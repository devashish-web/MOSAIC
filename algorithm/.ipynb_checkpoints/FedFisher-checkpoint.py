'''import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from algorithm.Base import BaseServer, BaseClient


# ============================================================
# Helpers
# ============================================================

def _infer_num_classes(data, args=None):
    if args is not None and hasattr(args, "num_classes"):
        return int(args.num_classes)
    return int(data.y.max().item()) + 1


def _extract_model_outputs(model, data):
    """
    Handles models returning either:
      - (hidden, log_probs)
      - log_probs only
    """
    out = model(data)
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            rep, logp = out[0], out[1]
        elif len(out) == 1:
            rep, logp = out[0], out[0]
        else:
            raise ValueError("Empty model output.")
    else:
        rep, logp = out, out
    if rep is None:
        rep = logp
    return rep, logp


# ============================================================
# Diagonal Fisher Computation
# ============================================================

def _compute_diagonal_fisher(model, data, num_classes, device):
    """
    Computes the diagonal of the Fisher Information Matrix at the
    current model parameters using the train-masked nodes.

    For classification with cross-entropy / NLL loss:
      F_diag[i] = E_y[ (d/dW_i log P(y|x,W))^2 ]

    In practice we approximate this using the empirical Fisher:
      F_diag = (1/N) * sum_n (grad_W log P(y_n|x_n,W))^2
    i.e. the element-wise squared gradient of the log-likelihood
    averaged over training samples. This is the standard diagonal
    Fisher approximation used in EWC, FisherMerge, and FedFisher(Diag).

    Returns:
        diag_fisher: dict { param_name -> 1D tensor of shape (numel,) }
    """
    model.eval()

    # Initialise accumulator
    diag_fisher = {
        name: torch.zeros_like(param.data).to(device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    train_mask = data.train_mask
    n_train = int(train_mask.sum().item())
    if n_train == 0:
        return diag_fisher

    # --- per-sample gradient accumulation ---
    # For efficiency we do ONE forward pass with the full training set
    # and accumulate the squared gradients, which equals the empirical
    # Fisher diagonal under i.i.d. samples.
    model.zero_grad()

    _, logp = _extract_model_outputs(model, data)
    logp = logp.float()

    # Use NLL loss over training nodes (sum reduction so we can divide by N)
    loss = F.nll_loss(
        logp[train_mask],
        data.y[train_mask],
        reduction="sum"
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Squared gradient / N  =  empirical Fisher diagonal
            diag_fisher[name] = (param.grad.detach() ** 2) / max(n_train, 1)

    model.zero_grad()
    return diag_fisher


# ============================================================
# FedFisher(Diag) Server Aggregation
# ============================================================

def _fedfisher_diag_aggregate(local_models_params, diag_fishers, size_weights=None):
    """
    Computes the FedFisher(Diag) global model analytically.

    From the paper (Proposition 2 + Lemma 1), the closed-form solution is:

        W* = (sum_i F_i)^{-1} * (sum_i F_i * W_i)

    where F_i is the diagonal Fisher (treated as a diagonal matrix).

    For the diagonal case this simplifies element-wise to:

        W*[j] = (sum_i F_i[j] * W_i[j]) / (sum_i F_i[j])

    i.e. each parameter is a Fisher-weighted average of local parameters.

    If all Fisher diagonals are zero for a parameter, fall back to
    a uniform average (= FedAvg) for that parameter.

    Args:
        local_models_params : list of state_dicts  (one per client)
        diag_fishers        : list of diag_fisher dicts (same order)
        size_weights        : optional 1-D array of dataset-size weights
                              (used only as a tiebreaker / fallback weight)

    Returns:
        global_state_dict : OrderedDict
    """
    K = len(local_models_params)
    assert K > 0, "No client models provided."
    assert len(diag_fishers) == K

    if size_weights is None:
        size_weights = np.ones(K, dtype=np.float64) / K
    else:
        size_weights = np.asarray(size_weights, dtype=np.float64)
        size_weights = size_weights / (size_weights.sum() + 1e-12)

    keys = list(local_models_params[0].keys())
    global_state = {}

    for key in keys:
        # Stack: (K, *param_shape)
        params = torch.stack(
            [local_models_params[k][key].float() for k in range(K)], dim=0
        )

        # Check if this key has a corresponding Fisher entry
        # (non-parameter buffers like running_mean won't have one)
        if key in diag_fishers[0]:
            # Fisher weights: (K, *param_shape)
            fishers = torch.stack(
                [diag_fishers[k][key].float() for k in range(K)], dim=0
            )

            # Weighted numerator and denominator
            # size_weights broadcast over param dims
            w = torch.tensor(size_weights, dtype=torch.float32)
            for _ in range(params.dim() - 1):
                w = w.unsqueeze(-1)

            weighted_fishers = fishers * w          # (K, *shape)
            numerator   = (weighted_fishers * params).sum(dim=0)
            denominator = weighted_fishers.sum(dim=0)

            # Where denominator is effectively zero -> fall back to FedAvg
            zero_mask = denominator.abs() < 1e-15
            global_param = torch.where(
                zero_mask,
                (params * w).sum(dim=0),            # FedAvg fallback
                numerator / denominator.clamp_min(1e-15)
            )
        else:
            # Buffer (e.g. BatchNorm running stats): simple weighted average
            w = torch.tensor(size_weights, dtype=torch.float32)
            for _ in range(params.dim() - 1):
                w = w.unsqueeze(-1)
            global_param = (params * w).sum(dim=0)

        global_state[key] = global_param.to(local_models_params[0][key].dtype)

    return global_state


# ============================================================
# FedFisher(Diag) with GD refinement (optional)
# ============================================================

def _fedfisher_diag_gd_refine(
    model,
    local_models_params,
    diag_fishers,
    size_weights,
    device,
    lr=0.01,
    steps=2000,
    eval_every=100,
    val_data=None,
):
    """
    Optional GD refinement from Algorithm 1 in the paper:

        W^{t+1} = W^t - lr * sum_i F_i (W^t - W_i)

    This refines the analytical solution when the diagonal Fisher
    approximation is imperfect. Can be skipped (steps=0) to use
    the pure closed-form solution.

    Args:
        model               : the global model (already initialised to W*)
        local_models_params : list of client state_dicts
        diag_fishers        : list of diagonal Fisher dicts
        size_weights        : dataset-size weights
        device              : torch.device
        lr                  : server learning rate η_S
        steps               : number of GD steps T
        eval_every          : checkpoint eval frequency
        val_data            : optional PyG Data for checkpointing

    Returns:
        best_state_dict or final state_dict
    """
    if steps <= 0:
        return {k: v.clone() for k, v in model.state_dict().items()}

    K = len(local_models_params)
    size_weights = np.asarray(size_weights, dtype=np.float64)
    size_weights = size_weights / (size_weights.sum() + 1e-12)

    # Pre-compute sum_i (F_i * W_i) and sum_i F_i for each param
    # These are constants throughout GD
    keys = [n for n, p in model.named_parameters() if p.requires_grad]

    # sum_i F_i  (diagonal, stored as tensor per param)
    sum_F = {}
    # sum_i F_i * W_i
    sum_FW = {}

    for key in keys:
        sf = torch.zeros_like(model.state_dict()[key]).float().to(device)
        sfw = torch.zeros_like(model.state_dict()[key]).float().to(device)
        for k in range(K):
            wk = float(size_weights[k])
            fk = diag_fishers[k][key].float().to(device)
            pk = local_models_params[k][key].float().to(device)
            sf  = sf  + wk * fk
            sfw = sfw + wk * fk * pk
        sum_F[key]  = sf
        sum_FW[key] = sfw

    # Use Adam for better convergence in practice (as in paper's appendix)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=0.01)

    best_val_acc = -1.0
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for t in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Gradient of FedFisher objective:
        # grad = sum_i F_i (W - W_i) = sum_F * W - sum_FW
        # We implement this as a pseudo-loss whose gradient equals the above
        loss = torch.tensor(0.0, device=device, requires_grad=False)
        for name, param in model.named_parameters():
            if name in sum_F:
                # 0.5 * sum_F * (W - W_i)^2 has gradient = sum_F * (W - W_i)
                residual = sum_F[name] * param.float() - sum_FW[name]
                # We need grad = residual, so loss = 0.5 * (residual^2 / sum_F)
                # Simpler: directly set grad manually
                param.grad = residual.to(param.dtype)

        # Step with the manually set gradients
        # (skip optimizer.step to avoid double computation)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
            model.zero_grad()

        if val_data is not None and (t + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                _, logp_v = _extract_model_outputs(model, val_data)
                pred = logp_v[val_data.val_mask].argmax(dim=1)
                ytrue = val_data.y[val_data.val_mask]
                val_acc = float((pred == ytrue).float().mean().item())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if val_data is not None and best_val_acc >= 0:
        return best_sd

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# Client
# ============================================================

class FedFisherClient(BaseClient):
    """
    FedFisher client: performs local training then computes
    the diagonal Fisher at the converged local model.

    Sends back:
        best_state   : trained model weights W_i
        diag_fisher  : diagonal Fisher F_i at W_i
    """

    def __init__(self, args, model, data):
        super().__init__(args, model, data)
        self.args = args
        self.device = torch.device(
            "cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = _infer_num_classes(data, args)
        self.local_epochs = args.sf_local_epochs
        self.eval_every   = args.sf_eval_every
        self.use_best     = args.sf_use_best_ckpt

        self.best_state  = None
        self.diag_fisher = None

    def train(self):
        """
        Local training + diagonal Fisher computation.
        Idempotent: if already trained, returns immediately.
        """
        if self.best_state is not None:
            return 0.0

        model     = self.model
        optimizer = self.optimizer

        best_val = -1.0
        best_sd  = None
        loss     = None

        # ---- local training ----
        for ep in range(self.local_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            _, out = _extract_model_outputs(model, self.data)
            loss = F.nll_loss(
                out[self.data.train_mask],
                self.data.y[self.data.train_mask]
            )
            loss.backward()
            optimizer.step()

            if self.use_best and (ep + 1) % self.eval_every == 0:
                if hasattr(self.data, "val_mask") and self.data.val_mask.sum().item() > 0:
                    model.eval()
                    with torch.no_grad():
                        _, out_v = _extract_model_outputs(model, self.data)
                    pred    = out_v[self.data.val_mask].argmax(dim=1).cpu()
                    ytrue   = self.data.y[self.data.val_mask].cpu()
                    val_acc = float((pred == ytrue).float().mean().item())
                    if val_acc > best_val:
                        best_val = val_acc
                        best_sd  = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }

        if self.use_best and best_sd is not None:
            model.load_state_dict(
                {k: v.to(self.device) for k, v in best_sd.items()}
            )

        # Cache trained weights
        self.best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        # ---- diagonal Fisher at W_i ----
        self.diag_fisher = _compute_diagonal_fisher(
            model=model,
            data=self.data,
            num_classes=self.num_classes,
            device=self.device,
        )
        # Move Fisher to CPU for communication efficiency
        self.diag_fisher = {
            k: v.detach().cpu()
            for k, v in self.diag_fisher.items()
        }

        return loss.item() if loss is not None else 0.0


# ============================================================
# Server
# ============================================================

class FedFisherServer(BaseServer):
    """
    FedFisher(Diag) Server.

    Algorithm (from paper):
      1. Collect local models W_i and diagonal Fishers F_i from clients.
      2. Initialise global model as FedAvg: W^(0) = mean(W_i).
      3. Run GD:  W^(t+1) = W^(t) - lr * sum_i F_i (W^(t) - W_i)
         OR use closed-form diagonal solution directly (default).
      4. Evaluate on test set.

    Key args (read from self.args):
        ff_use_gd         : bool  — run GD refinement (default False = closed-form only)
        ff_server_lr      : float — GD step size η_S (default 0.01)
        ff_server_steps   : int   — GD steps T (default 2000)
        ff_server_eval_every : int — checkpoint interval (default 100)
    """

    def __init__(self, args, clients, model, data, logger):
        super().__init__(args, clients, model, data, logger)
        self.args   = args
        self.device = torch.device(
            "cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu"
        )
        self.num_classes  = _infer_num_classes(data, args)
        self.model_ready  = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_active_client_ids(self):
        if (
            hasattr(self, "sampled_clients")
            and self.sampled_clients is not None
            and len(self.sampled_clients) > 0
        ):
            return list(self.sampled_clients)
        if isinstance(self.clients, dict):
            return list(self.clients.keys())
        return list(range(len(self.clients)))

    def _get_client(self, cid):
        if isinstance(self.clients, dict):
            return self.clients[cid]
        return self.clients[int(cid)]

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self):
        print("---------------------------")
        print("FedFisher(Diag): collecting local models and diagonal Fishers...")

        client_ids = self._get_active_client_ids()

        sizes              = []
        local_models_params = []
        diag_fishers       = []

        for cid in client_ids:
            client = self._get_client(cid)
            if client.best_state is None:
                client.train()

            n_train = int(client.data.train_mask.sum().item())
            sizes.append(n_train)
            local_models_params.append(client.best_state)
            diag_fishers.append(client.diag_fisher)

        # Dataset-size weights (used as fallback and for GD)
        total = sum(sizes) + 1e-12
        size_weights = [s / total for s in sizes]

        print(f"  Client ids    : {client_ids}")
        print(f"  Train sizes   : {sizes}")
        print(f"  Size weights  : {[round(w, 4) for w in size_weights]}")

        # ------ Fisher statistics (informational) ------
        for k, cid in enumerate(client_ids):
            fisher_norms = {
                name: float(diag_fishers[k][name].abs().mean().item())
                for name in diag_fishers[k]
            }
            total_fisher_norm = float(
                np.mean([v for v in fisher_norms.values()])
            )
            print(f"  Client {cid} mean |F_diag| : {total_fisher_norm:.6f}")

        # ------ Step 1: Closed-form diagonal FedFisher solution ------
        print("FedFisher(Diag): computing closed-form global model...")
        global_state = _fedfisher_diag_aggregate(
            local_models_params=local_models_params,
            diag_fishers=diag_fishers,
            size_weights=size_weights,
        )

        # Initialise global model with the closed-form solution
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in global_state.items()}
        )
        print("FedFisher(Diag): closed-form solution loaded into global model.")

        # ------ Step 2 (optional): GD refinement ------
        use_gd = bool(getattr(self.args, "ff_use_gd", False))

        if use_gd:
            lr          = float(getattr(self.args, "ff_server_lr",       0.01))
            steps       = int(getattr(self.args,   "ff_server_steps",    2000))
            eval_every  = int(getattr(self.args,   "ff_server_eval_every", 100))

            print(
                f"FedFisher(Diag): running GD refinement "
                f"(lr={lr}, steps={steps}, eval_every={eval_every})..."
            )

            # Use val_mask from the first client's data as a proxy
            # (server has no labelled data; use client 0 val set)
            val_data = self._get_client(client_ids[0]).data \
                if bool(getattr(self.args, "ff_gd_use_val", True)) else None

            best_sd = _fedfisher_diag_gd_refine(
                model=self.model,
                local_models_params=local_models_params,
                diag_fishers=diag_fishers,
                size_weights=size_weights,
                device=self.device,
                lr=lr,
                steps=steps,
                eval_every=eval_every,
                val_data=val_data,
            )
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_sd.items()}
            )
            print("FedFisher(Diag): GD refinement complete.")

        self.model_ready = True
        print("FedFisher(Diag): aggregation complete.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def global_evaluate(self):
        if not self.model_ready:
            self.aggregate()

        self.model.eval()
        with torch.no_grad():
            _, out = _extract_model_outputs(self.model, self.data)
            logp = out.float()

        test_mask = self.data.test_mask
        y_true    = self.data.y[test_mask].detach().cpu()
        y_prob    = torch.softmax(logp[test_mask], dim=1).detach().cpu()
        y_pred    = y_prob.argmax(dim=1)

        loss = F.nll_loss(
            torch.log(y_prob.clamp_min(1e-12)), y_true
        )
        acc  = float((y_pred == y_true).float().mean().item())

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.numpy(),
            y_pred.numpy(),
            average="macro",
            zero_division=0,
        )
        cm = confusion_matrix(
            y_true.numpy(),
            y_pred.numpy(),
            labels=list(range(self.num_classes)),
        )

        print("test_loss : "        + format(loss.item(), ".4f"))
        print("test_acc : "         + format(acc,         ".4f"))
        print("macro_precision : "  + format(float(precision), ".4f"))
        print("macro_recall : "     + format(float(recall),    ".4f"))
        print("macro_f1 : "         + format(float(f1),        ".4f"))
        print("confusion_matrix :")
        print(cm)

        if hasattr(self.logger, "write_test_loss"):
            self.logger.write_test_loss(loss.item())
        if hasattr(self.logger, "write_test_acc"):
            self.logger.write_test_acc(acc)
        if hasattr(self.logger, "write_test_f1"):
            self.logger.write_test_f1(float(f1))
        if hasattr(self.logger, "write_test_precision"):
            self.logger.write_test_precision(float(precision))
        if hasattr(self.logger, "write_test_recall"):
            self.logger.write_test_recall(float(recall))

        return {
            "loss":             float(loss.item()),
            "acc":              float(acc),
            "macro_precision":  float(precision),
            "macro_recall":     float(recall),
            "macro_f1":         float(f1),
            "confusion_matrix": cm,
        }'''



import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from algorithm.Base import BaseServer, BaseClient


# ============================================================
# Helpers
# ============================================================

def _infer_num_classes(data, args=None):
    if args is not None and hasattr(args, "num_classes"):
        return int(args.num_classes)
    return int(data.y.max().item()) + 1


def _extract_model_outputs(model, data):
    """
    Handles models returning either:
      - (hidden, log_probs)
      - log_probs only
    """
    out = model(data)
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            rep, logp = out[0], out[1]
        elif len(out) == 1:
            rep, logp = out[0], out[0]
        else:
            raise ValueError("Empty model output.")
    else:
        rep, logp = out, out
    if rep is None:
        rep = logp
    return rep, logp


def _payload_mb(obj):
    if obj is None:
        return 0.0
    if torch.is_tensor(obj):
        return (obj.numel() * obj.element_size()) / (1024 ** 2)
    if isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 ** 2)
    if isinstance(obj, dict):
        return sum(_payload_mb(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_payload_mb(v) for v in obj)
    if isinstance(obj, (float, np.floating, int, np.integer, bool)):
        return 8 / (1024 ** 2)
    return 0.0


# ============================================================
# Diagonal Fisher Computation
# ============================================================

def _compute_diagonal_fisher(model, data, num_classes, device):
    """
    Computes the diagonal of the Fisher Information Matrix at the
    current model parameters using the train-masked nodes.
    """
    model.eval()

    diag_fisher = {
        name: torch.zeros_like(param.data).to(device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    train_mask = data.train_mask
    n_train = int(train_mask.sum().item())
    if n_train == 0:
        return diag_fisher

    model.zero_grad()

    _, logp = _extract_model_outputs(model, data)
    logp = logp.float()

    loss = F.nll_loss(
        logp[train_mask],
        data.y[train_mask],
        reduction="sum"
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            diag_fisher[name] = (param.grad.detach() ** 2) / max(n_train, 1)

    model.zero_grad()
    return diag_fisher


# ============================================================
# FedFisher(Diag) Server Aggregation
# ============================================================

def _fedfisher_diag_aggregate(local_models_params, diag_fishers, size_weights=None):
    K = len(local_models_params)
    assert K > 0, "No client models provided."
    assert len(diag_fishers) == K

    if size_weights is None:
        size_weights = np.ones(K, dtype=np.float64) / K
    else:
        size_weights = np.asarray(size_weights, dtype=np.float64)
        size_weights = size_weights / (size_weights.sum() + 1e-12)

    keys = list(local_models_params[0].keys())
    global_state = {}

    for key in keys:
        params = torch.stack(
            [local_models_params[k][key].float() for k in range(K)], dim=0
        )

        if key in diag_fishers[0]:
            fishers = torch.stack(
                [diag_fishers[k][key].float() for k in range(K)], dim=0
            )

            w = torch.tensor(size_weights, dtype=torch.float32)
            for _ in range(params.dim() - 1):
                w = w.unsqueeze(-1)

            weighted_fishers = fishers * w
            numerator = (weighted_fishers * params).sum(dim=0)
            denominator = weighted_fishers.sum(dim=0)

            zero_mask = denominator.abs() < 1e-15
            global_param = torch.where(
                zero_mask,
                (params * w).sum(dim=0),
                numerator / denominator.clamp_min(1e-15)
            )
        else:
            w = torch.tensor(size_weights, dtype=torch.float32)
            for _ in range(params.dim() - 1):
                w = w.unsqueeze(-1)
            global_param = (params * w).sum(dim=0)

        global_state[key] = global_param.to(local_models_params[0][key].dtype)

    return global_state


# ============================================================
# FedFisher(Diag) with GD refinement (optional)
# ============================================================

def _fedfisher_diag_gd_refine(
    model,
    local_models_params,
    diag_fishers,
    size_weights,
    device,
    lr=0.01,
    steps=2000,
    eval_every=100,
    val_data=None,
):
    if steps <= 0:
        return {k: v.clone() for k, v in model.state_dict().items()}

    K = len(local_models_params)
    size_weights = np.asarray(size_weights, dtype=np.float64)
    size_weights = size_weights / (size_weights.sum() + 1e-12)

    keys = [n for n, p in model.named_parameters() if p.requires_grad]

    sum_F = {}
    sum_FW = {}

    for key in keys:
        sf = torch.zeros_like(model.state_dict()[key]).float().to(device)
        sfw = torch.zeros_like(model.state_dict()[key]).float().to(device)
        for k in range(K):
            wk = float(size_weights[k])
            fk = diag_fishers[k][key].float().to(device)
            pk = local_models_params[k][key].float().to(device)
            sf = sf + wk * fk
            sfw = sfw + wk * fk * pk
        sum_F[key] = sf
        sum_FW[key] = sfw

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=0.01)

    best_val_acc = -1.0
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for t in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss = torch.tensor(0.0, device=device, requires_grad=False)
        for name, param in model.named_parameters():
            if name in sum_F:
                residual = sum_F[name] * param.float() - sum_FW[name]
                param.grad = residual.to(param.dtype)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
            model.zero_grad()

        if val_data is not None and (t + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                _, logp_v = _extract_model_outputs(model, val_data)
                pred = logp_v[val_data.val_mask].argmax(dim=1)
                ytrue = val_data.y[val_data.val_mask]
                val_acc = float((pred == ytrue).float().mean().item())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if val_data is not None and best_val_acc >= 0:
        return best_sd

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# Client
# ============================================================

class FedFisherClient(BaseClient):
    """
    FedFisher client: performs local training then computes
    the diagonal Fisher at the converged local model.

    Sends back:
        best_state   : trained model weights W_i
        diag_fisher  : diagonal Fisher F_i at W_i
    """

    def __init__(self, args, model, data):
        super().__init__(args, model, data)
        self.args = args
        self.device = torch.device(
            "cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = _infer_num_classes(data, args)
        self.local_epochs = args.sf_local_epochs
        self.eval_every = args.sf_eval_every
        self.use_best = args.sf_use_best_ckpt

        self.best_state = None
        self.diag_fisher = None

    def train(self):
        if self.best_state is not None:
            return 0.0

        model = self.model
        optimizer = self.optimizer

        best_val = -1.0
        best_sd = None
        loss = None

        for ep in range(self.local_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            _, out = _extract_model_outputs(model, self.data)
            loss = F.nll_loss(
                out[self.data.train_mask],
                self.data.y[self.data.train_mask]
            )
            loss.backward()
            optimizer.step()

            if self.use_best and (ep + 1) % self.eval_every == 0:
                if hasattr(self.data, "val_mask") and self.data.val_mask.sum().item() > 0:
                    model.eval()
                    with torch.no_grad():
                        _, out_v = _extract_model_outputs(model, self.data)
                    pred = out_v[self.data.val_mask].argmax(dim=1).cpu()
                    ytrue = self.data.y[self.data.val_mask].cpu()
                    val_acc = float((pred == ytrue).float().mean().item())
                    if val_acc > best_val:
                        best_val = val_acc
                        best_sd = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }

        if self.use_best and best_sd is not None:
            model.load_state_dict(
                {k: v.to(self.device) for k, v in best_sd.items()}
            )

        self.best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        self.diag_fisher = _compute_diagonal_fisher(
            model=model,
            data=self.data,
            num_classes=self.num_classes,
            device=self.device,
        )
        self.diag_fisher = {
            k: v.detach().cpu()
            for k, v in self.diag_fisher.items()
        }

        return loss.item() if loss is not None else 0.0


# ============================================================
# Server
# ============================================================

class FedFisherServer(BaseServer):
    """
    FedFisher(Diag) Server.
    """

    def __init__(self, args, clients, model, data, logger):
        super().__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device(
            "cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = _infer_num_classes(data, args)
        self.model_ready = False

        # Communication accounting: client -> server only
        self.total_upload_mb = 0.0
        self.total_model_upload_mb = 0.0
        self.total_fisher_upload_mb = 0.0

    def _get_active_client_ids(self):
        if (
            hasattr(self, "sampled_clients")
            and self.sampled_clients is not None
            and len(self.sampled_clients) > 0
        ):
            return list(self.sampled_clients)
        if isinstance(self.clients, dict):
            return list(self.clients.keys())
        return list(range(len(self.clients)))

    def _get_client(self, cid):
        if isinstance(self.clients, dict):
            return self.clients[cid]
        return self.clients[int(cid)]

    def aggregate(self):
        print("---------------------------")
        print("FedFisher(Diag): collecting local models and diagonal Fishers...")

        client_ids = self._get_active_client_ids()

        sizes = []
        local_models_params = []
        diag_fishers = []

        round_model_upload_mb = 0.0
        round_fisher_upload_mb = 0.0

        for cid in client_ids:
            client = self._get_client(cid)
            if client.best_state is None:
                client.train()

            n_train = int(client.data.train_mask.sum().item())
            sizes.append(n_train)
            local_models_params.append(client.best_state)
            diag_fishers.append(client.diag_fisher)

            round_model_upload_mb += _payload_mb(client.best_state)
            round_fisher_upload_mb += _payload_mb(client.diag_fisher)

        round_upload_mb = round_model_upload_mb + round_fisher_upload_mb
        self.total_model_upload_mb += round_model_upload_mb
        self.total_fisher_upload_mb += round_fisher_upload_mb
        self.total_upload_mb += round_upload_mb

        print(f"Round model upload to server: {round_model_upload_mb:.2f} MB")
        print(f"Round Fisher upload to server: {round_fisher_upload_mb:.2f} MB")
        print(f"Round total upload to server: {round_upload_mb:.2f} MB")

        total = sum(sizes) + 1e-12
        size_weights = [s / total for s in sizes]

        print(f"  Client ids    : {client_ids}")
        print(f"  Train sizes   : {sizes}")
        print(f"  Size weights  : {[round(w, 4) for w in size_weights]}")

        for k, cid in enumerate(client_ids):
            fisher_norms = {
                name: float(diag_fishers[k][name].abs().mean().item())
                for name in diag_fishers[k]
            }
            total_fisher_norm = float(
                np.mean([v for v in fisher_norms.values()])
            )
            print(f"  Client {cid} mean |F_diag| : {total_fisher_norm:.6f}")

        print("FedFisher(Diag): computing closed-form global model...")
        global_state = _fedfisher_diag_aggregate(
            local_models_params=local_models_params,
            diag_fishers=diag_fishers,
            size_weights=size_weights,
        )

        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in global_state.items()}
        )
        print("FedFisher(Diag): closed-form solution loaded into global model.")

        use_gd = bool(getattr(self.args, "ff_use_gd", False))

        if use_gd:
            lr = float(getattr(self.args, "ff_server_lr", 0.01))
            steps = int(getattr(self.args, "ff_server_steps", 2000))
            eval_every = int(getattr(self.args, "ff_server_eval_every", 100))

            print(
                f"FedFisher(Diag): running GD refinement "
                f"(lr={lr}, steps={steps}, eval_every={eval_every})..."
            )

            val_data = self._get_client(client_ids[0]).data \
                if bool(getattr(self.args, "ff_gd_use_val", True)) else None

            best_sd = _fedfisher_diag_gd_refine(
                model=self.model,
                local_models_params=local_models_params,
                diag_fishers=diag_fishers,
                size_weights=size_weights,
                device=self.device,
                lr=lr,
                steps=steps,
                eval_every=eval_every,
                val_data=val_data,
            )
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_sd.items()}
            )
            print("FedFisher(Diag): GD refinement complete.")

        self.model_ready = True
        print("FedFisher(Diag): aggregation complete.")

    def global_evaluate(self):
        if not self.model_ready:
            self.aggregate()

        self.model.eval()
        with torch.no_grad():
            _, out = _extract_model_outputs(self.model, self.data)
            logp = out.float()

        test_mask = self.data.test_mask
        y_true = self.data.y[test_mask].detach().cpu()
        y_prob = torch.softmax(logp[test_mask], dim=1).detach().cpu()
        y_pred = y_prob.argmax(dim=1)

        loss = F.nll_loss(
            torch.log(y_prob.clamp_min(1e-12)), y_true
        )
        acc = float((y_pred == y_true).float().mean().item())

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.numpy(),
            y_pred.numpy(),
            average="macro",
            zero_division=0,
        )
        cm = confusion_matrix(
            y_true.numpy(),
            y_pred.numpy(),
            labels=list(range(self.num_classes)),
        )

        print("test_loss : " + format(loss.item(), ".4f"))
        print("test_acc : " + format(acc, ".4f"))
        print("macro_precision : " + format(float(precision), ".4f"))
        print("macro_recall : " + format(float(recall), ".4f"))
        print("macro_f1 : " + format(float(f1), ".4f"))
        print("confusion_matrix :")
        print(cm)

        if hasattr(self.logger, "write_test_loss"):
            self.logger.write_test_loss(loss.item())
        if hasattr(self.logger, "write_test_acc"):
            self.logger.write_test_acc(acc)
        if hasattr(self.logger, "write_test_f1"):
            self.logger.write_test_f1(float(f1))
        if hasattr(self.logger, "write_test_precision"):
            self.logger.write_test_precision(float(precision))
        if hasattr(self.logger, "write_test_recall"):
            self.logger.write_test_recall(float(recall))

        return {
            "loss": float(loss.item()),
            "acc": float(acc),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "confusion_matrix": cm,
        }

    def run(self):
        super().run()
        print("=" * 60)
        print(f"Total model upload to server: {self.total_model_upload_mb:.2f} MB")
        print(f"Total Fisher upload to server: {self.total_fisher_upload_mb:.2f} MB")
        print(f"Total client->server upload: {self.total_upload_mb:.2f} MB")
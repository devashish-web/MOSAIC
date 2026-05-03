import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected

from algorithm.Base import BaseClient, BaseServer


# ============================================================
# Utilities
# ============================================================

def _infer_num_classes(data, args=None):
    if args is not None and hasattr(args, "num_classes"):
        return int(args.num_classes)
    return int(data.y.max().item()) + 1


def _size_weights(sizes):
    arr = np.asarray(sizes, dtype=np.float64)
    return arr / (arr.sum() + 1e-12)


def _extract_model_outputs(model, data):
    """
    Normalise model output to (representation, log_probs).
    Accepts models that return either a tuple (rep, logp) or logp alone.
    """
    out = model(data)
    if isinstance(out, (tuple, list)):
        rep, logp = (out[0], out[1]) if len(out) >= 2 else (out[0], out[0])
    else:
        rep, logp = out, out
    return rep, logp


def _average_state_dicts(states, weights=None):
    k = len(states)
    if weights is None:
        weights = np.ones(k, dtype=np.float64) / max(k, 1)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)

    merged = {}
    for key in states[0]:
        stacked = torch.stack([state[key].float() for state in states])
        w = torch.tensor(weights, dtype=torch.float32).view(
            k, *([1] * (stacked.dim() - 1))
        )
        merged[key] = (stacked * w).sum(0).to(states[0][key].dtype)
    return merged


def _weighted_mean(vals, weights):
    vals = np.asarray(vals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    return float((vals * weights).sum())



def _imbalance_strength(counts):
    """Normalised entropy deficit: 0 = balanced, 1 = one dominant class."""
    counts = counts.float()
    if counts.sum().item() <= 0:
        return 0.0
    probs = counts / (counts.sum() + 1e-12)
    num_classes = probs.numel()
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum() / np.log(max(num_classes, 2))
    return float((1.0 - entropy).item())


# ============================================================
# Train-split graph utilities (no val/test leakage)
# ============================================================

def _train_edges(data):
    """Restrict edge index to train-train edges only."""
    if data.edge_index.size(1) == 0:
        return data.edge_index
    src, dst = data.edge_index
    mask = data.train_mask[src] & data.train_mask[dst]
    return data.edge_index[:, mask]


@torch.no_grad()
def _graph_homophily(data):
    edge_index = _train_edges(data)
    if edge_index.size(1) == 0:
        return 0.5
    src, dst = edge_index
    return float((data.y[src] == data.y[dst]).float().mean().item())


@torch.no_grad()
def _avg_train_degree(data):
    edge_index = _train_edges(data)
    num_train = int(data.train_mask.sum().item())
    return float(edge_index.size(1) / max(num_train, 1)) if num_train > 0 else 0.0


# ============================================================
# Client statistics
# ============================================================

@torch.no_grad()
def _label_transition_matrix(data, num_classes):
    """
    T[a,b] = P(y_dst=b | y_src=a) over train-train edges.
    Smoothing strength scales with class count and row-support sparsity.
    Returns the smoothed transition matrix and per-row support counts.
    """
    edge_index = _train_edges(data)

    if edge_index.size(1) == 0:
        trans = torch.full((num_classes, num_classes), 1.0 / num_classes)
        return trans, torch.zeros(num_classes)

    y_src = data.y[edge_index[0]].long()
    y_dst = data.y[edge_index[1]].long()
    counts = torch.bincount(
        y_src * num_classes + y_dst,
        minlength=num_classes ** 2,
    ).float().view(num_classes, num_classes)
    row_counts = counts.sum(1)

    pos_rows = row_counts[row_counts > 0]
    mean_support = float(pos_rows.mean().item()) if pos_rows.numel() > 0 else 0.0
    frac_supported = float((row_counts > 0).float().mean().item())
    class_risk = float(np.clip((np.log2(max(num_classes, 2)) - 3.0) / 3.0, 0.0, 1.0))
    smooth_risk = 0.5 * (1.0 - frac_supported) + 0.5 * class_risk
    lam = smooth_risk * min(2.0, 2.0 * num_classes / (mean_support + 1e-12))

    trans = (counts + lam) / (counts + lam).sum(1, keepdim=True).clamp_min(1e-12)
    return trans.cpu(), row_counts.cpu()


@torch.no_grad()
def _feature_prototypes(data, num_classes):
    """Class mean and class std in input feature space over train nodes."""
    x = data.x.detach()
    y = data.y
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    feat_dim = x.size(-1)
    means = torch.zeros(num_classes, feat_dim, dtype=x.dtype)
    stds = torch.zeros(num_classes, feat_dim, dtype=x.dtype)
    counts = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = data.train_mask & (y == c)
        n = int(mask.sum().item())
        counts[c] = n
        if n > 0:
            xc = x[mask].float()
            means[c] = xc.mean(0).to(x.dtype)
            stds[c] = xc.std(0, unbiased=False).to(x.dtype) if n > 1 else torch.zeros(feat_dim)

    return means.cpu(), stds.cpu(), counts.cpu()



def compute_client_statistics(data, num_classes):
    """Collect per-client structural summaries after local training."""
    transition, transition_row_counts = _label_transition_matrix(data, num_classes)
    feat_prototypes, feat_stds, class_counts = _feature_prototypes(data, num_classes)
    return {
        "transition": transition,
        "transition_row_counts": transition_row_counts,
        "feat_prototypes": feat_prototypes,
        "feat_proto_stds": feat_stds,
        "class_counts": class_counts,
        "homophily": _graph_homophily(data),
        "avg_train_degree": _avg_train_degree(data),
    }


# ============================================================
# Atlas aggregation
# ============================================================



def _transition_barycenter(transitions, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    trans_bar = sum(float(wk) * tk.float() for wk, tk in zip(weights, transitions))
    return trans_bar.cpu()



def _prototype_barycenter(prototypes, stds, counts, weights, min_std=1e-6):
    """
    Pooled mean/std barycenter weighted by client train-size and class counts.
    Std uses the law of total variance: E[sigma^2 + mu^2] - E[mu]^2.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    proto_num, proto_den, m2_num = None, None, None
    for wk, proto_k, std_k, count_k in zip(weights, prototypes, stds, counts):
        class_weight = float(wk) * count_k.float().view(-1, 1)
        second_moment = std_k.float().pow(2) + proto_k.float().pow(2)
        if proto_num is None:
            proto_num = class_weight * proto_k.float()
            proto_den = class_weight
            m2_num = class_weight * second_moment
        else:
            proto_num = proto_num + class_weight * proto_k.float()
            proto_den = proto_den + class_weight
            m2_num = m2_num + class_weight * second_moment

    proto_bar = proto_num / (proto_den + 1e-12)
    std_bar = ((m2_num / (proto_den + 1e-12) - proto_bar.pow(2)).clamp_min(0.0)).sqrt()

    valid = proto_den.squeeze(1) > 1e-12
    proto_bar[~valid] = 0.0
    std_bar[~valid] = 0.0
    if valid.any():
        std_bar[valid] = std_bar[valid].clamp_min(min_std)

    return proto_bar.cpu(), std_bar.cpu()



# ============================================================
# Pseudo-graph construction
# ============================================================

def _class_balanced_node_counts(global_class_counts, base_per_class, boost, max_mult=4.0):
    """Allocate more pseudo-nodes to minority classes."""
    priors = global_class_counts.float() / (global_class_counts.float().sum() + 1e-12)
    max_prior = float(priors.max().item()) if global_class_counts.sum().item() > 0 else 1.0

    counts = []
    for c in range(len(priors)):
        pc = float(priors[c].item())
        mult = min((max_prior / max(pc, 1e-12)) ** float(boost), max_mult) if pc > 0 else max_mult
        counts.append(max(2, int(round(base_per_class * mult))))
    return counts



def _sample_client_for_class(c, class_counts, weights, rng):
    """Sample a client proportional to weight * sqrt(class count)."""
    avail = [
        (k, float(weights[k]) * np.sqrt(float(count_k[c].item()) + 1e-12))
        for k, count_k in enumerate(class_counts)
        if float(count_k[c].item()) > 0
    ]
    if not avail:
        return int(rng.integers(0, len(class_counts)))

    idxs, probs = zip(*avail)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(idxs, p=probs))



def _estimate_global_feature_dispersion(feat_prototypes, class_counts, proto_bar):
    vals = []
    for proto_k, count_k in zip(feat_prototypes, class_counts):
        present = count_k > 0
        if present.sum().item() > 0:
            vals.append(
                float((proto_k[present].float() - proto_bar[present].float()).pow(2).mean().sqrt().item())
            )
    return float(np.mean(vals)) if vals else 0.0



def _derive_pseudo_params(atlas, args=None):
    """Derive pseudo-graph hyperparameters from atlas statistics."""
    global_class_counts = atlas["global_class_counts"].float()
    proto_bar = atlas["Fbar"].float()
    size_weights = np.asarray(atlas["size_weights"], dtype=np.float64)
    homophilies = np.asarray(atlas["homophilies"], dtype=np.float64)
    avg_degrees = np.asarray(atlas["avg_train_degrees"], dtype=np.float64)

    num_classes = int(global_class_counts.numel())
    avg_per_class = max(global_class_counts.sum().item() / max(num_classes, 1), 1.0)
    imbalance = _imbalance_strength(global_class_counts)
    feat_disp = _estimate_global_feature_dispersion(
        atlas["feat_prototypes"], atlas["class_counts"], proto_bar
    )
    mean_h = _weighted_mean(homophilies, size_weights)
    mean_deg = _weighted_mean(avg_degrees, size_weights)
    mix_ratio = feat_disp / (feat_disp + 1.0)

    base_per_class = int(np.clip(round(4.0 * np.sqrt(avg_per_class) * (1.0 + 0.5 * feat_disp)), 16, 96))
    minority_boost = float(np.clip(0.25 + 0.75 * imbalance, 0.25, 1.0))
    per_class_counts = _class_balanced_node_counts(global_class_counts, base_per_class, minority_boost)

    feat_mix_global = float(np.clip(0.50 + 0.25 * mix_ratio + 0.10 * (1.0 - mean_h), 0.45, 0.85))
    feat_noise = float(np.clip(0.01 + 0.06 * mix_ratio * (1.0 - mean_h), 0.0, 0.08))

    edge_per_node = int(np.clip(round((1.5 + np.log1p(max(mean_deg, 0.0))) * (0.75 + mean_h)), 4, 16))
    transition_edge_ratio = float(np.clip(0.35 + 0.40 * mean_h - 0.10 * mix_ratio, 0.25, 0.80))
    transition_edge_per_node = int(np.clip(round(edge_per_node * transition_edge_ratio), 2, max(edge_per_node - 1, 2)))
    knn_k = int(np.clip(round(edge_per_node * (1.25 - 0.50 * transition_edge_ratio)), 4, 20))

    if args is not None:
        if getattr(args, "mosaic_edge_per_node", None) is not None:
            edge_per_node = max(int(args.mosaic_edge_per_node), 2)
        if getattr(args, "mosaic_transition_edge_ratio", None) is not None:
            transition_edge_ratio = float(np.clip(args.mosaic_transition_edge_ratio, 0.05, 0.95))
            transition_edge_per_node = int(np.clip(round(edge_per_node * transition_edge_ratio), 1, edge_per_node))
        if getattr(args, "mosaic_knn_k", None) is not None:
            knn_k = max(int(args.mosaic_knn_k), 1)

    return {
        "base_per_class": base_per_class,
        "minority_boost": minority_boost,
        "feat_mix_global": feat_mix_global,
        "feat_noise": feat_noise,
        "edge_per_node": edge_per_node,
        "transition_edge_ratio": transition_edge_ratio,
        "transition_edge_per_node": transition_edge_per_node,
        "knn_k": knn_k,
        "per_class_counts": per_class_counts,
        "pseudo_nodes": int(np.sum(per_class_counts)),
        "imbalance_strength": imbalance,
        "feature_dispersion": feat_disp,
        "mean_homophily": mean_h,
        "mean_train_degree": mean_deg,
    }



def _build_transition_edges(y, class_to_nodes, trans_bar, edge_per_node, rng, device):
    """Sample edges according to the label transition matrix."""
    num_nodes = int(y.size(0))
    if num_nodes <= 1 or edge_per_node <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    trans_np = trans_bar.detach().cpu().numpy().astype(np.float64)
    edges = []
    for i in range(num_nodes):
        row = trans_np[int(y[i].item())].copy()
        row = row / (row.sum() + 1e-12)
        for _ in range(edge_per_node):
            dst_class = int(rng.choice(len(row), p=row))
            candidates = class_to_nodes[dst_class]
            if not candidates:
                continue
            j = int(rng.choice(candidates))
            if i != j:
                edges.append([i, j])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index



def _build_knn_edges(x, k, device):
    """Cosine kNN edges in pseudo-node feature space."""
    num_nodes = int(x.size(0))
    k = min(k, num_nodes - 1)
    if num_nodes <= 1 or k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    x_norm = x.float() / (x.float().norm(dim=1, keepdim=True) + 1e-12)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(-1e9)
    nbrs = torch.topk(sim, k=k, dim=1, largest=True).indices

    src = torch.arange(num_nodes, device=device).view(-1, 1).expand(-1, k).reshape(-1)
    dst = nbrs.reshape(-1)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.stack([src, dst])
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index



def _merge_edges(edge_indices, num_nodes, device):
    valid = [edge_index for edge_index in edge_indices if edge_index is not None and edge_index.numel() > 0]
    if not valid:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    edge_index = torch.cat(valid, dim=1)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index



def _make_train_val_masks(y, val_ratio=0.2, seed=0):
    """Stratified train/val split on pseudo-node labels."""
    y_np = y.detach().cpu().numpy()
    num_nodes = len(y_np)
    train_mask = torch.ones(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    rng = np.random.default_rng(seed)

    for c in np.unique(y_np):
        idx = np.where(y_np == c)[0]
        n_val = min(max(1, int(round(val_ratio * len(idx)))), len(idx) - 1)
        if n_val > 0:
            chosen = rng.choice(idx, size=n_val, replace=False)
            val_mask[chosen] = True
            train_mask[chosen] = False

    return train_mask, val_mask



def _build_pseudo_graph(atlas, args, device):
    """
    Synthesise a pseudo-graph from the structural atlas.

    Nodes: class-conditional Gaussian samples interpolated between the
           global prototype barycenter and a client-specific prototype.
    Edges: hybrid of transition-matrix-sampled edges and cosine kNN edges.
    """
    rng = np.random.default_rng(int(getattr(args, "mosaic_random_seed", 0)))
    proto_bar = atlas["Fbar"].float()
    std_bar = atlas["Fstd"].float()
    trans_bar = atlas["Tbar"].float()
    size_weights = np.asarray(atlas["size_weights"], dtype=np.float64)
    size_weights = size_weights / (size_weights.sum() + 1e-12)

    params = _derive_pseudo_params(atlas, args=args)
    mix_g = params["feat_mix_global"]
    noise = params["feat_noise"]

    xs, ys = [], []
    class_to_nodes = {c: [] for c in range(proto_bar.size(0))}
    node_id = 0

    for c in range(proto_bar.size(0)):
        for _ in range(params["per_class_counts"][c]):
            k = _sample_client_for_class(c, atlas["class_counts"], size_weights, rng)
            has_class = float(atlas["class_counts"][k][c].item()) > 0
            client_proto = atlas["feat_prototypes"][k][c].float().to(device) if has_class else proto_bar[c].to(device)
            client_std = (
                atlas["feat_proto_stds"][k][c].float().to(device).clamp_min(1e-6)
                if has_class else std_bar[c].to(device)
            )

            x_mean = mix_g * proto_bar[c].to(device) + (1.0 - mix_g) * client_proto
            x_std = (mix_g * std_bar[c].to(device) + (1.0 - mix_g) * client_std).clamp_min(1e-6)
            x = x_mean + (noise * x_std * torch.randn_like(x_mean) if noise > 0 else 0.0)

            xs.append(x)
            ys.append(c)
            class_to_nodes[c].append(node_id)
            node_id += 1

    x = torch.stack(xs).float().to(device)
    y = torch.tensor(ys, dtype=torch.long, device=device)

    trans_edge_index = _build_transition_edges(
        y,
        class_to_nodes,
        trans_bar.to(device),
        params["transition_edge_per_node"],
        rng,
        device,
    )
    knn_edge_index = _build_knn_edges(x, params["knn_k"], device)
    edge_index = _merge_edges([trans_edge_index, knn_edge_index], x.size(0), device)

    train_mask, val_mask = _make_train_val_masks(
        y,
        val_ratio=float(getattr(args, "mosaic_pseudo_val_ratio", 0.2)),
        seed=int(getattr(args, "mosaic_random_seed", 0)),
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=torch.zeros_like(train_mask),
    )
    data.pseudo_params = params
    data.transition_edge_index = trans_edge_index
    data.knn_edge_index = knn_edge_index
    return data


# ============================================================
# Expert gating
# ============================================================

def _zscore_across_experts(score_mat):
    """Normalise gate scores across experts per node: (K, N) -> (K, N)."""
    mean = score_mat.mean(0, keepdim=True)
    std = score_mat.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (score_mat - mean) / std



def _neighbor_avg_probs(edge_index, probs, include_self=True):
    """Mean predicted distribution over 1-hop neighbourhood: (N,C) -> (N,C)."""
    num_nodes, num_classes = probs.size()
    src, dst = edge_index

    agg = torch.zeros(num_nodes, num_classes, dtype=probs.dtype, device=probs.device)
    deg = torch.zeros(num_nodes, 1, dtype=probs.dtype, device=probs.device)
    agg.index_add_(0, dst, probs[src])
    deg.index_add_(0, dst, torch.ones(src.size(0), 1, dtype=probs.dtype, device=probs.device))

    if include_self:
        agg = agg + probs
        deg = deg + 1.0

    out = agg / deg.clamp_min(1.0)
    isolated = deg.squeeze(1) <= 0
    if isolated.any():
        out[isolated] = probs[isolated]
    return out



def _structural_consistency(probs, transition, edge_index, include_self=True):
    """
    Per-node negative KL divergence between observed neighbourhood distribution
    and the distribution expected from the label transition matrix.
    Higher = more consistent with client graph structure.
    Returns scores of shape (N,).
    """
    neigh = _neighbor_avg_probs(edge_index, probs, include_self)
    transition = transition.float().to(probs.device)
    expected = probs @ transition
    expected = expected / expected.sum(1, keepdim=True).clamp_min(1e-12)
    kl = (neigh * (torch.log(neigh.clamp_min(1e-12)) - torch.log(expected.clamp_min(1e-12)))).sum(1)
    return -kl



def _fixed_gate_temperature(args):
    """Fixed softmax temperature for structural-only expert gating."""
    if args is None:
        return 0.5
    if getattr(args, "sf_gate_temp", None) is not None:
        return max(float(args.sf_gate_temp), 1e-6)
    if getattr(args, "sf_gate_temp_base", None) is not None:
        return max(float(args.sf_gate_temp_base), 1e-6)
    return 0.5


# ============================================================
# Distillation helpers
# ============================================================

def _class_balanced_ce_weights(global_class_counts, beta=1.0):
    """Inverse-frequency class weights for cross-entropy, scaled to mean 1."""
    counts = global_class_counts.float()
    if beta <= 0:
        return torch.ones(counts.numel())
    weights = (counts.sum() + 1e-12) / (counts.numel() * (counts + 1e-12))
    weights = weights.pow(float(beta))
    return (weights / weights.mean().clamp_min(1e-12)).float()



def _fixed_kd_temperature(args):
    """Fixed KD temperature."""
    if args is None:
        return 1.5
    if getattr(args, "mosaic_kd_temp", None) is not None:
        return max(float(args.mosaic_kd_temp), 1e-6)
    return 1.5



def _fixed_distill_coefs(args):
    """Fixed KD and CE coefficients for the non-adaptive ablation."""
    kd_coef = float(getattr(args, "mosaic_kd_coef", 1.0)) if args is not None else 1.0
    hard_coef = float(getattr(args, "mosaic_hard_coef", 1.0)) if args is not None else 1.0
    ce_beta = float(getattr(args, "mosaic_ce_beta", 1.0)) if args is not None else 1.0
    return {
        "kd_coef": max(kd_coef, 0.0),
        "hard_coef": max(hard_coef, 0.0),
        "ce_beta": max(ce_beta, 0.0),
    }


# ============================================================
# Client
# ============================================================

class Non_Adaptive_MOSAICClient(BaseClient):
    """
    One-shot local training client.
    Trains a GNN on its local subgraph, then computes and caches
    structural statistics for server-side atlas construction.
    """

    def __init__(self, args, model, data):
        super().__init__(args, model, data)
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = _infer_num_classes(data, args)
        self.best_state = None
        self.stats = None

    def train(self):
        if self.best_state is not None:
            return 0.0

        model = self.model
        optimizer = self.optimizer
        best_val, best_state_dict, loss = -1.0, None, None

        for ep in range(self.args.sf_local_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, out = _extract_model_outputs(model, self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()

            if (
                self.args.sf_use_best_ckpt
                and (ep + 1) % self.args.sf_eval_every == 0
                and self.data.val_mask.sum().item() > 0
            ):
                model.eval()
                with torch.no_grad():
                    _, out_val = _extract_model_outputs(model, self.data)
                acc = float(
                    (out_val[self.data.val_mask].argmax(1) == self.data.y[self.data.val_mask])
                    .float()
                    .mean()
                    .item()
                )
                if acc > best_val:
                    best_val = acc
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

        if self.args.sf_use_best_ckpt and best_state_dict is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})

        self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.stats = compute_client_statistics(self.data, self.num_classes)
        return loss.item() if loss is not None else 0.0


# ============================================================
# Server
# ============================================================

class Non_Adaptive_MOSAICServer(BaseServer):
    """
    MOSAIC: Structural Atlas Distillation.

    Aggregation is one-shot:
      aggregate() -> builds atlas, pseudo-graph, teacher labels, distils student.
    global_evaluate() -> evaluates the distilled student on the held-out test set.
    """

    def __init__(self, args, clients, model, data, logger):
        super().__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = _infer_num_classes(data, args)
        self.atlas = None
        self.pseudo_data = None
        self._ready = False

    def _client_ids(self):
        if getattr(self, "sampled_clients", None):
            return list(self.sampled_clients)
        return list(self.clients.keys()) if isinstance(self.clients, dict) else list(range(len(self.clients)))

    def _get_client(self, cid):
        return self.clients[cid] if isinstance(self.clients, dict) else self.clients[int(cid)]

    def aggregate(self):
        print("=" * 60)
        print("MOSAIC: aggregating client experts ...")

        client_ids = self._client_ids()
        sizes, best_states = [], []
        transitions, transition_row_counts = [], []
        feat_prototypes, feat_stds = [], []
        class_counts = []
        homophilies, avg_degrees = [], []

        for cid in client_ids:
            client = self._get_client(cid)
            if client.best_state is None:
                client.train()

            stats = client.stats
            sizes.append(int(client.data.train_mask.sum().item()))
            best_states.append(client.best_state)
            transitions.append(stats["transition"])
            transition_row_counts.append(stats["transition_row_counts"])
            feat_prototypes.append(stats["feat_prototypes"])
            feat_stds.append(stats["feat_proto_stds"])
            class_counts.append(stats["class_counts"])
            homophilies.append(float(stats["homophily"]))
            avg_degrees.append(float(stats["avg_train_degree"]))

        size_weights = _size_weights(sizes)

        max_experts = int(getattr(self.args, "sf_max_experts", 0))
        if 0 < max_experts < len(client_ids):
            keep = sorted(np.argsort(-size_weights)[:max_experts].tolist())
            client_ids = [client_ids[i] for i in keep]
            sizes = [sizes[i] for i in keep]
            best_states = [best_states[i] for i in keep]
            transitions = [transitions[i] for i in keep]
            transition_row_counts = [transition_row_counts[i] for i in keep]
            feat_prototypes = [feat_prototypes[i] for i in keep]
            feat_stds = [feat_stds[i] for i in keep]
            class_counts = [class_counts[i] for i in keep]
            homophilies = [homophilies[i] for i in keep]
            avg_degrees = [avg_degrees[i] for i in keep]
            size_weights = _size_weights(sizes)
            print(f"  Pruned to {max_experts} experts: {client_ids}")

        self._log_federation_stats(
            client_ids,
            sizes,
            size_weights,
            homophilies,
            avg_degrees,
        )

        atlas_weights = size_weights
        global_class_counts = torch.stack([count_k.float() for count_k in class_counts]).sum(0)
        trans_bar = _transition_barycenter(transitions, atlas_weights)
        proto_bar, std_bar = _prototype_barycenter(
            feat_prototypes,
            feat_stds,
            class_counts,
            atlas_weights,
        )

        self.atlas = {
            "Tbar": trans_bar.cpu(),
            "Fbar": proto_bar.cpu(),
            "Fstd": std_bar.cpu(),
            "global_class_counts": global_class_counts.cpu(),
            "size_weights": size_weights,
            "atlas_weights": atlas_weights,
            "feat_prototypes": feat_prototypes,
            "feat_proto_stds": feat_stds,
            "class_counts": class_counts,
            "homophilies": homophilies,
            "avg_train_degrees": avg_degrees,
        }

        if getattr(self.args, "mosaic_init_from_experts", True):
            avg_state = _average_state_dicts(best_states, weights=atlas_weights)
            self.model.load_state_dict({k: v.to(self.device) for k, v in avg_state.items()})

        self.pseudo_data = _build_pseudo_graph(self.atlas, self.args, self.device)
        self._log_pseudo_graph(self.pseudo_data)

        teacher_probs = self._fuse_teacher_labels(
            pseudo_data=self.pseudo_data,
            client_ids=client_ids,
        )
        self._distil_student(teacher_probs)

        print("MOSAIC: done.")
        print("=" * 60)

    @torch.no_grad()
    def _fuse_teacher_labels(self, pseudo_data, client_ids):
        """Fuse client expert predictions on the pseudo-graph via structural-only gating."""
        pred_temp = max(float(getattr(self.args, "sf_pred_temp", 1.0)), 1e-6)
        probs_list = []

        for cid in client_ids:
            client = self._get_client(cid)
            client.model.eval()
            _, logp = _extract_model_outputs(client.model, pseudo_data)
            logp = logp.float().to(self.device)
            probs = logp.exp() if pred_temp == 1.0 else torch.softmax(logp / pred_temp, dim=1)
            probs_list.append(probs)

        probs = torch.stack(probs_list)
        struct_raw = torch.stack([
            _structural_consistency(
                probs[k],
                self._get_client(cid).stats["transition"],
                pseudo_data.edge_index.to(self.device),
            )
            for k, cid in enumerate(client_ids)
        ])

        struct_z = _zscore_across_experts(struct_raw)
        temp = _fixed_gate_temperature(self.args)
        gate_weights = torch.softmax(struct_z / max(temp, 1e-6), dim=0)
        fused = (gate_weights.unsqueeze(-1) * probs).sum(0)

        print(f"  Gate: structural_only=True  temp={temp:.3f}")
        return fused

    def _distil_student(self, teacher_probs):
        """
        Distil global student on pseudo-graph with KD + CE.

        KD and CE use fixed coefficients.
        CE uses inverse-frequency class weights.
        KD uses a fixed temperature.
        """
        model = self.model
        data = self.pseudo_data
        teacher_probs = teacher_probs.detach()

        lr = float(getattr(self.args, "mosaic_server_lr", getattr(self.args, "learning_rate", 1e-3)))
        wd = float(getattr(self.args, "mosaic_server_wd", getattr(self.args, "weight_decay", 5e-4)))
        epochs = int(getattr(self.args, "mosaic_server_epochs", 200))
        eval_every = int(getattr(self.args, "mosaic_server_eval_every", 10))
        use_best = bool(getattr(self.args, "mosaic_server_use_best_ckpt", True))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        distill_coefs = _fixed_distill_coefs(self.args)
        kd_temp = _fixed_kd_temperature(self.args)
        ce_beta = distill_coefs["ce_beta"]
        ce_weights = _class_balanced_ce_weights(self.atlas["global_class_counts"], beta=ce_beta).to(self.device)

        print(
            f"  Distil: kd_coef={distill_coefs['kd_coef']:.3f}  hard_coef={distill_coefs['hard_coef']:.3f}"
            f"  kd_temp={kd_temp:.3f}  ce_beta={ce_beta:.3f}"
        )

        best_score, best_state_dict, last_loss = -1e18, None, None

        for ep in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            _, logp = _extract_model_outputs(model, data)
            student_prob_t = torch.softmax(logp.float() / kd_temp, dim=1)
            teacher_prob_t = torch.softmax(torch.log(teacher_probs.clamp_min(1e-12)) / kd_temp, dim=1)

            kd_loss = F.kl_div(
                torch.log(student_prob_t[data.train_mask].clamp_min(1e-12)),
                teacher_prob_t[data.train_mask],
                reduction="batchmean",
            ) * (kd_temp ** 2)

            hard_loss = F.nll_loss(
                torch.log(student_prob_t[data.train_mask].clamp_min(1e-12)),
                data.y[data.train_mask],
                weight=ce_weights,
            )

            loss = distill_coefs["kd_coef"] * kd_loss + distill_coefs["hard_coef"] * hard_loss
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

            if use_best and (ep + 1) % eval_every == 0 and data.val_mask.sum().item() > 0:
                model.eval()
                with torch.no_grad():
                    _, logp_val = _extract_model_outputs(model, data)
                    prob_val = torch.softmax(logp_val.float(), dim=1)

                acc_val = float(
                    (prob_val[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean().item()
                )
                kl_val = F.kl_div(
                    torch.log(prob_val[data.val_mask].clamp_min(1e-12)),
                    teacher_probs[data.val_mask],
                    reduction="batchmean",
                ).item()
                score = acc_val - 0.1 * kl_val
                if score > best_score:
                    best_score = score
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

        if use_best and best_state_dict is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})

        self._ready = True
        print(f"  Distil done. final_loss={last_loss:.6f}")

    def global_evaluate(self):
        if not self._ready:
            self.aggregate()

        self.model.eval()
        with torch.no_grad():
            _, out = _extract_model_outputs(self.model, self.data)

        mask = self.data.test_mask
        y_true = self.data.y[mask].cpu()
        y_prob = torch.softmax(out.float()[mask], dim=1).cpu()
        y_pred = y_prob.argmax(1)

        loss = F.nll_loss(torch.log(y_prob.clamp_min(1e-12)), y_true)
        acc = float((y_pred == y_true).float().mean().item())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.numpy(), y_pred.numpy(), average="macro", zero_division=0
        )
        cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(self.num_classes)))

        print(f"test_loss      : {loss.item():.4f}")
        print(f"test_acc       : {acc:.4f}")
        print(f"macro_f1       : {f1:.4f}")
        print(f"macro_precision: {precision:.4f}")
        print(f"macro_recall   : {recall:.4f}")
        print("confusion_matrix:")
        print(cm)

        for attr, value in [
            ("write_test_loss", loss.item()),
            ("write_test_acc", acc),
            ("write_test_f1", float(f1)),
            ("write_test_precision", float(precision)),
            ("write_test_recall", float(recall)),
        ]:
            if hasattr(self.logger, attr):
                getattr(self.logger, attr)(value)

        return {
            "loss": float(loss.item()),
            "acc": acc,
            "macro_f1": float(f1),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "confusion_matrix": cm,
        }

    def _log_federation_stats(self, client_ids, sizes, size_weights, homophilies, avg_degrees):
        print(f"  Clients          : {client_ids}")
        print(f"  Train sizes      : {sizes}")
        print(f"  Size weights     : {np.round(size_weights, 4).tolist()}")
        print(f"  Homophilies      : {[round(h, 4) for h in homophilies]}")
        print(f"  Avg train degree : {[round(d, 4) for d in avg_degrees]}")

    def _log_pseudo_graph(self, pseudo_data):
        params = pseudo_data.pseudo_params
        print(f"  Pseudo nodes     : {params['pseudo_nodes']}")
        print(f"  Base/class       : {params['base_per_class']}  minority_boost={params['minority_boost']:.4f}")
        print(f"  Feat mix global  : {params['feat_mix_global']:.4f}  noise={params['feat_noise']:.4f}")
        print(
            f"  Edge/node        : {params['edge_per_node']}  "
            f"trans_ratio={params['transition_edge_ratio']:.4f}  knn_k={params['knn_k']}"
        )
        print(
            f"  Trans edges      : {pseudo_data.transition_edge_index.size(1)}  "
            f"kNN edges: {pseudo_data.knn_edge_index.size(1)}  total: {pseudo_data.edge_index.size(1)}"
        )
        print(
            f"  Homophily        : {params['mean_homophily']:.4f}  "
            f"imbalance={params['imbalance_strength']:.4f}  feat_disp={params['feature_dispersion']:.4f}"
        )


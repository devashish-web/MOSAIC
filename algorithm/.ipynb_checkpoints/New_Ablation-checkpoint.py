import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from algorithm.Base import BaseClient, BaseServer

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import DensityMatrix
except Exception as e:  # pragma: no cover
    QuantumCircuit = None
    StatePreparation = None
    DensityMatrix = None
    _QISKIT_IMPORT_ERROR = e
else:
    _QISKIT_IMPORT_ERROR = None


# =============================================================================
# Logging
# =============================================================================

class StepLogger:
    def __init__(self, prefix: str, enabled: bool = True):
        self.prefix = prefix
        self.enabled = enabled

    def log(self, msg: str):
        if self.enabled:
            print(f"[{self.prefix}] {msg}")

    @contextmanager
    def step(self, msg: str):
        if self.enabled:
            print(f"[{self.prefix}] {msg} ...")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                print(f"[{self.prefix}] {msg} done in {time.perf_counter() - t0:.2f}s")


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ClientAtom:
    client_id: int
    cls: int
    count: float
    z: np.ndarray
    rho: Optional[np.ndarray] = None


# =============================================================================
# Utilities
# =============================================================================


def _ensure_qiskit():
    if QuantumCircuit is None or StatePreparation is None or DensityMatrix is None:
        raise ImportError(
            "Qiskit is required for the quantum QBaryEnsemble path. "
            f"Original import error: {_QISKIT_IMPORT_ERROR}"
        )



def _payload_mb(obj) -> float:
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



def _infer_num_classes(data, args=None) -> int:
    if args is not None and hasattr(args, "num_classes"):
        return int(args.num_classes)
    return int(data.y.max().item()) + 1



def _extract_model_outputs(model, data):
    out = model(data)
    if isinstance(out, (tuple, list)):
        rep, logp = (out[0], out[1]) if len(out) >= 2 else (out[0], out[0])
    else:
        rep, logp = out, out
    return rep, logp



def _size_weights(sizes: Sequence[int]) -> np.ndarray:
    arr = np.asarray(sizes, dtype=np.float64)
    return arr / (arr.sum() + 1e-12)


@torch.no_grad()
def _compute_train_only_class_stats(rep: torch.Tensor, data, num_classes: int):
    rep = rep.detach().float()
    if rep.dim() == 1:
        rep = rep.unsqueeze(-1)

    means = torch.zeros(num_classes, rep.size(1), dtype=torch.float32, device=rep.device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=rep.device)

    for c in range(num_classes):
        mask = data.train_mask & (data.y == c)
        n = int(mask.sum().item())
        counts[c] = n
        if n <= 0:
            continue
        rc = rep[mask]
        means[c] = rc.mean(0)

    return {
        "latent_proto": means.cpu(),
        "class_counts": counts.cpu(),
    }


# =============================================================================
# Density-matrix geometry
# =============================================================================


def _hermitian(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)



def _project_psd_trace_one(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rho = _hermitian(np.asarray(rho, dtype=np.complex128))
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(np.real(evals), 0.0, None)
    out = (evecs * evals) @ evecs.conj().T
    tr = float(np.real(np.trace(out)))
    if tr <= eps:
        d = out.shape[0]
        out = np.eye(d, dtype=np.complex128) / d
    else:
        out = out / tr
    return _hermitian(out)



def _sqrtm_psd(rho: np.ndarray) -> np.ndarray:
    rho = _hermitian(np.asarray(rho, dtype=np.complex128))
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(np.real(evals), 0.0, None)
    return _hermitian((evecs * np.sqrt(evals)) @ evecs.conj().T)



def _invsqrtm_psd(rho: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    rho = _project_psd_trace_one(rho)
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(np.real(evals), eps, None)
    return _hermitian((evecs * (1.0 / np.sqrt(evals))) @ evecs.conj().T)



def _fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    rho = _project_psd_trace_one(rho)
    sigma = _project_psd_trace_one(sigma)
    sr = _sqrtm_psd(rho)
    middle = sr @ sigma @ sr
    sm = _sqrtm_psd(middle)
    val = float(np.real(np.trace(sm))) ** 2
    return float(np.clip(val, 0.0, 1.0))



def _bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    f = _fidelity(rho, sigma)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f))))



def _bures_barycenter(
    rhos: Sequence[np.ndarray],
    weights: Sequence[float],
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    if len(rhos) == 1:
        return _project_psd_trace_one(rhos[0])

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    rho = np.zeros_like(rhos[0], dtype=np.complex128)
    for w, r in zip(weights, rhos):
        rho = rho + float(w) * _project_psd_trace_one(r)
    rho = _project_psd_trace_one(rho)

    for _ in range(max_iter):
        sr = _sqrtm_psd(rho)
        isr = _invsqrtm_psd(rho)

        s = np.zeros_like(rho, dtype=np.complex128)
        for w, r in zip(weights, rhos):
            intermediate = sr @ _project_psd_trace_one(r) @ sr
            s = s + float(w) * _sqrtm_psd(intermediate)

        rho_next = isr @ s @ s @ isr
        rho_next = _project_psd_trace_one(rho_next)

        if np.linalg.norm(rho_next - rho, ord="fro") <= tol:
            rho = rho_next
            break
        rho = rho_next

    return _project_psd_trace_one(rho)



def _euclidean_density_mean(rhos: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    rho = np.zeros_like(rhos[0], dtype=np.complex128)
    for w, r in zip(weights, rhos):
        rho = rho + float(w) * _project_psd_trace_one(r)
    return _project_psd_trace_one(rho)


# =============================================================================
# PCA and amplitude encoding
# =============================================================================


def _resolve_pca_target_dim(args) -> int:
    n_qubits = int(getattr(args, "qbary_n_qubits", 3))
    state_dim = 2 ** n_qubits
    user_dim = getattr(args, "qbary_pca_dim", None)
    if user_dim is None:
        return state_dim
    user_dim = int(user_dim)
    if user_dim <= 0:
        return state_dim
    return min(user_dim, state_dim)



def _fit_server_pca(proto_list: Sequence[np.ndarray], target_dim: int):
    x = np.asarray(proto_list, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"PCA expects a 2D matrix, got shape={x.shape}")

    n_samples, n_features = x.shape
    effective_dim = int(min(max(target_dim, 1), n_samples, n_features))
    if effective_dim <= 0:
        raise ValueError("PCA target dimension must be positive")

    pca = PCA(n_components=effective_dim, svd_solver="auto", random_state=0)
    pca.fit(x)
    return {
        "mean": pca.mean_.astype(np.float64),
        "components": pca.components_.astype(np.float64),
        "explained_variance_ratio": getattr(
            pca,
            "explained_variance_ratio_",
            np.array([], dtype=np.float64),
        ).astype(np.float64),
        "effective_dim": effective_dim,
        "target_dim": int(target_dim),
        "input_dim": int(n_features),
    }



def _apply_server_pca(vec: np.ndarray, pca_meta: Dict) -> np.ndarray:
    x = np.asarray(vec, dtype=np.float64).ravel()
    centered = x - pca_meta["mean"]
    red = pca_meta["components"] @ centered
    target_dim = int(pca_meta["target_dim"])
    if red.size < target_dim:
        red = np.pad(red, (0, target_dim - red.size), mode="constant")
    elif red.size > target_dim:
        red = red[:target_dim]
    return red.astype(np.float64, copy=False)



def _pad_or_trim(z: np.ndarray, size: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size < size:
        return np.pad(z, (0, size - z.size), mode="constant")
    if z.size > size:
        return z[:size]
    return z



def _build_client_amplitude_encoding_circuit(z: np.ndarray, n_qubits: int) -> "QuantumCircuit":
    _ensure_qiskit()
    dim = 2 ** n_qubits
    amp = _pad_or_trim(z, dim).astype(np.float64)
    nrm = float(np.linalg.norm(amp))
    if nrm <= 1e-12:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / nrm

    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(amp.tolist(), normalize=True), range(n_qubits))
    return qc



def _encode_reduced_vector_to_density(z: np.ndarray, args) -> np.ndarray:
    _ensure_qiskit()
    n_qubits = int(getattr(args, "qbary_n_qubits", 3))
    qc = _build_client_amplitude_encoding_circuit(z, n_qubits)
    rho = np.asarray(DensityMatrix.from_instruction(qc).data, dtype=np.complex128)
    return _project_psd_trace_one(rho)


# =============================================================================
# Aggregation references and geometry scoring
# =============================================================================


def _group_atoms_by_class(atoms: Sequence[ClientAtom], num_classes: int) -> List[List[ClientAtom]]:
    grouped: List[List[ClientAtom]] = [[] for _ in range(num_classes)]
    for atom in atoms:
        grouped[int(atom.cls)].append(atom)
    return grouped



def _weighted_proto_mean(zs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    return np.sum(np.asarray(zs, dtype=np.float64) * weights[:, None], axis=0)



def _compute_reference_sets(
    atoms: Sequence[ClientAtom],
    num_classes: int,
    barycenter_mode: str,
    bary_iters: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    grouped = _group_atoms_by_class(atoms, num_classes)
    class_state_refs: Dict[int, np.ndarray] = {}
    class_proto_refs: Dict[int, np.ndarray] = {}
    state_map: Dict[Tuple[int, int], np.ndarray] = {}
    proto_map: Dict[Tuple[int, int], np.ndarray] = {}

    for c, bucket in enumerate(grouped):
        if not bucket:
            continue

        zs = []
        ws = []
        rhos = []
        for atom in bucket:
            key = (atom.client_id, atom.cls)
            proto_map[key] = np.asarray(atom.z, dtype=np.float64)
            zs.append(proto_map[key])
            ws.append(float(atom.count))
            if atom.rho is not None:
                state_map[key] = _project_psd_trace_one(atom.rho)
                rhos.append(state_map[key])

        class_proto_refs[c] = _weighted_proto_mean(zs, ws)

        if barycenter_mode == "bures":
            if not rhos:
                raise RuntimeError("Bures barycenter requested but no density states were provided.")
            class_state_refs[c] = _bures_barycenter(rhos, ws, max_iter=bary_iters)
        elif barycenter_mode == "euclidean_density":
            if not rhos:
                raise RuntimeError("Euclidean density mean requested but no density states were provided.")
            class_state_refs[c] = _euclidean_density_mean(rhos, ws)
        elif barycenter_mode == "proto_mean":
            pass
        else:
            raise ValueError(f"Unsupported qbary_barycenter_mode={barycenter_mode!r}")

    return class_state_refs, class_proto_refs, state_map, proto_map



def _proto_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))



def _relation_matrix(class_state_refs: Dict[int, np.ndarray], class_proto_refs: Dict[int, np.ndarray], num_classes: int, metric: str) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            if metric == "bures":
                if i in class_state_refs and j in class_state_refs:
                    mat[i, j] = _bures_distance(class_state_refs[i], class_state_refs[j])
            elif metric == "proto_euclidean":
                if i in class_proto_refs and j in class_proto_refs:
                    mat[i, j] = _proto_distance(class_proto_refs[i], class_proto_refs[j])
            else:
                raise ValueError(f"Unsupported relation metric: {metric!r}")
    return mat



def _compute_weights(
    client_ids: Sequence[int],
    client_stats: Sequence[Dict],
    state_map: Dict[Tuple[int, int], np.ndarray],
    proto_map: Dict[Tuple[int, int], np.ndarray],
    class_state_refs: Dict[int, np.ndarray],
    class_proto_refs: Dict[int, np.ndarray],
    size_weights: np.ndarray,
    num_classes: int,
    args,
    logger: StepLogger,
):
    gamma = float(getattr(args, "qbary_weight_gamma", 0.50))
    min_support = float(getattr(args, "qbary_min_class_support", 0.05))
    temp = float(getattr(args, "qbary_weight_temp", 1.5))
    geom_margin_coef = float(getattr(args, "qbary_geom_margin_coef", 1.00))
    logit_clip = float(getattr(args, "qbary_logit_clip", 20.0))
    floor = float(getattr(args, "qbary_weight_floor", 0.02))
    floor = float(np.clip(floor, 0.0, 0.25))
    classwise_blend = float(np.clip(getattr(args, "qbary_classwise_blend", 0.80), 0.0, 1.0))
    geometry_mode = str(getattr(args, "qbary_geometry_mode", "bures")).lower()
    ensemble_mode = str(getattr(args, "qbary_ensemble_mode", "blended")).lower()

    if ensemble_mode == "uniform":
        n_clients = len(client_ids)
        uniform = np.ones((n_clients, num_classes), dtype=np.float64) / max(n_clients, 1)
        logger.log("using uniform client weights")
        return np.ones(n_clients, dtype=np.float64) / max(n_clients, 1), uniform

    global_raw = np.zeros(len(client_ids), dtype=np.float64)
    classwise_raw = np.zeros((len(client_ids), num_classes), dtype=np.float64)

    for k, (cid, stats, sw) in enumerate(zip(client_ids, client_stats, size_weights)):
        counts = stats["class_counts"].numpy().astype(np.float64)
        g_num, g_den = 0.0, 0.0

        for c in range(num_classes):
            key = (int(cid), int(c))
            if counts[c] <= 0:
                continue

            if geometry_mode == "bures":
                if c not in class_state_refs or key not in state_map:
                    continue
                x = state_map[key]
                pos_dist = _bures_distance(x, class_state_refs[c])
                neg_dist = min(
                    [_bures_distance(x, class_state_refs[cp]) for cp in class_state_refs if cp != c] + [2.0]
                )
            elif geometry_mode == "proto_euclidean":
                if c not in class_proto_refs or key not in proto_map:
                    continue
                x = proto_map[key]
                pos_dist = _proto_distance(x, class_proto_refs[c])
                neg_dist = min(
                    [_proto_distance(x, class_proto_refs[cp]) for cp in class_proto_refs if cp != c] + [pos_dist]
                )
            else:
                raise ValueError(f"Unsupported qbary_geometry_mode={geometry_mode!r}")

            geom_margin = neg_dist - pos_dist
            support = max(float(counts[c]), min_support)
            logit = float(geom_margin_coef) * (geom_margin / max(temp, 1e-6))
            logit = float(np.clip(logit, -abs(logit_clip), abs(logit_clip)))
            score = (float(sw) ** 0.5) * (support ** gamma) * math.exp(logit)
            classwise_raw[k, c] = score
            g_num += support * score
            g_den += support

        global_raw[k] = g_num / max(g_den, 1e-12)
        logger.log(f"client={cid} global fusion score={global_raw[k]:.6f}")

    logits = np.log(global_raw + 1e-12) / max(temp, 1e-6)
    logits = logits - np.max(logits)
    global_weights = np.exp(logits)
    global_weights = global_weights / (global_weights.sum() + 1e-12)
    if floor > 0.0:
        global_weights = np.maximum(global_weights, floor)
        global_weights = global_weights / (global_weights.sum() + 1e-12)

    classwise_weights = np.zeros_like(classwise_raw)
    for c in range(num_classes):
        raw = np.clip(classwise_raw[:, c], 0.0, None)
        if raw.sum() <= 1e-12:
            v = global_weights.copy()
        else:
            v = raw / (raw.sum() + 1e-12)
            if floor > 0.0:
                v = np.maximum(v, floor)
                v = v / (v.sum() + 1e-12)
        v = classwise_blend * v + (1.0 - classwise_blend) * global_weights
        v = v / (v.sum() + 1e-12)
        classwise_weights[:, c] = v
        logger.log(f"class={c} classwise weights={np.round(classwise_weights[:, c], 4).tolist()}")

    logger.log(f"global weights={np.round(global_weights, 4).tolist()}")
    return global_weights, classwise_weights


# =============================================================================
# Client
# =============================================================================

class New_AblationClient(BaseClient):
    def __init__(self, args, model, data, client_id: Optional[int] = None):
        super().__init__(args, model, data)
        self.args = args
        self.client_id = client_id
        self.device = torch.device(
            f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = _infer_num_classes(data, args)
        self.best_state = None
        self.stats = None
        self.payload = None
        self._logger = StepLogger(
            f"QBARYENS-CLIENT-{self.client_id if self.client_id is not None else 'NA'}",
            enabled=bool(getattr(args, "qbary_verbose", True)),
        )

    def train(self):
        if self.best_state is not None:
            return 0.0

        best_val = -1.0
        best_state_dict = None
        last_loss = 0.0

        with self._logger.step("one-shot local training"):
            for ep in range(self.args.sf_local_epochs):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                _, out = _extract_model_outputs(self.model, self.data)
                loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.item())

                if (
                    getattr(self.args, "sf_use_best_ckpt", True)
                    and (ep + 1) % int(getattr(self.args, "sf_eval_every", 10)) == 0
                    and self.data.val_mask.sum().item() > 0
                ):
                    self.model.eval()
                    with torch.no_grad():
                        _, out_val = _extract_model_outputs(self.model, self.data)
                    acc = float(
                        (
                            out_val[self.data.val_mask].argmax(1)
                            == self.data.y[self.data.val_mask]
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    if acc > best_val:
                        best_val = acc
                        best_state_dict = {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }

        if getattr(self.args, "sf_use_best_ckpt", True) and best_state_dict is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})

        with self._logger.step("extracting train-only latent prototypes"):
            self.model.eval()
            with torch.no_grad():
                rep, _ = _extract_model_outputs(self.model, self.data)
            self.stats = _compute_train_only_class_stats(rep, self.data, self.num_classes)

        self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        payload = _payload_mb(self.best_state) + _payload_mb(self.stats)
        self._logger.log(
            f"payload to server ≈ {payload:.2f} MB | best_val={best_val:.4f} | final_loss={last_loss:.6f}"
        )
        return last_loss

    def prepare_payload(self, pca_meta: Dict):
        if self.stats is None:
            self.train()

        remove_quantum = bool(getattr(self.args, "qbary_remove_quantum", False))
        proto = self.stats["latent_proto"].numpy().astype(np.float64)
        counts = self.stats["class_counts"].numpy().astype(np.float64)

        reduced_proto_by_class: Dict[int, np.ndarray] = {}
        density_by_class: Dict[int, np.ndarray] = {}

        if not remove_quantum:
            _ensure_qiskit()

        for c in range(proto.shape[0]):
            if counts[c] <= 0:
                continue
            z = _apply_server_pca(proto[c], pca_meta)
            reduced_proto_by_class[int(c)] = z
            if not remove_quantum:
                density_by_class[int(c)] = _encode_reduced_vector_to_density(z, self.args)

        self.payload = {
            "reduced_proto_by_class": reduced_proto_by_class,
            "density_by_class": density_by_class,
            "class_counts": counts.copy(),
        }
        payload_mb = _payload_mb(reduced_proto_by_class) + _payload_mb(density_by_class)
        self._logger.log(f"server-side summary payload ≈ {payload_mb:.2f} MB")
        return self.payload


# =============================================================================
# Server
# =============================================================================

class New_AblationServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super().__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = _infer_num_classes(data, args)
        self.quantum_atlas = None
        self.global_weights = None
        self.classwise_weights = None
        self.final_client_ids = None
        self._ready = False
        self._logger = StepLogger(
            "QBARYENS-SERVER",
            enabled=bool(getattr(args, "qbary_verbose", True)),
        )

    def _client_ids(self):
        if getattr(self, "sampled_clients", None):
            return list(self.sampled_clients)
        return list(self.clients.keys()) if isinstance(self.clients, dict) else list(range(len(self.clients)))

    def _get_client(self, cid):
        return self.clients[cid] if isinstance(self.clients, dict) else self.clients[int(cid)]

    def _atoms_from_client_payloads(
        self,
        client_ids: Sequence[int],
        client_payloads: Sequence[Dict],
        size_weights: np.ndarray,
    ) -> List[ClientAtom]:
        atoms: List[ClientAtom] = []
        for cid, payload, sw in zip(client_ids, client_payloads, size_weights):
            counts = np.asarray(payload["class_counts"], dtype=np.float64)
            proto_map = payload["reduced_proto_by_class"]
            rho_map = payload["density_by_class"]
            for cls, z in proto_map.items():
                atoms.append(
                    ClientAtom(
                        client_id=int(cid),
                        cls=int(cls),
                        count=float(sw) * float(counts[int(cls)]),
                        z=np.asarray(z, dtype=np.float64),
                        rho=None if int(cls) not in rho_map else np.asarray(rho_map[int(cls)], dtype=np.complex128),
                    )
                )
        return atoms

    def aggregate(self):
        remove_quantum = bool(getattr(self.args, "qbary_remove_quantum", False))
        client_ids = self._client_ids()
        sizes = []
        client_stats = []
        total_received_mb = 0.0

        self._logger.log("=" * 60)
        self._logger.log("QBaryEnsemble-DensityGeometryOnly-Ablation: collecting one-shot client payloads")

        with self._logger.step("collecting client classical payloads"):
            for cid in client_ids:
                client = self._get_client(cid)
                if getattr(client, "client_id", None) is None:
                    client.client_id = cid
                if client.best_state is None:
                    client.train()
                sizes.append(int(client.data.train_mask.sum().item()))
                client_stats.append(client.stats)
                total_received_mb += _payload_mb(client.best_state) + _payload_mb(client.stats)

        size_weights = _size_weights(sizes)
        self._logger.log(f"server received classical payload ≈ {total_received_mb:.2f} MB")
        self._logger.log(f"client train sizes={sizes}")
        self._logger.log(f"size weights={np.round(size_weights, 4).tolist()}")

        all_proto = []
        for stats in client_stats:
            proto = stats["latent_proto"].numpy()
            counts = stats["class_counts"].numpy()
            for c in range(proto.shape[0]):
                if counts[c] > 0:
                    all_proto.append(proto[c])
        if not all_proto:
            raise RuntimeError("No class prototypes were extracted from clients.")

        target_pca_dim = _resolve_pca_target_dim(self.args)
        pca_meta = _fit_server_pca(all_proto, target_pca_dim)
        evr = pca_meta["explained_variance_ratio"]
        evr_sum = float(evr.sum()) if evr.size > 0 else 0.0
        self._logger.log(
            f"server PCA input_dim={pca_meta['input_dim']} -> effective_dim={pca_meta['effective_dim']} -> "
            f"target_dim={pca_meta['target_dim']} | explained_variance={evr_sum:.4f}"
        )

        client_payloads = []
        total_summary_mb = 0.0
        with self._logger.step("broadcasting PCA basis and collecting client summaries"):
            for cid in client_ids:
                client = self._get_client(cid)
                payload = client.prepare_payload(pca_meta)
                client_payloads.append(payload)
                total_summary_mb += _payload_mb(payload["reduced_proto_by_class"]) + _payload_mb(payload["density_by_class"])
        self._logger.log(f"server received summary payload ≈ {total_summary_mb:.2f} MB")

        atoms = self._atoms_from_client_payloads(
            client_ids=client_ids,
            client_payloads=client_payloads,
            size_weights=size_weights,
        )

        if remove_quantum:
            self._logger.log("running no-quantum ablation path")
            bary_mode = "proto_mean"
            geom_mode = "proto_euclidean"
            relation_metric = "proto_euclidean"
        else:
            bary_mode = str(getattr(self.args, "qbary_barycenter_mode", "bures")).lower()
            geom_mode = str(getattr(self.args, "qbary_geometry_mode", "bures")).lower()
            relation_metric = "bures" if bary_mode in {"bures", "euclidean_density"} else "proto_euclidean"

        class_state_refs, class_proto_refs, state_map, proto_map = _compute_reference_sets(
            atoms=atoms,
            num_classes=self.num_classes,
            barycenter_mode=bary_mode,
            bary_iters=int(getattr(self.args, "qbary_bary_iters", 40)),
        )

        # Make sure geometry mode follows the explicit no-quantum definition.
        setattr(self.args, "qbary_geometry_mode", geom_mode)

        relation = _relation_matrix(
            class_state_refs=class_state_refs,
            class_proto_refs=class_proto_refs,
            num_classes=self.num_classes,
            metric=relation_metric,
        )

        global_weights, classwise_weights = _compute_weights(
            client_ids=client_ids,
            client_stats=client_stats,
            state_map=state_map,
            proto_map=proto_map,
            class_state_refs=class_state_refs,
            class_proto_refs=class_proto_refs,
            size_weights=size_weights,
            num_classes=self.num_classes,
            args=self.args,
            logger=self._logger,
        )

        self.quantum_atlas = {
            "class_bary": class_state_refs,
            "class_proto_refs": class_proto_refs,
            "relation": relation,
            "relation_metric": relation_metric,
            "pca_mean": pca_meta["mean"],
            "pca_components": pca_meta["components"],
            "pca_explained_variance_ratio": pca_meta["explained_variance_ratio"],
            "pca_effective_dim": pca_meta["effective_dim"],
            "pca_target_dim": pca_meta["target_dim"],
            "state_encoding": None if remove_quantum else "amplitude",
            "barycenter_mode": bary_mode,
            "geometry_mode": geom_mode,
        }
        self.global_weights = np.asarray(global_weights, dtype=np.float64)
        self.classwise_weights = np.asarray(classwise_weights, dtype=np.float64)
        self.final_client_ids = list(client_ids)
        self._ready = True

        self._logger.log("QBaryEnsemble-DensityGeometryOnly-Ablation: done")
        self._logger.log("=" * 60)

    @torch.no_grad()
    def _ensemble_predict_proba(self, data, mask: torch.Tensor) -> torch.Tensor:
        if self.final_client_ids is None or self.classwise_weights is None:
            raise RuntimeError("Ensemble weights unavailable. Call aggregate() first.")

        probs_acc = None
        classwise = torch.tensor(self.classwise_weights, dtype=torch.float32, device=self.device)
        with self._logger.step("running client ensemble"):
            for k, cid in enumerate(self.final_client_ids):
                client = self._get_client(cid)
                client.model.eval()
                _, out = _extract_model_outputs(client.model, data)
                prob = torch.softmax(out.float()[mask], dim=1)
                contrib = prob * classwise[k].view(1, -1)
                probs_acc = contrib if probs_acc is None else (probs_acc + contrib)

        probs_acc = probs_acc / probs_acc.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return probs_acc

    def global_evaluate(self):
        if not self._ready:
            self.aggregate()

        mask = self.data.test_mask
        y_true = self.data.y[mask].cpu()
        with torch.no_grad():
            y_prob = self._ensemble_predict_proba(self.data, mask).detach().cpu()
        y_pred = y_prob.argmax(1)

        loss = F.nll_loss(torch.log(y_prob.clamp_min(1e-12)), y_true)
        acc = float((y_pred == y_true).float().mean().item())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.numpy(), y_pred.numpy(), average="macro", zero_division=0
        )
        cm = confusion_matrix(
            y_true.numpy(),
            y_pred.numpy(),
            labels=list(range(self.num_classes)),
        )

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


# =============================================================================
# Defaults / argparse-friendly knobs
# =============================================================================


def add_qbaryensemble_defaults(args):
    defaults = {
        "qbary_verbose": True,
        "qbary_n_qubits": 3,
        "qbary_bary_iters": 40,
        "qbary_weight_gamma": 0.50,
        "qbary_min_class_support": 0.05,
        "qbary_weight_temp": 1.5,
        "qbary_weight_floor": 0.02,
        "qbary_geom_margin_coef": 1.00,
        "qbary_logit_clip": 20.0,
        "qbary_classwise_blend": 0.80,
        "qbary_pca_dim": None,
        "qbary_remove_quantum": False,
        "qbary_barycenter_mode": "bures",
        "qbary_geometry_mode": "bures",
        "qbary_ensemble_mode": "blended",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args

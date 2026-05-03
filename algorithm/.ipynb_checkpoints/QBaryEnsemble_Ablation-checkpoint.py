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
    from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp
except Exception as e:  # pragma: no cover
    QuantumCircuit = None
    StatePreparation = None
    DensityMatrix = None
    Operator = None
    SparsePauliOp = None
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
class QuantumAtom:
    client_id: int
    cls: int
    count: float
    rho: np.ndarray
    z: np.ndarray


# =============================================================================
# Utilities
# =============================================================================


def _ensure_qiskit():
    if (
        QuantumCircuit is None
        or StatePreparation is None
        or DensityMatrix is None
        or Operator is None
        or SparsePauliOp is None
    ):
        raise ImportError(
            "Qiskit is required for QBaryEnsemble. "
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
    vars_ = torch.zeros(num_classes, rep.size(1), dtype=torch.float32, device=rep.device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=rep.device)

    for c in range(num_classes):
        mask = data.train_mask & (data.y == c)
        n = int(mask.sum().item())
        counts[c] = n
        if n <= 0:
            continue
        rc = rep[mask]
        means[c] = rc.mean(0)
        vars_[c] = rc.var(0, unbiased=False) if n > 1 else torch.zeros_like(means[c])

    return {
        "latent_proto": means.cpu(),
        "latent_var": vars_.cpu(),
        "class_counts": counts.cpu(),
    }


# =============================================================================
# Ablation helpers
# =============================================================================


def _get_ablation_config(args) -> Dict[str, object]:
    cfg = {
        "remove_quantum": bool(getattr(args, "qbary_remove_quantum", False)),
        "use_quantum_lens": bool(getattr(args, "qbary_use_quantum_lens", True)),
        "barycenter_mode": str(getattr(args, "qbary_barycenter_mode", "bures")).lower(),
        "geometry_mode": str(getattr(args, "qbary_geometry_mode", "bures")).lower(),
        "objective_mode": str(getattr(args, "qbary_objective_mode", "full")).lower(),
        "variance_mode": str(getattr(args, "qbary_variance_mode", "full")).lower(),
        "ensemble_mode": str(getattr(args, "qbary_ensemble_mode", "blended")).lower(),
        "use_geom_margin": bool(getattr(args, "qbary_use_geom_margin", True)),
        "use_feature_margin": bool(getattr(args, "qbary_use_feature_margin", True)),
        "use_variance_margin": bool(getattr(args, "qbary_use_variance_margin", True)),
        "use_variance_conf": bool(getattr(args, "qbary_use_variance_conf", True)),
        "use_support_term": bool(getattr(args, "qbary_use_support_term", True)),
    }

    if cfg["remove_quantum"]:
        # Force a fully classical path on reduced prototypes.
        cfg["use_quantum_lens"] = False
        cfg["barycenter_mode"] = "reduced_proto_mean"
        cfg["geometry_mode"] = "proto_euclidean"
        cfg["use_feature_margin"] = False
        setattr(args, "qbary_use_quantum_lens", False)
        setattr(args, "qbary_barycenter_mode", "reduced_proto_mean")
        setattr(args, "qbary_geometry_mode", "proto_euclidean")
        setattr(args, "qbary_use_feature_margin", False)

    variance_mode = cfg["variance_mode"]
    if variance_mode == "means_only":
        cfg["use_variance_margin"] = False
        cfg["use_variance_conf"] = False
    elif variance_mode == "conf_only":
        cfg["use_variance_margin"] = False
        cfg["use_variance_conf"] = True
    elif variance_mode == "margin_only":
        cfg["use_variance_margin"] = True
        cfg["use_variance_conf"] = False
    elif variance_mode != "full":
        raise ValueError(
            f"Unsupported qbary_variance_mode={variance_mode!r}. "
            "Use 'full', 'means_only', 'conf_only', or 'margin_only'."
        )

    if cfg["geometry_mode"] not in {"bures", "proto_euclidean", "proto_cosine"}:
        raise ValueError(
            f"Unsupported qbary_geometry_mode={cfg['geometry_mode']!r}. "
            "Use 'bures', 'proto_euclidean', or 'proto_cosine'."
        )
    if cfg["barycenter_mode"] not in {
        "bures",
        "euclidean_density",
        "density_medoid",
        "reduced_proto_mean",
        "reduced_proto_medoid",
    }:
        raise ValueError(
            f"Unsupported qbary_barycenter_mode={cfg['barycenter_mode']!r}."
        )
    if cfg["objective_mode"] not in {
        "within_only",
        "within_between",
        "within_between_entropy",
        "full",
    }:
        raise ValueError(
            f"Unsupported qbary_objective_mode={cfg['objective_mode']!r}."
        )
    if cfg["ensemble_mode"] not in {
        "uniform",
        "size_only",
        "global_only",
        "classwise_only",
        "blended",
    }:
        raise ValueError(
            f"Unsupported qbary_ensemble_mode={cfg['ensemble_mode']!r}."
        )

    if not cfg["use_quantum_lens"]:
        # No trainable lens -> identity transform, no SPSA training.
        setattr(args, "qbary_pqc_steps", 0)

    return cfg


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


def _weighted_density_medoid(rhos: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(rhos) == 1:
        return _project_psd_trace_one(rhos[0])
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    costs = []
    fixed = [_project_psd_trace_one(r) for r in rhos]
    for i, ri in enumerate(fixed):
        cost = 0.0
        for wj, rj in zip(weights, fixed):
            cost += float(wj) * float(np.linalg.norm(ri - rj, ord="fro"))
        costs.append(cost)
    return fixed[int(np.argmin(costs))]


def _weighted_proto_mean(zs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    z = np.sum(np.asarray(zs, dtype=np.float64) * weights[:, None], axis=0)
    return np.asarray(z, dtype=np.float64)


def _weighted_proto_medoid(zs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(zs) == 1:
        return np.asarray(zs[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    zs = [np.asarray(z, dtype=np.float64) for z in zs]
    costs = []
    for zi in zs:
        cost = 0.0
        for wj, zj in zip(weights, zs):
            cost += float(wj) * float(np.linalg.norm(zi - zj))
        costs.append(cost)
    return zs[int(np.argmin(costs))]


# =============================================================================
# Server PCA and client quantum encoding
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
    encoding = str(getattr(args, "qbary_state_encoding", "amplitude")).lower()
    if encoding == "amplitude":
        return min(user_dim, state_dim)
    return user_dim


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


def _apply_server_pca_diag_var(var_vec: np.ndarray, pca_meta: Dict) -> np.ndarray:
    v = np.asarray(var_vec, dtype=np.float64).ravel()
    comps = np.asarray(pca_meta["components"], dtype=np.float64)
    red_var = (comps ** 2) @ np.clip(v, 0.0, None)

    target_dim = int(pca_meta["target_dim"])
    if red_var.size < target_dim:
        red_var = np.pad(red_var, (0, target_dim - red_var.size), mode="constant")
    elif red_var.size > target_dim:
        red_var = red_var[:target_dim]
    return red_var.astype(np.float64, copy=False)


def _safe_standardize(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        return np.zeros(1, dtype=np.float64)
    mu = float(np.mean(z))
    sd = float(np.std(z))
    if sd <= 1e-12:
        return z - mu
    return (z - mu) / sd


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


def _build_client_angle_encoding_circuit(
    z: np.ndarray, n_qubits: int, layers: int
) -> "QuantumCircuit":
    _ensure_qiskit()
    vals = _safe_standardize(z)
    qc = QuantumCircuit(n_qubits)
    cursor = 0
    for _ in range(max(int(layers), 1)):
        for q in range(n_qubits):
            v = float(vals[cursor % len(vals)])
            qc.ry(float(np.tanh(v)), q)
            cursor += 1
        for q in range(n_qubits):
            v = float(vals[cursor % len(vals)])
            qc.rz(float(np.tanh(v)), q)
            cursor += 1
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    return qc


def _encode_reduced_vector_to_density(z: np.ndarray, args) -> np.ndarray:
    _ensure_qiskit()
    encoding = str(getattr(args, "qbary_state_encoding", "amplitude")).lower()
    n_qubits = int(getattr(args, "qbary_n_qubits", 3))

    if encoding == "amplitude":
        qc = _build_client_amplitude_encoding_circuit(z, n_qubits)
    elif encoding == "angle":
        qc = _build_client_angle_encoding_circuit(
            z=z,
            n_qubits=n_qubits,
            layers=int(getattr(args, "qbary_angle_layers", 2)),
        )
    else:
        raise ValueError(
            f"Unsupported qbary_state_encoding={encoding!r}. Use 'amplitude' or 'angle'."
        )

    rho = np.asarray(DensityMatrix.from_instruction(qc).data, dtype=np.complex128)
    return _project_psd_trace_one(rho)


# =============================================================================
# Server PQC lens and observable features
# =============================================================================


def _build_server_pqc(n_qubits: int, depth: int, theta: np.ndarray) -> "QuantumCircuit":
    _ensure_qiskit()
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.ry(float(theta[idx]), q)
            idx += 1
            qc.rz(float(theta[idx]), q)
            idx += 1
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.rx(float(theta[idx]), q)
            idx += 1
    return qc


def _unitary_from_theta(n_qubits: int, depth: int, theta: np.ndarray) -> np.ndarray:
    qc = _build_server_pqc(n_qubits, depth, theta)
    return np.asarray(Operator(qc).data, dtype=np.complex128)


def _observable_bank(n_qubits: int) -> List[np.ndarray]:
    _ensure_qiskit()
    bank: List[np.ndarray] = []

    for i in range(n_qubits):
        for p in ["X", "Y", "Z"]:
            chars = ["I"] * n_qubits
            chars[i] = p
            pauli_str = "".join(reversed(chars))
            bank.append(SparsePauliOp.from_list([(pauli_str, 1.0)]).to_matrix())

    if n_qubits > 1:
        for i in range(n_qubits - 1):
            for p in ["XX", "YY", "ZZ"]:
                chars = ["I"] * n_qubits
                chars[i] = p[0]
                chars[i + 1] = p[1]
                pauli_str = "".join(reversed(chars))
                bank.append(SparsePauliOp.from_list([(pauli_str, 1.0)]).to_matrix())

    return [np.asarray(o, dtype=np.complex128) for o in bank]


def _apply_unitary_to_density(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    return _project_psd_trace_one(u @ rho @ u.conj().T)


def _measurement_features_from_state(
    rho: np.ndarray, observables: Sequence[np.ndarray]
) -> np.ndarray:
    feats = [float(np.real(np.trace(rho @ obs))) for obs in observables]
    return np.asarray(feats, dtype=np.float64)


def _safe_cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _variance_confidence(v: np.ndarray) -> float:
    v = np.log1p(np.clip(np.asarray(v, dtype=np.float64), 0.0, None))
    return 1.0 / (1.0 + float(np.mean(v)))


def _variance_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.log1p(np.clip(np.asarray(a, dtype=np.float64), 0.0, None))
    b = np.log1p(np.clip(np.asarray(b, dtype=np.float64), 0.0, None))
    return float(np.mean((a - b) ** 2))


def _group_atoms_by_class(
    atoms: Sequence[QuantumAtom], num_classes: int
) -> List[List[QuantumAtom]]:
    grouped: List[List[QuantumAtom]] = [[] for _ in range(num_classes)]
    for atom in atoms:
        grouped[int(atom.cls)].append(atom)
    return grouped


def _pairwise_proto_relation_matrix(class_proto_refs: Dict[int, np.ndarray], metric: str, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        if i not in class_proto_refs:
            continue
        for j in range(num_classes):
            if j not in class_proto_refs:
                continue
            if metric == "proto_cosine":
                mat[i, j] = 1.0 - _safe_cosine_np(class_proto_refs[i], class_proto_refs[j])
            else:
                mat[i, j] = float(np.linalg.norm(class_proto_refs[i] - class_proto_refs[j]))
    return mat


def _quantum_relation_matrix(
    class_bary: Dict[int, np.ndarray], num_classes: int
) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        if i not in class_bary:
            continue
        for j in range(num_classes):
            if j not in class_bary:
                continue
            mat[i, j] = _bures_distance(class_bary[i], class_bary[j])
    return mat


# =============================================================================
# Score matrices
# =============================================================================


def _compute_class_reference_sets(
    grouped: Sequence[Sequence[QuantumAtom]],
    transformed_states: Dict[Tuple[int, int], np.ndarray],
    reduced_proto_map: Dict[Tuple[int, int], np.ndarray],
    barycenter_mode: str,
    bary_iters: int,
):
    class_state_refs: Dict[int, np.ndarray] = {}
    class_proto_refs: Dict[int, np.ndarray] = {}
    class_state_weights: Dict[int, np.ndarray] = {}

    density_modes = {"bures", "euclidean_density", "density_medoid"}

    for c, bucket in enumerate(grouped):
        if not bucket:
            continue
        ws = np.asarray([float(atom.count) for atom in bucket], dtype=np.float64)
        ws = ws / (ws.sum() + 1e-12)
        zs = [reduced_proto_map[(atom.client_id, atom.cls)] for atom in bucket]

        # Important for --qbary_remove_quantum and any reduced-prototype ablation:
        # only touch transformed_states when the chosen aggregation mode actually
        # needs density matrices. Otherwise classical runs would KeyError here
        # because transformed_states is intentionally empty.
        if barycenter_mode in density_modes:
            missing = [
                (atom.client_id, atom.cls)
                for atom in bucket
                if (atom.client_id, atom.cls) not in transformed_states
            ]
            if missing:
                raise KeyError(
                    "Density-based barycenter mode requires transformed states, "
                    f"but they were missing for keys={missing[:5]}"
                )
            rhos = [transformed_states[(atom.client_id, atom.cls)] for atom in bucket]

        if barycenter_mode == "bures":
            class_state_refs[c] = _bures_barycenter(rhos, ws, max_iter=bary_iters)
            class_proto_refs[c] = _weighted_proto_mean(zs, ws)
        elif barycenter_mode == "euclidean_density":
            class_state_refs[c] = _euclidean_density_mean(rhos, ws)
            class_proto_refs[c] = _weighted_proto_mean(zs, ws)
        elif barycenter_mode == "density_medoid":
            class_state_refs[c] = _weighted_density_medoid(rhos, ws)
            class_proto_refs[c] = _weighted_proto_mean(zs, ws)
        elif barycenter_mode == "reduced_proto_mean":
            class_proto_refs[c] = _weighted_proto_mean(zs, ws)
        elif barycenter_mode == "reduced_proto_medoid":
            class_proto_refs[c] = _weighted_proto_medoid(zs, ws)
        else:
            raise ValueError(f"Unsupported barycenter_mode={barycenter_mode!r}")
        class_state_weights[c] = ws

    return class_state_refs, class_proto_refs, class_state_weights


def _geometry_margin_for_client(
    key: Tuple[int, int],
    cls: int,
    transformed_states: Dict[Tuple[int, int], np.ndarray],
    reduced_proto_map: Dict[Tuple[int, int], np.ndarray],
    class_state_refs: Dict[int, np.ndarray],
    class_proto_refs: Dict[int, np.ndarray],
    geometry_mode: str,
) -> float:
    if geometry_mode == "bures":
        if cls not in class_state_refs or key not in transformed_states:
            return 0.0
        rho = transformed_states[key]
        pos = _bures_distance(rho, class_state_refs[cls])
        neg = min(
            [_bures_distance(rho, class_state_refs[c2]) for c2 in class_state_refs if c2 != cls] + [2.0]
        )
        return float(neg - pos)

    z = reduced_proto_map[key]
    if cls not in class_proto_refs:
        return 0.0
    if geometry_mode == "proto_euclidean":
        pos = float(np.linalg.norm(z - class_proto_refs[cls]))
        neg = min(
            [float(np.linalg.norm(z - class_proto_refs[c2])) for c2 in class_proto_refs if c2 != cls] + [pos]
        )
        return float(neg - pos)
    if geometry_mode == "proto_cosine":
        pos = _safe_cosine_np(z, class_proto_refs[cls])
        neg = max(
            [_safe_cosine_np(z, class_proto_refs[c2]) for c2 in class_proto_refs if c2 != cls] + [0.0]
        )
        return float(pos - neg)
    raise ValueError(f"Unsupported geometry_mode={geometry_mode!r}")


def _compute_score_matrices(
    client_ids: Sequence[int],
    client_stats: Sequence[Dict],
    transformed_states: Dict[Tuple[int, int], np.ndarray],
    feature_map: Dict[Tuple[int, int], np.ndarray],
    reduced_proto_map: Dict[Tuple[int, int], np.ndarray],
    class_state_refs: Dict[int, np.ndarray],
    class_proto_refs: Dict[int, np.ndarray],
    reduced_var_map: Dict[Tuple[int, int], np.ndarray],
    size_weights: np.ndarray,
    num_classes: int,
    args,
    ab_cfg: Dict[str, object],
):
    gamma = float(getattr(args, "qbary_weight_gamma", 0.50))
    min_support = float(getattr(args, "qbary_min_class_support", 0.05))
    temp = float(getattr(args, "qbary_weight_temp", 1.5))
    feature_margin_coef = float(getattr(args, "qbary_feature_margin_coef", 0.50))
    geom_margin_coef = float(getattr(args, "qbary_geom_margin_coef", 1.00))
    variance_margin_coef = float(getattr(args, "qbary_variance_margin_coef", 0.35))
    variance_conf_coef = float(getattr(args, "qbary_variance_conf_coef", 0.15))
    logit_clip = float(getattr(args, "qbary_logit_clip", 20.0))

    class_centers: Dict[int, np.ndarray] = {}
    class_var_centers: Dict[int, np.ndarray] = {}

    for c in range(num_classes):
        feats = []
        vars_: List[np.ndarray] = []
        ws = []
        var_ws: List[float] = []
        for cid, stats, sw in zip(client_ids, client_stats, size_weights):
            counts = stats["class_counts"].numpy().astype(np.float64)
            key = (int(cid), int(c))
            if counts[c] <= 0 or key not in feature_map:
                continue
            w = float(sw) * float(counts[c])
            feats.append(feature_map[key])
            ws.append(w)
            if key in reduced_var_map:
                vars_.append(reduced_var_map[key])
                var_ws.append(w)

        if feats:
            ws_arr = np.asarray(ws, dtype=np.float64)
            ws_arr = ws_arr / (ws_arr.sum() + 1e-12)
            class_centers[c] = np.sum(np.asarray(feats) * ws_arr[:, None], axis=0)
            if vars_:
                var_ws_arr = np.asarray(var_ws, dtype=np.float64)
                var_ws_arr = var_ws_arr / (var_ws_arr.sum() + 1e-12)
                class_var_centers[c] = np.sum(
                    np.asarray(vars_) * var_ws_arr[:, None], axis=0
                )

    global_raw = np.zeros(len(client_ids), dtype=np.float64)
    classwise_raw = np.zeros((len(client_ids), num_classes), dtype=np.float64)

    for k, (cid, stats, sw) in enumerate(zip(client_ids, client_stats, size_weights)):
        counts = stats["class_counts"].numpy().astype(np.float64)
        g_num, g_den = 0.0, 0.0

        for c in range(num_classes):
            key = (int(cid), int(c))
            if counts[c] <= 0 or key not in feature_map:
                continue
            if ab_cfg["geometry_mode"] == "bures" and c not in class_state_refs:
                continue
            if ab_cfg["geometry_mode"] != "bures" and c not in class_proto_refs:
                continue

            feat = feature_map[key]

            geom_margin = _geometry_margin_for_client(
                key=key,
                cls=c,
                transformed_states=transformed_states,
                reduced_proto_map=reduced_proto_map,
                class_state_refs=class_state_refs,
                class_proto_refs=class_proto_refs,
                geometry_mode=str(ab_cfg["geometry_mode"]),
            )

            feat_pos = _safe_cosine_np(feat, class_centers.get(c, feat))
            feat_neg = max(
                [_safe_cosine_np(feat, class_centers[cp]) for cp in class_centers if cp != c] + [0.0]
            )
            feat_margin = feat_pos - feat_neg

            var_margin = 0.0
            var_conf = 1.0
            if key in reduced_var_map:
                z_var = reduced_var_map[key]
                if bool(ab_cfg["use_variance_conf"]):
                    var_conf = _variance_confidence(z_var)
                if bool(ab_cfg["use_variance_margin"]) and c in class_var_centers:
                    pos_var = _variance_distance(z_var, class_var_centers[c])
                    neg_var = min(
                        [
                            _variance_distance(z_var, class_var_centers[cp])
                            for cp in class_var_centers
                            if cp != c
                        ]
                        + [pos_var]
                    )
                    var_margin = neg_var - pos_var

            support = max(float(counts[c]), min_support) if bool(ab_cfg["use_support_term"]) else 1.0
            support_factor = (support ** gamma) if bool(ab_cfg["use_support_term"]) else 1.0
            logit = 0.0
            if bool(ab_cfg["use_geom_margin"]):
                logit += float(geom_margin_coef) * (geom_margin / max(temp, 1e-6))
            if bool(ab_cfg["use_feature_margin"]):
                logit += float(feature_margin_coef) * feat_margin
            if bool(ab_cfg["use_variance_margin"]):
                logit += float(variance_margin_coef) * (var_margin / max(temp, 1e-6))
            if bool(ab_cfg["use_variance_conf"]):
                logit += float(variance_conf_coef) * np.log(var_conf + 1e-12)
            logit = float(np.clip(logit, -abs(logit_clip), abs(logit_clip)))

            score = (float(sw) ** 0.5) * float(support_factor) * math.exp(logit)
            classwise_raw[k, c] = score
            g_num += support * score
            g_den += support

        global_raw[k] = g_num / max(g_den, 1e-12)

    return global_raw, classwise_raw


def _normalized_weight_entropy(classwise_raw: np.ndarray) -> float:
    entropies = []
    raw = np.asarray(classwise_raw, dtype=np.float64)
    if raw.ndim != 2:
        return 0.0
    for c in range(raw.shape[1]):
        col = np.clip(raw[:, c], 0.0, None)
        active = col > 1e-12
        n_active = int(active.sum())
        if n_active <= 1:
            continue
        p = col[active]
        p = p / (p.sum() + 1e-12)
        ent = -float(np.sum(p * np.log(p + 1e-12))) / max(np.log(float(n_active)), 1e-12)
        entropies.append(ent)
    if not entropies:
        return 0.0
    return float(np.mean(entropies))


# =============================================================================
# Server quantum objective and PQC training
# =============================================================================


def _server_quantum_objective(
    theta: np.ndarray,
    atoms: Sequence[QuantumAtom],
    client_ids: Sequence[int],
    client_stats: Sequence[Dict],
    reduced_proto_map: Dict[Tuple[int, int], np.ndarray],
    reduced_var_map: Dict[Tuple[int, int], np.ndarray],
    size_weights: np.ndarray,
    args,
    num_classes: int,
    n_qubits: int,
    depth: int,
    observables: Sequence[np.ndarray],
    between_coef: float,
    weight_entropy_coef: float,
    bary_iters: int,
    ab_cfg: Dict[str, object],
):
    classical_only = bool(ab_cfg.get("remove_quantum", False))
    use_quantum_lens = bool(ab_cfg["use_quantum_lens"]) and not classical_only
    if use_quantum_lens and depth > 0 and theta.size > 0:
        u = _unitary_from_theta(n_qubits, depth, theta)
    else:
        u = np.eye(2 ** n_qubits, dtype=np.complex128) if not classical_only else None

    grouped = _group_atoms_by_class(atoms, num_classes)

    transformed_states: Dict[Tuple[int, int], np.ndarray] = {}
    feature_map: Dict[Tuple[int, int], np.ndarray] = {}
    within = 0.0
    total_w = 0.0

    for bucket in grouped:
        for atom in bucket:
            key = (atom.client_id, atom.cls)
            if classical_only:
                feature_map[key] = np.asarray(reduced_proto_map[key], dtype=np.float64)
            else:
                transformed = _apply_unitary_to_density(atom.rho, u)
                transformed_states[key] = transformed
                feature_map[key] = _measurement_features_from_state(transformed, observables)

    class_state_refs, class_proto_refs, _ = _compute_class_reference_sets(
        grouped=grouped,
        transformed_states=transformed_states,
        reduced_proto_map=reduced_proto_map,
        barycenter_mode=str(ab_cfg["barycenter_mode"]),
        bary_iters=bary_iters,
    )

    class_feature_centers: Dict[int, np.ndarray] = {}
    for c, bucket in enumerate(grouped):
        if not bucket:
            continue
        ws = np.asarray([float(atom.count) for atom in bucket], dtype=np.float64)
        ws = ws / (ws.sum() + 1e-12)
        feats = [feature_map[(atom.client_id, atom.cls)] for atom in bucket]
        center = np.sum(np.asarray(feats) * ws[:, None], axis=0)
        class_feature_centers[c] = center

        for atom, w, feat in zip(bucket, ws, feats):
            key = (atom.client_id, atom.cls)
            conf = 1.0
            if bool(ab_cfg["use_variance_conf"]) and key in reduced_var_map:
                conf = _variance_confidence(reduced_var_map[key])
            within += float(w) * conf * float(np.mean((feat - center) ** 2))
            total_w += float(w) * conf

    within = within / max(total_w, 1e-12)

    between_terms = []
    keys = sorted(class_feature_centers.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = _safe_cosine_np(class_feature_centers[keys[i]], class_feature_centers[keys[j]])
            between_terms.append(sim)
    between = float(np.mean(between_terms)) if between_terms else 0.0

    class_norm = [float(np.linalg.norm(class_feature_centers[c])) for c in keys]
    norm_reg = float(np.mean([(n - 1.0) ** 2 for n in class_norm])) if class_norm else 0.0

    _, classwise_raw = _compute_score_matrices(
        client_ids=client_ids,
        client_stats=client_stats,
        transformed_states=transformed_states,
        feature_map=feature_map,
        reduced_proto_map=reduced_proto_map,
        class_state_refs=class_state_refs,
        class_proto_refs=class_proto_refs,
        reduced_var_map=reduced_var_map,
        size_weights=size_weights,
        num_classes=num_classes,
        args=args,
        ab_cfg=ab_cfg,
    )
    weight_entropy = _normalized_weight_entropy(classwise_raw)

    objective_mode = str(ab_cfg["objective_mode"])
    loss = within
    if objective_mode in {"within_between", "within_between_entropy", "full"}:
        loss += float(between_coef) * between
    if objective_mode in {"within_between_entropy", "full"}:
        loss += float(weight_entropy_coef) * weight_entropy
    if objective_mode == "full":
        loss += 0.05 * norm_reg

    return loss, class_state_refs, class_proto_refs, transformed_states, feature_map


def _train_server_quantum_lens(
    atoms: Sequence[QuantumAtom],
    client_ids: Sequence[int],
    client_stats: Sequence[Dict],
    reduced_proto_map: Dict[Tuple[int, int], np.ndarray],
    reduced_var_map: Dict[Tuple[int, int], np.ndarray],
    size_weights: np.ndarray,
    args,
    num_classes: int,
    n_qubits: int,
    depth: int,
    train_steps: int,
    spsa_a: float,
    spsa_c: float,
    spsa_alpha: float,
    spsa_gamma: float,
    between_coef: float,
    weight_entropy_coef: float,
    bary_iters: int,
    logger: StepLogger,
    ab_cfg: Dict[str, object],
    eval_every: int = 5,
):
    classical_only = bool(ab_cfg.get("remove_quantum", False))
    if classical_only:
        observables = []
    else:
        _ensure_qiskit()
        observables = _observable_bank(n_qubits)

    if classical_only or not bool(ab_cfg["use_quantum_lens"]):
        theta = np.zeros(0, dtype=np.float64)
        loss, class_state_refs, class_proto_refs, transformed_states, feature_map = _server_quantum_objective(
            theta=theta,
            atoms=atoms,
            client_ids=client_ids,
            client_stats=client_stats,
            reduced_proto_map=reduced_proto_map,
            reduced_var_map=reduced_var_map,
            size_weights=size_weights,
            args=args,
            num_classes=num_classes,
            n_qubits=n_qubits,
            depth=0,
            observables=observables,
            between_coef=between_coef,
            weight_entropy_coef=weight_entropy_coef,
            bary_iters=bary_iters,
            ab_cfg=ab_cfg,
        )
        logger.log(f"classical path active | objective={loss:.6f}" if classical_only else f"quantum lens disabled | objective={loss:.6f}")
        return theta, class_state_refs, class_proto_refs, transformed_states, feature_map

    p = depth * n_qubits * 3
    theta = np.random.uniform(-0.1, 0.1, size=p).astype(np.float64)

    best_loss, best_state_refs, best_proto_refs, best_trans, best_feats = _server_quantum_objective(
        theta=theta,
        atoms=atoms,
        client_ids=client_ids,
        client_stats=client_stats,
        reduced_proto_map=reduced_proto_map,
        reduced_var_map=reduced_var_map,
        size_weights=size_weights,
        args=args,
        num_classes=num_classes,
        n_qubits=n_qubits,
        depth=depth,
        observables=observables,
        between_coef=between_coef,
        weight_entropy_coef=weight_entropy_coef,
        bary_iters=bary_iters,
        ab_cfg=ab_cfg,
    )
    best_theta = theta.copy()
    logger.log(
        f"initial quantum objective={best_loss:.6f} supported_classes={len(best_proto_refs)}/{num_classes}"
    )

    total_steps = max(int(train_steps), 1)

    with logger.step("training server PQC via SPSA"):
        for t in range(total_steps):
            ak = float(spsa_a / ((t + 1) ** max(spsa_alpha, 1e-6)))
            ck = float(spsa_c / ((t + 1) ** max(spsa_gamma, 1e-6)))
            delta = np.random.choice([-1.0, 1.0], size=p)

            l_plus, _, _, _, _ = _server_quantum_objective(
                theta=theta + ck * delta,
                atoms=atoms,
                client_ids=client_ids,
                client_stats=client_stats,
                reduced_proto_map=reduced_proto_map,
                reduced_var_map=reduced_var_map,
                size_weights=size_weights,
                args=args,
                num_classes=num_classes,
                n_qubits=n_qubits,
                depth=depth,
                observables=observables,
                between_coef=between_coef,
                weight_entropy_coef=weight_entropy_coef,
                bary_iters=bary_iters,
                ab_cfg=ab_cfg,
            )
            l_minus, _, _, _, _ = _server_quantum_objective(
                theta=theta - ck * delta,
                atoms=atoms,
                client_ids=client_ids,
                client_stats=client_stats,
                reduced_proto_map=reduced_proto_map,
                reduced_var_map=reduced_var_map,
                size_weights=size_weights,
                args=args,
                num_classes=num_classes,
                n_qubits=n_qubits,
                depth=depth,
                observables=observables,
                between_coef=between_coef,
                weight_entropy_coef=weight_entropy_coef,
                bary_iters=bary_iters,
                ab_cfg=ab_cfg,
            )

            ghat = ((l_plus - l_minus) / (2.0 * ck + 1e-12)) * delta
            theta = theta - ak * ghat
            theta = ((theta + np.pi) % (2.0 * np.pi)) - np.pi

            is_last = (t == total_steps - 1)
            if (t + 1) % eval_every == 0 or is_last:
                curr_loss, curr_state_refs, curr_proto_refs, curr_trans, curr_feats = _server_quantum_objective(
                    theta=theta,
                    atoms=atoms,
                    client_ids=client_ids,
                    client_stats=client_stats,
                    reduced_proto_map=reduced_proto_map,
                    reduced_var_map=reduced_var_map,
                    size_weights=size_weights,
                    args=args,
                    num_classes=num_classes,
                    n_qubits=n_qubits,
                    depth=depth,
                    observables=observables,
                    between_coef=between_coef,
                    weight_entropy_coef=weight_entropy_coef,
                    bary_iters=bary_iters,
                    ab_cfg=ab_cfg,
                )
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_theta = theta.copy()
                    best_state_refs = curr_state_refs
                    best_proto_refs = curr_proto_refs
                    best_trans = curr_trans
                    best_feats = curr_feats

    logger.log(f"best quantum objective={best_loss:.6f}")
    return best_theta, best_state_refs, best_proto_refs, best_trans, best_feats


# =============================================================================
# Global weight computation
# =============================================================================


def _compute_final_weights(
    client_ids: Sequence[int],
    client_stats: Sequence[Dict],
    transformed_states: Dict[Tuple[int, int], np.ndarray],
    feature_map: Dict[Tuple[int, int], np.ndarray],
    reduced_proto_map: Dict[Tuple[int, int], np.ndarray],
    class_state_refs: Dict[int, np.ndarray],
    class_proto_refs: Dict[int, np.ndarray],
    reduced_var_map: Dict[Tuple[int, int], np.ndarray],
    size_weights: np.ndarray,
    num_classes: int,
    args,
    ab_cfg: Dict[str, object],
    logger: StepLogger,
):
    mode = str(ab_cfg["ensemble_mode"])
    temp = float(getattr(args, "qbary_weight_temp", 1.5))
    floor = float(getattr(args, "qbary_weight_floor", 0.02))
    floor = float(np.clip(floor, 0.0, 0.25))
    classwise_blend = float(np.clip(getattr(args, "qbary_classwise_blend", 0.80), 0.0, 1.0))

    if mode == "uniform":
        global_weights = np.ones(len(client_ids), dtype=np.float64) / max(len(client_ids), 1)
        classwise_weights = np.tile(global_weights[:, None], (1, num_classes))
        return global_weights, classwise_weights

    if mode == "size_only":
        global_weights = np.asarray(size_weights, dtype=np.float64)
        global_weights = global_weights / (global_weights.sum() + 1e-12)
        classwise_weights = np.tile(global_weights[:, None], (1, num_classes))
        return global_weights, classwise_weights

    global_raw, classwise_raw = _compute_score_matrices(
        client_ids=client_ids,
        client_stats=client_stats,
        transformed_states=transformed_states,
        feature_map=feature_map,
        reduced_proto_map=reduced_proto_map,
        class_state_refs=class_state_refs,
        class_proto_refs=class_proto_refs,
        reduced_var_map=reduced_var_map,
        size_weights=size_weights,
        num_classes=num_classes,
        args=args,
        ab_cfg=ab_cfg,
    )

    for cid, raw in zip(client_ids, global_raw):
        logger.log(f"client={cid} global fusion score={raw:.6f}")

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
        classwise_weights[:, c] = v
        logger.log(f"class={c} classwise weights={np.round(v, 4).tolist()}")

    if mode == "global_only":
        classwise_weights = np.tile(global_weights[:, None], (1, num_classes))
    elif mode == "classwise_only":
        pass
    elif mode == "blended":
        for c in range(num_classes):
            v = classwise_blend * classwise_weights[:, c] + (1.0 - classwise_blend) * global_weights
            classwise_weights[:, c] = v / (v.sum() + 1e-12)
    else:
        raise ValueError(f"Unsupported qbary_ensemble_mode={mode!r}")

    logger.log(f"global weights={np.round(global_weights, 4).tolist()}")
    return global_weights, classwise_weights


# =============================================================================
# Client
# =============================================================================


class QBaryEnsemble_AblationClient(BaseClient):
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
        self.quantum_payload = None
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
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state_dict.items()}
            )

        with self._logger.step("extracting train-only latent prototypes and variances"):
            self.model.eval()
            with torch.no_grad():
                rep, _ = _extract_model_outputs(self.model, self.data)
            self.stats = _compute_train_only_class_stats(rep, self.data, self.num_classes)

        self.best_state = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        payload = _payload_mb(self.best_state) + _payload_mb(self.stats)
        self._logger.log(
            f"payload to server ≈ {payload:.2f} MB | "
            f"best_val={best_val:.4f} | final_loss={last_loss:.6f}"
        )
        return last_loss

    def prepare_quantum_payload(self, pca_meta: Dict):
        ab_cfg = _get_ablation_config(self.args)
        if not bool(ab_cfg["remove_quantum"]):
            _ensure_qiskit()
        if self.stats is None:
            self.train()

        proto = self.stats["latent_proto"].numpy().astype(np.float64)
        var_ = self.stats["latent_var"].numpy().astype(np.float64)
        counts = self.stats["class_counts"].numpy().astype(np.float64)

        density_by_class: Dict[int, np.ndarray] = {}
        reduced_proto_by_class: Dict[int, np.ndarray] = {}
        reduced_var_by_class: Dict[int, np.ndarray] = {}

        for c in range(proto.shape[0]):
            if counts[c] <= 0:
                continue
            z = _apply_server_pca(proto[c], pca_meta)
            z_var = _apply_server_pca_diag_var(var_[c], pca_meta)
            if not bool(ab_cfg["remove_quantum"]):
                rho = _encode_reduced_vector_to_density(z, self.args)
                density_by_class[int(c)] = rho
            reduced_proto_by_class[int(c)] = z
            reduced_var_by_class[int(c)] = z_var

        self.quantum_payload = {
            "density_by_class": density_by_class,
            "reduced_proto_by_class": reduced_proto_by_class,
            "reduced_var_by_class": reduced_var_by_class,
            "class_counts": counts.copy(),
        }
        payload_mb = (
            _payload_mb(density_by_class)
            + _payload_mb(reduced_proto_by_class)
            + _payload_mb(reduced_var_by_class)
        )
        payload_name = "classical reduced-prototype payload" if bool(ab_cfg["remove_quantum"]) else "quantum payload"
        self._logger.log(f"{payload_name} to server ≈ {payload_mb:.2f} MB")
        return self.quantum_payload


# =============================================================================
# Server
# =============================================================================


class QBaryEnsemble_AblationServer(BaseServer):
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
        self._ab_cfg = _get_ablation_config(args)

    def _client_ids(self):
        if getattr(self, "sampled_clients", None):
            return list(self.sampled_clients)
        return (
            list(self.clients.keys())
            if isinstance(self.clients, dict)
            else list(range(len(self.clients)))
        )

    def _get_client(self, cid):
        return (
            self.clients[cid]
            if isinstance(self.clients, dict)
            else self.clients[int(cid)]
        )

    def _atoms_from_client_quantum_payloads(
        self,
        client_ids: Sequence[int],
        client_quantum_payloads: Sequence[Dict],
        size_weights: np.ndarray,
    ) -> List[QuantumAtom]:
        atoms: List[QuantumAtom] = []
        classical_only = bool(self._ab_cfg.get("remove_quantum", False))
        for cid, payload, sw in zip(client_ids, client_quantum_payloads, size_weights):
            counts = np.asarray(payload["class_counts"], dtype=np.float64)
            source = payload["reduced_proto_by_class"] if classical_only else payload["density_by_class"]
            for cls in source.keys():
                rho = (
                    np.eye(1, dtype=np.complex128)
                    if classical_only
                    else np.asarray(payload["density_by_class"][int(cls)], dtype=np.complex128)
                )
                atoms.append(
                    QuantumAtom(
                        client_id=int(cid),
                        cls=int(cls),
                        count=float(sw) * float(counts[int(cls)]),
                        rho=rho,
                        z=np.asarray(payload["reduced_proto_by_class"][int(cls)], dtype=np.float64),
                    )
                )
        return atoms

    def aggregate(self):
        if not bool(self._ab_cfg.get("remove_quantum", False)):
            _ensure_qiskit()
        client_ids = self._client_ids()
        sizes = []
        client_stats = []
        total_received_mb = 0.0

        self._logger.log("=" * 60)
        self._logger.log("QBaryEnsemble-Hybrid: collecting one-shot client payloads")
        self._logger.log(f"ablation config={self._ab_cfg}")

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
            f"server PCA input_dim={pca_meta['input_dim']} -> "
            f"effective_dim={pca_meta['effective_dim']} -> "
            f"target_dim={pca_meta['target_dim']} | "
            f"explained_variance={evr_sum:.4f}"
        )

        client_quantum_payloads = []
        reduced_proto_map: Dict[Tuple[int, int], np.ndarray] = {}
        reduced_var_map: Dict[Tuple[int, int], np.ndarray] = {}
        quantum_received_mb = 0.0

        with self._logger.step(
            "broadcasting PCA basis and collecting client representations"
        ):
            for cid in client_ids:
                client = self._get_client(cid)
                payload = client.prepare_quantum_payload(pca_meta)
                client_quantum_payloads.append(payload)
                quantum_received_mb += (
                    _payload_mb(payload["density_by_class"])
                    + _payload_mb(payload["reduced_proto_by_class"])
                    + _payload_mb(payload["reduced_var_by_class"])
                )
                for cls, z in payload["reduced_proto_by_class"].items():
                    reduced_proto_map[(int(cid), int(cls))] = np.asarray(z, dtype=np.float64)
                for cls, z_var in payload["reduced_var_by_class"].items():
                    reduced_var_map[(int(cid), int(cls))] = np.asarray(z_var, dtype=np.float64)
        payload_kind = "classical reduced-prototype" if bool(self._ab_cfg.get("remove_quantum", False)) else "quantum"
        self._logger.log(
            f"server received {payload_kind} payload ≈ {quantum_received_mb:.2f} MB"
        )

        atoms = self._atoms_from_client_quantum_payloads(
            client_ids=client_ids,
            client_quantum_payloads=client_quantum_payloads,
            size_weights=size_weights,
        )

        theta, class_state_refs, class_proto_refs, transformed_states, feature_map = _train_server_quantum_lens(
            atoms=atoms,
            client_ids=client_ids,
            client_stats=client_stats,
            reduced_proto_map=reduced_proto_map,
            reduced_var_map=reduced_var_map,
            size_weights=size_weights,
            args=self.args,
            num_classes=self.num_classes,
            n_qubits=int(getattr(self.args, "qbary_n_qubits", 3)),
            depth=int(getattr(self.args, "qbary_pqc_depth", 2)),
            train_steps=int(getattr(self.args, "qbary_pqc_steps", 80)),
            spsa_a=float(getattr(self.args, "qbary_spsa_a", 0.08)),
            spsa_c=float(getattr(self.args, "qbary_spsa_c", 0.15)),
            spsa_alpha=float(getattr(self.args, "qbary_spsa_alpha", 0.602)),
            spsa_gamma=float(getattr(self.args, "qbary_spsa_gamma", 0.101)),
            between_coef=float(getattr(self.args, "qbary_between_coef", 0.50)),
            weight_entropy_coef=float(getattr(self.args, "qbary_weight_entropy_coef", 0.25)),
            bary_iters=int(getattr(self.args, "qbary_bary_iters", 40)),
            logger=self._logger,
            ab_cfg=self._ab_cfg,
            eval_every=int(getattr(self.args, "qbary_spsa_eval_every", 5)),
        )

        if self._ab_cfg["geometry_mode"] == "bures" and class_state_refs:
            relation = _quantum_relation_matrix(class_state_refs, self.num_classes)
            relation_metric = "bures_distance"
        else:
            relation = _pairwise_proto_relation_matrix(
                class_proto_refs,
                metric=str(self._ab_cfg["geometry_mode"]),
                num_classes=self.num_classes,
            )
            relation_metric = str(self._ab_cfg["geometry_mode"])

        self._logger.log(
            f"atlas support classes={len(class_proto_refs)}/{self.num_classes}"
        )

        global_weights, classwise_weights = _compute_final_weights(
            client_ids=client_ids,
            client_stats=client_stats,
            transformed_states=transformed_states,
            feature_map=feature_map,
            reduced_proto_map=reduced_proto_map,
            class_state_refs=class_state_refs,
            class_proto_refs=class_proto_refs,
            reduced_var_map=reduced_var_map,
            size_weights=size_weights,
            num_classes=self.num_classes,
            args=self.args,
            ab_cfg=self._ab_cfg,
            logger=self._logger,
        )

        self.quantum_atlas = {
            "theta": theta,
            "class_bary": class_state_refs,
            "class_proto_ref": class_proto_refs,
            "relation": relation,
            "relation_metric": relation_metric,
            "pca_mean": pca_meta["mean"],
            "pca_components": pca_meta["components"],
            "pca_explained_variance_ratio": pca_meta["explained_variance_ratio"],
            "pca_effective_dim": pca_meta["effective_dim"],
            "pca_target_dim": pca_meta["target_dim"],
            "state_encoding": str(
                getattr(self.args, "qbary_state_encoding", "amplitude")
            ).lower(),
            "barycenter_mode": str(self._ab_cfg["barycenter_mode"]),
            "geometry_mode": str(self._ab_cfg["geometry_mode"]),
            "objective_mode": str(self._ab_cfg["objective_mode"]),
            "ensemble_mode": str(self._ab_cfg["ensemble_mode"]),
            "remove_quantum": bool(self._ab_cfg.get("remove_quantum", False)),
        }
        self.global_weights = np.asarray(global_weights, dtype=np.float64)
        self.classwise_weights = np.asarray(classwise_weights, dtype=np.float64)
        self.final_client_ids = list(client_ids)
        self._ready = True

        self._logger.log("QBaryEnsemble-Hybrid: done")
        self._logger.log("=" * 60)

    @torch.no_grad()
    def _ensemble_predict_proba(self, data, mask: torch.Tensor) -> torch.Tensor:
        if self.final_client_ids is None or self.classwise_weights is None:
            raise RuntimeError("Ensemble weights unavailable. Call aggregate() first.")

        probs_acc = None
        classwise = torch.tensor(
            self.classwise_weights, dtype=torch.float32, device=self.device
        )
        with self._logger.step("running quantum-weighted client ensemble"):
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
        # Base method
        "qbary_verbose": True,
        "qbary_n_qubits": 3,
        "qbary_pqc_depth": 2,
        "qbary_pqc_steps": 80,
        "qbary_spsa_a": 0.08,
        "qbary_spsa_c": 0.15,
        "qbary_spsa_alpha": 0.602,
        "qbary_spsa_gamma": 0.101,
        "qbary_between_coef": 0.50,
        "qbary_weight_entropy_coef": 0.25,
        "qbary_bary_iters": 40,
        "qbary_weight_gamma": 0.50,
        "qbary_min_class_support": 0.05,
        "qbary_weight_temp": 1.5,
        "qbary_weight_floor": 0.02,
        "qbary_feature_margin_coef": 0.50,
        "qbary_geom_margin_coef": 1.00,
        "qbary_variance_margin_coef": 0.35,
        "qbary_variance_conf_coef": 0.15,
        "qbary_logit_clip": 20.0,
        "qbary_classwise_blend": 0.80,
        "qbary_state_encoding": "amplitude",   # {'amplitude', 'angle'}
        "qbary_pca_dim": None,
        "qbary_angle_layers": 2,
        "qbary_spsa_eval_every": 5,
        # Ablation controls
        "qbary_use_quantum_lens": True,
        "qbary_barycenter_mode": "bures",      # {'bures','euclidean_density','density_medoid','reduced_proto_mean','reduced_proto_medoid'}
        "qbary_geometry_mode": "bures",        # {'bures','proto_euclidean','proto_cosine'}
        "qbary_objective_mode": "full",        # {'within_only','within_between','within_between_entropy','full'}
        "qbary_variance_mode": "full",         # {'full','means_only','conf_only','margin_only'}
        "qbary_ensemble_mode": "blended",      # {'uniform','size_only','global_only','classwise_only','blended'}
        "qbary_remove_quantum": False,
        "qbary_use_geom_margin": True,
        "qbary_use_feature_margin": True,
        "qbary_use_variance_margin": True,
        "qbary_use_variance_conf": True,
        "qbary_use_support_term": True,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args



import math
import os
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
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import QiskitRuntimeService
except Exception as e:  # pragma: no cover
    QuantumCircuit = None
    transpile = None
    StatePreparation = None
    DensityMatrix = None
    AerSimulator = None
    NoiseModel = None
    QiskitRuntimeService = None
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


# =============================================================================
# Utilities
# =============================================================================

def _ensure_qiskit():
    if (
        QuantumCircuit is None
        or transpile is None
        or StatePreparation is None
        or DensityMatrix is None
        or AerSimulator is None
        or NoiseModel is None
        or QiskitRuntimeService is None
    ):
        raise ImportError(
            "Qiskit + Aer + qiskit-ibm-runtime are required for QBaryEnsemble_DensityGeometryOnly_Noisy. "
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


# =============================================================================
# Server PCA and noisy amplitude encoding
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


class IBMNoisyDensityRuntime:
    """
    Local Aer density-matrix simulator whose noise model is derived from a real
    IBM backend calibration.

    This final no-PQC method injects noise only during state preparation
    (amplitude encoding) and then uses the resulting noisy density matrices for
    Bures barycenter aggregation and geometry-based scoring.
    """

    def __init__(
        self,
        n_qubits: int,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        instance: Optional[str] = None,
        backend_name: Optional[str] = None,
        use_least_busy: bool = False,
        seed: int = 1234,
        verbose: bool = True,
    ):
        _ensure_qiskit()
        self.n_qubits = int(n_qubits)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        service_kwargs = {}
        if token:
            service_kwargs["token"] = token
        if channel:
            service_kwargs["channel"] = channel
        if instance:
            service_kwargs["instance"] = instance

        self.service = QiskitRuntimeService(**service_kwargs) if service_kwargs else QiskitRuntimeService()

        if backend_name:
            self.backend_device = self.service.backend(name=backend_name)
        elif use_least_busy:
            self.backend_device = self.service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=max(self.n_qubits, 1),
            )
        else:
            backends = self.service.backends(
                min_num_qubits=max(self.n_qubits, 1),
                operational=True,
                simulator=False,
            )
            if not backends:
                raise RuntimeError("No operational IBM backends available for noise extraction.")
            self.backend_device = backends[0]

        name_attr = getattr(self.backend_device, "name", None)
        self.backend_name = name_attr() if callable(name_attr) else str(name_attr)

        self.noise_model = NoiseModel.from_backend(self.backend_device)
        self.sim = AerSimulator(
            method="density_matrix",
            noise_model=self.noise_model,
            seed_simulator=self.seed,
        )

        if self.verbose:
            print(f"[IBM-NOISE] Using calibration-derived noise from backend: {self.backend_name}")
            print("[IBM-NOISE] Aer method=density_matrix | noise applied to state preparation only")

    def encode_density(self, z: np.ndarray, n_qubits: int) -> np.ndarray:
        qc = _build_client_amplitude_encoding_circuit(z, n_qubits)
        qc = qc.copy()
        qc.save_density_matrix(label="rho")

        tqc = transpile(
            qc,
            backend=self.sim,
            optimization_level=0,
            seed_transpiler=self.seed,
        )
        result = self.sim.run(tqc).result()
        data = result.data(0)
        rho = data["rho"]
        if hasattr(rho, "data"):
            rho = rho.data
        return _project_psd_trace_one(np.asarray(rho, dtype=np.complex128))


_NOISY_RUNTIME_CACHE: Dict[Tuple, IBMNoisyDensityRuntime] = {}


def _get_noisy_density_runtime(args, logger: Optional[StepLogger] = None) -> IBMNoisyDensityRuntime:
    token = getattr(args, "qbary_ibm_token", None) or os.environ.get("QISKIT_IBM_TOKEN")
    key = (
        int(getattr(args, "qbary_n_qubits", 3)),
        token,
        getattr(args, "qbary_ibm_channel", None),
        getattr(args, "qbary_ibm_instance", None),
        getattr(args, "qbary_noise_backend", None),
        bool(getattr(args, "qbary_use_least_busy", False)),
        int(getattr(args, "qbary_seed", 1234)),
    )
    if key not in _NOISY_RUNTIME_CACHE:
        if logger is not None:
            logger.log("initializing IBM backend-derived noisy density-matrix simulator")
        _NOISY_RUNTIME_CACHE[key] = IBMNoisyDensityRuntime(
            n_qubits=int(getattr(args, "qbary_n_qubits", 3)),
            token=token,
            channel=getattr(args, "qbary_ibm_channel", None),
            instance=getattr(args, "qbary_ibm_instance", None),
            backend_name=getattr(args, "qbary_noise_backend", None),
            use_least_busy=bool(getattr(args, "qbary_use_least_busy", False)),
            seed=int(getattr(args, "qbary_seed", 1234)),
            verbose=bool(getattr(args, "qbary_verbose", True)),
        )
    return _NOISY_RUNTIME_CACHE[key]


# =============================================================================
# Geometry-only aggregation and scoring
# =============================================================================

def _group_atoms_by_class(atoms: Sequence[QuantumAtom], num_classes: int) -> List[List[QuantumAtom]]:
    grouped: List[List[QuantumAtom]] = [[] for _ in range(num_classes)]
    for atom in atoms:
        grouped[int(atom.cls)].append(atom)
    return grouped


def _compute_class_barycenters(
    atoms: Sequence[QuantumAtom],
    num_classes: int,
    bary_iters: int,
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    grouped = _group_atoms_by_class(atoms, num_classes)
    class_bary: Dict[int, np.ndarray] = {}
    state_map: Dict[Tuple[int, int], np.ndarray] = {}

    for c, bucket in enumerate(grouped):
        if not bucket:
            continue
        rhos = []
        ws = []
        for atom in bucket:
            key = (atom.client_id, atom.cls)
            rho = _project_psd_trace_one(atom.rho)
            state_map[key] = rho
            rhos.append(rho)
            ws.append(float(atom.count))
        class_bary[c] = _bures_barycenter(rhos, ws, max_iter=bary_iters)

    return class_bary, state_map


def _quantum_relation_matrix(class_bary: Dict[int, np.ndarray], num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        if i not in class_bary:
            continue
        for j in range(num_classes):
            if j not in class_bary:
                continue
            mat[i, j] = _bures_distance(class_bary[i], class_bary[j])
    return mat


def _compute_geometry_only_weights(
    client_ids: Sequence[int],
    client_stats: Sequence[Dict],
    state_map: Dict[Tuple[int, int], np.ndarray],
    class_bary: Dict[int, np.ndarray],
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

    global_raw = np.zeros(len(client_ids), dtype=np.float64)
    classwise_raw = np.zeros((len(client_ids), num_classes), dtype=np.float64)

    for k, (cid, stats, sw) in enumerate(zip(client_ids, client_stats, size_weights)):
        counts = stats["class_counts"].numpy().astype(np.float64)
        g_num, g_den = 0.0, 0.0

        for c in range(num_classes):
            key = (int(cid), int(c))
            if counts[c] <= 0 or c not in class_bary or key not in state_map:
                continue

            rho = state_map[key]
            pos_dist = _bures_distance(rho, class_bary[c])
            neg_dist = min(
                [_bures_distance(rho, class_bary[cp]) for cp in class_bary if cp != c] + [2.0]
            )
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

class New_NoiseClient(BaseClient):
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
        self._quantum_runtime = None
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
            f"payload to server ≈ {payload:.2f} MB | "
            f"best_val={best_val:.4f} | final_loss={last_loss:.6f}"
        )
        return last_loss

    def prepare_quantum_payload(self, pca_meta: Dict):
        _ensure_qiskit()
        if self.stats is None:
            self.train()

        runtime = self._quantum_runtime or _get_noisy_density_runtime(self.args, self._logger)
        self._quantum_runtime = runtime

        proto = self.stats["latent_proto"].numpy().astype(np.float64)
        counts = self.stats["class_counts"].numpy().astype(np.float64)

        density_by_class: Dict[int, np.ndarray] = {}
        n_qubits = int(getattr(self.args, "qbary_n_qubits", 3))

        with self._logger.step("encoding reduced prototypes on IBM-derived noisy simulator"):
            for c in range(proto.shape[0]):
                if counts[c] <= 0:
                    continue
                z = _apply_server_pca(proto[c], pca_meta)
                rho = runtime.encode_density(z, n_qubits=n_qubits)
                density_by_class[int(c)] = rho

        self.quantum_payload = {
            "density_by_class": density_by_class,
            "class_counts": counts.copy(),
            "noise_backend_name": runtime.backend_name,
        }
        quantum_mb = _payload_mb(density_by_class)
        self._logger.log(f"quantum payload to server ≈ {quantum_mb:.2f} MB")
        return self.quantum_payload


# =============================================================================
# Server
# =============================================================================

class New_NoiseServer(BaseServer):
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

    def _atoms_from_client_quantum_payloads(
        self,
        client_ids: Sequence[int],
        client_quantum_payloads: Sequence[Dict],
        size_weights: np.ndarray,
    ) -> List[QuantumAtom]:
        atoms: List[QuantumAtom] = []
        for cid, payload, sw in zip(client_ids, client_quantum_payloads, size_weights):
            counts = np.asarray(payload["class_counts"], dtype=np.float64)
            for cls, rho in payload["density_by_class"].items():
                atoms.append(
                    QuantumAtom(
                        client_id=int(cid),
                        cls=int(cls),
                        count=float(sw) * float(counts[int(cls)]),
                        rho=np.asarray(rho, dtype=np.complex128),
                    )
                )
        return atoms

    def aggregate(self):
        _ensure_qiskit()
        client_ids = self._client_ids()
        sizes = []
        client_stats = []
        total_received_mb = 0.0

        self._logger.log("=" * 60)
        self._logger.log("QBaryEnsemble-DensityGeometryOnly-Noisy: collecting one-shot client payloads")

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

        runtime = _get_noisy_density_runtime(self.args, self._logger)

        client_quantum_payloads = []
        quantum_received_mb = 0.0
        with self._logger.step("broadcasting PCA basis and collecting noisy density states"):
            for cid in client_ids:
                client = self._get_client(cid)
                client._quantum_runtime = runtime
                payload = client.prepare_quantum_payload(pca_meta)
                client_quantum_payloads.append(payload)
                quantum_received_mb += _payload_mb(payload["density_by_class"])
        self._logger.log(
            f"server received quantum payload ≈ {quantum_received_mb:.2f} MB | "
            f"noise backend={runtime.backend_name}"
        )

        atoms = self._atoms_from_client_quantum_payloads(
            client_ids=client_ids,
            client_quantum_payloads=client_quantum_payloads,
            size_weights=size_weights,
        )

        class_bary, state_map = _compute_class_barycenters(
            atoms=atoms,
            num_classes=self.num_classes,
            bary_iters=int(getattr(self.args, "qbary_bary_iters", 40)),
        )

        relation = _quantum_relation_matrix(class_bary, self.num_classes)
        self._logger.log(f"atlas support classes={len(class_bary)}/{self.num_classes}")

        global_weights, classwise_weights = _compute_geometry_only_weights(
            client_ids=client_ids,
            client_stats=client_stats,
            state_map=state_map,
            class_bary=class_bary,
            size_weights=size_weights,
            num_classes=self.num_classes,
            args=self.args,
            logger=self._logger,
        )

        self.quantum_atlas = {
            "class_bary": class_bary,
            "relation": relation,
            "relation_metric": "bures_distance",
            "pca_mean": pca_meta["mean"],
            "pca_components": pca_meta["components"],
            "pca_explained_variance_ratio": pca_meta["explained_variance_ratio"],
            "pca_effective_dim": pca_meta["effective_dim"],
            "pca_target_dim": pca_meta["target_dim"],
            "state_encoding": "amplitude",
            "noise_backend_name": runtime.backend_name,
        }
        self.global_weights = np.asarray(global_weights, dtype=np.float64)
        self.classwise_weights = np.asarray(classwise_weights, dtype=np.float64)
        self.final_client_ids = list(client_ids)
        self._ready = True

        self._logger.log("QBaryEnsemble-DensityGeometryOnly-Noisy: done")
        self._logger.log("=" * 60)

    @torch.no_grad()
    def _ensemble_predict_proba(self, data, mask: torch.Tensor) -> torch.Tensor:
        if self.final_client_ids is None or self.classwise_weights is None:
            raise RuntimeError("Ensemble weights unavailable. Call aggregate() first.")

        probs_acc = None
        classwise = torch.tensor(self.classwise_weights, dtype=torch.float32, device=self.device)
        with self._logger.step("running geometry-weighted client ensemble"):
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
        "qbary_ibm_token": "g0vD2Nl_AQJKE6ZBZVeTw9z-uN6f7eu6vqsG6hgVFypg",
        "qbary_ibm_channel": "ibm_cloud",
        "qbary_ibm_instance": None,
        "qbary_noise_backend": None,
        "qbary_use_least_busy": True,
        "qbary_seed": 1234,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args

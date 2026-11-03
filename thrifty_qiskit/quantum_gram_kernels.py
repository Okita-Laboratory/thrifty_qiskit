"""
quantum_gram_kernels.py

Kernel plug-ins for gram_resampling.GramCache that reduce the number
or cost of quantum circuit evaluations.  All expose the same interface:

    kernel(X)        -> Gram (n, n)          (for GramCache)
    kernel.pairs(A, B) -> cross Gram (na, nb) (used by NystromKernel)

Contents
--------
1. DifferenceGridKernel   k(x,y)=f(x-y): evaluate f on G grid points, interpolate.
                          n(n-1)/2 circuits -> G circuits.
2. JointDifferenceGridKernel  k((x,y),(x',y')) = f(x-x', y-y'): G^2 grid.
                          Removes the n^4 problem of permuted joint encodings.
3. NystromKernel          K ~ K_nm K_mm^{-1} K_mn with m landmarks.
                          n^2 -> n*m + m^2 circuits.  Works for any kernel.
4. BatchedFidelityKernel  qiskit compute-uncompute fidelity kernel:
                          single PUB, transpile once, symmetry, adaptive shots.
5. is_translation_invariant   numerical check used before choosing 1/2.

Only 4 needs qiskit; 1-3 are pure numpy and accept any pairwise kernel.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

PairFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (A(na,d), B(nb,d)) -> (na, nb)


def _2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return X[:, None] if X.ndim == 1 else X


# ************************************************************
# 0. Translation-invariance check
# ************************************************************
def is_translation_invariant(pair_fn: PairFn, d: int = 1, n_test: int = 8,
                             tol: float = 1e-6, rng: Optional[np.random.Generator] = None) -> bool:
    """Return True if k(x+c, y+c) == k(x, y) for random x, y, c (per feature)."""
    rng = rng or np.random.default_rng(0)
    X = rng.uniform(0, 1, (n_test, d)); Y = rng.uniform(0, 1, (n_test, d))
    c = rng.uniform(-0.5, 0.5, (1, d))
    return bool(np.max(np.abs(pair_fn(X, Y) - pair_fn(X + c, Y + c))) < tol)


# ************************************************************
# 1. Difference-grid kernel (1-D)
# ************************************************************
class DifferenceGridKernel:
    """
    k(x, y) = f(x - y).  `pair_fn` is only called on the grid of differences.

    Parameters
    ----------
    pair_fn   : exact pairwise kernel (e.g. a quantum fidelity kernel).
    n_grid    : number of grid points G over [-range, range].
    check     : run is_translation_invariant() at construction (raises if False).
    """

    def __init__(self, pair_fn: PairFn, n_grid: int = 201, check: bool = True):
        if check and not is_translation_invariant(pair_fn, d=1):
            raise ValueError("pair_fn is not translation-invariant; use NystromKernel instead")
        self.pair_fn, self.n_grid = pair_fn, n_grid
        self.n_circuit_evals = 0
        self._grid = None
        self._f = None

    def _build(self, span: float) -> None:
        self._grid = np.linspace(-span, span, self.n_grid)
        # f(delta) = k(delta, 0): one row of pair evaluations
        self._f = self.pair_fn(self._grid[:, None], np.zeros((1, 1)))[:, 0]
        self.n_circuit_evals += self.n_grid

    def pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A, B = _2d(A)[:, 0], _2d(B)[:, 0]
        D = A[:, None] - B[None, :]
        span = float(np.max(np.abs(D))) * 1.0001 + 1e-12
        if self._grid is None or span > self._grid[-1]:
            self._build(span)
        return np.interp(D, self._grid, self._f)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.pairs(X, X)


# ************************************************************
# 2. Joint difference-grid kernel (2-D)
# ************************************************************
class JointDifferenceGridKernel:
    """
    k((x, y), (x', y')) = f(x - x', y - y').  Evaluates f on a G x G grid
    and interpolates bilinearly.  A permutation of X against fixed Y is then
    just a different set of (dx, dy) lookups: no new circuits.
    """

    def __init__(self, pair_fn: PairFn, n_grid: int = 61, check: bool = True):
        if check and not is_translation_invariant(pair_fn, d=2):
            raise ValueError("joint pair_fn is not translation-invariant")
        self.pair_fn, self.n_grid = pair_fn, n_grid
        self.n_circuit_evals = 0
        self._gx = self._gy = self._F = None

    def _build(self, sx: float, sy: float) -> None:
        self._gx = np.linspace(-sx, sx, self.n_grid)
        self._gy = np.linspace(-sy, sy, self.n_grid)
        GX, GY = np.meshgrid(self._gx, self._gy, indexing="ij")
        pts = np.column_stack([GX.ravel(), GY.ravel()])
        self._F = self.pair_fn(pts, np.zeros((1, 2)))[:, 0].reshape(self.n_grid, self.n_grid)
        self.n_circuit_evals += self.n_grid ** 2

    def _interp(self, DX: np.ndarray, DY: np.ndarray) -> np.ndarray:
        gx, gy, F = self._gx, self._gy, self._F
        ix = np.clip(np.searchsorted(gx, DX) - 1, 0, len(gx) - 2)
        iy = np.clip(np.searchsorted(gy, DY) - 1, 0, len(gy) - 2)
        tx = (DX - gx[ix]) / (gx[ix + 1] - gx[ix])
        ty = (DY - gy[iy]) / (gy[iy + 1] - gy[iy])
        return ((1 - tx) * (1 - ty) * F[ix, iy] + tx * (1 - ty) * F[ix + 1, iy]
                + (1 - tx) * ty * F[ix, iy + 1] + tx * ty * F[ix + 1, iy + 1])

    def pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A, B = _2d(A), _2d(B)
        DX = A[:, 0][:, None] - B[:, 0][None, :]
        DY = A[:, 1][:, None] - B[:, 1][None, :]
        sx = float(np.max(np.abs(DX))) * 1.0001 + 1e-12
        sy = float(np.max(np.abs(DY))) * 1.0001 + 1e-12
        if self._gx is None or sx > self._gx[-1] or sy > self._gy[-1]:
            self._build(max(sx, self._gx[-1] if self._gx is not None else 0),
                        max(sy, self._gy[-1] if self._gy is not None else 0))
        return self._interp(DX, DY)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.pairs(X, X)

    def gram_for_perm(self, X: np.ndarray, Y: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Gram of joint points (x_pi(i), y_i) without new circuits."""
        X, Y = _2d(X)[:, 0], _2d(Y)[:, 0]
        return self(np.column_stack([X[pi], Y]))


# ************************************************************
# 3. Nystrom kernel
# ************************************************************
class NystromKernel:
    """
    K ~ K_nm K_mm^{-1} K_mn using m landmark rows.  Circuit count:
    n*m (cross) + m*(m-1)/2 (landmark block) instead of n*(n-1)/2.

    landmarks : "random" | "kmeans" | explicit index array.
    ridge     : small regulariser added to K_mm before inversion.
    """

    def __init__(self, pair_fn: PairFn, m: int = 30, landmarks="random",
                 ridge: float = 1e-8, rng: Optional[np.random.Generator] = None):
        self.pair_fn, self.m, self.landmarks, self.ridge = pair_fn, m, landmarks, ridge
        self.rng = rng or np.random.default_rng(0)
        self.n_circuit_evals = 0

    def _pick(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if isinstance(self.landmarks, str) and self.landmarks == "random":
            return self.rng.choice(n, min(self.m, n), replace=False)
        if isinstance(self.landmarks, str) and self.landmarks == "kmeans":
            # simple Lloyd iterations on 1-d/low-d data; picks nearest real points
            C = X[self.rng.choice(n, min(self.m, n), replace=False)]
            for _ in range(20):
                lab = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1)
                C = np.array([X[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(len(C))])
            return np.unique(np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=0))
        return np.asarray(self.landmarks)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = _2d(X)
        idx = self._pick(X)
        L = X[idx]
        K_nm = self.pair_fn(X, L)                       # n*m circuits
        K_mm = K_nm[idx]                                # subset, no new circuits
        self.n_circuit_evals += X.shape[0] * len(idx)
        K_mm_inv = np.linalg.pinv(K_mm + self.ridge * np.eye(len(idx)))
        K = K_nm @ K_mm_inv @ K_nm.T
        # exact values on landmark rows/cols already; symmetrise numerically
        return 0.5 * (K + K.T)

    def pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.pair_fn(A, B)


# ************************************************************
# 4. Batched qiskit fidelity kernel
# ************************************************************
class BatchedFidelityKernel:
    """
    Compute-uncompute fidelity kernel  k(x, y) = |<0| U(y)^dag U(x) |0>|^2
    with one parametrised circuit, transpiled once, all pairs in ONE PUB.

    Parameters
    ----------
    feature_map : qiskit QuantumCircuit with `d` free parameters (e.g. z_feature_map(d, reps)).
    sampler     : a qiskit primitive Sampler (StatevectorSampler, AerSampler, runtime Sampler).
    backend     : if given, the circuit is transpiled to ISA once via generate_preset_pass_manager.
    shots       : shots per pair for stage 1.
    adaptive    : if not None, a tuple (select_fn, extra_shots).  After stage 1,
                  select_fn(K_stage1) -> boolean mask (n, n) of pairs to re-run
                  with extra_shots; results are pooled shot-weighted.
    resilience_level etc. are the caller's business: set them on the sampler's
    options (runtime) before passing it in; nothing here depends on them.
    """

    def __init__(self, feature_map, sampler, backend=None, shots: int = 2048,
                 adaptive: Optional[tuple] = None, optimization_level: int = 1):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        self.d = feature_map.num_parameters
        self.sampler, self.shots, self.adaptive = sampler, shots, adaptive
        x = ParameterVector("x", self.d)
        y = ParameterVector("y", self.d)
        qc = QuantumCircuit(feature_map.num_qubits)
        qc.compose(feature_map.assign_parameters(list(x)), inplace=True)
        qc.compose(feature_map.assign_parameters(list(y)).inverse(), inplace=True)
        qc.measure_all()
        self._params = list(x) + list(y)
        if backend is not None:
            from qiskit.transpiler import generate_preset_pass_manager
            pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
            qc = pm.run(qc)  # transpiled ONCE; only parameter values change afterwards
        self.circuit = qc
        self.n_jobs = 0
        self.n_circuit_evals = 0
        self.n_shots_total = 0

    # ---- low level: run a set of (x, y) parameter rows in one job ----
    def _run_rows(self, rows: np.ndarray, shots: int) -> np.ndarray:
        """rows: (N, 2d) -> fidelity estimates (N,)"""
        if len(rows) == 0:
            return np.zeros(0)
        job = self.sampler.run([(self.circuit, rows)], shots=shots)
        bits = job.result()[0].data.meas.array  # (N, shots, nbytes)
        self.n_jobs += 1
        self.n_circuit_evals += len(rows)
        self.n_shots_total += len(rows) * shots
        return (bits == 0).all(axis=-1).mean(axis=-1)

    def pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A, B = _2d(A), _2d(B)
        ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij")
        rows = np.column_stack([A[ia.ravel()], B[ib.ravel()]])
        return self._run_rows(rows, self.shots).reshape(len(A), len(B))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = _2d(X)
        n = X.shape[0]
        iu, ju = np.triu_indices(n, k=1)                 # symmetry: upper triangle only
        rows = np.column_stack([X[iu], X[ju]])
        f = self._run_rows(rows, self.shots)             # stage 1: one job
        K = np.eye(n)
        K[iu, ju] = f; K[ju, iu] = f
        if self.adaptive is not None:                    # stage 2: only selected pairs
            select_fn, extra = self.adaptive
            mask = np.triu(np.asarray(select_fn(K), bool), k=1)
            si, sj = np.where(mask)
            if len(si):
                f2 = self._run_rows(np.column_stack([X[si], X[sj]]), extra)
                pooled = (K[si, sj] * self.shots + f2 * extra) / (self.shots + extra)
                K[si, sj] = pooled; K[sj, si] = pooled
        return K


# ----------------------------------------------------------------------
# Convenience: selector for adaptive shots based on stage-1 uncertainty
# ----------------------------------------------------------------------
def select_uncertain(threshold: float = 0.5, z: float = 2.0, shots: int = 2048):
    """
    Return select_fn: re-run pairs whose stage-1 estimate is within z standard
    errors of `threshold` (i.e. whose value could plausibly fall on either side).
    """
    def _sel(K: np.ndarray) -> np.ndarray:
        se = np.sqrt(np.clip(K * (1 - K), 1e-12, None) / shots)
        return np.abs(K - threshold) < z * se
    return _sel

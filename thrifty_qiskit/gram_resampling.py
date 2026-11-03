"""
gram_resampling.py

Index-based resampling for kernel statistics.

Design rule
-----------
Only `GramCache.gram()` ever touches a kernel function (quantum circuit,
simulator, RBF, ...).  Every statistic and every resampling scheme below
takes Gram matrices only and manipulates them by index operations, so the
number of kernel evaluations is fixed at (#variables) x n(n-1)/2 regardless
of how many permutations / bootstraps / CV folds are run on top.

Supported index operations
--------------------------
- subset / split          : K[ix(idx, idx)]                (label permutation, CV)
- single-variable permute : K[ix(pi, pi)]                   (X permuted, Y fixed)
- bootstrap               : K[ix(idx, idx)] with repeats
- product kernel          : K_X * K_Y (Hadamard)            (separable joint encoding)

Not supported: non-separable joint encodings (e.g. entangled ZZ feature
map on (x_i, y_i)).  See `JointGram` for the explicit n^2-point fallback,
which is exact but only feasible for small n.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Dict, Hashable, Optional, Tuple

import numpy as np

KernelFn = Callable[[np.ndarray], np.ndarray]  # data (n, d) -> Gram (n, n)


# ----------------------------------------------------------------------
# Kernel functions (plug-ins).  Replace with a qiskit FidelityQuantumKernel
# wrapper for real hardware; the rest of the module does not change.
# ----------------------------------------------------------------------
def rbf_kernel(X: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    X = np.atleast_2d(X)
    if X.shape[0] == 1 and X.ndim == 2 and X.shape[1] > 1:
        X = X.T
    sq = np.sum(X**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    if gamma is None:
        med = np.median(d2[np.triu_indices_from(d2, k=1)])
        gamma = 1.0 / med if med > 0 else 1.0
    return np.exp(-gamma * d2)


def z_feature_kernel(X: np.ndarray, reps: int = 2) -> np.ndarray:
    """
    Analytic fidelity kernel of the 1-qubit-per-feature Z feature map
    (H, then P(2x)) repeated `reps` times, no entanglement:
        k(x, y) = prod_d cos^2( reps * (x_d - y_d) )
    Separable across features, so the product-kernel identity holds exactly.
    """
    X = np.atleast_2d(X)
    if X.shape[1] > 1 and X.shape[0] == 1:
        X = X.T
    K = np.ones((X.shape[0], X.shape[0]))
    for d in range(X.shape[1]):
        diff = X[:, d][:, None] - X[:, d][None, :]
        K *= np.cos(reps * diff) ** 2
    return K


def scale_to_angle(x: np.ndarray, max_angle: float = np.pi / 2) -> np.ndarray:
    """
    Map a 1-d variable to [0, max_angle].  Periodic feature-map kernels
    (cos^2 form) lose power if raw data spans more than one period, so
    scale BEFORE building the Gram matrix.  Scaling is a fixed function
    of the full sample, so it does not interfere with index resampling.
    """
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo) * max_angle


# ----------------------------------------------------------------------
# Evaluation cache
# ----------------------------------------------------------------------
class EvalCache:
    """
    Memoise an expensive evaluation and count how often it actually runs.

    This is the one place in a pipeline that is allowed to touch raw data (or a
    quantum device).  Everything above it works with the cached objects and with
    indices, so adding resampling on top never increases the number of
    evaluations.  GramCache below is the kernel-matrix instance of this pattern;
    causal-qiskit uses the same class for shadow-estimated reduced states and for
    simulator runs.
    """

    def __init__(self, fn):
        self.fn = fn
        self._cache: Dict[Hashable, object] = {}
        self.n_calls = 0

    def get(self, key: Hashable, *args, **kwargs):
        if key not in self._cache:
            self._cache[key] = self.fn(*args, **kwargs)
            self.n_calls += 1
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()
        self.n_calls = 0

    def __len__(self) -> int:
        return len(self._cache)


class GramCache(EvalCache):
    """Memoise Gram matrices per variable.  Counts kernel calls for auditing."""

    def __init__(self, kernel_fn: KernelFn):
        super().__init__(kernel_fn)
        self.kernel_fn = kernel_fn
        self.n_kernel_calls = 0

    @staticmethod
    def _key_from_array(data: np.ndarray) -> str:
        return hashlib.sha1(np.ascontiguousarray(data).tobytes()).hexdigest()

    def gram(self, data: np.ndarray, key: Optional[Hashable] = None) -> np.ndarray:
        """Return the Gram matrix of `data` (n, d); compute once per key."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if key is None:
            key = self._key_from_array(data)
        out = self.get(key, data)
        self.n_kernel_calls = self.n_calls
        return out

    def clear(self) -> None:
        super().clear()
        self.n_kernel_calls = 0


# ----------------------------------------------------------------------
# Index operations on Gram matrices
# ----------------------------------------------------------------------
def sub(K: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """K restricted to (or reordered / resampled by) index vector idx."""
    return K[np.ix_(idx, idx)]


def product_gram(*Ks: np.ndarray) -> np.ndarray:
    """Gram of a separable joint kernel k((x,y),(x',y')) = k_X(x,x') k_Y(y,y')."""
    out = np.ones_like(Ks[0])
    for K in Ks:
        out = out * K
    return out


# ----------------------------------------------------------------------
# Statistics: Gram matrices in, scalar out
# ----------------------------------------------------------------------
def center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic(Kx: np.ndarray, Ky: np.ndarray) -> float:
    n = Kx.shape[0]
    if n < 3:
        return 0.0
    return float(np.trace(center(Kx) @ center(Ky)) / (n * n))


def marginal_stat(K: np.ndarray) -> float:
    n = K.shape[0]
    if n < 3:
        return 0.0
    Kc = center(K)
    return float(np.trace(Kc @ Kc) / (n * n))


def mmd2(K: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, biased: bool = True) -> float:
    """MMD^2 between groups a and b, from the Gram matrix of the pooled sample."""
    Kaa, Kbb, Kab = K[np.ix_(idx_a, idx_a)], K[np.ix_(idx_b, idx_b)], K[np.ix_(idx_a, idx_b)]
    na, nb = len(idx_a), len(idx_b)
    if biased:
        return float(Kaa.mean() + Kbb.mean() - 2 * Kab.mean())
    ta = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
    tb = (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
    return float(ta + tb - 2 * Kab.mean())


def kernel_ridge_residual(Kz: np.ndarray, Ky: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """
    Residual Gram of Y after kernel-ridge regression on Z (KCIT building block).
    R = (I - Kz (Kz + lam I)^-1);  returns R Ky R^T.
    """
    n = Kz.shape[0]
    R = np.eye(n) - Kz @ np.linalg.solve(Kz + lam * np.eye(n), np.eye(n))
    return R @ Ky @ R.T


# ----------------------------------------------------------------------
# Resampling schemes: all operate on Gram matrices by indices
# ----------------------------------------------------------------------
def perm_test_single(Kx: np.ndarray, Ky: np.ndarray,
                     stat: Callable[[np.ndarray, np.ndarray], float],
                     n_perm: int, rng: np.random.Generator) -> Tuple[float, float]:
    """Permute X (rows+cols of Kx) while Y is fixed.  Returns (stat, p)."""
    n = Kx.shape[0]
    obs = stat(Kx, Ky)
    count = 0
    for _ in range(n_perm):
        pi = rng.permutation(n)
        if abs(stat(sub(Kx, pi), Ky)) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def perm_test_split(Kx: np.ndarray, Ky: np.ndarray, labels: np.ndarray,
                    stat: Callable[[np.ndarray, np.ndarray], float],
                    n_perm: int, rng: np.random.Generator) -> Tuple[float, float]:
    """
    Split-statistic permutation test (QCSI do-orientation):
    delta = stat(group 1) - stat(group 0), labels shuffled.
    """
    def delta(lbl):
        i0, i1 = np.where(lbl == 0)[0], np.where(lbl == 1)[0]
        return stat(sub(Kx, i1), sub(Ky, i1)) - stat(sub(Kx, i0), sub(Ky, i0))

    obs = delta(labels)
    count = 0
    for _ in range(n_perm):
        if abs(delta(rng.permutation(labels))) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def perm_test_mmd(K: np.ndarray, labels: np.ndarray, n_perm: int,
                  rng: np.random.Generator) -> Tuple[float, float]:
    """Two-sample MMD permutation test on a pooled Gram matrix."""
    def m(lbl):
        return mmd2(K, np.where(lbl == 0)[0], np.where(lbl == 1)[0])

    obs = m(labels)
    count = sum(m(rng.permutation(labels)) >= obs for _ in range(n_perm))
    return obs, (count + 1) / (n_perm + 1)


def bootstrap(Ks: Tuple[np.ndarray, ...], stat: Callable[..., float],
              n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of stat(*Ks) by resampling indices with replacement."""
    n = Ks[0].shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = stat(*(sub(K, idx) for K in Ks))
    return out


def kfold_indices(n: int, k: int, rng: np.random.Generator):
    """Yield (train_idx, test_idx) for k-fold CV; use with sub()."""
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


# ----------------------------------------------------------------------
# Fallback for non-separable joint encodings: explicit n^2-point Gram
# ----------------------------------------------------------------------
class JointGram:
    """
    For a joint kernel k((x,y),(x',y')) that does NOT factorise, precompute
    the Gram over ALL pairs (x_a, y_b), a,b in [n].  Then any permutation
    pi of X corresponds to selecting the n points (x_pi(i), y_i), i.e. flat
    indices pi(i)*n + i.  Exact, but O(n^4) memory: use only for small n
    (n=30 -> 810k entries; n=100 -> 1e8; n=200 infeasible).
    """

    def __init__(self, joint_kernel_fn: KernelFn, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, float).reshape(len(X), -1)
        Y = np.asarray(Y, float).reshape(len(Y), -1)
        self.n = len(X)
        grid = np.array([np.concatenate([X[a], Y[b]]) for a in range(self.n) for b in range(self.n)])
        self.K_all = joint_kernel_fn(grid)

    def gram_for_perm(self, pi: np.ndarray) -> np.ndarray:
        flat = np.asarray(pi) * self.n + np.arange(self.n)
        return self.K_all[np.ix_(flat, flat)]

    def gram_identity(self) -> np.ndarray:
        return self.gram_for_perm(np.arange(self.n))

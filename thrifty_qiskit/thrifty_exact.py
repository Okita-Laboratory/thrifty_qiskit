"""
thrifty_exact.py -- exact reconstruction of band-limited quantum kernels,
zero-cost bandwidth sweeps, aggregated tests, and circuit budgets.

Motivation
----------
Existing libraries already avoid the obvious duplicates: Qiskit's
FidelityQuantumKernel evaluates only the upper triangle and skips the diagonal
(`evaluate_duplicates`), FidelityStatevectorKernel caches statevectors per data
point, and PennyLane's square_kernel_matrix does the same by symmetry.  Those
reduce the constant, not the exponent: a Gram matrix still costs O(n^2) circuits
and a bandwidth sweep of B values costs B times that.

This module exploits a structural fact instead.  For a feature map that encodes
each x_j through Pauli rotations, the fidelity kernel is a trigonometric
polynomial whose frequency set Omega is fixed by the encoding gates alone
(Schuld, arXiv:2101.11020; see also the 2026 erratum arXiv:2608.24923), and for
the standard single-qubit encodings it depends on x and y only through the
difference:

    k(x, y) = f(x - y) = sum_{w in Omega} c_w exp(i w . (x - y)).

Two consequences are implemented here.

1. EXACT reconstruction, not interpolation.  |Omega| difference points determine
   every coefficient c_w by solving one linear system, so the whole Gram matrix
   follows at machine precision from |Omega| circuit evaluations.  Grid-and-
   interpolate schemes trade accuracy for cost; here there is no trade.

2. Bandwidth sweeps are free.  Scaling the data, x -> gamma x, is the same as
   scaling the difference, so k_gamma(x, y) = f(gamma (x - y)) uses the SAME
   coefficients.  A sweep over B bandwidths -- the standard remedy for
   exponential kernel concentration -- costs zero additional circuits, and so
   does an aggregated test over those bandwidths.

Also here:
  * CircuitBudget: a context manager that caps and asserts circuit executions,
    turning "how many circuits did that cost" into a checkable property.
  * cross_hsic_test: a permutation-free independence test (studentised
    cross-statistic, normal calibration) for the case where even index-level
    resampling is unwanted.

Limits
------
Everything here needs the kernel to be shift-invariant AND band-limited with a
small, known Omega.  `verify_reconstruction` checks both numerically before use
and refuses otherwise -- the default ZZ feature map, whose phi_2 = (pi-x)(pi-y)
term is not a function of x-y, is correctly rejected.  This is also exactly the
regime in which a classical surrogate exists, so the saving here is a practical
one and not evidence of quantum advantage.
"""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

PairFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return X[:, None] if X.ndim == 1 else X


# ************************************************************
# Circuit budget
# ************************************************************
class CircuitBudget:
    """
    Context manager that caps circuit executions and reports the total.

    Wrap any object exposing `n_circuit_evals` (the kernels in this package do)
    or pass a callable returning the running count.  On exit the total is
    checked against `limit`; `assert_exactly` turns the count into an equality
    test, which is what the unit tests use to pin the circuit budget of a whole
    pipeline.

        with CircuitBudget(kernel, limit=201) as b:
            K = kernel(X)
            p = perm_test(K, ...)          # adds nothing
        b.used                              # 201
    """

    def __init__(self, source, limit: Optional[int] = None,
                 assert_exactly: Optional[int] = None):
        self.source = source
        self.limit = limit
        self.assert_exactly = assert_exactly
        self.start = 0
        self.used = 0

    def _count(self) -> int:
        if hasattr(self.source, "n_circuit_evals"):
            return int(self.source.n_circuit_evals)
        return int(self.source())

    def __enter__(self) -> "CircuitBudget":
        self.start = self._count()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.used = self._count() - self.start
        if exc_type is not None:
            return False
        if self.assert_exactly is not None and self.used != self.assert_exactly:
            raise AssertionError(
                f"circuit budget: used {self.used}, expected exactly {self.assert_exactly}")
        if self.limit is not None and self.used > self.limit:
            raise AssertionError(f"circuit budget exceeded: used {self.used} > {self.limit}")
        return False


# ************************************************************
# Frequency set and design matrix
# ************************************************************
def frequency_set(feature_dim: int, reps: int = 1, gate_freq: float = 2.0) -> np.ndarray:
    """
    Omega for `reps` re-uploads of a gate whose generator gives amplitude
    frequency `gate_freq` (P(2x) -> 2, exp(-i x P/2) -> 1).
    Per feature the difference frequencies are gate_freq * {-reps, ..., reps},
    so |Omega| = (2 reps + 1)^feature_dim.
    """
    per = gate_freq * np.arange(-reps, reps + 1)
    return np.array(list(itertools.product(per, repeat=feature_dim)), dtype=float)


def _design(D: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Real design matrix [cos(w.D), sin(w.D)] for difference rows D (N, d)."""
    ph = D @ omega.T
    return np.hstack([np.cos(ph), np.sin(ph)])


def _difference_nodes(omega: np.ndarray, d: int, reps: int, gate_freq: float) -> np.ndarray:
    """
    Nodes at which to evaluate f(Delta).  A tensor grid of 2*reps+1 equispaced
    points per feature over one period 2*pi/gate_freq is unisolvent for the
    trigonometric polynomial with frequency set Omega (Nyquist).
    """
    m = 2 * reps + 1
    period = 2 * np.pi / gate_freq
    base = np.linspace(0.0, period, m, endpoint=False)
    return np.array(list(itertools.product(base, repeat=d)), dtype=float)


# ************************************************************
# Exact band-limited kernel
# ************************************************************
class ExactShiftInvariantKernel:
    """
    Reconstruct k(x, y) = f(x - y) exactly from |Omega| circuit evaluations.

    Parameters
    ----------
    pair_fn : the true (quantum) kernel, called as pair_fn(A, B) -> (na, nb).
              Used only during `fit`.
    d       : feature dimension.
    reps    : number of re-uploads per feature.
    gate_freq : 2.0 for P(2x)-style encodings, 1.0 for exp(-i x P/2).
    check   : verify shift-invariance and reconstruction before accepting.

    After `fit`, `__call__` and `pairs` cost no circuits, and `with_bandwidth`
    returns a kernel for a rescaled feature map at no extra circuit cost.
    """

    def __init__(self, pair_fn: PairFn, d: int, reps: int = 1, gate_freq: float = 2.0,
                 check: bool = True, tol: float = 1e-8):
        self.pair_fn, self.d, self.reps, self.gate_freq = pair_fn, d, reps, gate_freq
        self.check, self.tol = check, tol
        self.omega = frequency_set(d, reps, gate_freq)
        self.coef_: Optional[np.ndarray] = None
        self.gamma = 1.0
        self.n_circuit_evals = 0     # circuits that determine the kernel
        self.n_check_evals = 0       # circuits spent on the one-off validity check
        self.residual_ = None

    # ---- fitting ----
    def fit(self) -> "ExactShiftInvariantKernel":
        nodes = _difference_nodes(self.omega, self.d, self.reps, self.gate_freq)
        zero = np.zeros((1, self.d))
        vals = self.pair_fn(nodes, zero)[:, 0]          # f(Delta) = k(Delta, 0)
        self.n_circuit_evals += len(nodes)
        A = _design(nodes, self.omega)
        self.coef_, *_ = np.linalg.lstsq(A, vals, rcond=None)
        self.residual_ = float(np.max(np.abs(A @ self.coef_ - vals)))
        if self.check:
            self._verify()
        return self

    def _verify(self, n_test: int = 4, rng: Optional[np.random.Generator] = None) -> None:
        """
        One-off validity check: shift-invariance and exactness of the
        reconstruction, on n_test^2 random pairs.  Its cost is O(1) in the sample
        size and is counted separately in `n_check_evals`.
        """
        rng = rng or np.random.default_rng(0)
        X = rng.uniform(0, np.pi, (n_test, self.d))
        Y = rng.uniform(0, np.pi, (n_test, self.d))
        shift = rng.uniform(-0.5, 0.5, (1, self.d))
        K_true = self.pair_fn(X, Y)
        self.n_check_evals += n_test * n_test
        if np.max(np.abs(self.pair_fn(X + shift, Y + shift) - K_true)) > 1e-6:
            raise ValueError("kernel is not shift-invariant; exact reconstruction does not apply")
        self.n_check_evals += n_test * n_test
        if np.max(np.abs(self._eval(X, Y) - K_true)) > self.tol:
            raise ValueError("reconstruction failed: the kernel is not band-limited on this Omega")

    # ---- evaluation (no circuits) ----
    def _eval(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A, B = _2d(A), _2d(B)
        D = (A[:, None, :] - B[None, :, :]).reshape(-1, self.d) * self.gamma
        return (_design(D, self.omega) @ self.coef_).reshape(len(A), len(B))

    def pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            self.fit()
        return self._eval(A, B)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        K = self.pairs(X, X)
        return 0.5 * (K + K.T)

    # ---- bandwidth: free ----
    def with_bandwidth(self, gamma: float) -> "ExactShiftInvariantKernel":
        """
        Kernel of the feature map with data scaled by `gamma`, sharing this
        object's coefficients and circuit count.  Costs no circuits.
        """
        if self.coef_ is None:
            self.fit()
        other = ExactShiftInvariantKernel.__new__(ExactShiftInvariantKernel)
        other.__dict__.update(self.__dict__)
        other.gamma = self.gamma * float(gamma)
        other.n_circuit_evals = 0            # this view adds nothing
        return other

    def bandwidth_sweep(self, X: np.ndarray, gammas: Sequence[float]) -> List[np.ndarray]:
        """Gram matrices for every bandwidth, at zero additional circuit cost."""
        return [self.with_bandwidth(g)(X) for g in gammas]


# ************************************************************
# Statistics on Gram matrices
# ************************************************************
def center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic(Kx: np.ndarray, Ky: np.ndarray) -> float:
    n = Kx.shape[0]
    return float(np.trace(center(Kx) @ center(Ky)) / (n * n))


def perm_test(Kx: np.ndarray, Ky: np.ndarray, n_perm: int,
              rng: np.random.Generator) -> Tuple[float, float]:
    obs = hsic(Kx, Ky)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Kx))
        if hsic(Kx[np.ix_(p, p)], Ky) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


# ************************************************************
# Aggregated multi-bandwidth test (all bandwidths share one fit)
# ************************************************************
def aggregated_hsic_test(kernel: ExactShiftInvariantKernel, X: np.ndarray, Y: np.ndarray,
                         gammas: Sequence[float], n_perm: int,
                         rng: np.random.Generator) -> Tuple[float, dict]:
    """
    Test independence by aggregating HSIC over a set of bandwidths, calibrated by
    a single permutation loop that is shared across bandwidths (the maximum of
    the standardised statistics is permuted jointly, so the level is controlled
    without a Bonferroni correction).

    Circuit cost: whatever `kernel.fit()` used.  Adding bandwidths or
    permutations adds none.
    """
    Kxs = [kernel.with_bandwidth(g)(X) for g in gammas]
    Kys = [kernel.with_bandwidth(g)(Y) for g in gammas]
    n = len(_2d(X))

    def stats(perm=None):
        out = []
        for Kx, Ky in zip(Kxs, Kys):
            Kxp = Kx if perm is None else Kx[np.ix_(perm, perm)]
            out.append(hsic(Kxp, Ky))
        return np.array(out)

    null = np.array([stats(rng.permutation(n)) for _ in range(n_perm)])
    scale = null.std(axis=0) + 1e-12
    obs = stats()
    T_obs = np.max(obs / scale)
    T_null = np.max(null / scale, axis=1)
    p = (np.sum(T_null >= T_obs) + 1) / (n_perm + 1)
    return float(p), {"per_bandwidth_hsic": obs, "T_obs": float(T_obs),
                      "gammas": list(gammas)}


# ************************************************************
# Permutation-free test (cross-HSIC)
# ************************************************************
def cross_hsic_test(Kx: np.ndarray, Ky: np.ndarray,
                    rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Studentised cross-HSIC: split the sample in two, form a cross-statistic whose
    null distribution is asymptotically normal, and calibrate against a normal
    quantile.  No permutations at all (cf. Shekhar-Kim-Ramdas style
    permutation-free kernel tests).

    Returns (statistic, one-sided p-value).
    """
    rng = rng or np.random.default_rng(0)
    n = Kx.shape[0]
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2: 2 * (n // 2)]
    m = len(a)
    Cx = center(Kx[np.ix_(a, b)])
    Cy = center(Ky[np.ix_(a, b)])
    terms = np.sum(Cx * Cy, axis=1) / m            # per-row contributions
    mean = terms.mean()
    sd = terms.std(ddof=1) / np.sqrt(m) + 1e-15
    t = mean / sd
    from math import erfc, sqrt
    p = 0.5 * erfc(t / sqrt(2.0))
    return float(t), float(p)


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    def z_pair(A, B, reps=1):
        """Exact Z-feature-map fidelity: prod_d cos^2(reps*(a_d - b_d))."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        K = np.ones((A.shape[0], B.shape[0]))
        for d in range(A.shape[1]):
            K *= np.cos(reps * (A[:, d][:, None] - B[:, d][None, :])) ** 2
        return K

    def zz_like(A, B):
        """Non-shift-invariant toy (mimics the default ZZ map)."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        p2 = lambda X: (np.pi - X[:, 0]) * (np.pi - X[:, 1])
        return z_pair(A, B) * np.cos(p2(A)[:, None] - p2(B)[None, :]) ** 2

    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, np.pi, (n, 1))
    y = np.mod(0.8 * x + 0.3 * rng.normal(size=(n, 1)), np.pi)

    print("=== exact reconstruction (d=1, reps=1) ===")
    k = ExactShiftInvariantKernel(lambda A, B: z_pair(A, B, 1), d=1, reps=1).fit()
    K = k(x)
    exact = z_pair(x, x, 1)
    print(f"  |Omega| = {len(k.omega)}, circuits to determine the kernel = {k.n_circuit_evals}, "
          f"one-off validity check = {k.n_check_evals}, naive n(n-1)/2 = {n*(n-1)//2}")
    print(f"  max |K - exact| = {np.max(np.abs(K - exact)):.2e}   (fit residual {k.residual_:.1e})")
    print(f"  recovered coefficients: {np.round(k.coef_, 6)}")

    print("\n=== bandwidth sweep is free ===")
    gammas = [0.25, 0.5, 1.0, 2.0, 4.0]
    before = k.n_circuit_evals
    Ks = k.bandwidth_sweep(x, gammas)
    print(f"  {len(gammas)} bandwidths, circuits added = {k.n_circuit_evals - before}")
    for g, Kg in zip(gammas, Ks):
        ref = z_pair(g * x, g * x, 1)
        print(f"   gamma={g:4.2f}: max err {np.max(np.abs(Kg - ref)):.2e}, "
              f"mean off-diagonal {np.mean(Kg[~np.eye(n,dtype=bool)]):.3f}")

    print("\n=== circuit budget over a whole pipeline ===")
    k2 = ExactShiftInvariantKernel(lambda A, B: z_pair(A, B, 1), d=1, reps=1, check=False)
    with CircuitBudget(k2, limit=10) as budget:
        Kx = k2(x); Ky = k2.pairs(y, y)
        _, p = perm_test(Kx, Ky, 500, rng)
        _ = k2.bandwidth_sweep(x, gammas)
    print(f"  500 permutations + {len(gammas)} bandwidths: circuits used = {budget.used}, p = {p:.4f}")

    print("\n=== aggregated multi-bandwidth test ===")
    p_agg, info = aggregated_hsic_test(k, x, y, gammas, 300, rng)
    z = rng.uniform(0, np.pi, (n, 1))
    p_null, _ = aggregated_hsic_test(k, x, z, gammas, 300, rng)
    print(f"  dependent x,y : p = {p_agg:.4f}")
    print(f"  independent   : p = {p_null:.4f}")
    print(f"  per-bandwidth HSIC = {np.round(info['per_bandwidth_hsic'], 5)}")

    print("\n=== permutation-free cross-HSIC ===")
    Kx, Ky, Kz = k(x), k.pairs(y, y), k.pairs(z, z)
    t1, p1 = cross_hsic_test(Kx, Ky, np.random.default_rng(1))
    t0, p0 = cross_hsic_test(Kx, Kz, np.random.default_rng(1))
    print(f"  dependent : t = {t1:7.2f}, p = {p1:.2e}")
    print(f"  independent: t = {t0:7.2f}, p = {p0:.3f}")

    print("\n=== refusal on a non-shift-invariant map ===")
    try:
        ExactShiftInvariantKernel(zz_like, d=2, reps=1).fit()
    except ValueError as e:
        print(f"  rejected as expected: {e}")

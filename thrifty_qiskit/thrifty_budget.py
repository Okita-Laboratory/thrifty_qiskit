"""
thrifty_budget.py -- what becomes free once the kernel is |Omega| coefficients.

thrifty_exact reduces a band-limited, shift-invariant quantum kernel to a
finite coefficient vector determined by |Omega| circuit evaluations.  This module
is the consequence: every downstream question that normally costs circuits
becomes a closed-form classical computation on those coefficients.

    k_gamma(x, y) = c_0 + sum_{w > 0} [ a_w cos(w gamma d) + b_w sin(w gamma d) ],
                    d = x - y.

Four things follow.

1. ERROR BARS.  With s_g shots at node g the estimate f_hat_g is binomial, so
   the coefficient covariance is  Sigma_c = A^+ diag(f(1-f)/s) A^{+T}  and the
   variance of any kernel entry, and of HSIC, follows by linear propagation.
   `gram_uncertainty` and `hsic_uncertainty` return these without any circuit.

2. SHOT BUDGETING.  Inverting the same expression gives the number of shots
   needed for a target kernel accuracy (`shots_for_precision`), and the optimal
   split of a fixed budget across nodes (`optimal_shot_allocation`) -- nodes are
   not equally informative, so uniform allocation is wasteful.

3. BANDWIDTH SELECTION AT ZERO COST.  Bandwidth is the standard remedy for
   exponential kernel concentration and is usually chosen by re-measuring the
   Gram matrix for each candidate.  Because gamma only rescales the difference,
   `select_bandwidth_kta` optimises kernel-target alignment over a continuum of
   gamma with no further circuits.

4. CONCENTRATION DIAGNOSIS.  The variance of off-diagonal kernel values over the
   data distribution is a function of the coefficients, so
   `concentration_curve` reports where a feature map collapses towards a constant
   kernel -- again without circuits.

Depends on thrifty_exact for the reconstruction itself.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from thrifty_qiskit.thrifty_exact import ExactShiftInvariantKernel, _2d, center, hsic


# ************************************************************
# Canonical real basis (non-negative frequencies only)
# ************************************************************
def canonical_frequencies(d: int, reps: int, gate_freq: float = 2.0) -> np.ndarray:
    """
    Frequency vectors for a minimal real basis: the lexicographically
    non-negative half of Omega.  For each w in this set the basis contributes
    cos(w.delta) and, unless w = 0, sin(w.delta), giving |Omega| real degrees of
    freedom in total.
    """
    per = gate_freq * np.arange(-reps, reps + 1)
    full = np.array(list(itertools.product(per, repeat=d)), dtype=float)
    keep = []
    for w in full:
        nz = w[np.abs(w) > 0]
        if len(nz) == 0 or nz[0] > 0:          # drop w if its mirror -w is kept
            keep.append(w)
    return np.array(keep)


def canonical_design(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Design matrix [1 | cos(w.D) | sin(w.D)] for w != 0, in canonical order."""
    D = _2d(D)
    ph = D @ W.T
    cols = []
    for j, w in enumerate(W):
        if np.allclose(w, 0):
            cols.append(np.ones(len(D)))
        else:
            cols.append(np.cos(ph[:, j]))
            cols.append(np.sin(ph[:, j]))
    return np.column_stack(cols)


class CanonicalKernel:
    """
    Exact band-limited kernel in the canonical real basis, carrying the node
    values, their shot counts, and the resulting coefficient covariance.
    """

    def __init__(self, d: int, reps: int, gate_freq: float = 2.0):
        self.d, self.reps, self.gate_freq = d, reps, gate_freq
        self.W = canonical_frequencies(d, reps, gate_freq)
        self.n_dof = sum(1 if np.allclose(w, 0) else 2 for w in self.W)
        self.nodes = self._nodes()
        self.coef_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.node_values_: Optional[np.ndarray] = None
        self.shots_: Optional[np.ndarray] = None
        self.n_circuit_evals = 0

    def _nodes(self) -> np.ndarray:
        m = 2 * self.reps + 1
        base = np.linspace(0.0, 2 * np.pi / self.gate_freq, m, endpoint=False)
        return np.array(list(itertools.product(base, repeat=self.d)), dtype=float)

    # ---- fit ----
    def fit(self, pair_fn: Callable, shots: Optional[Sequence[int]] = None,
            rng: Optional[np.random.Generator] = None) -> "CanonicalKernel":
        f = pair_fn(self.nodes, np.zeros((1, self.d)))[:, 0]
        self.n_circuit_evals += len(self.nodes)
        if shots is not None:
            rng = rng or np.random.default_rng(0)
            shots = np.asarray(shots, float)
            p = np.clip(f, 0, 1)
            f = rng.binomial(shots.astype(int), p) / shots
        self.node_values_ = np.asarray(f, float)
        self.shots_ = None if shots is None else np.asarray(shots, float)
        A = canonical_design(self.nodes, self.W)
        self.A_ = A
        self.Ap_ = np.linalg.pinv(A)
        self.coef_ = self.Ap_ @ self.node_values_
        self.cov_ = self._coef_cov()
        return self

    def _coef_cov(self) -> Optional[np.ndarray]:
        if self.shots_ is None:
            return np.zeros((len(self.coef_), len(self.coef_)))
        p = np.clip(self.node_values_, 1e-9, 1 - 1e-9)
        var_f = p * (1 - p) / self.shots_
        return self.Ap_ @ np.diag(var_f) @ self.Ap_.T

    # ---- evaluation ----
    def pairs(self, A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        A, B = _2d(A), _2d(B)
        D = (A[:, None, :] - B[None, :, :]).reshape(-1, self.d) * gamma
        return (canonical_design(D, self.W) @ self.coef_).reshape(len(A), len(B))

    def __call__(self, X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        K = self.pairs(X, X, gamma)
        return 0.5 * (K + K.T)


# ************************************************************
# 1. Error bars: shot noise -> kernel entries -> HSIC
# ************************************************************
def _entry_design(ck: CanonicalKernel, X: np.ndarray, gamma: float,
                  symmetrised: bool = True) -> np.ndarray:
    """
    Rows b such that Gram.ravel() = b @ coef.  With `symmetrised` the entry is
    (k(d) + k(-d))/2, which is what CanonicalKernel.__call__ returns; the sine
    columns cancel, so the odd-frequency noise does not reach the Gram matrix.
    """
    X = _2d(X)
    D = (X[:, None, :] - X[None, :, :]).reshape(-1, ck.d) * gamma
    B = canonical_design(D, ck.W)
    if symmetrised:
        B = 0.5 * (B + canonical_design(-D, ck.W))
    return B


def gram_uncertainty(ck: CanonicalKernel, X: np.ndarray, gamma: float = 1.0,
                     symmetrised: bool = True) -> np.ndarray:
    """
    Standard deviation of each Gram entry induced by node shot noise.
    Symmetrising the matrix removes the sine (odd-frequency) contribution, so the
    default accounts for that cancellation; pass symmetrised=False for the raw
    one-sided estimate.
    """
    B = _entry_design(ck, X, gamma, symmetrised)
    var = np.einsum("ij,jk,ik->i", B, ck.cov_, B)
    n = len(_2d(X))
    return np.sqrt(np.clip(var, 0, None)).reshape(n, n)


def hsic_uncertainty(ck: CanonicalKernel, X: np.ndarray, Ky: np.ndarray,
                     gamma: float = 1.0) -> Tuple[float, float]:
    """
    HSIC(Kx, Ky) and its standard deviation, propagating the coefficient
    covariance through the linear map coef -> Kx -> HSIC.
    HSIC is linear in Kx for fixed Ky, so the propagation is exact:
    HSIC = tr(H Kx H Ky)/n^2 = <Kx, H Ky H>/n^2.
    """
    X = _2d(X)
    n = len(X)
    G = center(Ky) / (n * n)                            # HSIC = <Kx, G>
    B = _entry_design(ck, X, gamma)                     # Kx.ravel() = B @ coef
    g = G.ravel() @ B                                   # HSIC = g . coef
    value = float(g @ ck.coef_)
    sd = float(np.sqrt(max(g @ ck.cov_ @ g, 0.0)))
    return value, sd


# ************************************************************
# 2. Shot budgeting
# ************************************************************
def _node_leverage(ck: CanonicalKernel, X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    How much each node's variance contributes to the Gram matrix error:
    l_g = sum over entries of (B A^+ e_g)^2, computed once per configuration.
    """
    M = _entry_design(ck, X, gamma) @ ck.Ap_            # (n^2, G): sensitivity to node g
    return np.sum(M ** 2, axis=0)


def optimal_shot_allocation(ck: CanonicalKernel, X: np.ndarray, total_shots: int,
                            gamma: float = 1.0, min_shots: int = 64) -> np.ndarray:
    """
    Split `total_shots` across nodes to minimise the total Gram variance.
    With variance f(1-f)/s_g and leverage l_g, Lagrange gives
    s_g proportional to sqrt(l_g f_g (1 - f_g)) -- nodes whose value sits near
    1/2 and which influence many entries deserve more shots.
    """
    lev = _node_leverage(ck, X, gamma)
    p = np.clip(ck.node_values_, 1e-6, 1 - 1e-6)
    w = np.sqrt(lev * p * (1 - p))
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    G = len(w)
    # reserve the floor for every node, then split what is left by weight, so
    # that the total never exceeds the budget
    free = max(0, total_shots - G * min_shots)
    return (min_shots + np.floor(free * w)).astype(int)


def shots_for_precision(ck: CanonicalKernel, X: np.ndarray, target_sd: float,
                        gamma: float = 1.0, uniform: bool = False) -> int:
    """
    Total shots needed so that the RMS standard deviation of the Gram entries is
    at most `target_sd`.  Variance scales as 1/s, so one calibration run at a
    reference budget fixes the constant.
    """
    ref = 10_000
    s_ref = (np.full(len(ck.nodes), ref // len(ck.nodes)) if uniform
             else optimal_shot_allocation(ck, X, ref, gamma))
    saved = ck.shots_
    ck.shots_ = s_ref.astype(float)
    ck.cov_ = ck._coef_cov()
    sd = np.sqrt(np.mean(gram_uncertainty(ck, X, gamma) ** 2))
    ck.shots_ = saved
    ck.cov_ = ck._coef_cov()
    return int(np.ceil(ref * (sd / target_sd) ** 2))


# ************************************************************
# 3. Bandwidth selection at zero circuit cost
# ************************************************************
def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """KTA between a Gram matrix and the ideal kernel yy^T of labels y in {-1,+1}."""
    T = np.outer(y, y).astype(float)
    Kc, Tc = center(K), center(T)
    denom = np.linalg.norm(Kc) * np.linalg.norm(Tc)
    return float(np.sum(Kc * Tc) / denom) if denom > 0 else 0.0


def select_bandwidth_kta(ck: CanonicalKernel, X: np.ndarray, y: np.ndarray,
                         gammas: Sequence[float]) -> Tuple[float, np.ndarray]:
    """
    Choose the bandwidth maximising KTA.  Costs no circuits: every candidate
    reuses the same coefficients.  Returns (best gamma, KTA per gamma).
    """
    scores = np.array([kernel_target_alignment(ck(X, g), y) for g in gammas])
    return float(gammas[int(np.argmax(scores))]), scores


def select_bandwidth_hsic(ck: CanonicalKernel, X: np.ndarray, Ky: np.ndarray,
                          gammas: Sequence[float]) -> Tuple[float, np.ndarray]:
    """Bandwidth maximising HSIC against a fixed second kernel.  Also free."""
    scores = np.array([hsic(ck(X, g), Ky) for g in gammas])
    return float(gammas[int(np.argmax(scores))]), scores


# ************************************************************
# 4. Concentration diagnosis
# ************************************************************
def concentration_curve(ck: CanonicalKernel, X: np.ndarray,
                        gammas: Sequence[float]) -> Dict[str, np.ndarray]:
    """
    Variance of the off-diagonal kernel values as a function of bandwidth.
    A kernel that concentrates (all entries equal) has variance -> 0 and cannot
    separate anything however many shots are spent; this diagnoses it for free.
    Also reports the shots needed to resolve one standard deviation of the
    kernel distribution, which is the practical cost of working at that gamma.
    """
    X = _2d(X)
    n = len(X)
    off = ~np.eye(n, dtype=bool)
    var, mean, shots = [], [], []
    for g in gammas:
        K = ck(X, g)
        v = float(np.var(K[off]))
        m = float(np.mean(K[off]))
        var.append(v); mean.append(m)
        # shots at which binomial noise on a single entry matches the spread of
        # the kernel values themselves; below this the kernel cannot separate points
        shots.append(np.inf if v <= 0 else float(max(1.0, m * (1 - m) / v)))
    return {"gamma": np.asarray(gammas, float), "variance": np.array(var),
            "mean": np.array(mean), "shots_to_resolve": np.array(shots)}


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    def z_pair(A, B, reps=1):
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        K = np.ones((A.shape[0], B.shape[0]))
        for d in range(A.shape[1]):
            K *= np.cos(reps * (A[:, d][:, None] - B[:, d][None, :])) ** 2
        return K

    rng = np.random.default_rng(0)
    n = 150
    x = rng.uniform(0, np.pi, (n, 1))
    y_lab = np.sign(np.sin(2 * x[:, 0]))
    y_var = np.mod(0.8 * x + 0.3 * rng.normal(size=(n, 1)), np.pi)

    print("=== canonical basis ===")
    ck = CanonicalKernel(d=1, reps=1).fit(lambda A, B: z_pair(A, B, 1))
    print(f"  nodes = {len(ck.nodes)}, degrees of freedom = {ck.n_dof}, "
          f"circuits = {ck.n_circuit_evals}")
    print(f"  coefficients = {np.round(ck.coef_, 6)}   (const, cos2, sin2)")
    print(f"  max |K - exact| = {np.max(np.abs(ck(x) - z_pair(x, x, 1))):.2e}")

    print("\n=== 1. error bars from node shot noise (no extra circuits) ===")
    print("  (the whole Gram matrix moves coherently with only |Omega| noisy nodes,")
    print("   so the check averages the realised RMS error over 200 draws)")
    Ky_ref = z_pair(y_var, y_var, 1)
    for shots in (256, 4096):
        errs, hs = [], []
        pred_sd = pred_hsd = None
        for t in range(200):
            ckn = CanonicalKernel(d=1, reps=1).fit(
                lambda A, B: z_pair(A, B, 1), shots=[shots] * 3,
                rng=np.random.default_rng(1000 + t))
            errs.append(np.sqrt(np.mean((ckn(x) - z_pair(x, x, 1)) ** 2)))
            h, hsd = hsic_uncertainty(ckn, x, Ky_ref)
            hs.append(h)
            if t == 0:
                pred_sd = np.sqrt(np.mean(gram_uncertainty(ckn, x) ** 2))
                pred_hsd = hsd
        print(f"  shots/node={shots:5d}: predicted Gram sd {pred_sd:.4f}, "
              f"realised RMS {np.mean(errs):.4f} | "
              f"HSIC sd predicted {pred_hsd:.5f}, realised {np.std(hs):.5f}")

    print("\n=== 2. shot allocation ===")
    ckn = CanonicalKernel(d=1, reps=1).fit(lambda A, B: z_pair(A, B, 1),
                                           shots=[2000] * 3, rng=np.random.default_rng(2))
    alloc = optimal_shot_allocation(ckn, x, 6000)
    print(f"  node values {np.round(ckn.node_values_, 3)}")
    print(f"  uniform      {np.array([2000, 2000, 2000])}")
    print(f"  optimal      {alloc}")
    for name, s in (("uniform", np.array([2000.0] * 3)), ("optimal", alloc.astype(float))):
        ckn.shots_ = s; ckn.cov_ = ckn._coef_cov()
        print(f"    {name:8s}: RMS Gram sd = {np.sqrt(np.mean(gram_uncertainty(ckn, x)**2)):.5f}")
    print(f"  shots for Gram sd 0.005: optimal {shots_for_precision(ckn, x, 0.005)}, "
          f"uniform {shots_for_precision(ckn, x, 0.005, uniform=True)}")

    print("\n=== 3. bandwidth selection at zero circuit cost ===")
    gammas = np.linspace(0.1, 4.0, 40)
    before = ck.n_circuit_evals
    g_kta, kta = select_bandwidth_kta(ck, x, y_lab, gammas)
    g_hsic, hs = select_bandwidth_hsic(ck, x, z_pair(y_var, y_var, 1), gammas)
    print(f"  {len(gammas)} candidates, circuits added = {ck.n_circuit_evals - before}")
    print(f"  best gamma by KTA  = {g_kta:.2f} (KTA {kta.max():.3f} vs {kta[0]:.3f} at gamma=0.1)")
    print(f"  best gamma by HSIC = {g_hsic:.2f}")

    print("\n=== 4. concentration diagnosis ===")
    diag = concentration_curve(ck, x, [0.05, 0.2, 1.0, 4.0, 16.0])
    for g, v, m, s in zip(diag["gamma"], diag["variance"], diag["mean"], diag["shots_to_resolve"]):
        print(f"  gamma={g:5.2f}: mean {m:.3f}, variance {v:.2e}, "
              f"shots to resolve ~ {s:,.0f}")

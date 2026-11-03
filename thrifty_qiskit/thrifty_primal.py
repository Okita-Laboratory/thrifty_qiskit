"""
thrifty_primal.py -- the primal form of a band-limited quantum kernel, and
the stability of decisions made from it.

Once the kernel is |Omega| non-negative cosine coefficients (thrifty_psd),
two further things follow, both classical and both free.

(1) AN EXPLICIT FEATURE MAP.  Positivity means every a_w >= 0, so sqrt(a_w) is
    real and

        k(x, y) = a_0 + sum_{w>0} a_w cos(w . (x - y))
                = phi(x) . phi(y),
        phi(x)  = [ sqrt(a_0),
                    sqrt(a_w) cos(w . x), sqrt(a_w) sin(w . x)  for w > 0 ].

    This is an exact, finite, explicit feature map of dimension |Omega|.  A
    quantum-kernel SVM or ridge regression can therefore be solved in the primal:
    training becomes O(n |Omega|^2) instead of O(n^3), prediction O(|Omega|) per
    point instead of O(n_SV), and the n x n Gram matrix never has to be formed.
    Approximate versions of this (random or deterministic Fourier features) are
    used to scale quantum kernel machines; here the features are exact and their
    number is fixed by the encoding rather than chosen as an approximation
    parameter.

    The dimension is |Omega| = (2L+1)^d, so the primal form is the right choice
    in low feature dimension.  `feature_dimension_report` says where the
    crossover against the dual form lies.

(2) DECISION STABILITY.  The coefficient covariance from shot noise
    (thrifty_budget) is a full description of the uncertainty, so resampling
    coefficients from it answers questions that normally need repeated
    experiments: does the selected bandwidth move?  would the test have reached
    the opposite conclusion?  `coefficient_bootstrap` does this with no circuits.

Depends on thrifty_budget and thrifty_psd.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from thrifty_qiskit.thrifty_budget import CanonicalKernel, center, hsic, kernel_target_alignment
from thrifty_qiskit.thrifty_exact import _2d
from thrifty_qiskit.thrifty_psd import coefficient_layout, is_bochner_psd, psd_project


# ************************************************************
# (1) Explicit feature map
# ************************************************************
class ExplicitFeatures:
    """
    phi(x) with k(x, y) = phi(x) . phi(y), built from non-negative coefficients.

    `project` runs the Bochner projection first when the coefficients are not
    already valid, since sqrt(a_w) must be real.
    """

    def __init__(self, ck: CanonicalKernel, gamma: float = 1.0, project: bool = True):
        if project and not is_bochner_psd(ck):
            ck = psd_project(ck)
        if not is_bochner_psd(ck, tol=1e-8):
            raise ValueError("coefficients are not positive; run psd_project first")
        self.ck, self.gamma = ck, float(gamma)
        lay = coefficient_layout(ck)
        a = np.clip(ck.coef_[lay["cos"]], 0.0, None)
        self.W = ck.W
        self.amp = np.sqrt(a)
        self.is_zero = np.array([np.allclose(w, 0) for w in ck.W])
        self.dim = int(np.sum(np.where(self.is_zero, 1, 2)))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Feature matrix Phi of shape (n, dim)."""
        X = _2d(X) * self.gamma
        ph = X @ self.W.T
        cols = []
        for j in range(len(self.W)):
            if self.is_zero[j]:
                cols.append(np.full(len(X), self.amp[j]))
            else:
                cols.append(self.amp[j] * np.cos(ph[:, j]))
                cols.append(self.amp[j] * np.sin(ph[:, j]))
        return np.column_stack(cols)

    def gram(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        P = self(X)
        return P @ P.T if Y is None else P @ self(Y).T


def feature_dimension_report(d: int, reps: int, n: int) -> Dict[str, int]:
    """Cost of the primal versus the dual form for a given problem shape."""
    D = (2 * reps + 1) ** d
    return {"feature_dim": D,
            "primal_train": n * D * D,
            "dual_train": n ** 3,
            "primal_predict": D,
            "dual_predict": n,
            "primal_memory": n * D,
            "dual_memory": n * n}


# ---- primal learners (plain numpy, to keep the comparison honest) ----
def ridge_primal(Phi: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    D = Phi.shape[1]
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(D), Phi.T @ y)


def ridge_dual(K: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    n = K.shape[0]
    return np.linalg.solve(K + lam * np.eye(n), y)


# ************************************************************
# (2) Decision stability from the coefficient covariance
# ************************************************************
def coefficient_bootstrap(ck: CanonicalKernel, statistic: Callable[[CanonicalKernel], object],
                          n_boot: int = 200, rng: Optional[np.random.Generator] = None,
                          project: bool = True) -> List[object]:
    """
    Repeat the experiment in silico: resample each node value from its binomial
    sampling distribution, refit the coefficients, and recompute `statistic`.
    With `project`, every draw is pushed back onto the valid (Bochner) set, so
    the bootstrap stays inside the space of real kernels.  Costs no circuits.
    """
    rng = rng or np.random.default_rng(0)
    if ck.shots_ is None:
        return [statistic(ck) for _ in range(n_boot)]       # no noise: constant
    p = np.clip(ck.node_values_, 0.0, 1.0)
    out = []
    for _ in range(n_boot):
        f = rng.binomial(ck.shots_.astype(int), p) / ck.shots_
        draw = _view(ck)
        draw.node_values_ = f
        draw.coef_ = ck.Ap_ @ f
        draw.cov_ = draw._coef_cov()
        if project:
            draw = psd_project(draw)
        out.append(statistic(draw))
    return out


def _view(ck: CanonicalKernel) -> CanonicalKernel:
    other = CanonicalKernel.__new__(CanonicalKernel)
    other.__dict__.update(ck.__dict__)
    other.n_circuit_evals = 0
    return other


def bandwidth_stability(ck: CanonicalKernel, X: np.ndarray, y: np.ndarray,
                        gammas: Sequence[float], n_boot: int = 200,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """
    How often each candidate bandwidth would have been selected, had the nodes
    been measured again.  A selection that flips between draws is not supported
    by the data at that shot count.
    """
    gammas = list(gammas)

    def pick(k: CanonicalKernel) -> int:
        return int(np.argmax([kernel_target_alignment(k(X, g), y) for g in gammas]))

    picks = coefficient_bootstrap(ck, pick, n_boot, rng)
    counts = np.bincount(picks, minlength=len(gammas))
    best = int(np.argmax(counts))
    return {"gammas": np.array(gammas), "counts": counts,
            "selected": float(gammas[best]),
            "selection_frequency": float(counts[best] / max(counts.sum(), 1)),
            "point_estimate": float(gammas[pick(ck)])}


def decision_stability(ck: CanonicalKernel, X: np.ndarray, Ky: np.ndarray,
                       threshold: float, n_boot: int = 200,
                       rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Probability that a dependence decision (HSIC above a threshold) would flip
    under a fresh measurement of the same nodes.
    """
    vals = np.array(coefficient_bootstrap(ck, lambda k: hsic(k(X), Ky), n_boot, rng))
    point = hsic(ck(X), Ky)
    return {"point": float(point), "mean": float(vals.mean()), "sd": float(vals.std()),
            "p_above": float(np.mean(vals > threshold)),
            "flip_probability": float(np.mean((vals > threshold) != (point > threshold)))}


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
    n, d, reps = 300, 1, 2
    x = rng.uniform(0, np.pi, (n, d))
    y = np.sin(3 * x[:, 0]) + 0.1 * rng.normal(size=n)
    lab = np.sign(y)

    print("=== (1) explicit feature map ===")
    ck = CanonicalKernel(d=d, reps=reps).fit(lambda A, B: z_pair(A, B, 1))
    feat = ExplicitFeatures(ck)
    Phi = feat(x)
    K = ck(x)
    print(f"  |Omega| = {ck.n_dof}, feature dimension = {feat.dim}")
    print(f"  max |Phi Phi^T - K| = {np.max(np.abs(Phi @ Phi.T - K)):.2e}")

    lam = 1e-3
    w = ridge_primal(Phi, y, lam)
    alpha = ridge_dual(K, y, lam)
    xt = rng.uniform(0, np.pi, (50, d))
    pred_primal = feat(xt) @ w
    pred_dual = ck.pairs(xt, x) @ alpha
    print(f"  primal vs dual predictions: max difference {np.max(np.abs(pred_primal - pred_dual)):.2e}")

    rep = feature_dimension_report(d, reps, n)
    print(f"  training  : primal {rep['primal_train']:,} flops vs dual {rep['dual_train']:,}")
    print(f"  prediction: primal {rep['primal_predict']} vs dual {rep['dual_predict']} per point")
    print(f"  memory    : primal {rep['primal_memory']:,} vs dual {rep['dual_memory']:,}")
    print("  crossover in d (reps=2, n=300): primal is cheaper while (2L+1)^d < n")
    for dd in (1, 2, 3, 4, 5):
        r = feature_dimension_report(dd, 2, 300)
        tag = "primal" if r["feature_dim"] < 300 else "dual"
        print(f"    d={dd}: feature dim {r['feature_dim']:6d} -> prefer {tag}")

    print("\n=== (2) decision stability under shot noise (no circuits) ===")
    gammas = np.linspace(0.3, 3.0, 10)
    print("  which bandwidth would be selected if the nodes were measured again?")
    for shots in (16, 64, 2048):
        ckn = CanonicalKernel(d=d, reps=reps).fit(
            lambda A, B: z_pair(A, B, 1), shots=[shots] * (2 * reps + 1),
            rng=np.random.default_rng(shots))
        bs = bandwidth_stability(ckn, x, lab, gammas, n_boot=300,
                                 rng=np.random.default_rng(1))
        top = np.argsort(bs["counts"])[::-1][:3]
        share = ", ".join(f"{bs['gammas'][i]:.1f}:{bs['counts'][i]/bs['counts'].sum():.0%}"
                          for i in top if bs["counts"][i] > 0)
        print(f"   shots/node={shots:5d}: point estimate {bs['point_estimate']:.1f}; "
              f"draws pick {share}")

    print("\n  would the dependence decision flip?  (threshold set near the estimate)")
    y2 = np.mod(0.8 * x + 0.3 * rng.normal(size=(n, d)), np.pi)
    Ky = z_pair(y2, y2, 1)
    for shots, thr in ((64, 0.079), (2048, 0.079), (2048, 0.050)):
        ckn = CanonicalKernel(d=d, reps=reps).fit(
            lambda A, B: z_pair(A, B, 1), shots=[shots] * (2 * reps + 1),
            rng=np.random.default_rng(shots))
        ds = decision_stability(ckn, x, Ky, threshold=thr, n_boot=300,
                                rng=np.random.default_rng(2))
        print(f"   shots/node={shots:5d}, threshold {thr:.3f}: HSIC {ds['point']:.4f} "
              f"+- {ds['sd']:.4f}, flip probability {ds['flip_probability']:.2f}")

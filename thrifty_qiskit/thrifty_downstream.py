"""
thrifty_downstream.py -- four downstream methods that become free.

These are not new principles.  Each is an existing technique whose published
cost is dominated by kernel evaluations, and each collapses to a classical
computation once the kernel is |Omega| coefficients.  They are collected here
because together they show how far the consequence reaches.

(1) CENTROID KERNELS.  Comparing every point against a few class centroids
    rather than against every other point takes the training cost of a quantum
    kernel classifier from O(n^2) to O(n) circuit evaluations (Tscharke, Issel,
    Debus, IEEE QCE 2024).  With a reconstructed kernel the centroids can be
    placed in the feature space, moved, and re-evaluated at no circuit cost --
    and the "centroid" need not be a data point at all.

(2) KERNEL INDUCING POINTS.  Summarising a dataset by a small set of synthetic
    points requires many kernel evaluations while the inducing points are being
    optimised, which is what makes the idea expensive for quantum kernels
    (cf. kernel-inducing-point distillation).  Here the optimisation runs
    entirely on the coefficients.

(3) LEAVE-ONE-OUT.  For kernel ridge regression the leave-one-out error has a
    closed form in terms of the hat matrix, so model selection over
    regularisation and bandwidth needs no refitting -- and no circuits, since
    every bandwidth shares the same coefficients.

(4) INPUT GRADIENTS.  d k / d x is another trigonometric polynomial with the
    same frequencies, so gradients with respect to the data are exact and free.
    This gives attribution, sensitivity and gradient-based placement of
    inducing points without the parameter-shift evaluations they would normally
    require.

Depends on thrifty_budget, thrifty_psd and thrifty_primal.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from thrifty_qiskit.thrifty_budget import CanonicalKernel, canonical_design, center
from thrifty_qiskit.thrifty_exact import _2d
from thrifty_qiskit.thrifty_primal import ExplicitFeatures, ridge_dual, ridge_primal


# ************************************************************
# (1) Centroid kernels
# ************************************************************
class CentroidKernel:
    """
    Class centroids in the explicit feature space.

    The centroid of a class is the mean of its feature vectors, so the
    similarity of a point to a class is phi(x) . mu_c -- one inner product in
    |Omega| dimensions.  Because phi is exact, the centroids are exact too, and
    moving or re-weighting them costs nothing.
    """

    def __init__(self, ck: CanonicalKernel, gamma: float = 1.0):
        self.feat = ExplicitFeatures(ck, gamma=gamma)
        self.classes_: Optional[np.ndarray] = None
        self.mu_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CentroidKernel":
        P = self.feat(X)
        self.classes_ = np.unique(y)
        self.mu_ = np.stack([P[y == c].mean(axis=0) for c in self.classes_])
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Similarity of each point to each centroid: (n, n_classes)."""
        return self.feat(X) @ self.mu_.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.scores(X), axis=1)]

    def kernel_to_centroids(self, X: np.ndarray) -> np.ndarray:
        """The n x n_classes matrix that a centroid method would have measured."""
        return self.scores(X)


def centroid_cost(n: int, n_classes: int) -> Dict[str, int]:
    """Circuit evaluations for the centroid method, measured against the full Gram."""
    return {"full_gram": n * (n - 1) // 2, "centroid_direct": n * n_classes,
            "reconstructed": 0}


# ************************************************************
# (2) Kernel inducing points
# ************************************************************
def induced_prediction(ck: CanonicalKernel, Z: np.ndarray, w: np.ndarray,
                       X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Prediction of a model supported on inducing points Z with weights w."""
    return ck.pairs(X, Z, gamma) @ w


def fit_inducing_weights(ck: CanonicalKernel, Z: np.ndarray, X: np.ndarray,
                         y: np.ndarray, lam: float = 1e-3,
                         gamma: float = 1.0) -> np.ndarray:
    """Least-squares weights on a fixed set of inducing points."""
    Kxz = ck.pairs(X, Z, gamma)
    m = Kxz.shape[1]
    return np.linalg.solve(Kxz.T @ Kxz + lam * np.eye(m), Kxz.T @ y)


def optimise_inducing_points(ck: CanonicalKernel, X: np.ndarray, y: np.ndarray,
                             m: int, n_iter: int = 200, step: float = 0.05,
                             lam: float = 1e-3, gamma: float = 1.0,
                             rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Place m synthetic inducing points by gradient descent on the training error.
    Every evaluation of the objective and of its gradient uses the reconstructed
    kernel, so the whole optimisation costs no circuits.
    """
    rng = rng or np.random.default_rng(0)
    X = _2d(X)
    Z = X[rng.choice(len(X), m, replace=False)].copy()
    hist = []
    for it in range(n_iter):
        w = fit_inducing_weights(ck, Z, X, y, lam, gamma)
        r = induced_prediction(ck, Z, w, X, gamma) - y
        hist.append(float(np.mean(r ** 2)))
        # dL/dZ through K(X, Z): uses the analytic input gradient of the kernel
        G = kernel_gradient_wrt_second(ck, X, Z, gamma)        # (n, m, d)
        grad = np.einsum("n,m,nmd->md", 2 * r / len(X), w, G)
        Z -= step * grad
    w = fit_inducing_weights(ck, Z, X, y, lam, gamma)
    hist.append(float(np.mean((induced_prediction(ck, Z, w, X, gamma) - y) ** 2)))
    return {"Z": Z, "w": w, "mse": np.array(hist)}


# ************************************************************
# (3) Leave-one-out in closed form
# ************************************************************
def loo_error(K: np.ndarray, y: np.ndarray, lam: float) -> float:
    """
    Leave-one-out mean squared error of kernel ridge regression, in closed form:
        e_i = (alpha_i) / (H^{-1})_{ii},  H = K + lam I,  alpha = H^{-1} y.
    No refitting, no circuits.
    """
    n = len(y)
    H = K + lam * np.eye(n)
    Hi = np.linalg.inv(H)
    alpha = Hi @ y
    return float(np.mean((alpha / np.diag(Hi)) ** 2))


def select_by_loo(ck: CanonicalKernel, X: np.ndarray, y: np.ndarray,
                  gammas: Sequence[float], lams: Sequence[float]) -> Dict:
    """
    Joint model selection over bandwidth and regularisation by leave-one-out.
    The bandwidth grid is free (shared coefficients) and the LOO error needs no
    refitting, so the whole grid costs nothing.
    """
    best = (np.inf, None, None)
    table = np.empty((len(gammas), len(lams)))
    for i, g in enumerate(gammas):
        K = ck(X, g)
        for j, lam in enumerate(lams):
            e = loo_error(K, y, lam)
            table[i, j] = e
            if e < best[0]:
                best = (e, float(g), float(lam))
    return {"loo": best[0], "gamma": best[1], "lam": best[2],
            "table": table, "gammas": np.asarray(gammas), "lams": np.asarray(lams)}


# ************************************************************
# (4) Analytic input gradients
# ************************************************************
def _grad_design(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    d/dD of the canonical design row, returned as (N, dof, d).
    d/dD cos(w.D) = -sin(w.D) w ;  d/dD sin(w.D) = cos(w.D) w.
    """
    D = _2d(D)
    ph = D @ W.T
    blocks = []
    for j, w in enumerate(W):
        if np.allclose(w, 0):
            blocks.append(np.zeros((len(D), 1, D.shape[1])))
        else:
            c = np.cos(ph[:, j])[:, None] * w[None, :]
            s = -np.sin(ph[:, j])[:, None] * w[None, :]
            blocks.append(np.stack([s, c], axis=1))
    return np.concatenate(blocks, axis=1)


def kernel_gradient_wrt_first(ck: CanonicalKernel, A: np.ndarray, B: np.ndarray,
                              gamma: float = 1.0) -> np.ndarray:
    """d k(a_i, b_j) / d a_i, shape (na, nb, d).  Exact and free."""
    A, B = _2d(A), _2d(B)
    D = (A[:, None, :] - B[None, :, :]).reshape(-1, ck.d) * gamma
    G = _grad_design(D, ck.W)                                  # (na*nb, dof, d)
    out = np.einsum("kjd,j->kd", G, ck.coef_) * gamma
    return out.reshape(len(A), len(B), ck.d)


def kernel_gradient_wrt_second(ck: CanonicalKernel, A: np.ndarray, B: np.ndarray,
                               gamma: float = 1.0) -> np.ndarray:
    """d k(a_i, b_j) / d b_j: the same quantity with the opposite sign."""
    return -kernel_gradient_wrt_first(ck, A, B, gamma)


def prediction_gradient(ck: CanonicalKernel, Xtrain: np.ndarray, alpha: np.ndarray,
                        Xtest: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gradient of a kernel ridge prediction with respect to the test input."""
    G = kernel_gradient_wrt_first(ck, Xtest, Xtrain, gamma)     # (nt, ntr, d)
    return np.einsum("ijd,j->id", G, alpha)


def feature_sensitivity(ck: CanonicalKernel, Xtrain: np.ndarray, alpha: np.ndarray,
                        Xtest: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Mean absolute prediction gradient per feature: a cheap attribution."""
    return np.mean(np.abs(prediction_gradient(ck, Xtrain, alpha, Xtest, gamma)), axis=0)


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
    n, d, reps = 300, 2, 2
    X = rng.uniform(0, np.pi, (n, d))
    y = np.sin(2 * X[:, 0]) + 0.15 * rng.normal(size=n)     # feature 1 is irrelevant
    lab = np.sign(y)
    ck = CanonicalKernel(d=d, reps=reps).fit(lambda A, B: z_pair(A, B, 1))
    print(f"kernel reconstructed from {ck.n_circuit_evals} circuits "
          f"(|Omega| = {ck.n_dof}); everything below adds none\n")

    print("=== (1) centroid kernel ===")
    cc = CentroidKernel(ck).fit(X, lab)
    acc = float(np.mean(cc.predict(X) == lab))
    cost = centroid_cost(n, len(cc.classes_))
    print(f"  training accuracy {acc:.3f} with {len(cc.classes_)} centroids")
    print(f"  circuits: full Gram {cost['full_gram']:,}, direct centroid method "
          f"{cost['centroid_direct']:,}, reconstructed {cost['reconstructed']}")

    print("\n=== (2) kernel inducing points ===")
    out = optimise_inducing_points(ck, X, y, m=8, n_iter=150, step=0.08,
                                   rng=np.random.default_rng(1))
    K = ck(X)
    full = ridge_dual(K, y, 1e-3)
    mse_full = float(np.mean((K @ full - y) ** 2))
    print(f"  8 inducing points: MSE {out['mse'][0]:.4f} -> {out['mse'][-1]:.4f} "
          f"in {len(out['mse'])-1} steps")
    print(f"  full kernel ridge on all {n} points: MSE {mse_full:.4f}")
    print(f"  circuits used by the optimisation: 0")

    print("\n=== (3) leave-one-out model selection ===")
    gammas = np.linspace(0.2, 3.0, 15)
    lams = np.logspace(-5, 0, 12)
    sel = select_by_loo(ck, X, y, gammas, lams)
    print(f"  grid {len(gammas)} x {len(lams)} = {len(gammas)*len(lams)} fits, circuits 0")
    print(f"  best gamma {sel['gamma']:.2f}, lambda {sel['lam']:.1e}, LOO MSE {sel['loo']:.4f}")
    worst = sel["table"].max()
    print(f"  LOO ranges from {sel['loo']:.4f} to {worst:.4f} over the grid "
          f"({worst/sel['loo']:.0f}x), so the choice matters")

    print("\n=== (4) analytic input gradients ===")
    Kg = ck(X, sel["gamma"])
    alpha = ridge_dual(Kg, y, sel["lam"])
    Xt = rng.uniform(0, np.pi, (200, d))
    sens = feature_sensitivity(ck, X, alpha, Xt, sel["gamma"])
    print(f"  mean |d prediction / d x| per feature: {np.round(sens, 4)}")
    print(f"  (feature 0 carries the label, feature 1 does not)")
    eps = 1e-6
    g_num = np.zeros((5, d))
    for j in range(d):
        Xp = Xt[:5].copy(); Xp[:, j] += eps
        Xm = Xt[:5].copy(); Xm[:, j] -= eps
        g_num[:, j] = ((ck.pairs(Xp, X, sel["gamma"]) @ alpha
                        - ck.pairs(Xm, X, sel["gamma"]) @ alpha) / (2 * eps))
    g_an = prediction_gradient(ck, X, alpha, Xt[:5], sel["gamma"])
    print(f"  analytic vs finite difference: max |difference| "
          f"{np.max(np.abs(g_an - g_num)):.2e}")

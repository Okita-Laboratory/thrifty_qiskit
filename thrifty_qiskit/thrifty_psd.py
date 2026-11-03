"""
thrifty_psd.py -- positivity and noise correction in the coefficient domain.

Two problems reported in the quantum-kernel literature are solved here by
working on the |Omega| coefficients instead of the n x n matrix.

(1) INDEFINITE GRAM MATRICES.  Shot noise makes the estimated kernel matrix
    lose positive semidefiniteness, and the usual repairs act on the matrix
    itself -- eigenvalue thresholding, flipping or shifting (Hubregtsen et al.,
    PRA 105, 042431 (2022)).  Those cost O(n^3), must be redone for every sample
    set and every bandwidth, and change entries that were measured correctly.

    For a shift-invariant kernel, Bochner's theorem says positive definiteness
    is equivalent to non-negativity of the spectral measure.  With a finite
    frequency set that is simply

        b_w = 0  for all w      (the kernel is even),
        a_w >= 0 for all w      (non-negative cosine coefficients),

    so positivity is a non-negative least squares in |Omega| unknowns.  Solving
    it once yields a kernel that is positive semidefinite for EVERY sample set,
    EVERY sample size and EVERY bandwidth simultaneously.  Note that the
    symmetrisation used elsewhere in this package (K + K^T)/2 is exactly the
    b_w = 0 half of this projection.

(2) A CORRUPTED DIAGONAL.  On hardware the diagonal of a fidelity kernel, which
    is 1 by construction, is measured well below 1 -- Kiani (Next Materials,
    2026) reports 0.618 +- 0.12.  Under a global depolarising model the measured
    kernel is an affine image of the true one,

        k_meas(delta) = (1 - lam) k(delta) + lam / 2^q,

    and delta = 0 is already one of the |Omega| nodes, so lam is determined with
    no extra circuits at all:

        lam = (1 - k_meas(0)) / (1 - 2^-q).

    Inverting the affine map corrects every entry of the Gram matrix and every
    bandwidth at once.  Compare Clifford data regression, which needs a training
    set of extra circuits to learn the same kind of affine correction.

Depends on thrifty_budget for the canonical coefficient representation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls

from thrifty_qiskit.thrifty_budget import CanonicalKernel, canonical_design
from thrifty_qiskit.thrifty_exact import _2d


# ************************************************************
# Coefficient layout helpers
# ************************************************************
def coefficient_layout(ck: CanonicalKernel) -> Dict[str, np.ndarray]:
    """
    Index maps for the canonical basis [1 | cos(w.d), sin(w.d) for w != 0].
    Returns the positions of the cosine coefficients (including the constant)
    and of the sine coefficients.
    """
    cos_idx, sin_idx, col = [], [], 0
    for w in ck.W:
        if np.allclose(w, 0):
            cos_idx.append(col); col += 1
        else:
            cos_idx.append(col); sin_idx.append(col + 1); col += 2
    return {"cos": np.array(cos_idx, int), "sin": np.array(sin_idx, int)}


def is_bochner_psd(ck: CanonicalKernel, tol: float = 1e-10) -> bool:
    """Bochner test on the coefficients: even, with non-negative cosine part."""
    lay = coefficient_layout(ck)
    return bool(np.all(np.abs(ck.coef_[lay["sin"]]) <= tol)
                and np.all(ck.coef_[lay["cos"]] >= -tol))


# ************************************************************
# (1) PSD projection by non-negative least squares
# ************************************************************
def psd_project(ck: CanonicalKernel, weights: Optional[np.ndarray] = None,
                inplace: bool = False) -> CanonicalKernel:
    """
    Fit the node values with the sine coefficients removed and the cosine
    coefficients constrained to be non-negative.  The result is a kernel that is
    positive semidefinite on any sample set, at any size, for any bandwidth.

    `weights` optionally weights the nodes (e.g. by 1/variance when the nodes
    were measured with different shot counts), which turns the fit into a
    weighted NNLS.
    """
    lay = coefficient_layout(ck)
    A_cos = ck.A_[:, lay["cos"]]                       # design restricted to cosines
    f = ck.node_values_
    if weights is not None:
        w = np.sqrt(np.asarray(weights, float))
        A_cos, f = A_cos * w[:, None], f * w
    a, _ = nnls(A_cos, f)
    coef = np.zeros_like(ck.coef_)
    coef[lay["cos"]] = a                               # sines set to zero
    out = ck if inplace else _shallow_copy(ck)
    out.coef_ = coef
    return out


def _shallow_copy(ck: CanonicalKernel) -> CanonicalKernel:
    other = CanonicalKernel.__new__(CanonicalKernel)
    other.__dict__.update(ck.__dict__)
    other.n_circuit_evals = 0                          # a derived view costs nothing
    return other


def min_eigenvalue(ck: CanonicalKernel, X: np.ndarray, gamma: float = 1.0) -> float:
    return float(np.linalg.eigvalsh(ck(X, gamma)).min())


def eigenvalue_threshold(K: np.ndarray) -> np.ndarray:
    """Baseline repair acting on the matrix: clip negative eigenvalues.  O(n^3)."""
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    return (V * np.clip(w, 0, None)) @ V.T


# ************************************************************
# (2) Diagonal-anchored depolarising mitigation
# ************************************************************
def depolarizing_lambda(ck: CanonicalKernel, n_qubits: int) -> float:
    """
    Estimate the global depolarising strength from the delta = 0 node, whose
    noiseless value is exactly 1.  Costs no circuits: that node is already part
    of the reconstruction.
    """
    zero = int(np.argmin(np.linalg.norm(ck.nodes, axis=1)))
    k0 = float(ck.node_values_[zero])
    floor = 2.0 ** (-n_qubits)
    return float(np.clip((1.0 - k0) / (1.0 - floor), 0.0, 1.0 - 1e-12))


def mitigate_depolarizing(ck: CanonicalKernel, n_qubits: int,
                          lam: Optional[float] = None,
                          inplace: bool = False) -> Tuple[CanonicalKernel, float]:
    """
    Invert k_meas = (1 - lam) k + lam 2^-q on the coefficients.
    Only the constant coefficient is shifted; all coefficients are rescaled.
    Returns (corrected kernel, lam).
    """
    lam = depolarizing_lambda(ck, n_qubits) if lam is None else float(lam)
    floor = 2.0 ** (-n_qubits)
    lay = coefficient_layout(ck)
    const = int(lay["cos"][np.argmin(np.linalg.norm(ck.W, axis=1))])
    coef = ck.coef_ / (1.0 - lam)
    coef[const] -= lam * floor / (1.0 - lam)
    out = ck if inplace else _shallow_copy(ck)
    out.coef_ = coef
    out.node_values_ = (ck.node_values_ - lam * floor) / (1.0 - lam)
    return out, lam


def diagonal_error(ck: CanonicalKernel, X: np.ndarray, gamma: float = 1.0) -> float:
    """Mean deviation of the Gram diagonal from its exact value 1."""
    return float(np.mean(np.abs(np.diag(ck(X, gamma)) - 1.0)))


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
    n, q = 120, 1
    x = rng.uniform(0, np.pi, (n, 1))

    print("=== (1) positivity under shot noise ===")
    print("  The band limit is rarely known exactly.  Here reps=2 is declared for a")
    print("  kernel that is really reps=1, so two of the five coefficients are truly")
    print("  zero and shot noise pushes them negative -- the kernel then fails")
    print("  Bochner's condition and the Gram matrix is indefinite.")
    print(f"\n{'shots/node':>10s} {'indefinite (raw)':>17s} {'indefinite (PSD)':>17s} "
          f"{'mean min eig raw':>17s} {'rms err raw':>12s} {'rms err PSD':>12s}")
    exact = z_pair(x, x, 1)
    n_draw = 20
    for shots in (64, 256, 1024, 4096):
        bad_raw = bad_psd = 0
        eig_raw, e_raw, e_psd = [], [], []
        for t in range(n_draw):
            ck = CanonicalKernel(d=1, reps=2).fit(lambda A, B: z_pair(A, B, 1),
                                                  shots=[shots] * 5,
                                                  rng=np.random.default_rng(1000 * shots + t))
            pk = psd_project(ck)
            m_raw, m_psd = min_eigenvalue(ck, x), min_eigenvalue(pk, x)
            bad_raw += m_raw < -1e-8
            bad_psd += m_psd < -1e-8
            eig_raw.append(m_raw)
            e_raw.append(np.sqrt(np.mean((ck(x) - exact) ** 2)))
            e_psd.append(np.sqrt(np.mean((pk(x) - exact) ** 2)))
        print(f"{shots:10d} {bad_raw/n_draw:16.0%} {bad_psd/n_draw:17.0%} "
              f"{np.mean(eig_raw):17.3f} {np.mean(e_raw):12.4f} {np.mean(e_psd):12.4f}")

    print("\n  the projection holds for every sample size and bandwidth:")
    ck = CanonicalKernel(d=1, reps=2).fit(lambda A, B: z_pair(A, B, 1),
                                          shots=[256] * 5, rng=np.random.default_rng(1))
    pk = psd_project(ck)
    for m in (20, 100, 400):
        Xm = np.random.default_rng(m).uniform(0, np.pi, (m, 1))
        eigs = [min_eigenvalue(pk, Xm, g) for g in (0.3, 1.0, 3.0)]
        print(f"   n={m:4d}: min eig over gamma in (0.3,1,3) = {min(eigs):.2e}")
    print(f"  Bochner test on coefficients: raw {is_bochner_psd(ck)}, "
          f"projected {is_bochner_psd(pk)}")
    print(f"  baseline (eigenvalue thresholding) on one n=400 matrix repairs that matrix")
    print(f"  only, at O(n^3); the projection above is one NNLS in {ck.n_dof} unknowns and")
    print(f"  covers every sample set and bandwidth at once.")

    print("\n=== (2) corrupted diagonal (global depolarising) ===")
    lam_true = 0.4
    floor = 2.0 ** (-q)

    def noisy_pair(A, B):
        return (1 - lam_true) * z_pair(A, B, 1) + lam_true * floor

    ckn = CanonicalKernel(d=1, reps=1).fit(noisy_pair)
    # note: the depolarised kernel is still band-limited on the same Omega,
    # so the reconstruction stays exact; only its scale and offset change.
    print(f"  true depolarising strength   = {lam_true}")
    print(f"  measured diagonal            = {np.mean(np.diag(ckn(x))):.3f}  "
          f"(exact value is 1)")
    mit, lam_hat = mitigate_depolarizing(ckn, n_qubits=q)
    print(f"  estimated lambda from the delta=0 node = {lam_hat:.4f}  (no extra circuits)")
    print(f"  diagonal after correction    = {np.mean(np.diag(mit(x))):.6f}")
    print(f"  max |K - exact| before {np.max(np.abs(ckn(x) - z_pair(x, x, 1))):.4f}, "
          f"after {np.max(np.abs(mit(x) - z_pair(x, x, 1))):.2e}")

    print("\n  with shot noise on top, and the two corrections composed:")
    ck2 = CanonicalKernel(d=1, reps=1).fit(noisy_pair, shots=[2048] * 3,
                                           rng=np.random.default_rng(3))
    mit2, lam2 = mitigate_depolarizing(ck2, n_qubits=q)
    fin = psd_project(mit2)
    print(f"   lambda_hat = {lam2:.4f}; diagonal {np.mean(np.diag(fin(x))):.4f}; "
          f"min eig {min_eigenvalue(fin, x):.2e}; "
          f"max err {np.max(np.abs(fin(x) - z_pair(x, x, 1))):.4f}")

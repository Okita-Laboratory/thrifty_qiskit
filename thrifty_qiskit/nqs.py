"""
nqs.py -- Classical emulators of quantum states: neural quantum states and MPS.

Instead of running a circuit, represent |psi> classically and optimise:

(1) RBMState  (Carleo & Troyer 2017)
    psi(sigma) = exp(a.sigma) * prod_j 2cosh(b_j + sum_i W_ij sigma_i)
    Ground state by minimising <H> with stochastic reconfiguration (SR).
    Small n (<= 12) uses exact enumeration of all 2^n configurations, so
    the optimisation is deterministic; a Metropolis sampler is included
    for larger n.

(2) DMRG on an MPS
    Two-site DMRG with SVD truncation to bond dimension chi.  Exact when chi
    is large enough for the entanglement; the half-chain entanglement
    entropy is read off the singular values.

Both are exercised on the transverse-field Ising model of shadows
(H = -sum J_i Z_i Z_{i+1} - sum h_i X_i, open chain) so results can be
checked against exact diagonalisation.  No external data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from thrifty_qiskit import shadows as sh  # tfim_hamiltonian, ground_state (exact reference)

# ************************************************************
# (1) RBM neural quantum state with exact enumeration (or Metropolis)
# ************************************************************
def all_configs(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    return 1 - 2 * bits  # sigma in {+1,-1}, bit 0 -> +1, consistent with Z eigenvalues


class RBMState:
    """Real-parameter RBM (adequate for stoquastic H with positive ground state)."""

    def __init__(self, n: int, alpha: int = 2, rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng(0)
        self.n, self.m = n, alpha * n
        self.a = 0.01 * rng.normal(size=n)
        self.b = 0.01 * rng.normal(size=self.m)
        self.W = 0.01 * rng.normal(size=(n, self.m))

    # --- amplitudes and log-derivatives ---
    def log_psi(self, S: np.ndarray) -> np.ndarray:
        theta = self.b + S @ self.W
        return S @ self.a + np.sum(np.logaddexp(theta, -theta), axis=1)  # log 2cosh (up to const)

    def log_derivs(self, S: np.ndarray) -> np.ndarray:
        """O_k(sigma) = d log psi / d param_k, stacked [a | b | W.ravel()]."""
        t = np.tanh(self.b + S @ self.W)
        return np.hstack([S, t, (S[:, :, None] * t[:, None, :]).reshape(len(S), -1)])

    def n_params(self) -> int:
        return self.n + self.m + self.n * self.m

    def set_params(self, p: np.ndarray) -> None:
        n, m = self.n, self.m
        self.a, self.b, self.W = p[:n], p[n:n + m], p[n + m:].reshape(n, m)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.a, self.b, self.W.ravel()])

    # --- local energy of TFIM ---
    def local_energy(self, S: np.ndarray, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        lp = self.log_psi(S)
        diag = -np.sum(J * S[:, :-1] * S[:, 1:], axis=1)
        off = np.zeros(len(S))
        for i in range(self.n):
            Sf = S.copy(); Sf[:, i] *= -1
            off -= h[i] * np.exp(self.log_psi(Sf) - lp)
        return diag + off

    def dense_state(self) -> np.ndarray:
        S = all_configs(self.n)
        psi = np.exp(self.log_psi(S) - self.log_psi(S).max())
        return psi / np.linalg.norm(psi)


def vmc_ground_state(rbm: RBMState, J: np.ndarray, h: np.ndarray, n_iter: int = 300,
                     lr: float = 0.05, eps: float = 1e-3, n_samples: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None, verbose: bool = False) -> np.ndarray:
    """
    Stochastic reconfiguration.  n_samples=None -> exact enumeration of 2^n configs
    (deterministic); otherwise Metropolis sampling with single spin flips.
    Returns the energy history.
    """
    rng = rng or np.random.default_rng(0)
    n = rbm.n
    hist = []
    S_all = all_configs(n) if n_samples is None else None
    for it in range(n_iter):
        if n_samples is None:
            S = S_all
            p = np.exp(2 * (rbm.log_psi(S) - rbm.log_psi(S).max())); p /= p.sum()
        else:
            S = metropolis_sample(rbm, n_samples, rng)
            p = np.full(len(S), 1.0 / len(S))
        E = rbm.local_energy(S, J, h)
        O = rbm.log_derivs(S)
        Em = p @ E
        Om = p @ O
        F = (p * (E - Em)) @ (O - Om)                       # gradient of <H>
        Sm = (O - Om).T @ ((O - Om) * p[:, None])           # SR metric
        Sm[np.diag_indices_from(Sm)] += eps
        dp = np.linalg.solve(Sm, F)
        rbm.set_params(rbm.get_params() - lr * dp)
        hist.append(Em)
        if verbose and it % 50 == 0:
            print(f"  iter {it:4d}  E = {Em:.6f}")
    return np.array(hist)


def metropolis_sample(rbm: RBMState, n_samples: int, rng: np.random.Generator,
                      n_sweeps: int = 10, thermal: int = 50) -> np.ndarray:
    n = rbm.n
    s = rng.choice([-1, 1], n)
    lp = rbm.log_psi(s[None])[0]
    out = []
    for t in range(thermal + n_samples * n_sweeps):
        i = rng.integers(n)
        s2 = s.copy(); s2[i] *= -1
        lp2 = rbm.log_psi(s2[None])[0]
        if np.log(rng.uniform()) < 2 * (lp2 - lp):
            s, lp = s2, lp2
        if t >= thermal and (t - thermal) % n_sweeps == 0:
            out.append(s.copy())
    return np.array(out)


# ************************************************************
# (2) MPS + two-site DMRG
# ************************************************************
_I, _X, _Z = np.eye(2), sh._PAULI[0].astype(float), sh._PAULI[2].astype(float)


def tfim_mpo(J: np.ndarray, h: np.ndarray):
    """Site-dependent MPO, W[s] shape (Dl, Dr, 2, 2), open boundaries."""
    n = len(h)
    Ws = []
    for i in range(n):
        W = np.zeros((3, 3, 2, 2))
        W[0, 0] = _I; W[0, 1] = _Z; W[0, 2] = -h[i] * _X
        if i > 0:
            W[1, 2] = -J[i - 1] * _Z   # bond (i-1, i) is closed at site i
        W[2, 2] = _I
        if i == 0:
            W = W[0:1]
        if i == n - 1:
            W = W[:, 2:3]
        Ws.append(W)
    return Ws


class MPS:
    def __init__(self, n: int, chi: int, rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng(0)
        self.n, self.chi = n, chi
        self.A = []
        for i in range(n):
            dl = min(chi, 2 ** i, 2 ** (n - i))
            dr = min(chi, 2 ** (i + 1), 2 ** (n - i - 1))
            self.A.append(rng.normal(size=(dl, 2, dr)))
        self.S_mid = None  # singular values at middle bond after a sweep

    def dense_state(self) -> np.ndarray:
        psi = self.A[0]
        for A in self.A[1:]:
            psi = np.einsum("...a,asb->...sb", psi, A)
        psi = psi.reshape(-1)
        return psi / np.linalg.norm(psi)

    def entanglement_entropy_mid(self) -> float:
        s = self.S_mid ** 2
        s = s[s > 1e-16]
        return float(-np.sum(s * np.log(s)))


# index convention: L[a,b,c] = (bra bond, mpo bond, ket bond); W[b,e,s,u] = <s|op|u>
def _left_env(L, A, W):
    return np.einsum("abc,asd,bust,ctf->duf", L, A, W, A)


def _right_env(R, A, W):
    return np.einsum("asd,best,ctf,def->abc", A, W, A, R)


def dmrg(mps: MPS, Ws, n_sweeps: int = 6, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """Two-site DMRG; returns (energy, energy history per half-sweep)."""
    n = mps.n
    # right-normalise all sites (QR from the right) and build right environments
    for i in range(n - 1, 0, -1):
        dl, d, dr = mps.A[i].shape
        Q, Rm = np.linalg.qr(mps.A[i].reshape(dl, d * dr).T)
        mps.A[i] = Q.T.reshape(-1, d, dr)
        mps.A[i - 1] = np.einsum("asb,bc->asc", mps.A[i - 1], Rm.T)
    R = [None] * (n + 1)
    R[n] = np.ones((1, 1, 1))
    for i in range(n - 1, 0, -1):
        R[i] = _right_env(R[i + 1], mps.A[i], Ws[i])
    L = [None] * (n + 1)
    L[0] = np.ones((1, 1, 1))
    hist = []
    E = None

    def update(i, going_right):
        nonlocal E
        A1, A2 = mps.A[i], mps.A[i + 1]
        theta = np.einsum("asb,btc->astc", A1, A2)
        Lm, Rm, W1, W2 = L[i], R[i + 2], Ws[i], Ws[i + 1]
        # effective Hamiltonian on (dl, s, t, dr): bra (a,s,t,g), ket (c,u,v,h)
        Heff = np.einsum("abc,besu,eftv,gfh->astgcuvh", Lm, W1, W2, Rm)
        dims = theta.shape
        Hm = Heff.reshape(np.prod(dims), np.prod(dims))
        Hm = 0.5 * (Hm + Hm.T)
        w, v = np.linalg.eigh(Hm)
        E = w[0]
        theta = v[:, 0].reshape(dims)
        dl, s, t, dr = dims
        U, Sv, Vh = np.linalg.svd(theta.reshape(dl * s, t * dr), full_matrices=False)
        k = min(mps.chi, np.sum(Sv > tol))
        U, Sv, Vh = U[:, :k], Sv[:k], Vh[:k]
        Sv /= np.linalg.norm(Sv)
        if i == n // 2 - 1:
            mps.S_mid = Sv.copy()
        if going_right:
            mps.A[i] = U.reshape(dl, s, k)
            mps.A[i + 1] = (np.diag(Sv) @ Vh).reshape(k, t, dr)
            L[i + 1] = _left_env(L[i], mps.A[i], Ws[i])
        else:
            mps.A[i] = (U @ np.diag(Sv)).reshape(dl, s, k)
            mps.A[i + 1] = Vh.reshape(k, t, dr)
            R[i + 1] = _right_env(R[i + 2], mps.A[i + 1], Ws[i + 1])

    for sweep in range(n_sweeps):
        for i in range(n - 1):
            update(i, True)
        hist.append(E)
        for i in range(n - 2, -1, -1):
            update(i, False)
        hist.append(E)
        if sweep > 0 and abs(hist[-1] - hist[-3]) < 1e-12:
            break
    return float(E), np.array(hist)


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    n = 8
    J = np.ones(n - 1)
    print(f"TFIM chain n={n}: ground-state energy vs transverse field h (J=1)")
    print("  h    exact       RBM(alpha=2)  err       DMRG chi=2  err       DMRG chi=8  err       S_mid(chi=8)")
    for hv in (0.3, 0.7, 1.0, 1.5, 3.0):
        h = hv * np.ones(n)
        H = sh.tfim_hamiltonian(J, h)
        E0 = np.linalg.eigvalsh(H)[0]
        rbm = RBMState(n, alpha=2, rng=np.random.default_rng(1))
        E_rbm = vmc_ground_state(rbm, J, h, n_iter=300)[-1]
        Ws = tfim_mpo(J, h)
        E2, _ = dmrg(MPS(n, chi=2), Ws)
        m8 = MPS(n, chi=8); E8, _ = dmrg(m8, Ws)
        print(f"  {hv:.1f}  {E0:+.6f}  {E_rbm:+.6f}  {E_rbm-E0:.1e}   {E2:+.6f}  {E2-E0:.1e}   "
              f"{E8:+.6f}  {E8-E0:.1e}   {m8.entanglement_entropy_mid():.3f}")

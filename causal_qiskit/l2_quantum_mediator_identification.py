"""
l2_quantum_mediator_identification.py

Open problem 1 (theory document, Sec. 8): in the quantum causal model

        Lambda --> X --> Z --> Y
        Lambda ------------> Y          (Lambda unobserved quantum common cause)

with X a control node (measure Lambda_X, record x, prepare tau_x) and Z a
QUANTUM mediator (a fixed channel, its output state flows to Y without being
measured), Proposition 4.8 says P(y | do(X = tau)) is not identifiable from
observational data alone.  This file makes that statement quantitative and
finds which additional assumptions recover identification.

Method: partial identification by semidefinite programming.
  Unknowns  : the effective POVM  E~_y  on  Z_out (x) Lambda_Y  (E~_y = N_Y^dag(E_y))
              and the conditional states rho^(x') of Lambda_Y given x.
  Known     : the channel X_out -> Z_out (assumed characterised), the preparations
              tau_x, and the observational distribution P(x', y).
  Target    : P(y | do tau) = Tr[ E~_y (sigma_tau (x) rho_LambdaY) ],
              rho_LambdaY = sum_x' P(x') rho^(x')   (Lemma 4.5).
For fixed rho^(x') the constraints and target are LINEAR in E~, so
max/min over E~ is an SDP.  Outer bounds are obtained by sampling rho^(x')
(the bilinear part).  If the bounds collapse to a point, the target is identified.

Scenarios compared:
  S0  observational data only                          -> wide bounds (not identified)
  S1  + X randomises its preparation independently     -> collapses (but this IS an intervention)
  S2  + Lambda_Y is classical (dephased)               -> partial collapse
  S3  + N_Y known (E~ known up to E_y)                 -> collapses
  S4  Z is a control node (measure-and-prepare)        -> classical front-door exact (Prop. 4.6)

All models are qubits: Lambda = 2 qubits (Lambda_X, Lambda_Y), X_out/Z_out = 1 qubit.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import warnings

import cvxpy as cp
import numpy as np

warnings.filterwarnings("ignore")

I2 = np.eye(2)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]], complex),
    "Z": np.array([[1, 0], [0, -1]], complex),
}


# ************************************************************
# Ground-truth model (used only to GENERATE data; the identification
# procedure never sees rho_Lambda, N_Y or E_y directly)
# ************************************************************
def random_state(d: int, rng: np.random.Generator, rank: Optional[int] = None) -> np.ndarray:
    rank = rank or d
    G = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(G)
    return Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))


def random_povm(d: int, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    """k-outcome POVM on C^d: random positive operators, symmetrically normalised."""
    ops = [random_state(d, rng) * rng.uniform(0.2, 1.0) for _ in range(k)]
    S = sum(ops)
    Sinv = np.linalg.inv(_sqrtm_psd(S))
    return [Sinv @ o @ Sinv for o in ops]


def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def channel_from_unitary(U: np.ndarray, env_dim: int):
    """Channel rho -> Tr_env[U (rho (x) |0><0|_env) U^dag]  (Stinespring with fresh env)."""
    d = U.shape[0] // env_dim

    def N(rho):
        full = np.kron(rho, np.diag([1.0] + [0.0] * (env_dim - 1)))
        out = U @ full @ U.conj().T
        return _ptrace(out, dims=(d, env_dim), keep=0)
    return N


def _ptrace(rho: np.ndarray, dims: Tuple[int, int], keep: int) -> np.ndarray:
    dA, dB = dims
    r = rho.reshape(dA, dB, dA, dB)
    return np.einsum("ijkj->ik", r) if keep == 0 else np.einsum("ijil->jl", r)


class GroundTruth:
    def __init__(self, seed: int = 0, n_x: int = 2, n_y: int = 2, lam_classical: bool = False):
        rng = np.random.default_rng(seed)
        self.rng = rng
        # Lambda: 2-qubit state on Lambda_X (x) Lambda_Y, entangled unless lam_classical
        if lam_classical:
            p = rng.dirichlet(np.ones(4))
            self.rho_L = np.diag(p).astype(complex)
        else:
            self.rho_L = random_state(4, rng, rank=2)
        # X: POVM on Lambda_X, preparations tau_x on X_out
        self.F = random_povm(2, n_x, rng)
        self.tau = [random_state(2, rng, rank=1) for _ in range(n_x)]
        # Z: channel X_out -> Z_out (unitary + partial dephasing so it's not invertible)
        self.U_Z = random_unitary(2, rng)
        self.p_deph = 0.3
        # Y: unitary N_Y on Z_out (x) Lambda_Y then POVM E_y on the 2 qubits
        self.U_Y = random_unitary(4, rng)
        self.E = random_povm(4, n_y, rng)
        self.n_x, self.n_y = n_x, n_y

    # ---- known-to-analyst part: Z channel ----
    def N_Z(self, rho: np.ndarray) -> np.ndarray:
        r = self.U_Z @ rho @ self.U_Z.conj().T
        return (1 - self.p_deph) * r + self.p_deph * np.diag(np.diag(r))

    # ---- effective POVM on Z_out (x) Lambda_Y (unknown to analyst) ----
    def E_eff(self) -> List[np.ndarray]:
        return [self.U_Y.conj().T @ E @ self.U_Y for E in self.E]

    # ---- conditional states of Lambda_Y ----
    def cond_states(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        P = np.empty(self.n_x)
        rhos = []
        for x in range(self.n_x):
            unnorm = _ptrace(np.kron(self.F[x], I2) @ self.rho_L, (2, 2), keep=1)
            P[x] = np.trace(unnorm).real
            rhos.append(unnorm / P[x])
        return P, rhos

    # ---- data ----
    def observational(self) -> Tuple[np.ndarray, np.ndarray]:
        """P(x'), P(y|x') with Z acting as a quantum mediator."""
        P, rhos = self.cond_states()
        Pyx = np.empty((self.n_x, self.n_y))
        for x in range(self.n_x):
            sig = self.N_Z(self.tau[x])
            for y, Ee in enumerate(self.E_eff()):
                Pyx[x, y] = np.trace(Ee @ np.kron(sig, rhos[x])).real
        return P, Pyx

    def interventional(self, tau: np.ndarray) -> np.ndarray:
        P, rhos = self.cond_states()
        rho_LY = sum(P[x] * rhos[x] for x in range(self.n_x))
        sig = self.N_Z(tau)
        return np.array([np.trace(Ee @ np.kron(sig, rho_LY)).real for Ee in self.E_eff()])

    def randomised_prep_data(self, preps: List[np.ndarray]) -> np.ndarray:
        """P(y | x', r): X records x' but prepares preps[r] chosen independently (an intervention)."""
        P, rhos = self.cond_states()
        out = np.empty((self.n_x, len(preps), self.n_y))
        for x in range(self.n_x):
            for r, tau in enumerate(preps):
                sig = self.N_Z(tau)
                for y, Ee in enumerate(self.E_eff()):
                    out[x, r, y] = np.trace(Ee @ np.kron(sig, rhos[x])).real
        return out


# ************************************************************
# Partial identification by SDP
# ************************************************************
def _herm(n):
    return cp.Variable((n, n), hermitian=True)


def sdp_bounds(sigma_obs: List[np.ndarray], Pyx: np.ndarray, P: np.ndarray,
               rhos: List[np.ndarray], sigma_tau: np.ndarray, y_target: int,
               extra: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
               E_fixed: Optional[List[np.ndarray]] = None) -> Tuple[float, float]:
    """
    For FIXED conditional states rhos (list over x'), minimise / maximise
        Tr[ E~_y (sigma_tau (x) rho_LY) ]
    over POVMs {E~_y} on C^4 consistent with  Tr[E~_y (sigma_obs[x'] (x) rhos[x'])] = Pyx[x', y].
    extra: list of (sigma_r, rho_x, p_vec) giving additional linear constraints
           Tr[E~_y (sigma_r (x) rho_x)] = p_vec[y]   (randomised preparations).
    E_fixed: if given, E~ is fixed and only rhos vary (used for S3).
    """
    n_x, n_y = Pyx.shape
    rho_LY = sum(P[x] * rhos[x] for x in range(n_x))
    if E_fixed is not None:
        v = np.trace(E_fixed[y_target] @ np.kron(sigma_tau, rho_LY)).real
        return v, v
    E = [_herm(4) for _ in range(n_y)]
    cons = [Ey >> 0 for Ey in E] + [sum(E) == np.eye(4)]
    for x in range(n_x):
        A = np.kron(sigma_obs[x], rhos[x])
        for y in range(n_y):
            cons.append(cp.real(cp.trace(E[y] @ A)) == Pyx[x, y])
    for (sig_r, rho_x, pvec) in (extra or []):
        A = np.kron(sig_r, rho_x)
        for y in range(n_y):
            cons.append(cp.real(cp.trace(E[y] @ A)) == pvec[y])
    obj = cp.real(cp.trace(E[y_target] @ np.kron(sigma_tau, rho_LY)))
    lo = cp.Problem(cp.Minimize(obj), cons).solve(solver="CLARABEL")
    E_lo = [np.array(Ey.value) for Ey in E]
    hi = cp.Problem(cp.Maximize(obj), cons).solve(solver="CLARABEL")
    E_hi = [np.array(Ey.value) for Ey in E]
    sdp_bounds.last_extremal = (E_lo, E_hi)
    return float(lo), float(hi)


def outer_bounds(gt: GroundTruth, tau: np.ndarray, y_target: int, n_rho_samples: int,
                 scenario: str, rng: np.random.Generator,
                 preps: Optional[List[np.ndarray]] = None) -> Tuple[float, float, float]:
    """
    Sample conditional states rhos (respecting only what the analyst knows), take
    the union of SDP intervals.  Returns (lower, upper, truth).
    scenario: "S0" obs only; "S1" + randomised preps; "S2" Lambda_Y classical; "S3" N_Y known.
    """
    P, Pyx = gt.observational()
    sigma_obs = [gt.N_Z(t) for t in gt.tau]
    sigma_tau = gt.N_Z(tau)
    truth = gt.interventional(tau)[y_target]
    lo, hi = np.inf, -np.inf
    P_true, rhos_true = gt.cond_states()
    feasible = 0
    for s in range(n_rho_samples):
        if s == 0:
            rhos = rhos_true                      # include the truth so the interval is never empty
        elif scenario == "S2":
            rhos = [np.diag(rng.dirichlet(np.ones(2))).astype(complex) for _ in range(gt.n_x)]
        else:
            rhos = [random_state(2, rng) for _ in range(gt.n_x)]
        extra = None
        if scenario == "S1":
            data = gt.randomised_prep_data(preps)
            extra = [(gt.N_Z(preps[r]), rhos[x], data[x, r]) for x in range(gt.n_x) for r in range(len(preps))]
        E_fixed = gt.E_eff() if scenario == "S3" else None
        try:
            l, h = sdp_bounds(sigma_obs, Pyx, P, rhos, sigma_tau, y_target, extra=extra, E_fixed=E_fixed)
        except Exception:
            continue                              # infeasible sample: this rhos is inconsistent with data
        if not (np.isfinite(l) and np.isfinite(h)):
            continue
        feasible += 1
        lo, hi = min(lo, l), max(hi, h)
    return lo, hi, truth


# ************************************************************
# S4: Z as a control node -> classical front-door (Prop. 4.6), numerical check
# ************************************************************
def front_door_check(gt: GroundTruth, n_z: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    rng = gt.rng
    G = random_povm(2, n_z, rng)                  # Z measures its input
    tauZ = [random_state(2, rng, rank=1) for _ in range(n_z)]  # and prepares tau'_z
    P, rhos = gt.cond_states()
    rho_LY = sum(P[x] * rhos[x] for x in range(gt.n_x))
    Eeff = gt.E_eff()
    # observational P(z|x), P(y|z,x')
    Pzx = np.array([[np.trace(G[z] @ gt.N_Z(gt.tau[x])).real for z in range(n_z)] for x in range(gt.n_x)])
    Pyzx = np.array([[[np.trace(Eeff[y] @ np.kron(tauZ[z], rhos[x])).real for y in range(gt.n_y)]
                      for z in range(n_z)] for x in range(gt.n_x)])
    # true interventional for tau = tau_x
    true = np.array([[sum(np.trace(G[z] @ gt.N_Z(gt.tau[x])).real *
                          np.trace(Eeff[y] @ np.kron(tauZ[z], rho_LY)).real for z in range(n_z))
                      for y in range(gt.n_y)] for x in range(gt.n_x)])
    # front-door estimate
    est = np.array([[sum(Pzx[x, z] * sum(P[xp] * Pyzx[xp, z, y] for xp in range(gt.n_x)) for z in range(n_z))
                     for y in range(gt.n_y)] for x in range(gt.n_x)])
    return true, est


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    gt = GroundTruth(seed=3)
    tau = random_state(2, rng, rank=1)             # an intervention outside the preparation set
    preps = [random_state(2, rng, rank=1) for _ in range(4)]  # tomographically complete-ish set
    y = 0
    print("Quantum mediator Z, entangled Lambda.  Target: P(y=0 | do(X = tau))")
    gt_cl = GroundTruth(seed=3, lam_classical=True)
    for sc, name, g in [("S0", "observational only", gt),
                        ("S1", "+ randomised preparation at X (intervention)", gt),
                        ("S2", "Lambda classical (dephased), obs only", gt_cl),
                        ("S3", "+ N_Y known (E~ fixed, rho^(x') free)", gt)]:
        lo, hi, truth = outer_bounds(g, tau, y, n_rho_samples=40, scenario=sc, rng=rng, preps=preps)
        print(f"  {sc} {name:48s} bounds [{lo:.4f}, {hi:.4f}]  width {hi - lo:.4f}  truth {truth:.4f}")

    # constructive counterexample: two models with identical observational data
    P, Pyx = gt.observational()
    _, rhos_true = gt.cond_states()
    sig_obs = [gt.N_Z(t) for t in gt.tau]
    lo, hi = sdp_bounds(sig_obs, Pyx, P, rhos_true, gt.N_Z(tau), y)
    E_lo, E_hi = sdp_bounds.last_extremal
    dev = max(abs(np.trace(E[yy] @ np.kron(sig_obs[x], rhos_true[x])).real - Pyx[x, yy])
              for E in (E_lo, E_hi) for x in range(gt.n_x) for yy in range(gt.n_y))
    print(f"\nTwo explicit models (same rho^(x')): observational mismatch {dev:.1e}, "
          f"interventional P(y=0|do tau) = {lo:.4f} vs {hi:.4f}")

    print("\nS4  Z as control node (measure-and-prepare): classical front-door formula")
    true, est = front_door_check(gt)
    print(f"  max |front-door - truth| over x, y = {np.max(np.abs(true - est)):.2e}")

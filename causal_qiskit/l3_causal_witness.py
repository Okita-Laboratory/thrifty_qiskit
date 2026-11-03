"""
l3_causal_witness.py -- Open problem 4: a testable witness of causal
non-separability (indefinite causal order).

Objects
-------
Parties A and B, each with input and output qubit (A_I, A_O, B_I, B_O), a
final target system F and a control qubit C, both measured at the end.  The
target's initial state is fixed to |0>, so a process is a positive operator W
on A_I A_O B_I B_O F C with Tr W = d_{A_O} d_{B_O} = 4.
Qubit order: [A_I, A_O, B_I, B_O, F, C]  (6 qubits, 64-dim).

Definite order A -> B -> (F,C) means W is a quantum comb:
    Tr_{FC} W    = 1_{B_O} (x) W2,      Tr_{B_I} W2 = 1_{A_O} (x) W1,      Tr W1 = 1.
With the notation  _X W := Tr_X W (x) 1_X / d_X  (a projector), the first two
conditions are the linear constraints
    M1(W) := _{FC}W - _{B_O FC}W = 0,     M2(W) := _{B_I B_O FC}W - _{A_O B_I B_O FC}W = 0,
M1, M2 are commuting orthogonal projectors with M1 M2 = 0, so the projector
onto the comb subspace is  Pi_AB = 1 - M1 - M2  (Araujo et al. 2015).
W is CAUSALLY SEPARABLE if W = W_AB + W_BA with W_AB, W_BA >= 0 in the
respective comb subspaces (sub-normalised).

Witness SDP (dual characterisation)
    S is a causal witness  iff  there exist S_AB, S_BA >= 0 with
        Pi_AB(S) = Pi_AB(S_AB),   Pi_BA(S) = Pi_BA(S_BA).
    min Tr[S W]  s.t.  S witness,  Tr[S Omega] = 1     (Omega = white noise)
    equals  -r*, the random causal robustness of W:
    r* = min r s.t. (W + r Omega) is causally separable.
We solve both and check duality numerically.

Translation into an experiment
------------------------------
Tr[S W] is expanded in Pauli strings on the six qubits: Paulis on A_I, B_I
are measurements, on A_O, B_O preparations, on F, C final measurements.  So
Tr[S W] is a fixed linear combination of P(outcomes | settings) over a finite
list of local settings.  This is the form in which it can be estimated on any
platform where two operations A, B (two drive pulses on a spin, say) are
applied in an order controlled by a coherent degree of freedom C.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]], complex),
    "Z": np.array([[1, 0], [0, -1]], complex),
}
Q = {"AI": 0, "AO": 1, "BI": 2, "BO": 3, "F": 4, "C": 5}
NQ = 6
D = 2 ** NQ


# ************************************************************
# Process matrices
# ************************************************************
def quantum_switch(control: np.ndarray = None) -> np.ndarray:
    """W = |w><w| for the switch with target input |0>; Tr W = 4."""
    control = np.array([1, 1]) / np.sqrt(2) if control is None else np.asarray(control, complex)
    w = np.zeros((2,) * NQ, complex)
    for a in range(2):
        for b in range(2):
            # order 0: |0> -> A_I ; A_O -> B_I ; B_O -> F ; C=0
            w[0, a, a, b, b, 0] += control[0]
            # order 1: |0> -> B_I ; B_O -> A_I ; A_O -> F ; C=1
            w[b, a, 0, b, a, 1] += control[1]
    w = w.reshape(D)
    W = np.outer(w, w.conj())
    return W * (4 / np.trace(W).real)


def dephase_control(W: np.ndarray) -> np.ndarray:
    t = W.reshape([2] * (2 * NQ))
    out = np.zeros_like(t)
    for c in range(2):
        idx = [slice(None)] * (2 * NQ)
        idx[Q["C"]] = c; idx[NQ + Q["C"]] = c
        out[tuple(idx)] = t[tuple(idx)]
    return out.reshape(D, D)


def white_noise() -> np.ndarray:
    return np.eye(D) / 16.0                     # Tr = 4


def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Qm, R = np.linalg.qr(G)
    return Qm @ np.diag(np.diag(R) / np.abs(np.diag(R)))


def local_unitary_on(W: np.ndarray, U: np.ndarray, qubit: int) -> np.ndarray:
    ops = [np.eye(2)] * NQ; ops[qubit] = U
    V = ops[0]
    for o in ops[1:]:
        V = np.kron(V, o)
    return V @ W @ V.conj().T


def random_separable_process(rng: np.random.Generator) -> np.ndarray:
    """
    Convex mixture of an order-AB comb and an order-BA comb, each obtained from the
    corresponding branch of the switch by random unitaries on the wires (A_I, B_I, F).
    Separable by construction.
    """
    W0 = quantum_switch(np.array([1, 0]))
    W1 = quantum_switch(np.array([0, 1]))
    for q in ("AI", "BI", "F"):
        W0 = local_unitary_on(W0, random_unitary(2, rng), Q[q])
        W1 = local_unitary_on(W1, random_unitary(2, rng), Q[q])
    p = rng.uniform()
    W = p * W0 + (1 - p) * W1
    return W * (4 / np.trace(W).real)


# ************************************************************
# Trace-and-replace maps and comb projectors (numpy and cvxpy)
# ************************************************************
def _perm_matrix(order: List[int], n: int) -> np.ndarray:
    d = 2 ** n
    P = np.zeros((d, d))
    for idx in range(d):
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        g = [0] * n
        for pos, q in enumerate(order):
            g[q] = bits[pos]
        P[int("".join(map(str, g)), 2), idx] = 1
    return P


def _ptrace_np(A: np.ndarray, dims: List[int], axis: int) -> np.ndarray:
    n = len(dims)
    t = A.reshape(dims + dims)
    m = int(np.prod(dims)) // dims[axis]
    return np.trace(t, axis1=axis, axis2=n + axis).reshape(m, m)


def _ptrace_any(expr, dims: List[int], traced: Sequence[int]):
    e, d, qubits = expr, list(dims), list(range(len(dims)))
    for q in sorted(traced, reverse=True):
        pos = qubits.index(q)
        e = _ptrace_np(e, d, pos) if isinstance(e, np.ndarray) else cp.partial_trace(e, d, axis=pos)
        d.pop(pos); qubits.pop(pos)
    return e


def trace_replace(expr, traced: Sequence[int]):
    """_X(expr) = Tr_X expr (x) 1_X / d_X in global qubit order (numpy or cvxpy)."""
    traced = sorted(traced)
    kept = [q for q in range(NQ) if q not in traced]
    T = _ptrace_any(expr, [2] * NQ, traced)
    k = len(traced)
    Ik = np.eye(2 ** k) / 2 ** k
    big = np.kron(Ik, T) if isinstance(T, np.ndarray) else cp.kron(Ik, T)
    P = _perm_matrix(traced + kept, NQ)
    return P @ big @ P.T


def comb_projector(expr, order: str):
    """Pi_order(expr) = expr - M1(expr) - M2(expr)."""
    if order == "AB":
        so, si, fo = Q["BO"], Q["BI"], Q["AO"]
    else:
        so, si, fo = Q["AO"], Q["AI"], Q["BO"]
    FC = [Q["F"], Q["C"]]
    M1 = trace_replace(expr, FC) - trace_replace(expr, FC + [so])
    M2 = trace_replace(expr, FC + [so, si]) - trace_replace(expr, FC + [so, si, fo])
    return expr - M1 - M2


# ---- sparse matrix form of the projectors acting on vec(W) (column-major), for the SDPs ----
def _bits(idx: int) -> List[int]:
    return [(idx >> (NQ - 1 - i)) & 1 for i in range(NQ)]


def _idx(bits: List[int]) -> int:
    return int("".join(map(str, bits)), 2)


def trace_replace_matrix(traced: Sequence[int]) -> sp.csr_matrix:
    """Sparse (D^2 x D^2) matrix L with  vec(_X W) = L vec(W),  vec column-major."""
    traced = sorted(traced)
    k = len(traced)
    rows, cols, vals = [], [], []
    for i in range(D):
        bi = _bits(i)
        for j in range(D):
            bj = _bits(j)
            if any(bi[q] != bj[q] for q in traced):
                continue                                   # output is 1 on traced part
            out = i + D * j                                # column-major position of (i, j)
            for x in range(2 ** k):
                xb = [(x >> (k - 1 - t)) & 1 for t in range(k)]
                bi2, bj2 = list(bi), list(bj)
                for t, q in enumerate(traced):
                    bi2[q] = xb[t]; bj2[q] = xb[t]
                rows.append(out); cols.append(_idx(bi2) + D * _idx(bj2)); vals.append(1.0 / 2 ** k)
    return sp.csr_matrix((vals, (rows, cols)), shape=(D * D, D * D))


_LMAP_CACHE: Dict[Tuple[int, ...], sp.csr_matrix] = {}


def _L(traced: Sequence[int]) -> sp.csr_matrix:
    key = tuple(sorted(traced))
    if key not in _LMAP_CACHE:
        _LMAP_CACHE[key] = trace_replace_matrix(key)
    return _LMAP_CACHE[key]


def comb_projector_matrix(order: str) -> sp.csr_matrix:
    if order == "AB":
        so, si, fo = Q["BO"], Q["BI"], Q["AO"]
    else:
        so, si, fo = Q["AO"], Q["AI"], Q["BO"]
    FC = [Q["F"], Q["C"]]
    M1 = _L(FC) - _L(FC + [so])
    M2 = _L(FC + [so, si]) - _L(FC + [so, si, fo])
    return sp.identity(D * D, format="csr") - M1 - M2


def is_comb(W: np.ndarray, order: str, tol: float = 1e-8) -> bool:
    return bool(np.linalg.eigvalsh(W).min() > -tol and np.max(np.abs(comb_projector(W, order) - W)) < tol)


# ************************************************************
# SDPs
# ************************************************************
def causal_robustness(W: np.ndarray, noise: np.ndarray = None) -> float:
    noise = white_noise() if noise is None else noise
    WAB = cp.Variable((D, D), hermitian=True)
    WBA = cp.Variable((D, D), hermitian=True)
    r = cp.Variable(nonneg=True)
    PAB, PBA = comb_projector_matrix("AB"), comb_projector_matrix("BA")
    cons = [WAB >> 0, WBA >> 0,
            PAB @ cp.vec(WAB, order="F") == cp.vec(WAB, order="F"),
            PBA @ cp.vec(WBA, order="F") == cp.vec(WBA, order="F"),
            WAB + WBA == W + r * noise]
    cp.Problem(cp.Minimize(r), cons).solve(solver="SCS", eps=1e-7, max_iters=50000)
    return float(r.value)


def causal_witness(W: np.ndarray, noise: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """min Tr[S W] over witnesses with Tr[S noise] = 1.  Returns (value, S)."""
    noise = white_noise() if noise is None else noise
    S = cp.Variable((D, D), hermitian=True)
    SAB = cp.Variable((D, D), hermitian=True)
    SBA = cp.Variable((D, D), hermitian=True)
    PAB, PBA = comb_projector_matrix("AB"), comb_projector_matrix("BA")
    cons = [SAB >> 0, SBA >> 0,
            PAB @ cp.vec(S, order="F") == PAB @ cp.vec(SAB, order="F"),
            PBA @ cp.vec(S, order="F") == PBA @ cp.vec(SBA, order="F"),
            cp.real(cp.trace(S @ noise)) == 1]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(S @ W))), cons)
    prob.solve(solver="SCS", eps=1e-7, max_iters=50000)
    Sv = np.array(S.value); Sv = 0.5 * (Sv + Sv.conj().T)
    return float(prob.value), Sv


def witness_value(S: np.ndarray, W: np.ndarray) -> float:
    return float(np.trace(S @ W).real)


# ************************************************************
# Translation into local settings
# ************************************************************
def pauli_expansion(S: np.ndarray, tol: float = 1e-6) -> Dict[str, float]:
    out = {}
    for s in itertools.product("IXYZ", repeat=NQ):
        P = PAULIS[s[0]]
        for ch in s[1:]:
            P = np.kron(P, PAULIS[ch])
        c = np.trace(P @ S).real / D
        if abs(c) > tol:
            out["".join(s)] = c
    return out


def minimal_settings(coeffs: Dict[str, float]) -> List[str]:
    """Greedy cover of the Pauli terms by full settings in {X,Y,Z}^6 ('I' is compatible with any)."""
    remaining = set(coeffs)
    settings = []
    while remaining:
        best, best_cov = None, set()
        for full in itertools.product("XYZ", repeat=NQ):
            cov = {s for s in remaining if all(a == "I" or a == b for a, b in zip(s, full))}
            if len(cov) > len(best_cov):
                best, best_cov = "".join(full), cov
        settings.append(best)
        remaining -= best_cov
    return settings


def estimate_witness_from_shots(coeffs: Dict[str, float], W: np.ndarray, shots: int,
                                rng: np.random.Generator) -> float:
    """Each Pauli term's expectation Tr[P W]/4 in [-1,1] is replaced by a `shots`-sample estimate."""
    val = 0.0
    for s, c in coeffs.items():
        P = PAULIS[s[0]]
        for ch in s[1:]:
            P = np.kron(P, PAULIS[ch])
        e = np.clip(np.trace(P @ W).real / 4, -1, 1)
        k = rng.binomial(shots, (1 + e) / 2)
        val += c * (2 * k / shots - 1) * 4
    return val


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W_sw = quantum_switch()
    W_cl = dephase_control(W_sw)
    print("switch: Tr W =", round(np.trace(W_sw).real, 6), " rank", np.linalg.matrix_rank(W_sw),
          " | order-0 branch is an AB comb:", is_comb(quantum_switch([1, 0]), "AB"),
          " BA comb:", is_comb(quantum_switch([0, 1]), "BA"))

    r_sw = causal_robustness(W_sw)
    val, S = causal_witness(W_sw)
    print(f"random causal robustness r* = {r_sw:.4f};  witness optimum min Tr[S W] = {val:.4f}  (duality: -r*)")
    r_cl = causal_robustness(W_cl)
    print(f"dephased switch (classical mixture of orders): r* = {r_cl:.1e}")

    worst = min(witness_value(S, random_separable_process(rng)) for _ in range(30))
    print(f"min Tr[S W_sep] over 30 random separable processes = {worst:+.2e}  (must be >= 0)")
    print(f"Tr[S W_switch] = {witness_value(S, W_sw):+.4f}, Tr[S W_dephased] = {witness_value(S, W_cl):+.4f}, "
          f"Tr[S noise] = {witness_value(S, white_noise()):.4f}")

    print("\nnoise threshold: W_p = p W_switch + (1-p) noise;  detection iff Tr[S W_p] < 0")
    p_star = 1 / (1 + r_sw)
    for p in (1.0, 0.8, 0.6, 0.4, round(p_star, 3), 0.3):
        Wp = p * W_sw + (1 - p) * white_noise()
        print(f"  p={p:.3f}: Tr[S W_p] = {witness_value(S, Wp):+.4f}")
    print(f"  threshold p* = 1/(1+r*) = {p_star:.4f}")

    coeffs = pauli_expansion(S)
    sets = minimal_settings(coeffs)
    print(f"\nwitness: {len(coeffs)} Pauli terms, covered by {len(sets)} local settings "
          f"(A_I meas, A_O prep, B_I meas, B_O prep, F meas, C meas)")
    print("  largest terms:", [(s, round(c, 4)) for s, c in sorted(coeffs.items(), key=lambda kv: -abs(kv[1]))[:6]])

    print("\nfinite-shot estimation of Tr[S W_p] at p = 0.6 (per Pauli term):")
    Wp = 0.6 * W_sw + 0.4 * white_noise()
    for shots in (100, 1000, 10000):
        ests = np.array([estimate_witness_from_shots(coeffs, Wp, shots, rng) for _ in range(50)])
        print(f"  shots={shots:6d}: mean {ests.mean():+.4f}, std {ests.std():.4f}, P(detect) = {(ests < 0).mean():.2f}")

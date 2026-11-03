"""
l2_orientation_noncommuting.py -- Open problem 3: orienting edges with the
non-commutativity of observation and intervention.

Setting (process-operator picture, same as l2_qpc_cmi):
  node i : input qubit in_i  <- channel from parents' output qubits (unitary on
           parents' outs + fresh in_i),
           local operation L_i : unitary on in_i (x) fresh output qubits (one per child).
  "Observing" node i means measuring in_i in some basis B_i BEFORE L_i acts.

Classically, observing a variable never changes its descendants.  Quantum
mechanically the measurement of in_i in basis B_i is a non-commuting
operation: it dephases in_i in B_i, and that change propagates through L_i to
every descendant.  Ancestors are untouched (no-signalling).  Hence:

  Orientation rule (observation-only):
     D(i -> j) := max_{B_i, B_i'} || P(b_j | B_i) - P(b_j | B_i') ||_1  (maximised over B_j too)
     D(i -> j) > 0   iff   i is an ancestor of j        (generically)
     D(j -> i) = 0   whenever j is not an ancestor of i.
  Combined with the skeleton (l2_qpc_cmi), this orients every edge, using
  only randomised measurement bases -- no state preparation (do) is needed.

This file simulates the model with density matrices (<= 10 qubits), computes
D exactly, checks the rule on random DAGs, orients edges, and then repeats
with finite shots to measure the orientation error rate versus shots.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from thrifty_qiskit.gram_resampling import EvalCache  # the one place that touches the simulator

PAULI = {
    0: np.array([[0, 1], [1, 0]], complex),
    1: np.array([[0, -1j], [1j, 0]], complex),
    2: np.array([[1, 0], [0, -1]], complex),
}
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_SDG = np.diag([1, -1j])
_U = {0: _H, 1: _H @ _SDG, 2: np.eye(2)}   # basis -> computational


# ************************************************************
# Density-matrix utilities
# ************************************************************
def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(G)
    return Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))


def apply_unitary(rho: np.ndarray, U: np.ndarray, qubits: Sequence[int], n: int) -> np.ndarray:
    """rho -> (U on `qubits`) rho (U^dag), n-qubit density matrix."""
    k = len(qubits)
    t = rho.reshape([2] * (2 * n))
    Ut = U.reshape([2] * (2 * k))
    # left multiply
    t = np.tensordot(Ut, t, axes=(list(range(k, 2 * k)), list(qubits)))
    # tensordot puts the k new axes first; move them back to positions `qubits`
    t = np.moveaxis(t, list(range(k)), list(qubits))
    # right multiply by U^dag: act on column indices n+q
    Ud = U.conj().T.reshape([2] * (2 * k))
    cols = [n + q for q in qubits]
    t = np.tensordot(t, Ud, axes=(cols, list(range(k))))
    t = np.moveaxis(t, list(range(2 * n - k, 2 * n)), cols)
    return t.reshape(2 ** n, 2 ** n)


def dephase(rho: np.ndarray, basis: int, qubit: int, n: int) -> np.ndarray:
    """Measure `qubit` in Pauli basis `basis` and discard the outcome (Lueders dephasing)."""
    U = _U[basis]
    r = apply_unitary(rho, U, [qubit], n)
    P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    r = apply_unitary(r, P0, [qubit], n) + apply_unitary(r, P1, [qubit], n)
    return apply_unitary(r, U.conj().T, [qubit], n)


def marginal_probs(rho: np.ndarray, basis: int, qubit: int, n: int) -> np.ndarray:
    r = apply_unitary(rho, _U[basis], [qubit], n)
    t = r.reshape([2] * (2 * n))
    # trace out everything except `qubit`
    idx = list(range(n))
    keep = qubit
    red = np.einsum(t, idx + [q if q != keep else 2 * n for q in range(n)], [keep, 2 * n])
    p = np.real(np.diag(red))
    return np.clip(p, 0, None) / p.sum()


# ************************************************************
# Model
# ************************************************************
class SequentialQCM:
    def __init__(self, n: int, edges: List[Tuple[int, int]], rng: np.random.Generator):
        self.n, self.edges, self.rng = n, sorted(edges), rng
        self.parents = {i: [j for (j, k) in edges if k == i] for i in range(n)}
        self.children = {i: [k for (j, k) in edges if j == i] for i in range(n)}
        q = 0
        self.in_q, self.out_q = {}, {}
        for i in range(n):
            self.in_q[i] = q; q += 1
            for k in self.children[i]:
                self.out_q[(i, k)] = q; q += 1
        self.nq = q
        # channel unitaries (parents' outs + in_i) and local unitaries (in_i + outs of i)
        self.U_in = {i: random_unitary(2 ** (len(self.parents[i]) + 1), rng) for i in range(n)}
        self.U_loc = {i: random_unitary(2 ** (len(self.children[i]) + 1), rng) for i in range(n)}
        self.order = list(range(n))  # topological (edges go i<j)

    def ancestors(self, j: int) -> set:
        anc, stack = set(), list(self.parents[j])
        while stack:
            v = stack.pop()
            if v not in anc:
                anc.add(v); stack += self.parents[v]
        return anc

    def run(self, obs_basis: Dict[int, int]) -> np.ndarray:
        """
        Sequential evolution.  obs_basis[i] = Pauli basis in which in_i is measured
        (outcome discarded) right before L_i; nodes absent from obs_basis are not measured.
        Returns the final density matrix on all in/out qubits.
        """
        n = self.nq
        rho = np.zeros((2 ** n, 2 ** n), complex); rho[0, 0] = 1
        for i in self.order:
            qs = [self.out_q[(p, i)] for p in self.parents[i]] + [self.in_q[i]]
            rho = apply_unitary(rho, self.U_in[i], qs, n)
            if i in obs_basis:
                rho = dephase(rho, obs_basis[i], self.in_q[i], n)
            qs = [self.in_q[i]] + [self.out_q[(i, k)] for k in self.children[i]]
            rho = apply_unitary(rho, self.U_loc[i], qs, n)
        return rho


# ************************************************************
# Orientation statistic
# ************************************************************
def run_cache(model: SequentialQCM) -> EvalCache:
    """
    Cache of simulator runs, keyed on (observed node, basis).  The state after
    observing node i in basis B_i does not depend on which node is read out
    afterwards, so the same run serves every read-out basis B_j and every target j.
    Evolving the density matrix is the expensive step, and this cache plays the
    role GramCache plays for kernel matrices in thrifty-qiskit.
    """
    return EvalCache(lambda obs: model.run(obs))


def disturbance(model: SequentialQCM, i: int, j: int, shots: Optional[int] = None,
                rng: Optional[np.random.Generator] = None,
                cache: Optional[EvalCache] = None,
                bases: Sequence[int] = (0, 1, 2)) -> float:
    """
    D(i -> j) = max_{B_i, B_i', B_j} || P(b_j | B_i, B_j) - P(b_j | B_i', B_j) ||_1,
    where node j is measured in basis B_j on in_j (its own observation) and node i is
    observed in basis B_i.  All other nodes unobserved.  With `shots`, each
    distribution is replaced by a binomial sample.

    `bases` is the family of measurement bases the experimenter may choose from
    (0 = X, 1 = Y, 2 = Z).  The default is the full Pauli family.  Restricting it
    to (2,) is the classical observation: reading the computational basis only.
    A classical reversible circuit is invariant under that reading, so D vanishes
    identically -- which is why the classical limit of this statistic is silence.
    """
    cache = run_cache(model) if cache is None else cache
    bases = tuple(bases)
    rhos = [cache.get((i, Bi), {i: Bi}) for Bi in bases]
    D = 0.0
    for Bj in bases:
        ps = []
        for rho in rhos:
            p = marginal_probs(rho, Bj, model.in_q[j], model.nq)
            if shots is not None:
                p = np.bincount(rng.choice(2, shots, p=p), minlength=2) / shots
            ps.append(p)
        for a, b in itertools.combinations(ps, 2):
            D = max(D, float(np.abs(a - b).sum()))
    return D


def orient_edges(model: SequentialQCM, skeleton: List[Tuple[int, int]], eps: float,
                 shots: Optional[int] = None, rng: Optional[np.random.Generator] = None,
                 cache: Optional[EvalCache] = None,
                 bases: Sequence[int] = (0, 1, 2)
                 ) -> Dict[Tuple[int, int], Optional[str]]:
    """For each undirected skeleton edge {i,j} return '->', '<-', or None (undetermined/conflict)."""
    out = {}
    cache = run_cache(model) if cache is None else cache
    for i, j in skeleton:
        dij = disturbance(model, i, j, shots, rng, cache, bases)
        dji = disturbance(model, j, i, shots, rng, cache, bases)
        if dij > eps and dji <= eps:
            out[(i, j)] = "->"
        elif dji > eps and dij <= eps:
            out[(i, j)] = "<-"
        else:
            out[(i, j)] = None
    return out


def random_dag(n: int, rng: np.random.Generator, p_edge: float = 0.5) -> List[Tuple[int, int]]:
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.uniform() < p_edge]
    return edges or [(0, 1)]


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("(A) Exact disturbance D(i->j) on random DAGs (n=4): ancestor vs non-ancestor")
    anc_min, non_max = np.inf, 0.0
    for trial in range(10):
        edges = random_dag(4, rng)
        m = SequentialQCM(4, edges, rng)
        for i, j in itertools.permutations(range(4), 2):
            d = disturbance(m, i, j)
            if i in m.ancestors(j):
                anc_min = min(anc_min, d)
            else:
                non_max = max(non_max, d)
    print(f"   min D over ancestor pairs = {anc_min:.3f}   max D over non-ancestor pairs = {non_max:.1e}")

    print("\n(B) Orientation of skeleton edges (exact), 20 random DAGs")
    correct, total = 0, 0
    for trial in range(20):
        edges = random_dag(4, rng)
        m = SequentialQCM(4, edges, rng)
        o = orient_edges(m, edges, eps=1e-6)
        correct += sum(v == "->" for v in o.values()); total += len(o)
    print(f"   correctly oriented {correct}/{total} edges (truth is always i->j with i<j)")

    print("\n(C) Finite shots per (basis pair): orientation accuracy")
    for shots in (200, 1000, 5000):
        correct, undet, total = 0, 0, 0
        for trial in range(10):
            edges = random_dag(4, rng)
            m = SequentialQCM(4, edges, rng)
            eps = 4 / np.sqrt(shots)   # ~3 sigma of the null L1 difference, max over 9 pairs
            o = orient_edges(m, edges, eps=eps, shots=shots, rng=rng)
            correct += sum(v == "->" for v in o.values())
            undet += sum(v is None for v in o.values()); total += len(o)
        print(f"   shots={shots:5d}: correct {correct}/{total}, undetermined {undet}, wrong {total-correct-undet}")

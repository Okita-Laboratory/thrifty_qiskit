"""
l2_qpc_cmi.py -- Open problem 2: PC-type discovery with quantum conditional
mutual information (CMI), and its finite-sample behaviour.

Two results.

(A) NEGATIVE: CMI evaluated on the FINAL joint state of a circuit that follows a
    DAG does not respect d-separation.  Example: A -> B -> C by two CNOTs gives
    the GHZ state, for which I(A;C|B) = 1 although B d-separates A and C.
    So "PC with CMI on the output state" is not a consistent algorithm.

(B) POSITIVE, but different from classical PC: on the PROCESS OPERATOR of
    Barrett-Lorenz-Oreshkov (2019) -- each node i has an input qubit A_i^in and
    one output qubit per outgoing edge; the model is the product of Choi
    operators rho_{A_i^in | Pa(i)^out} (disjoint systems here, so the
    commutation requirement of the quantum Markov condition holds) -- with node
    system S_i = A_i^in (x) (outputs of i):
      * d-separated (i,j | C)  =>  I(S_i;S_j|S_C) = 0        (as in BLO);
      * d-connected through a chain or fork (i - k - j, k not conditioned)
                               =>  I(S_i;S_j|S_C) = 0 as well (faithfulness FAILS);
      * d-connected by conditioning on a collider  =>  CMI > 0 (as classically);
      * adjacent (i,j)         =>  I(S_i;S_j) > 0 generically.
    The second line is the key difference: the process operator excludes the
    nodes' local operations, and indirect (chain/fork) paths only exist
    THROUGH local operations, so the process operator carries no indirect
    correlations.  Consequently marginal MI (empty conditioning set) is
    non-zero exactly for adjacent pairs, PC's conditioning is unnecessary,
    and the skeleton is read off from PAIRWISE mutual information alone.  Orientation is not encoded in the data: every output
    qubit is maximally mixed (Tr_in J = 1), so which qubit is "in" and which
    is "out" is a labelling of the experiment, not something to be inferred.
    We verify all of this numerically, then estimate the pairwise MI from M
    classical shadows of the process operator (random Pauli measurements on
    every in/out qubit = process tomography) and measure the skeleton F1
    against M.

Everything is numpy; systems stay at <= 9 qubits.
"""

from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from thrifty_qiskit import shadows as sh  # basis-rotation matrices
from thrifty_qiskit.gram_resampling import EvalCache  # the one place that touches the estimator

# ************************************************************
# Linear algebra helpers
# ************************************************************
def vn_entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


def ptrace_keep(rho: np.ndarray, n: int, keep: Sequence[int]) -> np.ndarray:
    """Reduced state on qubits `keep` (in increasing order) of an n-qubit state."""
    keep = sorted(keep)
    r = rho.reshape([2] * (2 * n))
    trace_out = [q for q in range(n) if q not in keep]
    for k, q in enumerate(trace_out):
        # after k traces the tensor has 2*(n-k) axes; qubit q is at axis q-k (row) and (n-k)+(q-k) (col)
        ax = q - k
        r = np.trace(r, axis1=ax, axis2=(n - k) + ax)
    d = 2 ** len(keep)
    return r.reshape(d, d)


def cmi(rho: np.ndarray, n: int, A: Sequence[int], B: Sequence[int], C: Sequence[int]) -> float:
    A, B, C = list(A), list(B), list(C)
    S = lambda qs: vn_entropy(ptrace_keep(rho, n, qs)) if qs else 0.0
    return S(A + C) + S(B + C) - S(A + B + C) - S(C)


# ************************************************************
# (A) Final-state counterexample
# ************************************************************
def ghz_final_state_cmi() -> float:
    """A -> B -> C by CNOTs from |+00>: returns I(A;C|B) of the final state (= 1)."""
    psi = np.zeros(8, complex); psi[0] = psi[7] = 1 / np.sqrt(2)  # GHZ
    rho = np.outer(psi, psi.conj())
    return cmi(rho, 3, [0], [2], [1])


# ************************************************************
# (B) Process operator of a quantum causal model
# ************************************************************
def random_isometry(d_in: int, d_out: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(size=(d_out, d_in)) + 1j * rng.normal(size=(d_out, d_in))
    Q, _ = np.linalg.qr(G)
    return Q[:, :d_in]


def choi_of_random_channel(k_in: int, rng: np.random.Generator, env: Optional[int] = None) -> np.ndarray:
    """
    Unnormalised Choi operator J = sum_{ab} N(|a><b|) (x) |a><b| of a random channel
    N: (C^2)^{k_in} -> C^2 (one output qubit), realised by an isometry into out (x) env.
    Ordering of qubits in J: [out qubit] + [k_in input copies].  Tr_out J = 1_in.
    """
    d_in = 2 ** k_in
    env = env or max(2, d_in)                        # need 2*env >= d_in for an isometry
    V = random_isometry(d_in, 2 * env, rng)          # d_in -> out (2) x env
    J = np.zeros((2 * d_in, 2 * d_in), complex)
    for a in range(d_in):
        for b in range(d_in):
            Eab = np.zeros((d_in, d_in)); Eab[a, b] = 1
            Nab = V @ Eab @ V.conj().T                # on out x env
            Nab = Nab.reshape(2, env, 2, env)
            Nab = np.einsum("iaja->ij", Nab)         # trace out env
            J += np.kron(Nab, Eab)
    return J


class QuantumCausalModel:
    """
    DAG on n nodes.  Node i has one input qubit (index in_q[i]) and one output
    qubit per outgoing edge (index out_q[(i, j)] for edge i->j).
    Channel of node i: Pa(i)'s output qubits (those on edges into i) -> in_q[i].
    Process operator sigma = (x)_i J_i on all qubits; sigma_hat = sigma / Tr sigma.
    """

    def __init__(self, n: int, edges: List[Tuple[int, int]], rng: np.random.Generator,
                 dense_limit: int = 13):
        """
        `dense_limit` is the largest qubit count for which the dense process
        operator is materialised.  Above it `sigma_hat` is None and every
        computation goes through the factors, which is exact and much cheaper:
        the dense operator is 2^nq x 2^nq (16 GB at nq = 15) while a factor has
        at most 1 + max_parents qubits.
        """
        self.n, self.edges, self.rng = n, sorted(edges), rng
        self.dense_limit = dense_limit
        self.parents = {i: [j for (j, k) in edges if k == i] for i in range(n)}
        self.children = {i: [k for (j, k) in edges if j == i] for i in range(n)}
        # qubit indices
        q = 0
        self.in_q = {}
        self.out_q = {}
        for i in range(n):
            self.in_q[i] = q; q += 1
            for k in self.children[i]:
                self.out_q[(i, k)] = q; q += 1
        self.nq = q
        self.node_sys = {i: [self.in_q[i]] + [self.out_q[(i, k)] for k in self.children[i]] for i in range(n)}
        self.sigma_hat = self._build()

    # ------------------------------------------------------------------
    # Factorised access: everything below avoids the dense process operator.
    #
    # sigma is a tensor product of Choi factors on disjoint qubit sets, so the
    # reduced state on any subset factorises across them and the von Neumann
    # entropy is the SUM of the per-factor entropies of the kept qubits.  Each
    # factor has at most 1 + max_parents qubits, so this is cheap however many
    # vertices there are, while the dense operator is 2^nq x 2^nq and runs out
    # of memory at about 14 qubits.
    # ------------------------------------------------------------------
    def entropy_of_qubits(self, keep: Sequence[int]) -> float:
        """Exact von Neumann entropy of the reduced process operator on `keep`."""
        keep = set(keep)
        total = 0.0
        for rho, qs in self.factors:
            local = [t for t, g in enumerate(qs) if g in keep]
            if not local:
                continue
            if len(local) == len(qs):
                total += vn_entropy(rho)
            else:
                total += vn_entropy(ptrace_keep(rho, len(qs), local))
        return total

    def cmi_exact(self, i: int, j: int, C: Sequence[int] = ()) -> float:
        """I(S_i ; S_j | S_C) on the process operator, computed factor-wise."""
        A, B = self.node_sys[i], self.node_sys[j]
        Cq = [q for c in C for q in self.node_sys[c]]
        S = self.entropy_of_qubits
        return S(list(A) + Cq) + S(list(B) + Cq) - S(list(A) + list(B) + Cq) - S(Cq)

    def _build(self) -> Optional[np.ndarray]:
        # Build sigma as a tensor product over nodes, then permute qubits into global order.
        factors, order = [], []
        self.factors = []                                   # (normalised Choi state, its global qubits)
        for i in range(self.n):
            pa = self.parents[i]
            J = choi_of_random_channel(len(pa), self.rng)   # qubits: [in_q[i]] + [out_q[(p,i)] for p in pa]
            factors.append(J)
            qs = [self.in_q[i]] + [self.out_q[(p, i)] for p in pa]
            self.factors.append((J / np.trace(J).real, qs))
            order += qs
        if self.nq > self.dense_limit:
            return None                       # too large; use the factorised path
        sigma = factors[0]
        for F in factors[1:]:
            sigma = np.kron(sigma, F)
        # permute from `order` to global 0..nq-1
        perm = np.argsort(order)          # position in `order` of global qubit g
        nq = self.nq
        t = sigma.reshape([2] * (2 * nq))
        t = np.transpose(t, list(perm) + [nq + p for p in perm])
        sigma = t.reshape(2 ** nq, 2 ** nq)
        return sigma / np.trace(sigma).real

    # ---- graph queries ----
    def d_separated(self, i: int, j: int, C: FrozenSet[int]) -> bool:
        """Standard d-separation via Bayes-ball on the DAG."""
        anc = set(C)
        frontier = list(C)
        while frontier:
            v = frontier.pop()
            for p in self.parents[v]:
                if p not in anc:
                    anc.add(p); frontier.append(p)
        # reachable search: state (node, direction) direction: 'up' (from child) / 'down' (from parent)
        visited, stack = set(), [(i, "up")]
        while stack:
            v, dr = stack.pop()
            if (v, dr) in visited:
                continue
            visited.add((v, dr))
            if v == j and v not in C:
                return False
            if dr == "up" and v not in C:
                for p in self.parents[v]: stack.append((p, "up"))
                for c in self.children[v]: stack.append((c, "down"))
            elif dr == "down":
                if v not in C:
                    for c in self.children[v]: stack.append((c, "down"))
                if v in anc:
                    for p in self.parents[v]: stack.append((p, "up"))
        return True

    def cmi_nodes(self, i: int, j: int, C: Sequence[int], rho: Optional[np.ndarray] = None) -> float:
        if rho is None:
            return self.cmi_exact(i, j, C)    # factorised: no dense operator needed
        Cq = [q for c in C for q in self.node_sys[c]]
        return cmi(rho, self.nq, self.node_sys[i], self.node_sys[j], Cq)


def random_dag(n: int, rng: np.random.Generator, p_edge: float = 0.5) -> List[Tuple[int, int]]:
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.uniform() < p_edge]
    return edges


# ************************************************************
# Shadows of the process operator (mixed state) -> reduced density matrices
# ************************************************************
_U = {0: sh._H, 1: sh._H @ sh._SDG, 2: np.eye(2)}


def _rotated_diag(rho_t: np.ndarray, n: int, basis_row: np.ndarray) -> np.ndarray:
    """Diagonal of (U rho U^dag) for U = (x)_q u_{basis_row[q]}, using tensor contractions
    (cost 2^{2n} * n instead of 2^{3n})."""
    t = rho_t
    for qb in range(n):
        u = _U[basis_row[qb]]
        t = np.tensordot(u, t, axes=([1], [qb]))          # row index qb
        t = np.moveaxis(t, 0, qb)
        t = np.tensordot(u.conj(), t, axes=([1], [n + qb]))  # column index qb
        t = np.moveaxis(t, 0, n + qb)
    d = 2 ** n
    return np.real(np.diagonal(t.reshape(d, d)))


def sample_shadows_mixed(rho: np.ndarray, n: int, M: int, rng: np.random.Generator):
    bases = rng.integers(0, 3, (M, n))
    bits = np.empty((M, n), dtype=np.int8)
    rho_t = rho.reshape([2] * (2 * n))
    d = 2 ** n
    for m in range(M):
        p = _rotated_diag(rho_t, n, bases[m]); p = np.clip(p, 0, None); p /= p.sum()
        idx = rng.choice(d, p=p)
        bits[m] = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
    return bases, bits


# single-snapshot single-qubit factors  3 U^dag |b><b| U - I, indexed [basis, bit]
_FACTOR = np.empty((3, 2, 2, 2), complex)
for _B in range(3):
    for _b in range(2):
        _ket = _U[_B].conj().T @ np.eye(2)[:, [_b]]
        _FACTOR[_B, _b] = 3 * (_ket @ _ket.conj().T) - np.eye(2)


def sample_shadows_model(model: "QuantumCausalModel", M: int, rng: np.random.Generator):
    """
    Shadows of the process operator using its tensor-product structure: the Choi factors
    act on disjoint qubit sets, so each factor is sampled independently.  Snapshots with the
    same basis pattern on a factor share one rotated distribution.  Output is distributed
    identically to sample_shadows_mixed(model.sigma_hat, ...) but is exponentially cheaper.
    """
    n = model.nq
    bases = rng.integers(0, 3, (M, n))
    bits = np.empty((M, n), dtype=np.int8)
    for rho, qs in model.factors:
        k = len(qs)
        rho_t = rho.reshape([2] * (2 * k))
        sub = bases[:, qs]                                          # (M, k)
        keys = sub @ (3 ** np.arange(k))                            # pattern id
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            p = _rotated_diag(rho_t, k, sub[idx[0]]); p = np.clip(p, 0, None); p /= p.sum()
            draws = rng.choice(2 ** k, size=len(idx), p=p)
            for t, qb in enumerate(qs):
                bits[idx, qb] = (draws >> (k - 1 - t)) & 1
    return bases, bits


def shadow_reduced_state(bases: np.ndarray, bits: np.ndarray, keep: Sequence[int],
                         chunk: Optional[int] = None,
                         max_bytes: int = 256_000_000) -> np.ndarray:
    """Standard shadow estimator of the reduced state on `keep`: mean over snapshots of
    (x)_{q in keep} (3 U^dag|b><b|U - I).  Vectorised over snapshots in chunks."""
    keep = sorted(keep)
    d = 2 ** len(keep)
    M = bases.shape[0]
    if chunk is None:
        # the intermediate array is (chunk, d, d) complex: bound it
        chunk = max(1, min(M, int(max_bytes // max(16 * d * d, 1))))
    acc = np.zeros((d, d), complex)
    for start in range(0, M, chunk):
        sl = slice(start, min(start + chunk, M))
        f = _FACTOR[bases[sl, keep[0]], bits[sl, keep[0]]]          # (m, 2, 2)
        for q in keep[1:]:
            g = _FACTOR[bases[sl, q], bits[sl, q]]                    # (m, 2, 2)
            f = np.einsum("mij,mkl->mikjl", f, g).reshape(f.shape[0], f.shape[1] * 2, f.shape[2] * 2)
        acc += f.sum(axis=0)
    return acc / M


def project_to_state(A: np.ndarray) -> np.ndarray:
    """Nearest density matrix (Hermitise, clip negative eigenvalues, renormalise)."""
    A = 0.5 * (A + A.conj().T)
    w, v = np.linalg.eigh(A)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return np.eye(len(A)) / len(A)
    return (v * w) @ v.conj().T / w.sum()


def entropy_cache(bases, bits) -> EvalCache:
    """
    Cache of shadow-estimated von Neumann entropies, keyed on the qubit subset.
    Reconstructing a reduced state from M snapshots is the expensive step here, and
    the same subsets recur across node pairs and conditioning sets, so the cache
    plays exactly the role GramCache plays for kernel matrices in thrifty-qiskit.
    """
    def _S(qs):
        return vn_entropy(project_to_state(shadow_reduced_state(bases, bits, qs)))
    return EvalCache(_S)


def cmi_from_shadows(model: QuantumCausalModel, bases, bits, i: int, j: int, C: Sequence[int],
                     cache: Optional[EvalCache] = None) -> float:
    A, B = model.node_sys[i], model.node_sys[j]
    Cq = [q for c in C for q in model.node_sys[c]]
    cache = entropy_cache(bases, bits) if cache is None else cache

    def S(qs):
        return cache.get(tuple(sorted(qs)), sorted(qs)) if qs else 0.0

    return S(A + Cq) + S(B + Cq) - S(A + B + Cq) - S(Cq)


# ************************************************************
# PC skeleton with a CI oracle
# ************************************************************
def pc_skeleton(n: int, ci_test, max_cond: int = 2) -> List[Tuple[int, int]]:
    adj = {i: set(range(n)) - {i} for i in range(n)}
    for size in range(max_cond + 1):
        for i, j in itertools.combinations(range(n), 2):
            if j not in adj[i]:
                continue
            others = adj[i] - {j}
            for C in itertools.combinations(sorted(others), size):
                if ci_test(i, j, C):
                    adj[i].discard(j); adj[j].discard(i)
                    break
    return sorted((i, j) for i in range(n) for j in adj[i] if i < j)


def skeleton_f1(true_edges, est_edges) -> float:
    T, E = set(true_edges), set(est_edges)
    tp = len(T & E)
    if tp == 0:
        return 0.0
    p, r = tp / len(E), tp / len(T)
    return 2 * p * r / (p + r)


# ************************************************************
# Skeleton from pairwise MI
# ************************************************************
def pairwise_mi_skeleton(model: QuantumCausalModel, mi_fn, thr: float) -> List[Tuple[int, int]]:
    return sorted((i, j) for i, j in itertools.combinations(range(model.n), 2) if mi_fn(i, j) > thr)


def mi_threshold(M: int, n_qubits_pair: int) -> float:
    """Crude finite-sample threshold: shadow entropy bias scales like d^2/M with d = 2^(#qubits)."""
    return 0.05 + 4.0 ** n_qubits_pair / M


def adaptive_threshold(values: Sequence[float], k: float = 1.0) -> float:
    """
    Threshold calibrated on the estimates themselves.

    The shadow estimate of a mutual information is biased upward, and the bias
    depends on the subsystem dimension and on the number of snapshots but not on
    whether the pair is adjacent.  Non-adjacent pairs therefore sit in a cluster
    at the low end, and that cluster measures the bias and its spread.

    The null level is taken from the LOWER HALF of the values only, which cannot
    be dominated by edges unless the graph is denser than 50 percent.  Using the
    median of all values instead fails on small or dense graphs, where half the
    pairs are edges; a fixed per-pair threshold fails in the other direction,
    letting the false positives grow with the n^2/2 comparisons.

    Two constants correct for looking at half the distribution.  For a Gaussian
    null, the median of the lower half sits at -0.669 sigma rather than at the
    centre, and the MAD of that half is 0.388 sigma rather than 0.674, so

        centre  = median(lower) + 0.669 * sigma,
        sigma   = MAD(lower) / 0.388,

    and the cut is placed at centre + k * sigma.  Both constants were obtained
    numerically and hold for a Gaussian null.

    The default k = 1 is an empirical choice, tuned on the benchmark of
    l2_benchmark, not a false-positive-rate guarantee: the null here is the
    upward bias of a shadow entropy estimate, which is neither Gaussian nor
    identically distributed across pairs, and a nominally 3-sigma cut discards
    most of the true edges at n = 15.  Raise k when false positives are the
    expensive error, lower it when missing edges is.
    """
    v = np.asarray(list(values), float)
    if len(v) < 4:
        return float(v.min()) if len(v) else 0.0
    lo = v[v <= np.median(v)]
    med = float(np.median(lo))
    mad = float(np.median(np.abs(lo - med)))
    if mad <= 0:
        return med + 1e-9
    sigma = mad / 0.3884
    return med + 0.6693 * sigma + k * sigma


def pairwise_mi_skeleton_adaptive(model, mi_fn, k: float = 1.0
                                  ) -> List[Tuple[int, int]]:
    """Skeleton from pairwise MI with the threshold calibrated on the values."""
    pairs = [(i, j) for i in range(model.n) for j in range(i + 1, model.n)]
    vals = [mi_fn(i, j) for (i, j) in pairs]
    thr = adaptive_threshold(vals, k)
    return [p for p, v in zip(pairs, vals) if v > thr]


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("(A) Final-state CMI on GHZ chain A->B->C:  I(A;C|B) =", round(ghz_final_state_cmi(), 4),
          " (d-separated, but CMI != 0)")

    print("\n(B) Process operator, random DAGs (n=4, 20 models): CMI by adjacency / d-separation")
    sep_max, chain_max, coll_min, nonadj_marg_max, adj_min = 0.0, 0.0, np.inf, 0.0, np.inf
    for trial in range(20):
        edges = random_dag(4, rng, 0.5)
        m = QuantumCausalModel(4, edges, rng)
        for i, j in itertools.combinations(range(4), 2):
            adjacent = (i, j) in edges
            for size in range(3):
                for C in itertools.combinations([k for k in range(4) if k not in (i, j)], size):
                    v = m.cmi_nodes(i, j, C)
                    dsep = m.d_separated(i, j, frozenset(C))
                    if dsep:
                        sep_max = max(sep_max, v)
                    elif not adjacent:
                        # d-connected non-adjacent: through a collider in C (or its descendant)?
                        colliders = [k for k in C if i in m.parents[k] and j in m.parents[k]]
                        if colliders:
                            coll_min = min(coll_min, v)
                        else:
                            chain_max = max(chain_max, v)
            if not adjacent:
                nonadj_marg_max = max(nonadj_marg_max, m.cmi_nodes(i, j, ()))
            else:
                adj_min = min(adj_min, m.cmi_nodes(i, j, ()))
    print(f"   d-separated                      : max CMI = {sep_max:.1e}")
    print(f"   d-connected via chain/fork only  : max CMI = {chain_max:.1e}   (faithfulness fails)")
    print(f"   d-connected via collider in C    : min CMI = {coll_min:.3f}   (as classically)")
    print(f"   non-adjacent, C = empty          : max MI  = {nonadj_marg_max:.1e}")
    print(f"   adjacent,     C = empty          : min MI  = {adj_min:.3f}")

    print("\n(C) Skeleton from exact pairwise MI (no conditioning)")
    ok = 0
    for trial in range(20):
        edges = random_dag(4, rng, 0.5)
        m = QuantumCausalModel(4, edges, rng)
        est = pairwise_mi_skeleton(m, lambda i, j: m.cmi_nodes(i, j, ()), 1e-8)
        ok += est == sorted(edges)
    print(f"   exact skeleton recovered in {ok}/20 random models")

    print("\n(D) Finite sample: pairwise MI from M shadows of the process operator (n=4, 5 models each)")
    for M in (1000, 4000, 16000):
        f1s = []
        for trial in range(5):
            edges = random_dag(4, rng, 0.5) or [(0, 1)]
            m = QuantumCausalModel(4, edges, rng)
            bases, bits = sample_shadows_mixed(m.sigma_hat, m.nq, M, rng)
            cache = entropy_cache(bases, bits)
            est = pairwise_mi_skeleton(
                m, lambda i, j: cmi_from_shadows(m, bases, bits, i, j, (), cache),
                mi_threshold(M, 4))
            f1s.append(skeleton_f1(edges, est))
        print(f"   M={M:6d}: mean skeleton F1 = {np.mean(f1s):.2f}")

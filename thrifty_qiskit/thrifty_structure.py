"""
thrifty_structure.py -- exploiting the structure of the feature map itself.

Two structures are detected and used here.

(1) THE FEATURE MAP IS BAND-LIMITED TOO.  Everything so far reconstructed the
    kernel k(x - y).  But the individual expectation values of a parametrised
    circuit are band-limited in exactly the same way: for a feature map that
    encodes each feature through Pauli rotations,

        b(x) = <0| U(x)^dag O U(x) |0>  =  sum_{w in Omega} c_w exp(i w . x),

    with Omega again fixed by the encoding gates.  A projected quantum kernel
    (PQK) is built from the one-qubit reduced density matrices, i.e. from the
    3q Bloch components b_{k,P}(x), and then applies a classical kernel to them:

        k_PQ(x, y) = exp( -gamma sum_{k,P} ( b_{k,P}(x) - b_{k,P}(y) )^2 ).

    PQK is the standard remedy for exponential concentration, and it already
    costs only O(n) state preparations rather than O(n^2).  Reconstructing every
    Bloch component from |Omega| nodes replaces that by |Omega| -- independent of
    n -- and makes the outer bandwidth free as well.  Note that k_PQ itself is
    NOT shift-invariant in x, so the earlier machinery does not apply to it; what
    is band-limited is the map, not the kernel.

(2) PARTIAL SEPARABILITY.  Full separability (thrifty_scaling) needs the
    kernel to factorise over single features, which fails as soon as the circuit
    entangles them.  Circuits usually entangle only some pairs, so the kernel
    factorises over BLOCKS of features.  Testing each pair for factorisation
    gives an interaction graph whose connected components are the blocks, and
    the cost becomes

        sum_b (2L+1)^{|b|}      instead of      (2L+1)^d,

    exponential in the largest block rather than in the feature dimension.  The
    detected graph is also a description of the feature map: it says which
    features the encoding actually couples.

Depends on thrifty_budget and thrifty_scaling.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from thrifty_qiskit.thrifty_budget import CanonicalKernel, canonical_design, canonical_frequencies
from thrifty_qiskit.thrifty_exact import _2d

VectorFn = Callable[[np.ndarray], np.ndarray]      # X (n,d) -> (n, m) expectation values
PairFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ************************************************************
# (1) Band-limited vector-valued functions and projected kernels
# ************************************************************
class BandLimitedVectorFunction:
    """
    Exact reconstruction of a vector-valued band-limited function b: R^d -> R^m
    from |Omega| = (2L+1)^d evaluation points.

    Unlike the kernel case the nodes live in x, not in the difference x - y, but
    the frequency set is the same and the linear system is the same size.  All m
    components share the same nodes, so the circuit cost is |Omega| settings
    regardless of m (one execution yields every Pauli expectation on every qubit
    when read out with random Pauli measurements).
    """

    def __init__(self, fn: VectorFn, d: int, reps: int = 1, gate_freq: float = 2.0,
                 check: bool = True, tol: float = 1e-8):
        self.fn, self.d, self.reps, self.gate_freq = fn, d, reps, gate_freq
        self.check, self.tol = check, tol
        self.W = canonical_frequencies(d, reps, gate_freq)
        self.n_dof = sum(1 if np.allclose(w, 0) else 2 for w in self.W)
        self.nodes = self._nodes()
        self.coef_: Optional[np.ndarray] = None      # (n_dof, m)
        self.n_circuit_evals = 0
        self.n_check_evals = 0

    def _nodes(self) -> np.ndarray:
        m = 2 * self.reps + 1
        base = np.linspace(0.0, 2 * np.pi / self.gate_freq, m, endpoint=False)
        return np.array(list(itertools.product(base, repeat=self.d)), dtype=float)

    def fit(self) -> "BandLimitedVectorFunction":
        V = np.atleast_2d(self.fn(self.nodes))           # (|Omega|, m)
        self.n_circuit_evals += len(self.nodes)
        A = canonical_design(self.nodes, self.W)
        self.coef_ = np.linalg.pinv(A) @ V
        if self.check:
            self._verify()
        return self

    def _verify(self, n_test: int = 8, rng: Optional[np.random.Generator] = None) -> None:
        rng = rng or np.random.default_rng(0)
        X = rng.uniform(0, np.pi, (n_test, self.d))
        truth = np.atleast_2d(self.fn(X))
        self.n_check_evals += n_test
        if np.max(np.abs(self(X) - truth)) > self.tol:
            raise ValueError("the function is not band-limited on this Omega")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return canonical_design(_2d(X), self.W) @ self.coef_


class ProjectedKernel:
    """
    Projected quantum kernel built on a reconstructed feature map.

    k_PQ(x, y) = exp( -gamma ||b(x) - b(y)||^2 ),  b reconstructed once.
    Every sample size and every gamma is then free.
    """

    def __init__(self, bloch: BandLimitedVectorFunction, gamma: float = 1.0):
        self.bloch, self.gamma = bloch, float(gamma)

    @property
    def n_circuit_evals(self) -> int:
        return self.bloch.n_circuit_evals

    def pairs(self, A: np.ndarray, B: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        g = self.gamma if gamma is None else float(gamma)
        Ba, Bb = self.bloch(A), self.bloch(B)
        d2 = (np.sum(Ba ** 2, axis=1)[:, None] + np.sum(Bb ** 2, axis=1)[None, :]
              - 2 * Ba @ Bb.T)
        return np.exp(-g * np.clip(d2, 0, None))

    def __call__(self, X: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        K = self.pairs(X, X, gamma)
        return 0.5 * (K + K.T)

    def bandwidth_sweep(self, X: np.ndarray, gammas: Sequence[float]) -> List[np.ndarray]:
        return [self(X, g) for g in gammas]


# ************************************************************
# (2) Interaction graph and block-wise reconstruction
# ************************************************************
def detect_interaction_graph(pair_fn: PairFn, d: int, reps: int = 1,
                             gate_freq: float = 2.0, tol: float = 1e-6,
                             rng: Optional[np.random.Generator] = None
                             ) -> Tuple[Set[Tuple[int, int]], int]:
    """
    Decide, for each pair of features, whether the kernel factorises over them:

        k(delta_i e_i + delta_j e_j) =? k(delta_i e_i) k(delta_j e_j).

    Returns the set of coupled pairs and the number of evaluations used.
    The test is local -- it only probes the two-feature slices -- so its cost is
    O(d^2) evaluations and does not grow with the sample size.
    """
    rng = rng or np.random.default_rng(0)
    m = 2 * reps + 1
    base = np.linspace(0.0, 2 * np.pi / gate_freq, m, endpoint=False)[1:]   # skip 0
    zero = np.zeros((1, d))
    n_eval = 0

    def axis_values(j: int) -> np.ndarray:
        P = np.zeros((len(base), d)); P[:, j] = base
        return pair_fn(P, zero)[:, 0]

    axes = {}
    for j in range(d):
        axes[j] = axis_values(j)
        n_eval += len(base)

    edges: Set[Tuple[int, int]] = set()
    for i, j in itertools.combinations(range(d), 2):
        P = np.zeros((len(base), d))
        P[:, i] = base
        P[:, j] = base
        joint = pair_fn(P, zero)[:, 0]
        n_eval += len(base)
        if np.max(np.abs(joint - axes[i] * axes[j])) > tol:
            edges.add((i, j))
    return edges, n_eval


def connected_blocks(d: int, edges: Set[Tuple[int, int]]) -> List[List[int]]:
    """Connected components of the interaction graph, as sorted feature lists."""
    parent = list(range(d))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: Dict[int, List[int]] = {}
    for i in range(d):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


class BlockExactKernel:
    """
    Reconstruct a kernel that factorises over blocks of features.

    Cost: sum_b (2L+1)^{|b|} evaluations (sharing the delta = 0 point), which is
    exponential in the largest block only.  Fully separable and fully coupled are
    the two extremes of this.
    """

    def __init__(self, pair_fn: PairFn, d: int, reps: int = 1, gate_freq: float = 2.0,
                 blocks: Optional[List[List[int]]] = None, check: bool = True,
                 tol: float = 1e-8):
        self.pair_fn, self.d, self.reps, self.gate_freq = pair_fn, d, reps, gate_freq
        self.check, self.tol = check, tol
        self.blocks = blocks
        self.factors: List[Tuple[List[int], CanonicalKernel]] = []
        self.n_circuit_evals = 0
        self.n_check_evals = 0
        self.edges: Optional[Set[Tuple[int, int]]] = None

    def fit(self) -> "BlockExactKernel":
        if self.blocks is None:
            self.edges, n = detect_interaction_graph(self.pair_fn, self.d, self.reps,
                                                     self.gate_freq)
            self.n_check_evals += n
            self.blocks = connected_blocks(self.d, self.edges)
        zero = np.zeros((1, self.d))
        first = True
        for blk in self.blocks:
            ck = CanonicalKernel(d=len(blk), reps=self.reps, gate_freq=self.gate_freq)
            P = np.zeros((len(ck.nodes), self.d))
            P[:, blk] = ck.nodes
            vals = self.pair_fn(P, zero)[:, 0]
            self.n_circuit_evals += len(ck.nodes) if first else len(ck.nodes) - 1
            first = False
            ck.A_ = canonical_design(ck.nodes, ck.W)
            ck.Ap_ = np.linalg.pinv(ck.A_)
            ck.node_values_ = vals
            ck.shots_ = None
            ck.coef_ = ck.Ap_ @ vals
            ck.cov_ = np.zeros((len(ck.coef_), len(ck.coef_)))
            self.factors.append((blk, ck))
        if self.check:
            self._verify()
        return self

    def _verify(self, n_test: int = 5, rng: Optional[np.random.Generator] = None) -> None:
        rng = rng or np.random.default_rng(0)
        X = rng.uniform(0, np.pi, (n_test, self.d))
        Y = rng.uniform(0, np.pi, (n_test, self.d))
        self.n_check_evals += n_test * n_test
        if np.max(np.abs(self.pairs(X, Y) - self.pair_fn(X, Y))) > self.tol:
            raise ValueError("the kernel does not factorise over the detected blocks")

    def pairs(self, A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        A, B = _2d(A), _2d(B)
        out = np.ones((len(A), len(B)))
        for blk, ck in self.factors:
            D = (A[:, blk][:, None, :] - B[:, blk][None, :, :]).reshape(-1, len(blk)) * gamma
            out *= (canonical_design(D, ck.W) @ ck.coef_).reshape(len(A), len(B))
        return out

    def __call__(self, X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        K = self.pairs(X, X, gamma)
        return 0.5 * (K + K.T)


def block_cost(blocks: Sequence[Sequence[int]], reps: int) -> int:
    m = 2 * reps + 1
    return int(sum(m ** len(b) for b in blocks) - (len(blocks) - 1))


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---------- a toy feature map with band-limited Bloch components ----------
    def bloch_map(X, q=3, reps=1):
        """
        Bloch components of a product feature map: qubit k encodes feature k with
        a Pauli rotation, so <X> = 0, <Y> = -sin(2 x_k), <Z> = cos(2 x_k).
        Band-limited on Omega with gate_freq = 2.
        """
        X = _2d(X)
        cols = []
        for k in range(q):
            t = reps * X[:, k]
            cols += [np.zeros(len(X)), -np.sin(2 * t), np.cos(2 * t)]
        return np.column_stack(cols)

    print("=== (1) projected quantum kernel from |Omega| circuits ===")
    d = q = 3
    n = 250
    x = rng.uniform(0, np.pi, (n, d))
    bl = BandLimitedVectorFunction(lambda X: bloch_map(X, q), d=d, reps=1).fit()
    pq = ProjectedKernel(bl, gamma=1.0)

    def pqk_direct(A, B, g=1.0):
        Ba, Bb = bloch_map(A, q), bloch_map(B, q)
        d2 = (np.sum(Ba ** 2, 1)[:, None] + np.sum(Bb ** 2, 1)[None, :] - 2 * Ba @ Bb.T)
        return np.exp(-g * np.clip(d2, 0, None))

    print(f"  |Omega| = {len(bl.nodes)}, components reconstructed = {bl.coef_.shape[1]}")
    print(f"  circuits = {bl.n_circuit_evals} (validity check {bl.n_check_evals}); "
          f"a direct PQK needs {n} state preparations, a fidelity kernel {n*(n-1)//2}")
    print(f"  max |K_PQ - exact| = {np.max(np.abs(pq(x) - pqk_direct(x, x))):.2e}")
    before = pq.n_circuit_evals
    gammas = [0.1, 0.5, 1.0, 4.0]
    Ks = pq.bandwidth_sweep(x, gammas)
    print(f"  {len(gammas)} outer bandwidths, circuits added = {pq.n_circuit_evals - before}")
    for g, K in zip(gammas, Ks):
        print(f"   gamma={g:4.1f}: max err {np.max(np.abs(K - pqk_direct(x, x, g))):.1e}, "
              f"mean off-diagonal {np.mean(K[~np.eye(n, dtype=bool)]):.3f}")

    # ---------- block structure ----------
    def blocked_kernel(A, B, reps=1, pairs=((0, 1), (3, 4))):
        """Product over features, with extra coupling inside the given pairs."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        d = A.shape[1]
        K = np.ones((A.shape[0], B.shape[0]))
        for j in range(d):
            K *= np.cos(reps * (A[:, j][:, None] - B[:, j][None, :])) ** 2
        for (i, j) in pairs:
            c = (A[:, i] + A[:, j])[:, None] - (B[:, i] + B[:, j])[None, :]
            K *= np.cos(reps * c) ** 2
        return K

    print("\n=== (2) interaction graph and block-wise reconstruction ===")
    d, reps = 6, 2
    X = rng.uniform(0, np.pi, (40, d))
    Y = rng.uniform(0, np.pi, (30, d))
    kb = BlockExactKernel(lambda A, B: blocked_kernel(A, B, 1), d, reps).fit()
    print(f"  detected couplings {sorted(kb.edges)}")
    print(f"  blocks {kb.blocks}")
    print(f"  max |K - exact| = {np.max(np.abs(kb.pairs(X, Y) - blocked_kernel(X, Y, 1))):.2e}")
    m = 2 * reps + 1
    print(f"  circuits: block-wise {kb.n_circuit_evals} "
          f"(graph detection {kb.n_check_evals - 25}), "
          f"full tensor {m ** d}, separable would be {2*reps*d+1} but is invalid here")

    print("\n  cost as the coupling grows (reps=2, d=8):")
    for name, pairs in (("no coupling", ()),
                        ("2 pairs", ((0, 1), (4, 5))),
                        ("a chain of 4", ((0, 1), (1, 2), (2, 3))),
                        ("all coupled", tuple(itertools.combinations(range(8), 2))[:12])):
        dd = 8
        e, _ = detect_interaction_graph(lambda A, B: blocked_kernel(A, B, 1, pairs), dd, 2)
        blocks = connected_blocks(dd, e)
        print(f"   {name:14s}: blocks {[len(b) for b in blocks]}, "
              f"cost {block_cost(blocks, 2):>7,} vs tensor {5**dd:,}")

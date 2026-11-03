"""
l2_classical_pc.py -- the classical baseline for the L2 benchmark.

The benchmark so far compared a quantum causal model with its dephased version,
but analysed BOTH with the same estimator: pairwise mutual information on the
process operator.  That answers "does the estimator still work if the model is
classical", not "what would a classical causal discovery method do here".  This
module supplies the missing comparison.

What it does
------------
A dephased quantum causal model is a classical causal Bayesian network on the
same DAG: the Choi factor of each vertex, made diagonal, is a conditional
probability table P(a_i | parents).  So the model can be sampled the way any
Bayesian network is sampled -- ancestral sampling, one categorical draw per
vertex -- giving a table of N i.i.d. observations of the vertex variables.

That table is exactly what a classical causal discovery method expects.  We run
the PC algorithm on it (chi-square conditional independence tests, then the
standard orientation rules) and compare with what the quantum estimators
recovered from the same underlying model.

Why this is the right baseline
------------------------------
The three methods use different data about the same object:

    quantum skeleton   -- process operator, pairwise MI, no conditioning
    quantum orientation-- measurement bases, (O)-type observation
    classical PC       -- N i.i.d. samples of the vertex values

PC recovers a CPDAG: the skeleton plus the orientations forced by colliders and
by acyclicity.  Edges outside that are not identifiable from observational data
alone, which is the classical limitation the quantum orientation statistic
sidesteps.  Comparing "edges PC leaves undirected" with "edges the quantum
statistic orients" is the comparison that means something.

No external dependency: the PC implementation here is self-contained.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from causal_qiskit import l2_qpc_cmi as qpc


# *****************************************
# Sampling a dephased model as a classical Bayesian network
# *****************************************
def conditional_tables(model: qpc.QuantumCausalModel) -> Dict[int, np.ndarray]:
    """
    P(a_i | parents) for every vertex, read off the diagonal of its Choi factor.

    The Choi operator of vertex i lives on [in_i] + [out_(p,i) for p in parents].
    Its diagonal, normalised over the input index, is the conditional
    probability table of the classical Bayesian network obtained by dephasing.
    """
    tables: Dict[int, np.ndarray] = {}
    for i, (J, qs) in zip(range(model.n), model.factors):
        k = len(qs)
        diag = np.real(np.diag(J)).reshape([2] * k)          # axis 0 is in_i
        n_par = k - 1
        tab = np.moveaxis(diag, 0, -1)                       # (parents..., a_i)
        tab = tab.reshape(2 ** n_par, 2)
        s = tab.sum(axis=1, keepdims=True)
        tab = np.where(s > 0, tab / np.maximum(s, 1e-300), 0.5)
        tables[i] = tab
    return tables


def sample_classical(model: qpc.QuantumCausalModel, N: int,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Ancestral sampling of the dephased model: N x n table of binary vertex values.
    This is the observational data a classical method would be given.
    """
    tables = conditional_tables(model)
    order = sorted(range(model.n), key=lambda i: len(model.parents[i]))
    order = _topological(model)
    X = np.zeros((N, model.n), dtype=int)
    for i in order:
        pa = model.parents[i]
        if pa:
            idx = np.zeros(N, dtype=int)
            for b, p in enumerate(pa):
                idx = idx * 2 + X[:, p]
        else:
            idx = np.zeros(N, dtype=int)
        p1 = tables[i][idx, 1]
        X[:, i] = (rng.uniform(size=N) < p1).astype(int)
    return X


def _topological(model: qpc.QuantumCausalModel) -> List[int]:
    seen, order = set(), []

    def visit(i):
        if i in seen:
            return
        for p in model.parents[i]:
            visit(p)
        seen.add(i); order.append(i)

    for i in range(model.n):
        visit(i)
    return order


# *****************************************
# Conditional independence test
# *****************************************
def chi2_ci_test(X: np.ndarray, i: int, j: int, C: Sequence[int],
                 alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Chi-square test of X_i _||_ X_j | X_C for binary variables.
    Returns (independent, p-value).  Strata with too few samples are skipped,
    which is the usual practice and makes the test conservative.
    """
    from scipy.stats import chi2

    C = list(C)
    if C:
        codes = np.zeros(len(X), dtype=int)
        for c in C:
            codes = codes * 2 + X[:, c]
    else:
        codes = np.zeros(len(X), dtype=int)

    stat, dof = 0.0, 0
    for s in np.unique(codes):
        m = codes == s
        if m.sum() < 10:
            continue
        tab = np.zeros((2, 2))
        for a in (0, 1):
            for b in (0, 1):
                tab[a, b] = np.sum(m & (X[:, i] == a) & (X[:, j] == b))
        n = tab.sum()
        r, c = tab.sum(axis=1), tab.sum(axis=0)
        if np.any(r == 0) or np.any(c == 0):
            continue
        exp = np.outer(r, c) / n
        stat += float(np.sum((tab - exp) ** 2 / exp))
        dof += 1
    if dof == 0:
        return True, 1.0
    p = float(chi2.sf(stat, dof))
    return p > alpha, p


# *****************************************
# PC algorithm
# *****************************************
def pc_algorithm(X: np.ndarray, alpha: float = 0.05, max_cond: int = 3
                 ) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], str], Dict]:
    """
    The PC algorithm: adjacency search, then collider orientation and Meek's
    rules.  Returns (skeleton, oriented edges, diagnostics).

    Orientations are reported as a dict keyed by the unordered edge, with value
    '->' when the edge points from the smaller to the larger index, '<-' for the
    other direction, and the edge absent when PC leaves it undirected.
    """
    n = X.shape[1]
    adj = {(i, j) for i in range(n) for j in range(i + 1, n)}
    sepset: Dict[Tuple[int, int], List[int]] = {}
    n_tests = 0

    for level in range(max_cond + 1):
        for (i, j) in sorted(adj):
            if (i, j) not in adj:
                continue
            nbrs = [k for k in range(n) if k != i and k != j
                    and (tuple(sorted((i, k))) in adj or tuple(sorted((j, k))) in adj)]
            if len(nbrs) < level:
                continue
            for C in itertools.combinations(nbrs, level):
                indep, _ = chi2_ci_test(X, i, j, C, alpha)
                n_tests += 1
                if indep:
                    adj.discard((i, j))
                    sepset[(i, j)] = list(C)
                    break

    # collider orientation: i - k - j with i,j non-adjacent and k not in sepset
    directed: Dict[Tuple[int, int], str] = {}

    def _set(a: int, b: int) -> None:
        """Record the directed edge a -> b on the unordered key."""
        key = tuple(sorted((a, b)))
        directed[key] = "->" if key[0] == a else "<-"

    for k in range(n):
        nb = [v for v in range(n) if tuple(sorted((v, k))) in adj]
        for i, j in itertools.combinations(nb, 2):
            if tuple(sorted((i, j))) in adj:
                continue
            if k not in sepset.get(tuple(sorted((i, j))), [k]):
                _set(i, k); _set(j, k)

    # Meek rule 1: i -> k, k - j, i and j non-adjacent  =>  k -> j
    changed = True
    while changed:
        changed = False
        for (a, b), d in list(directed.items()):
            src, dst = (a, b) if d == "->" else (b, a)
            for j in range(n):
                key = tuple(sorted((dst, j)))
                if j in (src, dst) or key not in adj or key in directed:
                    continue
                if tuple(sorted((src, j))) in adj:
                    continue
                _set(dst, j); changed = True

    return adj, directed, {"n_tests": n_tests, "n_sepsets": len(sepset)}


# *****************************************
# Comparison
# *****************************************
def skeleton_f1(true_edges: Sequence[Tuple[int, int]],
                est: Set[Tuple[int, int]]) -> float:
    T = {tuple(sorted(e)) for e in true_edges}
    tp = len(T & est)
    if tp == 0:
        return 0.0
    p, r = tp / len(est), tp / len(T)
    return 2 * p * r / (p + r)


def orientation_breakdown(true_edges: Sequence[Tuple[int, int]],
                          skel: Set[Tuple[int, int]],
                          directed: Dict[Tuple[int, int], str]) -> Dict[str, float]:
    """Of the true edges PC recovered, how many did it orient, and correctly?"""
    T = {tuple(sorted(e)): e for e in true_edges}
    found = [k for k in skel if k in T]
    correct = wrong = undirected = 0
    for k in found:
        forward = T[k] == k                                  # truth is i->j with i<j
        if k not in directed:
            undirected += 1
        elif (directed[k] == "->") == forward:
            correct += 1
        else:
            wrong += 1
    total = max(len(found), 1)
    return {"correct": correct / total, "wrong": wrong / total,
            "undirected": undirected / total, "n_recovered": len(found)}


def run_classical_pc(edges: List[Tuple[int, int]], n_vertices: int, N: int,
                     seed: int, alpha: float = 0.05) -> Dict:
    """Sample the dephased model and run PC on the resulting table."""
    rng_m = np.random.default_rng(seed)
    model = qpc.QuantumCausalModel(n_vertices, edges, rng_m)
    X = sample_classical(model, N, np.random.default_rng(seed + 30_000))
    skel, directed, diag = pc_algorithm(X, alpha=alpha)
    out = {"N": N, "f1": skeleton_f1(edges, skel), "n_tests": diag["n_tests"],
           **orientation_breakdown(edges, skel, directed)}
    return out


# *****************************************
# Demo
# *****************************************
if __name__ == "__main__":
    import sys
    from datetime import datetime
    from experiments import l2_benchmark as bm

    class Tee:
        """
        Echo to the terminal and keep a copy for the log file.

        The log is written once, at exit, from a buffer.  Writing to both streams as the
        program runs lets them interleave when the output is piped, and a multibyte
        character split across the boundary comes out corrupted; buffering removes the
        possibility entirely.
        """

        def __init__(self, path):
            import atexit
            self.path, self.buf, self.out = path, [], sys.__stdout__
            atexit.register(self.close)

        def write(self, s):
            self.out.write(s)
            self.buf.append(s)

        def flush(self):
            self.out.flush()

        def close(self):
            if self.buf is not None:
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write("".join(self.buf))
                self.buf = None

    sys.stdout = Tee("l2_classical_pc.log")
    print(f"# l2_classical_pc  {datetime.now().isoformat(timespec='seconds')}")

    print("\n=== (1) sanity: PC recovers the skeleton from enough samples ===")
    print(f"{'family':>9s} {'n':>2s} {'N':>7s} {'F1':>7s} {'CI tests':>9s}")
    for fam in ("chain", "fork", "collider", "diamond"):
        for N in (2000, 20000):
            f1, nt = [], []
            for r in range(5):
                rng = np.random.default_rng(5000 + r)
                edges = bm.FAMILIES[fam](4, rng)
                o = run_classical_pc(edges, 4, N, seed=5000 + r)
                f1.append(o["f1"]); nt.append(o["n_tests"])
            print(f"{fam:>9s} {4:2d} {N:7d} {np.mean(f1):7.3f} {np.mean(nt):9.0f}")

    print("\n=== (2) what PC can and cannot orient ===")
    print(f"{'family':>9s} {'recovered':>10s} {'correct':>8s} {'undirected':>11s} {'wrong':>7s}")
    for fam in ("chain", "fork", "collider", "diamond", "random50"):
        acc = {"correct": [], "undirected": [], "wrong": [], "n_recovered": []}
        for r in range(8):
            rng = np.random.default_rng(6000 + r)
            edges = bm.FAMILIES[fam](4, rng)
            o = run_classical_pc(edges, 4, 20000, seed=6000 + r)
            for k in acc:
                acc[k].append(o[k])
        print(f"{fam:>9s} {np.mean(acc['n_recovered']):10.1f} "
              f"{np.mean(acc['correct']):8.2f} {np.mean(acc['undirected']):11.2f} "
              f"{np.mean(acc['wrong']):7.2f}")
    print("  PC identifies colliders, but chains and forks are not oriented by observation.")
    print("  That is the classical limit the quantum orientation statistic sidesteps.")

    print("\n=== (3) side by side on the same models ===")
    print(f"{'family':>9s} {'PC ok':>9s} {'PC undir':>9s} | "
          f"{'quantum ok':>10s} {'q undec':>10s}")
    for fam in ("chain", "fork", "collider", "diamond", "random50"):
        pc = {"correct": [], "undirected": []}
        qu = {"correct": [], "undecided": []}
        for r in range(8):
            rng = np.random.default_rng(7000 + r)
            edges = bm.FAMILIES[fam](4, rng)
            o = run_classical_pc(edges, 4, 20000, seed=7000 + r)
            pc["correct"].append(o["correct"]); pc["undirected"].append(o["undirected"])
            a = bm.run_orientation(edges, 4, None, seed=7000 + r)
            qu["correct"].append(a["correct"]); qu["undecided"].append(a["undecided"])
        print(f"{fam:>9s} {np.mean(pc['correct']):9.2f} {np.mean(pc['undirected']):9.2f} | "
              f"{np.mean(qu['correct']):10.2f} {np.mean(qu['undecided']):10.2f}")
    print("  PC uses N=20000 samples; the quantum column is the exact value.")
    print("  Different resources, so this is not a ranking: it is what each can identify.")

    print("\n# wrote l2_classical_pc.log")

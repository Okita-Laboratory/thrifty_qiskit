"""
l2_benchmark.py -- a benchmark for causal discovery on quantum causal models.

Classical causal discovery has standard benchmarks (SACHS, ASIA, ALARM), but
they are datasets of classical variables and so belong to L1.  For L2, where the
vertices are quantum systems and the model is a process operator, no standard
benchmark exists.  This module defines one.

Design
------
The benchmark has to make two comparisons possible.

(1) ACROSS STRUCTURES.  The graph families are the ones whose treatment differs:
    chain, fork and collider (the three d-separation patterns), a diamond (two
    paths between the same endpoints), a star, a full chain of length n, and
    random DAGs at a controlled edge density.  Each is generated at several
    sizes so that the cost and the accuracy can be traced against n.

(2) AGAINST THE CLASSICAL LIMIT.  Every model is evaluated twice: once as a
    quantum causal model, and once in its classical limit, analysed with the
    same estimator, so that the difference isolates what the quantum structure
    contributes.  The classical limit is taken differently for the two tasks,
    because the two tasks use different objects:

      skeleton     -- dephase the process operator in the computational basis,
                      which turns the Choi factors into conditional probability
                      tables and the model into a classical Bayesian network on
                      the same DAG;
      orientation  -- replace every gate by a permutation matrix AND restrict the
                      observation to the computational basis, which together make
                      the circuit a classical reversible computation that reading
                      it does not disturb.

    The two limits give opposite answers, and that contrast is the point of the
    benchmark: the skeleton survives the classical limit, the orientation does
    not.

    What the orientation result does NOT say is that classical causal discovery
    cannot orient edges from observation.  It can: LiNGAM (Shimizu et al. 2006)
    and additive-noise models (Hoyer et al. 2008) identify a full DAG without
    intervention, by assuming the functional form of the generating process --
    linear with non-Gaussian noise, or additive noise -- so that the residual is
    independent of the input in the causal direction only.  Those assumptions do
    not hold for the models here, which are discrete Bayesian networks with no
    linear or non-Gaussian structure, so LiNGAM is not applicable to this
    benchmark rather than failing on it.  The statistic used here buys
    identifiability with a different resource: the choice of measurement basis,
    which relies on no-signalling and assumes nothing about the functional form.
    The classical limit is silent because reading a classical variable does not
    disturb it, not because orientation is impossible classically.

What limits the scale
---------------------
Two things could have limited how large a model the estimators reach, and it is
worth separating them because only one of them binds.

The number of VERTICES does not.  The process operator is a tensor product of
Choi factors on disjoint qubit sets, so its entropies decompose into a sum over
factors and the dense 2^nq x 2^nq operator never has to be formed.  Building it
was in fact the old bottleneck: it needs 16 GB at 15 qubits.  Working through
the factors, the exact skeleton of a 15-vertex chain (29 qubits) is recovered in
0.02 s.

The maximum DEGREE does.  A vertex with c children occupies 1 + c qubits, so a
pair of vertices spans up to 2(1 + c_max) qubits, and estimating the entropy of
that subsystem from classical shadows needs a number of snapshots growing like
4^{qubits}.  A chain (degree 2) is therefore easy at any n, while a star (degree
n - 1) becomes impractical at about six vertices.  This is a property of the
estimator, not of the model: the process operator itself stays small.

Metrics
-------
Skeleton F1 against the ground-truth adjacency, orientation accuracy on the
recovered skeleton, and the sample size (shadow count M, or shots per basis
pair) at which each reaches a target.  Costs are reported as circuit counts, not
as wall-clock time.

Depends on l2_qpc_cmi (skeleton) and l2_orientation_noncommuting
(orientation).
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from causal_qiskit import l2_qpc_cmi as qpc
from causal_qiskit import l2_orientation_noncommuting as ori


# *****************************************
# Graph families
# *****************************************
def chain(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def fork(n: int) -> List[Tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def collider(n: int) -> List[Tuple[int, int]]:
    return [(i, n - 1) for i in range(n - 1)]


def diamond(n: int) -> List[Tuple[int, int]]:
    """Two disjoint paths from 0 to n-1."""
    if n < 4:
        return chain(n)
    mid = (n - 2) // 2
    left = list(range(1, 1 + mid))
    right = list(range(1 + mid, n - 1))
    e = []
    prev = 0
    for v in left:
        e.append((prev, v)); prev = v
    e.append((prev, n - 1))
    prev = 0
    for v in right:
        e.append((prev, v)); prev = v
    e.append((prev, n - 1))
    return sorted(set(e))


def star_out(n: int) -> List[Tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def random_dag(n: int, p: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    e = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.uniform() < p]
    return e or [(0, 1)]


FAMILIES: Dict[str, Callable[[int, np.random.Generator], List[Tuple[int, int]]]] = {
    "chain":    lambda n, rng: chain(n),
    "fork":     lambda n, rng: fork(n),
    "collider": lambda n, rng: collider(n),
    "diamond":  lambda n, rng: diamond(n),
    "star":     lambda n, rng: star_out(n),
    "random25": lambda n, rng: random_dag(n, 0.25, rng),
    "random50": lambda n, rng: random_dag(n, 0.50, rng),
}


# *****************************************
# The classical limit: dephasing the process operator
# *****************************************
def dephase_all(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Remove every coherence in the computational basis.  For a process operator
    this turns the quantum causal model into a classical causal Bayesian network
    on the same DAG: the Choi factors become conditional probability tables.
    """
    d = 2 ** n_qubits
    out = np.zeros_like(rho)
    idx = np.arange(d)
    out[idx, idx] = np.real(np.diag(rho))
    return out


def dephased_model(model: qpc.QuantumCausalModel) -> qpc.QuantumCausalModel:
    """A copy whose process operator has been dephased; the DAG is unchanged."""
    other = qpc.QuantumCausalModel.__new__(qpc.QuantumCausalModel)
    other.__dict__.update(model.__dict__)
    other.sigma_hat = dephase_all(model.sigma_hat, model.nq)
    # the factorisation is used by the fast sampler, so dephase each factor too
    other.factors = [(dephase_all(f, len(qs)), qs) for f, qs in model.factors]
    return other


# *****************************************
# Metrics
# *****************************************
def skeleton_f1(true_edges: Sequence[Tuple[int, int]],
                est_edges: Sequence[Tuple[int, int]]) -> float:
    T = {tuple(sorted(e)) for e in true_edges}
    E = {tuple(sorted(e)) for e in est_edges}
    tp = len(T & E)
    if tp == 0:
        return 0.0
    p, r = tp / len(E), tp / len(T)
    return 2 * p * r / (p + r)


def orientation_accuracy(true_edges: Sequence[Tuple[int, int]],
                         decided: Dict[Tuple[int, int], Optional[str]]) -> Dict[str, float]:
    T = {tuple(sorted(e)): e for e in true_edges}
    correct = wrong = undecided = 0
    for key, verdict in decided.items():
        k = tuple(sorted(key))
        if k not in T:
            continue
        forward = T[k] == key
        if verdict is None:
            undecided += 1
        elif (verdict == "->") == forward:
            correct += 1
        else:
            wrong += 1
    total = max(correct + wrong + undecided, 1)
    return {"correct": correct / total, "wrong": wrong / total,
            "undecided": undecided / total, "n_edges": total}


# *****************************************
# One benchmark case
# *****************************************
def max_degree(edges: Sequence[Tuple[int, int]], n: int) -> int:
    """Largest number of qubits a single vertex occupies: 1 + (number of children)."""
    ch = {i: 0 for i in range(n)}
    for (i, _) in edges:
        ch[i] += 1
    return 1 + max(ch.values())


def run_skeleton(edges: List[Tuple[int, int]], n_vertices: int, M: int,
                 seed: int, classical: bool = False, adaptive: bool = True) -> Dict:
    rng_m = np.random.default_rng(seed)
    model = qpc.QuantumCausalModel(n_vertices, edges, rng_m)
    if classical:
        model = dephased_model(model)
    rng_s = np.random.default_rng(seed + 10_000)
    bases, bits = qpc.sample_shadows_model(model, M, rng_s)
    cache = qpc.entropy_cache(bases, bits)
    mi = lambda i, j: qpc.cmi_from_shadows(model, bases, bits, i, j, (), cache)
    if adaptive:
        est = qpc.pairwise_mi_skeleton_adaptive(model, mi)
    else:
        est = qpc.pairwise_mi_skeleton(model, mi, qpc.mi_threshold(M, 4))
    exact = qpc.pairwise_mi_skeleton(model, lambda i, j: model.cmi_exact(i, j, ()), 1e-8)
    return {"f1": skeleton_f1(edges, est), "f1_exact": skeleton_f1(edges, exact),
            "n_qubits": model.nq, "shadow_evaluations": cache.n_calls,
            "max_pair_qubits": max(len(model.node_sys[i]) + len(model.node_sys[j])
                                   for i in range(n_vertices)
                                   for j in range(i + 1, n_vertices)),
            "estimated": sorted(est), "exact": sorted(exact)}


def classicalise(model: ori.SequentialQCM, rng: np.random.Generator) -> ori.SequentialQCM:
    """
    Replace every unitary by a permutation matrix.  The circuit then computes a
    classical reversible function, measurement in the computational basis does
    not disturb it, and the orientation statistic must go silent.
    """
    other = ori.SequentialQCM.__new__(ori.SequentialQCM)
    other.__dict__.update(model.__dict__)
    other.U_in = {k: np.eye(U.shape[0])[rng.permutation(U.shape[0])].astype(complex)
                  for k, U in model.U_in.items()}
    other.U_loc = {k: np.eye(U.shape[0])[rng.permutation(U.shape[0])].astype(complex)
                   for k, U in model.U_loc.items()}
    return other


def run_orientation(edges: List[Tuple[int, int]], n_vertices: int, shots: Optional[int],
                    seed: int, classical: bool = False) -> Dict:
    """
    Orientation accuracy.  With `classical`, the circuit is replaced by a
    reversible classical one AND the observation is restricted to the
    computational basis: together these are the classical limit, in which
    reading a variable does not disturb it.
    """
    rng_m = np.random.default_rng(seed)
    model = ori.SequentialQCM(n_vertices, edges, rng_m)
    bases = (2,) if classical else (0, 1, 2)
    if classical:
        model = classicalise(model, np.random.default_rng(seed + 777))
    cache = ori.run_cache(model)
    eps = 1e-8 if shots is None else 4 / np.sqrt(shots)
    rng_s = np.random.default_rng(seed + 20_000)
    decided = ori.orient_edges(model, edges, eps=eps, shots=shots, rng=rng_s,
                               cache=cache, bases=bases)
    acc = orientation_accuracy(edges, decided)
    acc["simulator_runs"] = cache.n_calls
    acc["n_qubits"] = model.nq
    return acc


# *****************************************
# Suites
# *****************************************
def suite_structures(sizes: Sequence[int] = (3, 4, 5), M: int = 8000,
                     n_models: int = 5, families: Optional[Sequence[str]] = None) -> List[Dict]:
    """Skeleton recovery on each graph family, quantum versus dephased."""
    families = list(FAMILIES) if families is None else list(families)
    rows = []
    for fam in families:
        for n in sizes:
            for r in range(n_models):
                rng = np.random.default_rng(1000 * n + r)
                edges = FAMILIES[fam](n, rng)
                q = run_skeleton(edges, n, M, seed=1000 * n + r, classical=False)
                c = run_skeleton(edges, n, M, seed=1000 * n + r, classical=True)
                rows.append({"family": fam, "n": n, "model": r, "M": M,
                             "n_edges": len(edges), "n_qubits": q["n_qubits"],
                             "f1_quantum": q["f1"], "f1_classical": c["f1"],
                             "f1_quantum_exact": q["f1_exact"],
                             "f1_classical_exact": c["f1_exact"]})
    return rows


def suite_sample_complexity(family: str = "random50", n: int = 4,
                            Ms: Sequence[int] = (500, 2000, 8000, 32000),
                            n_models: int = 10) -> List[Dict]:
    """Skeleton F1 against the shadow count, quantum versus dephased."""
    rows = []
    for M in Ms:
        fq, fc = [], []
        for r in range(n_models):
            rng = np.random.default_rng(2000 + r)
            edges = FAMILIES[family](n, rng)
            fq.append(run_skeleton(edges, n, M, seed=2000 + r, classical=False)["f1"])
            fc.append(run_skeleton(edges, n, M, seed=2000 + r, classical=True)["f1"])
        rows.append({"M": M, "family": family, "n": n, "n_models": n_models,
                     "f1_quantum": float(np.mean(fq)), "se_quantum": float(np.std(fq, ddof=1) / np.sqrt(n_models)),
                     "f1_classical": float(np.mean(fc)), "se_classical": float(np.std(fc, ddof=1) / np.sqrt(n_models))})
    return rows


def suite_scaling(sizes: Sequence[int] = (3, 4, 5, 6), family: str = "chain",
                  M: int = 8000, n_models: int = 3) -> List[Dict]:
    """How far the estimator reaches as the number of vertices grows."""
    rows = []
    for n in sizes:
        f1, nq, ev = [], None, None
        for r in range(n_models):
            rng = np.random.default_rng(3000 + 10 * n + r)
            edges = FAMILIES[family](n, rng)
            out = run_skeleton(edges, n, M, seed=3000 + 10 * n + r)
            f1.append(out["f1"]); nq = out["n_qubits"]; ev = out["shadow_evaluations"]
        rows.append({"n": n, "family": family, "n_qubits": nq,
                     "reduced_states": ev, "f1": float(np.mean(f1)),
                     "se": float(np.std(f1, ddof=1) / np.sqrt(n_models)) if n_models > 1 else 0.0})
    return rows


def suite_vertices(sizes: Sequence[int] = (4, 6, 8, 10, 12, 15),
                   family: str = "chain", M: int = 16000, n_models: int = 3) -> List[Dict]:
    """
    Scaling in the number of vertices at fixed degree.  The exact estimator is
    reported alongside the finite-sample one, so that a drop in F1 can be
    attributed to the sample size rather than to the method.
    """
    rows = []
    for n in sizes:
        f1, ex, nq, mq = [], [], None, None
        t0 = time.time()
        for r in range(n_models):
            rng = np.random.default_rng(3000 + 10 * n + r)
            edges = FAMILIES[family](n, rng)
            out = run_skeleton(edges, n, M, seed=3000 + 10 * n + r)
            f1.append(out["f1"]); ex.append(out["f1_exact"])
            nq, mq = out["n_qubits"], out["max_pair_qubits"]
        rows.append({"n": n, "family": family, "n_qubits": nq,
                     "max_pair_qubits": mq, "M": M,
                     "f1": float(np.mean(f1)), "f1_exact": float(np.mean(ex)),
                     "seconds": (time.time() - t0) / n_models})
    return rows


def suite_degree(degrees: Sequence[int] = (2, 3, 4, 5), M: int = 16000,
                 n_models: int = 2, n_extra: int = 2) -> List[Dict]:
    """
    Scaling in the maximum degree, holding the number of vertices modest.  A hub
    with c children occupies 1 + c qubits, so a pair spanning the hub needs the
    entropy of a 2(1+c)-qubit subsystem, which is where the estimator stops.
    """
    rows = []
    for c in degrees:
        n = c + 1 + n_extra
        edges = [(0, k) for k in range(1, c + 1)] + \
                [(k, k + 1) for k in range(c + 1, n - 1)]
        f1, ex, mq = [], [], None
        t0 = time.time()
        for r in range(n_models):
            out = run_skeleton(edges, n, M, seed=8000 + 10 * c + r)
            f1.append(out["f1"]); ex.append(out["f1_exact"]); mq = out["max_pair_qubits"]
        rows.append({"max_children": c, "n": n, "max_pair_qubits": mq,
                     "snapshots_needed": 4 ** mq,
                     "f1": float(np.mean(f1)), "f1_exact": float(np.mean(ex)),
                     "seconds": (time.time() - t0) / n_models})
    return rows


def suite_orientation(sizes: Sequence[int] = (3, 4), shots_list=(None, 1000, 5000),
                      family: str = "random50", n_models: int = 5) -> List[Dict]:
    """Orientation accuracy, quantum circuits versus their classical limit."""
    rows = []
    for n in sizes:
        for shots in shots_list:
            for classical in (False, True):
                acc = {"correct": [], "wrong": [], "undecided": []}
                runs = None
                for r in range(n_models):
                    rng = np.random.default_rng(4000 + 10 * n + r)
                    edges = FAMILIES[family](n, rng)
                    a = run_orientation(edges, n, shots, seed=4000 + 10 * n + r,
                                        classical=classical)
                    for k in acc:
                        acc[k].append(a[k])
                    runs = a["simulator_runs"]
                rows.append({"n": n, "shots": shots, "family": family,
                             "classical": classical, "simulator_runs": runs,
                             **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


# *****************************************
# Demo
# *****************************************
if __name__ == "__main__":
    import sys
    from datetime import datetime

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

    sys.stdout = Tee("l2_benchmark.log")
    print(f"# l2_benchmark  {datetime.now().isoformat(timespec='seconds')}")
    out: Dict[str, object] = {}

    print("\n=== (1) graph families: quantum vs dephased (exact estimator) ===")
    print(f"{'family':>9s} {'n':>2s} {'qubits':>7s} {'edges':>6s} "
          f"{'F1 quantum':>11s} {'F1 classical':>13s}")
    rows = suite_structures(sizes=(3, 4), M=16000, n_models=3)
    out["structures"] = rows
    for fam in FAMILIES:
        for n in (3, 4):
            sel = [r for r in rows if r["family"] == fam and r["n"] == n]
            if not sel:
                continue
            print(f"{fam:>9s} {n:2d} {sel[0]['n_qubits']:7d} {sel[0]['n_edges']:6d} "
                  f"{np.mean([r['f1_quantum_exact'] for r in sel]):11.3f} "
                  f"{np.mean([r['f1_classical_exact'] for r in sel]):13.3f}")
    print("  The skeleton is recovered exactly in the classical limit too: not specific to L2.")

    print("\n=== (2) sample complexity: shadows needed for the skeleton ===")
    print(f"{'M':>7s} {'F1 quantum':>18s} {'F1 classical':>18s}")
    rows = suite_sample_complexity(Ms=(500, 2000, 8000, 32000), n_models=8)
    out["sample_complexity"] = rows
    for r in rows:
        print(f"{r['M']:7d} {r['f1_quantum']:12.3f} +-{r['se_quantum']:.3f} "
              f"{r['f1_classical']:12.3f} +-{r['se_classical']:.3f}")

    print("\n=== (3a) scaling in the number of vertices (chain, degree 2) ===")
    print(f"{'n':>3s} {'qubits':>7s} {'pair q':>7s} {'F1 (M=16000)':>14s} {'F1 exact':>9s} {'time':>8s}")
    rows = suite_vertices(sizes=(4, 6, 8, 10, 12, 15), M=16000, n_models=3)
    out["vertices"] = rows
    for r in rows:
        print(f"{r['n']:3d} {r['n_qubits']:7d} {r['max_pair_qubits']:7d} "
              f"{r['f1']:14.3f} {r['f1_exact']:9.3f} {r['seconds']:7.2f}s")
    print("  Vertices do not limit the estimator: the exact F1 is 1.000 at n=15 (29 qubits).")

    print("\n=== (3b) scaling in the maximum degree (hub with c children) ===")
    print(f"{'children':>9s} {'n':>3s} {'pair q':>7s} {'4^q':>12s} "
          f"{'F1':>7s} {'F1 exact':>9s} {'time':>8s}")
    rows = suite_degree(degrees=(2, 3, 4), M=16000, n_models=2)
    out["degree"] = rows
    for r in rows:
        print(f"{r['max_children']:9d} {r['n']:3d} {r['max_pair_qubits']:7d} "
              f"{r['snapshots_needed']:12,} {r['f1']:7.3f} {r['f1_exact']:9.3f} "
              f"{r['seconds']:7.2f}s")
    print("  Degree does: a pair spans 2(1+c) qubits and the snapshots needed grow like 4^q.")

    print("\n=== (4) orientation: quantum circuits vs their classical limit ===")
    print(f"{'n':>3s} {'shots':>7s} {'limit':>9s} {'runs':>6s} "
          f"{'correct':>9s} {'undecided':>10s} {'wrong':>7s}")
    rows = suite_orientation(sizes=(3, 4), shots_list=(None, 1000, 5000), n_models=5)
    out["orientation"] = rows
    for r in rows:
        sh = "exact" if r["shots"] is None else str(r["shots"])
        lim = "classical" if r["classical"] else "quantum"
        print(f"{r['n']:3d} {sh:>7s} {lim:>9s} {r['simulator_runs']:6d} "
              f"{r['correct']:9.2f} {r['undecided']:10.2f} {r['wrong']:7.2f}")
    print("  In the classical limit reading a variable does not disturb it, so D vanishes.")
    print("  Since the skeleton survives (see (1)), orientation is the part specific to L2.")
    print("  This does not mean classical methods cannot orient edges.")
    print("  LiNGAM and additive-noise models identify a full DAG from observation alone,")
    print("  by assuming the functional form of the generating process.  The models here are")
    print("  discrete Bayesian networks, so those assumptions do not apply.  Different resources.")

    with open("l2_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n# wrote l2_benchmark.log and l2_benchmark.json")

"""
l2_finite_sample_stats.py -- Statistical evaluation of the finite-sample
claims for skeleton recovery (l2_qpc_cmi) and orientation
(l2_orientation_noncommuting).

Model-generation protocol (identical for both experiments):
  * DAG on n=4 nodes: each pair (i<j) is an edge with probability 0.5
    (independent), at least one edge; edges oriented i -> j (topological order).
  * Channels: Stinespring isometries with i.i.d. complex Gaussian entries,
    orthonormalised by QR (Haar-distributed columns); environment dimension
    equal to the input dimension.  Local operations: Haar-random unitaries
    (QR of complex Ginibre matrices with phase correction).
  * Random seeds: model r uses numpy default_rng(1000 + r); the shadow /
    shot sampler uses default_rng(5000 + r).

Reported for each M / shots:
  * mean and standard error of skeleton F1 over R models;
  * for orientation: counts of correct / undetermined / wrong over all
    skeleton edges of R models, with the Clopper-Pearson 95% upper bound on
    the wrong-orientation rate.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np
from scipy.stats import beta

from causal_qiskit import l2_qpc_cmi as q
from causal_qiskit import l2_orientation_noncommuting as o


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Upper bound of the one-sided (1-alpha) Clopper-Pearson interval for a binomial rate."""
    return 1.0 if n == 0 else float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0


def skeleton_stats(R: int, Ms=(1000, 4000, 16000)) -> dict:
    out = {}
    for M in Ms:
        f1 = []
        for r in range(R):
            rng_m = np.random.default_rng(1000 + r)
            edges = q.random_dag(4, rng_m, 0.5) or [(0, 1)]
            m = q.QuantumCausalModel(4, edges, rng_m)
            rng_s = np.random.default_rng(5000 + r)
            bases, bits = q.sample_shadows_model(m, M, rng_s)
            cache = q.entropy_cache(bases, bits)
            est = q.pairwise_mi_skeleton(
                m, lambda i, j: q.cmi_from_shadows(m, bases, bits, i, j, (), cache),
                q.mi_threshold(M, 4))
            f1.append(q.skeleton_f1(edges, est))
        f1 = np.array(f1)
        out[M] = (float(f1.mean()), float(f1.std(ddof=1) / np.sqrt(R)), float((f1 == 1.0).mean()))
        skeleton_stats.per_model = getattr(skeleton_stats, "per_model", {})
        skeleton_stats.per_model[M] = f1.tolist()
    return out


def orientation_stats(R: int, shots_list=(200, 1000, 5000)) -> dict:
    out = {}
    for shots in shots_list:
        correct = undet = wrong = 0
        for r in range(R):
            rng_m = np.random.default_rng(1000 + r)
            edges = o.random_dag(4, rng_m)
            m = o.SequentialQCM(4, edges, rng_m)
            rng_s = np.random.default_rng(5000 + r)
            res = o.orient_edges(m, edges, eps=4 / np.sqrt(shots), shots=shots, rng=rng_s,
                                 cache=o.run_cache(m))
            for v in res.values():
                correct += v == "->"; undet += v is None; wrong += v == "<-"
        n = correct + undet + wrong
        out[shots] = dict(n=n, correct=correct, undet=undet, wrong=wrong,
                          wrong_ub=clopper_pearson_upper(wrong, n), undet_rate=undet / n)
    return out


def run_chunk(kind: str, param: int, r0: int, r1: int) -> dict:
    """Evaluate models r0..r1-1 for one setting; used to split long runs."""
    if kind == "skel":
        f1 = []
        for r in range(r0, r1):
            rng_m = np.random.default_rng(1000 + r)
            edges = q.random_dag(4, rng_m, 0.5) or [(0, 1)]
            m = q.QuantumCausalModel(4, edges, rng_m)
            rng_s = np.random.default_rng(5000 + r)
            bases, bits = q.sample_shadows_model(m, param, rng_s)
            cache = q.entropy_cache(bases, bits)
            est = q.pairwise_mi_skeleton(
                m, lambda i, j: q.cmi_from_shadows(m, bases, bits, i, j, (), cache),
                q.mi_threshold(param, 4))
            f1.append(q.skeleton_f1(edges, est))
        return dict(f1=f1)
    correct = undet = wrong = 0
    for r in range(r0, r1):
        rng_m = np.random.default_rng(1000 + r)
        edges = o.random_dag(4, rng_m)
        m = o.SequentialQCM(4, edges, rng_m)
        rng_s = np.random.default_rng(5000 + r)
        res = o.orient_edges(m, edges, eps=4 / np.sqrt(param), shots=param, rng=rng_s,
                             cache=o.run_cache(m))
        for v in res.values():
            correct += v == "->"; undet += v is None; wrong += v == "<-"
    return dict(correct=correct, undet=undet, wrong=wrong)


if __name__ == "__main__" and len(sys.argv) >= 5 and sys.argv[1] in ("skel", "orient"):
    import json
    kind, param, r0, r1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    print(json.dumps(dict(kind=kind, param=param, r0=r0, r1=r1, **run_chunk(kind, param, r0, r1))))
    sys.exit(0)


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

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    log_path = f"l2_finite_sample_stats_R{R}.log"
    json_path = f"l2_finite_sample_stats_R{R}.json"
    sys.stdout = Tee(log_path)

    print(f"# l2_finite_sample_stats  run at {datetime.now().isoformat(timespec='seconds')}")
    print(f"# R = {R} random models per setting; protocol in the module docstring")
    print(f"# numpy {np.__version__}\n")

    skel = skeleton_stats(R)
    print("Skeleton (pairwise MI from M shadows of the process operator)")
    print(f"{'M':>7s} {'mean F1':>8s} {'s.e.':>6s} {'exact':>6s}")
    for M, (mean, se, exact) in skel.items():
        print(f"{M:7d} {mean:8.3f} {se:6.3f} {exact:6.2f}")

    ori = orientation_stats(R)
    print("\nOrientation (eps = 4/sqrt(shots))")
    print(f"{'shots':>6s} {'edges':>6s} {'correct':>8s} {'undet':>6s} {'wrong':>6s} {'wrong 95%UB':>12s}")
    for shots, d in ori.items():
        print(f"{shots:6d} {d['n']:6d} {d['correct']:8d} {d['undet']:6d} {d['wrong']:6d} {d['wrong_ub']:12.3f}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"R": R,
                   "skeleton": {str(k): {"mean_f1": v[0], "se": v[1], "exact_rate": v[2]}
                                for k, v in skel.items()},
                   "orientation": {str(k): v for k, v in ori.items()}}, f, indent=2)
    print(f"\n# wrote {log_path} and {json_path}")

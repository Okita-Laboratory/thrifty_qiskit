"""
l2_orientation_diagnose.py -- Locate and explain the wrong orientations in
the R=30 evaluation (l2_finite_sample_stats.py).

For every model r and every skeleton edge, this reports
  * the exact (infinite-shot) disturbances D(i->j) and D(j->i);
  * the finite-shot decision at each shots value;
  * whether a wrong decision is a structural exception of Proposition
    "orientation by observation" -- i.e. D(i->j) is itself at or below the
    threshold 4/sqrt(shots), so the true signal is undetectable -- or a
    statistical fluctuation (D(i->j) well above threshold but the sample
    landed on the wrong side).

It also reports, for the offending model, the marginal state of the ancestor's
input qubit (exception (a): maximally mixed) and the purity of the downstream
map's output (exception (b): the downstream channel discards the state).

Writes l2_orientation_diagnose.log and .json.

Run:  python3.10 l2_orientation_diagnose.py 30
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np

from causal_qiskit import l2_orientation_noncommuting as o


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

def diagnose(R: int, shots_list=(200, 1000, 5000)) -> dict:
    records = []
    for r in range(R):
        rng_m = np.random.default_rng(1000 + r)
        edges = o.random_dag(4, rng_m)
        m = o.SequentialQCM(4, edges, rng_m)
        # exact disturbances
        exact = {(i, j): (o.disturbance(m, i, j), o.disturbance(m, j, i)) for i, j in edges}
        # finite-shot decisions, same seeds as the evaluation script
        decisions = {}
        for shots in shots_list:
            rng_s = np.random.default_rng(5000 + r)
            decisions[shots] = o.orient_edges(m, edges, eps=4 / np.sqrt(shots), shots=shots, rng=rng_s)
        for (i, j) in edges:
            dij, dji = exact[(i, j)]
            rec = dict(model=r, edge=[i, j], n_edges=len(edges),
                       D_forward=dij, D_backward=dji,
                       decisions={s: decisions[s][(i, j)] for s in shots_list})
            # marginal state of the ancestor's input qubit (exception (a))
            rho = m.run({})
            p = o.marginal_probs(rho, 2, m.in_q[i], m.nq)
            px = o.marginal_probs(rho, 0, m.in_q[i], m.nq)
            py = o.marginal_probs(rho, 1, m.in_q[i], m.nq)
            bloch = np.array([px[0] - px[1], py[0] - py[1], p[0] - p[1]])
            rec["ancestor_bloch_norm"] = float(np.linalg.norm(bloch))
            records.append(rec)
    return records


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    shots_list = (200, 1000, 5000)
    sys.stdout = Tee("l2_orientation_diagnose.log")
    print(f"# l2_orientation_diagnose  run at {datetime.now().isoformat(timespec='seconds')}")
    print(f"# R = {R}, shots = {shots_list}\n")

    recs = diagnose(R, shots_list)

    wrong = [rc for rc in recs if any(v == "<-" for v in rc["decisions"].values())]
    print(f"total edges: {len(recs)}; edges wrong at some shots: {len(wrong)}\n")
    for rc in wrong:
        i, j = rc["edge"]
        print(f"model {rc['model']:2d}  edge {i}->{j}  (model has {rc['n_edges']} edges)")
        print(f"   exact D({i}->{j}) = {rc['D_forward']:.5f}   exact D({j}->{i}) = {rc['D_backward']:.2e}")
        print(f"   |Bloch| of ancestor input = {rc['ancestor_bloch_norm']:.4f}"
              f"   (0 = maximally mixed: exception (a))")
        for s in shots_list:
            thr = 4 / np.sqrt(s)
            tag = "signal below threshold" if rc["D_forward"] <= thr else "signal above threshold"
            print(f"   shots={s:5d}: decision {str(rc['decisions'][s]):>4s},"
                  f" threshold {thr:.4f}  -> {tag}")
        print()

    # summary over all edges: how many have D_forward below each threshold
    print("edges whose exact forward signal is below the threshold (undetectable):")
    for s in shots_list:
        thr = 4 / np.sqrt(s)
        k = sum(rc["D_forward"] <= thr for rc in recs)
        print(f"   shots={s:5d}: threshold {thr:.4f}, {k}/{len(recs)} edges undetectable")

    D = np.array([rc["D_forward"] for rc in recs])
    print(f"\nexact forward D over all edges: min {D.min():.5f}, median {np.median(D):.4f}, max {D.max():.4f}")

    with open("l2_orientation_diagnose.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2)
    print("\n# wrote l2_orientation_diagnose.log and l2_orientation_diagnose.json")

"""
hw_benchmark.py -- measure on real hardware what the simulator cannot show.

Everything reported so far was measured on a simulator, where a job carries no
queue and a circuit costs a fixed millisecond.  Two of the claims in the papers
only mean something on a device:

  (A) BATCHING.  Submitting the n(n-1)/2 pairs as one PUB in one job, instead of
      one job per pair, does not change the circuit count at all.  It changes the
      number of jobs, and on hardware the per-job overhead -- queueing,
      compilation, result retrieval -- dominates the wall clock.
  (B) TRANSPILE ONCE.  The circuit structure is identical for every pair; only
      the parameter values change.  Transpiling once in the constructor and
      binding parameters afterwards should remove a per-call cost that the
      simulator hides.

This script measures both, plus the fixed per-job overhead by regression, and
writes everything to a log and a JSON file.

It does NOT need much quantum time: the circuits are one qubit and the point is
the overhead, not the computation.  The expensive part is waiting in the queue,
so keep `--pairs-b` small.

Setup
-----
    pip install qiskit qiskit-ibm-runtime
    # once, to store the token from https://quantum.cloud.ibm.com
    python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
               S.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN', overwrite=True)"

Run
---
    # dry run against a local simulator first (no account needed)
    python3 -m experiments.hw_benchmark --backend local --pairs-a 190 --pairs-b 12

    # then the device; (B) submits `pairs-b` separate jobs, so keep it small
    nohup python -u hw_benchmark.py --pairs-a 190 --pairs-b 12 \
          > hw_benchmark.out 2>&1 &

Outputs hw_benchmark.log and hw_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


# **************************************************
# logging
# **************************************************
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

def log(msg: str = "") -> None:
    print(msg)


# **************************************************
# circuit
# **************************************************
def fidelity_circuit(feature_dim: int = 1, reps: int = 1):
    """Compute-uncompute circuit with free parameters x and y."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import z_feature_map

    fm = z_feature_map(feature_dimension=feature_dim, reps=reps)
    x = ParameterVector("x", feature_dim)
    y = ParameterVector("y", feature_dim)
    qc = QuantumCircuit(fm.num_qubits)
    qc.compose(fm.assign_parameters(list(x)), inplace=True)
    qc.compose(fm.assign_parameters(list(y)).inverse(), inplace=True)
    qc.measure_all()
    return qc


def pair_rows(n_points: int, feature_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Upper-triangle parameter rows (n_pairs, 2*feature_dim)."""
    X = rng.uniform(0, np.pi, (n_points, feature_dim))
    iu, ju = np.triu_indices(n_points, k=1)
    return np.column_stack([X[iu], X[ju]])


# **************************************************
# backend
# **************************************************
def get_sampler_and_backend(which: str):
    """Return (sampler, backend, name).  `which` is 'local' or a backend name/'least_busy'."""
    if which == "local":
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler(seed=0), None, "local StatevectorSampler"
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()
    backend = (service.least_busy(operational=True, simulator=False)
               if which == "least_busy" else service.backend(which))
    return SamplerV2(mode=backend), backend, backend.name


def job_metrics(job) -> Dict[str, Optional[float]]:
    """Queue and execution times from the runtime job, when available."""
    out: Dict[str, Optional[float]] = {"queue_s": None, "quantum_s": None}
    try:
        m = job.metrics()
        ts = m.get("timestamps", {}) or {}
        created, running, finished = ts.get("created"), ts.get("running"), ts.get("finished")
        if created and running:
            out["queue_s"] = _delta(created, running)
        usage = m.get("usage", {}) or {}
        out["quantum_s"] = usage.get("quantum_seconds") or m.get("usage_seconds")
    except Exception as e:                                   # local sampler, or no metrics
        out["error"] = str(e)
    return out


def _delta(a: str, b: str) -> float:
    from datetime import datetime as dt
    f = "%Y-%m-%dT%H:%M:%S.%fZ"
    def p(t):
        t = t.replace("+00:00", "Z")
        for fmt in (f, "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.strptime(t, fmt)
            except ValueError:
                continue
        return dt.fromisoformat(t.replace("Z", "+00:00")).replace(tzinfo=None)
    return (p(b) - p(a)).total_seconds()


# **************************************************
# measurements
# **************************************************
def run_batched(sampler, circuit, rows: np.ndarray, shots: int) -> Dict:
    """(A) one job, one PUB, all pairs as parameter values."""
    t0 = time.time()
    job = sampler.run([(circuit, rows)], shots=shots)
    submitted = time.time()
    res = job.result()
    done = time.time()
    bits = res[0].data.meas.array
    f = (bits == 0).all(axis=-1).mean(axis=-1)
    return {"mode": "batched", "n_pairs": int(len(rows)), "n_jobs": 1,
            "submit_s": submitted - t0, "wall_s": done - t0,
            "mean_fidelity": float(np.mean(f)), **job_metrics(job)}


def run_per_pair(sampler, circuit, rows: np.ndarray, shots: int) -> Dict:
    """(B) one job per pair -- the same circuits, many more jobs."""
    t0 = time.time()
    per_job, fids = [], []
    for r in rows:
        tj = time.time()
        job = sampler.run([(circuit, r.reshape(1, -1))], shots=shots)
        res = job.result()
        bits = res[0].data.meas.array
        fids.append(float((bits == 0).all(axis=-1).mean()))
        per_job.append({"wall_s": time.time() - tj, **job_metrics(job)})
    wall = time.time() - t0
    q = [d["queue_s"] for d in per_job if d.get("queue_s") is not None]
    return {"mode": "per_pair", "n_pairs": int(len(rows)), "n_jobs": int(len(rows)),
            "wall_s": wall, "wall_per_job_s": wall / max(len(rows), 1),
            "median_queue_s": float(np.median(q)) if q else None,
            "mean_fidelity": float(np.mean(fids)), "jobs": per_job}


def overhead_regression(sampler, circuit, rows: np.ndarray, sizes: List[int],
                        shots: int) -> Dict:
    """
    Wall-clock time of one job against the number of circuits in it.
    The intercept is the fixed per-job overhead, the slope the marginal cost of
    a circuit.  This is what makes batching worth doing.
    """
    pts = []
    for k in sizes:
        sub = rows[:k]
        t0 = time.time()
        job = sampler.run([(circuit, sub)], shots=shots)
        job.result()
        pts.append({"n_circuits": int(k), "wall_s": time.time() - t0, **job_metrics(job)})
    x = np.array([p["n_circuits"] for p in pts], float)
    y = np.array([p["wall_s"] for p in pts], float)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = intercept = float("nan")
    return {"points": pts, "slope_s_per_circuit": float(slope),
            "intercept_s_per_job": float(intercept)}


def transpile_cost(backend, circuit, n_repeat: int = 5,
                   optimization_level: int = 1) -> Dict:
    """Cost of transpiling once versus once per call."""
    if backend is None:
        return {"skipped": "no backend (local run)"}
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    t0 = time.time()
    isa = pm.run(circuit)
    once = time.time() - t0
    t0 = time.time()
    for _ in range(n_repeat):
        pm.run(circuit)
    repeated = (time.time() - t0) / n_repeat
    return {"transpile_once_s": once, "transpile_per_call_s": repeated,
            "isa_depth": int(isa.depth()), "isa_ops": {k: int(v) for k, v in isa.count_ops().items()}}


# **************************************************
# main
# **************************************************
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="least_busy",
                    help="'local', 'least_busy', or a backend name")
    ap.add_argument("--pairs-a", type=int, default=190,
                    help="pairs in the single batched job (A)")
    ap.add_argument("--pairs-b", type=int, default=12,
                    help="pairs submitted as separate jobs (B); keep small")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--feature-dim", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--skip-per-pair", action="store_true",
                    help="skip (B), which is the part that queues")
    args = ap.parse_args()

    sys.stdout = Tee("hw_benchmark.log")
    log(f"# hw_benchmark  {datetime.now().isoformat(timespec='seconds')}")
    log(f"# args: {vars(args)}")

    sampler, backend, name = get_sampler_and_backend(args.backend)
    log(f"# backend: {name}")

    circuit = fidelity_circuit(args.feature_dim, args.reps)
    if backend is not None:
        from qiskit.transpiler import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        circuit = pm.run(circuit)                            # transpile ONCE
        log(f"# transpiled once: depth {circuit.depth()}, ops {dict(circuit.count_ops())}")

    rng = np.random.default_rng(0)
    n_points = int(np.ceil((1 + np.sqrt(1 + 8 * args.pairs_a)) / 2))
    rows = pair_rows(n_points, args.feature_dim, rng)[: args.pairs_a]
    log(f"# {len(rows)} pairs (n = {n_points} points), {args.shots} shots each\n")

    out: Dict[str, object] = {"backend": name, "args": vars(args)}

    log("== (A) one job, all pairs ==")
    a = run_batched(sampler, circuit, rows, args.shots)
    out["batched"] = a
    log(f"   pairs {a['n_pairs']}, jobs 1, wall {a['wall_s']:.1f} s, "
        f"queue {a.get('queue_s')}, quantum {a.get('quantum_s')}")

    if not args.skip_per_pair:
        log(f"\n== (B) one job per pair ({args.pairs_b} jobs) ==")
        b = run_per_pair(sampler, circuit, rows[: args.pairs_b], args.shots)
        out["per_pair"] = b
        log(f"   pairs {b['n_pairs']}, jobs {b['n_jobs']}, wall {b['wall_s']:.1f} s "
            f"({b['wall_per_job_s']:.1f} s/job), median queue {b.get('median_queue_s')}")
        per_pair_extrapolated = b["wall_per_job_s"] * a["n_pairs"]
        out["extrapolated_per_pair_wall_s"] = per_pair_extrapolated
        log(f"   extrapolated to {a['n_pairs']} pairs: {per_pair_extrapolated:,.0f} s "
            f"vs {a['wall_s']:.1f} s batched "
            f"({per_pair_extrapolated / max(a['wall_s'], 1e-9):,.0f}x)")

    log("\n== per-job overhead by regression ==")
    sizes = [s for s in (1, 5, 20, 60, args.pairs_a) if s <= len(rows)]
    reg = overhead_regression(sampler, circuit, rows, sizes, args.shots)
    out["overhead"] = reg
    for p in reg["points"]:
        log(f"   {p['n_circuits']:5d} circuits: {p['wall_s']:8.2f} s")
    log(f"   fixed per-job overhead {reg['intercept_s_per_job']:.2f} s, "
        f"marginal {reg['slope_s_per_circuit']*1e3:.3f} ms/circuit")

    log("\n== transpile once vs per call ==")
    t = transpile_cost(backend, fidelity_circuit(args.feature_dim, args.reps))
    out["transpile"] = t
    log(f"   {t}")

    with open("hw_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    log("\n# wrote hw_benchmark.log and hw_benchmark.json")


if __name__ == "__main__":
    main()

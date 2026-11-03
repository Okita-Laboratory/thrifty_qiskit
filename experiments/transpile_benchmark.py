"""
transpile_benchmark.py -- where transpiling once actually pays.

The first hardware run measured the transpile cost on a 1-qubit, depth-9 circuit
and found no difference between transpiling once and transpiling on every call
(1.20 ms vs 1.05 ms).  That is the expected answer for a circuit that small: the
pass manager has almost nothing to do.  The question was asked at the wrong
scale.

Transpilation cost is set by the number of qubits, the depth, and how hard the
routing problem is -- not by the number of shots.  This script sweeps those and
reports the crossover, i.e. the size at which transpiling once becomes worth
doing for a Gram matrix of a given number of pairs.

It runs entirely on the classical side.  Transpilation only needs the backend's
coupling map and basis gates, so no job is submitted, no queue is entered and no
quantum time is consumed.  A saved account lets it use the real device
description; without one it falls back to a generic backend of the same size.

Run
---
    python3 -m experiments.transpile_benchmark                       # generic backend
    python3 -m experiments.transpile_benchmark --backend ibm_fez     # real coupling map
    python3 -m experiments.transpile_benchmark --max-qubits 20 --levels 0 1 2 3

Writes transpile_benchmark.log and transpile_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np


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

# *********************************
# circuits
# *********************************
def fidelity_circuit(feature_dim: int, reps: int = 1, entanglement: str = "full"):
    """
    Compute-uncompute circuit for a ZZ feature map: the kernel circuit actually
    used in quantum kernel methods, and the one whose transpilation is hard,
    because the ZZ terms need routing on a device with limited connectivity.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import zz_feature_map

    if feature_dim < 2:                                       # ZZ needs two qubits
        from qiskit.circuit.library import z_feature_map
        fm = z_feature_map(feature_dimension=feature_dim, reps=reps)
    else:
        fm = zz_feature_map(feature_dimension=feature_dim, reps=reps,
                            entanglement=entanglement)
    x = ParameterVector("x", feature_dim)
    y = ParameterVector("y", feature_dim)
    qc = QuantumCircuit(fm.num_qubits)
    qc.compose(fm.assign_parameters(list(x)), inplace=True)
    qc.compose(fm.assign_parameters(list(y)).inverse(), inplace=True)
    qc.measure_all()
    return qc


def get_backend(name: Optional[str], n_qubits: int):
    """Real backend description if available, otherwise a generic one."""
    if name:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            return QiskitRuntimeService().backend(name), name
        except Exception as e:                                # no account, offline
            print(f"  (could not load {name}: {e}; using a generic backend)")
    from qiskit.providers.fake_provider import GenericBackendV2
    return GenericBackendV2(num_qubits=max(n_qubits, 2), seed=0), f"generic-{max(n_qubits,2)}q"


# *********************************
# measurement
# *********************************
def time_transpile(backend, circuit, level: int, n_repeat: int = 5) -> Dict:
    """Seconds per pass-manager run, and the resulting ISA circuit size."""
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=level, backend=backend)
    pm.run(circuit)                                          # warm up caches
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        isa = pm.run(circuit)
        times.append(time.perf_counter() - t0)
    return {"transpile_s": float(np.median(times)),
            "transpile_s_min": float(np.min(times)),
            "isa_depth": int(isa.depth()),
            "isa_2q": int(sum(v for k, v in isa.count_ops().items()
                              if k in ("cz", "cx", "ecr", "rzz"))),
            "n_params": int(len(circuit.parameters))}


def sweep_qubits(backend_name: Optional[str], qubit_list: Sequence[int], reps: int,
                 level: int, n_repeat: int) -> List[Dict]:
    rows = []
    for q in qubit_list:
        backend, bname = get_backend(backend_name, q)
        qc = fidelity_circuit(q, reps)
        r = time_transpile(backend, qc, level, n_repeat)
        rows.append({"n_qubits": q, "reps": reps, "level": level,
                     "backend": bname, "src_depth": int(qc.depth()), **r})
    return rows


def sweep_levels(backend_name: Optional[str], n_qubits: int, reps: int,
                 levels: Sequence[int], n_repeat: int) -> List[Dict]:
    backend, bname = get_backend(backend_name, n_qubits)
    qc = fidelity_circuit(n_qubits, reps)
    return [{"n_qubits": n_qubits, "reps": reps, "level": lv, "backend": bname,
             **time_transpile(backend, qc, lv, n_repeat)} for lv in levels]


def crossover(transpile_s: float, n_pairs: Sequence[int], t_circ: float = 1.21,
              t_job: float = 14.4) -> Dict:
    """
    Fraction of the wall clock spent transpiling, if the pass manager were run
    once per pair instead of once for the whole Gram matrix.

    With the measured device costs, executing P pairs in one job takes
    t_job + P * t_circ seconds.  Transpiling per call adds P * transpile_s.
    """
    rows = []
    for P in n_pairs:
        exec_s = t_job + P * t_circ
        per_call = P * transpile_s
        rows.append({"n_pairs": int(P), "execute_s": exec_s,
                     "transpile_per_call_s": per_call,
                     "overhead_fraction": per_call / (exec_s + per_call)})
    return {"transpile_s": transpile_s, "t_circ": t_circ, "t_job": t_job,
            "rows": rows}


# *********************************
# main
# *********************************
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=None,
                    help="real backend name (e.g. ibm_fez); omit for a generic one")
    ap.add_argument("--qubits", type=int, nargs="+",
                    default=[1, 2, 4, 6, 8, 10, 12, 16])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    sys.stdout = Tee("transpile_benchmark.log")
    print(f"# transpile_benchmark  {datetime.now().isoformat(timespec='seconds')}")
    print(f"# args: {vars(args)}")
    print("# transpilation is classical: no job is submitted, no quantum time used\n")
    out: Dict[str, object] = {"args": vars(args)}

    print("== transpile cost against circuit size (ZZ feature map, level 1) ==")
    print(f"{'qubits':>7s} {'src depth':>10s} {'ISA depth':>10s} {'2q gates':>9s} "
          f"{'transpile':>11s}")
    rows = sweep_qubits(args.backend, args.qubits, args.reps, 1, args.repeat)
    out["by_qubits"] = rows
    for r in rows:
        print(f"{r['n_qubits']:7d} {r['src_depth']:10d} {r['isa_depth']:10d} "
              f"{r['isa_2q']:9d} {r['transpile_s']*1e3:8.1f} ms")

    big = rows[-1]
    print(f"\n== optimisation level (at {big['n_qubits']} qubits) ==")
    print(f"{'level':>6s} {'ISA depth':>10s} {'2q gates':>9s} {'transpile':>11s}")
    lv = sweep_levels(args.backend, big["n_qubits"], args.reps, args.levels, args.repeat)
    out["by_level"] = lv
    for r in lv:
        print(f"{r['level']:6d} {r['isa_depth']:10d} {r['isa_2q']:9d} "
              f"{r['transpile_s']*1e3:8.1f} ms")

    print("\n== what it costs to transpile once per pair ==")
    print("   (execution times from the ibm_fez measurement: 1.21 s/circuit, 14.4 s/job)")
    print(f"   {'qubits':>7s} {'level':>6s} {'transpile':>11s} {'19,900 pairs: exec':>20s} "
          f"{'transpile':>16s} {'share':>7s}")
    cases = []
    for r in rows:
        cases.append((r["n_qubits"], 1, r["transpile_s"]))
    for r in lv:
        if r["level"] != 1:
            cases.append((r["n_qubits"], r["level"], r["transpile_s"]))
    for q, level, ts in cases:
        cr = crossover(ts, [19900])
        row = cr["rows"][0]
        print(f"   {q:7d} {level:6d} {ts*1e3:8.1f} ms {row['execute_s']:17,.0f} s "
              f"{row['transpile_per_call_s']:13,.0f} s {row['overhead_fraction']:7.1%}")
        out.setdefault("crossover", []).append({"n_qubits": q, "level": level, **cr})

    ratio = big["transpile_s"] / rows[0]["transpile_s"]
    worst = max(lv, key=lambda r: r["transpile_s"])
    print(f"\n   transpile cost grows {ratio:,.0f}x from {rows[0]['n_qubits']} to "
          f"{big['n_qubits']} qubits, and another "
          f"{worst['transpile_s']/big['transpile_s']:,.0f}x from optimisation level 1 "
          f"to {worst['level']}.")
    print("   A circuit still executes in 1.21 s, so the ratio matters for large "
          "circuits at high optimisation levels.")

    with open("transpile_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n# wrote transpile_benchmark.log and transpile_benchmark.json")


if __name__ == "__main__":
    main()

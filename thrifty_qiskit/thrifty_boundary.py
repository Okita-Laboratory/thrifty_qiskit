"""
thrifty_boundary.py -- where the reconstruction stops working, and what is
left there.

Everything built on the coefficient representation needs two conditions: the
kernel must be a function of the difference x - y, and its frequency set Omega
must be small.  Both are properties of the encoding, not of the data, so they
can be decided before any large-scale measurement.  This module decides them and
reports the consequence.

Four structures are tested, in increasing generality:

    separable   k(delta) = prod_j f_j(delta_j)              cost 2Ld + 1
    block       k(delta) = prod_b f_b(delta_b)              cost sum_b (2L+1)^{|b|}
    tensor      k(delta) band-limited on Omega              cost (2L+1)^d
    none        neither                                     cost n(n-1)/2

The last row is the important one.  The standard entangling feature maps -- the
ZZ map of Havlicek et al., which is what "quantum kernel" usually means in
practice -- put a term phi(x_i, x_j) = (pi - x_i)(pi - x_j) in the phase, which
is not a function of x - y.  Every reconstruction here refuses it, correctly, and
what remains is Nystrom or direct evaluation.

That is not a gap in the method.  A kernel that is a difference function with a
small Omega is one whose classical description is short, which is exactly the
regime where a classical surrogate exists.  The boundary of the reconstruction is
therefore also the boundary of the region where a quantum advantage could be
claimed, and this module measures where it lies:

  * `classify_feature_map` decides which structure holds and what it costs;
  * `nystrom_quality` measures what the fallback buys, and at what error;
  * `concentration_scaling` measures the other wall -- for entangling maps the
    kernel values collapse towards a constant as the qubit count grows, so the
    shots needed to resolve them grow exponentially even when no reconstruction
    is attempted.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PairFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return X[:, None] if X.ndim == 1 else X


# ************************************************************
# Structure classification
# ************************************************************
def classify_feature_map(pair_fn: PairFn, d: int, reps: int = 1, n: int = 200,
                         gate_freq: float = 2.0, tol: float = 1e-6) -> Dict:
    """
    Decide which reconstruction applies to `pair_fn` and what it costs.

    Returns a record with the structure found, the circuit cost of the cheapest
    valid method, the cost of direct evaluation, and the reason for each refusal.
    Every test is local: its cost does not grow with the sample size.
    """
    from thrifty_qiskit.quantum_gram_kernels import is_translation_invariant
    from thrifty_qiskit.thrifty_exact import ExactShiftInvariantKernel
    from thrifty_qiskit.thrifty_scaling import SeparableExactKernel, method_costs
    from thrifty_qiskit.thrifty_structure import BlockExactKernel, block_cost, connected_blocks, \
        detect_interaction_graph

    out: Dict[str, object] = {"d": d, "reps": reps, "n": n,
                              "direct": n * (n - 1) // 2, "reasons": {}}

    shift = is_translation_invariant(pair_fn, d=d, tol=tol)
    out["shift_invariant"] = bool(shift)
    if not shift:
        out["reasons"]["shift"] = "k は差 x-y の関数ではない"
        out["structure"] = "none"
        out["cost"] = out["direct"]
        return out

    # separable?
    try:
        k = SeparableExactKernel(pair_fn, d, reps, gate_freq).fit()
        out["structure"] = "separable"
        out["cost"] = k.n_circuit_evals
        out["blocks"] = [[j] for j in range(d)]
        return out
    except ValueError as e:
        out["reasons"]["separable"] = str(e)

    # block structure?
    edges, _ = detect_interaction_graph(pair_fn, d, reps, gate_freq, tol=tol)
    blocks = connected_blocks(d, edges)
    if max(len(b) for b in blocks) < d:
        try:
            kb = BlockExactKernel(pair_fn, d, reps, gate_freq, blocks=blocks).fit()
            out["structure"] = "block"
            out["cost"] = kb.n_circuit_evals
            out["blocks"] = blocks
            out["edges"] = sorted(edges)
            return out
        except ValueError as e:
            out["reasons"]["block"] = str(e)

    # full tensor?
    try:
        kt = ExactShiftInvariantKernel(pair_fn, d, reps, gate_freq).fit()
        out["structure"] = "tensor"
        out["cost"] = kt.n_circuit_evals
        out["blocks"] = [list(range(d))]
        return out
    except ValueError as e:
        out["reasons"]["tensor"] = str(e)

    out["structure"] = "none"
    out["cost"] = out["direct"]
    out["blocks"] = [list(range(d))]
    return out


def summarise(rec: Dict) -> str:
    saving = rec["direct"] / max(rec["cost"], 1)
    return (f"structure={rec['structure']:>9s}  cost={rec['cost']:>8,}  "
            f"direct={rec['direct']:>8,}  ratio={saving:>8,.0f}x")


# ************************************************************
# What the fallback buys
# ************************************************************
def nystrom_quality(pair_fn: PairFn, X: np.ndarray, ms: Sequence[int],
                    rng: Optional[np.random.Generator] = None) -> List[Dict]:
    """
    Relative error and circuit cost of the Nystrom approximation, for the case
    where no reconstruction applies.  This is the only remaining reduction that
    does not require structure.
    """
    rng = rng or np.random.default_rng(0)
    X = _2d(X)
    n = len(X)
    exact = pair_fn(X, X)
    rows = []
    for m in ms:
        idx = rng.choice(n, min(m, n), replace=False)
        K_nm = pair_fn(X, X[idx])
        K_mm = K_nm[idx]
        K = K_nm @ np.linalg.pinv(K_mm + 1e-10 * np.eye(len(idx))) @ K_nm.T
        rows.append({"m": int(m), "circuits": int(n * len(idx)),
                     "rel_error": float(np.linalg.norm(K - exact) / np.linalg.norm(exact)),
                     "direct": n * (n - 1) // 2})
    return rows


def concentration_scaling(kernel_for_q: Callable[[int], PairFn], qs: Sequence[int],
                          n: int = 40, rng: Optional[np.random.Generator] = None) -> List[Dict]:
    """
    The second wall.  For entangling feature maps the off-diagonal kernel values
    concentrate towards a constant as the qubit count grows, so distinguishing
    two data points needs a number of shots that grows with the concentration.
    `shots_to_resolve` is the shot count at which the binomial standard error
    matches the spread of the kernel values themselves.
    """
    rng = rng or np.random.default_rng(0)
    rows = []
    for q in qs:
        pf = kernel_for_q(q)
        X = rng.uniform(0, np.pi, (n, q))
        K = pf(X, X)
        off = K[~np.eye(n, dtype=bool)]
        m, v = float(off.mean()), float(off.var())
        rows.append({"q": int(q), "mean": m, "variance": v,
                     "shots_to_resolve": float("inf") if v <= 0 else m * (1 - m) / v})
    return rows


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---------- reference maps ----------
    def z_prod(A, B, reps=1):
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        K = np.ones((A.shape[0], B.shape[0]))
        for j in range(A.shape[1]):
            K *= np.cos(reps * (A[:, j][:, None] - B[:, j][None, :])) ** 2
        return K

    def z_pairs_coupled(A, B, reps=1, pairs=((0, 1), (3, 4))):
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        K = z_prod(A, B, reps)
        for (i, j) in pairs:
            c = (A[:, i] + A[:, j])[:, None] - (B[:, i] + B[:, j])[None, :]
            K *= np.cos(reps * c) ** 2
        return K

    def zz_map_kernel(q: int, reps: int = 1) -> PairFn:
        """Exact fidelity kernel of the standard ZZ feature map, via statevectors."""
        from qiskit.circuit.library import zz_feature_map
        from qiskit.quantum_info import Statevector
        fm = zz_feature_map(feature_dimension=q, reps=reps)

        def pf(A, B):
            A, B = np.atleast_2d(A), np.atleast_2d(B)
            SA = [Statevector(fm.assign_parameters(a)) for a in A]
            SB = [Statevector(fm.assign_parameters(b)) for b in B]
            return np.array([[abs(sa.inner(sb)) ** 2 for sb in SB] for sa in SA])
        return pf

    print("=== structure of the feature map decides the cost ===")
    cases = [
        ("Z 積写像 (d=6)", lambda A, B: z_prod(A, B, 1), 6, 2),
        ("2 対結合 (d=6)", lambda A, B: z_pairs_coupled(A, B, 1), 6, 2),
        ("ZZ 特徴写像 (d=2)", zz_map_kernel(2), 2, 1),
        ("ZZ 特徴写像 (d=4)", zz_map_kernel(4), 4, 1),
    ]
    records = {}
    for name, pf, d, reps in cases:
        rec = classify_feature_map(pf, d, reps, n=200)
        records[name] = rec
        print(f"  {name:22s} {summarise(rec)}")
        for k, v in rec["reasons"].items():
            print(f"      {k:10s}: {v}")

    print("\n=== what remains when no structure applies: Nystrom ===")
    pf = zz_map_kernel(4)
    X = rng.uniform(0, np.pi, (120, 4))
    print(f"  {'m':>4s} {'circuits':>9s} {'direct':>8s} {'relative error':>15s}")
    for row in nystrom_quality(pf, X, [10, 20, 40, 80], np.random.default_rng(1)):
        flag = "  (more than direct evaluation)" if row["circuits"] > row["direct"] else ""
        print(f"  {row['m']:4d} {row['circuits']:9,} {row['direct']:8,} "
              f"{row['rel_error']:15.3f}{flag}")
    print("  Accuracy needs more landmarks, and the circuit count soon exceeds direct evaluation.")

    print("\n=== the other wall: concentration of the ZZ feature map ===")
    print(f"  {'qubits':>7s} {'mean':>8s} {'variance':>10s} {'shots to resolve':>17s}")
    for row in concentration_scaling(lambda q: zz_map_kernel(q), [2, 3, 4, 5, 6], n=30,
                                     rng=np.random.default_rng(2)):
        print(f"  {row['q']:7d} {row['mean']:8.4f} {row['variance']:10.2e} "
              f"{row['shots_to_resolve']:17,.0f}")

    print("\n  The reconstruction works exactly where the kernel has a short classical description.")
    print("  A quantum advantage is claimed outside that region, where only Nystrom and direct")
    print("  evaluation remain and the kernel values concentrate, raising the shots needed.")

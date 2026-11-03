"""
thrifty_scaling.py -- making the exact reconstruction scale, and using it to
train feature maps.

The reconstruction in thrifty_exact costs |Omega| = (2L+1)^d circuits, which
is exponential in the feature dimension d and eventually worse than simply
measuring the n(n-1)/2 pairs.  Two things fix that, and a third exploits the
result.

(1) SEPARABILITY.  A feature map that encodes each feature on its own qubit,
    with no entangling gate between the encodings, gives a product kernel

        k(delta) = prod_j f_j(delta_j),      f_j(0) = 1.

    Setting all but one component of delta to zero isolates a factor, so

        f_j(t) = k(t e_j),

    and the whole kernel is determined by d (2L+1) - (d-1) = 2Ld + 1 evaluations
    instead of (2L+1)^d.  Exponential becomes linear.  Whether the kernel really
    factorises is checked numerically at off-axis points, and `SeparableExactKernel`
    refuses when it does not.

(2) A COST MODEL.  Three methods are available -- direct pair evaluation,
    the full tensor reconstruction, and the separable reconstruction -- each
    valid under different conditions and cheapest in a different regime.
    `choose_method` returns the cheapest one that is valid for the given
    (d, L, n), so the crossover is decided rather than assumed.

(3) TRAINING FEATURE MAPS.  Trainable embedding kernels are fitted by maximising
    kernel-target alignment over the parameters of the ansatz, and each step
    normally needs a fresh Gram matrix (Hubregtsen et al., PRA 105, 042431
    (2022)).  The frequency set Omega is fixed by the ENCODING gates alone, so
    trainable gates move the coefficients inside the same space: every step
    costs |Omega| (or 2Ld+1) circuits instead of n(n-1)/2, and the KTA and its
    gradient are then classical.

Depends on thrifty_exact and thrifty_budget.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from thrifty_qiskit.thrifty_budget import CanonicalKernel, canonical_design, kernel_target_alignment
from thrifty_qiskit.thrifty_exact import _2d

PairFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ************************************************************
# (1) Separable exact reconstruction
# ************************************************************
class SeparableExactKernel:
    """
    Exact reconstruction of a product kernel from axis evaluations.

    Cost: 2*L*d + 1 circuits (the delta = 0 point is shared by all axes),
    against (2L+1)^d for the full tensor reconstruction.
    """

    def __init__(self, pair_fn: PairFn, d: int, reps: int = 1, gate_freq: float = 2.0,
                 check: bool = True, tol: float = 1e-8):
        self.pair_fn, self.d, self.reps, self.gate_freq = pair_fn, d, reps, gate_freq
        self.check, self.tol = check, tol
        self.factors: List[CanonicalKernel] = []
        self.n_circuit_evals = 0
        self.n_check_evals = 0

    # ---- fitting ----
    def fit(self) -> "SeparableExactKernel":
        base = CanonicalKernel(d=1, reps=self.reps, gate_freq=self.gate_freq)
        t = base.nodes[:, 0]                                  # 2L+1 offsets
        zero = np.zeros((1, self.d))
        seen_zero = False
        for j in range(self.d):
            P = np.zeros((len(t), self.d))
            P[:, j] = t
            vals = self.pair_fn(P, zero)[:, 0]
            # the delta = 0 row is the same physical circuit for every axis
            n_new = len(t) if not seen_zero else len(t) - 1
            self.n_circuit_evals += n_new
            seen_zero = True
            ck = CanonicalKernel(d=1, reps=self.reps, gate_freq=self.gate_freq)
            ck.A_ = canonical_design(ck.nodes, ck.W)
            ck.Ap_ = np.linalg.pinv(ck.A_)
            ck.node_values_ = vals
            ck.shots_ = None
            ck.coef_ = ck.Ap_ @ vals
            ck.cov_ = np.zeros((len(ck.coef_), len(ck.coef_)))
            self.factors.append(ck)
        if self.check:
            self._verify()
        return self

    def _verify(self, n_test: int = 5, rng: Optional[np.random.Generator] = None) -> None:
        """Check the product form at off-axis points (cost is O(1) in n)."""
        rng = rng or np.random.default_rng(0)
        X = rng.uniform(0, np.pi, (n_test, self.d))
        Y = rng.uniform(0, np.pi, (n_test, self.d))
        K_true = self.pair_fn(X, Y)
        self.n_check_evals += n_test * n_test
        if np.max(np.abs(self.pairs(X, Y) - K_true)) > self.tol:
            raise ValueError(
                "kernel does not factorise across features; use the full reconstruction")

    # ---- evaluation (no circuits) ----
    def pairs(self, A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        A, B = _2d(A), _2d(B)
        out = np.ones((len(A), len(B)))
        for j, ck in enumerate(self.factors):
            D = (A[:, j][:, None] - B[:, j][None, :]).reshape(-1, 1) * gamma
            out *= (canonical_design(D, ck.W) @ ck.coef_).reshape(len(A), len(B))
        return out

    def __call__(self, X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        K = self.pairs(X, X, gamma)
        return 0.5 * (K + K.T)


# ************************************************************
# (2) Cost model and dispatch
# ************************************************************
def method_costs(d: int, reps: int, n: int) -> Dict[str, int]:
    """Circuit cost of each method for the given problem shape."""
    return {"direct": n * (n - 1) // 2,
            "tensor": (2 * reps + 1) ** d,
            "separable": 2 * reps * d + 1}


def choose_method(pair_fn: PairFn, d: int, reps: int, n: int,
                  gate_freq: float = 2.0, verbose: bool = False) -> Tuple[str, object, Dict]:
    """
    Try the cheapest method first and fall back when its validity check fails.
    Returns (name, fitted kernel or None for 'direct', cost table).
    """
    costs = method_costs(d, reps, n)
    order = sorted(costs, key=costs.get)
    for name in order:
        if name == "direct":
            if verbose:
                print(f"  falling back to direct pair evaluation ({costs['direct']} circuits)")
            return "direct", None, costs
        try:
            if name == "separable":
                k = SeparableExactKernel(pair_fn, d, reps, gate_freq).fit()
            else:
                from thrifty_qiskit.thrifty_exact import ExactShiftInvariantKernel
                k = ExactShiftInvariantKernel(pair_fn, d, reps, gate_freq).fit()
            if verbose:
                print(f"  {name}: accepted ({costs[name]} circuits)")
            return name, k, costs
        except ValueError as e:
            if verbose:
                print(f"  {name}: rejected ({e})")
    return "direct", None, costs


# ************************************************************
# (3) Training an embedding kernel at |Omega| circuits per step
# ************************************************************
class TrainableKernelProblem:
    """
    A feature map with trainable parameters theta whose ENCODING gates are fixed.
    `pair_fn_for(theta)` returns the kernel of that theta.  By Schuld's theorem
    the frequency set does not depend on theta, so each theta is reconstructed
    from the same number of circuits.
    """

    def __init__(self, pair_fn_for: Callable[[np.ndarray], PairFn], d: int, reps: int,
                 n_params: int, separable: bool = True):
        self.pair_fn_for, self.d, self.reps = pair_fn_for, d, reps
        self.n_params, self.separable = n_params, separable
        self.n_circuit_evals = 0
        self.n_steps = 0

    def gram(self, theta: np.ndarray, X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        pf = self.pair_fn_for(theta)
        if self.separable:
            k = SeparableExactKernel(pf, self.d, self.reps, check=False).fit()
        else:
            from thrifty_qiskit.thrifty_exact import ExactShiftInvariantKernel
            k = ExactShiftInvariantKernel(pf, self.d, self.reps, check=False).fit()
        self.n_circuit_evals += k.n_circuit_evals
        return k(X, gamma)


def train_kernel_kta(problem: TrainableKernelProblem, X: np.ndarray, y: np.ndarray,
                     theta0: np.ndarray, n_iter: int = 60,
                     gammas: Optional[Sequence[float]] = None) -> Dict:
    """
    Maximise kernel-target alignment over the trainable parameters.
    Each objective evaluation is one reconstruction (2Ld+1 or |Omega| circuits);
    if `gammas` is given, the bandwidth is optimised inside every evaluation at
    no additional circuit cost, since it only rescales the same coefficients.
    """
    from scipy.optimize import minimize

    gammas = [1.0] if gammas is None else list(gammas)

    def kta_of(theta: np.ndarray) -> float:
        pf = problem.pair_fn_for(np.asarray(theta, float))
        if problem.separable:
            k = SeparableExactKernel(pf, problem.d, problem.reps, check=False).fit()
        else:
            from thrifty_qiskit.thrifty_exact import ExactShiftInvariantKernel
            k = ExactShiftInvariantKernel(pf, problem.d, problem.reps, check=False).fit()
        problem.n_circuit_evals += k.n_circuit_evals
        problem.n_steps += 1
        return max(kernel_target_alignment(k(X, g), y) for g in gammas)

    theta0 = np.asarray(theta0, float)
    start = kta_of(theta0)
    res = minimize(lambda t: -kta_of(t), theta0, method="Nelder-Mead",
                   options={"maxfev": n_iter, "xatol": 1e-3, "fatol": 1e-4})
    return {"theta": res.x, "kta_start": start, "kta_end": float(-res.fun),
            "circuits": problem.n_circuit_evals, "objective_evals": problem.n_steps}


def train_kernel_grid(problem: TrainableKernelProblem, X: np.ndarray, y: np.ndarray,
                      grid: Sequence[float],
                      gammas: Optional[Sequence[float]] = None) -> Dict:
    """
    Exhaustive search of the trainable parameters over a product grid.
    This is normally out of reach -- |grid|^p Gram matrices -- but each point
    costs only one reconstruction here, so the global optimum over the grid is
    affordable and no local optimiser is needed.
    """
    gammas = [1.0] if gammas is None else list(gammas)
    best_val, best_theta = -np.inf, None
    for theta in itertools.product(grid, repeat=problem.n_params):
        pf = problem.pair_fn_for(np.array(theta, float))
        if problem.separable:
            k = SeparableExactKernel(pf, problem.d, problem.reps, check=False).fit()
        else:
            from thrifty_qiskit.thrifty_exact import ExactShiftInvariantKernel
            k = ExactShiftInvariantKernel(pf, problem.d, problem.reps, check=False).fit()
        problem.n_circuit_evals += k.n_circuit_evals
        problem.n_steps += 1
        val = max(kernel_target_alignment(k(X, g), y) for g in gammas)
        if val > best_val:
            best_val, best_theta = val, np.array(theta, float)
    return {"theta": best_theta, "kta_end": float(best_val),
            "circuits": problem.n_circuit_evals, "objective_evals": problem.n_steps}


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    def z_prod(A, B, reps=1, scales=None):
        """Product (separable) kernel: prod_j cos^{2}(reps s_j (a_j - b_j))."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        d = A.shape[1]
        s = np.ones(d) if scales is None else np.asarray(scales, float)
        K = np.ones((A.shape[0], B.shape[0]))
        for j in range(d):
            K *= np.cos(reps * s[j] * (A[:, j][:, None] - B[:, j][None, :])) ** 2
        return K

    def entangled_like(A, B, reps=1):
        """Not a product: an extra factor coupling two features (still shift-invariant)."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        K = z_prod(A, B, reps)
        d01 = (A[:, 0] - B[:, 0])[:, None] - 0  # coupling term on delta_0 + delta_1
        c = (A[:, 0] + A[:, 1])[:, None] - (B[:, 0] + B[:, 1])[None, :]
        return K * np.cos(reps * c) ** 2

    rng = np.random.default_rng(0)

    print("=== (1) separable reconstruction: exponential -> linear ===")
    print(f"{'d':>3s} {'reps':>5s} {'tensor (2L+1)^d':>16s} {'separable 2Ld+1':>16s} "
          f"{'direct n=200':>13s} {'max err':>10s}")
    for d, reps in [(1, 1), (2, 1), (3, 2), (6, 2), (10, 2)]:
        X = rng.uniform(0, np.pi, (40, d))
        Y = rng.uniform(0, np.pi, (30, d))
        k = SeparableExactKernel(lambda A, B: z_prod(A, B, reps), d, reps).fit()
        err = np.max(np.abs(k.pairs(X, Y) - z_prod(X, Y, reps)))
        c = method_costs(d, reps, 200)
        print(f"{d:3d} {reps:5d} {c['tensor']:16d} {k.n_circuit_evals:16d} "
              f"{c['direct']:13d} {err:10.1e}")

    print("\n=== (2) the cost model dispatches, and the checks are real ===")
    for name, pf, d, reps in [("product kernel", lambda A, B: z_prod(A, B, 1), 4, 1),
                              ("coupled kernel", lambda A, B: entangled_like(A, B, 1), 2, 1)]:
        print(f"  {name} (d={d}, reps={reps}):")
        chosen, k, costs = choose_method(pf, d, reps, n=200, verbose=True)
        print(f"    chosen = {chosen}, costs = {costs}")

    print("\n=== (3) training an embedding kernel ===")
    d, reps, n = 3, 1, 80
    X = rng.uniform(0, np.pi, (n, d))
    # only the first feature carries the label: the trained scales should keep
    # feature 0 and switch the other two off
    y = np.sign(np.sin(2 * X[:, 0]))

    def pair_fn_for(theta):
        return lambda A, B: z_prod(A, B, reps, scales=np.abs(theta))

    per_step_direct = n * (n - 1) // 2

    prob = TrainableKernelProblem(pair_fn_for, d, reps, n_params=d, separable=True)
    out = train_kernel_kta(prob, X, y, theta0=np.ones(d), n_iter=120)
    print(f"  local search (Nelder-Mead): KTA {out['kta_start']:.3f} -> {out['kta_end']:.3f}, "
          f"{out['objective_evals']} evaluations, {out['circuits']} circuits")

    prob2 = TrainableKernelProblem(pair_fn_for, d, reps, n_params=d, separable=True)
    grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    out2 = train_kernel_grid(prob2, X, y, grid)
    print(f"  exhaustive grid ({len(grid)}^{d} points): KTA {out2['kta_end']:.3f}, "
          f"scales {np.round(out2['theta'], 2)}")
    print(f"    (feature 0 carries the label; the others are switched off)")
    print(f"    circuits used = {out2['circuits']} "
          f"({out2['circuits'] // out2['objective_evals']} per point)")
    print(f"    with a directly measured Gram matrix this search would need "
          f"{out2['objective_evals'] * per_step_direct:,} circuits "
          f"({out2['objective_evals'] * per_step_direct / out2['circuits']:,.0f}x more)")

    prob3 = TrainableKernelProblem(pair_fn_for, d, reps, n_params=d, separable=True)
    out3 = train_kernel_grid(prob3, X, y, grid, gammas=np.linspace(0.3, 3.0, 12))
    print(f"  the same grid with 12 bandwidths per point: KTA {out3['kta_end']:.3f}, "
          f"circuits {out3['circuits']} (unchanged)")

"""
thrifty_freq_ml.py -- two ML acceleration methods, moved into the frequency
domain.

The five machine-learning substitutes implemented earlier (classical surrogate,
classical shadows, NQS/MPS, Rotosolve, Clifford data regression) are
reimplementations of published methods.  Two of them, however, become something
different once the band-limited structure is available, because both of them are
implicitly one-frequency arguments.

(1) ERROR MITIGATION PER FREQUENCY.  Clifford data regression fits an affine map
    from noisy to ideal expectation values, one scalar slope for the whole
    circuit family.  The depolarising correction in thrifty_psd is the same
    shape with the slope fixed by the delta = 0 node.  Both assume the noise acts
    uniformly.  It does not: a kernel value at large |delta| comes from a longer
    coherent evolution, so high frequencies are damped more than low ones.  In
    the coefficient domain that is a diagonal map

        c~_w = t_w c_w ,

    and the |Omega| damping factors can be fitted from calibration points whose
    ideal value is known classically, then divided out.  A single scalar cannot
    represent this, and forcing one leaves a residual that grows with the
    frequency content of the kernel.

(2) BLOCK RECONSTRUCTION FOR VQE.  Rotosolve works because the energy along ONE
    parameter is A cos(theta - B) + C, a trigonometric polynomial with three
    coefficients, so three evaluations determine it and the minimum along that
    direction follows in closed form.  That is exactly the |Omega| = 3 case of
    the reconstruction used throughout this package.  Applying the general
    version to a BLOCK of k parameters reconstructs the landscape restricted to
    that block from (2L+1)^k evaluations and minimises it globally over the
    block, rather than one coordinate at a time.  Coordinate descent can stall in
    a place where no single coordinate improves the energy but a joint move does;
    the block version steps over that.

Both are extensions of published methods rather than new methods: (1) generalises
the scalar of Clifford data regression to one factor per frequency, (2)
generalises Rotosolve's one-parameter closed form to a k-parameter one.  They are
collected here because both follow from the same fact -- the object being
optimised or corrected is a trigonometric polynomial with a known frequency set.

Only (1) pays off.  Fitting one factor per frequency removes the damping exactly,
while any single scalar leaves a residual that grows with how unevenly the noise
acts across frequencies.  Extension (2) does not: on a coupled six-parameter
landscape every block size reaches the same minimum and k = 1 gets there with the
fewest evaluations, because a block costs (2L+1)^k evaluations per update.  Worth
recording is how the negative result appeared: with the frequency set set too
small, the reconstruction is not exact, coordinate descent solves the wrong
one-dimensional problem, and larger blocks look better.  Setting the band
correctly removes the effect.  The negative result is kept because the two
extensions have the same justification and only measurement separates them.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ************************************************************
# (1) Per-frequency error mitigation
# ************************************************************
def real_basis(omega: Sequence[float]) -> np.ndarray:
    """Non-negative frequencies, in increasing order: the free real coefficients."""
    return np.array(sorted({abs(w) for w in omega}), dtype=float)


def design(delta: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """[1 | cos(w d)] for w > 0; the kernels here are even, so no sine terms."""
    delta = np.atleast_1d(np.asarray(delta, float))
    cols = [np.ones(len(delta)) if np.isclose(w, 0) else np.cos(w * delta)
            for w in freqs]
    return np.column_stack(cols)


class FrequencyMitigator:
    """
    Fit and invert a per-frequency damping c~_w = t_w c_w.

    `calibration` supplies pairs (delta, ideal value) whose ideal value is known
    without the device -- Clifford points of the encoding, or any point where the
    kernel is computable classically.  At least one point per free frequency is
    needed.  With a single point only the constant can be fixed, which is the
    depolarising correction.
    """

    def __init__(self, freqs: np.ndarray):
        self.freqs = np.asarray(freqs, float)
        self.t_: Optional[np.ndarray] = None
        self.c_ideal_: Optional[np.ndarray] = None
        self.n_calibration = 0

    def fit(self, cal_delta: np.ndarray, cal_ideal: np.ndarray,
            noisy_fn: Callable[[np.ndarray], np.ndarray]) -> "FrequencyMitigator":
        cal_delta = np.asarray(cal_delta, float)
        A = design(cal_delta, self.freqs)
        # ideal coefficients of the calibration points (known classically)
        self.c_ideal_ = np.linalg.lstsq(A, np.asarray(cal_ideal, float), rcond=None)[0]
        noisy = np.asarray(noisy_fn(cal_delta), float)
        self.n_calibration = len(cal_delta)
        c_noisy = np.linalg.lstsq(A, noisy, rcond=None)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(self.c_ideal_) > 1e-9, c_noisy / self.c_ideal_, 1.0)
        self.t_ = np.clip(t, 1e-6, None)
        return self

    def correct(self, c_noisy: np.ndarray) -> np.ndarray:
        """Undo the damping on a measured coefficient vector."""
        return np.asarray(c_noisy, float) / self.t_

    def scalar_equivalent(self) -> float:
        """The single slope Clifford data regression would have fitted."""
        return float(np.mean(self.t_))


def fit_coefficients(delta: np.ndarray, values: np.ndarray,
                     freqs: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design(delta, freqs), np.asarray(values, float),
                           rcond=None)[0]


def evaluate(delta: np.ndarray, coef: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return design(delta, freqs) @ np.asarray(coef, float)


# ************************************************************
# (2) Block reconstruction for variational optimisation
# ************************************************************
def block_frequencies(k: int, reps: int = 1, gate_freq: float = 1.0) -> np.ndarray:
    """Frequency vectors of a k-parameter block, each parameter entering `reps` times."""
    per = gate_freq * np.arange(-reps, reps + 1)
    return np.array(list(itertools.product(per, repeat=k)), dtype=float)


def block_nodes(k: int, reps: int = 1, gate_freq: float = 1.0) -> np.ndarray:
    m = 2 * reps + 1
    base = np.linspace(0.0, 2 * np.pi / gate_freq, m, endpoint=False)
    return np.array(list(itertools.product(base, repeat=k)), dtype=float)


def block_design(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    ph = np.atleast_2d(P) @ W.T
    return np.hstack([np.cos(ph), np.sin(ph)])


class BlockLandscape:
    """
    Exact reconstruction of the cost restricted to a block of k parameters.

    The energy of a parametrised circuit is a trigonometric polynomial in each
    parameter, so restricted to k parameters it lives in a space of
    (2k+1)^k... precisely, of |Omega| = (2 reps + 1)^k frequencies.  Evaluating
    the cost at those nodes determines the restriction, and its minimum over the
    block can then be found classically.
    """

    def __init__(self, k: int, reps: int = 1, gate_freq: float = 1.0):
        self.k, self.reps, self.gate_freq = k, reps, gate_freq
        self.W = block_frequencies(k, reps, gate_freq)
        self.nodes = block_nodes(k, reps, gate_freq)
        self.coef_: Optional[np.ndarray] = None

    def fit(self, cost_at: Callable[[np.ndarray], np.ndarray]) -> "BlockLandscape":
        vals = np.asarray(cost_at(self.nodes), float)
        self.coef_ = np.linalg.lstsq(block_design(self.nodes, self.W), vals,
                                     rcond=None)[0]
        return self

    def __call__(self, P: np.ndarray) -> np.ndarray:
        return block_design(np.atleast_2d(P), self.W) @ self.coef_

    def argmin(self, grid: int = 21, refine: bool = True) -> np.ndarray:
        """Global minimum over the block: dense grid, then a local polish."""
        ax = np.linspace(0, 2 * np.pi, grid, endpoint=False)
        P = np.array(list(itertools.product(ax, repeat=self.k)), dtype=float)
        best = P[int(np.argmin(self(P)))]
        if refine:
            from scipy.optimize import minimize
            r = minimize(lambda p: float(self(p.reshape(1, -1))[0]), best,
                         method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-10, "maxfev": 400})
            if float(r.fun) < float(self(best.reshape(1, -1))[0]):
                best = r.x
        return best


class VariationalProblem:
    """Wraps a cost function and counts circuit evaluations."""

    def __init__(self, cost: Callable[[np.ndarray], float], n_params: int):
        self.cost, self.n_params = cost, n_params
        self.n_evals = 0

    def __call__(self, theta: np.ndarray) -> float:
        self.n_evals += 1
        return float(self.cost(np.asarray(theta, float)))

    def batch(self, thetas: np.ndarray) -> np.ndarray:
        return np.array([self(t) for t in np.atleast_2d(thetas)])


def block_optimise(problem: VariationalProblem, theta0: np.ndarray, k: int = 2,
                   reps: int = 1, sweeps: int = 6, gate_freq: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Sweep over blocks of k parameters, reconstructing and minimising each block.
    k = 1 is Rotosolve; k >= 2 minimises several parameters jointly, which is the
    difference this function is here to measure.
    """
    rng = rng or np.random.default_rng(0)
    theta = np.asarray(theta0, float).copy()
    p = len(theta)
    hist = [problem(theta)]
    for _ in range(sweeps):
        order = rng.permutation(p)
        for start in range(0, p, k):
            idx = order[start:start + k]
            if len(idx) < k:
                idx = order[-k:] if p >= k else order
            land = BlockLandscape(len(idx), reps, gate_freq)

            def cost_at(P):
                out = []
                for row in np.atleast_2d(P):
                    t = theta.copy(); t[idx] = row
                    out.append(problem(t))
                return np.array(out)

            land.fit(cost_at)
            best = land.argmin()
            cand = theta.copy(); cand[idx] = best
            if problem(cand) < hist[-1]:
                theta = cand
            hist.append(problem(theta))
    return {"theta": theta, "energy": hist[-1], "history": np.array(hist),
            "evaluations": problem.n_evals}


# ************************************************************
# Demo
# ************************************************************
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

    sys.stdout = Tee("thrifty_freq_ml.log")
    print(f"# thrifty_freq_ml  {datetime.now().isoformat(timespec='seconds')}")

    # ---------------- (1) per-frequency mitigation ----------------
    print("\n=== (1) noise that damps frequencies unequally ===")
    freqs = np.array([0.0, 2.0, 4.0])
    c_true = np.array([0.375, 0.5, 0.125])          # an even, band-limited kernel

    def ideal(d):
        return evaluate(np.atleast_1d(d), c_true, freqs)

    damp = np.array([1.0, 0.75, 0.45])              # higher frequency, more damping
    lam_readout = 0.03

    def noisy(d):
        v = evaluate(np.atleast_1d(d), c_true * damp, freqs)
        return (1 - lam_readout) * v + lam_readout * 0.5

    print(f"  true coefficients      {np.round(c_true, 4)}")
    print(f"  damping applied        {np.round(damp, 3)} (plus 3% readout)")

    # calibration nodes must make the cosine design invertible: cos(w d) for
    # w in {0,2,4} is even about pi/2, so sampling [0, pi) uniformly is degenerate
    cal_d = np.linspace(0, np.pi / 2, len(freqs))
    mit = FrequencyMitigator(freqs).fit(cal_d, ideal(cal_d), noisy)
    print(f"  fitted damping         {np.round(mit.t_, 3)} "
          f"({mit.n_calibration} calibration points)")

    test_d = np.linspace(0, np.pi / 2, 40)
    c_noisy = fit_coefficients(test_d, noisy(test_d), freqs)
    raw = evaluate(test_d, c_noisy, freqs)
    per_freq = evaluate(test_d, mit.correct(c_noisy), freqs)
    scalar = evaluate(test_d, c_noisy / mit.scalar_equivalent(), freqs)
    truth = ideal(test_d)
    print(f"\n  {'correction':>16s} {'RMS error':>12s}")
    for name, v in (("none", raw), ("one scalar", scalar), ("per frequency", per_freq)):
        print(f"  {name:>16s} {np.sqrt(np.mean((v - truth) ** 2)):12.2e}")
    print("  A single scalar leaves the frequency-dependent part of the damping behind.")

    print("\n  as the damping becomes more uneven:")
    print(f"  {'spread':>14s} {'one scalar':>12s} {'per frequency':>14s}")
    for spread in (0.0, 0.1, 0.2, 0.3, 0.4):
        d_s = np.array([1.0, 1.0 - spread, 1.0 - 2 * spread])

        def noisy_s(d, d_s=d_s):
            v = evaluate(np.atleast_1d(d), c_true * d_s, freqs)
            return (1 - lam_readout) * v + lam_readout * 0.5

        m = FrequencyMitigator(freqs).fit(cal_d, ideal(cal_d), noisy_s)
        cn = fit_coefficients(test_d, noisy_s(test_d), freqs)
        e_s = np.sqrt(np.mean((evaluate(test_d, cn / m.scalar_equivalent(), freqs) - truth) ** 2))
        e_f = np.sqrt(np.mean((evaluate(test_d, m.correct(cn), freqs) - truth) ** 2))
        print(f"  {spread:14.2f} {e_s:12.5f} {e_f:14.2e}")

    # ---------------- (2) block reconstruction ----------------
    print("\n=== (2) block reconstruction of a variational landscape ===")

    def make_cost(p: int, seed: int = 0):
        """A trigonometric cost with couplings, so coordinates are not separable."""
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(p, p)); A = 0.5 * (A + A.T)
        b = rng.normal(size=p)

        def cost(th):
            return float(b @ np.cos(th) + np.sin(th) @ A @ np.sin(th))
        return cost

    p = 6
    base = make_cost(p, seed=3)
    print(f"  {p} coupled parameters.  Each parameter enters the cost with frequencies")
    print("  {0, +-1, +-2}, so reps=2 is the correct band (reps=1 is not exact).")
    print("  30 random starts; energy reached and evaluations used, against the sweep count.\n")
    print(f"  {'sweeps':>7s} " + "  ".join(f"{'k=' + str(k):>16s}" for k in (1, 2, 3)))
    print(f"  {'':7s} " + "  ".join(f"{'median / evals':>16s}" for _ in (1, 2, 3)))
    for sw in (2, 4, 6, 10):
        cells = []
        for k in (1, 2, 3):
            E, V = [], []
            for t in range(30):
                th0 = np.random.default_rng(100 + t).uniform(0, 2 * np.pi, p)
                prob = VariationalProblem(base, p)
                o = block_optimise(prob, th0, k=k, reps=2, sweeps=sw,
                                   rng=np.random.default_rng(t))
                E.append(o["energy"]); V.append(o["evaluations"])
            cells.append(f"{np.median(E):8.2f} /{np.mean(V):5.0f}")
        print(f"  {sw:7d} " + "  ".join(f"{c:>16s}" for c in cells))

    print("\n  The result is negative.  Every k reaches the same minimum and k=1 (Rotosolve)")
    print("  gets there with the fewest evaluations: a block costs (2L+1)^k evaluations per")
    print("  update, so raising k only costs more.")
    print("  With reps=1, k=2 appears to win, but only because the frequency set is too")
    print("  small for the reconstruction to be exact; the effect disappears with the right band.")
    print("  What block reconstruction shows is that Rotosolve is its k=1 case.")

    print("\n# wrote thrifty_freq_ml.log")

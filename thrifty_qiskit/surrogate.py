"""
surrogate.py  --  Classical surrogates of quantum ML models.

Idea (Schuld, Sweke, Meyer 2021; Schreiber, Eisert, Meyer 2023):
a parametrised quantum circuit with Pauli-rotation data encoding computes
    f(x) = sum_{w in Omega} c_w exp(i w . x),
where the frequency set Omega is fixed by the ENCODING gates alone
(trainable gates only change the coefficients c_w).  Once |Omega| is known
and small, f can be fitted classically from N >= |Omega| samples and then
evaluated with no circuit at all.  If Omega is exponentially large, or the
encoding is not a Pauli rotation in x (e.g. the (pi-x)(pi-y) term of the
default ZZ feature map), the fit does not converge: that is the regime
where a classical surrogate is not available.

Data: generated here by running the circuit (Statevector = exact, or a
Sampler with shots = noisy).  No external dataset needed.

Contents
--------
frequency_set(...)      enumerate Omega for `reps` re-uploads of P(2x)-type gates
TrigSurrogate           least-squares fit of the trigonometric polynomial on Omega
RFFSurrogate            random Fourier features + ridge (when Omega is too large)
QuantumModel            circuit output <Z_0> as a function of x (with optional shots)
QuantumKernelModel      fidelity kernel k(x, y) as a function of (x, y)
generate_dataset(...)   sample inputs, evaluate the quantum model
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


# ************************************************************
# Frequency spectrum of Pauli-rotation encodings
# ************************************************************
def frequency_set(feature_dim: int, reps: int = 1, gate_freqs: Sequence[float] = (0.0, 2.0),
                  max_terms: Optional[int] = None) -> np.ndarray:
    """
    Omega for a circuit that encodes each feature `reps` times with a gate
    whose generator eigenvalues give amplitude frequencies `gate_freqs`
    (P(2x) = diag(1, e^{2ix}) -> {0, 2}).

    Expectation-value frequencies per feature are differences of sums:
        {sum_l a_l - sum_l b_l : a_l, b_l in gate_freqs}  (l = 1..reps)
    Omega is the Cartesian product over features -> |Omega| = (per-feature)^d.
    Returns an (|Omega|, d) array (only the half with w >= 0 lexicographically
    plus the zero vector would suffice for real f; we keep all for simplicity).
    """
    sums = {0.0}
    for _ in range(reps):
        sums = {s + g for s in sums for g in gate_freqs}
    per_feat = sorted({a - b for a in sums for b in sums})
    omega = np.array(list(itertools.product(per_feat, repeat=feature_dim)))
    if max_terms is not None and len(omega) > max_terms:
        raise ValueError(f"|Omega| = {len(omega)} exceeds max_terms={max_terms}: "
                         "surrogate not practical; consider RFFSurrogate")
    return omega


def _design_matrix(X: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Real feature matrix [cos(w.x), sin(w.x)] over w in Omega (dropping duplicates)."""
    ph = X @ omega.T                                   # (N, |Omega|)
    return np.hstack([np.cos(ph), np.sin(ph)])


# ************************************************************
# Surrogates
# ************************************************************
class TrigSurrogate:
    """Exact-class surrogate: least squares on the known frequency set."""

    def __init__(self, omega: np.ndarray, ridge: float = 1e-10):
        self.omega, self.ridge = np.asarray(omega, float), ridge
        self.coef_ = None

    @property
    def n_params(self) -> int:
        return 2 * len(self.omega)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrigSurrogate":
        A = _design_matrix(np.atleast_2d(X), self.omega)
        AtA = A.T @ A + self.ridge * np.eye(A.shape[1])
        self.coef_ = np.linalg.solve(AtA, A.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _design_matrix(np.atleast_2d(X), self.omega) @ self.coef_

    def spectrum(self) -> dict:
        """|c_w| per frequency vector (for inspection)."""
        m = len(self.omega)
        amp = np.sqrt(self.coef_[:m] ** 2 + self.coef_[m:] ** 2)
        return {tuple(w): float(a) for w, a in zip(self.omega, amp)}


class RFFSurrogate:
    """
    Random Fourier features + ridge.  Use when Omega is too large to enumerate:
    frequencies are sampled from Omega if given, else from N(0, sigma^2 I).
    """

    def __init__(self, n_features: int = 500, omega: Optional[np.ndarray] = None,
                 sigma: float = 2.0, ridge: float = 1e-6, rng: Optional[np.random.Generator] = None):
        self.n_features, self.omega, self.sigma, self.ridge = n_features, omega, sigma, ridge
        self.rng = rng or np.random.default_rng(0)
        self.W = self.b = self.coef_ = None

    def _feats(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W.T + self.b)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFFSurrogate":
        X = np.atleast_2d(X)
        d = X.shape[1]
        if self.omega is not None:
            idx = self.rng.integers(0, len(self.omega), self.n_features)
            self.W = np.asarray(self.omega, float)[idx]
        else:
            self.W = self.rng.normal(0, self.sigma, (self.n_features, d))
        self.b = self.rng.uniform(0, 2 * np.pi, self.n_features)
        A = self._feats(X)
        self.coef_ = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._feats(np.atleast_2d(X)) @ self.coef_


# ************************************************************
# Quantum models (data generators)
# ************************************************************
class QuantumModel:
    """
    f(x) = <psi(x; theta)| O |psi(x; theta)>,  psi = ansatz(theta) . feature_map(x) |0>.

    feature_map : qiskit circuit whose parameters are the input x (feature_dim of them)
    ansatz      : optional trainable circuit; theta fixed at construction (random or given)
    observable  : SparsePauliOp; default Z on qubit 0
    shots       : None -> exact Statevector; int -> sampled with StatevectorSampler
    """

    def __init__(self, feature_map, ansatz=None, theta: Optional[np.ndarray] = None,
                 observable=None, shots: Optional[int] = None, seed: int = 0):
        from qiskit.quantum_info import SparsePauliOp

        self.fm = feature_map
        self.d = feature_map.num_parameters
        self.nq = feature_map.num_qubits
        rng = np.random.default_rng(seed)
        if ansatz is not None:
            theta = rng.uniform(0, 2 * np.pi, ansatz.num_parameters) if theta is None else theta
            self.circuit = feature_map.compose(ansatz.assign_parameters(theta))
        else:
            self.circuit = feature_map
        self.observable = observable or SparsePauliOp.from_sparse_list(
            [("Z", [0], 1.0)], num_qubits=self.nq)
        self.shots, self.seed = shots, seed
        self.n_circuit_evals = 0

    def _exact(self, X: np.ndarray) -> np.ndarray:
        from qiskit.quantum_info import Statevector
        out = np.empty(len(X))
        for i, x in enumerate(X):
            sv = Statevector(self.circuit.assign_parameters(x))
            out[i] = np.real(sv.expectation_value(self.observable))
        return out

    def _shots(self, X: np.ndarray) -> np.ndarray:
        from qiskit.primitives import StatevectorEstimator
        est = StatevectorEstimator(default_precision=1 / np.sqrt(self.shots), seed=self.seed)
        res = est.run([(self.circuit, self.observable, X)]).result()[0]
        return np.asarray(res.data.evs, float)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        self.n_circuit_evals += len(X)
        return self._exact(X) if self.shots is None else self._shots(X)


class QuantumKernelModel:
    """k(x, y) = |<psi(y)|psi(x)>|^2 as a function of the concatenated input (x, y)."""

    def __init__(self, feature_map, shots: Optional[int] = None, seed: int = 0):
        self.fm, self.d, self.shots, self.seed = feature_map, feature_map.num_parameters, shots, seed
        self.n_circuit_evals = 0

    def __call__(self, XY: np.ndarray) -> np.ndarray:
        from qiskit.quantum_info import Statevector
        XY = np.atleast_2d(np.asarray(XY, float))
        self.n_circuit_evals += len(XY)
        out = np.empty(len(XY))
        rng = np.random.default_rng(self.seed)
        for i, row in enumerate(XY):
            a = Statevector(self.fm.assign_parameters(row[: self.d]))
            b = Statevector(self.fm.assign_parameters(row[self.d:]))
            p = float(abs(a.inner(b)) ** 2)
            out[i] = p if self.shots is None else rng.binomial(self.shots, p) / self.shots
        return out


def generate_dataset(model: Callable[[np.ndarray], np.ndarray], n: int, dim: int,
                     low: float = 0.0, high: float = np.pi,
                     rng: Optional[np.random.Generator] = None):
    """Uniform inputs in [low, high]^dim and the model's outputs."""
    rng = rng or np.random.default_rng(0)
    X = rng.uniform(low, high, (n, dim))
    return X, model(X)


# ************************************************************
# Demo (python3 -m thrifty_qiskit.surrogate)
# ************************************************************
if __name__ == "__main__":
    from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map

    rng = np.random.default_rng(0)
    print("=== (a) Z feature map, d=2, reps=2, entangling ansatz: surrogate is exact ===")
    d, reps = 2, 2
    qm = QuantumModel(z_feature_map(d, reps=reps), real_amplitudes(d, reps=2))
    omega = frequency_set(d, reps)
    Xtr, ytr = generate_dataset(qm, 4 * len(omega), d, rng=rng)
    Xte, yte = generate_dataset(qm, 500, d, rng=rng)
    s = TrigSurrogate(omega).fit(Xtr, ytr)
    print(f"|Omega|={len(omega)}, params={s.n_params}, train N={len(Xtr)}, "
          f"test max err={np.max(np.abs(s.predict(Xte) - yte)):.2e}, "
          f"circuits used={qm.n_circuit_evals - 500}")

    print("=== (b) same with shot noise: error floor ~ 1/sqrt(shots) ===")
    qm_s = QuantumModel(z_feature_map(d, reps=reps), real_amplitudes(d, reps=2), shots=2048)
    Xtr, ytr = generate_dataset(qm_s, 8 * len(omega), d, rng=rng)
    s = TrigSurrogate(omega).fit(Xtr, ytr)
    print(f"test RMSE={np.sqrt(np.mean((s.predict(Xte) - yte) ** 2)):.3f} (1/sqrt(2048)={1/np.sqrt(2048):.3f})")

    print("=== (c) |Omega| growth with reps and d ===")
    for dd, rr in [(1, 1), (1, 4), (2, 2), (3, 2), (4, 2), (6, 2)]:
        print(f"  d={dd} reps={rr}: |Omega|={len(frequency_set(dd, rr))}")

    print("=== (d) default ZZ feature map: (pi-x)(pi-y) term is not band-limited ===")
    qm_zz = QuantumModel(zz_feature_map(2, reps=1), real_amplitudes(2, reps=1))
    Xtr, ytr = generate_dataset(qm_zz, 1500, 2, rng=rng)
    Xte, yte = generate_dataset(qm_zz, 500, 2, rng=rng)
    for L in [1, 2, 4, 8]:
        om = frequency_set(2, L)
        if len(om) > 300:
            break
        s = TrigSurrogate(om).fit(Xtr, ytr)
        print(f"  assumed reps={L}: |Omega|={len(om):4d}, test max err={np.max(np.abs(s.predict(Xte) - yte)):.3f}")

    print("=== (e) kernel surrogate: k(x,y) for Z map = cos^2(x-y) ===")
    km = QuantumKernelModel(z_feature_map(1, reps=1))
    om = frequency_set(2, 1)  # frequencies on (x, y)
    XY, k = generate_dataset(km, 60, 2, rng=rng)
    s = TrigSurrogate(om).fit(XY, k)
    XYte, kte = generate_dataset(km, 300, 2, rng=rng)
    print(f"  test max err={np.max(np.abs(s.predict(XYte) - kte)):.2e}")
    big = {tuple(int(v) for v in w): round(a, 4) for w, a in s.spectrum().items() if a > 1e-3}
    print(f"  non-zero frequencies: {big}")

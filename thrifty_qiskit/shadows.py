"""
shadows.py -- Classical shadows + ML prediction of ground-state properties.

Two layers:

(1) Classical shadows (Huang, Kueng, Preskill 2020)
    M snapshots of random single-qubit Pauli-basis measurements of a state rho.
    Any k-local Pauli expectation is estimated from the SAME snapshots with
    variance <= 3^k / M, so one measurement budget serves many observables.

(2) ML for many-body properties (Huang et al., Science 2022)
    Train: for N random Hamiltonians H(a), prepare the ground state (here: exact
    diagonalisation stands in for the quantum computer), take M shadows, estimate
    a property y(a) (e.g. nearest-neighbour ZZ correlations).
    Predict y(a') for unseen a' with kernel ridge regression -- no quantum
    computation for new Hamiltonians.  This is the closest analogue of
    DFT -> MLIP: expensive device runs on few instances, cheap model afterwards.

Data: generated here (random couplings, exact ground states, sampled shadows).
Pure numpy; n <= 10 qubits is comfortable.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# single-qubit basis-change unitaries: measure X -> apply H, Y -> apply H S^dag, Z -> I
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]])
_U = {0: _H, 1: _H @ _SDG, 2: np.eye(2)}          # basis index 0=X, 1=Y (H S^dag), 2=Z
_PAULI = {0: np.array([[0, 1], [1, 0]]), 1: np.array([[0, -1j], [1j, 0]]), 2: np.diag([1, -1])}


# ************************************************************
# Hamiltonians and exact ground states (stand-in for the quantum device)
# ************************************************************
def _op_on(op: np.ndarray, i: int, n: int) -> np.ndarray:
    mats = [np.eye(2)] * n
    mats[i] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def tfim_hamiltonian(J: np.ndarray, h: np.ndarray, periodic: bool = False) -> np.ndarray:
    """H = -sum_i J_i Z_i Z_{i+1} - sum_i h_i X_i   (dense, 2^n x 2^n)."""
    n = len(h)
    Z, X = _PAULI[2], _PAULI[0]
    H = np.zeros((2 ** n, 2 ** n))
    bonds = n if periodic else n - 1
    for i in range(bonds):
        H -= J[i] * _op_on(Z, i, n) @ _op_on(Z, (i + 1) % n, n)
    for i in range(n):
        H -= h[i] * _op_on(X, i, n)
    return H


def ground_state(H: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(H)
    return v[:, 0].astype(complex)


def pauli_expectation(psi: np.ndarray, paulis: Sequence[Tuple[int, int]], n: int) -> float:
    """Exact <psi| prod_{(i,p)} P_i |psi>; paulis = [(qubit, 0/1/2), ...]."""
    op = np.eye(2 ** n, dtype=complex)
    for i, p in paulis:
        op = op @ _op_on(_PAULI[p], i, n)
    return float(np.real(psi.conj() @ op @ psi))


# ************************************************************
# (1) Classical shadows
# ************************************************************
def sample_shadows(psi: np.ndarray, M: int, rng: Optional[np.random.Generator] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    M random Pauli-basis measurements of |psi>.
    Returns bases (M, n) in {0,1,2} and bits (M, n) in {0,1}.
    """
    rng = rng or np.random.default_rng(0)
    n = int(np.log2(len(psi)))
    bases = rng.integers(0, 3, (M, n))
    bits = np.empty((M, n), dtype=np.int8)
    for m in range(M):
        U = _U[bases[m, 0]]
        for i in range(1, n):
            U = np.kron(U, _U[bases[m, i]])
        probs = np.abs(U @ psi) ** 2
        probs /= probs.sum()
        idx = rng.choice(len(probs), p=probs)
        bits[m] = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
    return bases, bits


class ShadowEstimator:
    """Estimate Pauli expectations from one set of shadows (median of means)."""

    def __init__(self, bases: np.ndarray, bits: np.ndarray, n_batches: int = 1):
        self.bases, self.bits, self.n_batches = bases, bits, n_batches
        self.M, self.n = bases.shape

    def _per_snapshot(self, paulis: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Single-snapshot estimator of a Pauli string: prod_i 3*(-1)^bit if basis matches else 0."""
        est = np.ones(self.M)
        for i, p in paulis:
            match = self.bases[:, i] == p
            est *= np.where(match, 3.0 * (1 - 2 * self.bits[:, i]), 0.0)
        return est

    def expectation(self, paulis: Sequence[Tuple[int, int]]) -> float:
        est = self._per_snapshot(paulis)
        if self.n_batches <= 1:
            return float(est.mean())
        return float(np.median([b.mean() for b in np.array_split(est, self.n_batches)]))

    def expectations(self, pauli_list) -> np.ndarray:
        return np.array([self.expectation(p) for p in pauli_list])


# ************************************************************
# (2) ML for ground-state properties
# ************************************************************
def random_tfim_params(n: int, N: int, rng: np.random.Generator,
                       J_range=(0.0, 2.0), h_range=(0.0, 2.0)) -> np.ndarray:
    """N parameter vectors a = (J_1..J_{n-1}, h_1..h_n)."""
    J = rng.uniform(*J_range, (N, n - 1))
    h = rng.uniform(*h_range, (N, n))
    return np.hstack([J, h])


def split_params(a: np.ndarray, n: int):
    return a[: n - 1], a[n - 1:]


def nn_zz_observables(n: int):
    """Nearest-neighbour ZZ correlators as Pauli lists."""
    return [[(i, 2), (i + 1, 2)] for i in range(n - 1)]


class PropertyDataset:
    """
    For each parameter vector a: ground state -> shadows -> estimated targets.
    `exact_targets` are kept for evaluation only (they need the full state).
    """

    def __init__(self, n: int, observables, M_shadows: int, rng: np.random.Generator):
        self.n, self.obs, self.M, self.rng = n, observables, M_shadows, rng
        self.n_state_preparations = 0
        self.n_snapshots = 0

    def build(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Y_shadow, Y_exact = [], []
        for a in A:
            J, h = split_params(a, self.n)
            psi = ground_state(tfim_hamiltonian(J, h))
            self.n_state_preparations += 1
            bases, bits = sample_shadows(psi, self.M, self.rng)
            self.n_snapshots += self.M
            Y_shadow.append(ShadowEstimator(bases, bits).expectations(self.obs))
            Y_exact.append([pauli_expectation(psi, o, self.n) for o in self.obs])
        return np.array(Y_shadow), np.array(Y_exact)


class KernelRidge:
    """RBF kernel ridge regression, multi-output.  gamma=None -> median heuristic."""

    def __init__(self, gamma: Optional[float] = None, ridge: float = 1e-1):
        self.gamma, self.ridge = gamma, ridge

    def _K(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-self.gamma * d2)

    def fit(self, A: np.ndarray, Y: np.ndarray) -> "KernelRidge":
        if self.gamma is None:
            d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
            self.gamma = 1.0 / np.median(d2[np.triu_indices(len(A), 1)])
        self.A = A
        K = self._K(A, A)
        self.alpha = np.linalg.solve(K + self.ridge * np.eye(len(A)), Y)
        return self

    def predict(self, A: np.ndarray) -> np.ndarray:
        return self._K(A, self.A) @ self.alpha


class LocalKernelRidge:
    """
    Locality-aware variant (spirit of Huang et al. 2022): predict observable
    on qubits S using only Hamiltonian parameters within distance `radius` of S.
    One KRR per observable; far-away couplings are ignored, which sharply cuts
    the effective input dimension and the sample complexity.
    """

    def __init__(self, n: int, observables, radius: int = 2, ridge: float = 1e-1):
        self.n, self.obs, self.radius, self.ridge = n, observables, radius, ridge
        self.models, self.masks = [], []

    def _mask(self, paulis) -> np.ndarray:
        sites = {i for i, _ in paulis}
        near = {j for i in sites for j in range(max(0, i - self.radius), min(self.n, i + self.radius + 1))}
        mJ = np.array([i in near or (i + 1) in near for i in range(self.n - 1)])
        mh = np.array([i in near for i in range(self.n)])
        return np.concatenate([mJ, mh])

    def fit(self, A: np.ndarray, Y: np.ndarray) -> "LocalKernelRidge":
        self.models, self.masks = [], []
        for k, o in enumerate(self.obs):
            m = self._mask(o)
            self.masks.append(m)
            self.models.append(KernelRidge(ridge=self.ridge).fit(A[:, m], Y[:, k]))
        return self

    def predict(self, A: np.ndarray) -> np.ndarray:
        return np.column_stack([mdl.predict(A[:, m]) for mdl, m in zip(self.models, self.masks)])


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 6
    print("=== (1) one shadow set, many observables ===")
    J, h = np.ones(n - 1), 0.8 * np.ones(n)
    psi = ground_state(tfim_hamiltonian(J, h))
    obs = nn_zz_observables(n) + [[(i, 0)] for i in range(n)] + [[(0, 2), (3, 2)]]
    for M in (200, 800, 3200):
        bases, bits = sample_shadows(psi, M, rng)
        est = ShadowEstimator(bases, bits, n_batches=5).expectations(obs)
        exact = np.array([pauli_expectation(psi, o, n) for o in obs])
        print(f"  M={M:5d}: {len(obs)} observables from the same snapshots, "
              f"max|err|={np.max(np.abs(est - exact)):.3f}  (bound ~ 3/sqrt(M)={3/np.sqrt(M):.3f})")

    print("=== (2) ML: predict NN-ZZ correlations for unseen Hamiltonians ===")
    obs = nn_zz_observables(n)
    ds = PropertyDataset(n, obs, M_shadows=500, rng=rng)
    A_test = random_tfim_params(n, 100, rng)
    _, Y_test_exact = ds.build(A_test)
    base = np.sqrt(np.mean((Y_test_exact - Y_test_exact.mean(0)) ** 2))
    print(f"  baseline RMSE (predict mean) = {base:.3f}")
    for N in (25, 50, 100, 200):
        A_tr = random_tfim_params(n, N, rng)
        Y_tr_shadow, _ = ds.build(A_tr)
        g = KernelRidge().fit(A_tr, Y_tr_shadow)
        l = LocalKernelRidge(n, obs, radius=2).fit(A_tr, Y_tr_shadow)
        r_g = np.sqrt(np.mean((g.predict(A_test) - Y_test_exact) ** 2))
        r_l = np.sqrt(np.mean((l.predict(A_test) - Y_test_exact) ** 2))
        print(f"  N={N:3d} training Hamiltonians ({N*500} snapshots): "
              f"global KRR RMSE={r_g:.3f}, local KRR RMSE={r_l:.3f}")
    print(f"  device cost: {ds.n_state_preparations} state preparations, {ds.n_snapshots} snapshots;"
          f" predictions for new Hamiltonians cost 0.")

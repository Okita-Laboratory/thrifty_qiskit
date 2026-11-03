"""
cdr_mitigation.py -- Learning-based error mitigation: Clifford Data Regression.

CDR (Czarnik, Arrasmith, Coles, Cincio 2021):
  1. Take the target circuit U(theta) (non-Clifford rotations).
  2. Build N training circuits U(theta_c) whose rotation angles are multiples
     of pi/2 (near the original angles).  These are Clifford circuits, so their
     IDEAL expectation values are computable classically (stabilizer simulation),
     while their NOISY values are measured on the device (or a noise model).
  3. Fit   ideal = a * noisy + b   (or a small polynomial) on the training pairs.
  4. Apply the fit to the noisy value of the target circuit.

Compared with zero-noise extrapolation (ZNE by unitary folding), CDR needs no
control over the noise level and learns the device's actual bias.

Everything here is self-generated:
  - noisy device   : density-matrix simulation with depolarising noise after
                     every gate (+ optional readout error), pure qiskit.quantum_info
  - ideal Clifford : StabilizerState (polynomial cost, scales to many qubits)
  - ideal target   : Statevector (for evaluation only; unavailable on real hardware)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# ************************************************************
# Circuit family: RY/RZ layers + linear CX ladder (all rotations parametrised)
# ************************************************************
def build_circuit(n: int, angles: np.ndarray, reps: int = 1):
    """angles: (reps+1, n, 2) -> RY(a0) RZ(a1) on every qubit, CX ladder between layers."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for r in range(reps + 1):
        for q in range(n):
            qc.ry(float(angles[r, q, 0]), q)
            qc.rz(float(angles[r, q, 1]), q)
        if r < reps:
            for q in range(n - 1):
                qc.cx(q, q + 1)
    return qc


def random_angles(n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 2 * np.pi, (reps + 1, n, 2))


def near_clifford_angles(angles: np.ndarray, rng: np.random.Generator, p_random: float = 0.3) -> np.ndarray:
    """Round each angle to the nearest multiple of pi/2, except with prob p_random pick a random one."""
    k = np.rint(angles / (np.pi / 2)).astype(int)
    rand = rng.integers(0, 4, angles.shape)
    mask = rng.uniform(size=angles.shape) < p_random
    k = np.where(mask, rand, k)
    return (k % 4) * (np.pi / 2)


# ************************************************************
# Simulators
# ************************************************************
class NoisyDevice:
    """Density-matrix simulator: depolarising after each gate, optional readout flips."""

    def __init__(self, p1: float = 0.005, p2: float = 0.03, readout: float = 0.0, shots: Optional[int] = None,
                 seed: int = 0):
        self.p1, self.p2, self.readout, self.shots = p1, p2, readout, shots
        self.rng = np.random.default_rng(seed)
        self.n_circuit_runs = 0

    @staticmethod
    def _depol(p: float, k: int):
        from qiskit.quantum_info import Kraus, Pauli
        import itertools
        paulis = [Pauli("".join(s)) for s in itertools.product("IXYZ", repeat=k)]
        d = 4 ** k
        ops = [np.sqrt(1 - p * (d - 1) / d) * paulis[0].to_matrix()] + \
              [np.sqrt(p / d) * P.to_matrix() for P in paulis[1:]]
        return Kraus(ops)

    def expectation(self, qc, observable) -> float:
        from qiskit.quantum_info import DensityMatrix, Operator
        self.n_circuit_runs += 1
        n = qc.num_qubits
        dm = DensityMatrix.from_int(0, 2 ** n)
        K1, K2 = self._depol(self.p1, 1), self._depol(self.p2, 2)
        for inst in qc.data:
            q = [qc.find_bit(b).index for b in inst.qubits]
            dm = dm.evolve(Operator(inst.operation), q)
            dm = dm.evolve(K1 if len(q) == 1 else K2, q)
        if self.readout > 0:
            # symmetric bit-flip readout error on every qubit == extra X-channel before measuring
            from qiskit.quantum_info import Kraus
            X = np.array([[0, 1], [1, 0]])
            Kr = Kraus([np.sqrt(1 - self.readout) * np.eye(2), np.sqrt(self.readout) * X])
            for q in range(n):
                dm = dm.evolve(Kr, [q])
        e = float(np.real(dm.expectation_value(observable)))
        if self.shots is not None:
            # observable is a sum of Pauli terms with eigenvalues +-1: crude shot noise
            var = float(np.real(dm.expectation_value(observable @ observable))) - e ** 2
            e += self.rng.normal(0, np.sqrt(max(var, 0) / self.shots))
        return e


def ideal_clifford(qc, observable) -> float:
    from qiskit.quantum_info import StabilizerState
    return float(np.real(StabilizerState(qc).expectation_value(observable)))


def ideal_statevector(qc, observable) -> float:
    from qiskit.quantum_info import Statevector
    return float(np.real(Statevector(qc).expectation_value(observable)))


# ************************************************************
# CDR
# ************************************************************
class CDR:
    def __init__(self, device: NoisyDevice, n_train: int = 20, degree: int = 1,
                 p_random: float = 0.3, rng: Optional[np.random.Generator] = None):
        self.dev, self.n_train, self.degree, self.p_random = device, n_train, degree, p_random
        self.rng = rng or np.random.default_rng(0)
        self.coef_ = None
        self.train_pairs_ = None

    def fit(self, n: int, reps: int, angles: np.ndarray, observable,
            min_nonzero_frac: float = 0.4, max_tries: int = 400) -> "CDR":
        """
        Training circuits are checked with the (free) stabilizer simulation BEFORE
        being sent to the device: Clifford circuits often give exactly 0 for a
        Pauli observable, and a training set of zeros cannot determine the fit.
        We keep at most (1-min_nonzero_frac)*n_train zero-valued circuits.
        """
        noisy, ideal = [], []
        n_zero_max = int((1 - min_nonzero_frac) * self.n_train)
        n_zero = 0
        for _ in range(max_tries):
            if len(ideal) >= self.n_train:
                break
            a = near_clifford_angles(angles, self.rng, self.p_random)
            qc = build_circuit(n, a, reps)
            v = ideal_clifford(qc, observable)
            if abs(v) < 1e-9:
                if n_zero >= n_zero_max:
                    continue
                n_zero += 1
            ideal.append(v)
            noisy.append(self.dev.expectation(qc, observable))   # device run only for kept circuits
        noisy, ideal = np.array(noisy), np.array(ideal)
        if np.std(ideal) < 1e-6:
            raise RuntimeError("degenerate CDR training set: increase p_random or max_tries")
        self.train_pairs_ = (noisy, ideal)
        self.coef_ = np.polyfit(noisy, ideal, self.degree)
        return self

    def mitigate(self, noisy_value: float) -> float:
        return float(np.polyval(self.coef_, noisy_value))


# ************************************************************
# ZNE by unitary folding (for comparison)
# ************************************************************
def fold_circuit(qc, scale: int):
    """Global folding: U -> U (U^dag U)^k, scale = 2k+1."""
    assert scale % 2 == 1
    folded = qc.copy()
    for _ in range((scale - 1) // 2):
        folded.compose(qc.inverse(), inplace=True)
        folded.compose(qc, inplace=True)
    return folded


def zne(device: NoisyDevice, qc, observable, scales: Sequence[int] = (1, 3, 5), degree: int = 1) -> float:
    xs, ys = [], []
    for s in scales:
        xs.append(s); ys.append(device.expectation(fold_circuit(qc, s), observable))
    return float(np.polyval(np.polyfit(xs, ys, degree), 0.0))


# ************************************************************
# Demo
# ************************************************************
if __name__ == "__main__":
    from qiskit.quantum_info import SparsePauliOp

    rng = np.random.default_rng(0)
    n, reps = 4, 2
    obs = SparsePauliOp.from_sparse_list([("ZZ", [1, 2], 1.0)], num_qubits=n)
    dev = NoisyDevice(p1=0.005, p2=0.03, readout=0.01)
    print(f"n={n}, reps={reps}, depolarising p1={dev.p1}, p2={dev.p2}, readout={dev.readout}")
    print(f"{'target':>6s} {'ideal':>8s} {'noisy':>8s} {'CDR':>8s} {'ZNE':>8s}   |err| noisy  CDR   ZNE")
    errs = []
    for t in range(8):
        angles = random_angles(n, reps, rng)
        qc = build_circuit(n, angles, reps)
        ideal = ideal_statevector(qc, obs)
        noisy = dev.expectation(qc, obs)
        cdr = CDR(dev, n_train=20, rng=np.random.default_rng(t)).fit(n, reps, angles, obs)
        mit = cdr.mitigate(noisy)
        z = zne(dev, qc, obs)
        errs.append((abs(noisy - ideal), abs(mit - ideal), abs(z - ideal)))
        print(f"{t:6d} {ideal:8.4f} {noisy:8.4f} {mit:8.4f} {z:8.4f}   "
              f"{errs[-1][0]:.4f} {errs[-1][1]:.4f} {errs[-1][2]:.4f}")
    e = np.array(errs)
    print(f"\nmean |err|: noisy {e[:,0].mean():.4f}, CDR {e[:,1].mean():.4f}, ZNE {e[:,2].mean():.4f}")
    print(f"device runs per target: CDR {cdr.n_train}+1 (training set reusable across observables), ZNE 3")
    print(f"CDR fit: ideal = {cdr.coef_[0]:.3f} * noisy + {cdr.coef_[1]:.4f}")

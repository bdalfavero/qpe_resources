import argparse
import json
import h5py
from math import sqrt, ceil
import numpy as np
import openfermion as of
from openfermionpyscf import run_pyscf
import qiskit
from qiskit.circuit.library import PauliEvolutionGate, phase_estimation
from qiskit.synthesis import LieTrotter
from qiskit import transpile
import quimb.tensor as qtn
from tensor_network_common import pauli_sum_to_mpo
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from qpe_trotter import (
    get_gate_counts,
    sample_eps2
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file for input.")
    parser.add_argument("output_file", type=str, help="HDF5 file for output.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    l = input_dict["l"]
    max_mpo_bond = input_dict["max_mpo_bond"]
    max_dmrg_bond = input_dict["max_dmrg_bond"]
    samples = int(input_dict["samples"])

    energy_error = 1e-3
    bond_length = 1.0 
    natoms = l
    geometry = [("H", (0, 0, i * bond_length)) for i in range(natoms)]
    molecule = of.MolecularData(geometry, basis="sto-3g", multiplicity=1, charge=natoms % 2)
    molecule = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
    fermion_hamiltonian = of.get_fermion_operator(molecule.get_molecular_hamiltonian())
    ham_jw = of.jordan_wigner(fermion_hamiltonian)
    nq = of.utils.count_qubits(ham_jw)
    nterms = len(ham_jw.terms)
    print(f"The Hamiltonian has {nterms} terms and acts on {nq} qubits.")
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_jw)
    qs = ham_cirq.qubits
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)

    # Get ground state by DMRG.
    dmrg = qtn.tensor_dmrg.DMRG(ham_mpo, max_dmrg_bond)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge!")
    ground_state = dmrg.state
    ground_energy = dmrg.energy.real
    print(f"Final DMRG energy: {ground_energy:4.5e}")

    # Approximate the norm of the Hamiltonian with the triangle inequality.
    # This is an upper bond on the norm, so we will have smaller tau than we should.
    coeffs = np.array([ps.coefficient for ps in ham_cirq])
    ham_norm = np.sum(np.abs(coeffs))
    evol_time = np.pi / (4. * ham_norm)
    print(f"Evolution time = {evol_time}")

    # Compute eps2 by sampling.
    sample_checkpoints, eps2_samples, _ = sample_eps2(ham_cirq, qs, ground_state, samples, max_mpo_bond=max_mpo_bond)
    eps2 = eps2_samples[-1]
    print(f"eps2 = {eps2}")
    dt = sqrt(energy_error / abs(eps2))
    print(f"dt = {dt}")
    num_steps = ceil(evol_time / dt)
    print(f"Requires {num_steps} steps.")

    # Get gate counts.
    evol_gate = PauliEvolutionGate(ham_qiskit, time=evol_time, synthesis=LieTrotter(reps=num_steps))
    ev_circuit = qiskit.QuantumCircuit(nq)
    ev_circuit.append(evol_gate, range(nq))
    ev_circuit_transpiled = qiskit.transpile(ev_circuit, basis_gates=["u3", "cx"])
    counts = get_gate_counts(ev_circuit_transpiled)
    qubit_nums = []
    gate_counts = []
    for qnum, count in counts.items():
        qubit_nums.append(qnum)
        gate_counts.append(count)

    f = h5py.File(args.output_file, "w")
    f.create_dataset("sample_checkpoints", data=np.array(sample_checkpoints))
    f.create_dataset("sampled_eps2", data=np.array(eps2_samples))
    f.create_dataset("dt", data=dt)
    f.create_dataset("evol_time", data=evol_time)
    f.create_dataset("num_steps", data=num_steps)
    f.create_dataset("qubit_nums", data=qubit_nums)
    f.create_dataset("gate_counts", data=gate_counts)
    f.close()

if __name__ == "__main__":
    main()
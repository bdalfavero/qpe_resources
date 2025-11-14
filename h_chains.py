import argparse
import json
import h5py
from math import sqrt, ceil
import numpy as np
from scipy.sparse.linalg import norm
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
    group_single_strings,
    trotter_perturbation,
    bits_for_epsilon,
    get_gate_counts,
    sample_eps2
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file for input.")
    parser.add_argument("output_file", type=str, help="JSON file for ouptut.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    l = input_dict["l"]
    max_mpo_bond = input_dict["max_mpo_bond"]
    max_mps_bond = input_dict["max_mps_bond"]
    energy_error = input_dict["energy_error"]
    nsamples = int(input_dict["samples"])

    # ham = of.fermi_hubbard(l, l, t, u, spinless=True)
    # ham_jw = of.transforms.jordan_wigner(ham)
    bond_length = 1.0 
    natoms = l
    geometry = [("H", (0, 0, i * bond_length)) for i in range(natoms)]
    molecule = of.MolecularData(geometry, basis="sto-3g", multiplicity=1, charge=0)
    molecule = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)  # To get ground state energy for comparison etc., also can use DMRG
    fermion_hamiltonian = of.get_fermion_operator(molecule.get_molecular_hamiltonian())
    ham_jw = of.jordan_wigner(fermion_hamiltonian)
    nterms = len(ham_jw.terms)
    print(f"Hamiltonian has {nterms} terms.")
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_jw)
    qs = ham_cirq.qubits
    nq = len(qs)
    print(f"Hamiltonian has {nq} qubits.")
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)

    # Get an approximate ground state using DMRG.
    dmrg = qtn.tensor_dmrg.DMRG(ham_mpo, max_mps_bond)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge!")
    ground_state = dmrg.state
    ground_energy = dmrg.energy.real
    print(f"Final DMRG energy: {ground_energy:4.5e}")

    if nq <= 10:
        ham_sparse = of.linalg.get_sparse_operator(ham_jw)
        ham_norm = norm(ham_sparse)
    else:
        # Approximate the norm of the Hamiltonian with the triangle inequality.
        # This is an upper bond on the norm, so we will have smaller tau than we should.
        coeffs = np.array([ps.coefficient for ps in ham_cirq])
        ham_norm = np.sum(np.abs(coeffs))
    evol_time = np.pi / (4. * ham_norm)
    print(f"Evolution time = {evol_time}")

    groups = [ps for ps in ham_cirq]
    v2 = trotter_perturbation(groups)
    v2_mpo = pauli_sum_to_mpo(v2, qs, max_mpo_bond)
    # Get energy from Mehendale Eqn. 8
    eps2 = (ground_state.H @ v2_mpo.apply(ground_state)).real
    print(f"eps2 = {eps2:4.5e}")
    dt = sqrt(energy_error / eps2)
    num_steps = ceil(evol_time / dt)
    print(f"dt = {dt:4.5e}, n_steps = {num_steps}")

    # Use the largest term to get a pessimistic bound.
    coeffs = np.array([ps.coefficient for ps in ham_cirq])
    i_max = np.argmax(np.abs(coeffs))
    max_coeff = coeffs[i_max]
    eps2_bound = (1. / 24) * 0.5 * max_coeff.real ** 3
    print(f"eps2_bound = {eps2_bound}")
    dt_bound = sqrt(energy_error / abs(eps2_bound)).real
    print(f"dt_bound = {dt_bound}")

    # Use the sampling method.
    sample_checkpoints, eps2_sampled, _ = sample_eps2(groups, qs, ground_state, nsamples, max_mpo_bond=max_mpo_bond)
    print(f"eps2_sampled = {eps2_sampled[-1]}")

    # # Synethsize a circuit with multiple ancillae (traditional QPE)
    evol_gate = PauliEvolutionGate(ham_qiskit, time=evol_time, synthesis=LieTrotter(reps=num_steps))
    # num_ancillae = bits_for_epsilon(energy_error)
    # print(f"Synthesizing QPE circuit with {num_ancillae} ancillae")
    # qpe_ckt = phase_estimation(num_ancillae, evol_gate)
    # print("Transpiling.")
    # qpe_ckt_transpiled = transpile(qpe_ckt, basis_gates=["u3", "cx"])
    # depth = qpe_ckt_transpiled.depth()
    # counts = get_gate_counts(qpe_ckt_transpiled)
    # print(f"Transpiled circuit has depth {depth}.")
    # print("Gate counts:")
    # for k, v in counts.items():
    #     print(f"{k}, {v}")
    
    # Synthesize a controlled Trotter step of time dt.
    print("Synthesizing SAPE circuit.")
    sape_ckt = qiskit.QuantumCircuit(nq + 1)
    controlled_evol_gate = evol_gate.control()
    sape_ckt.append(controlled_evol_gate, range(nq + 1))
    sape_transpiled = transpile(sape_ckt, basis_gates=["u3", "cx"])
    sape_depth = sape_transpiled.depth()
    sape_counts = get_gate_counts(sape_transpiled)
    print(f"Transpiled circuit has depth {sape_depth}.")
    print("Gate counts:")
    qubit_numbers = []
    gate_counts = []
    for k, v in sape_counts.items():
        print(f"{k}, {v}")
        qubit_numbers.append(k)
        gate_counts.append(v)
    
    f = h5py.File(args.output_file, "w")
    f.create_dataset("l", data=l)
    f.create_dataset("evol_time", data=evol_time)
    f.create_dataset("energy_error", data=energy_error)
    f.create_dataset("eps2_exact", data=eps2)
    f.create_dataset("eps2_bound", data=eps2_bound)
    f.create_dataset("sample_checkpoints", data=np.array(sample_checkpoints))
    f.create_dataset("eps2_samples", data=np.array(eps2_sampled))
    f.create_dataset("last_sample", data=eps2_sampled[-1])
    f.create_dataset("dt", data=dt)
    f.create_dataset("num_steps", data=num_steps)
    f.create_dataset("sape_depth", data=sape_depth)
    f.create_dataset("qubit_numbers", data=qubit_numbers)
    f.create_dataset("gate_counts", data=gate_counts)
    f.close()

if __name__ == "__main__":
    main()
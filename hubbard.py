import argparse
import json
import h5py
from math import sqrt, ceil
import numpy as np
from scipy.sparse.linalg import norm
import openfermion as of
import qiskit
from qiskit.circuit.library import PauliEvolutionGate, phase_estimation
from qiskit.synthesis import LieTrotter
from qiskit import transpile
import quimb.tensor as qtn
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.converters.openfermion_bridge import from_openfermion
from qtoolbox.grouping import sorted_insertion_grouping
from tensor_network_common import pauli_sum_to_mpo, mps_to_vector
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from qpe_trotter import (
    v2_pauli_sum,
    build_v2_terms,
    compute_expectation_parallel,
    compute_expectation_sequential,
    get_gate_counts
)
from kcommute import get_si_sets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file for input.")
    parser.add_argument("output_file", type=str, help="HDF5 file for ouptut.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    l1 = input_dict["l1"]
    l2 = input_dict["l2"]
    t = input_dict["t"]
    u = input_dict["u"]
    max_mpo_bond = input_dict["max_mpo_bond"]
    max_mps_bond = input_dict["max_mps_bond"]
    energy_error = input_dict["energy_error"]
    k = input_dict["k"]
    n_workers = input_dict["n_workers"]

    ham = of.fermi_hubbard(l1, l2, t, u, spinless=True)
    ham_jw = of.transforms.jordan_wigner(ham)
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
    ground_state_vec = mps_to_vector(ground_state)
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

    # Use the exact method.
    # Use SI.
    groups = get_si_sets(ham_cirq, k=k)
    print(f"Hamiltonian has {len(groups)} groups.")
    group_psums = [sum(group) for group in groups]
    v2 = v2_pauli_sum(group_psums)
    v2_mpo = pauli_sum_to_mpo(v2, qs, max_mpo_bond)
    # Get energy from Mehendale Eqn. 8
    eps2 = (ground_state.H @ v2_mpo.apply(ground_state)).real
    print(f"eps2 = {eps2:4.5e}")
    dt = sqrt(energy_error / eps2)
    num_steps = ceil(evol_time / dt)
    print(f"dt = {dt:4.5e}, n_steps = {num_steps}")

    # Use Jeremiah's quantum toolbox to compute eps2.
    terms = [from_openfermion(term, coeff, nq)
            for term, coeff in ham_jw.terms.items() if term]  # skip identity
    ham_qt = Hamiltonian(terms)
    print(f"Loaded Hamiltonian: {ham_qt.num_terms()} terms, {ham_qt.num_qubits()} qubits")
    group_collection = sorted_insertion_grouping(ham_qt, k=k)
    sym_groups = [list(g.paulis) for g in group_collection.groups]
    v2_terms = build_v2_terms(sym_groups)
    eps2_toolbox = compute_expectation_parallel(v2_terms, ground_state, nq, n_workers)
    eps2_err = abs(eps2_toolbox - eps2)
    print(f"eps2 from toolbox = {eps2_toolbox:4.5e}")
    print(f"Absolute error {eps2_err:4.5e}.")

    # # Use the largest term to get a pessimistic bound.
    # coeffs = np.array([ps.coefficient for ps in ham_cirq])
    # i_max = np.argmax(np.abs(coeffs))
    # max_coeff = coeffs[i_max]
    # eps2_bound = (1. / 24) * 0.5 * max_coeff.real ** 3
    # print(f"eps2_bound = {eps2_bound}")
    # dt_bound = sqrt(energy_error / abs(eps2_bound)).real
    # print(f"dt_bound = {dt_bound}")

    # Synethsize a circuit with multiple ancillae (traditional QPE)
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
    sape_ckt = qiskit.QuantumCircuit(nq)
    sape_ckt.append(evol_gate, range(nq))
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
    f.create_dataset("l1", data=l1)
    f.create_dataset("l2", data=l2)
    f.create_dataset("t", data=t)
    f.create_dataset("u", data=u)
    f.create_dataset("evol_time", data=evol_time)
    f.create_dataset("energy_error", data=energy_error)
    f.create_dataset("eps2_exact", data=eps2)
    f.create_dataset("eps2_toolbox", data=eps2_toolbox)
    f.create_dataset("dt", data=dt)
    f.create_dataset("num_steps", data=num_steps)
    f.create_dataset("sape_depth", data=sape_depth)
    f.create_dataset("qubit_numbers", data=qubit_numbers)
    f.create_dataset("gate_counts", data=gate_counts)
    f.close()

if __name__ == "__main__":
    main()
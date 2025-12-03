from time import perf_counter_ns
import json
from math import sqrt, ceil
import openfermion as of
import qiskit
from qiskit.circuit.library import PauliEvolutionGate, phase_estimation
from qiskit.synthesis import LieTrotter
from qiskit import transpile
from qiskit.qasm2 import dump
import quimb
import quimb.tensor as qtn
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.converters.openfermion_bridge import from_openfermion
from qtoolbox.grouping import sorted_insertion_grouping
from tensor_network_common import pauli_sum_to_mpo
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from qpe_trotter import (
    build_v2_terms,
    compute_expectation_parallel,
    get_gate_counts
)

def main():
    max_mpo_bond = 500
    max_mps_bond = 10
    energy_error = 1e-3
    evol_time = 0.1

    hamiltonian_file = "data/monomer_eqb"
    hamiltonian = of.jordan_wigner(
            of.get_fermion_operator(
        of.chem.MolecularData(filename=hamiltonian_file).get_molecular_hamiltonian()
        )
    )
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    qs = ham_cirq.qubits
    nq = len(qs)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)

    # Get an approximate ground state using DMRG.
    dmrg = qtn.tensor_dmrg.DMRG(ham_mpo, max_mps_bond)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge!")
    ground_state = dmrg.state
    ground_energy = dmrg.energy.real
    quimb.save_to_disk(ground_state, "data/water_mps.dump")
    print(f"Final DMRG energy: {ground_energy:4.5e}")

    # groups = group_single_strings(ham_cirq)
    # v2 = trotter_perturbation(groups)
    # v2_mpo = pauli_sum_to_mpo(v2, qs, max_mpo_bond)
    # # Get energy from Mehendale Eqn. 8
    # eps2 = (ground_state.H @ v2_mpo.apply(ground_state)).real
    start_time = perf_counter_ns()
    terms = [from_openfermion(term, coeff, nq)
            for term, coeff in ham_jw.terms.items() if term]  # skip identity
    ham = Hamiltonian(terms)
    print(f"Loaded Hamiltonian: {ham.num_terms()} terms, {ham.num_qubits()} qubits")
    group_collection = sorted_insertion_grouping(ham)
    sym_groups = [list(g.paulis) for g in group_collection.groups]
    end_time = perf_counter_ns()
    sorting_time = float(end_time - start_time)
    start_time = perf_counter_ns()
    v2_terms = build_v2_terms(sym_groups)
    end_time = perf_counter_ns()
    v2_time = float(end_time - start_time)
    # eps2_toolbox = compute_expectation_parallel(v2_terms, ground_state_vec, nq, n_workers)
    start_time = perf_counter_ns()
    eps2 = compute_expectation_parallel(v2_terms, ground_state, nq, n_workers)
    end_time = perf_counter_ns()
    expectation_time = float(end_time - start_time)
    print(f"eps2 = {eps2:4.5e}")
    dt = sqrt(energy_error / eps2)
    num_steps = ceil(evol_time / dt)
    print(f"dt = {dt:4.5e}, n_steps = {num_steps}")

    evol_gate = PauliEvolutionGate(ham_qiskit, time=evol_time, synthesis=LieTrotter(reps=num_steps))
    # num_ancillae = bits_for_epsilon(energy_error)
    # print(f"Synthesizing QPE circuit with {num_ancillae} ancillae")
    # qpe_ckt = phase_estimation(num_ancillae, evol_gate)
    qpe_ckt = qiskit.QuantumCircuit(nq)
    qpe_ckt.append(evol_gate, range(nq))
    dump(qpe_ckt, "data/water_qpe_circuit.qasm")
    qpe_ckt_transpiled = transpile(qpe_ckt, basis_gates=["u3", "cx"])
    depth = qpe_ckt_transpiled.depth()
    counts = get_gate_counts(qpe_ckt_transpiled)
    print("Gate counts:")
    for k, v in counts.items():
        print(f"{k}, {v}")
    
    output_dict = {
        "counts": counts,
        "energy": ground_energy,
        "depth": depth,
        "steps": num_steps,
        "sorting_time": sorting_time,
        "v2_time": v2_time,
        "expectation_time": expectation_time
    }
    with open("data/water_out.json", "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()
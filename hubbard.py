import argparse
import json
from math import sqrt, ceil
import numpy as np
import openfermion as of
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
    t = input_dict["t"]
    u = input_dict["u"]
    max_mpo_bond = input_dict["max_mpo_bond"]
    max_mps_bond = input_dict["max_mps_bond"]
    evol_time = input_dict["evol_time"]
    energy_error = input_dict["energy_error"]

    ham = of.fermi_hubbard(l, l, t, u, spinless=True)
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
    print(f"Final DMRG energy: {ground_energy:4.5e}")

    # Use the exact method.
    groups = group_single_strings(ham_cirq)
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
    eps2_bound = (-1. / 24) * 0.5 * max_coeff ** 3
    print(f"eps2_bound = {eps2_bound}")
    dt_bound = sqrt(energy_error / abs(eps2_bound))
    print(f"dt_bound = {dt_bound}")

    # Use the sampling method.
    # nsamples = 100_000
    # eps2_sampled = sample_eps2(groups, ground_state, nsamples, qs, max_mpo_bond)
    # print(f"eps2_sampled = {eps2_sampled}")

    # Synethsize a circuit with multiple ancillae (traditional QPE)
    evol_gate = PauliEvolutionGate(ham_qiskit, time=evol_time, synthesis=LieTrotter(reps=num_steps))
    num_ancillae = bits_for_epsilon(energy_error)
    print(f"Synthesizing QPE circuit with {num_ancillae} ancillae")
    qpe_ckt = phase_estimation(num_ancillae, evol_gate)
    print("Transpiling.")
    qpe_ckt_transpiled = transpile(qpe_ckt, basis_gates=["u3", "cx"])
    depth = qpe_ckt_transpiled.depth()
    counts = get_gate_counts(qpe_ckt_transpiled)
    print(f"Transpiled circuit has depth {depth}.")
    print("Gate counts:")
    for k, v in counts.items():
        print(f"{k}, {v}")
    
    # Synthesize a controlled Trotter step of time dt.
    print("Synthesizing SAPE circuit.")
    sape_ckt = qiskit.QuantumCircuit(nq + 1)
    controlled_evol_gate = evol_gate.control()
    sape_ckt.append(controlled_evol_gate, range(nq + 1))
    sape_transpiled = transpile(sape_ckt, basis_gates=["u3", "cx"])
    sape_depth = sape_transpiled.depth()
    sape_counts = get_gate_counts(sape_transpiled)
    print(f"Transpiled circuit has depth {depth}.")
    print("Gate counts:")
    for k, v in sape_counts.items():
        print(f"{k}, {v}")
    
    output_dict = {
        "l": l,
        "t": t,
        "u": u,
        "evol_time": evol_time,
        "energy_error": energy_error,
        "num_ancillae": num_ancillae,
        "dt": dt,
        "num_steps": num_steps,
        "depth": depth,
        "sape_depth": sape_depth,
        "counts": counts,
        "sape_counts": sape_counts
    }
    with open(args.output_file, "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()
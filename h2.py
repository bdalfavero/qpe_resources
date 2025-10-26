"""Get gate counts for the H2 molecule."""

import json
from math import sqrt, ceil
import openfermion as of
from openfermionpyscf import run_pyscf
from qiskit.circuit.library import PauliEvolutionGate, phase_estimation
from qiskit.synthesis import LieTrotter
from qiskit import transpile
from qiskit.qasm2 import dump
import quimb.tensor as qtn
from tensor_network_common import pauli_sum_to_mpo
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from qpe_trotter import (
    group_single_strings,
    trotter_perturbation,
    bits_for_epsilon,
    get_gate_counts
)

def main():
    max_mpo_bond = 100
    max_mps_bond = 15
    evol_time = 0.5
    energy_error = 1e-3
    n_elec = 4
    molec = "LiH"
    basis = "sto-3g"
    geometry = of.chem.geometry_from_pubchem(molec)
    multiplicity = 1
    molecule = of.chem.MolecularData(
        geometry, basis, multiplicity
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    print(f"HF energy:", molecule.hf_energy)
    print(f"FCI energy:", molecule.fci_energy)
    hamiltonian_fermi = molecule.get_molecular_hamiltonian()
    hamiltonian = of.transforms.jordan_wigner(hamiltonian_fermi)
    nterms = len(hamiltonian.terms)
    print(f"Hamiltonian has {nterms} terms.")
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, ham_cirq.qubits, max_mpo_bond)

    # Get an approximate ground state using DMRG.
    dmrg = qtn.tensor_dmrg.DMRG(ham_mpo, max_mps_bond)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge!")
    ground_state = dmrg.state
    ground_energy = dmrg.energy.real
    print(f"Final DMRG energy: {ground_energy:4.5e}")

    groups = group_single_strings(ham_cirq)
    v2 = trotter_perturbation(groups)
    v2_mpo = pauli_sum_to_mpo(v2, qs, max_mpo_bond)
    # Get energy from Mehendale Eqn. 8
    eps2 = (ground_state.H @ v2_mpo.apply(ground_state)).real
    print(f"eps2 = {eps2:4.5e}")
    dt = sqrt(energy_error / eps2)
    num_steps = ceil(evol_time / dt)
    print(f"dt = {dt:4.5e}, n_steps = {num_steps}")

    evol_gate = PauliEvolutionGate(ham_qiskit, time=evol_time, synthesis=LieTrotter(reps=num_steps))
    num_ancillae = bits_for_epsilon(energy_error)
    print(f"Synthesizing QPE circuit with {num_ancillae} ancillae")
    qpe_ckt = phase_estimation(num_ancillae, evol_gate)
    qpe_ckt_transpiled = transpile(qpe_ckt, basis_gates=["u3", "cx"])
    dump(qpe_ckt_transpiled, "data/h2_circuit.qasm")
    depth = qpe_ckt_transpiled.depth()
    counts = get_gate_counts(qpe_ckt_transpiled)
    print("Gate counts:")
    for k, v in counts.items():
        print(f"{k}, {v}")
    
    output_dict = {
        "hf_energy": molecule.hf_energy,
        "fci_energy": molecule.fci_energy,
        "dt": dt,
        "steps": num_steps,
        "ancillae": num_ancillae,
        "depth": depth,
        "counts": counts
    }
    with open("data/h2_output.json", "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()
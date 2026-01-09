from time import perf_counter_ns
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
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.converters.openfermion_bridge import from_openfermion
from qtoolbox.grouping import sorted_insertion_grouping
from tensor_network_common import pauli_sum_to_mpo
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from qpe_trotter import (
    v2_pauli_sum,
    build_v2_terms,
    compute_expectation_parallel,
    get_gate_counts
)
from kcommute import get_si_sets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file for input.")
    parser.add_argument("output_file", type=str, help="JSON file for ouptut.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    molec_name = input_dict["molec_name"]
    max_mpo_bond = input_dict["max_mpo_bond"]
    max_mps_bond = input_dict["max_mps_bond"]
    energy_error = input_dict["energy_error"]
    k = input_dict["k"]
    n_workers = input_dict["n_workers"]

    geometry = of.chem.geometry_from_pubchem(molec_name)

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

    ham_norm = ham_mpo.norm()
    evol_time = np.pi / (4. * ham_norm)
    print(f"Evolution time = {evol_time}")

    # Compute eps2
    terms = [from_openfermion(term, coeff, nq)
            for term, coeff in ham_jw.terms.items() if term]  # skip identity
    ham = Hamiltonian(terms)
    print(f"Loaded Hamiltonian: {ham.num_terms()} terms, {ham.num_qubits()} qubits")
    group_collection = sorted_insertion_grouping(ham)
    sym_groups = [list(g.paulis) for g in group_collection.groups]
    v2_terms = build_v2_terms(sym_groups, n_workers=n_workers)
    # eps2_toolbox = compute_expectation_parallel(v2_terms, ground_state_vec, nq, n_workers)
    eps2_toolbox = compute_expectation_parallel(v2_terms, ground_state, nq, n_workers)
    print(f"eps2 from toolbox = {eps2_toolbox:4.5e}")
    dt = sqrt(energy_error / eps2_toolbox)
    num_steps = ceil(evol_time / dt)
    print(f"dt = {dt:4.5e}, n_steps = {num_steps}")

    # Synthesize a controlled Trotter step of time dt.
    print("Synthesizing SAPE circuit.")
    evol_gate = PauliEvolutionGate(ham_qiskit, time=evol_time, synthesis=LieTrotter(reps=num_steps))
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
    f.create_dataset("molec_name", data=molec_name)
    f.create_dataset("nq", data=nq)
    f.create_dataset("nterms", data=nterms)
    f.create_dataset("evol_time", data=evol_time)
    f.create_dataset("energy_error", data=energy_error)
    f.create_dataset("eps2_exact", data=eps2_toolbox)
    f.create_dataset("dt", data=dt)
    f.create_dataset("num_steps", data=num_steps)
    f.create_dataset("sape_depth", data=sape_depth)
    f.create_dataset("qubit_numbers", data=qubit_numbers)
    f.create_dataset("gate_counts", data=gate_counts)
    f.close()




if __name__ == "__main__":
    main()
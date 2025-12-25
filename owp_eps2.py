import json
import argparse
import h5py
from math import ceil, sqrt
import numpy as np
import openfermion as of
from quimb.utils import load_from_disk
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.converters.openfermion_bridge import from_openfermion
from qtoolbox.grouping import sorted_insertion_grouping
from qpe_trotter import (
    v2_pauli_sum,
    build_v2_terms,
    build_v2_terms_parallel,
    compute_expectation_parallel,
    get_gate_counts
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file for input.")
    parser.add_argument("output_file", type=str, help="HDF5 file for ouptut.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    energy_error = input_dict["energy_error"]
    k = input_dict["k"]
    n_workers = input_dict["n_workers"]
    threshold = input_dict["threshold"]
    ham_dir = input_dict["ham_dir"] # Where to read Hamiltonian and ground state from.

    ground_state = load_from_disk(f"{ham_dir}/owp_ground_state_25.dat")
    ham_mpo = load_from_disk(f"{ham_dir}/owp_mpo_chi_1000.dat")
    hamiltonian = of.utils.load_operator(file_name="owp_631gd_22_ducc.data", data_directory=ham_dir)
    nterms_before = len(hamiltonian.terms)
    hamiltonian.compress(abs_tol=threshold)
    nterms_after = len(hamiltonian.terms)
    ham_jw = of.transforms.jordan_wigner(hamiltonian)
    nq = of.utils.count_qubits(ham_jw)

    evol_time = np.pi / ham_mpo.norm()

    # Use Jeremiah's quantum toolbox to compute eps2.
    terms = [from_openfermion(term, coeff, nq)
            for term, coeff in ham_jw.terms.items() if term]  # skip identity
    ham = Hamiltonian(terms)
    print(f"Loaded Hamiltonian: {ham.num_terms()} terms, {ham.num_qubits()} qubits")
    group_collection = sorted_insertion_grouping(ham, k=k)
    sym_groups = [list(g.paulis) for g in group_collection.groups]
    v2_terms = build_v2_terms_parallel(sym_groups, n_workers=n_workers)
    # eps2_toolbox = compute_expectation_parallel(v2_terms, ground_state_vec, nq, n_workers)
    eps2_toolbox = compute_expectation_parallel(v2_terms, ground_state, nq, n_workers)
    print(f"eps2 from toolbox = {eps2_toolbox:4.5e}")
    dt = sqrt(energy_error / eps2_toolbox)
    num_steps = ceil(evol_time / dt)
    print(f"dt = {dt:4.5e}, n_steps = {num_steps}")

    f = h5py.File(args.output_file, "w")
    f.create_dataset("nterms_before", data=nterms_before)
    f.create_dataset("nterms_after", data=nterms_after)
    f.create_dataset("evol_time", data=evol_time)
    f.create_dataset("energy_error", data=energy_error)
    f.create_dataset("eps2_exact", data=eps2_toolbox)
    f.create_dataset("dt", data=dt)
    f.create_dataset("num_steps", data=num_steps)
    f.close()

if __name__ == "__main__":
    main()
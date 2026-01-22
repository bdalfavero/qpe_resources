"""Evaluate eps2 by writing terms to the scratch directory."""

from typing import List
import pickle as pkl
from itertools import accumulate
from time import perf_counter_ns
import numpy as np
from scipy.sparse import csc_matrix, kron, identity
import cirq
import openfermion as of
from openfermionpyscf import run_pyscf
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
from convert import to_groups_of
from error_pert import get_v2_sarray
from qpe_trotter import v2_pauli_sum, v2_qubop, fast_commutator_sum
from kcommute import get_si_sets
from tensor_network_common import pauli_sum_to_mpo, mps_to_vector
from qtoolbox.core.pauli import PauliString
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.converters.openfermion_bridge import from_openfermion
from qtoolbox.grouping import sorted_insertion_grouping
from itertools import accumulate
from multiprocessing import Pool, cpu_count
from qpe_trotter import compute_expectation_sequential, build_v2_terms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def v2_initial_sums(sym_groups):
    nterms = len(sym_groups)
    sums_l2r = list(accumulate(sym_groups, lambda a, b: a + b))
    sums_r2l = list(reversed(list(accumulate(reversed(sym_groups), lambda a, b: a + b))))
    sums_r2l.append([])
    return sums_l2r, sums_r2l


def do_terms(groups_filename, sums_filename, indices):
    groups_file = open(groups_filename, "rb")
    sums_file = open(sums_filename, "rb")
    sym_groups = pkl.load(groups_file)
    sums_dict = pkl.load(sums_file)
    sums_l2r = sums_dict["sums_l2r"]
    sums_r2l = sums_dict["sums_r2l"]

    v2_terms = []
    for i in indices:
        V1 = fast_commutator_sum(sums_l2r[i-1], sym_groups[i])
        for t in fast_commutator_sum(V1, sums_r2l[i+1]):
            t.coeff *= -1/3
            v2_terms.append(t)
        for t in fast_commutator_sum(V1, sym_groups[i]):
            t.coeff *= -1/6
            v2_terms.append(t)
    return v2_terms


def main():
    # First the normal way.
    molec = "H2"
    basis = "sto-3g"
    n_elec = 2
    geometry = of.chem.geometry_from_pubchem(molec)
    multiplicity = 1
    molecule = of.chem.MolecularData(
        geometry, basis, multiplicity
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    print(f"HF energy:", molecule.hf_energy)
    print(f"FCI energy:", molecule.fci_energy)
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    hamiltonian_psum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)

    nq = of.utils.count_qubits(hamiltonian_qubop)
    nterms = len(hamiltonian_qubop.terms)
    print(f"Hamiltonian has {nq} qubits and {nterms} terms.")

    qs = cirq.LineQubit.range(nq)
    hamiltonian_mpo = pauli_sum_to_mpo(hamiltonian_psum, qs, 100)
    dmrg = qtn.DMRG(hamiltonian_mpo, bond_dims=15)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge.")
    ground_state = dmrg.state
    ground_state_vec = mps_to_vector(ground_state)

    terms = [from_openfermion(term, coeff, nq)
            for term, coeff in hamiltonian_qubop.terms.items() if term]  # skip identity
    ham = Hamiltonian(terms)

    # Group using quantum-toolbox's SI
    start_group = perf_counter_ns()
    group_collection = sorted_insertion_grouping(ham)
    time_grouping = perf_counter_ns() - start_group

    sym_groups = [list(g.paulis) for g in group_collection.groups]
    v2_terms = build_v2_terms(sym_groups)
    eps2_symplectic = compute_expectation_sequential(v2_terms, ground_state_vec, nq)
    print(eps2_symplectic)

    # Now with scratch directory.
    sums_l2r, sums_r2l = v2_initial_sums(sym_groups)
    sums_fname = "data/h2_sums.pkl"
    output_dict = {
        "sums_l2r": sums_l2r,
        "sums_r2l": sums_r2l
    }
    with open(sums_fname, "wb") as f:
        pkl.dump(output_dict, f)
    groups_fname = "data/h2_groups.pkl"
    with open(groups_fname, "wb") as f:
        pkl.dump(sym_groups, f)
    nterms = len(sym_groups)
    eps2_scratch = 0.
    for i in range(1, nterms):
        v2_terms = do_terms(groups_fname, sums_fname, [i])
        eps2_scratch_contrib = compute_expectation_sequential(v2_terms, ground_state_vec, nq)
        eps2_scratch += eps2_scratch_contrib
    print(eps2_scratch)

if __name__ == "__main__":
    main()
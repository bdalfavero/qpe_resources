import openfermion as of
from quimb.utils import save_to_disk
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from tensor_network_common import pauli_sum_to_mpo

def main():
    max_mpo_bond = 1_000
    threshold=1e-2

    hamiltonian = of.utils.load_operator(file_name="owp_631gd_22_ducc.data", data_directory="data")
    hamiltonian.compress(abs_tol=threshold)
    print(f"Compressed Fermionic Hamiltonian has {len(hamiltonian.terms)} term(s).")
    ham_jw = of.transforms.jordan_wigner(hamiltonian)
    nterms = len(ham_jw.terms)
    print(f"Hamiltonian has {nterms} terms.")
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_jw)
    qs = ham_cirq.qubits
    nq = len(qs)
    print(f"Hamiltonian has {nq} qubits.")
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)
    save_to_disk(ham_mpo, f"data/owp_mpo_chi_{max_mpo_bond}.dat")

if __name__ == "__main__":
    main()
"""This analysis is based on https://arxiv.org/abs/2312.13282"""

from typing import List, Dict
import cirq
import qiskit

def group_single_strings(ham: cirq.PauliSum) -> List[cirq.PauliSum]:
    """A partitioning where each partition contains a single string."""

    groups: List[cirq.PauliSum] = []
    for h in ham:
        groups.append(cirq.PauliSum.from_pauli_strings([h]))
    return groups


def trotter_perturbation(hamiltonian_terms: List[cirq.PauliSum]) -> cirq.PauliSum:
    """Get the first order of the perturbation Hamiltonian in Eqn. 7 of Mehendale et al.
    Note that this does not include the factor of dt^2.
    
    Arguments:
    hamiltonian_terms - The terms {H_1, ..., H_n} used in the Trotter decomposition.
    
    Returns:
    H' - The perturbation to the Hamiltonian to second order in dt."""

    M = len(hamiltonian_terms)
    # print(f"M = {M}")
    h_prime = cirq.PauliSum()
    # TODO Can I parallelize this? Maybe only over the mu loop.
    for mu in range(2 * M - 1):
        print(f"mu = {mu}")
        if mu >= M:
            i = mu - M
            h_mu = hamiltonian_terms[M - 1 - i]
        else:
            h_mu = hamiltonian_terms[mu]
        for nu in range(mu+1, 2 * M):
            # print(f"nu = {nu}")
            if nu >= M:
                i = nu - M
                h_nu = hamiltonian_terms[M - 1 - i]
            else:
                h_nu = hamiltonian_terms[nu]
            comm_nu_mu = h_nu * h_mu - h_mu * h_nu
            for nu_prime in range(nu, 2 * M):
                # print(f"nu_prime = {nu_prime}")
                if nu_prime >= M:
                    i = nu_prime - M
                    h_nu_prime = hamiltonian_terms[M - 1 - i]
                else:
                    h_nu_prime = hamiltonian_terms[nu_prime]
                comm_three = h_nu_prime * comm_nu_mu - comm_nu_mu * h_nu_prime
                if nu_prime == nu:
                    delta = 1.
                else:
                    delta = 0.
                h_prime += (1. - delta / 2.) * comm_three
    return -1. / 24. * h_prime


def bits_for_epsilon(eps: float, max_depth: int = 40) -> int:
    """Get the number of bits k s.t. 2^-k <= eps."""

    k = 0
    while 2 ** (-k) >= eps:
        k += 1
        if k > max_depth:
            raise ValueError(f"Max number of iterations exceeded: k={k}")
    return k


def get_gate_counts(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Get the counts of how many n-qubit gates there are."""

    counts: Dict[int, int] = {}
    for inst in circuit.data:
        inst_nq = len(inst.qubits)
        if inst_nq in counts.keys():
            counts[inst_nq] += 1
        else:
            counts[inst_nq] = 1
    return counts

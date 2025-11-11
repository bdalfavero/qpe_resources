"""This analysis is based on https://arxiv.org/abs/2312.13282"""

from typing import List, Dict, Tuple
from random import randint, random
import numpy as np
import cirq
import qiskit
from quimb.tensor.tensor_1d import MatrixProductState
from tensor_network_common import pauli_sum_to_mpo, mpo_mps_exepctation

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


def sample_eps2(
    hamiltonian: cirq.PauliSum, qs: List[cirq.Qid], ground_state: MatrixProductState,
    samples: int, interval: int=10, max_mpo_bond: int=100
) -> Tuple[List[int], List[float], int]:
    """Estimate eps2 by sampling terms uniformly."""

    hamiltonian_terms = [ps for ps in hamiltonian]
    M = len(hamiltonian_terms)
    # num_sum = num_summands(M)
    # print(f"{num_sum} terms in summation.")
    sample_checkpoints = []
    sample_num_checkpoints = []
    num_zero = 0
    running_estimate = 0.
    v2 = cirq.PauliSum()
    for n in range(samples):
        mu = np.random.randint(0, 2 * M - 1)
        nu = np.random.randint(mu + 1, 2 * M)
        nu_prime = np.random.randint(nu, 2 * M)
        p_mu = 1. / len(range(0, 2 * M - 1))
        p_nu = 1. / len(range(mu + 1, 2 * M))
        p_nu_prime = 1. / len(range(nu, 2 * M))
        p = p_mu * p_nu * p_nu_prime
        if mu >= M:
            i = mu - M
            h_mu = hamiltonian_terms[M - 1 - i]
        else:
            h_mu = hamiltonian_terms[mu]
        if nu >= M:
            i = nu - M
            h_nu = hamiltonian_terms[M - 1 - i]
        else:
            h_nu = hamiltonian_terms[nu]
        if nu_prime >= M:
            i = nu_prime - M
            h_nu_prime = hamiltonian_terms[M - 1 - i]
        else:
            h_nu_prime = hamiltonian_terms[nu_prime]
        if nu_prime == nu:
            delta = 1.
        else:
            delta = 0.
        # if not commutes(h_nu, h_mu, blocks):
        #     comm_nu_mu = 2. * h_nu * h_mu # = [H_nu, H_mu] when they anti-commute.
        #     if not commutes(h_nu_prime, comm_nu_mu, blocks):
        #         # This term contributes.
        #         comm_three = 2. * h_nu_prime * comm_nu_mu
        #         comm_three_mpo = pauli_string_to_mpo(comm_three, qs)
        #         mat_elem = ground_state.H @ comm_three_mpo.apply(ground_state)
        #         running_estimate += (1 - delta / 2) * mat_elem.real * num_sum
        #     else:
        #         num_zero += 1
        comm_nu_mu = h_nu * h_mu - h_mu * h_nu
        comm_three = h_nu_prime * comm_nu_mu - comm_nu_mu * h_nu_prime
        # print(mu, nu, nu_prime)
        # print(f"p = {p}")
        # print(1 - delta / 2)
        # print(comm_three)
        if len(comm_three) != 0:
            comm_three_mpo = pauli_sum_to_mpo(comm_three, qs, max_mpo_bond)
            mat_elem = ground_state.H @ comm_three_mpo.apply(ground_state)
            running_estimate += (1 - delta / 2) * mat_elem.real / p
            # v2 += (-1 / 24.) * (1 - delta / 2) * comm_three / p
        else:
            num_zero += 1
        if n % interval == 0 or n == samples - 1:
            sample_checkpoints.append((-1 / 24.) * running_estimate / float(n + 1))
            sample_num_checkpoints.append(n+1)
            # print("v2 =", v2 / float(n+1))
    return sample_num_checkpoints, sample_checkpoints, num_zero


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

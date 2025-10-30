"""This analysis is based on https://arxiv.org/abs/2312.13282"""

from typing import List, Dict, Tuple
from random import randint, random
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


def get_unweighted_psum(psum: cirq.PauliSum) -> Tuple[float, cirq.PauliSum]:
    """For a Paulisum sum_i c_i P_i, get the total weight w = sum_i |c_i|
    and the unweighted Paulisum \sum_i c_i P_i / w."""

    w = 0.
    for term in psum:
        w += abs(term.coefficient)
    return (w, psum / w)


def sample_eps2(
    hamiltonian_terms: List[cirq.PauliSum], phi0: MatrixProductState,
    nsamples: int, qs: List[cirq.Qid], max_bond: int
) -> float:
    """Estimate <phi0|V_2|phi0> by sampling commutator terms by the Metropolis-Hastings algorithm.
    Let H_mu = sum_i h_i A_i. The weight is w_mu = |sum_i h_i|.
    The unnormalized probability weight of each commutator is |(1 - delta_nu,nu') w_nu' w_nu w_mu|.
    We will propose steps uniformly at random, so Pr(x_1 -> x_2) == Pr(x_2 -> x_1). The acceptance ratio
    is then just the ratio of the unormalized probabilities. For each sample, we add on
    <phi0|sgn((1-delta/2)w_nu' w_nu w_mu) [A_nu', [A_nu, A_mu]]|phi0>."""

    def _weight_and_summand(mu, nu, nu_prime):
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
        w_mu, a_mu = get_unweighted_psum(h_mu)
        w_nu, a_nu = get_unweighted_psum(h_nu)
        w_nu_prime, a_nu_prime = get_unweighted_psum(h_nu_prime)
        # print(f"weights = {w_mu} {w_nu} {w_nu_prime}")
        weight = (1. - delta / 2.) * w_mu * w_nu * w_nu_prime
        assert weight >= 0.
        comm_nu_mu = a_nu * a_mu - a_mu * a_nu
        # print(f"comm_nu_mu = {comm_nu_mu}")
        comm_three = a_nu_prime * comm_nu_mu - comm_nu_mu * a_nu_prime
        if weight.real > 0.:
            summand = comm_three
        else:
            summand = -comm_three
        return abs(weight), summand

    M = len(hamiltonian_terms)
    eps2 = 0.
    # Start mu, nu, and nu' with some pre-chosen values.
    mu_old = 0
    nu_old = 0
    nu_prime_old = 0
    weight_old, summand_old = _weight_and_summand(mu_old, nu_old, nu_prime_old)
    for ns in range(nsamples):
        # print(f"On sample {ns}.")
        # Propose new mu, nu, nu'.
        mu_new = randint(0, 2 * M - 1)
        nu_new = randint(mu_new + 1, 2 * M)
        nu_prime_new = randint(nu_new, 2 * M)
        weight_new, summand_new = _weight_and_summand(mu_new, nu_new, nu_prime_new)
        # print(f"weight_new = {weight_new}")
        acceptance_ratio = min(1., weight_new / weight_old)
        r = random()
        if r <= acceptance_ratio:
            mu_old = mu_new
            nu_old = nu_new
            nu_prime_old = nu_prime_new
            # print("summand_new = ", summand_new)
            # print(f"summand_new has {len(summand_new)} terms")
            if len(summand_new) != 0:
                term_mpo = pauli_sum_to_mpo(summand_new, qs, max_bond)
                eps2 += (-1. / 24) * mpo_mps_exepctation(term_mpo, phi0)
        else:
            if len(summand_old) != 0:
                term_mpo = pauli_sum_to_mpo(summand_old, qs, max_bond)
                eps2 += (-1. / 24) * mpo_mps_exepctation(term_mpo, phi0)
    return eps2.real / nsamples


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

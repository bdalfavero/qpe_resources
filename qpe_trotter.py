"""This analysis is based on https://arxiv.org/abs/2312.13282"""

from itertools import accumulate
from typing import List, Dict, Tuple, Union
from random import randint, random
import numpy as np
import cirq
import openfermion as of
import qiskit
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
from tensor_network_common import pauli_sum_to_mpo, mpo_mps_exepctation
from qtoolbox.core.pauli import PauliString
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.converters.openfermion_bridge import from_openfermion
from qtoolbox.grouping import sorted_insertion_grouping
from qtoolbox.converters.openfermion_bridge import to_openfermion
from itertools import accumulate
from multiprocessing import Pool, cpu_count

def group_single_strings(ham: cirq.PauliSum) -> List[cirq.PauliSum]:
    """A partitioning where each partition contains a single string."""

    groups: List[cirq.PauliSum] = []
    for h in ham:
        groups.append(cirq.PauliSum.from_pauli_strings([h]))
    return groups


def commutator(a, b):
    return a * b - b * a


def v2_pauli_sum(hamiltonian_terms: List[cirq.PauliSum]) -> cirq.PauliSum:
    """"Calculates V2 the same way as in the get_v2_sarray function of error_pert.py"""
    
    nterms = len(hamiltonian_terms)
    term_sums_l2r = list(accumulate(hamiltonian_terms))
    temp = reversed(hamiltonian_terms)
    term_sums_r2l = list(accumulate(temp))
    term_sums_r2l = list(reversed(term_sums_r2l))
    term_sums_r2l.append(cirq.PauliSum())
    term_combs_V1_v2 = [(term_sums_l2r[i-1], hamiltonian_terms[i], term_sums_r2l[i+1]) for i in range (1, nterms)]
    v2 = cirq.PauliSum()
    for i,j,k in term_combs_V1_v2:
        V1_term = commutator(i, j)
        v2 += - commutator(V1_term, k)*1/3 - commutator(V1_term, j)*1/6
    return v2


def v2_qubop(hamiltonian_terms: List[of.QubitOperator]) -> of.QubitOperator:
    """"Calculates V2 the same way as in the get_v2_sarray function of error_pert.py"""
    
    nterms = len(hamiltonian_terms)
    term_sums_l2r = list(accumulate(hamiltonian_terms))
    temp = reversed(hamiltonian_terms)
    term_sums_r2l = list(accumulate(temp))
    term_sums_r2l = list(reversed(term_sums_r2l))
    term_sums_r2l.append(cirq.PauliSum())
    term_combs_V1_v2 = [(term_sums_l2r[i-1], hamiltonian_terms[i], term_sums_r2l[i+1]) for i in range (1, nterms)]
    v2 = cirq.PauliSum()
    for i,j,k in term_combs_V1_v2:
        V1_term = of.commutator(i, j)
        v2 += - of.commutator(V1_term, k)*1/3 - of.commutator(V1_term, j)*1/6
    return v2


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
        print(f"mu = {mu}/{2 * M - 1}")
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
    hamiltonian: cirq.PauliSum, qs: List[cirq.Qid], ground_state: Union[np.ndarray, MatrixProductState],
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
        comm_nu_mu = h_nu * h_mu - h_mu * h_nu
        comm_three = h_nu_prime * comm_nu_mu - comm_nu_mu * h_nu_prime
        # print(mu, nu, nu_prime)
        # print(f"p = {p}")
        # print(1 - delta / 2)
        # print(comm_three)
        if len(comm_three) != 0:
            if isinstance(ground_state, MatrixProductState):
                comm_three_mpo = pauli_sum_to_mpo(comm_three, qs, max_mpo_bond)
                mat_elem = ground_state.H @ comm_three_mpo.apply(ground_state)
                running_estimate += (1 - delta / 2) * mat_elem.real / p
            else:
                qubit_map = {q: i for i, q in enumerate(qs)}
                mat_elem = comm_three.expectation_from_state_vector(ground_state, qubit_map)
                running_estimate += (1 - delta / 2) * mat_elem.real / p
            v2 += (-1 / 24.) * (1 - delta / 2) * comm_three / p
        else:
            num_zero += 1
        if n % interval == 0 or n == samples - 1:
            sample_checkpoints.append((-1 / 24.) * running_estimate / float(n + 1))
            sample_num_checkpoints.append(n+1)
    return sample_num_checkpoints, sample_checkpoints, v2 / float(samples)


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


def fast_commutator_sum(a_terms, b_terms):
    result = []
    for ai in a_terms:
        for bj in b_terms:
            if not ai.commutes_with(bj):  
                p = ai.multiply(bj)       
                p.coeff *= 2              
                result.append(p)
    return result

def apply_pauli_to_state(x_bits, z_bits, n_qubits, state):
    dim = len(state)
    result = state.copy()

    x_bits_of, z_bits_of = 0, 0
    for q in range(n_qubits):
        if x_bits & (1 << q):
            x_bits_of |= (1 << (n_qubits - 1 - q))
        if z_bits & (1 << q):
            z_bits_of |= (1 << (n_qubits - 1 - q))

    if z_bits_of:
        indices = np.arange(dim)
        parity = np.zeros(dim, dtype=np.int8)
        for b in range(n_qubits):
            if z_bits_of & (1 << b):
                parity ^= ((indices >> b) & 1).astype(np.int8)
        result *= (1 - 2*parity)

    if x_bits_of:
        result = result[np.arange(dim) ^ x_bits_of]

    y_bits = x_bits & z_bits
    if y_bits:
        result *= (1j) ** bin(y_bits).count('1')

    return result

def build_v2_terms(sym_groups):
    nterms = len(sym_groups)
    sums_l2r = list(accumulate(sym_groups, lambda a, b: a + b))
    sums_r2l = list(reversed(list(accumulate(reversed(sym_groups), lambda a, b: a + b))))
    sums_r2l.append([])

    v2_terms = []
    for i in range(1, nterms):
        V1 = fast_commutator_sum(sums_l2r[i-1], sym_groups[i])
        for t in fast_commutator_sum(V1, sums_r2l[i+1]):
            t.coeff *= -1/3
            v2_terms.append(t)
        for t in fast_commutator_sum(V1, sym_groups[i]):
            t.coeff *= -1/6
            v2_terms.append(t)
    return v2_terms

def compute_expectation_sequential(v2_terms, psi, n_qubits):
    eps2 = 0.0
    for t in v2_terms:
        p_state = apply_pauli_to_state(t.x_bits, t.z_bits, n_qubits, psi)
        eps2 += (t.coeff * np.vdot(psi, p_state)).real
    return eps2


def pstring_mps_expectation(
    pstring: PauliString, state: MatrixProductState, qs: List[cirq.Qid],
    max_bond: int = 50
) -> float:
    """Compute the expectation of a PauliString."""

    assert len(qs) == len(state.tensor_map)

    pstring_of = to_openfermion(pstring)
    pstring_cirq = of.transforms.qubit_operator_to_pauli_sum(pstring_of)
    pstring_mpo = pauli_sum_to_mpo(pstring_cirq, qs, max_bond=max_bond)
    return (state.H @ state.gate_with_mpo(pstring_mpo)).real


def compute_expectation_batch(args):
    terms_data, psi, n_qubits = args
    if isinstance(psi, np.ndarray):
        dim = len(psi)
    elif isinstance(psi, MatrixProductState):
        dim = 2 ** len(psi.tensor_map)
    else:
        raise ValueError(f"psi if of type {type(psi)}.")
    qs = cirq.LineQubit.range(n_qubits)
    total = 0.0
    for x_bits, z_bits, coeff in terms_data:
        if isinstance(psi, np.ndarray):
            result = apply_pauli_to_state(x_bits, z_bits, n_qubits, psi)
            total += (coeff * np.vdot(psi, result)).real
        elif isinstance(psi, MatrixProductState):
            pstring = PauliString(x_bits, z_bits, coeff)
            total += pstring_mps_expectation(pstring, psi, qs)
        else:
            raise ValueError(f"psi if of type {type(psi)}.")
    return total

    # terms_data, psi, n_qubits = args
    # dim = len(psi)
    # total = 0.0
    # for x_bits, z_bits, coeff in terms_data:
    #     result = psi.copy()
    #     x_bits_of, z_bits_of = 0, 0
    #     for q in range(n_qubits):
    #         if x_bits & (1 << q):
    #             x_bits_of |= (1 << (n_qubits - 1 - q))
    #         if z_bits & (1 << q):
    #             z_bits_of |= (1 << (n_qubits - 1 - q))
    #     if z_bits_of:
    #         indices = np.arange(dim)
    #         parity = np.zeros(dim, dtype=np.int8)
    #         for b in range(n_qubits):
    #             if z_bits_of & (1 << b):
    #                 parity ^= ((indices >> b) & 1).astype(np.int8)
    #         result = result * (1 - 2*parity)
    #     if x_bits_of:
    #         result = result[np.arange(dim) ^ x_bits_of]
    #     y_bits = x_bits & z_bits
    #     if y_bits:
    #         result = result * ((1j) ** bin(y_bits).count('1'))
    #     total += (coeff * np.vdot(psi, result)).real
    # return total

def compute_expectation_parallel(v2_terms, psi, n_qubits, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count())  
    terms_data = [(t.x_bits, t.z_bits, t.coeff) for t in v2_terms]
    
    batch_size = len(terms_data) // n_workers + 1
    batches = [(terms_data[i:i+batch_size], psi, n_qubits) 
               for i in range(0, len(terms_data), batch_size)]
    
    with Pool(n_workers) as pool:
        results = pool.map(compute_expectation_batch, batches)
    return sum(results)


def to_groups_mpo(groups: List[List[cirq.PauliString]], qs: List[cirq.Qid], max_bond: int) -> List[MatrixProductOperator]:
    """Convert groups from SI into a list of MatrixProductOperators."""

    mpos: List[MatrixProductOperator] = []
    for group in groups:
        group_psum = sum(group)
        group_mpo = pauli_sum_to_mpo(group_psum, qs, max_bond=max_bond)
        mpos.append(group_mpo.copy())
    return mpos

def mpo_add_compress(
    mpo_a: MatrixProductOperator, mpo_b: MatrixProductOperator,
    compress: bool = False, max_bond: int = 100
) -> MatrixProductOperator:
    """Add two MPOs, optionally compressing them."""

    mpo_sum = mpo_a + mpo_b
    if compress:
        mpo_sum.compress(max_bond=max_bond)
    return mpo_sum


def mpo_commutator(mpo_a: MatrixProductOperator, mpo_b: MatrixProductOperator) -> MatrixProductOperator:
    """Take the commutator of two MPOs."""

    return mpo_a.apply(mpo_b) - mpo_b.apply(mpo_a)


def zeros_mpo(nsites: int, phys_dim: int=2) -> MatrixProductOperator:
    """Returns an MPO corresponding to a matirx of all zeros."""

    def fill_fun(shape):
        return np.zeros(shape, dtype=complex)
    
    return MatrixProductOperator.from_fill_fn(fill_fun, L=nsites, bond_dim=1, phys_dim=phys_dim)


def get_v2_contrib_mpo(fragments_list: List[MatrixProductOperator], psi: MatrixProductState, max_bond: int) -> float:
    """Calculate eps2 when the fragments are MPOs."""

    max_mpo_sites = max([len(mpo.tensor_map) for mpo in fragments_list])

    frags_len = len(fragments_list)
    frag_sums_l2r = list(accumulate(fragments_list, lambda a, b: mpo_add_compress(a, b, True, max_bond)))
    temp = reversed(fragments_list)
    frag_sums_r2l = list(accumulate(temp, lambda a, b: mpo_add_compress(a, b, True, max_bond)))
    frag_sums_r2l = list(reversed(frag_sums_r2l))
    frag_sums_r2l.append(zeros_mpo(max_mpo_sites))
    frag_combs_V1_v2 = [(frag_sums_l2r[i-1], fragments_list[i], frag_sums_r2l[i+1]) for i in range (1, frags_len)]
    eps2 = 0
    for i,j,k in frag_combs_V1_v2:
        V1_term = mpo_commutator(i, j)
        term1 = psi.H @ psi.gate_with_mpo(mpo_commutator(V1_term, k))
        term2 = psi.H @ psi.gate_with_mpo(mpo_commutator(V1_term, j))
        eps2 += - term1 *1/3 - term2 *1/6
    return eps2.real
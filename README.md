# QPE Resources

Here we will use the perturbative technique from [this paper](https://arxiv.org/abs/2312.13282) to see how what physical resources we need for quantum phase estimation.

The script `hubbard.py` computes gate counts for the Fermi-Hubbard model. The script expects an input file and an output file. Both are JSON. An example input file is given here:
```
{
    "l": 2,
    "t": 1.0,
    "u": 4.0,
    "max_mpo_bond": 100,
    "max_mps_bond": 15,
    "energy_error": 0.001,
    "evol_time": 0.1
}
```

## Notes on choosing $t$

For vanilla QPE:
1. Clearly we cannot make $t$ arbitrarily small. We only get $m$ bits of $tE$, so we want to ensure that $tE$ is neither too small nor too large.

For SAPE:
1. [The Harrow paper](https://arxiv.org/abs/2503.05647) says we need a time $t = 2^M$, where $M$ is the number of bits we need.
1. [This paper](https://quantum-journal.org/papers/q-2023-11-06-1165/) is cited by the Harrow paper as an example of robust QPE. They just frame everything in terms of a unitary $U$, not specifically getting energies from a Hamiltonian.
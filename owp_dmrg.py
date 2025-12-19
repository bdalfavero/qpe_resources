from warnings import warn
import quimb.tensor as qtn
from quimb.utils import load_from_disk, save_to_disk

def main():
    max_mps_bond = 25
    owp_mpo = load_from_disk("data/owp_mpo_chi_1000.dat")
    dmrg = qtn.DMRG(owp_mpo, bond_dims=max_mps_bond)
    converged = dmrg.solve(max_sweeps=100)
    if not converged:
        warn("DMRG did not converge!")
    psi = dmrg.state
    energy = dmrg.energy.real
    print(f"Got ground state energy {energy:5.6e}.")
    save_to_disk(psi, f"data/owp_ground_state_{max_mps_bond}.dat")

if __name__ == "__main__":
    main()
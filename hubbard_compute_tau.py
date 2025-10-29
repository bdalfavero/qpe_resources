"""See Section 4 of 'On Low Depth Algorithms for Quantum Phase Estimation'
https://quantum-journal.org/papers/q-2023-11-06-1165/"""

import argparse
import json
import numpy as np
from scipy.sparse.linalg import norm
import openfermion as of

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file for input.")
    # parser.add_argument("output_file", type=str, help="JSON file for ouptut.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    l = input_dict["l"]
    t = input_dict["t"]
    u = input_dict["u"]

    ham = of.fermi_hubbard(l, l, t, u, spinless=True)
    ham_jw = of.transforms.jordan_wigner(ham)
    ham_sparse = of.linalg.get_sparse_operator(ham_jw)

    ham_norm = norm(ham_sparse, ord=2)
    tau = np.pi / (4. * ham_norm)
    print(f"tau = {tau:4.5e}")

if __name__ == "__main__":
    main()
import json
import subprocess

t = 1.0
u = 4.0
max_mpo_bond = 100
max_mps_bond = 15
energy_error = 1e-3
evol_time = 0.1
l_values = [2, 4, 6, 8]
for l in l_values:
    print(f"Starting l = {l}")
    input_dict = {
        "l": l,
        "t": t,
        "u": u,
        "max_mpo_bond": max_mpo_bond,
        "max_mps_bond": max_mps_bond,
        "energy_error": energy_error,
        "evol_time": evol_time
    }
    input_fname = f"data/hubbard_l{l}_in.json"
    with open(input_fname, "w") as f:
        json.dump(input_dict, f)

    output_fname = f"data/hubbard_l{l}_out.json"
    status = subprocess.run(["python", "hubbard.py", input_fname, output_fname])
    status.check_returncode()
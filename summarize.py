#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys
import numpy as np
from hybrid_mdmc.filehandlers import hkmcmd_ArgumentParser
from hybrid_mdmc.system import *


def main(argv):
    parser = hkmcmd_ArgumentParser()
    args = parser.parse_args()
    summarize_system_state(args.system, args.prefix)
    return


def summarize_system_state(system, prefix):
    system_data = SystemData(system, f"{prefix}")
    system_data.read_json()
    system_state = SystemState(f"{prefix}.system_state.json")
    system_state.read_data_from_json()
    system_state.clean()
    progression_df = system_state.assemble_progression_df(sorted([mol.kind for mol in system_data.species]))
    data = progression_df.to_numpy()
    header = [str(_) for _ in progression_df.columns]
    np.savetxt(f"{prefix}.summary.txt", data, header="    ".join(header), comments="# ")
    return


if __name__ == "__main__":
    main(sys.argv[1:])

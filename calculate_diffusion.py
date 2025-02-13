#!/usr/bin/env python3
#
# Author:
#    Dylan Gilley
#    dgilley@purdue.edu


import argparse, sys, os
import numpy as np
import pandas as pd
from copy import deepcopy
from hybrid_mdmc import utility
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.diffusion import Diffusion
from hybrid_mdmc.sensitivity_analysis_diffusion_rates import (
    read_mvabfa_file,
    calculate_and_write_mvabfa,
)
from hybrid_mdmc.customargparse import HMDMC_ArgumentParser
from hybrid_mdmc.voxels import Voxels
from hybrid_mdmc.functions import gen_molecules
from hybrid_mdmc.mean_displacement import read_msd_file


def main(argv):

    # Use HMDMC_ArgumentParser to parse the command line.
    parser = HMDMC_ArgumentParser(auto=False)
    parser.add_default_args()
    parser.add_argument(
        "-diffusion_species",
        dest="diffusion_species",
        type=str,
        default="all",
        help='Species for which to calculate diffusion rates. Default is "all".',
    )
    parser.add_argument(
        "-method",
        dest="method",
        type=str,
        default="average_msd",
        help="Diffusion rate method to use. Default is 'average_msd'.",
    )
    parser.add_argument(
        "-filename_msd",
        dest="filename_msd",
        type=str,
        default="default",
        help="Filename of MSD file.",
    )
    setattr(parser, "args", parser.parse_args())
    if parser.args.filename_notebook == "default":
        parser.args.filename_notebook = parser.args.system + "_notebook.xlsx"
    if parser.args.filename_msd == "default":
        parser.args.filename_msd = parser.args.prefix + ".msdoutput.txt"
    if os.path.isfile(parser.args.filename_notebook):
        parser.read_notebook()
    parser.adjust_default_args()
    args = parser.args
    args.diffusion_species = args.diffusion_species.split()

    # Read in the data_file, diffusion_file, rxndf, and msf files.
    datafile_result = parse_data_file(
        args.filename_data, unwrap=True, atom_style=args.atom_style
    )
    atoms, box = datafile_result[0], datafile_result[5]
    masterspecies = parser.get_masterspecies_dict()

    # Create the Voxels object
    voxels_datafile = Voxels(box, args.number_of_voxels)
    atomtypes2moltype = {
        tuple(sorted([i[2] for i in v["Atoms"]])): k for k, v in masterspecies.items()
    }

    # Create and populate an instance of the "MoleculeList" class.
    molecules = gen_molecules(atoms, atomtypes2moltype, voxels_datafile)

    if args.method == "direct_flux":
        filename_mvabfa = args.prefix + ".mvabfa.txt"
        file_mvabfa = utility.FileTracker(filename_mvabfa)
        calculate_and_write_mvabfa(
            args.prefix,
            args.filename_trajectory,
            atoms,
            molecules,
            voxels_datafile,
            args.lammps_time_units_to_seconds_conversion,
            [0, -1, 1],
            file_mvabfa,
            logfile=None,
        )
        _, mvabfa_molecule_types, mvabfa, mvabfa_timesteps = read_mvabfa_file(
            filename_mvabfa
        )
        diffusion = Diffusion(
            args.prefix,
            args.filename_trajectory,
            atoms,
            molecules,
            voxels_datafile,
            time_conversion=args.lammps_time_units_to_seconds_conversion,
        )
        diffusion.molecular_voxel_assignments_by_frame_array = deepcopy(mvabfa)
        diffusion.timesteps = deepcopy(mvabfa_timesteps)
        diffusion.calculate_direct_voxel_transition_rates(
            average_across_voxel_neighbors=True
        )
        file_direct_rates = utility.FileTracker(args.prefix + ".direct_rates.txt")
        bar = "".join((["#"] * 100))
        file_direct_rates.write(f"\n\n{bar}\n")
        for sp in diffusion.direct_voxel_transition_rates.keys():
            output_df = pd.DataFrame(diffusion.direct_voxel_transition_rates[sp])
            file_direct_rates.write(f"\n\nSpecies {sp}\n")
            file_direct_rates.write(output_df.to_string(index=False, header=False))
        diffusion.calculate_diffusion_rates(
            starting_position_idxs=np.array(
                [list(range(np.prod(voxels_datafile.number_of_voxels)))] * 10
            ).flatten(),  # effectively 10 walks with a walker starting in each voxel
            number_of_steps=900,  # sensitivity analysis shows good response to 900
        )
        diffusion_rate = diffusion.diffusion_rates

    elif args.method == "average_msd":
        msd_data = read_msd_file(args.filename_msd)
        last_diffusion_step = np.max(list(msd_data.keys()))
        species = sorted(list(msd_data[last_diffusion_step][1].keys()))
        frames = np.array([float(_) for _ in msd_data[last_diffusion_step][0]]).flatten()
        frames = frames[1:] - frames[0]
        frames *= args.lammps_time_units_to_seconds_conversion
        mean_displacements = {
            sp: msd_data[last_diffusion_step][1][sp]["mean_displacement_avg"].flatten()
            for sp in species
        }
        diffusion_rate = {
            sp: np.zeros(
                (
                    np.prod(voxels_datafile.number_of_voxels),
                    np.prod(voxels_datafile.number_of_voxels),
                )
            )
            for sp in species
        }
        for sp in species:
            if sp not in args.diffusion_species and "all" not in args.diffusion_species:
                diffusion_rate[sp].fill(np.inf)
                continue
            mean_displacement_rate = np.mean(mean_displacements[sp] / frames)
            separation_distances = sorted(
                list(voxels_datafile.separation_groupings_dict.keys())
            )
            voxels_datafile.voxel_idxs_by_distance_groupings
            for i, sep in enumerate(separation_distances):
                diffusion_rate[sp][
                    voxels_datafile.voxel_idxs_by_distance_groupings[i]
                ] = mean_displacement_rate/sep


    file_diffusion = utility.FileTracker(args.filename_diffusion)
    file_diffusion.write(
        "\n\n---------------------------------------------------------------------------\n\n"
    )
    file_diffusion.write(f"\nDiffusionStep 0\n")
    for k, v in sorted(diffusion_rate.items()):
        output_df = pd.DataFrame(v)
        file_diffusion.write(f"\n\nDiffusion Rates for {k}\n")
        file_diffusion.write(output_df.to_string(index=False, header=False))

    return


if __name__ == "__main__":
    main(sys.argv)

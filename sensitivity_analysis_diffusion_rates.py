#!/usr/bin/env python3
#
# Author:
#    Dylan Gilley
#    dgilley@purdue.edu


import sys, os, datetime

import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Union

from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.diffusion import (
    Diffusion,
    calculate_molecular_voxel_assignments_by_frame_array,
)
from hybrid_mdmc.customargparse import HMDMC_ArgumentParser
from hybrid_mdmc.voxels import Voxels
from hybrid_mdmc.classes import MoleculeList
from hybrid_mdmc.mol_classes import AtomList
from functions import gen_molecules
from hybrid_mdmc import utility


def main(argv):

    # Use HMDMC_ArgumentParser to parse the command line.
    parser = HMDMC_ArgumentParser(auto=False)
    parser.add_default_args()
    parser.add_argument(
        "-species",
        dest="species",
        type=str,
        default="all",
        help="Species for which to conduct sensitivity analyses. Default: 'all'.",
    )
    parser.add_argument(
        "-trajectory_frames",
        dest="trajectory_frames",
        default="0 -1 1",
        type=str,
        help="Trajectory frames to use; 'start end every'. Default: '0 -1 1'",
    )
    parser.add_argument(
        "-direct_transition_frames",
        dest="direct_transition_frames",
        default="2 -1 1",
        type=str,
        help="Direct transition frames to use; 'start end every'. Default: '0 -1 1'",
    )
    parser.add_argument(
        "-random_walk_lengths",
        dest="random_walk_lengths",
        default="100 200 100",
        type=str,
        help="Random walk lengths to use; 'start end every'. Default: '100 200 100'.",
    )
    parser.add_argument(
        "-random_walk_replicates",
        dest="random_walk_replicates",
        default=10,
        type=int,
        help="Number of random walk replicates. Default: 10.",
    )
    parser.add_argument(
        "-random_walk_recursion_interval",
        dest="random_walk_recursion_interval",
        default=1,
        type=int,
        help="Recursion interval for random walk. Default: 1.",
    )
    parser.add_argument(
        "--calculate_direct_transition_sensitivity",
        dest="calculate_direct_transition_sensitivity",
        action="store_true",
        help="Calculate the sensitivity of direct transition rates to the diffusion rate.",
    )
    parser.add_argument(
        "--calculate_random_walk_sensitivity",
        dest="calculate_random_walk_sensitivity",
        action="store_true",
        help="Calculate the sensitivity of random walk rates to the diffusion rate.",
    )

    # Parse the command line.
    setattr(parser, "args", parser.parse_args())
    if parser.args.filename_notebook == "default":
        parser.args.filename_notebook = parser.args.system + "_notebook.xlsx"
    if os.path.isfile(parser.args.filename_notebook):
        parser.read_notebook()
    parser.adjust_default_args()
    args = parser.args

    # Adjust arguments
    args.species = args.species.split()
    trajectory_frames = [int(_) for _ in args.trajectory_frames.split()]
    direct_transition_frames = [int(_) for _ in args.direct_transition_frames.split()]
    random_walk_lengths = [int(_) for _ in args.random_walk_lengths.split()]
    random_walk_lengths = np.arange(
        random_walk_lengths[0], random_walk_lengths[1], random_walk_lengths[2]
    )
    random_walk_replicates = np.arange(1, args.random_walk_replicates + 1)
    random_walk_recursion_interval = int(args.random_walk_recursion_interval)

    # Prepare system
    file_log = utility.FileTracker(args.prefix + ".sa.log.txt")
    datafile_result = parse_data_file(
        args.filename_data, unwrap=True, atom_style=args.atom_style
    )
    atoms_datafile, box_datafile = datafile_result[0], datafile_result[5]
    masterspecies = parser.get_masterspecies_dict()
    species = [
        k for k in masterspecies.keys() if k in args.species or args.species == ["all"]
    ]
    voxels_datafile = Voxels(box_datafile, args.number_of_voxels)
    atomtypes2moltype = {
        tuple(sorted([i[2] for i in v["Atoms"]])): k for k, v in masterspecies.items()
    }
    molecules_datafile = gen_molecules(
        atoms_datafile, atomtypes2moltype, voxels_datafile
    )
    filename_mvabfa = (
        args.prefix + ".sa.mvabfa.txt"
    )  # name of file containing the "molecular voxel assignments by frame array" ("mvabfa")

    # Voxel transition rate sensitivity
    if args.calculate_direct_transition_sensitivity is True:
        file_log.write(
            f"\ncalculating direct transition rate sensitivity analysis... {datetime.datetime.now()} \n\n"
        )
        # If a molecular voxel assignments by frame array file does not exist, the trajectory file must be parsed to create one.
        if not os.path.isfile(filename_mvabfa):
            file_log.write(f"  {filename_mvabfa} not found.\n")
            file_log.write(
                f"  parsing {args.filename_trajectory} (frame {trajectory_frames[0]} to {trajectory_frames[1]} by every {trajectory_frames[2]} frames)... {datetime.datetime.now()} \n\n"
            )
            calculate_and_write_mvabfa(
                args.prefix,
                args.filename_trajectory,
                atoms_datafile,
                molecules_datafile,
                voxels_datafile,
                args.lammps_time_units_to_seconds_conversion,
                trajectory_frames,
                filename_mvabfa,
                logfile=file_log,
            )

        # Read the molecular voxel assignments by frame array file
        _, mvabfa_molecule_types, mvabfa, mvabfa_timesteps = read_mvabfa_file(
            filename_mvabfa
        )

        # Create the output files
        output_files = [
            utility.FileTracker(f"{args.prefix}.sa.direct_rates.{sp}.txt")
            for sp in species
        ]

        # Loop over each total number of frames, create a Diffusion instance, manually set the
        # molecular voxel assignments by frame array, calcule the direct transition rates, and
        # write the result to the output file.
        if direct_transition_frames[1] == -1:
            direct_transition_frames[1] = mvabfa.shape[0]
        for total_frames in np.arange(
            direct_transition_frames[0],
            direct_transition_frames[1],
            direct_transition_frames[2],
        ):
            if total_frames > mvabfa.shape[0]:
                break
            file_log.write(
                f"  calculating direct transition rates using the first {total_frames} trajectory frames... {datetime.datetime.now()} \n\n"
            )

            diffusion = Diffusion(
                args.prefix,
                args.filename_trajectory,
                atoms_datafile,
                molecules_datafile,
                voxels_datafile,
                time_conversion=args.lammps_time_units_to_seconds_conversion,
            )
            diffusion.molecular_voxel_assignments_by_frame_array = deepcopy(mvabfa)
            diffusion.molecular_voxel_assignments_by_frame_array = (
                diffusion.molecular_voxel_assignments_by_frame_array[:total_frames]
            )
            diffusion.timesteps = deepcopy(mvabfa_timesteps)
            diffusion.timesteps = diffusion.timesteps[:total_frames]
            diffusion.calculate_direct_voxel_transition_rates()
            for spidx, sp in enumerate(species):
                bar = "".join((["#"] * 100))
                output_df = pd.DataFrame(diffusion.direct_voxel_transition_rates[sp])
                output_files[spidx].write(f"\n\n{bar}\n")
                output_files[spidx].write(f"NumberOfFrames {total_frames}\n")
                output_files[spidx].write(
                    f"TrajectoryFrame {mvabfa_timesteps[total_frames]}\n"
                )
                output_files[spidx].write(
                    f"TotalTime {(diffusion.timesteps[-1]-diffusion.timesteps[0])*diffusion.time_conversion}\n"
                )
                output_files[spidx].write(f"DirectTransitionRates\n")
                output_files[spidx].write(
                    output_df.to_string(index=False, header=False)
                )

    # Random walk rate sensitivity
    if args.calculate_random_walk_sensitivity is True:
        file_log.write(
            f"\ncalculating random walk rate sensitivity analysis... {datetime.datetime.now()} \n\n"
        )

        direct_voxel_transition_rates = {}
        for sp in species:
            (
                number_of_frames,
                trajectory_frames,
                total_time,
                direct_voxel_transition_rates[sp],
            ) = read_direct_rates_files(f"{args.prefix}.sa.direct_rates.{sp}.txt")

        # declare output files
        output_files = [
            utility.FileTracker(f"{args.prefix}.sa.random_walk_rates.{sp}.txt")
            for sp in species
        ]

        # loop over walk lengths
        for walk_length in random_walk_lengths:
            file_log.write(
                f"  calculating random walk rates using a walk length of {walk_length}... {datetime.datetime.now()} \n"
            )

            # loop over walk replicates
            for rep in random_walk_replicates:
                file_log.write(f"    replicate {rep}... {datetime.datetime.now()} \n")

                # declare new diffusion instance
                diffusion = Diffusion(
                    args.prefix,
                    args.filename_trajectory,
                    atoms_datafile,
                    molecules_datafile,
                    voxels_datafile,
                    time_conversion=args.lammps_time_units_to_seconds_conversion,
                )

                # manually assign direct transition rates
                diffusion.direct_voxel_transition_rates = deepcopy(
                    direct_voxel_transition_rates
                )
                diffusion.direct_voxel_transition_rates = {
                    k: v[sorted(v.keys())[-1]]
                    for k, v in diffusion.direct_voxel_transition_rates.items()
                }

                # calculate random walk rates
                diffusion.calculate_diffusion_rates(
                    starting_position_idxs=np.arange(
                        diffusion.direct_voxel_transition_rates[species[0]].shape[0]
                    ),
                    number_of_steps=walk_length,
                    species=species,
                    recursion_interval=random_walk_recursion_interval,
                )

                # write to file
                for spidx, sp in enumerate(species):
                    bar = "".join((["#"] * 100))
                    output_df = pd.DataFrame(diffusion.diffusion_rates[sp])
                    output_files[spidx].write(f"\n\n{bar}\n")
                    output_files[spidx].write(f"RandomWalkLength {walk_length}\n")
                    output_files[spidx].write(f"RandomWalkReplicate {rep}\n")
                    output_files[spidx].write(f"DiffusionRates\n")
                    output_files[spidx].write(
                        output_df.to_string(index=False, header=False)
                    )

    return


def calculate_and_write_mvabfa(
    prefix: str,
    filename_trajectory: str,
    atoms_datafile: AtomList,
    molecules_datafile: MoleculeList,
    voxels_datafile: Voxels,
    time_conversion: float,
    trajectory_frames: list,
    filename_mvabfa: str,
    logfile: Union[None, utility.FileTracker] = None,
) -> None:
    """Calculate molecular voxel assignments by frame array and write to file.

    Parses the trajectory file to create the molecular voxel assignments by frame array. This array
    is then printed to a file, "filename_mvabfa".

    Parameters
    ----------
    prefix (str) : Name passed to Diffusion instance.
    filename_trajectory (str) : Filename of trajectory file.
    atoms_datafile (AtomList) : AtomList instance, created from datafile information.
    molecules_datafile (MoleculeList) : MoleculeList instance, created from datafile information.
    voxels_datafile (Voxels) : Voxels instance.
    time_conversion (float) : Time conversion factor for Diffusion instance.
    trajectory_frames (list) : List of trajectory frames to parse, [start, end, every].
    filename_mvabfa (str) : Filename of "molecular voxel assignments by frame array" file.

    Returns
    -------
    None
    """

    diffusion = Diffusion(
        prefix,
        filename_trajectory,
        atoms_datafile,
        molecules_datafile,
        voxels_datafile,
        time_conversion=time_conversion,
    )
    diffusion.calculate_molecular_voxel_assignments_by_frame(
        start=trajectory_frames[0],
        end=trajectory_frames[1],
        every=trajectory_frames[2],
        logfile=logfile,
    )
    output = np.concatenate(
        (
            molecules_datafile.ids.reshape(1, -1),
            molecules_datafile.mol_types.reshape(1, -1),
            diffusion.molecular_voxel_assignments_by_frame_array,
        )
    )
    output = pd.DataFrame(
        output,
        index=["MoleculeIDs", "MoleculeTypes"]
        + [
            f"Frame {i}"
            for i in range(
                1,
                diffusion.molecular_voxel_assignments_by_frame_array.shape[0] + 1,
            )
        ],
    )
    file_mvabfa = utility.FileTracker(filename_mvabfa)
    file_mvabfa.write(f"# array calculated by parsing {filename_trajectory}\n\n\n")
    file_mvabfa.write(output.to_string(index=True, header=False))
    timesteps_string = " ".join(map(str, diffusion.timesteps))
    file_mvabfa.write(f"\n\ntimesteps {timesteps_string}\n")

    return


def read_mvabfa_file(
    filename_mvabfr: str,
) -> tuple:
    """Read molecular voxel assignments by frame array from file.

    Parameters
    ----------
    filename_mvabfr (str) : Filename of "molecular voxel assignments by frame array" file.

    Returns
    -------
    list
        IDs of M molecules.
    list
        Types of M molecules.
    np.ndarray
        Molecular voxel assignments by frame array, N frames by M molecules.
    list
        Timestep for each of the N frames.
    """
    mvabfa = {}
    with open(filename_mvabfr) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 0:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "MoleculeIDs":
                molecule_ids = fields[1:]
                continue
            if fields[0] == "MoleculeTypes":
                molecule_types = fields[1:]
                continue
            if fields[0] == "Frame":
                mvabfa[int(fields[1])] = [int(_) for _ in fields[2:]]
                continue
            if fields[0] == "timesteps":
                timesteps = [int(_) for _ in fields[1:]]
                continue
    return (
        molecule_ids,
        molecule_types,
        np.array([v for k, v in sorted(mvabfa.items())]),
        timesteps,
    )


def read_direct_rates_files(filename):
    direct_transition_rates = {}
    number_of_frames, trajectory_frame, total_time = [], [], []
    with open(filename, "r") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 0:
                continue
            if "#" in fields[0]:
                continue
            if fields[0] == "NumberOfFrames":
                number_of_frames.append(int(fields[1]))
                continue
            if fields[0] == "TrajectoryFrame":
                trajectory_frame.append(int(fields[1]))
                continue
            if fields[0] == "TotalTime":
                total_time.append(float(fields[1]))
                continue
            if fields[0] == "DirectTransitionRates":
                flag = True
                voxel_idx = 0
                direct_transition_rates[number_of_frames[-1]] = {}
                continue
            if flag:
                direct_transition_rates[number_of_frames[-1]][voxel_idx] = [
                    float(_) for _ in fields
                ]
                voxel_idx += 1
                continue
    return (
        number_of_frames,
        trajectory_frame,
        total_time,
        {
            k: np.array([vv for kk, vv in sorted(v.items())])
            for k, v in sorted(direct_transition_rates.items())
        },
    )


if __name__ == "__main__":
    main(sys.argv)

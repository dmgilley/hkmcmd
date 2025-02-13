#!/usr/bin/env python3
#
# Author:
#    Dylan Gilley
#    dgilley@purdue.edu


import argparse, sys
import numpy as np
import pandas as pd
from copy import deepcopy
from hybrid_mdmc import utility
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.frame_generator import frame_generator
from hybrid_mdmc.parsers import read_notebook
from hybrid_mdmc.functions import gen_molecules


def main(argv):

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="label")
    parser.add_argument("-atom_style", dest="atom_style", default="full")
    parser.add_argument("-filename_data", dest="filename_data", default="default")
    parser.add_argument(
        "-filename_notebook", dest="filename_notebook", default="default"
    )
    parser.add_argument("-filename_trj", dest="filename_trj", default="default")
    parser.add_argument("-filename_output", dest="filename_output", default="default")
    parser.add_argument("-parsing_frames", dest="parsing_frames", default="0 -1 1")
    parser.add_argument("-step", dest="step", default="0")
    args = parser.parse_args()

    # Set default filenames, format args
    if args.filename_data == "default":
        args.filename_data = args.label + ".end.data"
    if args.filename_notebook == "default":
        args.filename_notebook = args.label.split("-")[0] + "_notebook.xlsx"
    if args.filename_trj == "default":
        args.filename_trj = args.label + ".diffusion.lammpstrj"
    if args.filename_output == "default":
        args.filename_output = args.label + ".msdoutput.txt"
    args.parsing_frames = [int(_) for _ in args.parsing_frames.split()]

    # Initialize handler
    handler = MSDHandler(
        args.label,
        args.atom_style,
        args.filename_data,
        args.filename_notebook,
        args.filename_trj,
        args.filename_output,
        parsing_frames=args.parsing_frames,
    )
    handler.read_data()
    handler.read_notebook()
    handler.parse_trajectory_file()
    handler.calculate_msd()

    output_file = utility.FileTracker(handler.filename_output)
    output_file.write_separation()
    for species in sorted(handler.mean_displacements[0].keys()):
        output_file.write(f"\nstep {args.step}\n")
        output_file.write_array(
            ("frames ", np.array([int(_) for _ in handler.trajectory_frames]))
        )
        output_file.write(f"species {species}\n")
        output_file.write_array(
            ("mean_displacement_avg ", handler.mean_displacements[0][species])
        )
        output_file.write_array(
            ("mean_displacement_std ", handler.mean_displacements[1][species])
        )
        output_file.write_array(
            (
                "mean_squared_displacement_avg ",
                handler.mean_squared_displacements[0][species],
            )
        )
        output_file.write_array(
            (
                "mean_squared_displacement_std ",
                handler.mean_squared_displacements[1][species],
            )
        )

    return


class MSDHandler:

    def __init__(
        self,
        label,
        atom_style,
        filename_data,
        filename_notebook,
        filename_trj,
        filename_output,
        parsing_frames=[0, -1, 1],
    ):
        self.label = label
        self.atom_style = atom_style
        self.filename_data = filename_data
        self.filename_notebook = filename_notebook
        self.filename_trj = filename_trj
        self.filename_output = filename_output
        self.parsing_frames = parsing_frames
        self.adjacency_list = None
        self.notebook_dict = None
        self.atomtypes2moltype = None
        self.trajectory_molecular_cog_array = None
        self.trajectory_molecule_types = None
        self.trajectory_frames = None
        self.mean_displacements = None
        self.mean_squared_displacements = None
        return

    def read_data(self):
        atoms, bonds, angles, dihedrals, impropers, box, adj_mat, extra_prop = (
            parse_data_file(self.filename_data, unwrap=True, atom_style=self.atom_style)
        )
        self.adjacency_list = [
            [count_j for count_j, j in enumerate(i) if j != 0]
            for count_i, i in enumerate(adj_mat)
        ]
        return

    def read_notebook(self):
        self.notebook_dict = read_notebook(self.filename_notebook)
        self.atomtypes2moltype = {
            tuple(sorted([i[2] for i in v["Atoms"]])): k
            for k, v in self.notebook_dict["masterspecies"].items()
        }
        return

    def parse_trajectory_file(self):
        (
            self.trajectory_molecular_cog_array,
            self.trajectory_molecule_types,
            self.trajectory_frames,
        ) = assemble_molecular_cog(
            self.filename_trj,
            self.adjacency_list,
            self.atomtypes2moltype,
            start=self.parsing_frames[0],
            end=self.parsing_frames[1],
            every=self.parsing_frames[2],
        )
        return

    def calculate_msd(self):
        (
            mean_displacements_avg,
            mean_displacements_std,
            mean_squared_displacements_avg,
            mean_squared_displacements_std,
        ) = calculate_msd(
            self.trajectory_molecular_cog_array,
            self.trajectory_molecule_types,
            self.trajectory_frames,
        )
        self.mean_displacements = (mean_displacements_avg, mean_displacements_std)
        self.mean_squared_displacements = (
            mean_squared_displacements_avg,
            mean_squared_displacements_std,
        )
        return


def assemble_molecular_cog(
    filename_trj, adj_list, atomtypes2moltype, start=0, end=-1, every=1
):
    molecular_cog = {}
    molecular_ids = None
    for atoms_thisframe, timestep_thisframe, box_thisframe in frame_generator(
        filename_trj,
        start=start,
        end=end,
        every=every,
        unwrap=True,
        adj_list=adj_list,
        return_prop=False,
    ):
        molecules_thisframe = gen_molecules(atoms_thisframe, atomtypes2moltype)
        if molecular_ids is None:
            molecular_ids = molecules_thisframe.ids
        if not np.all(molecular_ids == molecules_thisframe.ids):
            raise ValueError("Molecular IDs do not match across frames.")
        molecules_thisframe.get_cog(atoms_thisframe)
        molecular_cog[timestep_thisframe] = molecules_thisframe.cogs
    frames = sorted(list(molecular_cog.keys()))
    molecular_cog = [v for k, v in sorted(molecular_cog.items())]
    molecular_cog_array = np.array(molecular_cog)
    molecular_cog_array = np.transpose(molecular_cog_array, (1, 2, 0))
    return (
        molecular_cog_array,
        molecules_thisframe.mol_types,
        frames,
    )


def calculate_msd(molecular_cog_array, molecule_types, frames):

    number_of_frames = len(frames)
    molecule_types_set = sorted(list(set(molecule_types)))
    dict_ = {mt: np.zeros(number_of_frames - 1) for mt in molecule_types_set}
    mean_displacements_avg = deepcopy(dict_)
    mean_displacements_std = deepcopy(dict_)
    mean_squared_displacements_avg = deepcopy(dict_)
    mean_squared_displacements_std = deepcopy(dict_)
    for mt in molecule_types_set:
        cog_array = deepcopy(molecular_cog_array)
        cog_array = cog_array[molecule_types == mt]
        for separation in range(1, number_of_frames):
            diffs = [
                cog_array[:, :, (i + separation)] - cog_array[:, :, i]
                for i in range(number_of_frames - separation)
            ]
            displacements = np.array(
                [np.sqrt(np.sum(diff**2, axis=1)) for diff in diffs]
            ).flatten()
            squared_displacements = displacements**2
            mean_displacements_avg[mt][separation - 1] = np.mean(displacements)
            mean_displacements_std[mt][separation - 1] = np.std(displacements)
            mean_squared_displacements_avg[mt][separation - 1] = np.mean(
                squared_displacements
            )
            mean_squared_displacements_std[mt][separation - 1] = np.std(
                squared_displacements
            )
    return (
        mean_displacements_avg,
        mean_displacements_std,
        mean_squared_displacements_avg,
        mean_squared_displacements_std,
    )


def read_msd_file(filename):
    result = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    step = None
    frames = []
    species = None
    data = {}
    
    for line in lines:
        line = line.strip()
        if len(line.split()) == 0:
            continue
        if line.split()[0] == '#':
            continue
        if "-" in line.split()[0]:
            continue
        if line.startswith('step'):
            if step != int(line.split()[1]):
                if step is not None:
                    result[step] = (frames, data)
                step = int(line.split()[1])
                frames = []
                data = {}
        elif line.startswith('frames'):
            frames = list(map(int, line.split()[1:]))
        elif line.startswith('species'):
            species = line.split()[1]
            data[species] = {
                'mean_displacement_avg': [],
                'mean_displacement_std': [],
                'mean_squared_displacement_avg': [],
                'mean_squared_displacement_std': []
            }
        elif line.startswith('mean_displacement_avg'):
            data[species]['mean_displacement_avg'] = np.array(list(map(float, line.split()[1:])))
        elif line.startswith('mean_displacement_std'):
            data[species]['mean_displacement_std'] = np.array(list(map(float, line.split()[1:])))
        elif line.startswith('mean_squared_displacement_avg'):
            data[species]['mean_squared_displacement_avg'] = np.array(list(map(float, line.split()[1:])))
        elif line.startswith('mean_squared_displacement_std'):
            data[species]['mean_squared_displacement_std'] = np.array(list(map(float, line.split()[1:])))
    
    if step is not None:
        result[step] = (frames, data)
    
    return result


if __name__ == "__main__":
    main(sys.argv[1:])

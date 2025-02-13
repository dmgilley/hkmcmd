#!/usr/bin/env python3
#
# Author:
#    Dylan Gilley
#    dgilley@purdue.edu


import argparse, sys
import numpy as np
import pandas as pd
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.frame_generator import frame_generator
from hybrid_mdmc.parsers import read_notebook
from hybrid_mdmc.functions import calculate_HKMCMD_molecule_types_from_LAMMPS_atom_types


def main(argv):

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="replicate_name")
    parser.add_argument(dest="HKMCMD_molecule_types")
    parser.add_argument(
        "-filename_notebook", dest="filename_notebook", default="default"
    )
    parser.add_argument("-filename_data", dest="filename_data", default="default")
    parser.add_argument("-filename_trj", dest="filename_trj", default="default")
    parser.add_argument("-filename_output", dest="filename_output", default="default")
    parser.add_argument("-atom_style", dest="atom_style", default="full")
    parser.add_argument("-frames", dest="frames", default="0 1 1000")
    args = parser.parse_args()

    # Set default filenames
    if args.filename_notebook == "default":
        args.filename_notebook = args.replicate_name + "_notebook.xlsx"
    if args.filename_data == "default":
        args.filename_data = args.replicate_name + ".in.data"
    if args.filename_trj == "default":
        args.filename_trj = args.replicate_name + ".diffusion.lammpstrj"
    if args.filename_output == "default":
        args.filename_output = args.replicate_name + ".msdoutput.txt"

    # Format arguments
    args.HKMCMD_molecule_types = args.HKMCMD_molecule_types.split()
    args.frames = [int(_) for _ in args.frames.split()]


    atoms, bonds, angles, dihedrals, impropers, box, adj_mat, extra_prop = (
        parse_data_file(args.filename_data, unwrap=True, atom_style=args.atom_style)
    )
    masterspecies = parser.get_masterspecies_dict()
    atomtypes2moltype = {
        tuple(sorted([i[2] for i in v["Atoms"]])): k for k, v in masterspecies.items()
    }
    molecules = gen_molecules(atoms, atomtypes2moltype)

    # Initialize MSDHandler and calculate MSD
    msd = MSDHandler(
        args.replicate_name,
        args.HKMCMD_molecule_types,
        filename_notebook=args.filename_notebook,
        filename_data=args.filename_data,
        filename_trj=args.filename_trj,
        filename_output=args.filename_output,
        atom_style=args.atom_style,
        frames=args.frames,
    )
    msd.read_notebook()
    msd.parse_data_file()
    msd.get_centers_of_mass()
    msd.calculate_msd()

    # Write output
    with open(msd.filename_output, "w") as f:
        f.write("timesteps\n{}\n\n".format([_ for _ in msd.timesteps]))
        f.write("boxes\n{}\n\n".format([_ for _ in msd.boxes]))
        f.write("msd_mean\n{}\n\n".format([_ for _ in msd.msd_mean]))
        f.write("msd_std\n{}\n\n".format([_ for _ in msd.msd_std]))

    return


class MSDHandler:

    def __init__(
        self,
        replicate_name,
        HKMCMD_molecule_types,
        filename_notebook="default",
        filename_data="default",
        filename_trj="default",
        filename_output="default",
        atom_style="full",
        frames=[0, 1, 100],
    ):

        # Set default filenames
        if filename_notebook == "default":
            filename_notebook = replicate_name + "_notebook.xlsx"
        if filename_data == "default":
            filename_data = replicate_name + ".in.data"
        if filename_trj == "default":
            filename_trj = replicate_name + ".diffusion.lammpstrj"
        if filename_output == "default":
            filename_output = replicate_name + ".msdoutput.txt"

        self.replicate_name = replicate_name
        self.filename_notebook = filename_notebook
        self.filename_data = filename_data
        self.filename_trj = filename_trj
        self.filename_output = filename_output
        self.HKMCMD_molecule_types = HKMCMD_molecule_types
        self.atom_style = atom_style
        self.frames = frames

        return

    def read_notebook(self):

        notebook_dict = read_notebook(self.filename_notebook)
        self.header = notebook_dict["header"]
        self.starting_species = notebook_dict["starting_species"]
        self.masterspecies = notebook_dict["masterspecies"]
        self.reaction_data = notebook_dict["reaction_data"]
        self.initial_MD_init_dict = notebook_dict["initial_MD_init_dict"]
        self.cycled_MD_init_dict = notebook_dict["cycled_MD_init_dict"]

        return

    def parse_data_file(self):

        outputs = parse_data_file(
            self.filename_data,
            atom_style=self.atom_style,
            preserve_atom_order=False,
            preserve_bond_order=False,
            preserve_angle_order=False,
            preserve_dihedral_order=False,
            preserve_improper_order=False,
            tdpd_conc=[],
            unwrap=False,
        )

        labels = [
            "atoms",
            "bonds",
            "angles",
            "dihedrals",
            "impropers",
            "box",
            "adj_mat",
            "extra_prop",
        ]

        for idx, val in enumerate(labels):
            setattr(self, val, outputs[idx])

        return

    def get_centers_of_mass(self):

        if not hasattr(self, "masterspecies"):
            self.read_notebook()
        if not hasattr(self, "atoms"):
            self.parse_data_file()

        data_atoms_HKMCMD_molecule_types = (
            calculate_HKMCMD_molecule_types_from_LAMMPS_atom_types(
                self.atoms, self.masterspecies
            )
        )
        centers_of_mass_IDs = [
            atomID
            for idx, atomID in enumerate(self.atoms.ids)
            if data_atoms_HKMCMD_molecule_types[idx] in self.HKMCMD_molecule_types
        ]
        centers_of_mass = np.zeros(
            (
                len(centers_of_mass_IDs),
                int((self.frames[2] - self.frames[0]) / self.frames[1]),
            )
        )
        timesteps = []
        boxes = []
        frame_idx = 0

        for atom, timestep, box in frame_generator(
            self.filename_trj,
            start=self.frames[0],
            end=self.frames[2] - 1,
            every=self.frames[1],
            unwrap=False,
        ):
            idxs = atom.get_idx(ids=centers_of_mass_IDs)
            centers_of_mass[:, frame_idx] = np.array(
                [
                    np.sqrt(atom.x[idx] ** 2 + atom.y[idx] ** 2 + atom.z[idx] ** 2)
                    for idx in idxs
                ]
            )
            timesteps.append(timestep)
            boxes.append(box)
            frame_idx += 1

        setattr(self, "centers_of_mass", centers_of_mass)
        setattr(self, "timesteps", np.array([int(_) for _ in timesteps]))
        setattr(self, "boxes", boxes)

        return

    def calculate_msd(self):

        if not hasattr(self, "centers_of_mass"):
            self.get_centers_of_mass()

        # Get the number of atoms and frames
        num_atoms, num_frames = self.centers_of_mass.shape

        # Initialize an array to store MSD values
        msd_values = np.zeros(num_frames - 1)
        msd_std = np.zeros(num_frames - 1)

        # Loop over all possible frame differences (lag times)
        for lag in range(1, num_frames):
            # Calculate the squared displacements for the given lag time
            displacements = (
                self.centers_of_mass[:, lag:] - self.centers_of_mass[:, :-lag]
            )
            squared_displacements = displacements**2

            # Average over all atoms and all pairs of frames with the given lag
            msd_values[lag - 1] = np.mean(squared_displacements)
            msd_std[lag - 1] = np.std(squared_displacements)

        setattr(self, "msd_mean", msd_values)
        setattr(self, "msd_std", msd_std)

        return


if __name__ == "__main__":
    main(sys.argv[1:])

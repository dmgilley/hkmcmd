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

        data_atoms_HKMDMC_molecule_types = (
            calculate_HKMCMD_molecule_types_from_LAMMPS_atom_types(
                self.atoms, self.masterspecies
            )
        )
        centers_of_mass_IDs = [
            atomID
            for idx, atomID in enumerate(self.atoms.ids)
            if data_atoms_HKMDMC_molecule_types[idx] in self.HKMCMD_molecule_types
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


def read_notebook(filename_notebook):
    notebook = pd.read_excel(
        filename_notebook, sheet_name=None, index_col=None, header=None
    )

    # System
    dict_ = {}
    for index, row in notebook["System"].iterrows():
        if row[1] == "-":
            continue
        if row[1] == "false":
            row[1] = False
        if row[1] == "true":
            row[1] = True
        dict_[row[0]] = row[1]

    # Header
    dict_["header"] = {"masses": []}
    for index, row in notebook["Header"].iterrows():
        if "types" in row[0]:
            dict_["header"]["_".join(row[0].split())] = int(row[1])
        elif "mass" in row[0]:
            dict_["header"]["masses"].append(
                tuple([int(row[0].split()[1]), float(row[1])])
            )

    # Species
    dict_["starting_species"] = {}
    dict_["masterspecies"] = {}
    notebook["Species"].dropna(how="all", inplace=True)
    category_series = notebook["Species"].iloc[:, 0]
    category_series.ffill(inplace=True)
    categories = [
        (index, value, notebook["Species"].loc[index, 1])
        for index, value in category_series.items()
    ]
    msf_categories = ["Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"]
    row_map = {
        key: {v[2]: v[0] for v in categories if v[1] == key} for key in msf_categories
    }
    species = None
    for column_idx in notebook["Species"].iloc[:, 2:].columns:
        column = notebook["Species"].loc[:, column_idx]
        if not pd.isna(column[0]):
            species = column[0]
            dict_["masterspecies"][species] = {_: [] for _ in msf_categories}
            dict_["starting_species"][species] = int(column[2])
        # atoms
        if not pd.isna(column[row_map["Atoms"]["ID"]]):
            dict_["masterspecies"][species]["Atoms"].append(
                [
                    int(column[row_map["Atoms"]["ID"]]),
                    0,
                    int(column[row_map["Atoms"]["type"]]),
                    float(column[row_map["Atoms"]["charge"]]),
                    float(column[row_map["Atoms"]["x"]]),
                    float(column[row_map["Atoms"]["y"]]),
                    float(column[row_map["Atoms"]["z"]]),
                ]
            )
        # interactions
        for interaction in msf_categories[1:]:
            if not pd.isna(column[row_map[interaction]["ID"]]):
                list_ = [
                    int(column[row_map[interaction]["ID"]]),
                    int(column[row_map[interaction]["type"]]),
                    int(column[row_map[interaction]["i"]]),
                    int(column[row_map[interaction]["j"]]),
                ]
                if "k" in row_map[interaction].keys():
                    list_.append(int(column[row_map[interaction]["k"]]))
                if "l" in row_map[interaction].keys():
                    list_.append(int(column[row_map[interaction]["l"]]))
                dict_["masterspecies"][species][interaction].append(list_)

    # Reactions
    dict_["reaction_data"] = {}
    row_map = {val: idx for idx, val in notebook["Reactions"].iloc[:, 0].items()}
    dict_["reaction_data"] = {
        int(column[row_map["reaction ID"]]): {
            "reactant_molecules": column[row_map["reactant(s)"]].split(","),
            "product_molecules": column[row_map["product(s)"]].split(","),
            "A": column[row_map["A (1/s)"]],
            "b": column[row_map["b"]],
            "Ea": column[row_map["Ea (kcal/mol)"]],
        }
        for idx, column in notebook["Reactions"].iloc[:, 1:].items()
    }

    # Initial MD
    dict_["initial_MD_init_dict"] = {}
    for index, row in notebook["Initial MD"].iterrows():
        if "run" in row[0]:
            dict_["initial_MD_init_dict"][row[0]] = list(row[1:])
        else:
            dict_["initial_MD_init_dict"][row[0]] = row[1]

    # Cycled MD
    dict_["cycled_MD_init_dict"] = {}
    for index, row in notebook["Cycled MD"].iterrows():
        if "run" in row[0]:
            dict_["cycled_MD_init_dict"][row[0]] = list(row[1:])
        else:
            dict_["cycled_MD_init_dict"][row[0]] = row[1]

    return dict_


def calculate_HKMCMD_molecule_types_from_LAMMPS_atom_types(atoms, master_species):

    tuple_of_LAMMPS_atom_types_to_HKMCMD_molecule_type = {}
    for k, v in master_species.items():
        tuple_of_LAMMPS_atom_types_to_HKMCMD_molecule_type[
            tuple(sorted([i[2] for i in v["Atoms"]]))
        ] = k

    HKMCMD_molecule_types = np.array(["not assigned"] * len(atoms.ids)).flatten()
    for LAMMPS_molecule_ID in sorted(list(set(atoms.mol_id))):
        idxs = [
            idx for idx, LmID in enumerate(atoms.mol_id) if LmID == LAMMPS_molecule_ID
        ]
        HKMCMD_molecule_types[idxs] = (
            tuple_of_LAMMPS_atom_types_to_HKMCMD_molecule_type[
                tuple(sorted(atoms.lammps_type[idxs]))
            ]
        )

    return HKMCMD_molecule_types


if __name__ == "__main__":
    main(sys.argv[1:])

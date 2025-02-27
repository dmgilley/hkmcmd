#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import os, json, datetime
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from hybrid_mdmc.particle_interactions import *


def frame_generator(
    name, start=0, end=-1, every=1, unwrap=False, adj_list=None, return_prop=False
):

    # key: lammps keyword, value: Atom attribute
    candidates = {
        "id": "ID",
        "mol": "molecule_ID",
        "type": "lammps_type",
        "mass": "mass",
        "q": "charge",
        "xu": "x",
        "yu": "y",
        "zu": "z",
        "x": "x",
        "y": "y",
        "z": "z",
        "vx": "vx",
        "vy": "vy",
        "vz": "vz",
        "fx": "fx",
        "fy": "fy",
        "fz": "fz",
    }

    # Parse Trajectories
    frame = -1  # Frame counter (total number of frames in the trajectory)
    frame_count = -1  # Frame counter (number of parsed frames in the trajectory)
    frame_flag = 0  # Flag for marking the start of a parsed frame
    atom_flag = 0  # Flag for marking the start of a parsed Atom data block
    N_atom_flag = 0  # Flag for marking the place the number of atoms should be updated
    atom_count = 0  # Atom counter for each frame
    box_flag = 0  # Flag for marking the start of the box parse
    box_count = -1  # Line counter for keeping track of the box dimensions.

    # Open the trajectory file for reading
    with open(name, "r") as f:

        # Iterate over the lines of the original trajectory file
        for lines in f:

            fields = lines.split()

            # Find the start of each frame and check if it is included in the user-requested range
            if len(fields) == 2 and fields[1] == "TIMESTEP":
                frame += 1
                if (
                    frame >= start
                    and (frame <= end or end == -1)
                    and (frame - start) % every == 0
                ):
                    frame_flag = 1
                    frame_count += 1
                elif frame > end and end != -1:
                    break
            # Parse commands for when a user-requested frame is being parsed
            if frame_flag == 1:

                # Header parse commands
                if atom_flag == 0 and N_atom_flag == 0 and box_flag == 0:
                    if len(fields) > 2 and fields[1] == "ATOMS":
                        atom_flag = 1
                        # read in all properties
                        prop = {_: [-1 for k in range(N_atoms)] for _ in fields[2:]}
                        ind_dict = {
                            _: fields.index(_) - 2 for _ in fields[2:]
                        }  # key: property name in lammpstrj, value: field index
                        # if frame_count == 0:
                        #   print("The following properties are found in lammpstrj file: {}".format(' '.join(list(ind_dict.keys()))))
                        continue
                    if len(fields) > 2 and fields[1] == "NUMBER":
                        N_atom_flag = 1
                        continue

                    if len(fields) > 2 and fields[1] == "BOX":
                        box_flag = 1
                        continue

                    if len(fields) == 1:
                        timestep = fields[0]
                        continue

                # Update the number of atoms in each frame
                if N_atom_flag == 1:

                    # Intialize total geometry of the molecules being parsed in this frame
                    # Note: from here forward the N_current acts as a counter of the number of atoms that have been parsed from the trajectory.
                    N_atoms = int(fields[0])
                    prop = {
                        candidates[_]: [-1 for k in range(N_atoms)] for _ in candidates
                    }
                    N_atom_flag = 0
                    continue

                # Read in box dimensions
                if box_flag == 1:
                    # initilize box array
                    if box_count == -1:
                        # cubic  box only has 2 columns
                        if len(fields) == 2:
                            box = np.zeros([3, 2])
                        # crystal has 3 columns
                        else:
                            box = np.zeros([3, 3])

                    box_count += 1
                    box[box_count] = [float(_) for _ in fields]

                    # After all box data has been parsed, save the box_lengths/2 to temporary variables for unwrapping coordinates and reset flags/counters
                    if box_count == 2:
                        box_count = -1
                        box_flag = 0
                    continue

                # Parse relevant atoms
                if atom_flag == 1:
                    for _ in prop:
                        if _ in ["id", "mol", "type"]:
                            prop[_][atom_count] = int(fields[ind_dict[_]])
                        else:
                            prop[_][atom_count] = float(fields[ind_dict[_]])
                    atom_count += 1

                    # Reset flags once all atoms have been parsed
                    if atom_count == N_atoms:

                        frame_flag = 0
                        atom_flag = 0
                        atom_count = 0

                        # Sort based on ids
                        prop["id"], sort_ind = list(
                            zip(
                                *sorted(
                                    [
                                        (k, count_k)
                                        for count_k, k in enumerate(prop["id"])
                                    ]
                                )
                            )
                        )
                        for _ in prop:
                            prop[_] = np.array([prop[_][k] for k in sort_ind])

                        # Populate atom with prop dictionary
                        # properties not supported by atomlis class will not be parsed
                        # if those omitted properties are still desired, use return_prop flag to return original prop dictionary
                        # initilize atomlist
                        atoms_list = [
                            Atom(
                                **{
                                    candidates[key]: prop[key][atom_idx]
                                    for key in candidates.keys()
                                    if key in prop.keys()
                                }
                            )
                            for atom_idx, atom_ID in enumerate(prop["id"])
                        ]

                        # Upwrap the geometry
                        if unwrap is True:
                            atoms_list = unwrap_atoms_list(
                                atoms_list, box, adj_list=adj_list
                            )

                        if return_prop:
                            yield atoms_list, timestep, box, prop
                        else:
                            yield atoms_list, timestep, box, None


def parse_data_file(
    data_file,
    atom_style="full",
    preserve_atom_order=False,
    preserve_bond_order=False,
    preserve_angle_order=False,
    preserve_dihedral_order=False,
    preserve_improper_order=False,
    tdpd_conc=[],
    unwrap=False,
):

    # Initialize temporary dictionaries for the atoms, bonds, angle, dihedrals, and impropers, as well as final list/dictionary objects for box, masses, velocities, and extra_prop.
    temp_atoms = {}
    temp_bonds = {}
    temp_angles = {}
    temp_dihedrals = {}
    temp_impropers = {}
    box = []
    masses = {}
    velocities = {}
    ellipsoids = {}
    extra_prop = {}

    # Create lists of atom/bond/angles/dihedral id's, if the order is to be preserved
    if preserve_atom_order:
        atom_ids = []
    if preserve_bond_order:
        bond_ids = []
    if preserve_angle_order:
        angle_ids = []
    if preserve_dihedral_order:
        dihedral_ids = []
    if preserve_improper_order:
        improper_ids = []

    # This dictionary describes the atom attirbutes listed in the data file, based on the lammps atom style.
    # Key: LAMMPS atom style
    # Value: list of AtomList attributes, in the order of the LAMMPS Atoms section's columns
    lammps_atom_attributes_options = {
        "angle": ["atom_id", "mol_id", "lammps_type", "x", "y", "z"],
        "atomic": ["atom_id", "lammps_type", "x", "y", "z"],
        "body": ["atom_id", "lammps_type", "bodyflag", "mass", "x", "y", "z"],
        "bond": ["atom_id", "mol_id", "lammps_type", "x", "y", "z"],
        "charge": ["atom_id", "lammps_type", "charge", "x", "y", "z"],
        "dipole": [
            "atom_id",
            "lammps_type",
            "charge",
            "x",
            "y",
            "z",
            "mux",
            "muy",
            "muz",
        ],
        "dpd": ["atom_id", "lammps_type", "theta", "x", "y", "z"],
        "edpd": ["atom_id", "lammps_type", "edpd_temp", "edpd_cv", "x", "y", "z"],
        "electron": [
            "atom_id",
            "lammps_type",
            "charge",
            "spin",
            "eradius",
            "x",
            "y",
            "z",
        ],
        "ellipsoid": [
            "atom_id",
            "lammps_type",
            "ellipsoidflag",
            "density",
            "x",
            "y",
            "z",
        ],
        "full": ["atom_id", "mol_id", "lammps_type", "charge", "x", "y", "z"],
        "line": [
            "atom_id",
            "mol_id",
            "lammps_type",
            "lineflag",
            "density",
            "x",
            "y",
            "z",
        ],
        "mdpd": ["atom_id", "lammps_type", "rho", "x", "y", "z"],
        "mesont": [
            "atom_id",
            "mol_id",
            "lammps_type",
            "bond_nt",
            "mass",
            "mradius",
            "mlength",
            "buckling",
            "x",
            "y",
            "z",
        ],
        "molecular": ["atom_id", "mol_id", "lammps_type", "x", "y", "z"],
        "peri": ["atom_id", "lammps_type", "volume", "density", "x", "y", "z"],
        "smd": [
            "atom_id",
            "lammps_type",
            "mol_id",
            "volume",
            "mass",
            "kradius",
            "cradius",
            "x0",
            "y0",
            "z0",
            "x",
            "y",
            "z",
        ],
        "sph": ["atom_id", "lammps_type", "rho", "esph", "cv", "x", "y", "z"],
        "sphere": ["atom_id", "lammps_type", "diameter", "density", "x", "y", "z"],
        "spin": ["atom_id", "lammps_type", "x", "y", "z", "spx", "spy", "spz"],
        "tdpd": ["atom_id", "lammps_type", "x", "y", "z"]
        + [conc for conc in tdpd_conc],
        "templpate": [
            "atom_id",
            "lammps_type",
            "mol_id",
            "template_index",
            "template_atom",
            "x",
            "y",
            "z",
        ],
        "tri": [
            "atom_id",
            "mol_id",
            "lammps_type",
            "triangleflag",
            "density",
            "x",
            "y",
            "z",
        ],
        "wavepacket": [
            "atom_id",
            "lammps_type",
            "charge",
            "spin",
            "eradius",
            "etag",
            "cs_re",
            "cs_im",
            "x",
            "y",
            "z",
        ],
    }

    lammps2Atom = {
        "atom_id": "ID",
        "charge": "charge",
        "lammps_type": "lammps_type",
        "mass": "mass",
        "mol_id": "molecule_ID",
        "x": "x",
        "y": "y",
        "z": "z",
    }

    # This list replicates the specific atom attributes list from the previous dictionary.
    # Creating a new object titled "att_list" simplifies the rest of the script, as the list of atom attributes for the specific data file is called multiple times.
    if atom_style.split()[0] == "hybrid":
        att_list = ["atom_id", "lammps_type", "x", "y", "z"]
        for style in atom_style.split()[1:]:
            temp_list = [
                att
                for att in lammps_atom_attributes_options[style]
                if att not in att_list
            ]
            att_list += temp_list
    else:
        att_list = lammps_atom_attributes_options[atom_style]

    # Initialize the flag to be used during the parsing, as well as a list of flag options.
    flag = None
    flag_options = [
        "Atoms",
        "Masses",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
        "Velocities",
        "Ellipsoid",
    ]

    # Parse the data file.
    with open(data_file, "r") as f:
        for line in f:
            fields = line.split()

            # Skip over blank lines and any comments.
            if fields == []:
                continue
            if fields[0] == "#":
                continue

            # Skip over coefficient lines, if present
            if "Bond Coeffs" in line:
                flag = None
                continue
            if "Pair Coeffs" in line:
                flag = None
                continue
            if "Angle Coeffs" in line:
                flag = None
                continue
            if "Dihedral Coeffs" in line:
                flag = None
                continue
            if "Improper Coeffs" in line:
                flag = None
                continue

            # Check for updates to the flag.
            if fields[0] in flag_options:
                flag = fields[0]
                continue

            # Parse the actual data, based on the flag.
            if "xlo" in line or "ylo" in line or "zlo" in line:
                box.append([float(fields[0]), float(fields[1])])
                continue
            if "xy" in line:
                box.append([float(fields[0]), float(fields[1]), float(fields[2])])
            if flag == "Masses":
                masses[int(fields[0])] = float(fields[1])
                continue
            if flag == "Atoms":
                temp_atoms[int(fields[0])] = [float(i) for i in fields[1:]]
                if preserve_atom_order:
                    atom_ids.append(int(fields[0]))
                continue
            if flag == "Bonds":
                temp_bonds[int(fields[0])] = [int(i) for i in fields[1:]]
                if preserve_bond_order:
                    bond_ids.append(int(fields[0]))
                continue
            if flag == "Angles":
                temp_angles[int(fields[0])] = [int(i) for i in fields[1:]]
                if preserve_angle_order:
                    angle_ids.append(int(fields[0]))
                continue
            if flag == "Dihedrals":
                temp_dihedrals[int(fields[0])] = [int(i) for i in fields[1:]]
                if preserve_dihedral_order:
                    dihedral_ids.append(int(fields[0]))
                continue
            if flag == "Impropers":
                temp_impropers[int(fields[0])] = [int(i) for i in fields[1:]]
                if preserve_improper_order:
                    improper_ids.append(int(fields[0]))
                continue
            if flag == "Velocities":
                velocities[int(fields[0])] = [float(i) for i in fields[1:]]
                continue
            if flag == "Ellipsoids":
                ellipsoids[int(fields[0])] = [float(i) for i in fields[1:]]

    # Create the atoms list
    if preserve_atom_order is False:
        atom_ids = sorted([key for key in temp_atoms.keys()])
    extra_prop = {
        att: [temp_atoms[ID][att_idx] for ID in atom_ids]
        for att_idx, att in enumerate(att_list)
        if att not in lammps2Atom.keys()
    }
    atoms_list = [Atom(ID=ID) for ID in atom_ids]
    for idx, attr in enumerate(att_list[1:]):
        if attr in lammps2Atom.keys():
            for atom in atoms_list:
                setattr(atom, lammps2Atom[attr], temp_atoms[atom.ID][idx])
    for atom in atoms_list:
        setattr(atom, "mass", masses[atom.lammps_type])
    if unwrap is True:
        atoms_list = unwrap_atoms_list(atoms_list, box)

    # Create the bonds list
    bonds_list = None
    if preserve_bond_order is False:
        bond_ids = sorted([key for key in temp_bonds.keys()])
    if len(bond_ids) != 0:
        bonds_list = [
            IntraMode(
                ID=ID,
                kind=temp_bonds[ID][0],
                atom_IDs=[temp_bonds[ID][1], temp_bonds[ID][2]],
            )
            for idx, ID in enumerate(bond_ids)
        ]

    # Create the angles list
    angles_list = None
    if preserve_angle_order is False:
        angle_ids = sorted([key for key in temp_angles.keys()])
    if len(angle_ids) != 0:
        angles_list = [
            IntraMode(
                ID=ID,
                kind=temp_angles[ID][0],
                atom_IDs=[temp_angles[ID][1], temp_angles[ID][2], temp_angles[ID][3]],
            )
            for idx, ID in enumerate(angle_ids)
        ]

    # Create the dihedrals list
    dihedrals_list = None
    if preserve_dihedral_order is False:
        dihedral_ids = sorted([key for key in temp_dihedrals.keys()])
    if len(dihedral_ids) != 0:
        dihedrals_list = [
            IntraMode(
                ID=ID,
                kind=temp_dihedrals[ID][0],
                atom_IDs=[
                    temp_dihedrals[ID][1],
                    temp_dihedrals[ID][2],
                    temp_dihedrals[ID][3],
                    temp_dihedrals[ID][4],
                ],
            )
            for idx, ID in enumerate(dihedral_ids)
        ]

    # Create the impropers list
    impropers_list = None
    if preserve_improper_order is False:
        improper_ids = sorted([key for key in temp_impropers.keys()])
    if len(improper_ids) != 0:
        impropers_list = [
            IntraMode(
                ID=ID,
                kind=temp_impropers[ID][0],
                atom_IDs=[
                    temp_impropers[ID][1],
                    temp_impropers[ID][2],
                    temp_impropers[ID][3],
                    temp_impropers[ID][4],
                ],
            )
            for idx, ID in enumerate(improper_ids)
        ]

    return (
        atoms_list,
        bonds_list,
        angles_list,
        dihedrals_list,
        impropers_list,
        box,
        extra_prop,
    )


def parse_diffusion_file(filename):
    data, diffusion_step, species = {}, None, None
    with open(filename, "r") as f:
        for line in f:
            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if "---" in fields[0]:
                continue
            if fields[0] == "DiffusionStep":
                diffusion_step = int(fields[1])
                data[diffusion_step] = {}
                continue
            if fields[0] == "Diffusion":
                species = fields[3]
                data[diffusion_step][species] = []
                continue
            if species is not None:
                data[diffusion_step][species].append([float(_) for _ in fields])
                continue
    diffusion_steps = sorted(list(data.keys()))
    species = sorted(list(data[diffusion_steps[0]].keys()))
    return {sp: np.array([data[ds][sp] for ds in diffusion_steps]) for sp in species}


class FileTracker:

    def __init__(self, name, overwrite=False):
        self.name = name
        if overwrite is True or not os.path.exists(self.name):
            self.create()
        return

    def create(self):
        with open(self.name, "w") as f:
            f.write(f"# File created {datetime.datetime.now()}\n")
        return

    def write(self, string):
        with open(self.name, "a") as f:
            f.write(string)
        return

    def write_separation(self, sep="-"):
        with open(self.name, "a") as f:
            f.write(f"\n{sep * 100}\n")
        return

    def write_array(self, array, as_str=True):
        preamble = ""
        if type(array) is tuple:
            preamble, array = array[0], array[1]
        if as_str is True:
            array = np.array_str(array, max_line_width=1e99)[1:-1]
        with open(self.name, "a") as f:
            f.write(f"{preamble}{array}\n")
        return


class hkmcmd_ArgumentParser:

    def __init__(self):
        self.parser = ArgumentParser()
        self.add_positional_arguments()
        return

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
        return

    def add_positional_arguments(self):
        self.add_argument("system", type=str, help="Name of the system")
        self.add_argument("prefix", type=str, help="Prefix of the system")
        return

    def add_optional_arguments(self):
        for file in [
            "json",
            "data",
            "trajectory",
            "diffusion",
            "system_state",
            "reaction",
            "settings",
        ]:
            self.add_argument(
                f"-filename_{file}",
                dest=f"filename_{file}",
                type=str,
                default=None,
            )
        self.add_argument(
            "-diffusion_cycle", dest="diffusion_cycle", type=int, default=0
        )
        self.add_argument(
            "--debug", dest="debug", default=False, action="store_const", const=True
        )
        return

    def adjust_default_args(self):
        if self.args.filename_json is None:
            self.args.filename_json = self.args.system + ".json"
        if self.args.filename_data is None:
            self.args.filename_data = self.args.prefix + ".end.data"
        if self.args.filename_trajectory is None:
            self.args.filename_trajectory = self.args.prefix + ".diffusion.lammpstrj"
        if self.args.filename_diffusion is None:
            self.args.filename_diffusion = self.args.prefix + ".diffusion.txt"
        if self.args.filename_system_state is None:
            self.args.filename_system_state = self.args.prefix + ".system_state.json"
        if self.args.filename_reaction is None:
            self.args.filename_reaction = self.args.prefix + ".reaction.txt"
        if self.args.filename_settings is None:
            self.args.filename_settings = self.args.prefix + ".in.settings"
        return

    def parse_args(self):
        self.add_optional_arguments()
        self.args = self.parser.parse_args()
        self.adjust_default_args()
        return self.args

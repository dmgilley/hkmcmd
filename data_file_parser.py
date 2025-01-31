#!/usr/bin/env python3
#
# Author:
#    Dylan Gilley
#    dgilley@purdue.edu
#
# History:
#    Feb2022 - Initial creation
#

import numpy as np
from hybrid_mdmc.mol_classes import AtomList, IntraModeList


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

    ############################# 72 Characters ############################

    """Parser for LAMMPS data files.


    Parameters
    ----------
    data_file: str
        The LAMMPS data file to parse. If present, ignores force field
        coefficients.

    atom_style: str, optional
        LAMMPS "atom style;" signals the attributes and their order present
        in the data file. See https://docs.lammps.org/read_data.html for
        details. If using the hybrid style, declare the word "hybrid"
        followed by each substyle, all in the same string.
        E.g. 'hybrid full ellipsoid'(default: 'full')

    preserve_atom_order: boolean, optional
        If True, the order of the atoms data in the returned atoms object
        will be the same as that of the data file. If False, the order of
        the data in the returned atoms object will be sorted by atom id.
        Setting to False (i.e. sorting the data by id) speeds parsing.
        (default: False)

    preserve_bond_order: boolean, optional
        If True, the order of the bond data in the returned bonds object
        will be the same as that of the data file. If False, the order of
        the data in the returned bonds object will be sorted by bond id.
        Setting to False (i.e. sorting the data by id) speeds parsing.
        (default: False)

    preserve_angle_order: boolean, optional
        If True, the order of the angle data in the returned angles object
        will be the same as that of the data file. If False, the order of
        the data in the returned angles object will be sorted by angle id.
        Setting to False (i.e. sorting the data by id) speeds parsing.
        (default: False)

    preserve_dihedral_order: boolean, optional
        If True, the order of the dihedral data in the returned dihedrals
        object will be the same as that of the data file. If False, the
        order ofthe data in the returned dihedrals object will be sorted
        by dihedral id. Setting to False (i.e. sorting the data by id)
        speeds parsing. (default: False)

    preserve_improper_order: boolean, optional
        If True, the order of the improper data in the returned impropers
        object will be the same as that of the data file. If False, the
        order ofthe data in the returned impropers object will be sorted
        by improper id. Setting to False (i.e. sorting the data by id)
        speeds parsing. (default: False)


    Returns
    -------
    atoms: instance of AtomList.

    bonds: instance of IntraModeList.

    angles: instance of IntraModeList.

    dihedrals: instance of IntraModeList.

    impropers: instance of IntraModeList.

    box: dimensions of the simulation cell.
        [[xmin,xmax], [ymin,ymax], [zmin,zmax]] (cubic)
        [[xmin,xmax], [ymin,ymax], [zmin,zmax], [xy,xz,yz]] (non-cubic)

    adj_dict: adjacency dictionary.
        { atom id: [bonded atom id 1, bonded atom id 2, ...] }

    adj_mat: NxN adjacency matrix.
        If there is a bond between atoms with ids "i" and "j", the
        matrix will have a a vlaue of 1 in row (i-1), column (j-1) and
        in row (j-1), column (i-1). Otherwise, a value of 0. The adjacency
        matrix assumse the atom ids range from 1 to N, without any
        gaps (inclusive).

    extra_prop: a dictionary of atom properties not handled by the AtomList
        class. { property: [values, indexed to atoms.ids] }


    Errors
    ------
    "Error! Unknown mass for atom(s) ___...":
        Error signifies that the atomic mass for the specified atom(s), as
        read from the LAMMPS data file, is/are not among the keys of the
        "mass_to_element" dictionary. To continue, verify the atomic masses
        in the LAMMPS data file. If these are correct, you may need to add
        the element to the "mass_to_element" dictionary.


    Additional Notes
    ----------------


    Future Work
    -----------
    """

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

    # Create a dictionary for mapping atomic mass to element.
    # Not all elements are mapped; this may need to be edited if your element isn't included.
    # In using this dictionary, it is expected that the passed key will be rounded; i.e. if the atom has a mass of 12.011, pass "12" as the key (or, round(12.011) ) to receive the value "C".
    mass_to_element = {
        1: "H",
        4: "He",
        7: "Li",
        9: "Be",
        11: "B",
        12: "C",
        14: "N",
        16: "O",
        19: "F",
        20: "Ne",
        23: "Na",
        24: "Mg",
        27: "Al",
        28: "Si",
        31: "P",
        32: "S",
        36: "Cl",
        40: "Ar",
        39: "K",
        40: "Ca",
        48: "Ti",
        52: "Cr",
        55: "Mn",
        56: "Fe",
        64: "Cu",
        65: "Zn",
        80: "Br",
        84: "Kr",
        108: "Ag",
        119: "Sn",
        127: "I",
        131: "Xe",
        195: "Pt",
        197: "Au",
        207: "Pb",
    }

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

    # Create the atoms object, sorting by atom id if the order is not being preserved.
    if not preserve_atom_order:
        atom_ids = sorted([key for key in temp_atoms.keys()])
    atoms = AtomList(ids=atom_ids)

    # Create the adjacency dictionary and the adjacency matrix.
    # The adjacency matrix is created under the assumption that the atom ids range from 1 to N, inclusive (N = number of atoms in system).
    adj_dict = {atom_id: set() for atom_id in atoms.ids}
    adj_mat = np.zeros((len(atoms.ids), len(atoms.ids)))
    for bond_id, atom_list in temp_bonds.items():
        adj_dict[atom_list[1]].add(atom_list[2])
        adj_dict[atom_list[2]].add(atom_list[1])
        adj_mat[atom_list[2] - 1, atom_list[1] - 1] = 1
        adj_mat[atom_list[1] - 1, atom_list[2] - 1] = 1
    for atom_id, bonded_set in adj_dict.items():
        adj_dict[atom_id] = sorted(list(bonded_set))

    # Unwrap atom coordinates, if requested.
    if unwrap:

        # Grab the box dimensions.
        lx = box[0][1] - box[0][0]
        ly = box[1][1] - box[1][0]
        lz = box[2][1] - box[2][0]

        # Create a list of unwrapped atom ids.
        unwrapped = []

        # Loop over all atom ids.
        for i in atoms.ids:

            # If the atom id is in the unwrapped list, all atoms in its molecule should be unwrapped already, and this atom can be skipped.
            if i in unwrapped:
                continue

            # Begin a walk along the molecular graph.
            # The molecular graph is cumulatively built up, stored in the "unwrap_list" object, initially seeded with atom "i".
            unwrap_list = [i]
            unwrapped += [i]
            for j in unwrap_list:

                # "new" holds the atom ids of the atoms bonded to atom "j".
                new = [k for k in adj_dict[j] if k not in unwrapped]

                # Loop over the "new" list, unwrapping the coordinates of the "k" atom, adding the atom id to the "unwrapped" list.
                for k in new:

                    # x-dimension
                    while (
                        temp_atoms[k][att_list.index("x") - 1]
                        - temp_atoms[j][att_list.index("x") - 1]
                    ) > lx / 2.0:
                        temp_atoms[k][att_list.index("x") - 1] -= lx
                    while (
                        temp_atoms[k][att_list.index("x") - 1]
                        - temp_atoms[j][att_list.index("x") - 1]
                    ) < -lx / 2.0:
                        temp_atoms[k][att_list.index("x") - 1] += lx

                    # y-dimension
                    while (
                        temp_atoms[k][att_list.index("y") - 1]
                        - temp_atoms[j][att_list.index("y") - 1]
                    ) > ly / 2.0:
                        temp_atoms[k][att_list.index("y") - 1] -= ly
                    while (
                        temp_atoms[k][att_list.index("y") - 1]
                        - temp_atoms[j][att_list.index("y") - 1]
                    ) < -ly / 2.0:
                        temp_atoms[k][att_list.index("y") - 1] += ly

                    # z-dimension
                    while (
                        temp_atoms[k][att_list.index("z") - 1]
                        - temp_atoms[j][att_list.index("z") - 1]
                    ) > lz / 2.0:
                        temp_atoms[k][att_list.index("z") - 1] -= lz
                    while (
                        temp_atoms[k][att_list.index("z") - 1]
                        - temp_atoms[j][att_list.index("z") - 1]
                    ) < -lz / 2.0:
                        temp_atoms[k][att_list.index("z") - 1] += lz

                    unwrapped += [k]

                unwrap_list += new

    # Add the appropriate atomistic characteristics tot eh atoms object.
    if "lammps_type" in att_list:
        atoms.append(
            lammps_type=[
                int(temp_atoms[ids][att_list.index("lammps_type") - 1])
                for ids in atom_ids
            ]
        )
    if "charge" in att_list:
        atoms.append(
            charge=[temp_atoms[ids][att_list.index("charge") - 1] for ids in atom_ids]
        )
    if "mol_id" in att_list:
        atoms.append(
            mol_id=[
                int(temp_atoms[ids][att_list.index("mol_id") - 1]) for ids in atom_ids
            ]
        )
    if "x" in att_list:
        atoms.append(x=[temp_atoms[ids][att_list.index("x") - 1] for ids in atom_ids])
    if "y" in att_list:
        atoms.append(y=[temp_atoms[ids][att_list.index("y") - 1] for ids in atom_ids])
    if "z" in att_list:
        atoms.append(z=[temp_atoms[ids][att_list.index("z") - 1] for ids in atom_ids])
    if len(velocities) != 0:
        atoms.append(
            vx=[velocities[ids][0] for ids in atom_ids],
            vy=[velocities[ids][1] for ids in atom_ids],
            vz=[velocities[ids][2] for ids in atom_ids],
        )
    if len(ellipsoids) != 0:
        for ids in atom_ids:
            if ids not in ellipsoids.keys():
                ellipsoids[ids] = [None for i in range(7)]
        atoms.append(
            q1=[ellipsoids[ids][3] for ids in atom_ids],
            q2=[ellipsoids[ids][4] for ids in atom_ids],
            q3=[ellipsoids[ids][5] for ids in atom_ids],
            q4=[ellipsoids[ids][6] for ids in atom_ids],
        )

    # Check if there are atom properties not covered in the AtomList. If so, fill the extras dictionary.
    filled_attributes = []
    for att_name, att_values in atoms.__dict__.items():
        if len(att_values) != 0:
            filled_attributes.append(att_name)
    for att_name in att_list:
        if att_name not in filled_attributes:
            extra_prop[att_name] = [
                temp_atoms[ids][att_list.index(att_name) - 1] for ids in atom_ids
            ]
    if len(ellipsoids) != 0:
        extra_prop["ellipsoid shapex"] = [ellipsoids[ids][0] for ids in atom_ids]
        extra_prop["ellipsoid shapey"] = [ellipsoids[ids][1] for ids in atom_ids]
        extra_prop["ellipsoid shapez"] = [ellipsoids[ids][2] for ids in atom_ids]

    # Add the masses and elements to the atoms object.
    atoms.append(mass=[masses[atom_type] for atom_type in atoms.lammps_type])
    error_mass = []
    elements_list = []
    for idx, atom_mass in enumerate(atoms.mass):
        try:
            elements_list.append(mass_to_element[round(atom_mass)])
        except KeyError:
            error_mass.append(atoms.ids[idx])
            elements_list.append("CG-{}".format(atoms.lammps_type[idx]))
    # if len(error_mass)!=0:
    # print('Warning! Unknown mass for atom(s) {}.\nIf this is not a CG system, check the \"mass_to_element\" dictionary in data_file_parser, and/or your data file\'s atomic masses.'.format(error_mass))
    atoms.append(element=elements_list)

    # Create the bonds object, sorting by bond id if the order is not being preserved.
    if not preserve_bond_order:
        bond_ids = sorted([key for key in temp_bonds.keys()])
    bonds = IntraModeList(
        ids=bond_ids,
        lammps_type=[temp_bonds[ids][0] for ids in bond_ids],
        atom_ids=[[temp_bonds[ids][1], temp_bonds[ids][2]] for ids in bond_ids],
    )

    # Create the angles object, sorting by angle id if the order is not being preserved.
    if not preserve_angle_order:
        angle_ids = sorted([key for key in temp_angles.keys()])
    angles = IntraModeList(
        ids=angle_ids,
        lammps_type=[temp_angles[ids][0] for ids in angle_ids],
        atom_ids=[
            [temp_angles[ids][1], temp_angles[ids][2], temp_angles[ids][3]]
            for ids in angle_ids
        ],
    )

    # Create the dihedrals object, sorting by dihedral id if the order is not being preserved.
    if not preserve_dihedral_order:
        dihedral_ids = sorted([key for key in temp_dihedrals.keys()])
    dihedrals = IntraModeList(
        ids=dihedral_ids,
        lammps_type=[temp_dihedrals[ids][0] for ids in dihedral_ids],
        atom_ids=[
            [
                temp_dihedrals[ids][1],
                temp_dihedrals[ids][2],
                temp_dihedrals[ids][3],
                temp_dihedrals[ids][4],
            ]
            for ids in dihedral_ids
        ],
    )

    # Create the impropers object, sorting by improper id if the order is not being preserved.
    if not preserve_improper_order:
        improper_ids = sorted([key for key in temp_impropers.keys()])
    impropers = IntraModeList(
        ids=improper_ids,
        lammps_type=[temp_impropers[ids][0] for ids in improper_ids],
        atom_ids=[
            [
                temp_impropers[ids][1],
                temp_impropers[ids][2],
                temp_impropers[ids][3],
                temp_impropers[ids][4],
            ]
            for ids in improper_ids
        ],
    )

    return atoms, bonds, angles, dihedrals, impropers, box, adj_mat, extra_prop

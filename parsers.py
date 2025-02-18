#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

# Imports
import numpy as np
import pandas as pd


def parse_rxndf(rxn_file):
    """Parser for 'reaction details file' (.rxndf)

    Parameters
    ----------
    rxn_file: str
        Name of the reaction detail file.

    Returns Dictionary
    ------------------
    ~keys~
    Reaction number: int
    ~values~
    'reactant_molecules': list
        A list of the molecule type(s) that participate in the
        given reaction. Format: [molecule_type, ...]
    'product_molecules': list
        A list of the molecule type(s) that result from the
        given reaction. Format: [molecule_type, ...]
    'A': [float]
        A in k=A(T^b)exp(-Ea/RT). ()
    'b': [float]
        b in k=A(T^b)exp(-Ea/RT). ()
    'Ea': [float]
        Ea in k=A(T^b)exp(-Ea/RT). ()
    """

    rxndf = {}
    keywords = [
        "reactant_molecules",
        "product_molecules",
        "radial_dist",
        "A",
        "b",
        "Ea",
    ]
    with open(rxn_file, "r") as f:
        for line in f:
            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "Reaction":
                rxn_num = int(fields[2])
                rxndf[rxn_num] = {k: [] for k in keywords}
                continue
            if fields[0].strip(":") in keywords:
                key = fields[0].strip(":")
                if key == "reactant_molecules" or key == "product_molecules":
                    rxndf[rxn_num][key] = [i for i in fields[1:]]
                    continue
                rxndf[rxn_num][key] = [float(fields[1])]

    return rxndf


def parse_msf(msf_file):
    """Parser for master species file (.msf)

    Parameters
    ----------
    msf_file: str
        Name of the master species file.

    Returns Dictionary
    ------------------
    ~keys~
    Species: str
    ~values~
    'Atoms': list of lists
        [[atom_id,mol_id,atom_type,q,x,y,z], ...]
    'Bonds' or 'Angles' or 'Dihedrals' or 'Impropers': list of lists
        [[interaction_id,interaction_type,atom_i_id,atom_j_id, ...], ...]
    """

    msf = {}
    keywords = ["Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"]
    flag = None
    with open(msf_file, "r") as f:
        for line in f:
            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "Species:":
                sp = fields[1]
                msf[sp] = {}
                continue
            if fields[0] in keywords:
                flag = fields[0]
                continue
            if flag == "Atoms":
                if flag not in msf[sp].keys():
                    msf[sp][flag] = []
                msf[sp]["Atoms"].append(
                    [int(i) for i in fields[:3]] + [float(i) for i in fields[3:]]
                )
                continue
            if flag:
                if flag not in msf[sp].keys():
                    msf[sp][flag] = []
                msf[sp][flag].append([int(i) for i in fields])
                continue

    return msf


def parse_diffusion(diffusion_file):
    """Parser for diffusion file (.diffusion)

    Parameters
    ----------
    diffusion_file: str
        Name of the diffusion file.

    Returns
    -------
    voxels: dictionary
        keys - voxel number (int)
        values - list of lists of floats
            [[xmin, xmax],[ymin,ymax],[zmin,zmax]]
    diffusion_matrix: np.array
        Entries are diffusion coefficients, going from
        voxel num. (row index - 1) to voxel num. (column index - 1).
    """

    voxels = {}
    diffusion_list = []
    flag = None
    with open(diffusion_file, "r") as f:
        for line in f:
            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "voxel":
                v = int(fields[1])
                voxels[v] = []
                continue
            if "bounds" in fields[0]:
                voxels[v] += [[float(fields[1]), float(fields[2])]]
                continue
            if fields[0] == "diffusion_matrix":
                flag = "dm"
                continue
            if flag == "dm":
                diffusion_list.append([float(i) for i in fields])
                continue
    diffusion_matrix = np.array(diffusion_list)

    return voxels, diffusion_matrix


def parse_header(header_file):
    """Parser for reading a header file (.header)

    Parameters
    ----------
    header_file: str
        Name of the header file

    Returns
    -------
    header: dictionary
        keys - str
            atom_types, bond_types, angle_types, dihedral_types,
            improper_types, or masses
        values - int or list of tuples
            > for the 'types' items, and integer describing the
            number of that type of interaction
            > for the 'masses' item, [ (atom type, mass), ... ]
    """

    header = {"masses": {}}
    flag = None

    int_types = ["atom", "bond", "angle", "dihedral", "improper"]

    with open(header_file, "r") as f:
        for line in f:

            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "Masses":
                flag = "Masses"
                continue

            if not flag and len(fields) == 3:
                if fields[1] in int_types and fields[2] == "types":
                    header[fields[1] + "_types"] = int(fields[0])
                    continue

            if flag == "Masses":
                try:
                    header["masses"][int(fields[0])] = float(fields[1])
                except:
                    header["masses"] = [
                        (k, header["masses"][k])
                        for k in sorted(list(header["masses"].keys()))
                    ]
                    return header

    header["masses"] = [
        (k, header["masses"][k]) for k in sorted(list(header["masses"].keys()))
    ]

    return header


def parse_concentration(conc_file):
    """Parser for a concentration file (.concentration)

    Parameters
    ----------
    conc_file: str
        Name of the concentration file

    Returns
    -------
    counts: dict of dict of dict
        keys: [int][int][str]
            Step number, then reaction cycle number,
            then species name.
        values: int
            Number of species.

    times: dict of dict
        keys: [int][int]
            Step number, then reaction cycle number.
        values: float
            Time update.

    selected_rxns: dict of dict of list of int
        keys: [int][int]
            Step number, then reaction cycle number.
        values: list of int
            Selected reaction types.
    """

    counts, times, selected_rxns = {}, {}, {}
    flag = None
    step, rc = 0, 0
    with open(conc_file, "r") as f:
        for line in f:
            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if "--" in line:
                continue
            if fields[0] == "DiffusionStep" or fields[0] == "Step":
                step = int(fields[1])
                if step in counts.keys():
                    print(
                        "Error! Multiple DiffusionSteps labeled with the same ID in {}. Sending kill code...".format(
                            conc_file
                        )
                    )
                    return "kill", "kill", "kill"
                counts[step], times[step], selected_rxns[step] = {}, {}, {}
                continue
            if fields[0] == "ReactionCycle":
                rc = int(fields[1])
                if rc in counts[step].keys():
                    print(
                        "Error! Multiple ReactionCycles in DiffusionStep {} are labeled with the same ID in {}. Sending kill code...".format(
                            step, conc_file
                        )
                    )
                    return "kill", "kill", "kill"
                counts[step][rc] = {}
                continue
            if fields[0] == "MoleculeCounts":
                for i in list(range(len(fields)))[1::2]:
                    counts[step][rc][fields[i]] = int(fields[i + 1])
                continue
            if fields[0] == "SelectedReactionTypes":
                selected_rxns[step][rc] = [int(i) for i in fields[1:]]
                continue
            if fields[0] == "Time":
                times[step][rc] = float(fields[1])
                continue

    # Check that all entries have the same species
    if (
        len(
            set(
                [
                    tuple(sorted(list(counts[step][rc].keys())))
                    for step in counts.keys()
                    for rc in counts[step].keys()
                ]
            )
        )
        > 1
    ):
        print(
            "Warning! parse_concentrations found differing species lists for different steps/reaction cycles.\n"
            + "Entries will be padded such that all entries have the same species list.\n"
            + "Steps/reaction cycles will have species with counts of 0 added, where necessary."
        )
        species = sorted(
            set(
                [
                    sp
                    for species in [
                        sorted(list(counts[step][rc].keys()))
                        for step in counts.keys()
                        for rc in counts[step].keys()
                    ]
                    for sp in species
                ]
            )
        )
        for step in counts.keys():
            for rc in counts[step].keys():
                for sp1 in [
                    sp0 for sp0 in species if sp0 not in counts[step][rc].keys()
                ]:
                    counts[step][rc][sp1] = 0

    # Fill in missing selected_rxns, if needed
    missing_sr = []
    for step in counts.keys():
        if step not in selected_rxns.keys():
            selected_rxns[step] = {}
        for rc in [
            k for k in counts[step].keys() if k not in selected_rxns[step].keys()
        ]:
            selected_rxns[step][rc] = []
            missing_sr.append((step, rc))
    if missing_sr:
        if (0, 0) not in missing_sr:
            pass
            #print(
            #    "Warning! Missing entries for selected rxns for the following DiffusionStep/ReactionCycle pairs: {}".format(
            #        " ".join(["{},{}".format(i[0], i[1]) for i in missing_sr])
            #    )
            #)

    return counts, times, selected_rxns


def parse_scale(scale_file, windowsize_MDMCcycles=1e10):
    """Parser for a scaling file (.scale)

    Parameters
    ----------
    scale_file: str
        Name of the scaling file

    Returns
    -------
    scale: dict of dict of dict
        keys: [int][int][int]
            Step number, then reaction cycle number,
            then reaction type
        values: float
            Scaling.
    """

    scale = {}
    flag = None
    step, rc = 0, 0
    rxn_set = set()
    with open(scale_file, "r") as f:
        for line in f:
            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if "--" in line:
                continue
            if fields[0] in ["DiffusionStep", "Step"]:
                step = int(fields[1])
                if step in scale.keys():
                    print(
                        "parse_scale: Error! Multiple DiffusionSteps are labeled with identical IDs in scale file {}. Returning kill code...".format(
                            scale_file
                        )
                    )
                    return "kill", "kill"
                scale[step] = {}
                continue
            if fields[0] == "ReactionCycle":
                rc = int(fields[1])
                if rc in scale[step].keys():
                    print(
                        "parse_scale: Error! Multiple ReactionCycles are labeled with identical IDs in DiffusionStep {}, scale file {}. Returning kill code...".format(
                            step, scale_file
                        )
                    )
                    return "kill", "kill"
                scale[step][rc] = {}
                continue
            if fields[0] == "ReactionTypes":
                scale[step][rc] = [int(i) for i in fields[1:]]
                rxn_set.update(set(scale[step][rc]))
                continue
            if fields[0] == "ReactionScaling":
                if len(scale[step][rc]) != len(fields[1:]):
                    print("parse_scale: Error! ")
                    print(
                        "Lengths of ReactionType and ReactionScaling for DiffusionStep {}, ReactionCycle {} in scale file {} differ. ".format(
                            step, rc, scale_file
                        )
                    )
                    print("Sending kill code...")
                    return "kill", "kill"
                scale[step][rc] = {
                    scale[step][rc][idx]: float(fields[1 + idx])
                    for idx in range(len(fields[1:]))
                }
                continue
    rxn_set = sorted(list(rxn_set))

    # Create a list of MDMC cycles
    MDMCcycles = [
        (num, tup)
        for num, tup in enumerate(
            [
                (step, rc)
                for step in sorted(scale.keys())
                for rc in sorted(scale[step].keys())
            ]
        )
    ]
    windowsize_MDMCcycles = np.min([len(MDMCcycles), windowsize_MDMCcycles])
    MDMCcycles = MDMCcycles[int(len(MDMCcycles) - windowsize_MDMCcycles) :]

    # Print a warning if padding will occur
    missing_rxn = []
    for step, v in scale.items():
        for rc, vv in v.items():
            if np.all(sorted(list(vv.keys())) != rxn_set):
                missing_rxn.append((step, rc))
    if missing_rxn:
        print(
            "parse_scale: Warning! The following DiffusionStep/ReactionCyclepairs are missing reaction types present in other cycles: {}".format(
                " ".join(["{},{}".format(i[0], i[1]) for i in missing_rxn])
            )
        )
        print(
            "Padding these cycles by adding the missing reaction types witha scaling of 1.0"
        )

    # Create the scale dataframe, filling in missing rxn scales with 1.0000
    scale_data = {rxn: [1.0] * len(MDMCcycles) for rxn in rxn_set}
    for idx, tup in enumerate(MDMCcycles):
        for rxn, sc in scale[tup[1][0]][tup[1][1]].items():
            scale_data[rxn][idx] = sc
    scale = pd.DataFrame(
        index=[i[0] for i in MDMCcycles], columns=rxn_set, data=scale_data
    )

    return scale, MDMCcycles


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

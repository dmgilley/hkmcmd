#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import numpy as np
import pandas as pd
from scipy.spatial.distance import *
from copy import deepcopy
from sklearn import linear_model
from hybrid_mdmc.classes import MoleculeList, ReactionList


def voxels2voxelsmap(voxels):
    """Creates an object to map x/y/z indices to a voxel

    Parameters
    ----------
    voxels: dictionary
        - keys: int
            > voxel ID
        - values: list of lists
            > each sublist is [lmin,lmax]

    Returns
    -------
    voxelmap: dictionary
        > keys: tuple
            - (xidx,yidx,zidx): indices of minimum voxel bounds,
              indexed to x/y/z arrays
        > values: list
            - [ voxel ID, [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    x: np.array
        holds the left-most bounds for voxels, in the x direction
    y: np.array
        holds the left-most bounds for voxels, in the y direction
    z: np.array
        holds the left-most bounds for voxels, in the z direction
    """

    voxelsb = {tuple([tuple(i) for i in v]): k for k, v in voxels.items()}
    x = sorted(set([i for bounds in [v[0] for v in voxels.values()] for i in bounds]))
    y = sorted(set([i for bounds in [v[1] for v in voxels.values()] for i in bounds]))
    z = sorted(set([i for bounds in [v[2] for v in voxels.values()] for i in bounds]))
    voxelsmap = {}
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            for k in range(len(z) - 1):
                xx = [x[i], x[i + 1]]
                yy = [y[j], y[j + 1]]
                zz = [z[k], z[k + 1]]
                voxelsmap[(i, j, k)] = [
                    voxelsb[(tuple(xx), tuple(yy), tuple(zz))],
                    xx,
                    yy,
                    zz,
                ]
                if i == len(x) - 2:
                    voxelsmap[(i + 1, j, k)] = [
                        voxelsb[(tuple(xx), tuple(yy), tuple(zz))],
                        xx,
                        yy,
                        zz,
                    ]
                if j == len(y) - 2:
                    voxelsmap[(i, j + 1, k)] = [
                        voxelsb[(tuple(xx), tuple(yy), tuple(zz))],
                        xx,
                        yy,
                        zz,
                    ]
                if k == len(z) - 2:
                    voxelsmap[(i, j, k + 1)] = [
                        voxelsb[(tuple(xx), tuple(yy), tuple(zz))],
                        xx,
                        yy,
                        zz,
                    ]

    return voxelsmap, np.array(x), np.array(y), np.array(z)


def gen_molecules(atoms, atomtypes2moltype, voxels=None):
    """Creates an instance of the MoleculeList class

    Parameters
    ----------
    atoms: instance of AtomList class
    atomtypes2moltype: dict
        > keys: tuple
            - sorted tuple of atom types in the molecule
        > values: str
            - species name
    voxelmap: dict
        > keys: tuple
            - (xidx,yidx,zidx): indices of minimum voxel bounds,
               indexed to x/y/z arrays
        > values: list
            - [ voxel ID, [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    voxelsx: np.array
        holds the left-most bounds for voxels, in the x direction
    voxelsy: np.array
        holds the left-most bounds for voxels, in the y direction
    voxelsz: np.array
        holds the left-most bounds for voxels, in the z direction

    Returns
    -------
    MoleculeList: instance of MoleculeList
    """

    molecules_dict = {m: {} for m in set(atoms.mol_id)}
    for m in molecules_dict.keys():
        atom_idxs = [idx for idx in range(len(atoms.ids)) if atoms.mol_id[idx] == m]
        molecules_dict[m]["atom_IDs"] = sorted(atoms.ids[atom_idxs].tolist())
        molecules_dict[m]["cog"] = np.mean(
            np.array([atoms.x[atom_idxs], atoms.y[atom_idxs], atoms.z[atom_idxs]]),
            axis=1,
        )
        molecules_dict[m]["type"] = atomtypes2moltype[
            tuple(sorted(atoms.lammps_type[atom_idxs]))
        ]
        molecules_dict[m]["voxel"] = []
        if voxels is not None:
            molecules_dict[m]["voxel"] = voxels.assign_voxel_ID_to_given_COG(
                molecules_dict[m]["cog"]
            )
    mkeys = sorted(list(molecules_dict.keys()))
    return MoleculeList(
        ids=mkeys,
        atom_ids=[molecules_dict[m]["atom_IDs"] for m in mkeys],
        cogs=[molecules_dict[m]["cog"] for m in mkeys],
        mol_types=[molecules_dict[m]["type"] for m in mkeys],
        voxel_idxs=[molecules_dict[m]["voxel"] for m in mkeys],
    )


def get_rxns_serial(
    molecules, voxels, diffusion_rate, rxnscaling, rxn_data, minimum_diffusion_rate
):
    """Creates instance of class hybridmdmc.classes.ReactionList.

    Parameters
    ----------


    Returns
    -------
    rxns: instance of class HybridMDMC_Classes.ReactionList
    """
    # The total reaction rate is calculated as follows:
    # timeofdiffusion = 1 / diffusionrate
    #   diffusionrate is the rate of diffusion from voxel i to voxel j
    #   for bimolecular rxns, where the first reactant is in voxel i
    #   and the second is in voxel j
    # timeofrxn = 1 / (rawrxnrate * rxnscaling)
    # totalrxnrate = 1 / (timeofdiffusion + timeofrxn)
    #
    # It is assumed that if diffusion/reaction scaling is NOT
    # requested, all diffusion rates are np.inf/all reaction scalings
    # are 1.

    voxelID2idx = {vID: idx for idx, vID in enumerate(voxels.IDs)}
    reactions, number_of_reactions = {}, 0
    cycle = rxnscaling.index[-1]

    for (
        rxn_type
    ) in rxn_data.keys():  # Loop over every possible reaction in the rxn_data

        # Unimolecular reactions
        if len(rxn_data[rxn_type]["reactant_molecules"]) == 1:
            reactive_molecule_idxs_i = [
                idx
                for idx, molID in enumerate(molecules.ids)
                if molecules.mol_types[idx] in rxn_data[rxn_type]["reactant_molecules"]
            ]
            reactions.update(
                {
                    reaction_index
                    + 1
                    + number_of_reactions: [
                        rxn_type,  # reaction type
                        [molecules.ids[molecule_index]],  # list of molecule ID
                        rxn_data[rxn_type]["rawrate"]
                        * rxnscaling.loc[cycle, rxn_type],  # reactione event rate
                    ]
                    for reaction_index, molecule_index in enumerate(
                        reactive_molecule_idxs_i
                    )
                }
            )
            number_of_reactions = len(reactions)
            continue

        # Bimolecular reactions
        elif len(rxn_data[rxn_type]["reactant_molecules"]) == 2:
            reactive_molecule_idxs_i = [
                idx
                for idx, molID in enumerate(molecules.ids)
                if molecules.mol_types[idx]
                == rxn_data[rxn_type]["reactant_molecules"][0]
            ]
            reactive_molecule_idxs_j = [
                idx
                for idx, molID in enumerate(molecules.ids)
                if molecules.mol_types[idx]
                == rxn_data[rxn_type]["reactant_molecules"][1]
            ]
            diffusion_rates_forward = np.array(
                [
                    [
                        diffusion_rate[molecules.mol_types[molidx_i]][
                            voxelID2idx[molecules.voxel_idxs[molidx_i]],
                            voxelID2idx[molecules.voxel_idxs[molidx_j]],
                        ]
                        for molidx_j in reactive_molecule_idxs_j
                    ]
                    for molidx_i in reactive_molecule_idxs_i
                ],
            )
            diffusion_rates_reverse = np.array(
                [
                    [
                        diffusion_rate[molecules.mol_types[molidx_j]][
                            voxelID2idx[molecules.voxel_idxs[molidx_j]],
                            voxelID2idx[molecules.voxel_idxs[molidx_i]],
                        ]
                        for molidx_i in reactive_molecule_idxs_i
                    ]
                    for molidx_j in reactive_molecule_idxs_j
                ],
            )
            diffusion_rates = np.maximum(
                diffusion_rates_reverse, diffusion_rates_forward
            )
            upper_triangle_indices = np.triu_indices_from(diffusion_rates, k=1)
            filtered_indices = [
                (i, j)
                for i, j in zip(*upper_triangle_indices)
                if diffusion_rates[i, j] > minimum_diffusion_rate
            ]

            mask = np.nonzero(diffusion_rates <= minimum_diffusion_rate)
            times_of_diffusion = 1 / np.where(
                diffusion_rates > minimum_diffusion_rate, diffusion_rates, 1
            )
            times_of_diffusion[mask] = np.inf

            #            times_of_diffusion = 1 / diffusion_rates
            times_of_rxn = 1 / (
                rxn_data[rxn_type]["rawrate"] * rxnscaling.loc[cycle, rxn_type]
            )
            reactions.update(
                {
                    reaction_index
                    + 1
                    + number_of_reactions: [
                        rxn_type,  # reaction type
                        [
                            molecules.ids[reactive_molecule_idxs_i[value[0]]],
                            molecules.ids[reactive_molecule_idxs_j[value[1]]],
                        ],  # list of molecule IDs
                        1
                        / (
                            times_of_diffusion[value] + times_of_rxn
                        ),  # reaction event rate
                    ]
                    for reaction_index, value in enumerate(filtered_indices)
                }
            )
            number_of_reactions = len(reactions)
            continue

        # More than bimolecular rxn
        elif len(rxn_data[rxn_type]["reactant_molecules"]) > 2:
            print(
                "Error! hybridmdmc.functions.get_rxns does not currently support reactions between more than 2 molecules."
            )

    reactions = ReactionList(
        ids=sorted(list(reactions.keys())),
        rxn_types=[
            reactions[k][0] for k in sorted(list(reactions.keys()))
        ],  # reaction types
        reactants=[
            reactions[k][1] for k in sorted(list(reactions.keys()))
        ],  # lists of reactant molecule(s) ID(s)
        rates=[reactions[k][2] for k in sorted(list(reactions.keys()))],
    )  # rates

    return reactions


def get_rxns(
    molecules,
    vox,
    voxelID2idx,
    diffusion_rates,
    rxnscaling,
    delete,
    rxn_data,
    diffusion_cutoff,
    temp,
):
    """Creates instance of class hybridmdmc.classes.ReactionList.

    Parameters
    ----------
    molecules: instance of class HybridMDMC_Classes.MoleculeList
    vox: int
        Voxel number.
    voxelID2idx: dict
        Maps voxel IDs to their index in the diffusion matrix.
    diffusion_matrix: np.ndarray
        NxN array, where N is the number of voxels.
        An entry represents the diffusion coefficient going from
        voxel number (row index - 1) to voxel number (column index - 1).
    delete: list of int
        Molecules IDs that are scheduled for deletion.
    rxn_data: dict
        keys: int
            Reaction number.
        values: dict
            'reactant_molecules': list
                A list of the molecule type(s) that participate in the
                given reaction. Format: [molecule_type, ...]
            'product_molecules': list
                A list of the molecule type(s) that result from the
                given reaction. Format: [molecule_type, ...]
            'A': [float]
                A in k=A(T^b)exp(-Ea/RT).
            'b': [float]
                b in k=A(T^b)exp(-Ea/RT).
            'Ea': [float]
                Ea in k=A(T^b)exp(-Ea/RT).
    diffusion_cutoff: float
        Cutoff distance for considering diffusion.
    temp: float
        Temperature.

    Returns
    -------
    rxns: instance of class HybridMDMC_Classes.ReactionList
    """
    # The total reaction rate is calculated as follows:
    # timeofdiffusion = 1 / diffusionrate
    #   diffusionrate is the rate of diffusion from voxel i to voxel j
    #   for bimolecular rxns, where the first reactant is in voxel i
    #   and the second is in voxel j
    # timeofrxn = 1 / (rawrxnrate * rxnscaling)
    # totalrxnrate = 1 / (timeofdiffusion + timeofrxn)
    #
    # It is assumed that if diffusion/reaction scaling is NOT
    # requested, all diffusion rates are np.inf/all reaction scalings
    # are 1.

    molID2idx = {
        _: idx for idx, _ in enumerate(molecules.ids)
    }  # Create dictionary to map molecule IDs to their index in the molecules object
    rmol_IDs = sorted(
        [
            molecules.ids[idx]
            for idx in range(len(molecules.ids))
            if molecules.voxel_idxs[idx] == vox
        ]
    )  # Get molecule IDs for the reactive species
    rxns, rxn_count = {}, 1
    cycle = rxnscaling.index[-1]

    for (
        rxn_type
    ) in rxn_data.keys():  # Loop over every possible reaction in the rxn_data
        for rmID in rmol_IDs:  # Loop over each reactive molecule in the voxel
            # if rmID in delete: continue # If this molecule has already reacted, skip it
            if (
                molecules.mol_types[molID2idx[rmID]]
                not in rxn_data[rxn_type]["reactant_molecules"]
            ):
                continue  # Skip if this molecule cannot undergo this rxn
            reactant_types = deepcopy(rxn_data[rxn_type]["reactant_molecules"])
            reactant_types.remove(
                molecules.mol_types[molID2idx[rmID]]
            )  # List of remaining reactant molecule types, if any, after removing the active molecule type

            # Unimolecular rxn
            if len(reactant_types) == 0:
                timeofdiffusion = 0
                timeofrxn = 1 / (
                    rxn_data[rxn_type]["rawrate"] * rxnscaling.loc[cycle, rxn_type]
                )
                rxns[rxn_count] = {
                    "rxn_type": rxn_type,
                    "reactants": [rmID],
                    "rate": 1 / (timeofdiffusion + timeofrxn),
                }
                rxn_count += 1
                continue

            # Bimolecular rxn
            if len(reactant_types) == 1:
                partner_IDs = [
                    _
                    for _ in molecules.ids
                    if diffusion_rates[molecules.mol_types[molID2idx[rmID]]][
                        voxelID2idx[vox], voxelID2idx[molecules.voxel_idxs[molID2idx[_]]]
                    ]
                    >= diffusion_cutoff
                    and molecules.mol_types[molID2idx[_]] == reactant_types[0]
                    # and ID not in delete
                    and _ != rmID
                ]
                for pID in partner_IDs:
                    timeofdiffusion = 1 / (
                        diffusion_rates[molecules.mol_types[molID2idx[rmID]]][
                            voxelID2idx[vox],
                            voxelID2idx[molecules.voxel_idxs[molID2idx[pID]]],
                        ]
                    )
                    timeofrxn = 1 / (
                        rxn_data[rxn_type]["rawrate"] * rxnscaling.loc[cycle, rxn_type]
                    )
                    rxns[rxn_count] = {
                        "rxn_type": rxn_type,
                        "reactants": [rmID, pID],
                        "rate": 1 / (timeofdiffusion + timeofrxn),
                    }
                    rxn_count += 1
                continue

            # More than bimolecular rxn
            if len(reactant_types) > 1:
                print(
                    "Error! hybridmdmc.functions.get_rxns does not currently support reactions between more than 2 molecules. Crashed in voxel {}, reaction number {}.".format(
                        vox, rxn_type
                    )
                )

    rxns = ReactionList(
        ids=sorted(list(rxns.keys())),
        rxn_types=[rxns[k]["rxn_type"] for k in sorted(list(rxns.keys()))],
        reactants=[rxns[k]["reactants"] for k in sorted(list(rxns.keys()))],
        rates=[rxns[k]["rate"] for k in sorted(list(rxns.keys()))],
    )

    return rxns


def update_topology(
    msf, molecules, add, delete, atoms, bonds, angles, dihedrals, impropers
):
    """Updates the topology a given system.

    Parameters
    ----------
    msf: dict
        keys: str
            Species name.
        values: dict
            'Atoms': list of lists
                [[atom ID, mol ID, atom type, q, x, y, z], ...]
            'Bonds'/'Angles'/'Dihedrals'/'Impropers': list of lists
                [[interaction ID, interaction type, atom i ID, atom j ID, ...] ...]
    molecules: instance of class HybridMDMC_Classes.MoleculeList
    add: dict
        keys: int
            New molecule ID.
        values: list
            [ species type (str), cog (np.ndarray) ]
    delete: list of int
        Molecule IDs scheduled for deletion.
    atoms: instance of class mol_classes.AtomList
    bonds: instance of class mol_classes.IntraModeList
    angles: instance of class mol_classes.IntraModeList
    dihedrals: instance of class mol_classes.IntraModeList
    impropers: instance of class mol_classes.IntraModeList

    Returns
    -------
    atoms: instance of class mol_classes.AtomList
    bonds: instance of class mol_classes.IntraModeList
    angles: instance of class mol_classes.IntraModeList
    dihedrals: instance of class mol_classes.IntraModeList
    impropers: instance of class mol_classes.IntraModeList
    """

    # Delete the appropriate atoms and interactions
    if len(delete) != 0:
        del_atom_ids = [
            i
            for l in [
                molecules.atom_ids[idx]
                for idx, ID in enumerate(molecules.ids)
                if ID in delete
            ]
            for i in l
        ]
        atoms.del_idx(list(np.array(del_atom_ids) - 1), reassign_lammps_type=False)
        bonds.del_by_atom_ids(
            del_atom_ids, reassign_lammps_type=False, old2new_ids=atoms.old2new_ids
        )
        angles.del_by_atom_ids(
            del_atom_ids, reassign_lammps_type=False, old2new_ids=atoms.old2new_ids
        )
        dihedrals.del_by_atom_ids(
            del_atom_ids, reassign_lammps_type=False, old2new_ids=atoms.old2new_ids
        )
        impropers.del_by_atom_ids(
            del_atom_ids, reassign_lammps_type=False, old2new_ids=atoms.old2new_ids
        )

    # Adjust the mol numbers
    old2new_molids = {
        m: m_idx + 1 for m_idx, m in enumerate(sorted(list(set(atoms.mol_id))))
    }
    atoms.mol_id = np.array([old2new_molids[m] for m in atoms.mol_id])

    # Add the appropriate atoms and interactions
    mol_count = len(set(atoms.mol_id))
    atom_count = len(atoms.ids)
    interaction_types = ["Bonds", "Angles", "Dihedrals", "Impropers"]
    interaction_counts = {
        "Bonds": len(bonds.ids),
        "Angles": len(angles.ids),
        "Dihedrals": len(dihedrals.ids),
        "Impropers": len(impropers.ids),
    }
    new_atoms = {}
    new_interactions = {i: {} for i in interaction_types}

    # Loop through the "add" object, adding the new atoms and interactions to their respective dictionaries
    # Adding the items to dictionaries then converting to lists in one step is faster
    for k, v in add.items():

        # Create a dictionary to map this species' atom ids to the system's atom ids
        ref2sys_atomids = {
            a[0]: atom_count + a_idx + 1 for a_idx, a in enumerate(msf[v[0]]["Atoms"])
        }

        # Add the new atoms, with appropriate system ids and adjusted coordinates
        avg_coords = np.mean(np.array([a[4:] for a in msf[v[0]]["Atoms"]]), axis=0)
        for a in msf[v[0]]["Atoms"]:
            new_atoms[ref2sys_atomids[a[0]]] = [
                mol_count + 1,
                a[2],
                a[3],
                a[4] - avg_coords[0] + v[1][0],
                a[5] - avg_coords[1] + v[1][1],
                a[6] - avg_coords[2] + v[1][2],
            ]
        mol_count += 1
        atom_count += len(msf[v[0]]["Atoms"])

        # Add the new bonding interactions, with the appropriate ids
        for interaction in interaction_types:
            if interaction not in msf[v[0]].keys():
                continue
            for i in msf[v[0]][interaction]:
                new_interactions[interaction][interaction_counts[interaction] + 1] = [
                    i[1]
                ] + [ref2sys_atomids[j] for j in i[2:]]
                interaction_counts[interaction] += 1

    # Add all the atoms and interactions to the respective objects
    akeys = sorted(list(new_atoms.keys()))
    bkeys = sorted(list(new_interactions["Bonds"].keys()))
    ankeys = sorted(list(new_interactions["Angles"].keys()))
    dkeys = sorted(list(new_interactions["Dihedrals"].keys()))
    ikeys = sorted(list(new_interactions["Impropers"].keys()))
    atoms.append(
        ids=akeys,
        mol_id=[new_atoms[k][0] for k in akeys],
        lammps_type=[new_atoms[k][1] for k in akeys],
        charge=[new_atoms[k][2] for k in akeys],
        x=[new_atoms[k][3] for k in akeys],
        y=[new_atoms[k][4] for k in akeys],
        z=[new_atoms[k][5] for k in akeys],
    )
    bonds.append(
        ids=bkeys,
        lammps_type=[new_interactions["Bonds"][bk][0] for bk in bkeys],
        atom_ids=[new_interactions["Bonds"][bk][1:] for bk in bkeys],
    )
    angles.append(
        ids=ankeys,
        lammps_type=[new_interactions["Angles"][ank][0] for ank in ankeys],
        atom_ids=[new_interactions["Angles"][ank][1:] for ank in ankeys],
    )
    dihedrals.append(
        ids=dkeys,
        lammps_type=[new_interactions["Dihedrals"][dk][0] for dk in dkeys],
        atom_ids=[new_interactions["Dihedrals"][dk][1:] for dk in dkeys],
    )
    impropers.append(
        ids=ikeys,
        lammps_type=[new_interactions["Impropers"][ik][0] for ik in ikeys],
        atom_ids=[new_interactions["Impropers"][ik][1:] for ik in ikeys],
    )

    # Adjust charges to nuetralize system
    atoms.charge = np.array(
        [i - np.sum(atoms.charge) / len(atoms.charge) for i in atoms.charge]
    )

    return atoms, bonds, angles, dihedrals, impropers


def find_dist_same(geo, box):
    """Calculates MIC distance between points.

    Calculates the minimum image convention (MIC) distance between points,
    as supplied in a single array (hence "same"). Distance is Minkowski
    distance of p=1.0.

    Parameters
    ----------
    geo: np.ndarray
        Nx3 array, where each row holds the x, y, and z coordinates.
    box: list of lists of floats
        [[ xmin, xmax ], ... ]

    Returns
    -------
    rs: np.ndarray
        NxN array, where an entry is the distance between original point
        of row index and original point of column index.
    """

    rs = np.zeros((geo.shape[0], geo.shape[0]))
    for i in range(3):
        dist = squareform(pdist(geo[:, i : i + 1], "minkowski", p=1.0))
        l, l2 = box[i][1] - box[i][0], (box[i][1] - box[i][0]) / 2.0
        while not (dist <= l2).all():
            dist -= l * (dist > l2)
            dist = np.abs(dist)
        rs += dist**2
    rs = np.sqrt(rs)

    return rs


def get_progression(
    counts, times, selected_rxns, reaction_types, species, windowsize=1e10
):
    """Converts history information into a progression DataFrame.

    Given the system history, this function returns a DataFrame
    containing the molecule concentrations and the number of times each
    reaction ha been selected over the past MDMC cycles.

    Parameters
    ----------
    counts: dict
        - counts[diffusion step][rxn step][species] = count
        - counts[int][int][str] = int
    times: dict
        - times[diffusion step][rxn step] = time
        - times[int][int] = float
    selected_rxns: dict
        - selected_rxns[diffusion step][rxn step][rxn num] = count
        - selected_rxns[int][int][int] = int
    reaction_types: list
        - list of reaction types
    species: list
        - list of species
    window_size: int
        - number of previous cycles to keep in the progression object

    Returns
    -------
    progression: pandas.DataFrame
        - indices: mdmc cycles
        - columns: 'time', reactions, species
        - entries: time entry, or reaction count,
                   or species number fraction for each cycle
    mdmc_cycles: list
        - enries are tuples; (mdmc cycle, (diffusion step, rxn step))
    """

    # Create a list of relevant MDMC cycles, with each entry containing
    # the new MDMD cycle (0 indexed) number and the (step,rc)
    mdmc_cycles = [
        (num, tup)
        for num, tup in enumerate(
            [
                (step, rc)
                for step in sorted(counts.keys())
                for rc in sorted(counts[step].keys())
            ]
        )
    ]
    windowsize = np.min([windowsize, len(mdmc_cycles)])
    mdmc_cycles = mdmc_cycles[int(len(mdmc_cycles) - windowsize) :]

    # Declare the progression columns and data objects
    index_ = [i[0] for i in mdmc_cycles]
    columns_ = ["time"] + reaction_types + species
    data_ = {}

    # Fill in the data
    data_["time"] = [times[i[1][0]][i[1][1]] for i in mdmc_cycles]
    for r in reaction_types:
        data_[r] = [selected_rxns[i[1][0]][i[1][1]].count(r) for i in mdmc_cycles]
    for s in species:
        data_[s] = [float(counts[i[1][0]][i[1][1]][s]) for i in mdmc_cycles]

    # Create the progression DataFrame
    progression = pd.DataFrame(data=data_, columns=columns_, index=index_)

    # Change from molecule counts to molecule number fraction
    # progression.loc[:,species[0]:species[-1]] = progression.loc[:,species[0]:species[-1]].div(
    #    progression.loc[:,species[0]:species[-1]].sum(axis=1),axis=0)

    return progression, mdmc_cycles


def update_progression(
    progression, molecules, species, time, selected_rxns, windowsize_MDMCcycles=1e10
):

    add_index = [progression.index[-1] + 1]
    add_columns = progression.columns
    add_data = {"time": [time]}
    for c in [_ for _ in progression.columns if type(_) == int]:
        add_data[c] = [selected_rxns.count(c)]
    for sp in species:
        add_data[sp] = [float(len([1 for _ in molecules.mol_types if _ == sp]))]
    add_df = pd.DataFrame(data=add_data, index=add_index, columns=add_columns)
    # add_df.loc[:,species[0]:species[-1]] = add_df.loc[:,species[0]:species[-1]].div(add_df.loc[:,species[0]:species[-1]].sum(axis=1),axis=0)
    progression = pd.concat([progression, add_df])
    if progression.shape[0] > windowsize_MDMCcycles:
        progression = progression.drop(labels=progression.index[0], axis=0)

    return progression


def write_tracking_file(
    file_name, lines, driver='"Driver name"', time='"datetime"', step=None, new=False
):

    parameter = "a"
    file_type = "Unknown file type"
    if new:
        parameter = "w"
        if file_name.lower()[-4:] == "conc":
            file_type = "Concentration"
        if file_name.lower()[-3:] == "log":
            file_type = "Log"
        if file_name.lower()[-5:] == "scale":
            file_type = "Scaling"

    with open(file_name, parameter) as f:
        if new:
            f.write(
                "# {} file created for tracking progress in a Hybrid MD/MC simulation.\n".format(
                    file_type
                )
            )
            f.write("# Create with {} on {}\n".format(driver, time))
        f.write("\n{}\n\nDiffusionStep {}\n".format("-" * 75, step))
        for l in lines:
            f.write(l)

    return


def get_rxnmatrix(rxndata, masterspecies):

    rxndict = {s: {n: 0 for n in rxndata.keys()} for s in masterspecies.keys()}
    for n in rxndata.keys():
        for s in rxndata[n]["reactant_molecules"]:
            rxndict[s][n] -= 1
        for s in rxndata[n]["product_molecules"]:
            rxndict[s][n] += 1

    return pd.DataFrame(
        data=rxndict,
        columns=sorted(list(masterspecies.keys())),
        index=sorted(list(rxndata.keys())),
    )


def get_PSSrxns(
    rxnmatrix,
    reaction_scaling,
    progression,
    windowsize_slope,
    windowsize_rxnselection,
    scalingcriteria_concentration_slope,
    scalingcriteria_concentration_cycles,
    scalingcriteria_rxnselection_count,
):
    """Determine the reactions that are in psuedo steady state.

    Given certain criteria and the system history, this function
    returns a list of reactions that are in psuedo steady state.

    Parameters
    ----------
    """

    # If the windowsize_slope exceeds the current number of MDMC cycles,
    # return an empty list. By definition, no reaction can be at PSS.
    if windowsize_slope > len(progression):
        return []

    # Reset the windowsize_rxnselection if it is greater than the length of
    # the progression df.
    windowsize_rxnselection = np.min([windowsize_rxnselection, len(progression)])

    # For each species, determine the number of cycles that have occured
    # for which the slope of the concentration has met the
    # scalingcriteria_concentration_slope criteria.
    cycles_of_constant_slope = {
        _: get_cycles_of_constant_slope(
            progression.loc[:, _].to_numpy(),
            windowsize_slope,
            scalingcriteria_concentration_slope,
            number_of_windows=scalingcriteria_concentration_cycles,
        )
        for _ in progression.columns
        if _ != "time" and type(_) == str
    }

    # Loop over the reactions, checking each for PSS based on the
    # desired criteria.
    PSSrxns = []
    true_dynamics_steps = [
        idx
        for idx in progression.index[-windowsize_rxnselection:]
        if np.all(reaction_scaling.loc[idx] == 1.0)
    ]
    for rxntype in [_ for _ in progression.columns if _ != 0 and type(_) != str]:

        # Create a list of all species involved in this reaction,
        # reactants AND products.
        species_rxn = [_ for _ in rxnmatrix.columns if rxnmatrix.loc[rxntype, _] != 0]

        # Check that all of the species for this reaction have an unchanging
        # slope of concentration. If not, exit the loop for this rxntype.
        if not np.all(
            np.array([cycles_of_constant_slope[_] for _ in species_rxn])
            >= scalingcriteria_concentration_cycles
        ):
            continue

        # Check that this rxntype has been selected the desired number
        # of times over the desired number of previous cycles. If not,
        # exit the loop for this rxntype.
        # if not np.sum(progression.loc[progression.index[-windowsize_rxnselection]:,rxntype]) >= scalingcriteria_rxnselection_count:
        if (
            not np.sum([progression.loc[idx, rxntype] for idx in true_dynamics_steps])
            >= scalingcriteria_rxnselection_count
        ):
            continue

        # If this point is reached, the reaction may be added to the
        # PSS list.
        PSSrxns.append(rxntype)

    return PSSrxns


def scalerxns(
    reaction_scaling,
    PSSrxns,
    windowsize_scalingpause,
    scalingfactor_adjuster,
    scalingfactor_minimum,
    rxnlist="all",
):

    # Declare the last cycle
    lastcycle = reaction_scaling.index[-1]

    # Handle the default rxnlist
    if rxnlist == "all":
        rxnlist = sorted(list(reaction_scaling.columns))

    # If the number of cycles that have passed since the last scaling factor was adjusted
    # is less than the windowsize_scalingpause, return unscaled reactions.
    if (
        np.min(
            [
                get_cyclesofconstantscaling(reaction_scaling.loc[:, rxntype].to_numpy())
                for rxntype in reaction_scaling.columns
            ]
        )
        < windowsize_scalingpause
    ):
        reaction_scaling.loc[lastcycle + 1] = [1.0 for _ in reaction_scaling.columns]
        return reaction_scaling

    # Create a dictionary to track the new reaction scaling
    newscaling = {
        rxntype: reaction_scaling.loc[lastcycle, rxntype]
        for rxntype in reaction_scaling.columns
    }

    # Loop over the rxnlist and scale reactions
    for rxntype in rxnlist:

        # If this reaction is not in PSS, upscale it.
        if rxntype not in PSSrxns:
            newscaling[rxntype] /= scalingfactor_adjuster
            continue

        # If this reaction is in PSS, downscale it.
        if rxntype in PSSrxns:
            newscaling[rxntype] *= scalingfactor_adjuster
            continue

    # Append the new reaction scaling to rxnscaling
    reaction_scaling.loc[lastcycle + 1] = [
        newscaling[rxntype] for rxntype in reaction_scaling.columns
    ]

    # If all reactions are scaled, upscale
    while not np.any(reaction_scaling.loc[lastcycle + 1] >= 1.0):
        reaction_scaling.loc[lastcycle + 1] /= scalingfactor_adjuster

    # Adjust scalings that are outside of the scaling limits.
    reaction_scaling[reaction_scaling > 1.0] = 1.0
    reaction_scaling[reaction_scaling < scalingfactor_minimum] = scalingfactor_minimum

    return reaction_scaling


def get_cyclesofconstantscaling(reaction_scaling: np.array) -> int:
    """Determine the number of cycles that the scaling factor has remained constant.

    Given an array of reaction rate scaling factors, this
    function calculates the number of steps that the scaling
    factor has remained unchanged, counted from the most recent step.

    Parameters
    ----------
    reaction_scaling : numpy array
        An array of reaction rate scaling factors.

    Returns
    -------
    int
        The number of cycles that the scaling factor has remained constant.

    Edge Cases
    ----------
    If the reaction_scaling array has no length, return a value of 0.

    If the reaction_scaling array has no changes, return the length of the array.

    If the reaction_scaling changed between the previous step and the current step, return 1.
        e.g. np.array([1.0,1.0,0.1]) returns 1.
    """
    if len(reaction_scaling) == 0:
        return 0
    change_indices = np.where(np.diff(reaction_scaling) != 0)[0] + 1
    if len(change_indices) == 0:
        return len(reaction_scaling)
    return len(reaction_scaling) - change_indices[-1]


def get_cycles_of_constant_slope(
    particle_count: np.array,
    windowsize_slope: int,
    scalingcriteria_concentration_slope: float,
    number_of_windows: int = 1,
) -> int:
    """Determine the number of cycles that the slope of the particle count has remained constant.

    Parameters
    ----------
    particle_count : numpy.array
        An array containing the particle count of a species over time.

    windowsize_slope : int
        The number of cycles to use when calculating the slope.

    scalingcriteria_concentration_slope : float
        The maximum slope of the concentration that is considered "constant".

    number_of_windows : int
        The number of windows over which to calculate the slope.

    Returns
    -------
    int
        The number of cycles that the slope of the concentration has remained constant.
    """

    # If the progression dataframe has no length, return a value of 0.
    if len(particle_count) == 0 or len(particle_count) < windowsize_slope:
        return 0

    # Adjust the steps
    steps = number_of_windows + windowsize_slope - 1
    if steps > len(particle_count):
        steps = len(particle_count)
    particle_count = particle_count[-steps:]

    # Calculate the absolute value of the slopes using the provided window size.
    slope = np.array(
        [
            np.abs(
                linear_model.LinearRegression()
                .fit(
                    np.array(range(windowsize_slope)).reshape(-1, 1),
                    particle_count[idx : idx + windowsize_slope],
                )
                .coef_[0]
            )
            for idx in range(0, len(particle_count) - windowsize_slope + 1)
        ]
    )

    # If all of the cycles satisfy the scalingcriteria_concentration_slope, return the length of slope.
    if np.all(slope <= scalingcriteria_concentration_slope):
        return len(slope)

    # Otherwise, return the index of the first instance where the slope does NOT satisfy scalingcriteria_concentration_slope.
    return np.argwhere(slope[::-1] > scalingcriteria_concentration_slope)[0][0]

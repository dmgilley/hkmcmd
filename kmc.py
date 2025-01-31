#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

# Imports
import numpy as np
from copy import deepcopy


def rfkmc_rxn(rxns, rxn_data, molecules, translate_distance=0.0):
    """Performs a single rejection-free KMC event selection and time update.

    Parameters
    ----------
    rxns: instance of ReactionList class

    rxn_data: dictionary
        Returned from 'parse_rxndf'

    molecules: instance of MoleculeList class

    translate_distance: float, optional
        Distance to translate molecules after reaction (Angstrom).
        Default: 0.0 Angstrom

    Returns
    -------
    add: dictionary
        > keys - int
            Molecule number
        > values - list
            [ str,          np.array ]
            [ species type, cog ]

    delete: list of int
        List of reactant molecule IDs to delete

    dt: float
        Time update.
    """

    adjust = np.sqrt(translate_distance**2 / 3)
    molID2idx = {ID: idx for idx, ID in enumerate(molecules.ids)}
    u1 = np.random.random()
    rxn_idx = np.argwhere(np.cumsum(rxns.rates) >= np.sum(rxns.rates) * u1)[0][0]
    u2 = 0
    while u2 == 0:
        u2 = np.random.random()
    dt = -np.log(u2) / np.sum(rxns.rates)
    delete = [rxns.reactants[rxn_idx]]
    rt = rxns.rxn_types[rxn_idx]
    reactants = rxns.reactants[rxn_idx]
    products = rxn_data[rt]["product_molecules"]
    cog = [molecules.cogs[molID2idx[i]] for i in reactants]
    if len(reactants) == 1:
        if len(products) == 1:
            add = {1: [products[0], cog[0]]}
        if len(products) == 2:
            add = {1: [products[0], cog[0] - adjust], 2: [products[1], cog[0] + adjust]}
    if len(reactants) == 2:
        if len(products) == 1:
            add = {1: [products[0], cog[1]]}
        if len(products) == 2:
            if (
                molecules.mol_types[molID2idx[reactants[0]]]
                == rxn_data[rt]["reactant_molecules"][0]
            ):
                add = {1: [products[0], cog[0]], 2: [products[1], cog[1]]}
            else:
                add = {1: [products[0], cog[1]], 2: [products[1], cog[0]]}

    return add, delete, dt


def spkmc_rxn(rxns, rxn_data, molecules, Rmax, translate_distance=0.0):
    """Performs a single synchronous parallel KMC event selection and time update.

    Parameters
    ----------
    rxns: instance of ReactionList class

    rxn_data: dictionary
        Returned from 'parse_rxndf'

    molecules: instance of MoleculeList class

    Rmax: float
        Maximum total rate of reactions among all voxels

    translate_distance: float, optional
        Distance to translate molecules after reaction (Angstrom).
        Default: 0.0 Angstrom

    Returns
    -------
    add: dictionary
        > keys - int
            Molecule number
        > values - list
            [ str,          np.array ]
            [ species type, cog ]

    delete: list of int
        List of reactant molecule IDs to delete

    dt: float
        Time update

    rxn_types: int
        Reaction type selected.
        Reaction type 0 is the code for null event.
    """

    adjust = np.sqrt(translate_distance**2 / 3)
    molID2idx = {ID: idx for idx, ID in enumerate(molecules.ids)}
    u2 = 0
    while u2 == 0:
        u2 = np.random.random()
    dt = -np.log(u2) / Rmax
    u1 = np.random.random()
    rates = deepcopy(rxns.rates)
    # rates = rates.tolist()
    rates.append(Rmax - np.sum(rates))
    rxn_idx = np.argwhere(np.cumsum(rates) >= np.sum(rates) * u1)[0][0]
    if rxn_idx == len(rates) - 1:
        return {}, [], dt, 0
    delete = [rxns.reactants[rxn_idx]]
    rt = rxns.rxn_types[rxn_idx]
    reactants = rxns.reactants[rxn_idx]
    products = rxn_data[rt]["product_molecules"]
    cog = [molecules.cogs[molID2idx[i]] for i in reactants]
    if len(reactants) == 1:
        if len(products) == 1:
            add = {1: [products[0], cog[0]]}
        if len(products) == 2:
            add = {1: [products[0], cog[0] - adjust], 2: [products[1], cog[0] + adjust]}
    if len(reactants) == 2:
        if len(products) == 1:
            add = {1: [products[0], cog[1]]}
        if len(products) == 2:
            if (
                molecules.mol_types[molID2idx[reactants[0]]]
                == rxn_data[rt]["reactant_molecules"][0]
            ):
                add = {1: [products[0], cog[0]], 2: [products[1], cog[1]]}
            else:
                add = {1: [products[0], cog[1]], 2: [products[1], cog[0]]}

    return add, delete, dt, rt


def serial_kmc_rxn(rxns, rxn_data, molecules, translate_distance=0.0):
    """Performs a single synchronous parallel KMC event selection and time update.

    Parameters
    ----------
    rxns: instance of ReactionList class

    rxn_data: dictionary
        Returned from 'parse_rxndf'

    molecules: instance of MoleculeList class

    Rmax: float
        Maximum total rate of reactions among all voxels

    translate_distance: float, optional
        Distance to translate molecules after reaction (Angstrom).
        Default: 0.0 Angstrom

    Returns
    -------
    add: dictionary
        > keys - int
            Molecule number
        > values - list
            [ str,          np.array ]
            [ species type, cog ]

    delete: list of int
        List of reactant molecule IDs to delete

    dt: float
        Time update

    rxn_types: int
        Reaction type selected.
        Reaction type 0 is the code for null event.
    """

    adjust = np.sqrt(translate_distance**2 / 3)
    molID2idx = {ID: idx for idx, ID in enumerate(molecules.ids)}
    u2 = 0
    rates = deepcopy(rxns.rates)
    while u2 == 0:
        u2 = np.random.random()
    dt = -np.log(u2) / np.sum(rates)
    u1 = np.random.random()
    rxn_idx = np.argwhere(np.cumsum(rates) >= np.sum(rates) * u1)[0][0]
    delete = rxns.reactants[rxn_idx]
    rt = rxns.rxn_types[rxn_idx]
    reactants = rxns.reactants[rxn_idx]
    products = rxn_data[rt]["product_molecules"]
    cog = [molecules.cogs[molID2idx[i]] for i in reactants]
    if len(reactants) == 1:
        if len(products) == 1:
            add = {1: [products[0], cog[0]]}
        if len(products) == 2:
            add = {1: [products[0], cog[0] - adjust], 2: [products[1], cog[0] + adjust]}
    if len(reactants) == 2:
        if len(products) == 1:
            add = {1: [products[0], cog[1]]}
        if len(products) == 2:
            if (
                molecules.mol_types[molID2idx[reactants[0]]]
                == rxn_data[rt]["reactant_molecules"][0]
            ):
                add = {1: [products[0], cog[0]], 2: [products[1], cog[1]]}
            else:
                add = {1: [products[0], cog[1]], 2: [products[1], cog[0]]}

    return add, delete, dt, rt

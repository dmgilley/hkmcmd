#!usr/bin/env python3
# Author
#    Dylan M. Gilley
#    dgilley@purdue.edu


import numpy as np


class MoleculeList:
    """Class for arrays of all molecule attributes.

    Attributes
    -----------
    ids: numpy array of integers
        ID of each molecule.
    atom_ids: list of lists of integers
        Atom ids in each molecule.
    cogs: numpy array of numpy arrays
        Center of geometry of each molecule.
    mol_types: numpy array of strings
        Molecule type of each molecule, as defined in the \"msf\" file.
    voxels: numpy array of integers
        Voxel number of each molecule.
    rxn_ids: list of list of integers
        Reaction ID of each possible reactions for each molecule.

    Methods
    -----------

    Errors
    -----------
    "Error! Mismatch in the length of MoleculeList attributes. Exiting..."
        Self-explanatory

    Additional notes
    -----------

    Future work
    -----------
    """

    # Initialize
    def __init__(
        self, ids=[], atom_ids=[], cogs=[], mol_types=[], voxel_idxs=[], rxn_ids=[]
    ):
        self.ids = np.array(ids)
        self.atom_ids = atom_ids
        self.cogs = np.array(cogs)
        self.mol_types = np.array(mol_types)
        self.voxel_idxs = np.array(voxel_idxs)
        self.rxn_ids = rxn_ids

        # Checks
        all_keys = ["ids", "atom_ids", "cogs", "mol_types", "voxel_idxs", "rxn_ids"]
        Ls = {_: len(self.__dict__[_]) for _ in all_keys if len(self.__dict__[_]) != 0}
        if len(set([Ls[k] for k in Ls.keys()])) > 1:
            print(
                "Error! Mismatch in the length of MoleculeList attributes. Exiting..."
            )
            print(
                "Length of each attribute is: {}".format(
                    ", ".join(["{}: {}".format(k, Ls[k]) for k in Ls.keys()])
                )
            )
            quit()

    # Append info for new molecules
    def append(
        self, ids=[], atom_ids=[], cogs=[], mol_types=[], voxel_idxs=[], rxn_ids=[]
    ):
        self.ids = np.array(list(self.ids) + ids)
        self.atom_ids = self.atom_ids + atom_ids
        self.cogs = np.array(list(self.cogs) + cogs)
        self.mol_types = np.array(list(self.mol_types) + mol_types)
        self.voxel_idxs = np.array(list(self.voxel_idxs) + voxel_idxs)
        self.rxn_ids = self.rxn_ids + rxn_ids

    # Get ids based on a specific value of an attribute
    def get_idx(
        self, ids=[], atom_ids=[], cogs=[], mol_types=[], voxels=[], rxn_ids=[]
    ):
        if ids != []:
            return [_ for _, v in enumerate(self.ids) if v in ids]
        if atom_ids != []:
            return [
                _
                for _, v in enumerate(self.atom_ids)
                if sum([1 for i in v if i in atom_ids])
            ]
        if cogs != []:
            return [_ for _, v in enumerate(self.cogs) if v in cogs]
        if mol_types != []:
            return [_ for _, v in enumerate(self.mol_types) if v in mol_types]
        if voxels != []:
            return [_ for _, v in enumerate(self.voxel_idxs) if v in voxels]
        if rxn_ids != []:
            return [
                _
                for _, v in enumerate(self.rxn_ids)
                if sum([1 for i in v if i in rxn_ids])
            ]

    # Delete entries from all attributes at the supplied indices
    def del_idx(self, idx, reassign_ids=True):
        self.atom_ids = [v for _, v in enumerate(self.atom_ids) if _ not in idx]
        self.cogs = np.array([v for _, v in enumerate(self.cogs) if _ not in idx])
        self.mol_types = np.array(
            [v for _, v in enumerate(self.mol_types) if _ not in idx]
        )
        self.voxel_idxs = np.array(
            [v for _, v in enumerate(self.voxel_idxs) if _ not in idx]
        )
        self.rxn_ids = [v for _, v in enumerate(self.rxn_ids) if _ not in idx]

        # Reassign continuous ids from 1,2... onwards, not just delete entries.
        if reassign_ids:
            self.ids = [v for _, v in enumerate(self.ids) if _ not in idx]
            self.old2new_ids = {v: _ + 1 for _, v in enumerate(self.ids)}
            self.new2old_ids = {_ + 1: v for _, v in enumerate(self.ids)}
            self.ids = np.array(list(range(1, len(self.ids) + 1)))
        else:
            self.ids = np.array([v for _, v in enumerate(self.ids) if _ not in idx])

    def get_atomID2moleculeID(self):
        return {
            aID: self.ids[midx]
            for midx in range(len(self.ids))
            for aID in self.atom_ids[midx]
        }

    def get_cog(self, atomslist, box=[]):
        self.cog = np.array(
            [
                np.mean(
                    np.array(
                        [
                            [atomslist.x[idx], atomslist.y[idx], atomslist.z[idx]]
                            for idx, ID in enumerate(atomslist.ids)
                            if ID in matoms
                        ]
                    ),
                    axis=0,
                )
                for matoms in self.atom_ids
            ]
        )
        # Wrap back into the box, if it's provided
        if len(box):
            for d in range(3):
                while not np.all(self.cog[:, d] >= box[d][0]):
                    self.cog[:, d] = np.where(
                        self.cog[:, d] >= box[d][0],
                        self.cog[:, d],
                        self.cog[:, d] + (box[d][1] - box[d][0]),
                    )
                while not np.all(self.cog[:, d] <= box[d][1]):
                    self.cog[:, d] = np.where(
                        self.cog[:, d] <= box[d][1],
                        self.cog[:, d],
                        self.cog[:, d] - (box[d][1] - box[d][0]),
                    )

    def assign_voxel_idxs(self, voxel_origins, voxel_boundaries):
        self.voxel_idxs = assign_voxel_idxs_to_COGs(
            self.cogs, voxel_origins, voxel_boundaries
        )
        return self.voxel_idxs


def assign_voxel_idxs_to_COGs(
    COGs: list,
    voxel_origins: list,
    voxel_boundaries: list,
) -> list:
    voxel_origin_to_idx = {origin: idx for idx, origin in enumerate(voxel_origins)}
    x_voxel_minima = np.array(voxel_boundaries)[:, 0, 0]
    y_voxel_minima = np.array(voxel_boundaries)[:, 1, 0]
    z_voxel_minima = np.array(voxel_boundaries)[:, 2, 0]
    COG_nearest_minima = [
        tuple(
            [
                x_voxel_minima[np.argwhere(COG[0] >= x_voxel_minima)[-1][0]],
                y_voxel_minima[np.argwhere(COG[1] >= y_voxel_minima)[-1][0]],
                z_voxel_minima[np.argwhere(COG[2] >= z_voxel_minima)[-1][0]],
            ]
        )
        for COG in COGs
    ]
    return np.array([voxel_origin_to_idx[origin] for origin in COG_nearest_minima])


class ReactionList:
    """Class for arrays of reaction information.

    Attributes
    -----------
    ids: list of integers
        Reaction IDs (i.e. system number assigned to specific reaction).
    rxn_types: list of integers
        Reaction types, as specified in the rxndf file.
    reactants: lists of lists of integers
        Reactants for the specific reaction, as specified by the molecule ID.
    rates: list of floats
        Reaction rates.

    Methods
    -----------

    Errors
    -----------
    "Error! Mismatch in the length of ReactionList attributes. Exiting..."
        Self-explanatory

    Additional notes
    -----------

    Future work
    -----------
    """

    def __init__(self, ids=[], rxn_types=[], reactants=[], rates=[]):
        self.ids = ids
        self.rxn_types = rxn_types
        self.reactants = reactants
        self.rates = rates

        # Checks
        all_keys = ["ids", "rxn_types", "reactants", "rates"]
        Ls = {_: len(self.__dict__[_]) for _ in all_keys if len(self.__dict__[_]) != 0}
        if len(set([Ls[k] for k in Ls.keys()])) > 1:
            print(
                "Error! Mismatch in the length of ReactionList attributes. Exiting..."
            )
            print(
                "Length of each attribute is: {}".format(
                    ", ".join(["{}: {}".format(k, Ls[k]) for k in Ls.keys()])
                )
            )
            quit()

    # Get idx based on a specific value of an attribute
    def get_idx(self, rxn_types=[], reactants=[]):
        if rxn_types != []:
            return [_ for _, v in enumerate(self.rxn_types) if v in rxn_types]
        if reactants != []:
            return sorted(
                set(
                    [
                        i
                        for i, v in enumerate(self.reactants)
                        if sum([1 for _ in v if _ in reactants])
                    ]
                )
            )

    # Delete entries from all attributes at the supplied indices
    def del_idx(self, idx):
        self.ids = [v for _, v in enumerate(self.ids) if _ not in idx]

    # Delete entries from all attributes if the mode is formed by the atoms with indices supplied. Basically, remove all modes containing the atoms to be removed.
    def del_by_reactants(self, reactants):
        idx = self.get_idx(reactants=reactants)
        self.del_idx(idx)

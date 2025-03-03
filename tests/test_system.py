#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import unittest
import numpy as np
from copy import deepcopy
from hybrid_mdmc.interactions import get_interactions_lists_from_molcules_list
from hybrid_mdmc.filehandlers import parse_data_file
from hybrid_mdmc.system import *


class TestSystemState(unittest.TestCase):

    def setUp(self):
        self.system_state = SystemState(filename="fiction.system_state.json")
        self.system_state.read_data_from_json()

        # Read and clean the system data file
        self.system_data = SystemData("fiction", "fiction")
        self.system_data.read_json()
        self.system_data.clean()
        for reaction in self.system_data.reactions:
            reaction.calculate_raw_rate(188, method="Arrhenius")

        # Read the data file
        (
            atoms_list,
            bonds_list,
            angles_list,
            dihedrals_list,
            impropers_list,
            box,
            _,
        ) = parse_data_file(
            "fiction.in.data",
            atom_style="full",
        )
        self.molecules_list = [
            Molecule(ID=mID)
            for mID in sorted(list(set([atom.molecule_ID for atom in atoms_list])))
        ]

        molecule_kinds = ["A", "A", "A", "A2", "A2B2", "AB2C", "AC"]
        voxel_idxs = [1, 2, 3, 4, 5, 6, 7]
        for idx, molecule in enumerate(self.molecules_list):
            molecule.fill_lists(
                atoms_list=atoms_list,
                bonds_list=bonds_list,
                angles_list=angles_list,
                dihedrals_list=dihedrals_list,
                impropers_list=impropers_list,
            )
            molecule.kind = molecule_kinds[idx]
            for atom in molecule.atoms:
                atom.molecule_kind = molecule.kind
            molecule.voxel_idx = voxel_idxs[idx]

        return

    def test_check_atoms_list(self):

        atoms_list = (get_interactions_lists_from_molcules_list(self.molecules_list))[0]
        check = self.system_state.check_atoms_list(atoms_list)
        self.assertIs(check, True)

        with self.assertRaises(ValueError):
            self.system_state.check_atoms_list(atoms_list[1:])

        with self.assertRaises(ValueError):
            atoms_list_bad = deepcopy(atoms_list)
            atoms_list_bad[0].ID = 100
            self.system_state.check_atoms_list(atoms_list_bad)

        with self.assertRaises(ValueError):
            atoms_list_bad = deepcopy(atoms_list)
            atoms_list_bad[0].molecule_ID = 100
            self.system_state.check_atoms_list(atoms_list_bad)

        with self.assertRaises(ValueError):
            atoms_list_bad = deepcopy(atoms_list)
            atoms_list_bad[0].molecule_kind = "wrong"
            self.system_state.check_atoms_list(atoms_list_bad)

        return

    def test_get_molecule_kind(self):

        molecule_kind = self.system_state.get_molecule_kind(1)
        self.assertEqual(molecule_kind, "A")

        molecule_kind = self.system_state.get_molecule_kind(4)
        self.assertEqual(molecule_kind, "A2")

        molecule_kind = self.system_state.get_molecule_kind(5)
        self.assertEqual(molecule_kind, "A2B2")

        molecule_kind = self.system_state.get_molecule_kind(6)
        self.assertEqual(molecule_kind, "AB2C")

        molecule_kind = self.system_state.get_molecule_kind(7)
        self.assertEqual(molecule_kind, "AC")

        with self.assertRaises(ValueError):
            self.system_state.get_molecule_kind(100)

        return

    def test_assemble_reaction_scaling_df(self):
        df = self.system_state.assemble_reaction_scaling_df()
        self.assertEqual(df.shape, (5, 3))
        self.assertEqual(df.columns.tolist(), [1, 2, 3])
        self.assertEqual(df.index.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(
            np.all(df.to_numpy()),
            np.all(
                np.array(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 0.1, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                )
            ),
        )
        return

    def test_assemble_progression_df(self):
        df = self.system_state.assemble_progression_df(
            sorted([molecule.kind for molecule in self.system_data.species])
        )
        self.assertEqual(df.shape, (5, 9))
        self.assertEqual(
            sorted([str(_) for _ in df.columns.tolist()]),
            sorted(
                [str(_) for _ in ["A", "A2", "A2B2", "AB2C", "AC", "time", 1, 2, 3]]
            ),
        )
        self.assertEqual(df.loc[:, "A"].tolist(), [3, 1, 3, 1, 3])
        self.assertEqual(df.loc[:, "A2"].tolist(), [1, 2, 1, 2, 1])
        self.assertEqual(df.loc[:, "A2B2"].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(df.loc[:, "AB2C"].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(df.loc[:, "AC"].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(df.loc[:, "time"].tolist(), [0.0, 1.3, 1.4, 4.7, 5.9])
        self.assertEqual(df.loc[:, 1].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(df.loc[:, 2].tolist(), [0, 1, 0, 1, 0])
        self.assertEqual(df.loc[:, 3].tolist(), [0, 0, 1, 0, 1])

        return

    def test_update_molecules(self):
        self.system_state.update_molecules(self.molecules_list)
        self.assertEqual(self.system_state.molecule_IDs.shape, (6, 15))
        self.assertEqual(
            self.system_state.molecule_IDs[-1, :].tolist(),
            [1, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7],
        )
        self.assertEqual(self.system_state.molecule_kinds.shape, (6, 15))
        self.assertEqual(
            self.system_state.molecule_kinds[-1, :].tolist(),
            [
                "A",
                "A",
                "A",
                "A2",
                "A2",
                "A2B2",
                "A2B2",
                "A2B2",
                "A2B2",
                "AB2C",
                "AB2C",
                "AB2C",
                "AB2C",
                "AC",
                "AC",
            ],
        )
        return

    def test_update_reactions(self):
        df = self.system_state.assemble_reaction_scaling_df()
        new = pd.DataFrame(
            {1: [1e-2], 2: [1e-3], 3: [1e-1]}, index=[5], columns=[1, 2, 3]
        )
        df = pd.concat([df, new], ignore_index=False)
        selections = [1, 0, 0]
        self.system_state.update_reactions(selections, df)
        self.assertEqual(
            np.all(self.system_state.reaction_selections),
            np.all(
                np.array(
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                    ]
                )
            ),
        )
        self.assertEqual(
            np.all(self.system_state.reaction_scalings),
            np.all(
                np.array(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 0.1, 1.0],
                        [1.0, 1.0, 1.0],
                        [1e-2, 1e-3, 1e-1],
                    ]
                )
            ),
        )
        return

    def test_update_steps(self):
        self.system_state.update_steps(3, 1.0)
        self.assertEqual(self.system_state.diffusion_steps.tolist(), [0, 1, 1, 2, 2, 3])
        self.assertEqual(self.system_state.reactive_steps.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(
            self.system_state.times.tolist(), [0.0, 1.3, 1.4, 4.7, 5.9, 6.9]
        )
        return


if __name__ == "__main__":
    unittest.main()

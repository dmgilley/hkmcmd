#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import unittest
import numpy as np
import pandas as pd
from copy import deepcopy
from hybrid_mdmc.system import SystemData
from hybrid_mdmc.filehandlers import (
    parse_data_file,
)
from hybrid_mdmc.interactions import *
from hybrid_mdmc.reaction import *


class TestReaction(unittest.TestCase):

    def setUp(self):

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
            molecule.voxel_idx = voxel_idxs[idx]

        return

    def test_get_reactive_events_list(self):

        diffusion_rates_dict_matrix = {
            "A": np.zeros((8, 8)),
            "A2": np.zeros((8, 8)),
            "AB2C": np.zeros((8, 8)),
            "A2B2": np.zeros((8, 8)),
            # "AC": np.zeros((8,8)),
        }
        diffusion_rates_dict_matrix["A"][1, 2] = 1e10
        diffusion_rates_dict_matrix["A"][1, 3] = 0.0
        diffusion_rates_dict_matrix["A"][2, 1] = 1e2
        diffusion_rates_dict_matrix["A"][2, 3] = 1e6
        diffusion_rates_dict_matrix["A"][3, 1] = 1e2
        diffusion_rates_dict_matrix["A"][3, 2] = 1e4
        diffusion_rates_dict_matrix["A2"][4, 6] = 1e8
        reaction_scaling_df = pd.DataFrame(
            {1: [1.0], 2: [1.0], 3: [1.0], 4: [1.0]}, index=[1]
        )
        reaction_templates_list = deepcopy(self.system_data.reactions)
        minimum_diffusion_rate = 0.0
        reactive_events_list = get_reactive_events_list(
            self.molecules_list,
            diffusion_rates_dict_matrix,
            reaction_scaling_df,
            reaction_templates_list,
            minimum_diffusion_rate,
            avoid_double_counts=True,
        )
        self.assertEqual(len(reactive_events_list), 5)
        self.assertEqual(
            [_.event_rate for _ in reactive_events_list],
            [
                18188952.211601835,  # 4 -> 6
                22183559.71651404,  # 1 -> 2
                99.99955021774281,  # 3 -> 1
                956957.5535552616,  # 2 -> 3
                22232880.158981726,  # unimolecular
            ],
        )
        self.assertEqual(
            [
                [_.ID for _ in event.reactant_molecules]
                for event in reactive_events_list
            ],
            [[4, 6], [1, 2], [1, 3], [2, 3], [4]],
        )

        # allow double counts
        diffusion_rates_dict_matrix = {
            "A": np.zeros((8, 8)),
            "A2": np.zeros((8, 8)),
            "AB2C": np.zeros((8, 8)),
            "A2B2": np.zeros((8, 8)),
            "AC": np.zeros((8, 8)),
        }
        diffusion_rates_dict_matrix["A"][1, 2] = 1e10
        diffusion_rates_dict_matrix["A"][1, 3] = 0.0
        diffusion_rates_dict_matrix["A"][2, 1] = 1e2
        diffusion_rates_dict_matrix["A"][2, 3] = 1e6
        diffusion_rates_dict_matrix["A"][3, 1] = 1e2
        diffusion_rates_dict_matrix["A"][3, 2] = 1e4
        diffusion_rates_dict_matrix["A2"][4, 6] = 1e8
        reaction_scaling_df = pd.DataFrame(
            {1: [1.0], 2: [1.0], 3: [1.0], 4: [1.0]}, index=[1]
        )
        reaction_templates_list = deepcopy(self.system_data.reactions)
        minimum_diffusion_rate = 0.0
        reactive_events_list = get_reactive_events_list(
            self.molecules_list,
            diffusion_rates_dict_matrix,
            reaction_scaling_df,
            reaction_templates_list,
            minimum_diffusion_rate,
            avoid_double_counts=False,
        )
        self.assertEqual(len(reactive_events_list), 8)
        self.assertEqual(
            [_.event_rate for _ in reactive_events_list],
            [
                18188952.211601835,  # 4 -> 6
                22183559.71651404,  # 1 -> 2
                99.99955021774281,  # 1 -> 3
                22183559.71651404,  # 2 -> 1
                956957.5535552616,  # 2 -> 3
                99.99955021774281,  # 3 -> 1
                956957.5535552616,  # 3 -> 2
                22232880.158981726,  # unimolecular
            ],
        )
        self.assertEqual(
            [
                [_.ID for _ in event.reactant_molecules]
                for event in reactive_events_list
            ],
            [[4, 6], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [4]],
        )

        # min diffusion rate cutoff
        diffusion_rates_dict_matrix = {
            "A": np.zeros((8, 8)),
            "A2": np.zeros((8, 8)),
            "AB2C": np.zeros((8, 8)),
            "A2B2": np.zeros((8, 8)),
            "AC": np.zeros((8, 8)),
        }
        diffusion_rates_dict_matrix["A"][1, 2] = 0.0
        diffusion_rates_dict_matrix["A"][1, 3] = 0.0
        diffusion_rates_dict_matrix["A"][2, 1] = 1e10
        diffusion_rates_dict_matrix["A"][2, 3] = 0.0
        diffusion_rates_dict_matrix["A"][3, 1] = 0.0
        diffusion_rates_dict_matrix["A"][3, 2] = 0.0
        diffusion_rates_dict_matrix["A2"][4, 6] = 1e8
        reaction_scaling_df = pd.DataFrame(
            {1: [1.0], 2: [1.0], 3: [1.0], 4: [1.0]}, index=[1]
        )
        reaction_templates_list = deepcopy(self.system_data.reactions)
        minimum_diffusion_rate = 1e-99
        reactive_events_list = get_reactive_events_list(
            self.molecules_list,
            diffusion_rates_dict_matrix,
            reaction_scaling_df,
            reaction_templates_list,
            minimum_diffusion_rate,
            avoid_double_counts=True,
        )
        self.assertEqual(len(reactive_events_list), 3)
        self.assertEqual(
            [_.event_rate for _ in reactive_events_list],
            [
                18188952.211601835,  # 4 -> 6
                22183559.71651404,  # 2 -> 1
                22232880.158981726,  # unimolecular
            ],
        )
        self.assertEqual(
            [
                [_.ID for _ in event.reactant_molecules]
                for event in reactive_events_list
            ],
            [[4, 6], [1, 2], [4]],
        )

        return

    def test_kMC_event_selection(self):

        event_list = [
            Reaction(ID=1,event_rate=0.0),
            Reaction(ID=2,event_rate=1e2),
            Reaction(ID=3,event_rate=1e3),
        ]
        event_selection = [0]*1000
        time = [0]*1000
        for idx in range(1000):
            event, dt = kMC_event_selection(event_list)
            event_selection[idx] = event.ID
            time[idx] = dt

        self.assertEqual(event_selection.count(1), 0)
        self.assertGreater(event_selection.count(2), 71)
        self.assertLess(event_selection.count(3), 929)
        self.assertLess(event_selection.count(2), 111)
        self.assertGreater(event_selection.count(3), 889)
        self.assertGreater(np.mean(time), 0.0009*0.8)
        self.assertLess(np.mean(time), 0.00090000*1.2)

        return

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import unittest
import numpy as np
from copy import deepcopy
from hkmcmd import system, interactions, io
from hkmcmd.diffusion import calculate_direct_transition_rates, perform_random_walks, calculate_average_first_time_between_positions, assign_points_to_voxels, clean_voxel_assignments_list, calculate_mvabfa_fuzzy_boundary


class TestCalculateDiffusion(unittest.TestCase):

    def setUp(self):

        # Read and clean the system data file
        self.system_data = system.SystemData("fiction", "fiction")
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
        ) = io.parse_data_file(
            "fiction.in.data",
            atom_style="full",
        )
        self.molecules_list = [
            interactions.Molecule(ID=mID)
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

    def test_calculate_direct_transition_rates(self):

        mvabfa = np.array(
            [
                [1, 2, 3, 4, 1, 2, 3],
                [3, 2, 4, 1, 3, 2, 1],
                [3, 2, 1, 4, 2, 3, 1],
                [1, 3, 2, 4, 1, 2, 4],
            ]
        )

        local_rates_dict = calculate_direct_transition_rates(
            7,  # total_number_of_voxels: int,
            mvabfa,  # molecular_voxel_assignment_by_frame_array: np.ndarray,
            0.5,  # adjacent_transition_time: float,
            self.molecules_list,  # molecules_list: list,
        )

        self.assertEqual(
            np.all(local_rates_dict["A"]),
            np.all(
                (1 / 0.5)
                * np.array(
                    [
                        [
                            0,
                            1,
                            1,
                            0,
                        ],
                        [
                            0,
                            2,
                            1,
                            0,
                        ],
                        [
                            1,
                            0,
                            1,
                            1,
                        ],
                        [
                            1,
                            0,
                            0,
                            0,
                        ],
                    ]
                )
            ),
        )

        self.assertEqual(
            np.all(local_rates_dict["A2"]),
            np.all(
                (1 / 0.5)
                * np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            1,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            1,
                            0,
                            0,
                            1,
                        ],
                    ]
                )
            ),
        )

        self.assertEqual(
            np.all(local_rates_dict["A2B2"]),
            np.all(
                (1 / 0.5)
                * np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            0,
                        ],
                        [
                            1,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            1,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                    ]
                )
            ),
        )

        self.assertEqual(
            np.all(local_rates_dict["AB2C"]),
            np.all(
                (1 / 0.5)
                * np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            1,
                            1,
                            0,
                        ],
                        [
                            0,
                            1,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                    ]
                )
            ),
        )

        self.assertEqual(
            np.all(local_rates_dict["AC"]),
            np.all(
                (1 / 0.5)
                * np.array(
                    [
                        [
                            1,
                            0,
                            0,
                            1,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            1,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                    ]
                )
            ),
        )

        return

    def test_perform_random_walks(self):
        transfer_rates = np.array([[10, 25, 50, 100]] * 4)
        starting_position_idxs = np.array([0, 1, 2, 3])
        number_of_steps = 100_000
        number_of_steps = 1_000
        walkers_position, walkers_time = perform_random_walks(
            transfer_rates,
            starting_position_idxs,
            number_of_steps,
        )
        position_counts = np.unique(walkers_position, return_counts=True)
        molecules_list = [
            interactions.Molecule(ID=1, kind=1),
            interactions.Molecule(ID=2, kind=1),
            interactions.Molecule(ID=3, kind=1),
            interactions.Molecule(ID=4, kind=1),
        ]
        local_rates = calculate_direct_transition_rates(
            4,
            walkers_position,
            1,
            molecules_list,
        )[1]
        transfer_rates_sum = (
            np.array([[np.sum(transfer_rates, axis=1)] * 4]).reshape(4, 4).T
        )
        local_rates_sum = np.array([[np.sum(local_rates, axis=1)] * 4]).reshape(4, 4).T
        for row in range(4):
            for col in range(4):
                if row == col:
                    self.assertEqual(0.0, local_rates[row, col])
                else:
                    self.assertAlmostEqual(
                        transfer_rates[row, col] / np.sum(transfer_rates, axis=1)[row],
                        local_rates[row, col] / np.sum(local_rates, axis=1)[row],
                        delta=0.05,
                    )
        return

    def test_calculate_average_first_time_between_positions(self):

        walkers_position = np.array(
            [
                [0, 1, 2, 3],
                [2, 2, 3, 2],
                [0, 3, 1, 3],
                [1, 0, 2, 2],
            ]
        )
        walkers_time = np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
            ]
        )
        average_time = calculate_average_first_time_between_positions(
            walkers_position,
            walkers_time,
            4,
            recursion_interval=1,
        )

        self.assertTrue(
            np.all(
                average_time
                == np.array(
                    [
                        [
                            np.inf,
                            2,
                            1,
                            np.inf,
                        ],
                        [
                            3,
                            np.inf,
                            1,
                            2,
                        ],
                        [
                            1.5,
                            2,
                            np.inf,
                            1,
                        ],
                        [
                            1,
                            1,
                            4 / 3,
                            np.inf,
                        ],
                    ]
                )
            ),
        )

        return

    def test_assign_points_to_voxels(self):

        voxels = voxels.Voxels(box=[[-9, 9], [-9, 9], [-9, 9]], number_of_voxels=3)
        points = np.array(
            [
                [-9.0, -9.0, -9.0],
                [-10, -0.9, 0.5],  # [8, -0.9, 0.5]
                [0.0, 0.0, 0.0],
            ]
        )
        voxel_idxs = assign_points_to_voxels(points, voxels, fuzz=2.0)
        self.assertEqual(
            voxel_idxs,
            [
                [0, 2, 6, 8, 18, 20, 24, 26],
                [4, 22],
                [13],
            ],
        )
        voxel_idxs = assign_points_to_voxels(points, voxels, fuzz=0.0)
        self.assertEqual(voxel_idxs, [[0], [22], [13]])
        return

    def test_clean_voxel_assignments_list(self):
        voxel_idxs = [2, [2], [5, 3, 4, 2], [7, 6, 2], [7, 6]]
        voxel_idxs = clean_voxel_assignments_list(voxel_idxs)
        self.assertEqual(voxel_idxs[:-1], [2, 2, 2, 2])
        self.assertTrue(voxel_idxs[-1] in [6, 7])
        return

    def test_calculate_mvabfa_fuzzy_boundary(self):

        molecules_list = [
            interactions.Molecule(ID=1, kind="A", atoms=[interactions.Atom(ID=1, kind=1)]),
            interactions.Molecule(ID=2, kind="A", atoms=[interactions.Atom(ID=2, kind=1)]),
        ]

        # Without fuzzy boundary
        molecule_IDs, timesteps, mvabfa = calculate_mvabfa_fuzzy_boundary(
            "diffusion_fiction.diffusion.lammpstrj",
            molecules_list,
            number_of_voxels=[2, 2, 2],
            fuzz=0.0,
        )
        self.assertEqual(molecule_IDs, [1, 2])
        self.assertEqual(timesteps, [1, 2, 3])
        self.assertEqual(mvabfa[:,0].tolist(), [0, 7, 5])
        self.assertEqual(mvabfa[:,1].tolist(), [2, 6, 1])

        # With fuzzy boundary
        molecule_IDs, timesteps, mvabfa = calculate_mvabfa_fuzzy_boundary(
            "diffusion_fiction.diffusion.lammpstrj",
            molecules_list,
            number_of_voxels=[2, 2, 2],
            fuzz=1.0,
        )
        self.assertEqual(molecule_IDs, [1, 2])
        self.assertEqual(timesteps, [1, 2, 3])
        self.assertEqual(mvabfa[:,0].tolist(), [0, 0, 5])
        self.assertEqual(mvabfa[:,1].tolist(), [2, 2, 2])


if __name__ == "__main__":
    unittest.main()

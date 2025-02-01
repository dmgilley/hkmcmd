#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import unittest
import numpy as np
from copy import deepcopy
from hybrid_mdmc.voxels import (
    Voxels,
    canonicalize_voxel_inputs,
    create_voxel_boundaries_dictionary,
    calculate_shaft_overlap_idxs_1D,
    find_voxel_neighbors_with_shaft_overlap_method,
    assign_voxel_idx_to_given_COG,
    get_voxel_distance_groupings,
)


class TestVoxels(unittest.TestCase):

    def setUp(self):
        self.voxel = Voxels(
            box=[[-10, 10], [0, 4], [1, 10]],
            number_of_voxels=[5, 4, 3],
            xbounds=[-10, -6, -2, 2, 6, 10],
            ybounds=[0, 1, 2, 3, 4],
            zbounds=[1, 4, 7, 10],
        )

    def test_canonicalize_voxel_inputs(self):

        box_truth = [[-10, 10], [0, 4], [1, 10]]
        number_of_voxels_truth = [5, 4, 3]
        xbounds_truth = [-10, -6, -2, 2, 6, 10]
        ybounds_truth = [0, 1, 2, 3, 4]
        zbounds_truth = [1, 4, 7, 10]

        # all inputs
        box, number_of_voxels, xbounds, ybounds, zbounds = canonicalize_voxel_inputs(
            box=deepcopy(box_truth),
            number_of_voxels=deepcopy(number_of_voxels_truth),
            xbounds=deepcopy(xbounds_truth),
            ybounds=deepcopy(ybounds_truth),
            zbounds=deepcopy(zbounds_truth),
        )
        self.assertEqual(box, box_truth)
        self.assertEqual(number_of_voxels, number_of_voxels_truth)
        self.assertEqual(xbounds, xbounds_truth)
        self.assertEqual(ybounds, ybounds_truth)
        self.assertEqual(zbounds, zbounds_truth)

        # box and number_of_voxels
        box, number_of_voxels, xbounds, ybounds, zbounds = canonicalize_voxel_inputs(
            box=deepcopy(box_truth),
            number_of_voxels=deepcopy(number_of_voxels_truth),
        )
        self.assertEqual(box, box_truth)
        self.assertEqual(number_of_voxels, number_of_voxels_truth)
        self.assertEqual(xbounds, xbounds_truth)
        self.assertEqual(ybounds, ybounds_truth)
        self.assertEqual(zbounds, zbounds_truth)

        # box and bounds
        box, number_of_voxels, xbounds, ybounds, zbounds = canonicalize_voxel_inputs(
            box=deepcopy(box_truth),
            xbounds=deepcopy(xbounds_truth),
            ybounds=deepcopy(ybounds_truth),
            zbounds=deepcopy(zbounds_truth),
        )
        self.assertEqual(box, box_truth)
        self.assertEqual(number_of_voxels, number_of_voxels_truth)
        self.assertEqual(xbounds, xbounds_truth)
        self.assertEqual(ybounds, ybounds_truth)
        self.assertEqual(zbounds, zbounds_truth)

        # number_of_voxels and bounds
        box, number_of_voxels, xbounds, ybounds, zbounds = canonicalize_voxel_inputs(
            number_of_voxels=deepcopy(number_of_voxels_truth),
            xbounds=deepcopy(xbounds_truth),
            ybounds=deepcopy(ybounds_truth),
            zbounds=deepcopy(zbounds_truth),
        )
        self.assertEqual(box, box_truth)
        self.assertEqual(number_of_voxels, number_of_voxels_truth)
        self.assertEqual(xbounds, xbounds_truth)
        self.assertEqual(ybounds, ybounds_truth)
        self.assertEqual(zbounds, zbounds_truth)

    def test_canonicalize_voxel_inputs_exceptions(self):

        box_truth = [[-10, 10], [0, 4], [1, 10]]
        number_of_voxels_truth = [5, 4, 3]
        xbounds_truth = [-10, -6, -2, 2, 6, 10]
        ybounds_truth = [0, 1, 2, 3, 4]
        zbounds_truth = [1, 4, 7, 10]

        with self.assertRaises(ValueError):
            box, number_of_voxels, xbounds, ybounds, zbounds = (
                canonicalize_voxel_inputs(
                    box=[[-10, 10], [0, 4]],  # Invalid box dimensions
                    number_of_voxels=deepcopy(number_of_voxels_truth),
                    xbounds=deepcopy(xbounds_truth),
                    ybounds=deepcopy(ybounds_truth),
                    zbounds=deepcopy(zbounds_truth),
                )
            )

        with self.assertRaises(ValueError):
            box, number_of_voxels, xbounds, ybounds, zbounds = (
                canonicalize_voxel_inputs(
                    box=deepcopy(box_truth),
                    number_of_voxels=[5, 4, 2], # Invalid number of voxels
                    xbounds=deepcopy(xbounds_truth),
                    ybounds=deepcopy(ybounds_truth),
                    zbounds=deepcopy(zbounds_truth),
                )
            )

        with self.assertRaises(ValueError):
            box, number_of_voxels, xbounds, ybounds, zbounds = (
                canonicalize_voxel_inputs(
                    box=deepcopy(box_truth),
                    number_of_voxels=deepcopy(number_of_voxels_truth),
                    xbounds=deepcopy(xbounds_truth)[:-1], # Invalid xbounds
                    ybounds=deepcopy(ybounds_truth),
                    zbounds=deepcopy(zbounds_truth),
                )
            )

    def test_create_voxel_boundaries_dictionary(self):
        boundaries_dict = create_voxel_boundaries_dictionary(
            number_of_voxels=[2, 2, 2],
            xbounds=[0, 0.5, 1],
            ybounds=[0, 0.5, 1],
            zbounds=[0, 0.5, 1],
        )
        expected_dict = {
            0: [[0, 0.5], [0, 0.5], [0, 0.5]],
            1: [[0, 0.5], [0, 0.5], [0.5, 1]],
            2: [[0, 0.5], [0.5, 1], [0, 0.5]],
            3: [[0, 0.5], [0.5, 1], [0.5, 1]],
            4: [[0.5, 1], [0, 0.5], [0, 0.5]],
            5: [[0.5, 1], [0, 0.5], [0.5, 1]],
            6: [[0.5, 1], [0.5, 1], [0, 0.5]],
            7: [[0.5, 1], [0.5, 1], [0.5, 1]],
        }
        self.assertEqual(boundaries_dict, expected_dict)

    def test_calculate_shaft_overlap_idxs_1D(self):
        primary_bounds = np.array([0, 0.5])
        comparison_bounds = np.array([[0, 0.5], [0.5, 1], [1, 1.5]])
        overlap_idxs = calculate_shaft_overlap_idxs_1D(
            primary_bounds, comparison_bounds
        )
        self.assertTrue(np.array_equal(overlap_idxs, np.array([0, 1])))

    def test_find_voxel_neighbors_with_shaft_overlap_method(self):
        voxel_bounds = np.array(
            [
                [[0, 0.5], [0, 0.5], [0, 0.5]],
                [[0, 0.5], [0, 0.5], [0.5, 1]],
                [[0, 0.5], [0.5, 1], [0, 0.5]],
                [[0, 0.5], [0.5, 1], [0.5, 1]],
                [[0.5, 1], [0, 0.5], [0, 0.5]],
                [[0.5, 1], [0, 0.5], [0.5, 1]],
                [[0.5, 1], [0.5, 1], [0, 0.5]],
                [[0.5, 1], [0.5, 1], [0.5, 1]],
            ]
        )
        neighbors = find_voxel_neighbors_with_shaft_overlap_method(voxel_bounds)
        expected_neighbors = {
            0: [1, 2, 4],
            1: [0, 3, 5],
            2: [0, 3, 6],
            3: [1, 2, 7],
            4: [0, 5, 6],
            5: [1, 4, 7],
            6: [2, 4, 7],
            7: [3, 5, 6],
        }
        self.assertEqual(neighbors, expected_neighbors)

    def test_assign_voxel_idx_to_given_COG(self):
        COG = [0.25, 0.25, 0.25]
        voxel_idx = assign_voxel_idx_to_given_COG(
            COG, [0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1], [[0, 1], [0, 1], [0, 1]]
        )
        self.assertEqual(voxel_idx, 0)

    def test_get_voxel_distance_groupings(self):
        distances, groupings = get_voxel_distance_groupings(self.voxel)
        expected_distances = [0.5, 0.7071067811865476, 1.0]
        self.assertTrue(np.allclose(distances, expected_distances))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import unittest
import numpy as np
from copy import deepcopy
from hkmcmd.voxels import (
    Voxels,
    canonicalize_voxel_inputs,
    create_voxel_boundaries_dictionary,
    calculate_shaft_overlap_idxs_1D,
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
        return

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
        return

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
                    number_of_voxels=[5, 4, 2],  # Invalid number of voxels
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
                    xbounds=deepcopy(xbounds_truth)[:-1],  # Invalid xbounds
                    ybounds=deepcopy(ybounds_truth),
                    zbounds=deepcopy(zbounds_truth),
                )
            )
        return

    def test_create_voxel_boundaries_dictionary(self):
        boundaries_dict = create_voxel_boundaries_dictionary(
            number_of_voxels=[2, 4, 3],
            xbounds=[-9, -3, -1],
            ybounds=[1, 3, 5, 7, 9],
            zbounds=[-1, 2, 3, 4],
        )
        expected_dict = {
            0: [[-9, -3], [1, 3], [-1, 2]],
            1: [[-9, -3], [1, 3], [2, 3]],
            2: [[-9, -3], [1, 3], [3, 4]],
            3: [[-9, -3], [3, 5], [-1, 2]],
            4: [[-9, -3], [3, 5], [2, 3]],
            5: [[-9, -3], [3, 5], [3, 4]],
            6: [[-9, -3], [5, 7], [-1, 2]],
            7: [[-9, -3], [5, 7], [2, 3]],
            8: [[-9, -3], [5, 7], [3, 4]],
            9: [[-9, -3], [7, 9], [-1, 2]],
            10: [[-9, -3], [7, 9], [2, 3]],
            11: [[-9, -3], [7, 9], [3, 4]],
            12: [[-3, -1], [1, 3], [-1, 2]],
            13: [[-3, -1], [1, 3], [2, 3]],
            14: [[-3, -1], [1, 3], [3, 4]],
            15: [[-3, -1], [3, 5], [-1, 2]],
            16: [[-3, -1], [3, 5], [2, 3]],
            17: [[-3, -1], [3, 5], [3, 4]],
            18: [[-3, -1], [5, 7], [-1, 2]],
            19: [[-3, -1], [5, 7], [2, 3]],
            20: [[-3, -1], [5, 7], [3, 4]],
            21: [[-3, -1], [7, 9], [-1, 2]],
            22: [[-3, -1], [7, 9], [2, 3]],
            23: [[-3, -1], [7, 9], [3, 4]],
        }
        self.assertEqual(boundaries_dict, expected_dict)
        return

    def test_calculate_shaft_overlap_idxs_1D(self):
        bounds_of_primary_voxel = np.array([1, 2])
        bounds_of_comparison_voxels = np.array(
            [
                [3, 0, 0],  # no, broken
                [0, -1, 0],  # no, broken
                [0, 0.99999999, 0],  # no
                [0, 1, 1],  # yes
                [0, 1.00000001, 1],  # yes
                [0, 1.99999999, 1],  # yes
                [0, 2, 1],  # yes
                [0, 2.00000001, 1],  # yes
                [1, 1.99999999, 1],  # yes
                [1, 2, 1],  # yes
                [1, 2.00000001, 1],  # yes
                [1.99999999, 3, 1],  # yes
                [2, 3, 1],  # yes
                [2.00000001, 3, 0],  # no
                [3, 4, 0],  # no
            ]
        )
        expected_overlap_idxs = np.argwhere(
            bounds_of_comparison_voxels[:, 2] == 1
        ).flatten()
        bounds_of_comparison_voxels = bounds_of_comparison_voxels[:, 0:2]
        overlap_idxs = calculate_shaft_overlap_idxs_1D(
            bounds_of_primary_voxel, bounds_of_comparison_voxels
        )
        np.testing.assert_array_equal(overlap_idxs, expected_overlap_idxs)
        return
    
    def test_find_voxel_neighbors_with_shaft_overlap_method(self):
        return


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

# Imports
import numpy as np
import os
import sys


def calc_voxels(num_voxels, box, xbounds=[], ybounds=[], zbounds=[]):
    """ """
    # Define the voxels
    voxel_bounds = []
    given_bounds = [xbounds, ybounds, zbounds]
    dimensions = ["x", "y", "z"]
    for d in range(3):
        if len(given_bounds[d]):

            # Check for inconsistent bounds
            if box[d] != [given_bounds[d][0], given_bounds[d][-1]]:
                print("Error!")
                print(
                    "  Inconsistent boundaries in {}/calc_voxels.calc_voxels".format(
                        os.path.abspath(os.getcwd())
                    )
                )
                print(
                    "  Given box bounds for {} dimension: {}".format(
                        dimensions[d], box[d]
                    )
                )
                print(
                    "  Specified voxel bounds for {} dimension: {}".format(
                        dimensions[d], given_bounds[d]
                    )
                )
                return None

            voxel_bounds.append(given_bounds[d])
            continue
        voxel_bounds.append(
            [
                np.min(box[d])
                + (np.max(box[d]) - np.min(box[d])) / int(num_voxels[d]) * i
                for i in range(int(num_voxels[d]))
            ]
            + [np.max(box[d])]
        )
    voxels = {}
    count = 0
    for i in range(len(voxel_bounds[0]) - 1):
        for j in range(len(voxel_bounds[1]) - 1):
            for k in range(len(voxel_bounds[2]) - 1):
                idx = [i, j, k]
                voxels[count] = [
                    [voxel_bounds[d][idx[d]], voxel_bounds[d][idx[d] + 1]]
                    for d in range(3)
                ]
                count += 1
    return voxels


def get_box_fromvoxels(voxels):
    box = [
        [
            np.min([voxels[_][didx][0] for _ in voxels.keys()]),
            np.max([voxels[_][didx][1] for _ in voxels.keys()]),
        ]
        for didx in range(3)
    ]
    return box

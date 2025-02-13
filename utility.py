#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import datetime, os
import numpy as np
import pandas as pd
import scipy.spatial.distance as sds
from copy import deepcopy
from itertools import chain


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
    np.ndarray
        NxN array, where an entry is the distance between original point
        of row index and original point of column index.
    """

    rs = np.zeros((geo.shape[0], geo.shape[0]))
    for i in range(3):
        dist = sds.squareform(sds.pdist(geo[:, i : i + 1], "minkowski", p=1.0))
        l, l2 = box[i][1] - box[i][0], (box[i][1] - box[i][0]) / 2.0
        while not (dist <= l2).all():
            dist -= l * (dist > l2)
            dist = np.abs(dist)
        rs += dist**2
    rs = np.sqrt(rs)

    return rs


def flatten_nested_list(list_):
    while True in [isinstance(_, list) for _ in list_]:
        list_ = list(chain.from_iterable(list_))
    return list_


def histo_getbins(times, num_of_bins=100, tmin=None, tmax=None):
    if not tmax:
        tmax = np.min([np.max(_) for _ in times])
    if tmin is None:
        tmin = np.max([np.min(_) for _ in times])
    binsize = (tmax - tmin) / num_of_bins
    return (
        [[tmin + i * binsize, tmin + (i + 1) * binsize] for i in range(num_of_bins)],
        tmin,
        tmax,
    )


def combine_sample_set_statistics(observations, means, stds):
    observation_combined = np.sum(observations, axis=-1)
    mean_combined = np.sum(means * observations, axis=-1) / observation_combined
    variances = stds**2
    ds = (means - mean_combined[...,np.newaxis]) ** 2
    std_combined = np.sqrt(
        np.sum(observations * (variances + ds), axis=-1) / observation_combined
    )
    return observation_combined, mean_combined, std_combined


def calc_Ea(A, k, T):
    # A in same units as k; usually 1/s
    # k in same units as A; usually 1/s
    # T in K
    # Ea in kcal/mol
    R = 0.00198588  # kcal / mol K
    return -R * T * np.log(k / A / T)


def LJtimesteps_to_seconds(LJtimesteps, m, s, e, lammps_stepsize=0.001):
    return LJtimesteps * lammps_stepsize * np.sqrt(m) * s / np.sqrt(e)


class FileTracker:

    def __init__(self, name, overwrite=False):
        self.name = name
        if overwrite is True or not os.path.exists(self.name):
            self.create()
        return
    
    def create(self):
        with open(self.name, "w") as f:
            f.write(f"# File created {datetime.datetime.now()}\n")
        return

    def write(self, string):
        with open(self.name, "a") as f:
            f.write(string)
        return
    
    def write_separation(self, sep="-"):
        with open(self.name, "a") as f:
            f.write(f"\n{sep * 100}\n")
        return
    
    def write_array(self, array, asstr=True):
        preamble = ""
        if type(array) is tuple:
            preamble, array = array[0], array[1]
        if asstr is True:
            array = np.array_str(array,max_line_width=1e99)[1:-1]
        with open(self.name, "a") as f:
            f.write(f"{preamble}{array}\n")
        return


def read_direct_voxel_transition_rates(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    direct_voxel_transition_rates = {}
    parsing = False
    for line in lines:
        if line.split() == []:
            continue
        if "Frame" in line:
            frame = int(line.split()[1])
            direct_voxel_transition_rates[frame] = []
            parsing = True
        elif parsing is True:
            direct_voxel_transition_rates[frame].append(
                [float(i) for i in line.split()]
            )
    return {frame: np.array(v) for frame, v in direct_voxel_transition_rates.items()}

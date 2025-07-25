#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import numpy as np
from typing import Union
from hybrid_mdmc import utility


class Voxels:
    """Class for voxel information.

    NOTE voxels are zero indexed, and their IDs are simply their
    indices for all array and list attributes.

    Attributes
    -----------
    box: list
        Box dimensions, [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ].
    number of voxels: list
        Number of voxels in each dimension (length of three).
    xbounds: list
        X bounds for voxels, flattened. Voxel with index i has x bounds
            [xbounds[i], xbounds[i+1]].
    ybounds: list
        Y bounds for voxels, flattened. Voxel with index i has y bounds
            [ybounds[i], ybounds[i+1]].
    zbounds: list
        Z bounds for voxels, flattened. Voxel with index i has z bounds
            [zbounds[i], zbounds[i+1]].
    boundaries_dict: dict
        Keys: voxel idx
        Values: [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    IDs: np.ndarray
        Voxel IDs (equal to voxel indices).
    boundaries: list
        Voxel boundaries, [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    origins: list
        Voxel origin coordinates, [x,y,z].
    centers: np.ndarray
        Voxel center coordinates, [x,y,z].
    neighbors_dict: dict
        Keys: voxel idx
        Values: list of voxel indices that are neighbors to the key voxel.
    separation_groupings_dict: dict
        Keys: separation distance
        Values: list of tuples of voxel indices that are separated by the key
            separation distance.

    Methods
    -----------
    """

    def __init__(
        self,
        box: Union[None, list, np.ndarray] = None,
        number_of_voxels: Union[None, int, float, list, np.ndarray] = None,
        xbounds: Union[None, list, np.ndarray] = None,
        ybounds: Union[None, list, np.ndarray] = None,
        zbounds: Union[None, list, np.ndarray] = None,
    ):

        self.box, self.number_of_voxels, self.xbounds, self.ybounds, self.zbounds = (
            canonicalize_voxel_inputs(box, number_of_voxels, xbounds, ybounds, zbounds)
        )
        self.boundaries_dict = self.create_voxel_boundaries_dictionary()
        self.IDs = np.arange(len(self.boundaries_dict)).flatten()
        self.boundaries = [self.boundaries_dict[_] for _ in self.IDs]
        self.origins = [tuple(np.array(_)[:, 0]) for _ in self.boundaries]
        self.centers = np.array(
            [
                [
                    np.mean(boundary_list[0]),
                    np.mean(boundary_list[1]),
                    np.mean(boundary_list[2]),
                ]
                for boundary_list in self.boundaries
            ]
        )
        self.neighbors_dict = self.find_voxel_neighbors()
        self.separation_groupings_dict = self.get_separation_distance_groupings_dict()
        self.voxel_idxs_by_distance_groupings = self.get_voxel_idxs_by_distance_groupings()

        return

    def create_voxel_boundaries_dictionary(self):
        return create_voxel_boundaries_dictionary(
            number_of_voxels=self.number_of_voxels,
            xbounds=self.xbounds,
            ybounds=self.ybounds,
            zbounds=self.zbounds,
        )

    def find_voxel_neighbors(self):
        voxel_bounds = np.array(
            [[[b[d][0], b[d][1]] for d in range(3)] for b in self.boundaries]
        )
        return find_voxel_neighbors_with_shaft_overlap_method(voxel_bounds)

    def assign_voxel_ID_to_given_COG(self, COG, voxel_idx_to_ID=None):
        return assign_voxel_idx_to_given_COG(
            COG,
            self.xbounds,
            self.ybounds,
            self.zbounds,
            self.box,
        )

    def get_separation_distance_groupings_dict(self):
        distances, groupings = get_voxel_distance_groupings(self)
        return {
            d: list(zip(groupings[didx][0], groupings[didx][1]))
            for didx, d in enumerate(distances)
        }
    
    def get_voxel_idxs_by_distance_groupings(self):
        return [
            tuple([np.array(_) for _ in zip(*v)])
            for k, v in sorted(self.separation_groupings_dict.items())
        ]


def canonicalize_voxel_inputs(
    box: Union[None, list, np.ndarray] = None,
    number_of_voxels: Union[None, int, float, list, np.ndarray] = None,
    xbounds: Union[None, list, np.ndarray] = None,
    ybounds: Union[None, list, np.ndarray] = None,
    zbounds: Union[None, list, np.ndarray] = None,
) -> tuple:
    """Canonicalizes inputs for voxelization.

    Parameters
    ----------
    box: None | list | np.ndarray
        Box dimensions.
    number_of_voxels: None | int | float | list | np.ndarray
        Number of voxels in each dimension.
    xbounds: None | list | np.ndarray
        X bounds for voxels.
    ybounds: None | list | np.ndarray
        Y bounds for voxels.
    zbounds: None | list | np.ndarray
        Z bounds for voxels.

    Returns
    -------
    box: list
        Box dimensions, [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ].
    number of voxels: list
        Number of voxels in each dimension (length of three).
    xbounds: list
        X bounds for voxels, flattened. Voxel with index i has x bounds
            [xbounds[i], xbounds[i+1]].
    ybounds: list
        Y bounds for voxels, flattened. Voxel with index i has y bounds
            [ybounds[i], ybounds[i+1]].
    zbounds: list
        Z bounds for voxels, flattened. Voxel with index i has z bounds
            [zbounds[i], zbounds[i+1]].
    """

    # Canonicalize box
    if box is None:
        if None in [xbounds, ybounds, zbounds]:
            raise ValueError(
                f"""
                Either box dimenions or ALL voxel bounds must be specified.
                    Given box: {box}
                    Given xbounds: {xbounds}
                    Given ybounds: {ybounds}
                    Given zbounds: {zbounds}
                """
            )
        box = [None] * 6
    if type(box) is np.ndarray:
        box = box.flatten().tolist()
    if len(box) == 3:
        box = utility.flatten_nested_list(box)
    if len(box) not in [0, 6]:
        raise ValueError(
            f"""
            Box dimensions are either in an incorrect format, or specified dimensionality is not currently supported.             
                Given box: {box}
                Expected [x_min,x_max,y_min,y_max,z_min,z_max] or [[x_min,x_max], [y_min,y_max], [z_min,z_max]]
            """
        )
    box = (
        np.array([float(_) if _ is not None else None for _ in box]).flatten().tolist()
    )

    # Canonicalize number_of_voxels
    if number_of_voxels is None:
        if None in [xbounds, ybounds, zbounds]:
            raise ValueError(
                f"""
                Either number_of_voxels or ALL voxel bounds must be specified.
                    Given number_of_voxels: {number_of_voxels}
                    Given xbounds: {xbounds}
                    Given ybounds: {ybounds}
                    Given zbounds: {zbounds}
                """
            )
        number_of_voxels = [len(xbounds) - 1, len(ybounds) - 1, len(zbounds) - 1]
    if type(number_of_voxels) is np.ndarray:
        number_of_voxels = number_of_voxels.flatten().tolist()
    if type(number_of_voxels) in [int, float]:
        number_of_voxels = [int(number_of_voxels)]
    if len(number_of_voxels) == 1:
        number_of_voxels = [number_of_voxels[0]] * 3

    # Canonicalize xbounds, ybounds, and zbounds
    if xbounds is None:
        xbounds = np.linspace(box[0], box[1], number_of_voxels[0] + 1).tolist()
    if ybounds is None:
        ybounds = np.linspace(box[2], box[3], number_of_voxels[1] + 1).tolist()
    if zbounds is None:
        zbounds = np.linspace(box[4], box[5], number_of_voxels[2] + 1).tolist()
    if type(xbounds) is np.ndarray:
        xbounds = xbounds.flatten().tolist()
    if type(ybounds) is np.ndarray:
        ybounds = ybounds.flatten().tolist()
    if type(zbounds) is np.ndarray:
        zbounds = zbounds.flatten().tolist()

    # If a box dimension wasn't set, pull it from the voxel bounds
    box = np.array(box).flatten()
    box = np.where(
        box == None,
        [xbounds[0], xbounds[-1], ybounds[0], ybounds[-1], zbounds[0], zbounds[-1]],
        box,
    )
    box = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]]]

    # Consistency check - box and x/y/z bounds
    error_box, error_bounds, error_dimension = [], [], []
    for d, dbounds in enumerate([xbounds, ybounds, zbounds]):
        if box[d] != [dbounds[0], dbounds[-1]]:
            error_box.append(box[d])
            error_bounds.append([dbounds[0], dbounds[-1]])
            error_dimension.append(["x", "y", "z"][d])
    if len(error_box):
        raise ValueError(
            f"""
            Inconsistent boundaries.
                Given box bounds for {error_dimension} dimension: {error_box}
                Specified voxel bounds for {error_dimension} dimension: {error_bounds}
            """
        )

    # Consistency check - number of voxels and bounds
    error_voxels, error_bounds, error_dimension = [], [], []
    for d, dbounds in enumerate([xbounds, ybounds, zbounds]):
        if number_of_voxels[d] != len(dbounds) - 1:
            error_voxels.append(number_of_voxels[d])
            error_bounds.append(len(dbounds) - 1)
            error_dimension.append(["x", "y", "z"][d])
    if len(error_voxels):
        raise ValueError(
            f"""
            Inconsistent number of voxels.
                Given number of voxels for {error_dimension} dimension: {error_voxels}
                Specified voxel bounds for {error_dimension} dimension: {error_bounds}
            """
        )

    return box, number_of_voxels, xbounds, ybounds, zbounds


def create_voxel_boundaries_dictionary(
    number_of_voxels: list,
    xbounds: list,
    ybounds: list,
    zbounds: list,
) -> tuple:
    """Creates a dictionary of voxel boundaries.

    Parameters
    ----------
    number of voxels: list
        Number of voxels in each dimension (length of three).
    xbounds: list
        X bounds for voxels, flattened. Voxel with index i has x bounds
            [xbounds[i], xbounds[i+1]].
    ybounds: list
        Y bounds for voxels, flattened. Voxel with index i has y bounds
            [ybounds[i], ybounds[i+1]].
    zbounds: list
        Z bounds for voxels, flattened. Voxel with index i has z bounds
            [zbounds[i], zbounds[i+1]].

    Returns
    -------
    voxel_boundaries_dict: dict
        Keys: voxel idx
        Values: [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ]
    """

    voxel_boundaries_dict, count = {}, 0
    for i in range(len(xbounds) - 1):
        for j in range(len(ybounds) - 1):
            for k in range(len(zbounds) - 1):
                voxel_boundaries_dict[count] = [
                    [xbounds[i], xbounds[i + 1]],
                    [ybounds[j], ybounds[j + 1]],
                    [zbounds[k], zbounds[k + 1]],
                ]
                count += 1
    if len(voxel_boundaries_dict) != np.prod(number_of_voxels):
        raise ValueError(
            "Entries in voxel_boundaries_dict does not match the number of voxels defined by the boundaries."
        )
    return voxel_boundaries_dict


def calculate_shaft_overlap_idxs_1D(
    bounds_of_primary_voxel: np.ndarray, bounds_of_comparison_voxels: np.ndarray
) -> np.ndarray:
    """Finds indices of comparison voxels that overlap with the primary voxel.

    This function assumes that voxel bounds define an infinitely long
    shaft. It compares the shaft boundaries of the primary voxel to all
    comparison voxels, returning the indices of the comparison voxels
    whose shafts overlap with the shaaft of the primary voxel.

    non overlapping    overlapping    overlapping
     |  |                |  |            |  |
     |  |                |  |            |  |
    ---------------    -----------    -----------
          |  |              |  |           |  |
          |  |              |  |           |  |

    Parameters
    ----------
    bounds_of_primary_voxel: np.ndarray
        1D array of length 2, [min, max].
    bounds_of_comparison_voxels: np.ndarray
        2D array of shape (N, 2), where N is the number of comparison voxels.

    Returns
    -------
    np.ndarray
        1D array of indices of comparison voxels that overlap with the primary voxel.
    """

    min_a = bounds_of_primary_voxel[0]
    max_a = bounds_of_primary_voxel[1]
    min_b = bounds_of_comparison_voxels[:, 0]
    max_b = bounds_of_comparison_voxels[:, 1]
    return np.argwhere(
        np.logical_and(np.logical_not(min_a > max_b), np.logical_not(max_a < min_b))
    ).flatten()


def find_voxel_neighbors_with_shaft_overlap_method(voxel_bounds: np.ndarray) -> tuple:
    """Finds neighbors of each voxel using the shaft overlap method.

    Parameters
    ----------
    voxel_bounds: np.ndarray
        2D array of shape (N, 3, 2), where N is the number of voxels.
        Entry [i, j, 0] is the j-th dimension minimum of the i-th voxel.
        Entry [i, j, 1] is the j-th dimension maximum of the i-th voxel.

    Returns
    -------
    voxel_neighbors: dict
        Keys: voxel index
        Values: list of voxel indices that are neighbors to the key voxel.
    """

    if voxel_bounds.shape[0] == 1:
        return {0: []}

    box_minima = [
        np.min(voxel_bounds[:, dimension, 0], axis=None) for dimension in range(3)
    ]
    box_maxima = [
        np.max(voxel_bounds[:, dimension, 1], axis=None) for dimension in range(3)
    ]
    voxel_idxs_on_minima = [
        [
            voxel_idx
            for voxel_idx, voxel_bounds in enumerate(voxel_bounds)
            if voxel_bounds[dimension, 0] == box_minima[dimension]
        ]
        for dimension in range(3)
    ]
    voxel_idxs_on_maxima = [
        [
            voxel_idx
            for voxel_idx, voxel_bounds in enumerate(voxel_bounds)
            if voxel_bounds[dimension, 1] == box_maxima[dimension]
        ]
        for dimension in range(3)
    ]
    voxel_neighbors = {}
    for primary_voxel_idx, primary_voxel_bounds in enumerate(voxel_bounds):
        neighbor_idxs = np.delete(np.arange(voxel_bounds.shape[0]), primary_voxel_idx)
        for dimension in range(3):
            bounds_of_comparison_voxels = voxel_bounds[
                [
                    vidx
                    for vidx in np.arange(voxel_bounds.shape[0])
                    if vidx in neighbor_idxs
                ]
            ][:, dimension]
            if primary_voxel_bounds[dimension, 0] == box_minima[dimension]:
                bounds_of_comparison_voxels = np.concatenate(
                    (
                        bounds_of_comparison_voxels,
                        np.array(
                            [
                                [-np.inf, box_minima[dimension]]
                                for vidx in voxel_idxs_on_maxima[dimension]
                                if vidx in neighbor_idxs
                            ]
                        ),
                    )
                )
                neighbor_idxs = np.concatenate(
                    (
                        neighbor_idxs,
                        np.array(
                            [
                                vidx
                                for vidx in voxel_idxs_on_maxima[dimension]
                                if vidx in neighbor_idxs
                            ]
                        ).flatten(),
                    )
                )
            if primary_voxel_bounds[dimension, 1] == box_maxima[dimension]:
                bounds_of_comparison_voxels = np.concatenate(
                    (
                        bounds_of_comparison_voxels,
                        np.array(
                            [
                                [box_maxima[dimension], np.inf]
                                for vidx in voxel_idxs_on_minima[dimension]
                                if vidx in neighbor_idxs
                            ]
                        ),
                    )
                )
                neighbor_idxs = np.concatenate(
                    (
                        neighbor_idxs,
                        np.array(
                            [
                                vidx
                                for vidx in voxel_idxs_on_minima[dimension]
                                if vidx in neighbor_idxs
                            ]
                        ).flatten(),
                    )
                )
            shaft_overlap_idxs = calculate_shaft_overlap_idxs_1D(
                primary_voxel_bounds[dimension],
                bounds_of_comparison_voxels,
            )
            neighbor_idxs = neighbor_idxs[shaft_overlap_idxs]
            neighbor_idxs = np.sort(np.unique(neighbor_idxs))
        voxel_neighbors[primary_voxel_idx] = neighbor_idxs
    return voxel_neighbors


def assign_voxel_idx_to_given_COG(
    COG: Union[list, np.ndarray],
    xbounds: list,
    ybounds: list,
    zbounds: list,
    box: list,
) -> int:
    """Assigns voxel index to given Center of Geometry.

    For a given COG, the corresponding voxel index in which the COG is
    located is returned.

    NOTE this function saves time by assuming that the voxel indices
    are assigned for x, then y, then z.

    Parameters
    ----------
    COG: list | np.ndarray
        Center of geometry, [x,y,z].
    xbounds: list
        X bounds for voxels, flattened. Voxel with index i has x bounds
            [xbounds[i], xbounds[i+1]].
    ybounds: list
        Y bounds for voxels, flattened. Voxel with index i has y bounds
            [ybounds[i], ybounds[i+1]].
    zbounds: list
        Z bounds for voxels, flattened. Voxel with index i has z bounds
            [zbounds[i], zbounds[i+1]].
    box: list
        Box dimensions, [ [xmin,xmax], [ymin,ymax], [zmin,zmax] ].

    Returns
    -------
    int
        Voxel idx.
    """

    # COG s/b flattened array
    if type(COG) is list:
        COG = np.array(COG)
    COG = COG.flatten()

    # wrap COG into box
    for i in range(3):
        while COG[i] < box[i][0]:
            COG[i] += box[i][1] - box[i][0]
        while COG[i] > box[i][1]:
            COG[i] -= box[i][1] - box[i][0]

    # find x/y/z indices, then calculate voxel index assuming voxel
    # indices are assigned for x, then y, then z
    xidx = np.argwhere(COG[0] >= xbounds[:-1])[-1][0]
    yidx = np.argwhere(COG[1] >= ybounds[:-1])[-1][0]
    zidx = np.argwhere(COG[2] >= zbounds[:-1])[-1][0]
    vidx = (
        xidx * (len(ybounds) - 1) * (len(zbounds) - 1)
        + yidx * (len(zbounds) - 1)
        + zidx
    )
    return vidx


def get_voxel_distance_groupings(voxels: Voxels) -> tuple:
    """Returns voxel distance groupings.

    Given a Voxels instance, this function will find the set of
    separation distances between all voxels, and will assign each
    voxel-voxel pair to the corresponding distance group.

    Parameters
    ----------
    voxels: Voxels
        Voxel object.

    Returns
    -------
    neighbor_distances: list
        List of distances between voxels.
    neighbor_idxs: list
        List of lists of voxel indices that are neighbors to the key voxel.
    """
    distance = utility.find_dist_same(voxels.centers, voxels.box)
    neighbor_distances = sorted(list(set(distance.flatten())))[1:]
    return neighbor_distances, [np.where(distance == d) for d in neighbor_distances]

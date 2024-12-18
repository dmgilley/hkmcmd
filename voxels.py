#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu 


# Imports
import numpy as np
from typing import Union


class Voxels():
    """Class for voxel information.

    Attributes
    -----------
    box: list of lists of floats
        Box dimensions.
    num_voxels: list of integers
        Number of voxels in each dimension.
    xbounds: list of floats
        X bounds for voxels.
    ybounds: list of floats
        Y bounds for voxels.
    zbounds: list of floats
        Z bounds for voxels.

    Methods
    -----------
    """

    def __init__(
            self,
            box: Union[list, None] = None,
            number_of_voxels: Union[list, None] = None,
            xbounds: Union[list, None] = None,
            ybounds: Union[list, None] = None,
            zbounds: Union[list, None] = None,
            ):

        box,number_of_voxels,xbounds,ybounds,zbounds = parse_voxel_inputs(box,number_of_voxels,xbounds,ybounds,zbounds)
        self.box = box
        self.number_of_voxels = number_of_voxels
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.zbounds = zbounds
        self.create_voxel_boundaries_dictionary()
        self.voxel_IDs = np.array(sorted(list(self.voxel_boundaries_dict.keys())))
        self.voxel_boundaries = [self.voxel_boundaries_dict[_] for _ in self.voxel_IDs]
        self.voxel_origins = [tuple(np.array(_)[:,0]) for _ in self.voxel_boundaries]

        return
    
    def create_voxel_boundaries_dictionary(self, rewrite=False):
        if not hasattr(self,'voxel_boundaries_dict') or rewrite is True:
            setattr(self,'voxel_boundaries_dict',create_voxel_boundaries_dictionary(
                box=self.box,
                number_of_voxels=self.number_of_voxels,
                xbounds=self.xbounds,
                ybounds=self.ybounds,
                zbounds=self.zbounds,
                ))
        return self.voxel_boundaries_dict
    
    def find_voxel_neighbors(self, include_touching: bool = True, rewrite: bool = False):
        if not hasattr(self, 'voxel_neigbors_dict') or rewrite is True:
            if not hasattr(self,'voxel_boundaries_dict'):
                self.create_voxel_boundaries_dictionary()
            if not hasattr(self,'voxel_boundaries'):
                self.voxel_boundaries = [self.voxel_boundaries_dict[_] for _ in self.voxel_IDs]
            setattr(self, 'voxel_neighbors_dict', find_voxel_neighbors_with_shaft_overlap_method(
                voxel_bounds=np.array(self.voxel_boundaries),
                voxel_IDs=self.voxel_IDs,
                include_touching=include_touching))
        return
    
    def assign_voxel_ID_to_given_COG(self, COG, voxel_idx_to_ID=None):
        return assign_voxel_ID_to_given_COG(
            COG,
            self.xbounds,
            self.ybounds,
            self.zbounds,
            self.box,
            voxel_idx_to_ID=voxel_idx_to_ID)


def parse_voxel_inputs(            
        box: Union[list, None] = None,
        number_of_voxels: Union[list, None] = None,
        xbounds: Union[list, None] = None,
        ybounds: Union[list, None] = None,
        zbounds: Union[list, None] = None,
        ):
    
    # Adjust box argument
    if box is None:
        if None in [xbounds,ybounds,zbounds]:
            raise ValueError(
                """
                Either box dimenions or ALL voxel bounds must be specified.
                Given box: {}
                Given xbounds: {}
                Given ybounds: {}
                Given zbounds: {}
                """.format(box,xbounds,ybounds,zbounds))
        box = [None]*6
    if len(box) == 3:
        box = np.array(box).flatten()
    if len(box) not in [0,6]:
        raise ValueError(
            """
            Box dimensions are either in an incorrect format, or specified dimensions are not currently supported.             
            Given box: {}
            Expected [x_min,x_max,y_min,y_max,z_min,z_max] or [[x_min,x_max], [y_min,y_max], [z_min,z_max]]
            """.format(box))
    # convert box values to floats
    box = np.array([float(b) if b is not None else None for b in box]).flatten()

    # Adjust number_of_voxels argument
    if type(number_of_voxels) is np.ndarray:
        number_of_voxels = number_of_voxels.flatten().tolist()
    if number_of_voxels is None:
        if None in [xbounds,ybounds,zbounds]:
            raise ValueError(
                """
                Either number_of_voxels or ALL voxel bounds must be specified.
                Given number_of_voxels: {}
                Given xbounds: {}
                Given ybounds: {}
                Given zbounds: {}
                """.format(number_of_voxels,xbounds,ybounds,zbounds))
        number_of_voxels = [len(xbounds)-1,len(ybounds)-1,len(zbounds)-1]
    if len(number_of_voxels) == 1:
        number_of_voxels = [number_of_voxels[0]]*3

    # Adjust xbounds, ybounds, and zbounds arguments
    if xbounds is None:
        xbounds = np.linspace( box[0], box[1], number_of_voxels[0]+1 ).tolist()
    if ybounds is None:
        ybounds = np.linspace( box[2], box[3], number_of_voxels[1]+1 ).tolist()
    if zbounds is None:
        zbounds = np.linspace( box[4], box[5], number_of_voxels[2]+1 ).tolist()

    # If a box dimension wasn't set, pull it from the voxel bounds
    box = np.where(box == None, [xbounds[0],xbounds[-1],ybounds[0],ybounds[-1],zbounds[0],zbounds[-1]], box)
    box = [[box[0],box[1]], [box[2],box[3]], [box[4],box[5]]]

    # Consistency check - box and x/y/z bounds
    error_box = []
    error_bounds = []
    error_dimension = []
    for d,dbounds in enumerate([xbounds,ybounds,zbounds]):
        if box[d] != [dbounds[0],dbounds[-1]]:
            error_box.append(box[d])
            error_bounds.append([dbounds[0],dbounds[-1]])
            error_dimension.append(['x','y','z'][d])
    if len(error_box):
        raise ValueError(
            """
            Inconsistent boundaries.
            Given box bounds for {} dimension: {}
            Specified voxel bounds for {} dimension: {}
            """.format(error_dimension,error_box,error_dimension,error_bounds))

    # Consistency check - number of voxels and bounds
    error_voxels = []
    error_bounds = []
    error_dimension = []
    for d,dbounds in enumerate([xbounds,ybounds,zbounds]):
        if number_of_voxels[d] != len(dbounds)-1:
            error_voxels.append(number_of_voxels[d])
            error_bounds.append(len(dbounds)-1)
            error_dimension.append(['x','y','z'][d])
    if len(error_voxels):
        raise ValueError(
            """
            Inconsistent number of voxels.
            Given number of voxels for {} dimension: {}
            Specified voxel bounds for {} dimension: {}
            """.format(error_dimension,error_voxels,error_dimension,error_bounds))

    return box,number_of_voxels,xbounds,ybounds,zbounds


def create_voxel_boundaries_dictionary(            
        box: Union[list, None] = None,
        number_of_voxels: Union[list, None] = None,
        xbounds: Union[list, None] = None,
        ybounds: Union[list, None] = None,
        zbounds: Union[list, None] = None,
        ):
    box,number_of_voxels,xbounds,ybounds,zbounds = parse_voxel_inputs(box,number_of_voxels,xbounds,ybounds,zbounds)
    voxel_boundaries_dict, count = {}, 0
    for i in range(len(xbounds)-1):
        for j in range(len(ybounds)-1):
            for k in range(len(zbounds)-1):
                voxel_boundaries_dict[count] = [ [xbounds[i],xbounds[i+1]], [ybounds[j],ybounds[j+1]], [zbounds[k],zbounds[k+1]], ]
                count += 1
    if len(voxel_boundaries_dict) != np.prod(number_of_voxels):
        raise ValueError('Entries in voxel_boundaries_dict does not match the number of voxels defined by the boundaries.')
    return voxel_boundaries_dict


def get_box_from_voxel_boundaries_dict(voxel_boundaries_dict):
    return [[
        np.min([voxel_boundaries_dict[_][didx][0] for _ in voxel_boundaries_dict.keys()]),
        np.max([voxel_boundaries_dict[_][didx][1] for _ in voxel_boundaries_dict.keys()])] for didx in range(3) ]


def calculate_shaft_overlap_idxs_1D(
        bounds_of_primary_voxel: np.ndarray,
        bounds_of_comparison_voxels: np.ndarray,
        inclusive: bool = True) -> tuple:
    
    min_a = bounds_of_primary_voxel[0]
    max_a = bounds_of_primary_voxel[1]
    min_b = bounds_of_comparison_voxels[:,0]
    max_b = bounds_of_comparison_voxels[:,1]

    if inclusive is True:
        overlap_idxs = np.argwhere(np.logical_and( np.logical_not(min_a > max_b), np.logical_not(max_a < min_b) )).flatten()
    else:
        overlap_idxs = np.argwhere(np.logical_and( np.logical_not(min_a >= max_b), np.logical_not(max_a <= min_b) )).flatten()

    return overlap_idxs


def find_voxel_neighbors_with_shaft_overlap_method(
        voxel_bounds: np.ndarray, # "number of voxels" x "number of dimensions" x 2
        voxel_IDs: Union[np.ndarray, None] = None, # flattened array pf size "number of voxels"
        include_touching: bool = True) -> tuple:

    if voxel_IDs is None:
        voxel_IDs = np.arange(len(voxel_bounds)).flatten()
    box_minima = [np.min(voxel_bounds[:,dimension,0], axis=None) for dimension in range(3)]
    voxels_on_minima = [[
        voxel_IDs[voxel_idx] for voxel_idx,voxel_bounds in enumerate(voxel_bounds) if voxel_bounds[dimension,0] == box_minima[dimension]
        ] for dimension in range(3)]
    box_maxima = [np.max(voxel_bounds[:,dimension,1], axis=None) for dimension in range(3)]
    voxels_on_maxima = [[
        voxel_IDs[voxel_idx] for voxel_idx,voxel_bounds in enumerate(voxel_bounds) if voxel_bounds[dimension,1] == box_maxima[dimension]
        ] for dimension in range(3)]
    voxel_neighbors = {}

    for primary_voxel_idx,primary_voxel_bounds in enumerate(voxel_bounds):
        neighbor_IDs = np.delete(voxel_IDs, primary_voxel_idx)
        for dimension in range(3):
            bounds_of_comparison_voxels = voxel_bounds[[vidx for vidx,vID in enumerate(voxel_IDs) if vID in neighbor_IDs]][:,dimension]
            if primary_voxel_bounds[dimension,0] == box_minima[dimension]:
                bounds_of_comparison_voxels = np.concatenate((
                    bounds_of_comparison_voxels,
                    np.array([[-np.inf,box_minima[dimension]] for vID in voxels_on_maxima[dimension] if vID in neighbor_IDs])))
                neighbor_IDs = np.concatenate((
                    neighbor_IDs,
                    np.array([vID for vID in voxels_on_maxima[dimension] if vID in neighbor_IDs]).flatten()))
            if primary_voxel_bounds[dimension,1] == box_maxima[dimension]:
                bounds_of_comparison_voxels = np.concatenate((
                    bounds_of_comparison_voxels,
                    np.array([[box_maxima[dimension],np.inf] for vID in voxels_on_minima[dimension] if vID in neighbor_IDs])))
                neighbor_IDs = np.concatenate((
                    neighbor_IDs,
                    np.array([vID for vID in voxels_on_minima[dimension] if vID in neighbor_IDs]).flatten()))
            shaft_overlap_idxs = calculate_shaft_overlap_idxs_1D(
                primary_voxel_bounds[dimension],
                bounds_of_comparison_voxels,
                inclusive=include_touching)
            neighbor_IDs = neighbor_IDs[shaft_overlap_idxs]
            neighbor_IDs = np.sort(np.unique(neighbor_IDs))
        voxel_neighbors[voxel_IDs[primary_voxel_idx]] = neighbor_IDs
            
    return voxel_neighbors
    

def assign_voxel_ID_to_given_COG(
        COG,
        xbounds,
        ybounds,
        zbounds,
        box, 
        voxel_idx_to_ID=None):
    
    # COG s/b flattened array
    if type(COG) is list:
        COG = np.array(COG)
    COG = COG.flatten()

    # wrap COG into box
    for i in range(3):
        while COG[i] < box[i][0]:
            COG[i] += box[i][1]-box[i][0]
        while COG[i] > box[i][1]:
            COG[i] -= box[i][1]-box[i][0]

    # find x/y/z indices, then calculate voxel index assuming voxel
    # indices are assigned for x, then y, then z
    xidx = np.argwhere(COG[0] >= xbounds[:-1])[-1][0]
    yidx = np.argwhere(COG[1] >= ybounds[:-1])[-1][0]
    zidx = np.argwhere(COG[2] >= zbounds[:-1])[-1][0]
    vidx = xidx*(len(ybounds)-1)*(len(zbounds)-1) + yidx*(len(zbounds)-1) + zidx

    # if voxel_idx_to_ID is given, return the ID
    if voxel_idx_to_ID is not None:
        return voxel_idx_to_ID[vidx]
    
    # otherwise, assume idx = ID
    return vidx
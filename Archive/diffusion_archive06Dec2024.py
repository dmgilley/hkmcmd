#!/usr/bin/env python
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu

import numpy as np
from collections import defaultdict
from hybrid_mdmc.frame_generator import frame_generator
from hybrid_mdmc.classes import MoleculeList
from hybrid_mdmc.calc_voxels import calc_voxels
from functions import *
from hybrid_mdmc.parsers import *
from hybrid_mdmc.data_file_parser import *

# ToDo
#   - documentation
#   - rewrite using a voxel class
#   - parallelize; get_diffusionratematrix_from_transitionratematrix
#     performs MANY independent calculations. Parallelization would
#     provide tremendous speed-up.
#   - check that frame_generator will behave appropriately if it
#     attempts to unwrap trajectories that are provided unwrapped


class DiffusionGraph:
    """Class designed to hold diffusion information.

    This class creates objects used for finding diffusion rates
    with Dijsktra's Algorithm.

    Attributes
    ----------
    edges: dictionary
        keys - int
            node ID
        values - list of int
            node IDs to which the key node ID connects
    weights: dictionary
        keys - tuple (int,int)
            (starting nodeID, ending nodeID)
        value - float
            "weight" (rate, time, etc.) connecting the nodes

    Methods
    -------
    add_edge(from_node,to_node,weight)
    """

    def __init__(self, edges=defaultdict(list), weights={}):
        self.edges = edges
        self.weights = weights

    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight

    def dijkstra(self, initial, end):
        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()
        while current_node != end:
            visited.add(current_node)
            destinations = self.edges[current_node]
            weight_to_current_node = shortest_paths[current_node][1]
            for next_node in destinations:
                weight = (
                    self.weights[(current_node, next_node)] + weight_to_current_node
                )
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)
            next_destinations = {
                node: shortest_paths[node]
                for node in shortest_paths
                if node not in visited
            }
            if not next_destinations:
                return None, np.inf
                # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        # Work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]
        total_weight = np.sum(
            [self.weights[(path[idx], path[idx + 1])] for idx in range(len(path) - 1)]
        )
        return path, total_weight


def get_DiffusionGraph_from_matrix(matrix):
    """Calculates a DiffusionGraph object from a matrix.

    Given a square matrix containing some sort of weight between nodes,
    this function returns an instance of "DiffusionGraph." Node IDs are
    assigned as the row/column index. Weight values are the
    corresponding matrix values.
    """
    graph = DiffusionGraph(
        edges={
            idx: [col for col, _ in enumerate(matrix[idx, :]) if _]
            for idx in range(len(matrix))
        },
        weights={
            (ridx, cidx): val
            for ridx, row in enumerate(matrix)
            for cidx, val in enumerate(row)
            if val
        },
    )
    return graph


def get_voxels_byframe(
    traj_file, atoms_datafile, molecules_datafile, num_voxels, start=0, end=-1, every=1
):
    """ """
    # Create a dictionary to hold the voxel number of each molecule for
    # each frame. Entries are a list of voxels, indexed to
    # molecules_datafile.ids, and keys are the timestep.
    voxels_byframe = {}

    # Calculate an adjacency list for use in frame_generator using the
    # datefile_atoms object. This is required for unwrapping atom
    # trajectories.
    adj_list = [
        [idx for idx, _ in enumerate(atoms_datafile.mol_id) if idx != aidx and _ == mol]
        for aidx, mol in enumerate(atoms_datafile.mol_id)
    ]

    # Loop over all of the requested frames.
    for atoms_thisframe, timestep, box_thisframe in frame_generator(
        traj_file,
        start=start,
        end=end,
        every=every,
        unwrap=True,
        adj_list=adj_list,
        return_prop=False,
    ):

        # Calculate the voxels object based on the this timestep's box
        # and the requested "num_voxels."
        voxels_thisframe = calc_voxels(num_voxels, box_thisframe)

        # Create a temporary MoleculeList object to hold the molecule
        # information for this timestep. By using the unchanging
        # molecules_datafile object to assign ids, all temporary
        # molecule objects and subsequent voxel lists will be
        # consistently indexed.
        molecules_thisframe = MoleculeList(
            ids=molecules_datafile.ids,
            mol_types=molecules_datafile.mol_types,
            atom_ids=molecules_datafile.atom_ids,
        )

        # Calculate the COG and voxel for each molecule in this
        # timestep. frame_generator unwraps trajectories (if they are
        # not already provided unwrapped). The COG is calculated
        # using unwrapped atomic positions, then is wrapped back into
        # the box so that the molecule is assigned to the correct
        # voxel.
        molecules_thisframe.get_cog(atoms_thisframe, box=box_thisframe)
        molecules_thisframe.get_voxels(voxels_thisframe)

        # Add the voxel list for this timestep to the voxels_byframe
        # dictionary. Entries in the list of assigned voxels are voxel
        # IDs as created with calc_voxels; calc_voxels assigns IDs as
        # the index of voxels, counted as (x0,y0,z0,), ... (x0,y0,zf),
        # (x0,y1,z0), ... (x0,yf,zf), (x1,y0,z0), ... (xf,yf,zf).
        voxels_byframe[int(timestep)] = molecules_thisframe.voxel_idxs

    return voxels_byframe


def calc_diffusionrate(
    trjfile,
    atoms_datafile,
    box,
    masterspecies_info,
    num_voxels,
    xbounds=[],
    ybounds=[],
    zbounds=[],
    start=0,
    end=-1,
    every=1,
    lammps_stepsize=1,
    lammps_time_units_to_seconds_conversion=1e-15,
):
    """ """
    # Calculate the voxels and voxel mapping objects based on the
    # provided number of voxels and the provided box. The actual voxel
    # parameters are not used. They are only created to provide a
    # consistent len(voxel) used throughout the function, and so that
    # the molecules_datafile object can be created with the already
    # existing gen_molecules function. The voxel assignments of
    # molecules_datafile are not used anywhere in this function.
    voxels = calc_voxels(
        num_voxels, box, xbounds=xbounds, ybounds=ybounds, zbounds=zbounds
    )
    voxelsmap, voxelsx, voxelsy, voxelsz = voxels2voxelsmap(voxels)
    atomtypes2moltype = {
        tuple(sorted([_[2] for _ in v["Atoms"]])): k
        for k, v in masterspecies_info.items()
    }

    # Generate a molecule object based on the datafile information.
    molecules_datafile = gen_molecules(
        atoms_datafile, atomtypes2moltype, voxelsmap, voxelsx, voxelsy, voxelsz
    )

    # Calculate the assigned voxel for each molecule in each frame. The
    # list of assigned voxels in each timestep consists of voxel IDs
    # (which, as provided by calc_voxels, are equivalent to voxel
    # indices).
    voxels_byframe = get_voxels_byframe(
        trjfile,
        atoms_datafile,
        molecules_datafile,
        num_voxels,
        start=start,
        end=end,
        every=every,
    )

    # For each species, create a "voxel transition" array. The ith,jth
    # entry is the total number of transitions from voxel i to voxel j
    # for that species over the entire trajectory. Arrays are held in a
    # dictionary.
    voxel_transitions = {
        _: np.zeros((len(voxels), len(voxels))) for _ in masterspecies_info.keys()
    }
    timesteps = sorted(voxels_byframe.keys())
    for midx, type_ in enumerate(molecules_datafile.mol_types):
        voxel_list = np.array([voxels_byframe[time][midx] for time in timesteps])
        voxel_list_shifted = np.roll(voxel_list, -1)
        transitions = np.column_stack((voxel_list, voxel_list_shifted))
        to_from, count = np.unique(transitions[:-1, :], axis=0, return_counts=True)
        for idx, tf in enumerate((to_from)):
            voxel_transitions[type_][tf[0] - 1, tf[1] - 1] += count[idx]

    # Calculate a DiffusionGraph object for the diffusion rates, then
    # create a DiffusionGraph object for the diffusion times. Creating
    # the rates object first will make a smaller object by not
    # including the transitions that have values of 0.
    time_in_lammps_units = (timesteps[-1] - timesteps[0]) * lammps_stepsize
    time_in_seconds = time_in_lammps_units * lammps_time_units_to_seconds_conversion
    diffusion_rate = {
        k: get_DiffusionGraph_from_matrix(v / (time_in_seconds))
        for k, v in voxel_transitions.items()
    }
    diffusion_time = {
        k: DiffusionGraph(
            edges=v.edges, weights={kk: 1 / vv for kk, vv in v.weights.items()}
        )
        for k, v in diffusion_rate.items()
    }

    # Perform Dijkstra's on the diffusion time. This must be performed
    # on the diffusion time and not the diffusion rate, because
    # Dijkstra's finds the shortest path. Afterwards, the diffusion
    # time is converted to diffusion rate.
    diffusion_rate = {}
    for k, v in diffusion_time.items():
        diffusion_rate[k] = np.zeros((len(voxels), len(voxels)))
        for row in range(len(voxels)):
            for col in range(len(voxels)):
                path, totaltime = v.dijkstra(row, col)
                diffusion_rate[k][row, col] = 1 / totaltime

    return diffusion_rate

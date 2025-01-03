#!/usr/bin/env python
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu


import numpy as np
from typing import Union
from hybrid_mdmc.frame_generator import frame_generator
from hybrid_mdmc.classes import MoleculeList
from hybrid_mdmc.mol_classes import AtomList
from hybrid_mdmc.voxels import Voxels


class Diffusion():

    def __init__(self,
        name: str,
        filename_trajectory: str,
        atoms_datafile: AtomList,
        molecules_datafile: MoleculeList,
        voxels: Voxels,
        time_conversion: float = 1.0
        ):
        
        self.name = name
        self.filename_trajectory = filename_trajectory
        self.atoms_datafile = atoms_datafile
        self.molecules_datafile = molecules_datafile
        self.number_of_voxels = voxels.number_of_voxels
        self.voxels = voxels
        self.time_conversion = time_conversion # desired_units = current_units * time_conversion

        return
    
    def parse_trajectory_file(self, start=0, end=-1, every=1, time_conversion=None):
        if time_conversion is not None:
            self.time_conversion = time_conversion
        voxels_by_frame_array, total_time = assemble_voxels_by_frame_array(
            self.filename_trajectory,
            self.atoms_datafile,
            self.molecules_datafile,
            self.number_of_voxels,
            start=start, end=end, every=every
        )
        setattr(self, 'voxels_by_frame_array', voxels_by_frame_array)
        setattr(self, 'total_time', total_time*self.time_conversion)
        return
    
    def calculate_direct_voxel_transition_rates(self):
        if not hasattr(self, 'voxels_by_frame_array'):
            self.parse_trajectory_file()
        direct_voxel_transition_rates = calculate_direct_voxel_transition_rates(
            np.prod(self.number_of_voxels),
            self.voxels_by_frame_array,
            self.total_time,
            self.molecules_datafile
        )
        setattr(self, 'direct_voxel_transition_rates', direct_voxel_transition_rates)
        return

    def perform_random_walks(self, starting_position_idxs='all', number_of_steps=None, species='all'):
        if not hasattr(self, 'direct_voxel_transition_rates'):
            self.calculate_direct_voxel_transition_rates()
        if starting_position_idxs == 'all':
            starting_position_idxs = np.arange(np.prod(self.number_of_voxels)).flatten()
        if number_of_steps is None:
            number_of_steps = 4*np.prod(self.number_of_voxels)
        if species == 'all':
            species = self.molecules_datafile.mol_types
        elif not isinstance(species, list):
            species = [species]
        self.voxels.find_voxel_neighbors()
        noneighbor_mask = np.array([[j_idx in self.voxels.voxel_neighbors_dict[i_idx] for j_idx in range(np.prod(self.number_of_voxels))] for i_idx in range(np.prod(self.number_of_voxels))])
        walkers_positions, walkers_times = {}, {}
        for sp in species:
            transfer_rates = np.where(noneighbor_mask, self.direct_voxel_transition_rates[sp], 0)
            empty_rows = np.argwhere(np.sum(transfer_rates,axis=1) == 0).flatten()
            transfer_rates[empty_rows,empty_rows] = np.inf # if poition is a trap, set rate to self as np.inf
            walkers_positions[sp], walkers_times[sp] = perform_random_walks(
                transfer_rates,
                starting_position_idxs,
                number_of_steps)
        setattr(self, 'walkers_position', walkers_positions)
        setattr(self, 'walkers_time', walkers_times)
        return

    def calculate_average_first_time_between_positions(self, species='all'):
        if not hasattr(self, 'walkers_position'):
            self.perform_random_walks(species)
        if species == 'all':
            species = self.molecules_datafile.mol_types
        elif not isinstance(species, list):
            species = [species]
        average_first_time_between_positions = {}
        for sp in species:
            average_first_time_between_positions[sp] = calculate_average_first_time_between_positions(
                self.walkers_position[sp],
                self.walkers_time[sp],
                np.prod(self.number_of_voxels)
            )
        setattr(self, 'average_first_time_between_positions', average_first_time_between_positions)
        return

    def calculate_diffusion_rates(self, species='all'):
        if not hasattr(self, 'average_first_time_between_positions'):
            self.calculate_average_first_time_between_positions(species)
        if species == 'all':
            species = self.molecules_datafile.mol_types
        elif not isinstance(species, list):
            species = [species]
        diffusion_rates = {}
        for sp in species:
            diffusion_rates[sp] = 1/(self.average_first_time_between_positions[sp])
        setattr(self, 'diffusion_rates', diffusion_rates)
        return
    

def calculate_molecule_COGs(molecule_list,atom_list,box=[]):
    COGs = np.array([
        np.mean(np.array([
            [atom_list.x[idx],atom_list.y[idx],atom_list.z[idx]]
            for idx,ID in enumerate(atom_list.ids) if ID in matoms]),axis=0)
            for matoms in molecule_list.atom_ids
    ])
    if len(box) != 0:
        for d in range(3):
            while not np.all(COGs[:,d] >= box[d][0]):
                COGs[:,d] = np.where(COGs[:,d] >= box[d][0], COGs[:,d], COGs[:,d] + (box[d][1] - box[d][0]))
            while not np.all(COGs[:,d] <= box[d][1]):
                COGs[:,d] = np.where(COGs[:,d] <= box[d][1], COGs[:,d], COGs[:,d] - (box[d][1] - box[d][0]))
    return COGs


def assign_voxel_IDs_to_COGs(
        COGs: list,
        voxels: Voxels):
    voxel_origin_to_ID = {origin:voxels.voxel_IDs[idx] for idx,origin in enumerate(voxels.voxel_origins)}
    x_voxel_minima = np.array(voxels.voxel_boundaries)[:,0,0]
    y_voxel_minima = np.array(voxels.voxel_boundaries)[:,1,0]
    z_voxel_minima = np.array(voxels.voxel_boundaries)[:,2,0]
    COG_nearest_minima = [tuple([
        x_voxel_minima[np.argwhere(COG[0] >= x_voxel_minima)[-1][0]],
        y_voxel_minima[np.argwhere(COG[1] >= y_voxel_minima)[-1][0]],
        z_voxel_minima[np.argwhere(COG[2] >= z_voxel_minima)[-1][0]],]
    ) for COG in COGs]
    return [voxel_origin_to_ID[origin] for origin in COG_nearest_minima]


def assemble_voxels_by_frame_array(
        filename_trajectory,
        atoms_datafile,
        molecules_datafile,
        number_of_voxels,
        start=0, end=-1, every=1
        ):
    voxels_by_frame_dict = {}
    all_timesteps = []
    adj_list = [
        [idx for idx, _ in enumerate(
            atoms_datafile.mol_id) if idx != aidx and _ == mol]
        for aidx, mol in enumerate(atoms_datafile.mol_id)
    ]
    for atoms_thisframe, timestep, box_thisframe in frame_generator(
            filename_trajectory,
            start=start, end=end, every=every, unwrap=True,
            adj_list=adj_list,
            return_prop=False):
        all_timesteps.append(int(timestep))
        voxels_thisframe = Voxels(box_thisframe, number_of_voxels)
        molecules_thisframe = MoleculeList(
            ids=molecules_datafile.ids,
            mol_types=molecules_datafile.mol_types,
            atom_ids=molecules_datafile.atom_ids
        )
        molecules_thisframe_COGs = calculate_molecule_COGs(molecules_thisframe, atoms_thisframe, box=box_thisframe)
        molecules_thisframe_voxel_IDs = assign_voxel_IDs_to_COGs(molecules_thisframe_COGs, voxels_thisframe)
        voxels_by_frame_dict[int(timestep)] = molecules_thisframe_voxel_IDs
    voxels_by_frame_array = np.array([value for key,value in sorted(voxels_by_frame_dict.items())])
    return voxels_by_frame_array, all_timesteps[-1]-all_timesteps[0]


def calculate_direct_voxel_transition_rates(total_number_of_voxels, voxels_by_frame, total_time, molecules_datafile):
    voxel_transition_counts = {
        _: np.zeros((total_number_of_voxels, total_number_of_voxels))
        for _ in set(molecules_datafile.mol_types)
    }
    for midx, type_ in enumerate(molecules_datafile.mol_types):
        voxel_list = voxels_by_frame[:,midx].flatten()
        voxel_list_shifted = np.roll(voxel_list, -1)
        transitions = np.column_stack((voxel_list, voxel_list_shifted))
        to_from, count = np.unique(transitions[:-1, :], axis=0, return_counts=True)
        for idx, tf in enumerate((to_from)):
            voxel_transition_counts[type_][tf[0]-1, tf[1]-1] += count[idx]
    direct_voxel_transition_rates = {mol_type: vt_counts/total_time for mol_type,vt_counts in voxel_transition_counts.items()}
    return direct_voxel_transition_rates


def perform_random_walks(transfer_rates, starting_position_idxs, number_of_steps):
    rate_sums = np.sum(transfer_rates, axis=1)
    transfer_rates = np.cumsum(transfer_rates, axis=1)
    walkers_position = np.zeros((number_of_steps+1, len(transfer_rates)), dtype=int)
    walkers_position[0] = starting_position_idxs.flatten()
    walkers_time = np.zeros((number_of_steps+1, len(transfer_rates)), dtype=np.float64)
    for step in range(number_of_steps):
        u1 = np.random.rand(1, starting_position_idxs.flatten().shape[0])*np.array([rate_sums[idx] for idx in walkers_position[step]]).flatten()
        walkers_position[step+1] = [np.argwhere(transfer_rates[walkers_position[step,idx]] >= u)[0][0] for idx,u in enumerate(u1[0])]
        u2 = 1 - np.random.rand(1, starting_position_idxs.flatten().shape[0]) # exclude 0
        u2 = u2.flatten()[0]
        dt = -np.log(u2)/rate_sums[walkers_position[step]]
        walkers_time[step+1] = walkers_time[step] + dt
    return walkers_position, walkers_time


def calculate_average_first_time_between_positions(walkers_positions, walkers_times, number_of_positions):
    transition_counts = np.zeros((number_of_positions,number_of_positions))
    transition_times = np.zeros((number_of_positions,number_of_positions))
    for colidx in range(walkers_positions.shape[1]):
        zipped_positiontime = list(zip(walkers_positions[:,colidx], walkers_times[:,colidx]))
        for i_idx,i_value in enumerate(zipped_positiontime[:-1]):
            remaining = np.array(zipped_positiontime[i_idx+1:])
            first_idx = [ np.min(np.where(remaining[:,0] == j_value)) for j_value in np.unique(remaining[:,0]) if j_value != i_value[0]]
            for j_value in remaining[first_idx]:
                transition_counts[i_value[0],int(j_value[0])] += 1
                transition_times[i_value[0],int(j_value[0])] += j_value[1] - i_value[1]
    transition_times = np.where(transition_counts == 0, np.inf, transition_times)
    return transition_times/transition_counts

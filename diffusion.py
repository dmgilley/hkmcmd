#!/usr/bin/env python
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu


import datetime
import numpy as np
import pandas as pd
from typing import Union
from hybrid_mdmc import utility
from hybrid_mdmc.frame_generator import frame_generator
from hybrid_mdmc.classes import MoleculeList
from hybrid_mdmc.mol_classes import AtomList
from hybrid_mdmc.voxels import Voxels


class Diffusion:
    """Class for calculating diffusion rates.

    Attributes
    ----------
    name (str): The name of the system.
    filename_trajectory (str): The filename of the trajectory file.
    atoms_datafile (AtomList): The AtomList instance assembled from the datafile information.
    molecules_datafile (MoleculeList): The MoleculeList instance assembled from the datafile
        information.
    number_of_voxels (list): The number of voxels in each dimension.
    voxels (Voxels): The Voxels instance assembled from the datafile information.
    time_conversion (float):  The time conversion factor.
        desired_units = current_units * time_conversion
    molecular_voxel_assignments_by_frame_array (None | np.ndarray): Array of N frames by M
        molecules, where entry i,j is the voxel in which molecule j is located at frame i.
    timesteps (None | list): The timesteps of the frames in the trajectory file.
    direct_voxel_transition_rates (None | dict)
        key (str): The molecule type.
        value (np.ndarray): Direct voxel transition rates, where entry i,j is the average
            transition rate from voxel i to voxel j.
    random_walk_positions (None | dict)
        key (str): The molecule type.
        value (np.ndarray): NxM array, where an entry is the position index of a walker at a
            given step.
    random_walk_times (None | dict)
        key (str): The molecule type.
        value (np.ndarray): NxM array, where an entry is the time of a walker at a given step.
    random_walk_average_first_time_between_positions (None | dict)
        key (str): The molecule type.
        value (np.ndarray): LxL array, where entry i,j is the average time a walker travels from
            voxel i to voxel j, for all L possible voxels.
    diffusion_rates (None | dict)
        key (str): The molecule type.
        value (float): Diffusion rate, where entry i,j is the average diffusion rate from voxel i
            to voxel j.
    """

    def __init__(
        self,
        name: str,
        filename_trajectory: str,
        atoms_datafile: AtomList,
        molecules_datafile: MoleculeList,
        voxels: Voxels,
        time_conversion: float = 1.0,
    ):

        self.name = name
        self.filename_trajectory = filename_trajectory
        self.atoms_datafile = atoms_datafile
        self.molecules_datafile = molecules_datafile
        self.number_of_voxels = voxels.number_of_voxels
        self.voxels = voxels
        self.time_conversion = time_conversion
        self.molecular_voxel_assignments_by_frame_array = None
        self.timesteps = None
        self.direct_voxel_transition_rates = None
        self.random_walk_positions = None
        self.random_walk_times = None
        self.random_walk_average_first_time_between_positions = None
        self.diffusion_rates = None

        return

    def calculate_molecular_voxel_assignments_by_frame(
        self,
        start: int = 0,
        end: int = -1,
        every: int = 1,
        logfile: Union[None, utility.FileTracker] = None,
        file_mvabfa: Union[None, utility.FileTracker] = None,
    ) -> tuple:
        """Calculate the molecular voxel assignments by frame.

        Uses the calculate_molecular_voxel_assignments_by_frame_array function to parse the
        trajectory file to create the array and log the associated timesteps. Both objects are
        stored as attributes of the class and returned by this method. The logged timesteps are
        unaltered (i.e. there are the exact values listed in the trajectory file).

        Parameters
        ----------
        start (int, optional): The first frame in the trajectory file to parse. Default: 0.
        end (int, optional): The last frame in the trajectory file to parse. Default: -1.
        every (int, optional): The interval between frames to parse. Default: 1.

        Returns
        -------
        np.ndarray
            Array of N frames by M molecules, where entry i,j is the voxel in which molecule j is
            located at frame i.
        list
            The timesteps of the frames in the trajectory file.
        """
        result = calculate_molecular_voxel_assignments_by_frame_array(
            self.filename_trajectory,
            self.atoms_datafile,
            self.molecules_datafile,
            self.number_of_voxels,
            start=start,
            end=end,
            every=every,
            logfile=logfile,
            file_mvabfa=file_mvabfa,
        )
        self.molecular_voxel_assignments_by_frame_array = result[0]
        self.timesteps = result[1]
        return self.molecular_voxel_assignments_by_frame_array, self.timesteps

    def calculate_direct_voxel_transition_rates(
        self,
        time_conversion: Union[None, float] = None,
        start: int = 0,
        end: int = -1,
        every: int = 1,
        average_across_voxel_neighbors: bool = False,
    ) -> dict:
        """Calculate the direct voxel transition rates.

        Uses the calculate_direct_voxel_transition_rates function to calculate the direct voxel
        transition rates. The rates are stored as an attribute of the class and returned by this
        method. The time conversion factor is applied to the total time elapsed between the first
        and last timesteps in the trajectory file. If the instances does not have a molecular voxel
        assignments by frame array attribute, this method will call the
        calculate_molecular_voxel_assignments_by_frame method to parse the trajectory file.

        Parameters
        ----------
        time_conversion (float, optional): The time conversion factor. Default: None.
        start (int, optional): The first frame in the trajectory file to parse. Default: 0.
        end (int, optional): The last frame in the trajectory file to parse. Default: -1.
        every (int, optional): The interval between frames to parse. Default: 1.

        Returns
        dict
            key (str): The molecule type.
            value (np.ndarray): Direct voxel transition rates, where entry i,j is the average
                transition rate from voxel i to voxel j.
        """

        if self.molecular_voxel_assignments_by_frame_array is None:
            self.calculate_molecular_voxel_assignments_by_frame(
                start=start, end=end, every=every
            )
        if time_conversion is None:
            time_conversion = self.time_conversion
        self.direct_voxel_transition_rates = calculate_direct_voxel_transition_rates(
            np.prod(self.number_of_voxels),
            self.molecular_voxel_assignments_by_frame_array,
            # (self.timesteps[-1] - self.timesteps[0]) * time_conversion, # ERROR ERROR ERROR
            (self.timesteps[1] - self.timesteps[0]) * time_conversion,
            self.molecules_datafile,
        )
        if average_across_voxel_neighbors is True:
            for voxel_idxs in self.voxels.voxel_idxs_by_distance_groupings:
                for species in self.direct_voxel_transition_rates.keys():
                    self.direct_voxel_transition_rates[species][voxel_idxs] = np.mean(
                        self.direct_voxel_transition_rates[species][voxel_idxs]
                    )
        return self.direct_voxel_transition_rates

    def perform_random_walks(
        self,
        starting_position_idxs: Union[None, list, np.ndarray] = None,
        number_of_steps: Union[None, int] = None,
        species: Union[str, list] = "all",
    ) -> tuple:
        """Perform a set of random walks using perform_random_walks function.

        M independent walks will be simulated, each taking N steps through the L possible positions
        (i.e. the L voxel indices). The transfer rates are the direct voxel transition rates, with
        a mask applied to remove transitions between nonneighboring voxels. If the starting
        positions are not specified, one walker is placed in all voxels. If the number of steps is
        not specified, the number of steps will be 4 times the number of voxels.

        Parameters
        ----------
        starting_position_idxs (list | np.ndarray, optional): The starting position indices for
            each walker. Default: None.
        number_of_steps (int, optional): The number of steps each walk will take. Default: None.
        species (str, optional): The species to perform the random walks for. Default: "all".

        Returns
        -------
        dict
            key (str): The molecule type.
            value (np.ndarray): NxM array, where an entry is the position index of a walker at a
                given step.
        dict
            key (str): The molecule type.
            value (np.ndarray): NxM array, where an entry is the time of a walker at a given step.
        """
        if starting_position_idxs is None:
            starting_position_idxs = np.arange(np.prod(self.number_of_voxels)).flatten()
        starting_position_idxs = np.array(starting_position_idxs).flatten()
        if number_of_steps is None:
            number_of_steps = 4 * np.prod(self.number_of_voxels)
        if not isinstance(species, list):
            species = [species]
        if species[0] == "all":
            species = self.molecules_datafile.mol_types
        noneighbor_mask = np.array(
            [
                [
                    j_idx in self.voxels.neighbors_dict[i_idx]
                    for j_idx in range(np.prod(self.number_of_voxels))
                ]
                for i_idx in range(np.prod(self.number_of_voxels))
            ]
        )
        self.random_walk_positions, self.random_walk_times = {}, {}
        for sp in species:
            transfer_rates = np.where(
                noneighbor_mask == True, self.direct_voxel_transition_rates[sp], 0
            )
            self.random_walk_positions[sp], self.random_walk_times[sp] = (
                perform_random_walks(
                    transfer_rates, starting_position_idxs, number_of_steps
                )
            )
        return self.random_walk_positions, self.random_walk_times

    def calculate_average_first_time_between_positions(
        self,
        species: Union[str, list] = "all",
        recursion_interval: int = 1,
    ) -> dict:
        """Calculate the average first time between positions.

        Uses the calculate_average_first_time_between_positions function to calculate the average
        first time between voxels of the random walks previously preformed by the instance. The
        average times are stored as an attribute of the class and returned by this method.

        Parameters
        ----------
        species (str | list, optional): The species on which to operate. Default: "all".
        recursion_interval (int, optional): The interval at which to recurse through the position
            array. Default: 1.

        Returns
        -------
        dict
            key (str): The molecule type.
            value (np.ndarray): LxL array, where entry i,j is the average time a walker travels
                from voxel i to voxel j, for all L possible voxels.
        """

        if not isinstance(species, list):
            species = [species]
        if species[0] == "all":
            species = self.molecules_datafile.mol_types
        self.random_walk_average_first_time_between_positions = {}
        for sp in species:
            self.random_walk_average_first_time_between_positions[sp] = (
                calculate_average_first_time_between_positions(
                    self.random_walk_positions[sp],
                    self.random_walk_times[sp],
                    np.prod(self.number_of_voxels),
                    recursion_interval=recursion_interval,
                )
            )
        return self.random_walk_average_first_time_between_positions

    def calculate_diffusion_rates(
        self,
        starting_position_idxs: Union[None, list, np.ndarray] = None,
        number_of_steps: Union[None, int] = None,
        species: Union[str, list] = "all",
        recursion_interval: int = 1,
    ) -> dict:
        """Calculate the diffusion rates.

        Uses the calculate_diffusion_rates function to calculate the average diffusion rates
        between all voxels. The rates are stored as an attribute of the class and returned by this
        method. If the instance foes not have a random walk average first time between positions
        attribute, this method will call the perform_random_walks and
        calculate_average_first_time_between_positions methods.

        Parameters
        ----------
        starting_position_idxs (list | np.ndarray, optional): The starting position indices for
            each walker. Default: None.
        number_of_steps (int, optional): The number of steps each walk will take. Default: None.
        species (str, optional): The species to perform the random walks for. Default: "all".
        recursion_interval (int, optional): The interval at which to recurse through the position
            array. Default: 1.

        Returns
        -------
        dict
            key (str): The molecule type.
            value (float): Diffusion rate, where entry i,j is the average diffusion rate from voxel
                i to voxel j.
        """

        if not isinstance(species, list):
            species = [species]
        if species[0] == "all":
            species = sorted(list(set(self.molecules_datafile.mol_types)))
        if self.random_walk_average_first_time_between_positions is None:
            self.perform_random_walks(
                starting_position_idxs=starting_position_idxs,
                number_of_steps=number_of_steps,
                species=species,
            )
            self.calculate_average_first_time_between_positions(
                species=species, recursion_interval=recursion_interval
            )
        self.diffusion_rates = {}
        for sp in species:
            self.diffusion_rates[sp] = 1 / (
                self.random_walk_average_first_time_between_positions[sp]
            )
        return self.diffusion_rates


def calculate_molecular_voxel_assignments_by_frame_array(
    filename_trajectory: str,
    atoms_datafile: AtomList,
    molecules_datafile: MoleculeList,
    number_of_voxels: list,
    start: int = 0,
    end: int = -1,
    every: int = 1,
    logfile: Union[None, utility.FileTracker] = None,
    file_mvabfa: Union[None, utility.FileTracker] = None,
) -> tuple:
    """
    Calculate the "molecular voxel assignment by frame" array.

    This function calculates the "molecular voxel assignment by frame" array, which is a 2D array
    containing the voxel idx/ID in which each molecule is located for each frame. This is done by
    parsing through the trajectory file, calculating the center of geometry (COG) of each molecule,
    and assigning the COG to the voxel in which it is located.

    Parameters
    ----------
    filename_trajectory (str): The filename of the trajectory file.
    atoms_datafile (AtomList): The AtomList instance assembled from the datafile information.
    molecules_datafile (MoleculeList): The MoleculeList instance assembled from the datafile
        information.
    number_of_voxels (list): The number of voxels in each dimension.
    start (int, optional): The first frame in the trajectory file to parse. Default: 0.
    end (int, optional): The last frame in the trajectory file to parse. Default: -1.
    every (int, optional): The interval between frames to parse. Default: 1.

    Returns
    -------
    np.ndarray
        Array of N frames by M molecules, where entry i,j is the voxel in which molecule j is
        located at frame i.
    list
        The timesteps of the frames in the trajectory file.
    """

    molecular_voxel_assignment_by_frame_dict = {}
    all_timesteps, sub_timesteps = [], []
    adj_list = [
        [idx for idx, _ in enumerate(atoms_datafile.mol_id) if idx != aidx and _ == mol]
        for aidx, mol in enumerate(atoms_datafile.mol_id)
    ]
    generator_step = 1
    for atoms_thisframe, timestep, box_thisframe in frame_generator(
        filename_trajectory,
        start=start,
        end=end,
        every=every,
        unwrap=True,
        adj_list=adj_list,
        return_prop=False,
    ):
        all_timesteps.append(int(timestep))
        sub_timesteps.append(int(timestep))
        voxels_thisframe = Voxels(box_thisframe, number_of_voxels)
        molecules_thisframe = MoleculeList(
            ids=molecules_datafile.ids,
            mol_types=molecules_datafile.mol_types,
            atom_ids=molecules_datafile.atom_ids,
        )
        molecules_thisframe.get_cog(atoms_thisframe, box=box_thisframe)
        molecules_thisframe.assign_voxel_idxs(
            voxels_thisframe.origins, voxels_thisframe.boundaries
        )
        molecular_voxel_assignment_by_frame_dict[int(timestep)] = (
            molecules_thisframe.voxel_idxs
        )

        if logfile is not None and generator_step % 20 == 0:
            logfile.write(
                f"    {generator_step} frames parsed... {datetime.datetime.now()}\n"
            )
        if file_mvabfa is not None and generator_step % 40 == 0:
            molecular_voxel_assignment_by_frame_array = np.array(
                [
                    value
                    for key, value in sorted(
                        molecular_voxel_assignment_by_frame_dict.items()
                    )
                ]
            )
            output = pd.DataFrame(
                molecular_voxel_assignment_by_frame_array,
                index=[f"Frame {t}" for t in sub_timesteps],
            )
            file_mvabfa.write(f"{output.to_string(header=False, index=True)}\n")
            molecular_voxel_assignment_by_frame_dict = {}
            sub_timesteps = []
        generator_step += 1
    if file_mvabfa is not None:
        molecular_voxel_assignment_by_frame_array = np.array(
            [
                value
                for key, value in sorted(
                    molecular_voxel_assignment_by_frame_dict.items()
                )
            ]
        )
        output = pd.DataFrame(
            molecular_voxel_assignment_by_frame_array,
            index=[f"Frame {t}" for t in sub_timesteps],
        )
        file_mvabfa.write(f"{output.to_string(header=False, index=True)}\n")
        return None, all_timesteps
    molecular_voxel_assignment_by_frame_array = np.array(
        [
            value
            for key, value in sorted(molecular_voxel_assignment_by_frame_dict.items())
        ]
    )
    return (
        molecular_voxel_assignment_by_frame_array,
        all_timesteps,
    )


def calculate_direct_voxel_transition_rates(
    total_number_of_voxels: int,
    molecular_voxel_assignment_by_frame_array: np.ndarray,
    adjacent_transition_time: float,
    molecules_datafile: MoleculeList,
) -> dict:
    """
    Calculate the direct voxel transition rates.

    This function calculates the direct voxel transition rates, which are the rates at which
    molecules transition from one voxel to another. This is done by counting the number of
    transitions from one voxel to another for each molecule type, and dividing by time between
    adjacent frames in teh trajectory file.

    NOTE No neighbor mask is applied.
    NOTE Rates for a molecule to remain in a voxel (i.e. the matrix diagonal) ARE included.

    Parameters
    ----------
    total_number_of_voxels (int): Total number of voxels.
    molecular_voxel_assignment_by_frame_array (np.ndarray): Array of N frames by M molecules, where
        entry i,j is the voxel in which molecule j is located at frame i.
    adjacent_transition_time (float): Time between adjacent frames.
    molecules_datafile (MoleculeList): MoleculeList instance indexed to the column order of the
        molecular_voxel_assignment_by_frame_array.

    Returns
    -------
    dict
        key (str): The molecule type.
        value (np.ndarray): Direct voxel transition rates, where entry i,j is the average
            transition rate from voxel i to voxel j.
    """

    voxel_transition_counts = {
        _: np.zeros((total_number_of_voxels, total_number_of_voxels))
        for _ in set(molecules_datafile.mol_types)
    }
    for midx, type_ in enumerate(molecules_datafile.mol_types):
        voxel_list = molecular_voxel_assignment_by_frame_array[:, midx].flatten()
        voxel_list_shifted = np.roll(voxel_list, -1)
        transitions = np.column_stack((voxel_list, voxel_list_shifted))
        to_from, count = np.unique(transitions[:-1, :], axis=0, return_counts=True)
        for idx, tf in enumerate((to_from)):
            voxel_transition_counts[type_][tf[0], tf[1]] += count[idx]
    return {
        mol_type: vt_counts / adjacent_transition_time
        for mol_type, vt_counts in voxel_transition_counts.items()
    }


def perform_random_walks(
    transfer_rates: np.ndarray,
    starting_position_idxs: np.ndarray,
    number_of_steps: int,
) -> tuple:
    """Perform random walks.

    Perform random walks on a set of transfer rates. All rates of "remaining in place" are removed
    (i.e. the diagonal on the transfer rates array is zeroed). M independent walks will be
    simulated, each taking N steps through L possible positions.

    Parameters
    ----------
    transfer_rates (np.ndarray): LxL array, where entry i,j is the average transition rate from
        position i to position j.
    starting_position_idxs (np.ndarray): Mx1 array, where an entry is the starting position index
        for each of M total independent walkers.
    number_of_steps (int): The N steps each walk will take.

    Returns
    -------
    np.ndarray
        NxM array, where entry i,j is the position index of walker j at step i.
    np.ndarray
        NxM array, where entry i,j is the time of walker j at step i.
    """

    np.fill_diagonal(transfer_rates, 0)
    transfer_rates_sum = np.sum(transfer_rates, axis=1)
    transfer_rates_cumsum = np.cumsum(transfer_rates, axis=1)
    walkers_position = np.zeros(
        (number_of_steps + 1, starting_position_idxs.shape[0]), dtype=int
    )
    walkers_position[0] = starting_position_idxs.flatten()
    walkers_time = np.zeros(
        (number_of_steps + 1, starting_position_idxs.shape[0]), dtype=np.float64
    )
    for step in range(number_of_steps):
        u1 = np.random.rand(1, walkers_position.shape[1]).flatten()
        u1 *= np.array([transfer_rates_sum[idx] for idx in walkers_position[step]])
        walkers_position[step + 1] = np.array(
            [
                np.argwhere(transfer_rates_cumsum[walkers_position[step, idx]] >= u)[0][
                    0
                ]
                for idx, u in enumerate(u1)
            ]
        ).flatten()
        u2 = 1 - np.random.rand(1, walkers_position.shape[1]).flatten()
        dt = np.array(
            [
                -np.log(u) / transfer_rates_sum[walkers_position[step, idx]]
                for idx, u in enumerate(u2)
            ]
        ).flatten()
        walkers_time[step + 1] = walkers_time[step] + dt
    return walkers_position, walkers_time


def calculate_average_first_time_between_positions(
    walkers_position: np.ndarray,
    walkers_time: np.ndarray,
    number_of_positions: int,
    recursion_interval: int = 1,
) -> np.ndarray:
    """Calculate the average first time between positions.

    Calculate the average first time between positions given position and time arrays of a set of M
    random walks of N steps each.

    NOTE Position "labels" are expected to be their indices, zero indexed.

    Parameters
    ----------
    walkers_position (np.ndarray): NxM array, where entry i,j is the position index of walker j at
        step j.
    walkers_time (np.ndarray): NxM array, where entry i,j is the time of walker j at step i.
    number_of_positions (int): The number of possible positions.
    recursion_interval (int, optional): The interval at which to recurse through the position
        array. Default: 1.

    Returns
    -------
    np.ndarray
        LxL array, where entry i,j is the average time a walker travels from position i to position
        j, for all L possible positions.
    """

    transition_counts = np.zeros((number_of_positions, number_of_positions))
    transition_times = np.zeros((number_of_positions, number_of_positions))
    for colidx in range(walkers_position.shape[1]):
        zipped_positiontime = list(
            zip(walkers_position[:, colidx], walkers_time[:, colidx])
        )

        for i_idx in range(0, len(zipped_positiontime) - 1, recursion_interval):
            i_value = np.array(zipped_positiontime[i_idx])
            remaining = np.array(zipped_positiontime[i_idx + 1 :])
            first_idx = [
                np.min(np.where(remaining[:, 0] == j_value))
                for j_value in np.unique(remaining[:, 0])
                if j_value != i_value[0]
            ]
            for j_value in remaining[first_idx]:
                transition_counts[int(i_value[0]), int(j_value[0])] += 1
                transition_times[int(i_value[0]), int(j_value[0])] += (
                    j_value[1] - i_value[1]
                )
    return (
        np.where(transition_counts == 0, np.inf, transition_times) / transition_counts
    )

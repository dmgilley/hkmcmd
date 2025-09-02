#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import numpy as np
from copy import deepcopy
from sklearn import linear_model
from hkmcmd import interactions


def get_reactive_events_list(
    molecules_list,
    diffusion_rates_dict_matrix,
    reaction_scaling_df,
    reaction_templates_list,
    minimum_diffusion_rate,
    avoid_double_counts=True,
):

    event_dict, event_ID = {}, 1

    for reaction_template in reaction_templates_list:

        reaction_rate = (
            reaction_template.rawrate
            * reaction_scaling_df.loc[
                reaction_scaling_df.index[-1], reaction_template.kind
            ]
        )

        # Unimolecular reactions
        if len(reaction_template.reactant_molecules) == 1:
            reactive_molecule_idxs_i = [
                idx
                for idx, molecule in enumerate(molecules_list)
                if molecule.kind == reaction_template.reactant_molecules[0].kind
            ]
            for molecule_idx_i in reactive_molecule_idxs_i:
                event_dict[event_ID] = interactions.Reaction(
                    ID=event_ID,
                    kind=reaction_template.kind,
                    reactant_molecules=[molecules_list[molecule_idx_i]],
                    event_rate=reaction_rate,
                    event_rate_units="1/s",
                )
                event_ID += 1

        # Bimolecular reactions
        elif len(reaction_template.reactant_molecules) == 2:
            reactive_molecule_idxs_i = [
                idx
                for idx, molecule in enumerate(molecules_list)
                if molecule.kind == reaction_template.reactant_molecules[0].kind
            ]
            reactive_molecule_idxs_j = [
                idx
                for idx, molecule in enumerate(molecules_list)
                if molecule.kind == reaction_template.reactant_molecules[1].kind
            ]
            for molecule_idx_i in reactive_molecule_idxs_i:
                inner_loop_list = deepcopy(reactive_molecule_idxs_j)
                if molecule_idx_i in inner_loop_list:
                    inner_loop_list.remove(molecule_idx_i)
                if (
                    reaction_template.reactant_molecules[0].kind
                    == reaction_template.reactant_molecules[1].kind
                    and avoid_double_counts is True
                ):
                    inner_loop_list = [
                        idx_j
                        for idx_j in reactive_molecule_idxs_j
                        if idx_j > molecule_idx_i
                    ]
                for molecule_idx_j in inner_loop_list:
                    drate = np.max([
                        diffusion_rates_dict_matrix[
                            reaction_template.reactant_molecules[0].kind
                        ][
                            molecules_list[molecule_idx_i].voxel_idx,
                            molecules_list[molecule_idx_j].voxel_idx,
                        ],
                        diffusion_rates_dict_matrix[
                            reaction_template.reactant_molecules[0].kind
                        ][
                            molecules_list[molecule_idx_j].voxel_idx,
                            molecules_list[molecule_idx_i].voxel_idx,
                        ],]
                    )
                    if drate < minimum_diffusion_rate:
                        continue
                    dtime = np.inf
                    if drate != 0:
                        dtime = 1 / drate
                    event_dict[event_ID] = interactions.Reaction(
                        ID=event_ID,
                        kind=reaction_template.kind,
                        reactant_molecules=[
                            molecules_list[molecule_idx_i],
                            molecules_list[molecule_idx_j],
                        ],
                        event_rate=1 / ((1 / reaction_rate) + dtime),
                        event_rate_units="1/s",
                    )
                    event_ID += 1

        # More than bimolecular rxn
        elif len(reaction_template.reactant_molecules) > 2:
            raise ValueError(
                "Reactions between more than 2 molecules are not supported."
            )

    return [v for k, v in sorted(event_dict.items())]


def kMC_event_selection(event_list):
    rates = [event.event_rate for event in event_list]
    u2 = 0
    while u2 == 0:
        u2 = np.random.random()
    dt = -np.log(u2) / np.sum(rates)
    u1 = np.random.random()
    event_idx = np.argwhere(np.cumsum(rates) >= np.sum(rates) * u1)[0][0]
    return event_list[event_idx], dt


def get_PSSrxns(
    reactions_list,
    reaction_scaling,
    system_state_summary_df,
    windowsize_slope,
    windowsize_rxnselection,
    scalingcriteria_concentration_slope,
    scalingcriteria_concentration_cycles,
    scalingcriteria_rxnselection_count,
):
    """Determine the reactions that are in psuedo steady state.

    Given certain criteria and the system history, this function
    returns a list of reactions that are in psuedo steady state.

    Parameters
    ----------
    """

    # If the windowsize_slope exceeds the current number of MDMC cycles,
    # return an empty list. By definition, no reaction can be at PSS.
    #if windowsize_slope > len(system_state_summary_df):
    #    return []
    windowsize_slope = np.min([windowsize_slope, len(system_state_summary_df)])

    # Reset the windowsize_rxnselection if it is greater than the length of
    # the system_state_df df.
    windowsize_rxnselection = np.min(
        [windowsize_rxnselection, len(system_state_summary_df)]
    )

    # For each species, determine the number of cycles that have occured
    # for which the slope of the concentration has met the
    # scalingcriteria_concentration_slope criteria.
    cycles_of_constant_slope = {
        _: get_cycles_of_constant_slope(
            system_state_summary_df.loc[:, _].to_numpy(),
            windowsize_slope,
            scalingcriteria_concentration_slope,
            number_of_windows=scalingcriteria_concentration_cycles,
        )
        for _ in system_state_summary_df.columns
        if _ != "time" and type(_) == str
    }

    # Loop over the reactions, checking each for PSS based on the
    # desired criteria.
    PSSrxns = []
    true_dynamics_steps = [
        idx
        for idx in system_state_summary_df.index[-windowsize_rxnselection:]
        if np.all(reaction_scaling.loc[idx] == 1.0)
    ]
    for reaction in reactions_list:

        rxntype = reaction.kind

        # Create a list of all species involved in this reaction,
        # reactants AND products.
        species_rxn = [molecule.kind for molecule in reaction.reactant_molecules] + [
            molecule.kind for molecule in reaction.product_molecules
        ]

        # Check that all of the species for this reaction have an unchanging
        # slope of concentration. If not, exit the loop for this rxntype.
        if not np.all(
            np.array([cycles_of_constant_slope[_] for _ in species_rxn])
            >= scalingcriteria_concentration_cycles
        ):
            continue

        # Check that this rxntype has been selected the desired number
        # of times over the desired number of previous cycles. If not,
        # exit the loop for this rxntype.
        # if not np.sum(system_state_df.loc[system_state_df.index[-windowsize_rxnselection]:,rxntype]) >= scalingcriteria_rxnselection_count:
        if (
            not np.sum(
                [
                    system_state_summary_df.loc[idx, rxntype]
                    for idx in true_dynamics_steps
                ]
            )
            >= scalingcriteria_rxnselection_count
        ):
            continue

        # If this point is reached, the reaction may be added to the
        # PSS list.
        PSSrxns.append(rxntype)

    return PSSrxns


def scalerxns(
    reaction_scaling,
    PSSrxns,
    windowsize_scalingpause,
    scalingfactor_adjuster,
    scalingfactor_minimum,
    rxnlist="all",
):

    # Declare the last cycle
    lastcycle = reaction_scaling.index[-1]

    # Handle the default rxnlist
    if rxnlist == "all":
        rxnlist = sorted(list(reaction_scaling.columns))

    # If the number of cycles that have passed since the last scaling factor was adjusted
    # is less than the windowsize_scalingpause, return unscaled reactions.
    if (
        np.min(
            [
                get_cyclesofconstantscaling(reaction_scaling.loc[:, rxntype].to_numpy())
                for rxntype in reaction_scaling.columns
            ]
        )
        < windowsize_scalingpause
    ):
        reaction_scaling.loc[lastcycle + 1] = [1.0 for _ in reaction_scaling.columns]
        return reaction_scaling

    # Create a dictionary to track the new reaction scaling
    newscaling = {
        rxntype: reaction_scaling.loc[lastcycle, rxntype]
        for rxntype in reaction_scaling.columns
    }

    # Loop over the rxnlist and scale reactions
    for rxntype in rxnlist:

        # If this reaction is not in PSS, upscale it.
        if rxntype not in PSSrxns:
            newscaling[rxntype] /= scalingfactor_adjuster
            continue

        # If this reaction is in PSS, downscale it.
        if rxntype in PSSrxns:
            newscaling[rxntype] *= scalingfactor_adjuster
            continue

    # Append the new reaction scaling to rxnscaling
    reaction_scaling.loc[lastcycle + 1] = [
        newscaling[rxntype] for rxntype in reaction_scaling.columns
    ]

    # If all reactions are scaled, upscale
    while not np.any(reaction_scaling.loc[lastcycle + 1] >= 1.0):
        reaction_scaling.loc[lastcycle + 1] /= scalingfactor_adjuster

    # Adjust scalings that are outside of the scaling limits.
    reaction_scaling[reaction_scaling > 1.0] = 1.0
    reaction_scaling[reaction_scaling < scalingfactor_minimum] = scalingfactor_minimum

    return reaction_scaling


def get_cyclesofconstantscaling(reaction_scaling: np.array) -> int:
    """Determine the number of cycles that the scaling factor has remained constant.

    Given an array of reaction rate scaling factors, this
    function calculates the number of steps that the scaling
    factor has remained unchanged, counted from the most recent step.

    Parameters
    ----------
    reaction_scaling : numpy array
        An array of reaction rate scaling factors.

    Returns
    -------
    int
        The number of cycles that the scaling factor has remained constant.

    Edge Cases
    ----------
    If the reaction_scaling array has no length, return a value of 0.

    If the reaction_scaling array has no changes, return the length of the array.

    If the reaction_scaling changed between the previous step and the current step, return 1.
        e.g. np.array([1.0,1.0,0.1]) returns 1.
    """
    if len(reaction_scaling) == 0:
        return 0
    change_indices = np.where(np.diff(reaction_scaling) != 0)[0] + 1
    if len(change_indices) == 0:
        return len(reaction_scaling)
    return len(reaction_scaling) - change_indices[-1]


def get_cycles_of_constant_slope(
    particle_count: np.array,
    windowsize_slope: int,
    scalingcriteria_concentration_slope: float,
    number_of_windows: int = 1,
) -> int:
    """Determine the number of cycles that the slope of the particle count has remained constant.

    Parameters
    ----------
    particle_count : numpy.array
        An array containing the particle count of a species over time.

    windowsize_slope : int
        The number of cycles to use when calculating the slope.

    scalingcriteria_concentration_slope : float
        The maximum slope of the concentration that is considered "constant".

    number_of_windows : int
        The number of windows over which to calculate the slope.

    Returns
    -------
    int
        The number of cycles that the slope of the concentration has remained constant.
    """

    # If the system_state_df dataframe has no length, return a value of 0.
    if len(particle_count) == 0 or len(particle_count) < windowsize_slope:
        return 0

    # Adjust the steps
    steps = int(number_of_windows + windowsize_slope - 1)
    if steps > len(particle_count):
        steps = len(particle_count)
    particle_count = particle_count[-steps:]

    # Calculate the absolute value of the slopes using the provided window size.
    slope = np.array(
        [
            np.abs(
                linear_model.LinearRegression()
                .fit(
                    np.array(range(windowsize_slope)).reshape(-1, 1),
                    particle_count[idx : idx + windowsize_slope],
                )
                .coef_[0]
            )
            for idx in range(0, len(particle_count) - windowsize_slope + 1)
        ]
    )

    # If all of the cycles satisfy the scalingcriteria_concentration_slope, return the length of slope.
    if np.all(slope <= scalingcriteria_concentration_slope):
        return len(slope)

    # Otherwise, return the index of the first instance where the slope does NOT satisfy scalingcriteria_concentration_slope.
    return np.argwhere(slope[::-1] > scalingcriteria_concentration_slope)[0][0]

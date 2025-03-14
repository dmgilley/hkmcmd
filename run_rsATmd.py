#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys, warnings
import numpy as np
from hybrid_mdmc import utility
from hybrid_mdmc.interactions import (
    Molecule,
    update_molecules_list_IDs,
    remove_overlaps_from_molecules_list,
    update_molecules_list_with_reaction,
    get_interactions_lists_from_molcules_list,
)
from hybrid_mdmc.voxels import Voxels
from hybrid_mdmc.reaction import (
    kMC_event_selection,
    get_PSSrxns,
    scalerxns,
    get_reactive_events_list,
)
from hybrid_mdmc.system import SystemState, SystemData
from hybrid_mdmc.filehandlers import (
    LammpsInitHandler,
    write_lammps_data,
    hkmcmd_ArgumentParser,
    parse_data_file,
    parse_diffusion_file,
)


# Main argument
def main(argv):
    """Driver for conducting rs@md simulation.
    
    Modification of run_hkmcmd.py to perform "rs@md[rate]" simulations, as described in
    "Biedermann et al., J. Chem. Theory Comput. 17, 1074-1085 (2021)." Candidate lists are
    assembled using the idea of voxels; molecules must be within the same voxel to be considered a
    reactive pair. Intervoxel diffusion rates are set to 0, and the diffusion_rate cutoff is set to
    np.inf. 
    """

    # Parse the command line arguments
    parser = hkmcmd_ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    args = parser.parse_args()

    # Read and clean the system data file
    system_data = SystemData(args.system, args.prefix, filename_json=args.filename_json)
    system_data.read_json()
    system_data.clean()
    for reaction in system_data.reactions:
        reaction.calculate_raw_rate(
            system_data.hkmcmd["temperature_rxn"], method="Arrhenius"
        )

    if args.debug is True:
        breakpoint()

    # Check that system_data is consistent with rs@md[rate] requirements
    issues = []
    if system_data.hkmcmd["diffusion_cutoff"] != np.inf:
        system_data.hkmcmd["diffusion_cutoff"] = np.inf
        issues.append("diffusion_cutoff")
    if system_data.hkmcmd["scale_rates"] is True:
        system_data.hkmcmd["scale_rates"] = False
        issues.append("scale_rates")
    if system_data.scaling_diffusion["well-mixed"] is False:
        system_data.scaling_diffusion["well-mixed"] = True
        issues.append("well-mixed")
    if args.verbose:
        warnings.warn(f"rs@md[rate] reset the following settings: {issues}.")

    # Calculate reaction window time
    rxn_dt = system_data.MD_cycling["run_steps"][3] * system_data.MD_cycling["run_stepsize"][3] * system_data.lammps["time_conversion"]

    # Read the data file
    (
        atoms_list,
        bonds_list,
        angles_list,
        dihedrals_list,
        impropers_list,
        box,
        _,
    ) = parse_data_file(
        args.filename_data,
        atom_style=system_data.lammps["atom_style"],
        preserve_atom_order=False,
        preserve_bond_order=False,
        preserve_angle_order=False,
        preserve_dihedral_order=False,
        preserve_improper_order=False,
        tdpd_conc=[],
        unwrap=False,
    )

    # Create the Voxels object
    voxels_datafile = Voxels(box, system_data.scaling_diffusion["number_of_voxels"])

    # Create the SystemState instance
    system_state = SystemState(filename=args.filename_system_state)
    system_state.read_data_from_json()

    # Check for consistency between SystemState and Datafile atoms
    system_state.check_atoms_list(atoms_list)

    # Create molecules_list
    molecules_list = [
        Molecule(ID=ID)
        for ID in sorted(list(set([atom.molecule_ID for atom in atoms_list])))
    ]
    for molecule in molecules_list:
        molecule.fill_lists(
            atoms_list=atoms_list,
            bonds_list=bonds_list,
            angles_list=angles_list,
            dihedrals_list=dihedrals_list,
            impropers_list=impropers_list,
        )
        molecule.kind = system_state.get_molecule_kind(molecule.ID)
        molecule.assign_voxel_idx(voxels_datafile)

    if args.debug is True:
        breakpoint()

    # Set specific diffusion rates for rs@md[rate]
    diffusion_rates_dict_matrix = {
        sp: np.ones((np.prod(voxels_datafile.number_of_voxels), np.prod(voxels_datafile.number_of_voxels)))*np.inf
        for sp in [molecule.kind for molecule in system_data.species]
    }
    for sp in diffusion_rates_dict_matrix:
        diffusion_rates_dict_matrix[sp].fill(0)
        np.fill_diagonal(diffusion_rates_dict_matrix[sp], np.inf)

    # Begin the KMC loop
    moleculecount_starting = len(molecules_list)
    moleculecount_current = len(molecules_list)
    Reacting = True
    while Reacting:

        # Assemble DataFrames from the system_state instance.
        # Later, the functions that use these DataFrames can be updated to take the system_state instance.
        reaction_scaling_df = system_state.assemble_reaction_scaling_df()
        progression_df = system_state.assemble_progression_df(sorted([molecule.kind for molecule in system_data.species]))

        # Assemble a list of all possible reactions
        reactive_events_list = get_reactive_events_list(
            molecules_list,
            diffusion_rates_dict_matrix,
            reaction_scaling_df,
            system_data.reactions,
            system_data.hkmcmd["diffusion_cutoff"],
            avoid_double_counts=system_data.hkmcmd["avoid_double_counts"],
        )

        # Select a reaction
        selected_reactive_events = kMC_event_selection_rsATmd(reactive_events_list, rxn_dt)
        for event in selected_reactive_events:
            event.create_product_molecules(
                [
                    reaction
                    for reaction in system_data.reactions
                    if reaction.kind == event.kind
                ][0]
            )

        if args.debug is True:
            breakpoint()

        # Update the system with the selected reaction
        molecules_list = update_molecules_list_with_reaction_rsATmd(
            molecules_list,
            selected_reactive_events,
            box,
            tolerance=0.0000_0010,
            maximum_iterations=None,
        )
        system_state.update_molecules(molecules_list)
        system_state.update_reactions([event.kind for event in selected_reactive_events],reaction_scaling_df)
        system_state.update_steps(args.diffusion_cycle, rxn_dt)
        system_state.check_consistency()

        # Check for completion.
        Reacting = False

    # Write the LAMMPS data file
    atoms_list, bonds_list, angles_list, dihedrals_list, impropers_list = (
        get_interactions_lists_from_molcules_list(molecules_list)
    )

    if args.debug is True:
        breakpoint()

    write_lammps_data(
        args.prefix + ".in.data",
        atoms_list,
        bonds_list=bonds_list,
        angles_list=angles_list,
        dihedrals_list=dihedrals_list,
        impropers_list=impropers_list,
        box=box,
        num_atoms=len(atoms_list),
        num_bonds=len(bonds_list),
        num_angles=len(angles_list),
        num_dihedrals=len(dihedrals_list),
        num_impropers=len(impropers_list),
        num_atom_types=system_data.lammps.get("atom_types", None),
        num_bond_types=system_data.lammps.get("bond_types", None),
        num_angle_types=system_data.lammps.get("angle_types", None),
        num_dihedral_types=system_data.lammps.get("dihedral_types", None),
        num_improper_types=system_data.lammps.get("improper_types", None),
        masses=system_data.atomic_masses,
        charge=system_data.lammps.get("charged_atoms", None),
        wrap=False,
    )

    # Write the LAMMPS init file
    lammps_init = LammpsInitHandler(
        prefix=args.prefix,
        settings_file_name=args.filename_settings,
        data_file_name=args.prefix + ".in.data",
        **system_data.MD_cycling,
    )
    lammps_init.write()

    # Write the system_state file
    system_state.write_to_json(args.filename_system_state)

    return


def update_molecules_list_with_reaction_rsATmd(
    molecules_list,
    reactive_events,
    box,
    tolerance=0.0000_0010,
    maximum_iterations=None,
):
    reactant_molecules = utility.flatten_nested_list([event.reactant_molecules for event in reactive_events])
    product_molecules = utility.flatten_nested_list([event.product_molecules for event in reactive_events])
    molecules_list = [
        molecule
        for molecule in molecules_list
        if molecule.ID not in [_.ID for _ in reactant_molecules]
    ]
    molecules_list += product_molecules
    molecules_list = update_molecules_list_IDs(molecules_list, reset_atom_IDs=False)
    molecules_list = remove_overlaps_from_molecules_list(
        molecules_list, box, tolerance=tolerance, maximum_iterations=maximum_iterations
    )
    return molecules_list


def kMC_event_selection_rsATmd(event_list, dt):
    selected_events = []
    reacted_atomIDs = []
    for event in event_list:
        reactive_atomIDs = utility.flatten_nested_list([mol.atom_IDs for mol in event.reactant_molecules])
        if len(np.intersect1d(reactive_atomIDs, reacted_atomIDs)) > 0:
            continue
        u2 = 0
        while u2 == 0:
            u2 = np.random.random()
        if event.event_rate * dt > u2:
            selected_events.append(event)
            reacted_atomIDs += reactive_atomIDs
    return selected_events


if __name__ == "__main__":
    main(sys.argv[1:])

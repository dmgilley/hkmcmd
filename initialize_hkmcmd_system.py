#!/usr/bin/env python3
# Author
#    Dylan M. Gilley
#    dgilley@purdue.edu


import sys
import numpy as np
from copy import deepcopy
from hybrid_mdmc.particle_interactions import update_molecules_list_IDs
from hybrid_mdmc.filehandlers_lammps import write_lammps_data, LammpsInitHandler
from hybrid_mdmc.filehandlers_general import hkmcmd_ArgumentParser
from hybrid_mdmc.simulation_system import SystemData, SystemState
from hybrid_mdmc import utility


def main(argv):

    # Parse the command line arguments
    parser = hkmcmd_ArgumentParser()
    args = parser.parse_args()


    # Read and clean the system data file
    system_data = SystemData(args.system, args.prefix)
    system_data.read_json()
    system_data.clean()

    # Determine requested number of molecules
    molecule_counts = [
        system_data.lammps.get("starting_" + species.kind, 0)
        for species in system_data.species
    ]
    molecules_list = utility.flatten_nested_list(
        [
            [deepcopy(molecule) for _ in range(molecule_counts[idx])]
            for idx, molecule in enumerate(system_data.species)
        ]
    )

    # Determine the necessary box dimensions
    nodes_perside = int(np.ceil(np.sum(molecule_counts) ** (1.0 / 3.0)))
    spacing = 6
    box_length = (nodes_perside + 2) * spacing
    centers = np.array(
        [
            [x, y, z]
            for x in np.linspace(
                -box_length / 2 + spacing, box_length / 2 - spacing, num=nodes_perside
            )
            for y in np.linspace(
                -box_length / 2 + spacing, box_length / 2 - spacing, num=nodes_perside
            )
            for z in np.linspace(
                -box_length / 2 + spacing, box_length / 2 - spacing, num=nodes_perside
            )
        ]
    )

    # Place the molecules
    np.random.shuffle(molecules_list)
    np.random.shuffle(centers)
    molecules_list = update_molecules_list_IDs(molecules_list, reset_atom_IDs=True)
    for idx, molecule in enumerate(molecules_list):

        # Translate and rotate the molecule
        molecule.calculate_cog()
        molecule.translate(centers[idx])
        for axis in ["x", "y", "z"]:
            molecule.rotate(np.random.uniform(0, 2 * np.pi), axis)

    # Assemble the system
    atoms_list = utility.flatten_nested_list([molecule.atoms for molecule in molecules_list])
    bonds_list = utility.flatten_nested_list([molecule.bonds for molecule in molecules_list if molecule.bonds is not None])
    angles_list = utility.flatten_nested_list([molecule.angles for molecule in molecules_list if molecule.angles is not None])
    dihedrals_list = utility.flatten_nested_list([molecule.dihedrals for molecule in molecules_list if molecule.dihedrals is not None])
    impropers_list = utility.flatten_nested_list([molecule.impropers for molecule in molecules_list if molecule.impropers is not None])
    box = [[-box_length / 2, box_length / 2]]*3

    # Initialize the system state file
    system_state = SystemState(
        filename=args.filename_system_state,
        atom_IDs=[atom.ID for atom in atoms_list.sort(key=lambda atom: atom.ID)],
        reaction_kinds=sorted([reaction.kind for reaction in system_data.reactions]),
        diffusion_steps = [0],
        reactive_steps = [0],
        times = [0.0],
        molecule_IDs = [atom.molecule_ID for atom in atoms_list.sort(key=lambda atom: atom.ID)],
        molecule_kinds = [atom.molecule_kind for atom in atoms_list.sort(key=lambda atom: atom.ID)],
        reaction_selections = [0 for _ in system_data.reactions],
        reaction_scalings = [1.0 for _ in system_data.reactions],
    )
    system_state.write_to_json()

    # Write the initial LAMMPS data file
    write_lammps_data(
        args.prefix+".in.data",
        atoms_list,
        bonds_list=bonds_list,
        angles_list=angles_list,
        dihedrals_list=dihedrals_list,
        impropers_list=impropers_list,
        box=box,
        num_atoms=None,
        num_bonds=None,
        num_angles=None,
        num_dihedrals=None,
        num_impropers=None,
        num_atom_types=system_data.lammps.get("atom_types", None),
        num_bond_types=system_data.lammps.get("bond_types", None),
        num_angle_types=system_data.lammps.get("angle_types", None),
        num_dihedral_types=system_data.lammps.get("dihedral_types", None),
        num_improper_types=system_data.lammps.get("improper_types", None),
        masses=system_data.atomic_masses,
        charge=system_data.lammps.get("charged_atoms", None),
        wrap=False,
    )

    # Write the initial LAMMPS input file
    lammps_init = LammpsInitHandler(
        prefix=args.prefix,
        settings_file_name=args.filename_settings,
        data_file_name=args.prefix+".in.data",
        **system_data.MD_initial
    )
    lammps_init.write()

    return


if __name__ == "__main__":
    main(sys.argv[1:])

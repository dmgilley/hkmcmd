#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys
import numpy as np
from copy import deepcopy
from hkmcmd import io, utility, system, interactions


def main(argv):

    # Parse command line arguments
    parser = io.HkmcmdArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    # Read system data
    system_data = system.SystemData(args.system, args.prefix, filename_json=args.filename_json)
    system_data.read_json()
    system_data.clean()

    # Read the data file
    (
        atoms_list,
        bonds_list,
        angles_list,
        dihedrals_list,
        impropers_list,
        box,
        _,
    ) = io.parse_data_file(
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

    # Create the SystemState instance
    system_state = system.SystemState(filename=args.filename_system_state)
    system_state.read_data_from_json()

    # Check for consistency between SystemState and Datafile atoms
    system_state.check_atoms_list(atoms_list)

    # Create molecules_list
    molecules_list = [
        interactions.Molecule(ID=ID)
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

    # Create molecular cog array
    (
        molecular_cog_array,
        molecule_types,
        frames,
        box,
    ) = assemble_molecular_cog(
        args.filename_trajectory,
        molecules_list,
        start = system_data.hkmcmd["msd_start"],
        end = system_data.hkmcmd["msd_end"],
        every = system_data.hkmcmd["msd_every"],
    )

    # Calculate MSD
    (
        mean_displacements_avg,
        mean_displacements_std,
        mean_squared_displacements_avg,
        mean_squared_displacements_std,
    ) = calculate_msd(molecular_cog_array, molecule_types, len(frames), box)

    # Write MSD to file
    times = np.array(frames) * system_data.lammps_timestep_size * system_data.lammps["time_conversion"]
    filename_output = f"{args.prefix}.msd.txt"
    write_msd_file(
        filename_output,
        args.diffusion_cycle,
        times,
        mean_displacements_avg,
        mean_displacements_std,
        mean_squared_displacements_avg,
        mean_squared_displacements_std,
        overwrite=args.overwrite,
    )

    return


def assemble_molecular_cog(
    filename_trajectory: str,
    molecules_list: list,
    start: int = 0,
    end: int = -1,
    every: int = 1,
) -> tuple:
    molecules_list = deepcopy(molecules_list)
    atoms_list, _, _, _, _ = interactions.get_interactions_lists_from_molcules_list(molecules_list)
    adj_list = [
        [
            idx2
            for idx2, atom2 in enumerate(atoms_list)
            if idx2 != idx1 and atom2.molecule_ID == atom1.molecule_ID
        ]
        for idx1, atom1 in enumerate(atoms_list)
    ]
    atomID2idx = {atom.ID: idx for idx, atom in enumerate(atoms_list)}
    molecular_cog = {}
    for atoms_list_thisframe, timestep_thisframe, box_thisframe, _ in io.frame_generator(
        filename_trajectory,
        start=start,
        end=end,
        every=every,
        unwrap=True,
        adj_list=adj_list,
        return_prop=False,
    ):
        box_thisframe = box_thisframe.flatten()
        for molecule in molecules_list:
            for atom in molecule.atoms:
                atom.x = atoms_list_thisframe[atomID2idx[atom.ID]].x
                atom.y = atoms_list_thisframe[atomID2idx[atom.ID]].y
                atom.z = atoms_list_thisframe[atomID2idx[atom.ID]].z
            molecule.calculate_cog(box=box_thisframe)
        molecular_cog[int(timestep_thisframe)] = [
            molecule.cog.flatten().tolist() for molecule in molecules_list
        ]
    frames = sorted(list(molecular_cog.keys()))
    molecular_cog = [v for k, v in sorted(molecular_cog.items())]
    molecular_cog = np.array(molecular_cog) # shape (n_frames, n_molecules, n_dimensions)
    return (
        molecular_cog,
        [mol.kind for mol in molecules_list],
        frames,
        [[box_thisframe[0], box_thisframe[1]],[box_thisframe[2], box_thisframe[3]],[box_thisframe[4], box_thisframe[5]]],
    )


def calculate_msd(
    molecular_cog_array: np.ndarray,
    molecule_types: list,
    number_of_frames: int,
    box: list,
) -> tuple:

    molecule_types_set = sorted(list(set(molecule_types)))
    dict_ = {mt: np.zeros(number_of_frames - 1) for mt in molecule_types_set}
    mean_displacements_avg = deepcopy(dict_)
    mean_displacements_std = deepcopy(dict_)
    mean_squared_displacements_avg = deepcopy(dict_)
    mean_squared_displacements_std = deepcopy(dict_)
    for mt in molecule_types_set:
        cog_array = deepcopy(molecular_cog_array)
        indices = [i for i, x in enumerate(molecule_types) if x == mt]
        cog_array = cog_array[:, indices, :]
        differences = [utility.find_dist_same(cog_array[:, midx, :], box) for midx in range(cog_array.shape[1])]
        for separation in range(1, number_of_frames):
            diffs_indices = tuple(list(zip(*[[i, i + separation] for i in range(number_of_frames - separation)])))
            displacements = np.array(utility.flatten_nested_list([_[diffs_indices] for _ in differences])).flatten()
            squared_displacements = displacements**2
            mean_displacements_avg[mt][separation - 1] = np.mean(displacements)
            mean_displacements_std[mt][separation - 1] = np.std(displacements)
            mean_squared_displacements_avg[mt][separation - 1] = np.mean(
                squared_displacements
            )
            mean_squared_displacements_std[mt][separation - 1] = np.std(
                squared_displacements
            )
    return (
        mean_displacements_avg,
        mean_displacements_std,
        mean_squared_displacements_avg,
        mean_squared_displacements_std,
    )


def write_msd_file(
    filename: str,
    diffusion_cycle: int,
    times: np.ndarray,
    displacements_avg: dict,
    displacements_std: dict,
    squared_displacements_avg: dict,
    squared_displacements_std: dict,
    overwrite: bool = False,
) -> None :
    output_file = io.FileTracker(filename, overwrite=overwrite)
    output_file.write_separation()
    output_file.write(f"\ndiffusion_cycle {diffusion_cycle}\n")
    output_file.write_array(("times (s) ", times))
    for species in sorted(displacements_avg.keys()):
        output_file.write(f"\nspecies {species}\n")
        output_file.write_array(("displacement_avg ", displacements_avg[species]))
        output_file.write_array(("displacement_std ", displacements_std[species]))
        output_file.write_array(("squared_displacement_avg ",squared_displacements_avg[species],))
        output_file.write_array(("squared_displacement_std ",squared_displacements_std[species],))
    return


def read_msd_file(filename):
    """
    mean_displacements_avg
    mean_displacements_std
    mean_squared_displacements_avg
    mean_squared_displacements_std
    times
    diffusion_cycle
    """

    diffusion_cycles = []
    times = {}
    species_list = []
    displacement_avg = {}
    displacement_std = {}
    squared_displacement_avg = {}
    squared_displacement_std = {}
    with open(filename, "r") as file:
        for line in file:
            fields = line.split()
            if len(fields) == 0:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "diffusion_cycle":
                cycle = int(fields[1])
                diffusion_cycles.append(cycle)
                continue
            if fields[0] == "times":
                times[cycle] = np.array(list(map(float, fields[2:])))
                continue
            if fields[0] == "species":
                species = fields[1]
                if species not in species_list:
                    species_list.append(species)
                    displacement_avg[species] = {}
                    displacement_std[species] = {}
                    squared_displacement_avg[species] = {}
                    squared_displacement_std[species] = {}
                continue
            if fields[0] == "displacement_avg":
                displacement_avg[species][cycle] = np.array(list(map(float, fields[1:])))
                continue
            if fields[0] == "displacement_std":
                displacement_std[species][cycle] = np.array(list(map(float, fields[1:])))
                continue
            if fields[0] == "squared_displacement_avg":
                squared_displacement_avg[species][cycle] = np.array(list(map(float, fields[1:])))
                continue
            if fields[0] == "squared_displacement_std":
                squared_displacement_std[species][cycle] = np.array(list(map(float, fields[1:])))
                continue

    diffusion_cycles = sorted(diffusion_cycles)
    times = [times[cycle] for cycle in diffusion_cycles]
    displacement_avg = {species: [displacement_avg[species][cycle] for cycle in diffusion_cycles] for species in species_list}
    displacement_std = {species: [displacement_std[species][cycle] for cycle in diffusion_cycles] for species in species_list}
    squared_displacement_avg = {species: [squared_displacement_avg[species][cycle] for cycle in diffusion_cycles] for species in species_list}
    squared_displacement_std = {species: [squared_displacement_std[species][cycle] for cycle in diffusion_cycles] for species in species_list}
    
    return (
        diffusion_cycles,
        times,
        displacement_avg,
        displacement_std,
        squared_displacement_avg,
        squared_displacement_std,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

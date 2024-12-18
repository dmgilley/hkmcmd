#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

import os,sys,datetime,math
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.lammps_files_classes import write_lammps_data, LammpsInitHandler
from hybrid_mdmc.customargparse import HMDMC_ArgumentParser
from hybrid_mdmc.classes import *
from hybrid_mdmc.parsers import *
from hybrid_mdmc.kmc import *
from hybrid_mdmc.functions import *
#from hybrid_mdmc.calc_voxels import *
from hybrid_mdmc.voxels import Voxels
from hybrid_mdmc.diffusion import Diffusion

# Main argument
def main(argv):
    """Driver for conducting Hybrid MDMC simulation.
    """

    # Use HMDMC_ArgumentParser to parse the command line.
    parser = HMDMC_ArgumentParser()
    args = parser.args

    # Read in the data_file, diffusion_file, rxndf, and msf files.
    atoms, bonds, angles, dihedrals, impropers, box, adj_mat, extra_prop = parse_data_file(
        args.filename_data, unwrap=True, atom_style=args.atom_style)
    rxndata = parser.get_reaction_data_dict()
    masterspecies = parser.get_masterspecies_dict()
    data_header = parser.get_data_header_dict()
    rxnmatrix = get_rxnmatrix(rxndata, masterspecies)

    if args.debug:
        breakpoint()

    # Calculate the raw reaction rate for each reaction using the
    # Eyring equation. Ea is expected to be in kcal/mol, temperature is
    # expected to be in K, and the resulting units will be equivalent
    # to the untis of rxndata[r]['A'].
    R = 0.00198588  # kcal/mol/K
    for r in rxndata.keys():
        rxndata[r]['rawrate'] = \
            rxndata[r]['A'] * \
            args.temperature_rxn**rxndata[r]['b'] * \
            np.exp(-rxndata[r]['Ea']/args.temperature_rxn/R)

    if args.debug:
        breakpoint()

    # Create a voxels dictionary and related voxel mapping parameters
    # using the calc_voxels function and the datafile information.
    #voxels = calc_voxels(
    #    args.number_of_voxels, box,
    #    xbounds=args.x_bounds,
    #    ybounds=args.y_bounds,
    #    zbounds=args.z_bounds)
    #voxelsmap, voxelsx, voxelsy, voxelsz = voxels2voxelsmap(voxels)
    #voxelID2idx = {k: idx for idx, k in enumerate(sorted(list(voxels.keys())))}
    voxels_datafile = Voxels(box, args.number_of_voxels)
    atomtypes2moltype = {tuple(sorted([i[2] for i in v['Atoms']])):k for k,v in masterspecies.items()}

    # Create and populate an instance of the "MoleculeList" class.
    molecules = gen_molecules(atoms, atomtypes2moltype, voxels_datafile)

    # Check for consistency among the tracking files.
    tfs = [
        os.path.exists(f)
        for fidx, f in enumerate([args.filename_concentration, args.filename_scale, args.filename_log, args.filename_diffusion])
        if [True, args.scalerates, args.log, True][fidx]
    ]
    if len(set(tfs)) != 1:
        tf_names = [args.filename_concentration, args.filename_scale, args.filename_log, args.filename_diffusion]
        tf_requested = [True, args.scalerates, args.log, True]
        print('Error! Inconsistent exsitence of tracking files.')
        print('  File Name                Requested   Exists')
        for idx in range(len(tf_names)):
            r, e = 'True', 'True'
            if not tf_requested[idx]:
                r = 'False'
            if not os.path.exists(tf_names[idx]):
                e = 'False'
            print('  {:<24s} {:<9s}   {:<6s}'.format(tf_names[idx], r, e))
        print('Exiting...')
        quit()

    # Create/append the tracking files.
    for f in [
        (True, args.filename_concentration),
        (args.scalerates, args.filename_scale),
        (args.log, args.filename_log),
        (True, args.filename_diffusion)
    ]:
        if f[0]:
            lines, new = [], False
            if not os.path.exists(f[1]) and '.concentration' in f[1]:
                lines = [
                    'ReactionCycle  0\n',
                    'MoleculeCounts {}\n'.format(
                        ' '.join([
                            '{} {}'.format(
                                sp, len([i for i in molecules.mol_types if i == sp]))
                            for sp in sorted(masterspecies.keys())
                        ])
                    ),
                    'Time 0\n'
                ]
                new = True
            if not os.path.exists(f[1]) and '.scale' in f[1]:
                lines = [
                    'ReactionCycle  0\n',
                    'ReactionTypes {}\n'.format(
                        ' '.join(['{}'.format(i) for i in sorted(rxndata.keys())])),
                    'ReactionScaling {}\n'.format(
                        ' '.join(['1.0000' for i in sorted(rxndata.keys())]))
                ]
                new = True
            if not os.path.exists(f[1]) and '.log' in f[1]:
                new = True
            if not os.path.exists(f[1]) and '.diffusion' in f[1]:
                new = True
            write_tracking_file(
                f[1], lines,
                driver=sys.argv[0],
                time=datetime.datetime.now(),
                step=args.diffusion_step, new=new
            )

    if args.debug:
        breakpoint()

    # If rate scaling is requested, parse the concentration and scale
    # files and create the progression object.
    rxnscaling = pd.DataFrame(
        index=[1],
        columns=sorted(rxndata.keys()),
        data={_: 1 for _ in rxndata.keys()}
    )
    if args.scalerates:
        counts, times, selected_rxns = parse_concentration(args.filename_concentration)
        if counts == 'kill':
            quit()
        progression, MDMCcycles_progression = get_progression(
            counts, times, selected_rxns,
            [0]+list(sorted(rxndata.keys())),
            list(sorted(masterspecies.keys())))
        rxnscaling, MDMCcycles_scaling = parse_scale(args.filename_scale)

    # Check that the progression and scale files are consistent.
    if not np.all(
        sorted(list(rxndata.keys())) == sorted(list(rxnscaling.columns))
    ):
        print(
            'Error! ' +
            'The keys of the reaction data dictionary do not match the ' +
            'column headers of the reaction scaling dataframe. ' +
            'Exiting...'
        )
        if args.log:
            with open(args.logfile, 'a') as f:
                f.write('rxndata keys: {}\n'.format(
                    sorted(list(rxndata.keys())))
                )
                f.write('rxnscaling columns: {}\n'.format(
                    sorted(list(rxnscaling.columns)))
                )
        quit()
    if not np.all(
        sorted(MDMCcycles_progression) == sorted(MDMCcycles_scaling)
    ):
        print(
            'Error! ' +
            'MDMC cycles generated from \"get_progression\" and ' +
            '\"parse_scale\" are inconsistent. ' +
            'Exiting...'
        )
        if args.log:
            with open(args.logfile, 'a') as f:
                f.write('MDMCcycles_progression {}\n'.format(
                    sorted(MDMCcycles_progression))
                )
                f.write('MDMCcycles_scaling: {}\n'.format(
                    sorted(list(MDMCcycles_scaling)))
                )
        quit()

    if args.debug:
        breakpoint()

    # If requested, calculate the diffusion rate for each species.
    #reactivespecies = {k:v for k,v in masterspecies.items() if k in set([i for l in [_['reactant_molecules'] for _ in rxndata.values()] for i in l])}
    diffusion_rate = {
        _: np.full((len(voxels_datafile.voxel_IDs), len(voxels_datafile.voxel_IDs)), fill_value=np.inf)
        #for _ in reactivespecies.keys()
        for _ in masterspecies.keys()
    }
    if args.debug:
        breakpoint()
    if args.well_mixed is False:
        diffusion = Diffusion(
            'toy320.60A',
            args.filename_trajectory,
            atoms,
            molecules,
            voxels_datafile,
            time_conversion=args.lammps_time_units_to_seconds_conversion)
        diffusion.parse_trajectory_file(start=0, end=-1, every=1)
        diffusion.calculate_direct_voxel_transition_rates()
        diffusion.perform_random_walks(number_of_steps=864,species='A')
        diffusion.calculate_average_first_time_between_positions(species='A')
        diffusion.calculate_diffusion_rates(species='A')
        diffusion_rate['A'] = diffusion.diffusion_rates['A']

    # Append the diffusion file
    with open(args.filename_diffusion,'a') as f:
        for k,v in sorted(diffusion_rate.items()):
            f.write('\nDiffusion Rates for {}\n'.format(k))
            for row in v:
                f.write('{}\n'.format(' '.join([str(_) for _ in row])))

    # Begin the KMC loop
    molecount_starting, molecount_current = len(
        sorted(set(atoms.mol_id))), len(sorted(set(atoms.mol_id)))
    add, delete, selected_rxn_types = {}, [], []
    Reacting, rxn_cycle = True, 1
    while Reacting:

        if args.debug:
            breakpoint()

        # Perform reaction scaling, if requested.
        if args.scalerates:
            PSSrxns = get_PSSrxns(
                rxnmatrix,
                rxnscaling,
                progression,
                args.windowsize_slope,
                args.windowsize_rxnselection,
                args.scaling_criteria_concentration_slope,
                args.scaling_criteria_concentration_cycles,
                args.scaling_criteria_rxnselection_count
            )
            rxnscaling = scalerxns(
                rxnscaling,
                PSSrxns,
                args.windowsize_scalingpause,
                args.scalingfactor_adjuster,
                args.scalingfactor_minimum,
                rxnlist='all',
            )

        if args.debug:
            breakpoint()

        rxns = get_rxns_serial(
            molecules, voxels_datafile, diffusion_rate,
            rxnscaling, rxndata, args.diffusion_cutoff)


        if args.debug:
            breakpoint()

        # Execute reaction(s)
        add, delete, dt, selected_rt = serial_kmc_rxn(
            rxns, rxndata, molecules, translate_distance=0.5)
        molecount_current -= len(delete)
        selected_rxn_types += [selected_rt]

        if args.debug:
            breakpoint()

        # Check for completion.
        if molecount_current < (1-args.change_threshold)*molecount_starting:
            Reacting = False

        # Check for erroneous double deletion
        if len(set(delete)) != len(delete):
            print('Error! Molecules deleted twice. Exiting...')
            # quit()

        # If requested, print debugging information to the log file
        if args.log:
            with open(args.filename_log, 'a') as f:
                for idx in range(len(molecules.ids)):
                    print('{} {} {} {}'.format(
                        molecules.ids[idx],
                        molecules.mol_types[idx],
                        molecules.voxels[idx],
                        molecules.atom_ids[idx])
                    )
                f.write('add: {}\n'.format(add))
                f.write('delete: {}\n'.format(delete))
                f.write('selected_rxn_types: {}\n\n'.format(selected_rxn_types))

        # Update the topology
        atoms, bonds, angles, dihedrals, impropers = update_topology(
            masterspecies, molecules, add, delete, atoms, bonds, angles, dihedrals, impropers)
        
        if args.debug:
            breakpoint()

        # Create new molecules object
        molecules = gen_molecules(atoms, atomtypes2moltype, voxels_datafile)

        # Update the progression objects, if rate scaling is being performed
        if args.scalerates:
            progression = update_progression(progression, molecules, list(
                sorted(masterspecies.keys())), dt, selected_rxn_types)

        # pdb breakpoint
        if args.debug:
            breakpoint()

        # Update the tracking files
        lines = ['\nReactionCycle {}\n'.format(rxn_cycle),
                 'MoleculeCounts {}\n'.format(' '.join(['{} {}'.format(sp, len(
                     [i for i in molecules.mol_types if i == sp])) for sp in sorted(masterspecies.keys())])),
                 'SelectedReactionTypes {}\n'.format(
            ' '.join([str(i) for i in selected_rxn_types])),
            'Time {}\n'.format(dt)]
        with open(args.filename_concentration, 'a') as f:
            for l in lines:
                f.write(l)

        lines = ['\nReactionCycle {}\n'.format(rxn_cycle),
                 'ReactionTypes {}\n'.format(
                     ' '.join(['{}'.format(i) for i in sorted(rxndata.keys())])),
                 'ReactionScaling {}\n'.format(' '.join(['{}'.format(rxnscaling.loc[list(rxnscaling.index)[-1], i]) for i in sorted(rxndata.keys())]))]
        with open(args.filename_scale, 'a') as f:
            for l in lines:
                f.write(l)

        # Update the tracking
        rxn_cycle += 1
        add, delete, selected_rxn_types = {}, [], []

    # Get image flags for all atoms
    box_length_x = np.abs(box[0][1] - box[0][0])
    image_flags_x = np.where( atoms.x > box[0][1],  np.ceil((atoms.x - box[0][1]) / box_length_x),             0)
    image_flags_x = np.where( atoms.x < box[0][0], np.floor((atoms.x - box[0][0]) / box_length_x), image_flags_x)

    box_length_y = np.abs(box[1][1] - box[1][0])
    image_flags_y = np.where( atoms.y > box[1][1],  np.ceil((atoms.y - box[1][1]) / box_length_y),             0)
    image_flags_y = np.where( atoms.y < box[1][0], np.floor((atoms.y - box[1][0]) / box_length_y), image_flags_y)

    box_length_z = np.abs(box[2][1] - box[2][0])
    image_flags_z = np.where( atoms.z > box[2][1],  np.ceil((atoms.z - box[2][1]) / box_length_z),             0)
    image_flags_z = np.where( atoms.z < box[2][0], np.floor((atoms.z - box[2][0]) / box_length_z), image_flags_z)
    
    # Catch atomic overlaps
    overlaps = True
    overlap_tolerance = 0.0000_0010
    while overlaps is True:

        # wrap coordinates and check for overlaps
        atoms.x -= image_flags_x * box_length_x
        atoms.y -= image_flags_y * box_length_y
        atoms.z -= image_flags_z * box_length_z
        repeats_wrapped = []
        for idx_i,id_i in enumerate(atoms.ids):
            for idx_j,id_j in enumerate(atoms.ids[idx_i+1:],start=idx_i+1):
                if not math.isclose(atoms.x[idx_i], atoms.x[idx_j], rel_tol=0, abs_tol=overlap_tolerance): continue
                if not math.isclose(atoms.y[idx_i], atoms.y[idx_j], rel_tol=0, abs_tol=overlap_tolerance): continue
                if not math.isclose(atoms.z[idx_i], atoms.z[idx_j], rel_tol=0, abs_tol=overlap_tolerance): continue
                repeats_wrapped += [idx_i]
                break
        if len(repeats_wrapped) != 0:
            print('Found overlapping wrapped atoms! Removing {} overlaps...'.format(len(repeats_wrapped)))
        for _ in repeats_wrapped:
            atoms.x[_] += overlap_tolerance
            atoms.y[_] += overlap_tolerance
            atoms.z[_] += overlap_tolerance

        # then unwrap and check again
        atoms.x += image_flags_x * box_length_x
        atoms.y += image_flags_y * box_length_y
        atoms.z += image_flags_z * box_length_z
        repeats_unwrapped = []
        for idx_i,id_i in enumerate(atoms.ids):
            for idx_j,id_j in enumerate(atoms.ids[idx_i+1:],start=idx_i+1):
                if not math.isclose(atoms.x[idx_i], atoms.x[idx_j], rel_tol=0, abs_tol=overlap_tolerance): continue
                if not math.isclose(atoms.y[idx_i], atoms.y[idx_j], rel_tol=0, abs_tol=overlap_tolerance): continue
                if not math.isclose(atoms.z[idx_i], atoms.z[idx_j], rel_tol=0, abs_tol=overlap_tolerance): continue
                repeats_unwrapped += [idx_i]
                break
        if len(repeats_unwrapped) != 0:
            print('Found overlapping unwrapped atoms! Removing {} overlaps...'.format(len(repeats_unwrapped)))
        for _ in repeats_unwrapped:
            atoms.x[_] += overlap_tolerance
            atoms.y[_] += overlap_tolerance
            atoms.z[_] += overlap_tolerance

        # did we get them all?
        if len(repeats_wrapped) == 0 and len(repeats_unwrapped) == 0:
            overlaps = False

    # Write the LAMMPS data file
    write_lammps_data(args.filename_writedata, atoms, bonds, angles, dihedrals,
                      impropers, box, header=data_header, charge=args.charged_atoms)
    
    # Write the LAMMPS init file
    init_writer = LammpsInitHandler(
        prefix = args.prefix,
        settings_file_name = args.filename_settings,
        data_file_name = args.filename_writedata,
        **parser.cycled_MD_init_dict
    )
    init_writer.write()

    return


if __name__ == '__main__':
    main(sys.argv[1:])

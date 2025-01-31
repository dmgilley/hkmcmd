#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu

import os,sys,datetime
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.lammps_files_classes import write_lammps_data, write_lammps_init
from hybrid_mdmc.customargparse import HMDMC_ArgumentParser
from hybrid_mdmc.classes import *
from hybrid_mdmc.parsers import *
from hybrid_mdmc.kmc import *
from functions import *
from hybrid_mdmc.calc_voxels import *
from hybrid_mdmc.diffusion import *

# Main argument
def main(argv):
    """Driver for conducting Hybrid MDMC simulation.
    """

    # Use HMDMC_ArgumentParser to parse the command line.
    parser = HMDMC_ArgumentParser()
    parser.HMDMC_parse_args()
    parser.adjust_default_args()
    args = parser.args

    # Read in the data_file, diffusion_file, rxndf, and msf files.
    atoms, bonds, angles, dihedrals, impropers, box, adj_mat, extra_prop = parse_data_file(
        args.data_file, unwrap=True, atom_style=args.atom_style)
    rxndata = parse_rxndf(args.rxndf)
    masterspecies = parse_msf(args.msf)
    data_header = parse_header(args.header)
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
            rxndata[r]['A'][0] * \
            args.temp**rxndata[r]['b'][0] * \
            np.exp(-rxndata[r]['Ea'][0]/args.temp/R)

    if args.debug:
        breakpoint()

    # Create a voxels dictionary and related voxel mapping parameters
    # using the calc_voxels function and the datafile information.
    voxels = calc_voxels(
        args.num_voxels, box,
        xbounds=args.x_bounds,
        ybounds=args.y_bounds,
        zbounds=args.z_bounds)
    voxelsmap, voxelsx, voxelsy, voxelsz = voxels2voxelsmap(voxels)
    voxelID2idx = {k: idx for idx, k in enumerate(sorted(list(voxels.keys())))}
    atomtypes2moltype = {}
    for k, v in masterspecies.items():
        atomtypes2moltype[tuple(sorted([i[2] for i in v['Atoms']]))] = k

    # Create and populate an instance of the "MoleculeList" class.
    molecules = gen_molecules(atoms, atomtypes2moltype,
                              voxelsmap, voxelsx, voxelsy, voxelsz)

    # Check for consistency among the tracking files.
    tfs = [
        os.path.exists(f)
        for fidx, f in enumerate([args.conc_file, args.scale_file, args.log_file, args.diffusion_file])
        if [True, args.scalerates, args.log, True][fidx]
    ]
    if len(set(tfs)) != 1:
        tf_names = [args.conc_file, args.scale_file, args.log_file, args.diffusion_file]
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
        (True, args.conc_file),
        (args.scalerates, args.scale_file),
        (args.log, args.log_file),
        (True, args.diffusion_file)
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
        counts, times, selected_rxns = parse_concentration(args.conc_file)
        if counts == 'kill':
            quit()
        progression, MDMCcycles_progression = get_progression(
            counts, times, selected_rxns,
            [0]+list(sorted(rxndata.keys())),
            list(sorted(masterspecies.keys())))
        rxnscaling, MDMCcycles_scaling = parse_scale(args.scale_file)

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
    diffusion_rate = {
        _: np.full((len(voxels), len(voxels)), fill_value=np.inf)
        for _ in masterspecies.keys()
    }
    if not args.well_mixed:
        diffusion_rate = calc_diffusionrate(
            args.trj_file,
            atoms,
            box,
            masterspecies,
            args.num_voxels,
            xbounds=args.x_bounds,
            ybounds=args.y_bounds,
            zbounds=args.z_bounds
        )

    # Append the diffusion file
    with open(args.diffusion_file,'a') as f:
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

        # Declare the voxel list.
        vox_list = sorted(voxels.keys())

        # Perform reaction scaling, if requested.
        if args.scalerates:
            PSSrxns = get_PSSrxns(
                rxnmatrix,
                progression,
                args.windowsize_slope,
                args.windowsize_rxnselection,
                args.scalingcriteria_concentration_slope,
                args.scalingcriteria_concentration_cycles,
                args.scalingcriteria_rxnselection_count
            )
            rxnscaling = scalerxns(
                rxnscaling,
                progression,
                PSSrxns,
                args.windowsize_scalingpause,
                args.scalingfactor_adjuster,
                args.scalingfactor_minimum,
                rxnlist='all',
            )

        if args.debug:
            breakpoint()

        # Loop over each voxel to get the reactions available in each voxel.
        rmax_byvoxel = []
        for vox in vox_list:

            # Create and populate an instance of the "ReactionList" class.
            rxns = get_rxns(molecules, vox, voxelID2idx, diffusion_rate,
                            rxnscaling, delete, rxndata, args.diffusion_cutoff, args.temp)
            rmax_byvoxel.append(np.sum(rxns.rates))

        # Calculate the maximum voxel total reaction rate.
        Rmax = np.max(rmax_byvoxel)

        # pdb breakpoint
        if args.debug:
            breakpoint()

        # Loop over each voxel again. To prioritize memory over speed,
        # reactions are rediscoverd for each voxel. Then, each voxel
        # undergoes KMC selection of a reaction.
        for vox in vox_list:

            # Recalculate the rxns objects. This prioritizes memory
            # over speed. To prioritize speed over memory, save the
            # rxns object for each voxel when calculated above.
            rxns = get_rxns(
                molecules, vox, voxelID2idx, diffusion_rate,
                rxnscaling, delete, rxndata, args.diffusion_cutoff, args.temp)

            # Execute reaction(s)
            temp_delete, selected_rt = [], 0
            if len(rxns.ids) > 0:
                temp_add, temp_delete, dt, selected_rt = spkmc_rxn(
                    rxns, rxndata, molecules, Rmax, translate_distance=2.0)
                if len(set(delete) & set([i for _ in temp_delete for i in _])) > 0:
                    selected_rxn_types += [0]
                    continue
                add.update({k+len(add): v for k, v in temp_add.items()})
                delete += [i for _ in temp_delete for i in _]
            molecount_current -= len(temp_delete)
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
            with open(args.log_file, 'a') as f:
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

        # pdb breakpoint
        if args.debug:
            breakpoint()

        # Update the topology
        atoms, bonds, angles, dihedrals, impropers = update_topology(
            masterspecies, molecules, add, delete, atoms, bonds, angles, dihedrals, impropers)

        # Create new molecules object
        molecules = gen_molecules(
            atoms, atomtypes2moltype, voxelsmap, voxelsx, voxelsy, voxelsz)

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
        with open(args.conc_file, 'a') as f:
            for l in lines:
                f.write(l)

        lines = ['\nReactionCycle {}\n'.format(rxn_cycle),
                 'ReactionTypes {}\n'.format(
                     ' '.join(['{}'.format(i) for i in sorted(rxndata.keys())])),
                 'ReactionScaling {}\n'.format(' '.join(['{}'.format(rxnscaling.loc[list(rxnscaling.index)[-1], i]) for i in sorted(rxndata.keys())]))]
        with open(args.scale_file, 'a') as f:
            for l in lines:
                f.write(l)

        # Update the tracking
        rxn_cycle += 1
        add, delete, selected_rxn_types = {}, [], []

    # Write the resulting data and init files
    init = {
        'settings': args.settings,
        'prefix': args.prefix,
        'data': args.write_data,
        'thermo_freq': 1000,
        'avg_freq': 1000,
        'dump4avg': 100,
        'coords_freq': 1000,
        'units': args.lammps_units,
        'run_name': ['relax', 'equil', 'diffusion'],
        'run_type': ['nve/limit', 'npt', 'npt'],
        'run_steps': [args.relax, args.diffusion*10, args.diffusion],
        'run_temp': [[args.temp, args.temp, 10.0], [args.temp, args.temp, 100.0], [args.temp, args.temp, 100.0]],
        'run_press': [[args.press, args.press, 100.0], [args.press, args.press, 100.0], [args.press, args.press, 100.0]],
        'run_timestep': [0.25, 1.0, 1.0],
        'restart': False,
        'reset_steps': False,
        'thermo_keywords': ['temp', 'press', 'ke', 'pe']}
    init['pair_style'] = 'lj/cut 9.0'
    init['kspace_style'] = None
    init['angle_style'] = 'harmonic'
    init['dihedral_style'] = 'opls'
    init['improper_style'] = 'cvff'
    init['neigh_modify'] = 'every 1 delay 0 check yes one 10000'
    init['coords_freq'] = 20
    if not args.charged_atoms:
        init['atom_style'] = 'molecular'
    if args.lammps_units == 'lj':
        init['run_timestep'] = [0.001, 0.001, 0.001]
    write_lammps_data(args.write_data, atoms, bonds, angles, dihedrals,
                      impropers, box, header=data_header, charge=args.charged_atoms)
    write_lammps_init(init, args.write_init, step_restarts=False,
                      final_restart=False, final_data=True)

    return


if __name__ == '__main__':
    main(sys.argv[1:])

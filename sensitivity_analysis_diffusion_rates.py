#!/usr/bin/env python3
#
# Author:
#    Dylan Gilley
#    dgilley@purdue.edu


import sys, os, datetime
import numpy as np
import pandas as pd
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.diffusion import Diffusion,calculate_direct_voxel_transition_rates
from hybrid_mdmc.customargparse import HMDMC_ArgumentParser
from hybrid_mdmc.voxels import Voxels
from hybrid_mdmc.functions import gen_molecules


def main(argv):

    # Use HMDMC_ArgumentParser to parse the command line.
    parser = HMDMC_ArgumentParser(auto=False)
    parser.add_default_args()
    parser.add_argument('-diffusion_species', dest='diffusion_species', type=str, default='all',
                        help='Species for which to calculate diffusion rates. Default is "all".')
    parser.add_argument('-trajectory_frames', dest='trajectory_frames', default='0 -1 1', type=str)
    parser.add_argument('-voxel_transition_frames', dest='voxel_transition_frames', default=None, type=str,
                        help='Frames at which to calculate voxel transition rates. Default is to use only the last frame.')
    parser.add_argument('-calculate_diffusion_rates', dest='calculate_diffusion_rates', default=True, type=bool,
                        help='Calculate diffusion rates.')
    parser.add_argument('-maximum_walk_steps', dest='maximum_walk_steps', default='10 100 10', type=str,
                        help='Maximum number of walk steps to perform. Default is 10 100 10.')
    parser.add_argument('-number_of_walk_replicates', dest='number_of_walk_replicates', default=10, type=int,
                        help='Number of walk replicates to perform. Default is 10.')
    parser.add_argument('--no-calculate_voxel_transition_rates', dest='calculate_voxel_transition_rates', action="store_false",
                        help='Calculate voxel transition rates.')

    
    setattr(parser, 'args', parser.parse_args())
    if parser.args.filename_notebook == 'default':
        parser.args.filename_notebook = parser.args.system + '_notebook.xlsx'
    if os.path.isfile(parser.args.filename_notebook):
        parser.read_notebook()
    parser.adjust_default_args()
    args = parser.args
    args.diffusion_species = args.diffusion_species.split()
    trajectory_frames = [int(_) for _ in args.trajectory_frames.split()]
    voxel_transition_frames = []
    if args.voxel_transition_frames is not None:
        voxel_transition_frames = args.voxel_transition_frames.split()
    maximum_walk_steps = [int(i) for i in args.maximum_walk_steps.split()]
    number_of_walk_replicates = args.number_of_walk_replicates

    # Read in the data_file, diffusion_file, rxndf, and msf files.
    logfile = FileTracker(args.prefix+'.sadr_log.txt')
    datafile_result = parse_data_file(args.filename_data, unwrap=True, atom_style=args.atom_style)
    atoms, box = datafile_result[0],datafile_result[5]
    masterspecies = parser.get_masterspecies_dict()
    diffusion_species = [k for k in masterspecies.keys() if k in args.diffusion_species or args.diffusion_species == ['all']]

    # Create the Voxels object
    voxels_datafile = Voxels(box, args.number_of_voxels)
    atomtypes2moltype = {tuple(sorted([i[2] for i in v['Atoms']])):k for k,v in masterspecies.items()}

    # Create and populate an instance of the "MoleculeList" class.
    molecules = gen_molecules(atoms, atomtypes2moltype, voxels_datafile)

    # Initialize Diffusion and calculate rate for each of the requested species
    diffusion = Diffusion(
        args.prefix,
        args.filename_trajectory,
        atoms,
        molecules,
        voxels_datafile,
        time_conversion=args.lammps_time_units_to_seconds_conversion)
    
    # If requested, calculate the voxel transition rates.
    if args.calculate_voxel_transition_rates is True:
        logfile.write(f'voxel transition rates requested\n')
        vtfile = {species: FileTracker(args.prefix+f'.voxel_transition_rates.{species}.running.txt') for species in diffusion_species}
        vtfile_final = {species: FileTracker(args.prefix+f'.voxel_transition_rates.{species}.final.txt')for species in diffusion_species}
        logfile.write(f'  {datetime.datetime.now()} -- Parsing trajectory file...\n')
        diffusion.parse_trajectory_file(start=trajectory_frames[0], end=trajectory_frames[1], every=trajectory_frames[2], return_timesteps=True)
        if args.voxel_transition_frames is None:
            voxel_transition_frames = [diffusion.voxels_by_frame_array.shape[0],diffusion.voxels_by_frame_array.shape[0],1]
        logfile.write(f'  {datetime.datetime.now()} -- Calculating direct voxel transition rates...\n')
        for maximum_frame in np.arange(int(voxel_transition_frames[0]),int(voxel_transition_frames[1]),int(voxel_transition_frames[2])):
            logfile.write(f'    {datetime.datetime.now()} -- frame {diffusion.timesteps[maximum_frame]}...\n')
            voxels_by_frame = diffusion.voxels_by_frame_array[:maximum_frame]
            direct_voxel_transition_rates = {
                k: pd.DataFrame(v) for k,v in 
                calculate_direct_voxel_transition_rates(
                    np.prod(diffusion.number_of_voxels),
                    voxels_by_frame,
                    maximum_frame*diffusion.time_conversion,
                    diffusion.molecules_datafile).items()}
            for species in diffusion_species:
                vtfile[species].write(f'\nFrame {diffusion.timesteps[maximum_frame]}\n')
                vtfile[species].write(f'{direct_voxel_transition_rates[species].to_string(header=False,index=False)}\n')
        vtfile_final[species].write(f'\nFrame {diffusion.timesteps[maximum_frame]}\n')
        vtfile_final[species].write(f'{direct_voxel_transition_rates[species].to_string(header=False,index=False)}\n')
    
    # If requested, calculate the diffusion rates.
    if args.calculate_diffusion_rates is True:
        logfile.write(f'\ndiffusion rates requested\n')
        diffusion_rates_files = {species: FileTracker(args.prefix+f'.diffusion_rates.{species}.txt') for species in diffusion_species}
        if not hasattr(diffusion, 'direct_voxel_transition_rates'):
            logfile.write(f'  {datetime.datetime.now()} -- attempting to read in direct voxel transition rates...\n')
            try:
                dvtr = {
                    species: read_direct_voxel_transition_rates(args.prefix+f'.voxel_transition_rates.{species}.final.txt')
                    for species in diffusion_species}
                setattr(diffusion, "direct_voxel_transition_rates", {species: v[list(v.keys())[0]] for species,v in dvtr.items()})
            except:
                logfile.write(f'  {datetime.datetime.now()} -- failed to read in direct voxel transition rates. Exiting.\n')
                return
        for species in args.diffusion_species:
            logfile.write(f'  {datetime.datetime.now()} -- Calculating diffusion rates for species {species}...\n')
            for walk_steps in np.arange(maximum_walk_steps[0],maximum_walk_steps[1],maximum_walk_steps[2]):
                diffusion_rates_files[species].write(f'\n~Total walk steps {walk_steps}~\n')
                for replicate in range(1,number_of_walk_replicates+1):
                    logfile.write(f'    {datetime.datetime.now()} -- Performing random walk {replicate}/{number_of_walk_replicates} for {walk_steps} steps...\n')
                    diffusion.perform_random_walks(number_of_steps=walk_steps,species=species)
                    diffusion.calculate_average_first_time_between_positions(species=species)
                    diffusion.calculate_diffusion_rates(species=species)
                    diffusion_rate_dataframe = pd.DataFrame(diffusion.diffusion_rates[species])
                    diffusion_rates_files[species].write(f'\n~Walk {replicate}~\n')
                    diffusion_rates_files[species].write(f'{diffusion_rate_dataframe.to_string(header=False,index=False)}\n')
    return


class FileTracker:

    def __init__(self, name):
        self.name = name
        with open(self.name,'w') as f:
            f.write(f'File created {datetime.datetime.now()}\n\n')
        return
    
    def write(self, string):
        with open(self.name,'a') as f:
            f.write(string)
        return


def read_direct_voxel_transition_rates(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    direct_voxel_transition_rates = {}
    parsing = False
    for line in lines:
        if line.split() == []:
            continue
        if 'Frame' in line:
            frame = int(line.split()[1])
            direct_voxel_transition_rates[frame] = []
            parsing = True
        elif parsing is True:
            direct_voxel_transition_rates[frame].append([float(i) for i in line.split()])
    return {frame: np.array(v) for frame, v in direct_voxel_transition_rates.items()}


if __name__ == '__main__':
    main(sys.argv)

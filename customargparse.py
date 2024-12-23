#!/usr/bin/env python3
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu

import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from hybrid_mdmc.parsers import *

class HMDMC_ArgumentParser(ArgumentParser):

    def __init__(self, auto=True):
        ArgumentParser.__init__(self)
        if auto is True:
            self.add_default_args()
            self.args = self.parse_args()
            if self.args.filename_notebook == 'default':
                self.args.filename_notebook = self.args.system + '_notebook.xlsx'
            if os.path.isfile(self.args.filename_notebook):
                self.read_notebook()
            self.adjust_default_args()
        return

    def add_default_args(self):

        # Positional arguments
        self.add_argument(dest='system', type=str,
                            help='Name of the system')
        self.add_argument(dest='prefix', type=str,
                            help='Prefix of the system')

        # Optional arguments - file names
        self.add_argument('-filename_notebook', dest='filename_notebook', type=str, default='default',
                            help='Name of the excel notebook containing the simulation parameters')
        self.add_argument('-filename_data', dest='filename_data', type=str, default='default',
                            help='Name of the LAMMPS data file to read in.')
        self.add_argument('-filename_trajectory', dest='filename_trajectory', type=str, default='default',
                            help='Name of the lammps trajectory file. If not provided, the prefix is prepended to ".lammpstrj"')
        self.add_argument('-filename_concentration', dest='filename_concentration', type=str, default='default',
                            help='Name of the concentration file. If not provided, the prefix is prepended to ".concentration"')
        self.add_argument('-filename_log', dest='filename_log', type=str, default='default',
                            help='Name of the log file. If not provided, the prefix is prepended to ".log"')
        self.add_argument('-filename_scale', dest='filename_scale', type=str, default='default',
                            help='Name of the scale file. If not provided, the prefix is prepended to ".scale"')
        self.add_argument('-filename_diffusion', dest='filename_diffusion', type=str, default='default',
                            help='Name of the diffusion file. If not provided, the prefix is prepended to ".diffusion"')
        self.add_argument('-filename_settings', dest='filename_settings', type=str, default='default',
                            help='Name of the filename_settings file for LAMMPS MD run. If not provided, the prefix is preprended to ".in.filename_settings"')
        self.add_argument('-filename_writedata', dest='filename_writedata', type=str, default='default',
                            help='Name of the data file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.data"')
        self.add_argument('-filename_writeinit', dest='filename_writeinit', type=str, default='default',
                            help='Name of the init file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.init"')
        self.add_argument('-filename_header', dest='filename_header', type=str, default='default',
                            help='Name of the data filename_header file. If not provided, the prefix is prepended to ".filename_header"')
        self.add_argument('-filename_rxndf', dest='filename_rxndf', type=str, default='default',
                            help='Name of the filename_rxndf file. If not provided, the prefix is preprended to ".filename_rxndf"')
        self.add_argument('-filename_msf', dest='filename_msf', type=str, default='default',
                            help='Name of the filename_msf file. If not provided, the prefix is preprended to ".filename_msf"')

        # Optional arguments - MD information
        self.add_argument('-atom_style', dest='atom_style', default='full',
                            help='LAMMPS atom style. Default: full')
        self.add_argument('-lammps_units', dest='lammps_units', default='real',
                          help='LAMMPS units. Default: real')
        self.add_argument('-temperature_MD', dest='temperature_MD', default=298.0,
                            help='Temperature of the simulation (K). Default: 298.0 K')
        self.add_argument('-press', dest='press', default=1.0,
                            help='Pressure of the simulation (K). Default: 1.0')
        self.add_argument('-relax', dest='relax', default=1.0e3,
                            help='Length of the relaxation nve/lim run. Default: 1.0e3 (fs)')
        self.add_argument('-diffusion', dest='diffusion', default=1.0e4,
                            help='Length of the diffusion npt run. Default: 1.0e4 (fs)')
        
        # Optional arguments - diffusion information
        self.add_argument('-number_of_voxels', dest='number_of_voxels', type=str, default='3 3 3',
                            help='Space delimited string of the number of voxels in the x, y, and z directions. Default: 3 3 3')
        self.add_argument('-x_bounds', dest='x_bounds', type=str, default='',
                            help='Space delimited string defining custom bounds for the x-axis. Overrides first entry in -number_of_voxels')
        self.add_argument('-y_bounds', dest='y_bounds', type=str, default='',
                            help='Space delimited string defining custom bounds for the y-axis. Overrides second entry in -number_of_voxels')
        self.add_argument('-z_bounds', dest='z_bounds', type=str, default='',
                            help='Space delimited string defining custom bounds for the z-axis. Overrides third entry in -number_of_voxels')

        # Optional arguments - KMC parameters
        self.add_argument('-temperature_rxn', dest='temperature_rxn', default=298.0,
                            help='Temperature of the simulation (K). Default: 298.0 K')
        self.add_argument('-change_threshold', dest='change_threshold', default=0.10,
                            help='Fraction of original molecules to react before this reactive KMC cycle completes.')
        self.add_argument('-diffusion_cutoff', dest='diffusion_cutoff', default=0.4,
                            help='Minimum diffusion coefficient to consider for possible reaction between voxels.')
        self.add_argument('-kmc_type', dest='kmc_type', default='rejection_free',
                            help='Type of KMC to perform in each voxel.')
        self.add_argument('-diffusion_step', dest='diffusion_step', default=0,
                            help='Diffusion step. Default: 0.')
        self.add_argument('-scalerates', dest='scalerates', default='cumulative',
                            help='Option for scaling reaction rates when species mole fractions go stagnant. Options: cumulative, static, off.')
        self.add_argument('-scaling_criteria_rollingmean_stddev', dest='scaling_criteria_rollingmean_stddev', default=0.1,
                            help='Maximum (less than or equal to) standard deviation of the rolling mean of the number fraction of a molecular species'+\
                            'for that number fraction to be considered stagnant.')
        self.add_argument('-scaling_criteria_rollingmean_cycles', dest='scaling_criteria_rollingmean_cycles', default=3,
                            help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')
        self.add_argument('-scaling_criteria_concentration_slope', dest='scaling_criteria_concentration_slope', default=0.1,
                            help='Maximum (less than or equal to) slope of the number fraction of a molecular species for that number fraction to be considered stagnant.')
        self.add_argument('-scaling_criteria_concentration_cycles', dest='scaling_criteria_concentration_cycles', default=1,
                            help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')
        self.add_argument('-scaling_criteria_rxnselection_count', dest='scaling_criteria_rxnselection_count', default=1,
                            help='Minimum number of times (greater than or equal to) that a reaciton must be selected in the previous _ steps in order ot be a candidate for scaling.')
        self.add_argument('-windowsize_rollingmean', dest='windowsize_rollingmean', default=3,
                            help='Window size for the calculation of the rolling mean, measured in MDMC cycles.')
        self.add_argument('-windowsize_slope', dest='windowsize_slope', default=5,
                            help='Window size for the calculation of the slope of concentration, measured in MDMC cycles.')
        self.add_argument('-windowsize_scalingpause', dest='windowsize_scalingpause', default=3,
                            help='Number of MDMC cycles after a reaciton is scaled or unscaled before it can be scaled again.')
        self.add_argument('-windowsize_rxnselection', dest='windowsize_rxnselection', default=10,
                            help='Window size for checking the number of times a reaction has been selected.')
        self.add_argument('-scalingfactor_adjuster', dest='scalingfactor_adjuster', default=0.1,
                            help='Quantity which a reaction rate is multiplied by to scale that rate.')
        self.add_argument('-scalingfactor_minimum', dest='scalingfactor_minimum', default=1e-6,
                            help='Minimum reaction rate scaling factor for all reactions.')
        self.add_argument('--charged_atoms', action='store_true')
        self.add_argument('--no-charged_atoms', dest='charged_atoms', action='store_false')
        self.set_defaults(charged_atoms=True)

        # Optional arguments - flags
        self.add_argument('--distance_criteria', dest='distance_criteria', default=False, action='store_const', const=True)
        self.add_argument('--well_mixed', dest='well_mixed', default=False, action='store_const', const=True)
        self.add_argument('--debug', dest='debug', default=False, action='store_const', const=True)
        self.add_argument('--log', dest='log', default=False, action='store_const', const=True)

    def read_notebook(self):
        notebook = pd.read_excel(self.args.filename_notebook,sheet_name=None,index_col=None,header=None)

        # System
        for index, row in notebook['System'].iterrows():
            if row[1] == '-': continue
            if row[1] == 'false':
                row[1] = False
            if row[1] == 'true':
                row[1] = True
            self.args.__dict__[row[0]] = row[1]

        # Header
        setattr(self, 'header', {'masses': []})
        for index, row in notebook['Header'].iterrows():
            if 'types' in row[0]:
                self.header['_'.join(row[0].split())] = int(row[1])
            elif 'mass' in row[0]:
                self.header['masses'].append(tuple([int(row[0].split()[1]), float(row[1])]))

        # Species
        setattr(self, 'starting_species', {})
        setattr(self, 'masterspecies', {})
        notebook['Species'].dropna(how='all', inplace=True)
        category_series = notebook['Species'].iloc[:,0]
        category_series.ffill(inplace=True)
        categories = [(index,value,notebook['Species'].loc[index,1]) for index, value in category_series.items()]
        msf_categories = ['Atoms','Bonds','Angles','Dihedrals','Impropers']
        row_map = {key:{
                    v[2]:v[0] for v in categories if v[1] == key}
                        for key in msf_categories}
        species = None
        for column_idx in notebook['Species'].iloc[:,2:].columns:
            column = notebook['Species'].loc[:,column_idx]
            if not pd.isna(column[0]):
                species = column[0]
                self.masterspecies[species] = {_:[] for _ in msf_categories}
                self.starting_species[species] = int(column[2])
            # atoms
            if not pd.isna(column[row_map['Atoms']['ID']]):
                self.masterspecies[species]['Atoms'].append([
                    int(column[row_map['Atoms']['ID']]),
                    0,
                    int(column[row_map['Atoms']['type']]),
                    float(column[row_map['Atoms']['charge']]),
                    float(column[row_map['Atoms']['x']]),
                    float(column[row_map['Atoms']['y']]),
                    float(column[row_map['Atoms']['z']]),
                ])
            # interactions
            for interaction in msf_categories[1:]:
                if not pd.isna(column[row_map[interaction]['ID']]):
                    list_ = [
                        int(column[row_map[interaction]['ID']]),
                        int(column[row_map[interaction]['type']]),
                        int(column[row_map[interaction]['i']]),
                        int(column[row_map[interaction]['j']])]
                    if 'k' in row_map[interaction].keys():
                        list_.append(int(column[row_map[interaction]['k']]))
                    if 'l' in row_map[interaction].keys():
                        list_.append(int(column[row_map[interaction]['l']]))
                    self.masterspecies[species][interaction].append(list_)

        # Reactions
        setattr(self, 'reaction_data', {})
        row_map = {val:idx for idx,val in notebook['Reactions'].iloc[:,0].items()}
        self.reaction_data = {
            int(column[row_map['reaction ID']]): {
                'reactant_molecules': column[row_map['reactant(s)']].split(','),
                'product_molecules': column[row_map['product(s)']].split(','),
                'A': column[row_map['A (1/s)']],
                'b': column[row_map['b']],
                'Ea': column[row_map['Ea (kcal/mol)']],
            }
            for idx,column in notebook['Reactions'].iloc[:,1:].items()
        }

        # Initial MD
        setattr(self, 'initial_MD_init_dict', {})
        for index,row in notebook['Initial MD'].iterrows():
            if 'run' in row[0]:
                self.initial_MD_init_dict[row[0]] = list(row[1:])
            else:
                self.initial_MD_init_dict[row[0]] = row[1]
        
        # Cycled MD
        setattr(self, 'cycled_MD_init_dict', {})
        for index,row in notebook['Cycled MD'].iterrows():
            if 'run' in row[0]:
                self.cycled_MD_init_dict[row[0]] = list(row[1:])
            else:
                self.cycled_MD_init_dict[row[0]] = row[1]

        return

    def adjust_default_args(self):
        num_starting_args = len(self.args.__dict__)

        # Handle file names
        if self.args.filename_notebook == 'default':
            self.args.filename_notebook = self.args.system + '.xlsx'
        if self.args.filename_data == 'default':
            self.args.filename_data = self.args.prefix + '.end.data'
        if self.args.filename_rxndf == 'default':
            self.args.filename_rxndf = self.args.system+'.rxndf'
        if self.args.filename_msf == 'default':
            self.args.filename_msf = self.args.system+'.msf'
        if self.args.filename_settings == 'default':
            self.args.filename_settings = self.args.system+'.in.settings'
        if self.args.filename_header == 'default':
            self.args.filename_header = self.args.system+'.header'
        if self.args.filename_writedata == 'default':
            self.args.filename_writedata = self.args.prefix+'.in.data'
        if self.args.filename_writeinit == 'default':
            self.args.filename_writeinit = self.args.prefix+'.in.init'
        if self.args.filename_trajectory == 'default':
            self.args.filename_trajectory = self.args.prefix+'.diffusion.lammpstrj'
        if self.args.filename_concentration == 'default':
            self.args.filename_concentration = self.args.prefix+'.concentration'
        if self.args.filename_log == 'default':
            self.args.filename_log = self.args.prefix+'.log'
        if self.args.filename_scale == 'default':
            self.args.filename_scale = self.args.prefix+'.scale'
        if self.args.filename_diffusion == 'default':
            self.args.filename_diffusion = self.args.prefix+'.diffusion'

        # Handle MD information
        self.args.temperature_rxn = float(self.args.temperature_rxn)
        self.args.temperature_MD = float(self.args.temperature_MD)
        self.args.press = float(self.args.press)
        self.args.relax = float(self.args.relax)
        self.args.diffusion = float(self.args.diffusion)

        # Handle diffusion information
        self.args.number_of_voxels = [int(float(_)) for _ in self.args.number_of_voxels.split()]
        self.args.x_bounds = [int(float(_)) for _ in self.args.x_bounds.split()]
        self.args.y_bounds = [int(float(_)) for _ in self.args.y_bounds.split()]
        self.args.z_bounds = [int(float(_)) for _ in self.args.z_bounds.split()]

        # Handle KMC parameters
        self.args.change_threshold = float(self.args.change_threshold)
        self.args.diffusion_cutoff = float(self.args.diffusion_cutoff)
        self.args.kmc_type = str(self.args.kmc_type)
        self.args.diffusion_step = int(float(self.args.diffusion_step))
        self.args.scalerates = str(self.args.scalerates)
        self.args.scaling_criteria_rollingmean_stddev = float(self.args.scaling_criteria_rollingmean_stddev)
        self.args.scaling_criteria_rollingmean_cycles = int(float(self.args.scaling_criteria_rollingmean_cycles))
        self.args.scaling_criteria_concentration_slope = float(self.args.scaling_criteria_concentration_slope)
        self.args.scaling_criteria_concentration_cycles = int(float(self.args.scaling_criteria_concentration_cycles))
        self.args.scaling_criteria_rxnselection_count = int(float(self.args.scaling_criteria_rxnselection_count))
        self.args.windowsize_rollingmean = int(float(self.args.windowsize_rollingmean))
        self.args.windowsize_slope = int(float(self.args.windowsize_slope))
        self.args.windowsize_scalingpause = int(float(self.args.windowsize_scalingpause))
        self.args.windowsize_rxnselection = int(float(self.args.windowsize_rxnselection))
        self.args.scalingfactor_adjuster = float(self.args.scalingfactor_adjuster)
        self.args.scalingfactor_minimum = float(self.args.scalingfactor_minimum)

        # Check that all arguments were appropriately handled
        if len(self.args.__dict__) != num_starting_args:
            print('Error! The number of arguments changed during argument adjustment.'+\
                '\nPlease check hybrid_mdmc.customargparse.HMDMC_ArgumentParser.adjust_default_args()'+\
                '\nExiting...')
            quit()

    def get_reaction_data_dict(self):
        if hasattr(self, 'reaction_data'):
            return self.reaction_data
        return parse_rxndf(self.args.filename_rxndf)
    
    def get_masterspecies_dict(self):
        if hasattr(self, 'masterspecies'):
            return self.masterspecies
        return parse_msf(self.args.filename_msf)
    
    def get_data_header_dict(self):
        if hasattr(self, 'header'):
            return self.header
        return parse_header(self.args.filename_header)

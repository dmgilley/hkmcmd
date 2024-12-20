#!/usr/bin/env python3
# Author
#   Dylan M. Gilley
#   dgilley@purdue.edu


import argparse,sys


# Main function
def main(argv):
    """Write the bash file for initializing and running a new hybridmdmc Toy Problem.
    """

    # Create the parser
    parser = argparse.ArgumentParser()

    # Positional argument(s)
    parser.add_argument(dest='system', type=str,
                        help='System name (str).')
    parser.add_argument(dest='prefix', type=str,
                        help='Prefix (str).')

    # Optional arguments
    parser.add_argument('-filename_notebook', dest='filename_notebook', type=str, default='default',
                         help='Name of the parameters excel file (str). Default: default')
    parser.add_argument('-queue', dest='queue', type=str, default='bsavoie',
                        help='Queue on which to run (str). Defualt: bsavoie')
    parser.add_argument('-nodes', dest='nodes', type=int, default=1,
                       help='Number of nodes on which to run (int). Default: 1')
    parser.add_argument('-cores', dest='cores', type=int, default=16,
                        help='Number of cores on which to run (int). Defualt: 16')
    parser.add_argument('-timelim', dest='timelim', type=str, default='06:00:00',
                        help='Time limit of run (str). Default: 03:00:00')
    parser.add_argument('-reactive_loops', dest='reactive_loops', type=int, default=2000,
                        help='Number of reactive loops.')
    #parser.add_argument('--serial', action='store_true')
    #parser.add_argument('--synchronousparallel', dest='serial', action='store_false')
    #parser.set_defaults(serial=True)
    
    # Parse the line arguments
    args = parser.parse_args()
    
    mainscript = '~/bin/hybrid_mdmc/hybridmdmc.py'
    #if args.serial:
    #    mainscript = '~/bin/hybrid_mdmc/hybridmdmc_serial.py'

    # Write the bash file
    with open(args.prefix+'_hmdmc.sh', 'w') as f:
        f.write(
            """\
#!/bin/bash
#
#SBATCH --job-name {}
#SBATCH -o {}.slurm.out
#SBATCH -e {}.slurm.err
#SBATCH -A {}
#SBATCH -N {}
#SBATCH -n {}
#SBATCH -t {}

# Adjust modules
module purge
module load gcc/12.2.0
module load openmpi/4.1.4
module load lammps/20220623

# Write out job information
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $SLURM_NODELIST"
echo "Running on node(s): $SLURM_NNODES"
echo "Number of processors: $SLURM_NPROCS"
echo "Current working directory: $SLURM_SUBMIT_DIR"
scontrol -dd show job $SLURM_JOB_ID | grep "CPU_IDs"
            
# User supplied shell commands
cd $SLURM_SUBMIT_DIR

# Run script
echo "Start time: $(date)"

# System prep
python ~/bin/hybrid_mdmc/gen_initial_hybridmdmc.py {} {} -filename_notebook {}
mpirun -n {} lmp -in {}.in.init > {}.lammps.out
cp {}.in.init               {}_prep.in.init
cp {}.in.data               {}_prep.in.data
cp {}.end.data              {}_prep.end.data
cp {}.lammps.out            {}_prep.lammps.out
cp {}.lammps.log            {}_prep.lammps.log
cp {}.thermo.avg            {}_prep.thermo.avg
cp {}.shrink.lammpstrj      {}_prep.shrink.lammpstrj
cp {}.diffusion.lammpstrj   {}_prep.diffusion.lammpstrj

# Reactive loop
for i in `seq 0 {}`; do

    echo "Loop step ${{i}} ($(date)) "


    # Calculate MSD
    echo "  calculating msd ($(date)) ..."
    python ~/bin/hybrid_mdmc/calculate_MSD.py {} 'A' -filename_notebook '{}_notebook.xlsx' -frames '500 10 1000' -filename_output {}.msdoutput.${{i}}.txt
    retVal=$?
    if [ $retVal -ne 0 ]; then
        exit $retVal
    fi


            
    # Run RMD script
    echo "  running hybridmdmc ($(date)) ..."
    python {} {} {} -filename_notebook {} -diffusion_step ${{i}}
    retVal=$?
    if [ $retVal -ne 0 ]; then
        exit $retVal
    fi

    # Run MD
    echo "  running MD ($(date)) ..."
    mpirun -n {} lmp -in {}.in.init > {}.lammps.out
    retVal=$?
    if [ $retVal -ne 0 ]; then
        exit $retVal
    fi

    echo ""

done

echo "End time: $(date)"
""".format(
    args.prefix+'_hmdmc', args.prefix+'_hmdmc', args.prefix+'_hmdmc', args.queue, args.nodes, args.cores, args.timelim,
    args.system, args.prefix, args.filename_notebook,
    args.cores, args.prefix, args.prefix,
    args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix,
    args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix,
    args.reactive_loops,
    args.prefix, args.system, args.prefix,
    mainscript, args.system, args.prefix, args.filename_notebook,
    args.cores, args.prefix, args.prefix,
))

    return


if __name__ == '__main__':
    main(sys.argv[1:])

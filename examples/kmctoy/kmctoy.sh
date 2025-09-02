#!/bin/bash

# Run script
echo "Start time: $(date)"
echo ""

# System prep
python ~/bin/hybrid_mdmc/initialize_hkmcmd_system.py kmctoy kmctoy
echo "running initial MD ($(date)) ..."
echo ""
mpirun -n 8 lmp -in kmctoy.in.init > kmctoy.lammps.out
return_value=$?
if [ $return_value -ne 0 ]; then
    exit $return_value
fi
cp kmctoy.in.init               kmctoy_prep.in.init
cp kmctoy.in.data               kmctoy_prep.in.data
cp kmctoy.end.data              kmctoy_prep.end.data
cp kmctoy.lammps.out            kmctoy_prep.lammps.out
cp kmctoy.lammps.log            kmctoy_prep.lammps.log
cp kmctoy.thermo.avg            kmctoy_prep.thermo.avg
cp kmctoy.shrink.lammpstrj      kmctoy_prep.shrink.lammpstrj
cp kmctoy.diffusion.lammpstrj   kmctoy_prep.diffusion.lammpstrj

# Reactive loop
for i in `seq 1 200`; do

    echo "Loop step ${i} ($(date)) "

    # Run RMD script
    echo "  running hybridmdmc ($(date)) ..."
    python ~/bin/hybrid_mdmc/run_hkmcmd.py kmctoy kmctoy -diffusion_cycle ${i}
    return_value=$?
    if [ $return_value -ne 0 ]; then
        exit $return_value
    fi

    # Run MD
    echo "  running MD ($(date)) ..."
    mpirun -n 8 lmp -in kmctoy.in.init > kmctoy.lammps.out
    return_value=$?
    if [ $return_value -ne 0 ]; then
        exit $return_value
    fi

    echo ""

done

echo "End time: $(date)"

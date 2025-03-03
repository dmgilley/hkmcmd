

python3 ../../initialize_hkmcmd_system.py fiction fiction-1

for step in 1 2 3 4; do
    
    cp fiction-1.in.data fiction-1.end.data
    python3 ../../run_hkmcmd.py fiction fiction-1 -diffusion_cycle ${step}

done

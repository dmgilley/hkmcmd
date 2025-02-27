

To Test - Level 1
---------------------------------------------------------------------------------------------------


A. utility.py

   i. unwrap_coordinates() -> checked 24Feb2025

   ii. wrap_coordinates() -> checked 23Feb2025


B. particle_interactions.pu

   i. Atom
      1. wrap() -> checked 23Feb2025
      2. make_jsonable() -> checked 23Feb2025
      
   ii. IntraMode
      1. make_jsonable() -> checked 23Feb2025

   iii. Molecule
      1. check_for_json_inputs() -> checked 23Feb2025
      2. fill_lists() -> checked 24Feb2025
      3. unwrapatomic_coordinates() -> checked 24Feb2025
      4. calculate_cog() -> checked 24Feb2025
      5. assign_voxel_idxs_tuple() -> checked 24Feb2025
      6. translate_molecule() -> checked 24Feb2025
      7. atom_IDs @property -> checked 23Feb2025
      8. make_jsonable() -> checked 23Feb2025

   iv. Reaction
      1. check_for_json_inputs() -> checked 23Feb2025
      2. calculate_product_positions() -> checked 24Feb2025
      3. create_product_molecules() -> checked 24Feb2025
      4. calculate_rawrate() -> checked 24Feb2025
      5. make_jsonable() -> checked 23Feb2025

   v. unwrap_atoms_list() -> checked 24Feb2025


C. filehandlers_lammps.py

   i. write_lammps_data() -> checked 24Feb2025

   ii. LammpsInitHandler -> checked 24Feb2025
      1. generate_run_lines() -> checked 24Feb2025
      2. write() -> checked 24Feb2025


D. filehandlers_general.py

   i. frame_generator() -> checked 24Feb2025
   
   ii. parse_data_file() -> checked 24Feb2025

   iii. parse_diffusion_file() -> checked


E. voxels.py -> checked


F. kmc.py -> checked 24Feb2025


G. create_initial_lammps_files.py -> checked 25Feb2025


To Test - Level 2
---------------------------------------------------------------------------------------------------


A. hybridKMCMD.py


B. functions.py


C. filehandlers_general.py
   
   i. parse_system_state_file()

   ii. parse_reaction_file()

   iii. hkmcmd_ArgumentParser


To Test - Level 3
---------------------------------------------------------------------------------------------------


A. mean_displacement.py


B. diffusion.py



Other
---------------------------------------------------------------------------------------------------
concatenate_files.py
pure_kmc.py
sensitivity_analysis_diffusion_rates.py
write_hybridmdmc_bash.py

#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import unittest
import numpy as np
from copy import deepcopy
from hybrid_mdmc.system import *
from hybrid_mdmc.filehandlers import parse_data_file
from hybrid_mdmc.calculate_mean_displacement import *


class TestSystemState(unittest.TestCase):

    def setUp(self):

        # Read system data
        system_data = SystemData("msd_fiction", "msd_fiction", filename_json="msd_fiction.json")
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
        ) = parse_data_file(
            "msd_fiction.end.data",
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
        system_state = SystemState(filename="msd_fiction.system_state.json")
        system_state.read_data_from_json()

        # Check for consistency between SystemState and Datafile atoms
        system_state.check_atoms_list(atoms_list)

        # Create molecules_list
        self.molecules_list = [
            Molecule(ID=ID)
            for ID in sorted(list(set([atom.molecule_ID for atom in atoms_list])))
        ]
        for molecule in self.molecules_list:
            molecule.fill_lists(
                atoms_list=atoms_list,
                bonds_list=bonds_list,
                angles_list=angles_list,
                dihedrals_list=dihedrals_list,
                impropers_list=impropers_list,
            )
            molecule.kind = system_state.get_molecule_kind(molecule.ID)

        return

    def test_assemble_molecular_cog(self):

        # Create molecular cog array
        (
            molecular_cog_array,
            molecule_types,
            frames,
            box,
        ) = assemble_molecular_cog(
            "msd_fiction.diffusion.lammpstrj",
            self.molecules_list,
            start = 0,
            end = -1,
            every = 1,
        )

        setattr(self, "molecular_cog_array", molecular_cog_array)
        setattr(self, "molecule_types", molecule_types)
        setattr(self, "frames", frames)
        setattr(self, "box", box)

        self.assertEqual(
            molecular_cog_array.tolist(),
            np.array([
                [
                    [ -2.5, -2.5,  -2.5, ],
                    [  0.0,  0.0,   0.0, ],
                ],
                [
                    [  4.9,  4.9,   4.9, ],
                    [  3.0,  0.0,   0.0, ],
                ],
                [
                    [  2.5, -2.5,   2.5, ],
                    [  3.0,  3.0,   0.0, ],
                ],
                [
                    [  3.5, -2.5,   2.5, ],
                    [  3.0,  3.5, -0.75, ],
                ]
            ]).tolist()
        )
        self.assertEqual(
            molecule_types,
            ["A", "A2"]
        )
        self.assertEqual(
            frames,
            [20,30,40,50]
        )
        self.assertEqual(
            box,
            [ [-5.25, 5.25], [-5.25, 5.25], [-5.25, 5.25] ]
        )
        return

    def test_calculate_mean_displacement(self):

        # Create molecular cog array
        (
            molecular_cog_array,
            molecule_types,
            frames,
            box,
        ) = assemble_molecular_cog(
            "msd_fiction.diffusion.lammpstrj",
            self.molecules_list,
            start = 0,
            end = -1,
            every = 1,
        )

        (
        mean_displacements_avg,
        mean_displacements_std,
        mean_squared_displacements_avg,
        mean_squared_displacements_std,
        ) = calculate_msd(molecular_cog_array, molecule_types, len(frames), box)

        ref = {'A': [3.655365159096, 5.616999735582603, 6.726812023536855], 'A2': [2.3004626062886655, 3.9110479764691863, 4.670385423067351]}
        here = mean_displacements_avg
        for sp in ["A","A2"]:
            for idx in range(3):
                self.assertEqual(here[sp][idx], ref[sp][idx])

        ref = {'A': [1.9039359811542345, 1.4540680762828733, 0.0], 'A2': [0.9892952695736963, 0.3315927106500982, 0.0]}
        here = mean_displacements_std
        for sp in ["A","A2"]:
            for idx in range(3):
                self.assertEqual(here[sp][idx], ref[sp][idx])

        ref = {'A': [16.986666666666665, 33.665, 45.25], 'A2': [6.270833333333333, 15.406249999999998, 21.812500000000004]}
        here = mean_squared_displacements_avg
        for sp in ["A","A2"]:
            for idx in range(3):
                self.assertEqual(here[sp][idx], ref[sp][idx])

        ref = {'A': [11.733218749440502, 16.335000000000008, 0.0], 'A2': [3.859624513976572, 2.5937499999999982, 0.0]}
        here = mean_squared_displacements_std
        for sp in ["A","A2"]:
            for idx in range(3):
                self.assertEqual(here[sp][idx], ref[sp][idx])


        return


if __name__ == "__main__":
    unittest.main()

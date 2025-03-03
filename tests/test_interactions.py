#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import unittest
import numpy as np
from copy import deepcopy
from hybrid_mdmc.system import SystemData
from hybrid_mdmc.voxels import Voxels
from hybrid_mdmc.filehandlers import parse_data_file
from hybrid_mdmc.interactions import *


class TestAtom(unittest.TestCase):

    atom = Atom(ID=1, x=0.0, y=0.0, z=0.0)

    def test_init(self):
        with self.assertRaises(ValueError):
            Atom(ID=-1)
        return

    def test_wrap(self):
        self.atom.wrap(
            [
                [-3, 3],  # -> 0.0
                [4.225, 7.465],  # -> 6.48, multiple wraps
                [0.838, 10.325],  # -> 9.487
            ]
        )
        self.assertEqual(self.atom.x, 0.0)
        self.assertEqual(self.atom.y, 6.48)
        self.assertEqual(self.atom.z, 9.487)
        return


class TestMolecule(unittest.TestCase):

    system_data = SystemData("combustion", "combustion")
    system_data.read_json()
    system_data.clean()
    methanol, oxygen, water, carbon_dioxide = system_data.species
    (
        atoms_list,
        bonds_list,
        angles_list,
        dihedrals_list,
        impropers_list,
        box,
        _,
    ) = parse_data_file(
        "combustion.in.data",
        atom_style="full",
    )

    def test_creation(self):
        """Tests fill_lists and atom_IDs"""
        molecule = Molecule(ID=3)  # pull a methanol
        molecule.fill_lists(
            atoms_list=self.atoms_list,
            bonds_list=self.bonds_list,
            angles_list=self.angles_list,
            dihedrals_list=self.dihedrals_list,
            impropers_list=self.impropers_list,
        )
        self.assertEqual(molecule.atom_IDs, [10, 11, 12, 13, 14, 15])
        self.assertEqual(molecule.atoms[0].x, -0.03416307)
        self.assertEqual([bond.ID for bond in molecule.bonds], [8, 9, 10, 11, 12])
        self.assertEqual(
            [angle.ID for angle in molecule.angles], [9, 10, 11, 12, 13, 14, 15]
        )
        self.assertEqual([dihedral.ID for dihedral in molecule.dihedrals], [4, 5, 6])
        self.assertEqual([improper.ID for improper in molecule.impropers], [4, 5, 6])
        return

    def test_spatial_changes(self):
        """Tests unwrap_atomic_coordinates, calculate_cog, assign_voxel_idx, translate, and rotate"""
        atom1 = Atom(ID=1, x=0.0, y=0.0, z=0.0)
        atom2 = Atom(ID=2, x=-2.0, y=-2.0, z=-2.0)
        atom3 = Atom(ID=3, x=2.0, y=2.0, z=2.0)
        molecule = Molecule(ID=1, atoms=[atom1, atom2, atom3])

        # unwrap to no effect
        box = [[-3, 3], [-3, 3], [-3, 3]]
        molecule.unwrap_atomic_coordinates(box)
        self.assertEqual([atom1.x, atom2.x, atom3.x], [0.0, -2.0, 2.0])
        self.assertEqual([atom1.y, atom2.y, atom3.y], [0.0, -2.0, 2.0])
        self.assertEqual([atom1.z, atom2.z, atom3.z], [0.0, -2.0, 2.0])
        self.assertEqual(
            [atom1.x, atom2.x, atom3.x], [atom.x for atom in molecule.atoms]
        )
        self.assertEqual(
            [atom1.x, atom2.x, atom3.x], [atom.x for atom in molecule.atoms]
        )
        self.assertEqual(
            [atom1.x, atom2.x, atom3.x], [atom.x for atom in molecule.atoms]
        )

        # unwrap to original positions
        atom2.x, atom2.y, atom2.z = 8.0, 8.0, 8.0
        box = [
            [0, 10],
            [0, 10],
            [0, 10],
        ]
        molecule.unwrap_atomic_coordinates(box)
        self.assertEqual([atom1.x, atom2.x, atom3.x], [0.0, -2.0, 2.0])
        self.assertEqual([atom1.y, atom2.y, atom3.y], [0.0, -2.0, 2.0])
        self.assertEqual([atom1.z, atom2.z, atom3.z], [0.0, -2.0, 2.0])
        self.assertEqual(
            [atom1.x, atom2.x, atom3.x], [atom.x for atom in molecule.atoms]
        )
        self.assertEqual(
            [atom1.x, atom2.x, atom3.x], [atom.x for atom in molecule.atoms]
        )
        self.assertEqual(
            [atom1.x, atom2.x, atom3.x], [atom.x for atom in molecule.atoms]
        )

        # calc cog
        molecule.calculate_cog()
        self.assertTrue(np.all(molecule.cog == np.array([0.0, 0.0, 0.0])))

        # test assign_voxel_idx
        box = [[-3, 3], [0, 8], [2, 5]]
        voxels = Voxels(box=box, number_of_voxels=[3, 4, 3])
        atom1 = Atom(ID=1, x=-2.0, y=1.7, z=1.0)
        atom2 = Atom(ID=2, x=0.0, y=2.3, z=4.0)
        molecule = Molecule(ID=1, atoms=[atom1, atom2])
        molecule.assign_voxel_idx(voxels)
        molecule.assign_voxel_idx(voxels)
        self.assertEqual(molecule.voxel_idx[0], 17)
        atom1 = Atom(ID=1, x=-2.0, y=1.7, z=0.999999)
        atom2 = Atom(ID=2, x=0.0, y=2.3, z=4.0)
        molecule = Molecule(ID=1, atoms=[atom1, atom2])
        molecule.assign_voxel_idx(voxels)
        molecule.assign_voxel_idx(voxels)
        self.assertEqual(molecule.voxel_idx[0], 16)

        # test translate
        atom1 = Atom(ID=1, x=0.0, y=0.0, z=0.0)
        atom2 = Atom(ID=2, x=-2.0, y=-2.0, z=-2.0)
        atom3 = Atom(ID=3, x=2.0, y=2.0, z=2.0)
        molecule = Molecule(ID=1, atoms=[atom1, atom2, atom3])
        molecule.calculate_cog()
        self.assertEqual(np.all(molecule.cog), np.all(np.array([0.0, 0.0, 0.0])))
        molecule.translate(np.array([1.0, 1.0, 1.0]))
        self.assertEqual(np.all(molecule.cog), np.all(np.array([1.0, 1.0, 1.0])))
        self.assertEqual([atom1.x, atom2.x, atom3.x], [1.0, -1.0, 3.0])
        self.assertEqual([atom1.y, atom2.y, atom3.y], [1.0, -1.0, 3.0])
        self.assertEqual([atom1.z, atom2.z, atom3.z], [1.0, -1.0, 3.0])

        # test rotate
        molecule.rotate(np.pi / 2, "x")
        self.assertEqual(np.all(molecule.cog), np.all(np.array([1.0, 1.0, 1.0])))
        for val in [(atom1.x, 1.0), (atom1.y, 1.0), (atom1.z, 1.0)]:
            self.assertAlmostEqual(val[0], val[1], places=8)
        for val in [(atom2.x, -1.0), (atom2.y, 3.0), (atom2.z, -1.0)]:
            self.assertAlmostEqual(val[0], val[1], places=8)
        for val in [(atom3.x, 3.0), (atom3.y, -1.0), (atom3.z, 3.0)]:
            self.assertAlmostEqual(val[0], val[1], places=8)

        return

    def test_ID_adjustments(self):
        """Tests adjust_IDs, adjust_atom_IDs, adjust_intramode_IDs, and clean_IDs"""
        atom1 = Atom(ID=3, x=0.0, y=0.0, z=0.0)
        atom2 = Atom(ID=1, x=0.0, y=0.0, z=2.0)
        atom3 = Atom(ID=9, x=0.0, y=0.0, z=2.0)
        atom4 = Atom(ID=5, x=0.0, y=0.0, z=2.0)
        atom5 = Atom(ID=8, x=0.0, y=0.0, z=2.0)
        molecule = Molecule(
            ID=5,
            kind="fiction",
            atoms=[
                Atom(ID=3, lammps_type=1, x=0.0, y=0.0, z=0.0),
                Atom(ID=1, lammps_type=5, x=1.0, y=0.0, z=0.0),
                Atom(ID=9, lammps_type=5, x=2.0, y=0.0, z=0.0),
                Atom(ID=5, lammps_type=2, x=3.0, y=0.0, z=0.0),
                Atom(ID=8, lammps_type=9, x=2.0, y=0.0, z=1.0),
            ],
            bonds=[
                IntraMode(ID=8, kind=1, atom_IDs=[3, 1]),
                IntraMode(ID=2, kind=2, atom_IDs=[1, 9]),
                IntraMode(ID=3, kind=3, atom_IDs=[9, 5]),
                IntraMode(ID=7, kind=4, atom_IDs=[9, 8]),
            ],
            angles=[
                IntraMode(ID=1, kind=1, atom_IDs=[3, 1, 9]),
                IntraMode(ID=2, kind=2, atom_IDs=[1, 9, 5]),
                IntraMode(ID=3, kind=2, atom_IDs=[1, 9, 8]),
                IntraMode(ID=4, kind=3, atom_IDs=[5, 9, 8]),
            ],
            dihedrals=[
                IntraMode(ID=4, kind=1, atom_IDs=[3, 1, 9, 5]),
                IntraMode(ID=3, kind=2, atom_IDs=[3, 1, 9, 8]),
            ],
            impropers=[
                IntraMode(ID=6, kind=5, atom_IDs=[9, 1, 5, 8]),
            ],
        )
        molecule.clean_IDs(new_ID=3)
        self.assertEqual(molecule.ID, 3)
        self.assertEqual(molecule.kind, "fiction")
        self.assertEqual(
            [[atom.ID, atom.lammps_type, atom.x, atom.y, atom.z] for atom in molecule.atoms],
            [
                [1, 1, 0.0, 0.0, 0.0],
                [2, 5, 1.0, 0.0, 0.0],
                [3, 5, 2.0, 0.0, 0.0],
                [4, 2, 3.0, 0.0, 0.0],
                [5, 9, 2.0, 0.0, 1.0],
             ]
        )
        self.assertEqual(
            [[bond.ID, bond.kind, bond.atom_IDs] for bond in molecule.bonds],
            [
                [1, 1, [1, 2]],
                [2, 2, [2, 3]],
                [3, 3, [3, 4]],
                [4, 4, [3, 5]],
            ]
        )
        self.assertEqual(
            [[angle.ID, angle.kind, angle.atom_IDs] for angle in molecule.angles],
            [
                [1, 1, [1, 2, 3]],
                [2, 2, [2, 3, 4]],
                [3, 2, [2, 3, 5]],
                [4, 3, [4, 3, 5]],
            ]
        )
        self.assertEqual(
            [[dihedral.ID, dihedral.kind, dihedral.atom_IDs] for dihedral in molecule.dihedrals],
            [
                [1, 1, [1, 2, 3, 4]],
                [2, 2, [1, 2, 3, 5]],
            ]
        )
        self.assertEqual(
            [[improper.ID, improper.kind, improper.atom_IDs] for improper in molecule.impropers],
            [
                [1, 5, [3, 2, 4, 5]],
            ]
        )
        return


if __name__ == "__main__":
    unittest.main()

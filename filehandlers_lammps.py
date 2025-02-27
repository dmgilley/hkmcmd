#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import numpy as np
from textwrap import dedent


def write_lammps_data(
    file_name,
    atoms_list,
    bonds_list=None,
    angles_list=None,
    dihedrals_list=None,
    impropers_list=None,
    box=None,
    num_atoms=None,
    num_bonds=None,
    num_angles=None,
    num_dihedrals=None,
    num_impropers=None,
    num_atom_types=None,
    num_bond_types=None,
    num_angle_types=None,
    num_dihedral_types=None,
    num_improper_types=None,
    masses=None,
    charge=True,
    wrap=True,
):

    if bonds_list is None:
        bonds_list = []
    if angles_list is None:
        angles_list = []
    if dihedrals_list is None:
        dihedrals_list = []
    if impropers_list is None:
        impropers_list = []
    if box is None:
        box = [
            [
                np.min([atom.x for atom in atoms_list]),
                np.max([atom.x for atom in atoms_list]),
            ],
            [
                np.min([atom.y for atom in atoms_list]),
                np.max([atom.y for atom in atoms_list]),
            ],
            [
                np.min([atom.z for atom in atoms_list]),
                np.max([atom.z for atom in atoms_list]),
            ],
        ]
    if num_atoms is None:
        num_atoms = len(atoms_list)
    if num_bonds is None:
        num_bonds = len(bonds_list)
    if num_angles is None:
        num_angles = len(angles_list)
    if num_dihedrals is None:
        num_dihedrals = len(dihedrals_list)
    if num_impropers is None:
        num_impropers = len(impropers_list)
    if num_atom_types is None:
        num_atom_types = len(set([atom.lammps_type for atom in atoms_list]))
    if num_bond_types is None:
        num_bond_types = len(set([bond.kind for bond in bonds_list]))
    if num_angle_types is None:
        num_angle_types = len(set([angle.kind for angle in angles_list]))
    if num_dihedral_types is None:
        num_dihedral_types = len(set([dihedral.kind for dihedral in dihedrals_list]))
    if num_improper_types is None:
        num_improper_types = len(set([improper.kind for improper in impropers_list]))
    if masses is None:
        masses = sorted(
            list(set([(atom.lammps_type, atom.mass) for atom in atoms_list]))
        )

    if wrap is True:
        for atom in atoms_list:
            atom.wrap(box)

    with open(file_name, "w") as f:

        f.write(
            f"""LAMMPS data file
                
{num_atoms:<5d} atoms
{num_atom_types:<5d} atom types
{num_bonds:<5d} bonds
{num_bond_types:<5d} bond types
{num_angles:<5d} angles
{num_angle_types:<5d} angle types
{num_dihedrals:<5d} dihedrals
{num_dihedral_types:<5d} dihedral types
{num_impropers:<5d} impropers
{num_improper_types:<5d} improper types

{box[0][0]:<12.8f} {box[0][1]:<12.8f} xlo xhi
{box[1][0]:<12.8f} {box[1][1]:<12.8f} ylo yhi
{box[2][0]:<12.8f} {box[2][1]:<12.8f} zlo zhi

"""
        )

        f.write("Masses\n\n")
        for mass in masses:
            f.write(f"{mass[0]:<5d} {mass[1]:<8.4f}\n")

        f.write("\nAtoms\n\n")
        if charge:
            for atom in atoms_list:
                f.write(
                    f"{atom.ID:<5d} {atom.molecule_ID:<5d} {atom.lammps_type:<5d} {atom.charge:>14.8f} {atom.x:>14.8f} {atom.y:>14.8f} {atom.z:>14.8f}\n"
                )
        if not charge:
            for atom in atoms_list:
                f.write(
                    f"{atom.ID:<5d} {atom.molecule_ID:<5d} {atom.lammps_type:<5d} {atom.x:>14.8f} {atom.y:>14.8f} {atom.z:>14.8f}\n"
                )

        if len(bonds_list) > 0:
            f.write("\nBonds\n\n")
            for intramode in bonds_list:
                f.write(
                    f"{intramode.ID:<4d} {intramode.kind:<4d} {intramode.atom_IDs[0]:<4d} {intramode.atom_IDs[1]:<4d}\n"
                )

        if len(angles_list) > 0:
            f.write("\nAngles\n\n")
            for intramode in angles_list:
                f.write(
                    f"{intramode.ID:<4d} {intramode.kind:<4d} {intramode.atom_IDs[0]:<4d} {intramode.atom_IDs[1]:<4d} {intramode.atom_IDs[2]:<4d}\n"
                )

        if len(dihedrals_list) > 0:
            f.write("\nDihedrals\n\n")
            for intramode in dihedrals_list:
                f.write(
                    f"{intramode.ID:<4d} {intramode.kind:<4d} {intramode.atom_IDs[0]:<4d} {intramode.atom_IDs[1]:<4d} {intramode.atom_IDs[2]:<4d} {intramode.atom_IDs[3]:<4d}\n"
                )

        if len(impropers_list) > 0:
            f.write("\nImpropers\n\n")
            for intramode in impropers_list:
                f.write(
                    f"{intramode.ID:<4d} {intramode.kind:<4d} {intramode.atom_IDs[0]:<4d} {intramode.atom_IDs[1]:<4d} {intramode.atom_IDs[2]:<4d} {intramode.atom_IDs[3]:<4d}\n"
                )

    return


class LammpsInitHandler:

    def __init__(
        self,
        prefix="default",
        settings_file_name="default.in.settings",
        data_file_name="default.in.data",
        thermo_freq=100,
        coords_freq=100,
        avg_calculate_every=50,
        avg_number_of_steps=10,
        avg_stepsize=5,
        units="lj",
        atom_style="full",
        dimension=3,
        newton="on",
        pair_style="lj/cut 3.0",
        bond_style="harmonic",
        angle_style="harmonic",
        dihedral_style="opls",
        improper_style="cvff",
        run_name=["equil1"],
        run_style=["npt"],
        run_stepsize=[1000000],
        run_temperature=[[298.0, 298.0, 100.0]],
        run_pressure_volume=[[1.0, 1.0, 100.0]],
        run_steps=[1.0],
        thermo_keywords=["temp", "press", "ke", "pe"],
        neigh_modify="every 1 delay 10 check yes one 10000",
        write_trajectories=True,
        write_intermediate_restarts=True,
        write_final_data=True,
        write_final_restarts=True,
    ) -> None:

        # Set attributes using the default values
        self.prefix = prefix
        self.settings_file_name = settings_file_name
        self.data_file_name = data_file_name
        self.thermo_freq = thermo_freq
        self.coords_freq = coords_freq
        self.avg_calculate_every = avg_calculate_every
        self.avg_number_of_steps = avg_number_of_steps
        self.avg_stepsize = avg_stepsize
        self.units = units
        self.atom_style = atom_style
        self.dimension = dimension
        self.newton = newton
        self.pair_style = pair_style
        self.bond_style = bond_style
        self.angle_style = angle_style
        self.dihedral_style = dihedral_style
        self.improper_style = improper_style
        self.run_name = run_name
        self.run_style = run_style
        self.run_stepsize = run_stepsize
        self.run_temperature = run_temperature
        self.run_pressure_volume = run_pressure_volume
        self.run_steps = run_steps
        self.thermo_keywords = thermo_keywords
        if type(self.thermo_keywords) == str:
            self.thermo_keywords = self.thermo_keywords.split()
        self.neigh_modify = neigh_modify
        self.write_trajectories = write_trajectories
        self.write_intermediate_restarts = write_intermediate_restarts
        self.write_final_data = write_final_data
        self.write_final_restarts = write_final_restarts

        return

    def generate_run_lines(
        self, name, style, timestep, steps, temperature, pressure_volume
    ):

        fixes = []
        dumps = []

        if style == "other":
            return "\n{}\n".format(pressure_volume)

        lines = dedent(
            """
        #===========================================================
        # {} ({})
        #===========================================================

        timestep {}
        velocity all create {} {}
        run_style verlet
        """.format(
                name, style, timestep, temperature.split()[0], np.random.randint(1, 1e6)
            )
        ).rstrip()

        if self.write_trajectories == True:
            lines += dedent(
                """
            dump {} all custom {} {}.{}.lammpstrj id mol type xu yu zu vx vy vz
            dump_modify {} sort id format float %20.10g
            """.format(
                    name, self.coords_freq, self.prefix, name, name
                )
            ).rstrip()
            dumps += [name]

        if style == "nvt deform":
            lines += dedent(
                """
            fix {}_deform all deform {}
            fix {}_nvt all nvt temp {}
            """.format(
                    name, pressure_volume, name, temperature
                )
            ).rstrip()
            fixes += ["{}_deform".format(name), "{}_nvt".format(name)]

        if style == "nve/limit":
            lines += dedent(
                """
            fix {} all nve/limit {}
            """.format(
                    name, pressure_volume
                )
            ).rstrip()
            fixes += ["{}".format(name)]

        if style == "nvt":
            lines += dedent(
                """
            fix {} all nvt temp {}
            """.format(
                    name, temperature
                )
            ).rstrip()
            fixes += ["{}".format(name)]

        if style == "nve":
            lines += dedent(
                """
            fix {} all nve
            """.format(
                    name
                )
            ).rstrip()
            fixes += ["{}".format(name)]

        lines += dedent(
            """
        run {}
        """.format(
                steps
            )
        ).rstrip()

        for fix in fixes:
            lines += dedent(
                """
            unfix {}
            """.format(
                    fix
                )
            ).rstrip()

        for dump in dumps:
            lines += dedent(
                """
            undump {}
            """.format(
                    dump
                )
            ).rstrip()

        if self.write_intermediate_restarts:
            lines += dedent(
                """
            write_restart {}.restart
            """.format(
                    name
                )
            ).rstrip()

        lines += "\n"

        return lines

    def write(self):
        file = open(self.prefix + ".in.init", "w")
        file.write(
            dedent(
                """\
            # LAMMPS init file

            #===========================================================
            # Initialize System
            #===========================================================

            # System definition
            units {}
            dimension {}
            newton {}
            boundary p p p
            atom_style {}
            neigh_modify {}
            
            # Force-field definition
            special_bonds   lj 0.0 0.0 0.0 coul 0.0 0.0 0.0
            pair_style      {}
            pair_modify     shift yes mix sixthpower
            bond_style      {}
            angle_style     {}
            dihedral_style  {}
            improper_style  {}

            # Data, settings, and log files setup
            read_data {}
            include {}
            log {}.lammps.log
            thermo_style custom {}
            thermo_modify format float %14.6f
            thermo {}

            # Thermodynamic averages file setup
            # "Nevery Nrepeat Nfreq": On every "Nfreq" steps, take the averages by using "Nrepeat" previous steps, counted every "Nevery"
            {}            fix averages all ave/time {} {} {} v_calc_{} file {}.thermo.avg format %20.10g
            """.format(
                    self.units,
                    self.dimension,
                    self.newton,
                    self.atom_style,
                    self.neigh_modify,
                    self.pair_style,
                    self.bond_style,
                    self.angle_style,
                    self.dihedral_style,
                    self.improper_style,
                    self.data_file_name,
                    self.settings_file_name,
                    self.prefix,
                    " ".join(self.thermo_keywords),
                    self.thermo_freq,
                    "            ".join(
                        [
                            "variable calc_{} equal {}\n".format(k, k)
                            for k in self.thermo_keywords
                        ]
                    ),
                    self.avg_stepsize,
                    self.avg_number_of_steps,
                    self.avg_calculate_every,
                    " v_calc_".join(self.thermo_keywords),
                    self.prefix,
                )
            )
        )

        for run_idx, run_name in enumerate(self.run_name):
            file.write(
                self.generate_run_lines(
                    run_name,
                    self.run_style[run_idx],
                    self.run_stepsize[run_idx],
                    self.run_steps[run_idx],
                    self.run_temperature[run_idx],
                    self.run_pressure_volume[run_idx],
                )
            )

        file.write(
            dedent(
                """\
                   
            #===========================================================
            # Clean and exit
            #===========================================================

            unfix averages
            """
            )
        )
        if self.write_final_data:
            file.write("write_data {}.end.data\n".format(self.prefix))
        if self.write_final_data:
            file.write("write_restart {}.end.restart".format(self.prefix))

        file.close()
        return

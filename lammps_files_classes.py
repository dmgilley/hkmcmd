#!/usr/bin/env python

import numpy as np
from textwrap import dedent


# Simple function for testing whether a given string can be made into a float
def isfloat_str(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# Function for writing a lammps data file, given a "data" dictionary
def write_lammps_data(
    file_name, atoms, bonds, angles, dihedrals, impropers, box, header={}, charge=True
):

    header_defaults = {
        "atoms": atoms.ids.flatten().shape[0],
        "atom_types": len(set(atoms.lammps_type)),
        "bonds": bonds.ids.flatten().shape[0],
        "bond_types": len(set(bonds.lammps_type)),
        "angles": angles.ids.flatten().shape[0],
        "angle_types": len(set(angles.lammps_type)),
        "dihedrals": dihedrals.ids.flatten().shape[0],
        "dihedral_types": len(set(dihedrals.lammps_type)),
        "impropers": impropers.ids.flatten().shape[0],
        "improper_types": len(set(impropers.lammps_type)),
    }
    if "masses" not in header.keys():
        header_defaults["masses"] = sorted(
            list(
                set(
                    [
                        (atoms.lammps_type[idx], atoms.mass[idx])
                        for idx in range(len(atoms.ids))
                    ]
                )
            )
        )

    for keyword in header_defaults.keys():
        if keyword not in header:
            header[keyword] = header_defaults[keyword]

    with open(file_name, "w") as f:

        f.write("LAMMPS data file.\n\n")

        f.write("{:<5d} {}\n".format(header["atoms"], "atoms"))
        f.write("{:<5d} {}\n".format(header["atom_types"], "atom types"))
        f.write("{:<5d} {}\n".format(header["bonds"], "bonds"))
        f.write("{:<5d} {}\n".format(header["bond_types"], "bond types"))
        f.write("{:<5d} {}\n".format(header["angles"], "angles"))
        f.write("{:<5d} {}\n".format(header["angle_types"], "angle types"))
        f.write("{:<5d} {}\n".format(header["dihedrals"], "dihedrals"))
        f.write("{:<5d} {}\n".format(header["dihedral_types"], "dihedral types"))
        f.write("{:<5d} {}\n".format(header["impropers"], "impropers"))
        f.write("{:<5d} {}\n\n".format(header["improper_types"], "improper types"))

        f.write(
            "{:<12.8f} {:<12.8f} {} {}\n".format(box[0][0], box[0][1], "xlo", "xhi")
        )
        f.write(
            "{:<12.8f} {:<12.8f} {} {}\n".format(box[1][0], box[1][1], "ylo", "yhi")
        )
        f.write(
            "{:<12.8f} {:<12.8f} {} {}\n\n".format(box[2][0], box[2][1], "zlo", "zhi")
        )

        f.write("Masses\n\n")
        for mass in header["masses"]:
            f.write("{:<5d} {:<8.4f}\n".format(mass[0], mass[1]))

        f.write("\nAtoms\n\n")
        if charge:
            for idx, atom_id in enumerate(atoms.ids):
                f.write(
                    "{:<5d} {:<5d} {:<5d} {:>14.8f} {:>14.8f} {:>14.8f} {:>14.8f}\n".format(
                        atom_id,
                        atoms.mol_id[idx],
                        atoms.lammps_type[idx],
                        atoms.charge[idx],
                        atoms.x[idx],
                        atoms.y[idx],
                        atoms.z[idx],
                    )
                )
        if not charge:
            for idx, atom_id in enumerate(atoms.ids):
                f.write(
                    "{:<5d} {:<5d} {:<5d} {:>14.8f} {:>14.8f} {:>14.8f}\n".format(
                        atom_id,
                        atoms.mol_id[idx],
                        atoms.lammps_type[idx],
                        atoms.x[idx],
                        atoms.y[idx],
                        atoms.z[idx],
                    )
                )

        if bonds.ids.flatten().shape[0] > 0:
            f.write("\nBonds\n\n")
            for idx, bond_id in enumerate(bonds.ids):
                f.write(
                    "{:<4d} {:<4d} {:<4d} {:<4d}\n".format(
                        bond_id,
                        bonds.lammps_type[idx],
                        bonds.atom_ids[idx][0],
                        bonds.atom_ids[idx][1],
                    )
                )

        if angles.ids.flatten().shape[0] > 0:
            f.write("\nAngles\n\n")
            for idx, angle_id in enumerate(angles.ids):
                f.write(
                    "{:<4d} {:<4d} {:<4d} {:<4d} {:<4d}\n".format(
                        angle_id,
                        angles.lammps_type[idx],
                        angles.atom_ids[idx][0],
                        angles.atom_ids[idx][1],
                        angles.atom_ids[idx][2],
                    )
                )

        if dihedrals.ids.flatten().shape[0] > 0:
            f.write("\nDihedrals\n\n")
            for idx, dihedral_id in enumerate(dihedrals.ids):
                f.write(
                    "{:<4d} {:<4d} {:<4d} {:<4d} {:<4d} {:<4d}\n".format(
                        dihedral_id,
                        dihedrals.lammps_type[idx],
                        dihedrals.atom_ids[idx][0],
                        dihedrals.atom_ids[idx][1],
                        dihedrals.atom_ids[idx][2],
                        dihedrals.atom_ids[idx][3],
                    )
                )

        if impropers.ids.flatten().shape[0] > 0:
            f.write("\nImpropers\n\n")
            for idx, improper_id in enumerate(impropers.ids):
                f.write(
                    "{:<4d} {:<4d} {:<4d} {:<4d} {:<4d} {:<4d}\n".format(
                        improper_id,
                        impropers.lammps_type[idx],
                        impropers.atom_ids[idx][0],
                        impropers.atom_ids[idx][1],
                        impropers.atom_ids[idx][2],
                        impropers.atom_ids[idx][3],
                    )
                )

    return


def parse_lammps_out(fname):

    data = {}
    flag = False
    with open(fname, "r") as f:
        for line in f:
            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "Step":
                flag = True
                keywords = fields[1:]
                continue
            if len(fields) > 7:
                if fields[0] == "Loop" and fields[1] == "time":
                    flag = False
                    continue
            if flag:
                data[int(fields[0])] = [float(i) for i in fields[1:]]

    return keywords, data


def parse_data_header_and_masses(fname):

    header = {"masses": {}}
    flag = None

    int_types = ["atom", "bond", "angle", "dihedral", "improper"]

    with open(fname, "r") as f:
        for line in f:

            fields = line.split()
            if fields == []:
                continue
            if fields[0] == "#":
                continue
            if fields[0] == "Masses":
                flag = "Masses"
                continue

            if not flag and len(fields) == 3:
                if fields[1] in int_types and fields[2] == "types":
                    header[fields[1] + "_types"] = int(fields[0])
                    continue

            if flag == "Masses":
                try:
                    header["masses"][int(fields[0])] = float(fields[1])
                except:
                    header["masses"] = [
                        (k, header["masses"][k])
                        for k in sorted(list(header["masses"].keys()))
                    ]
                    return header

    header["masses"] = [
        (k, header["masses"][k]) for k in sorted(list(header["masses"].keys()))
    ]
    return header


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

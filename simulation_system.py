#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dgilley@purdue.edu


import json, os
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Union
from hybrid_mdmc import utility
from hybrid_mdmc.particle_interactions import Molecule, Reaction


class SystemState:
    """Class to store the state of the system.

    Attributes
    ----------
    filename_output : str
        Filename to write the output to.
    atom_IDs : np.ndarray (N,)
        Flat array of the N atom IDs.
    reaction_kinds : np.ndarray (R,)
        Flat array of the R reaction IDs.
    diffusion_steps : np.ndarray (S,)
        Flat array of the S diffusion steps.
    reactive_steps : np.ndarray (S,)
        Flat array of the S reactive steps.
    times : np.ndarray (S,)
        Flat array of the S times.
    molecule_IDs : np.ndarray (S,N)
        Array of the molecule ID for each of the N atoms in each of the S steps.
    molecule_kinds : np.ndarray (S,N)
        Array of the molecule kind for each of the N atoms in each of the S steps.
    reaction_selections : np.ndarray (S,R)
        Array of the reaction selection for each of the S steps.
    reaction_scalings : np.ndarray (S,R)
        Array of the reaction scaling for each of the S steps.

    Methods
    -------
    """

    def __init__(
        self,
        filename: Union[None, str] = None,
        atom_IDs: Union[None, list, np.ndarray] = None,
        reaction_kinds: Union[None, list, np.ndarray] = None,
        diffusion_steps: Union[None, list, np.ndarray] = None,
        reactive_steps: Union[None, list, np.ndarray] = None,
        times: Union[None, list, np.ndarray] = None,
        molecule_IDs: Union[None, np.ndarray] = None,
        molecule_kinds: Union[None, np.ndarray] = None,
        reaction_selections: Union[None, np.ndarray] = None,
        reaction_scalings: Union[None, np.ndarray] = None,
    ):
        self.filename = filename
        self.atom_IDs = atom_IDs
        self.reaction_kinds = reaction_kinds
        self.diffusion_steps = diffusion_steps
        self.reactive_steps = reactive_steps
        self.times = times
        self.molecule_IDs = molecule_IDs
        self.molecule_kinds = molecule_kinds
        self.reaction_selections = reaction_selections
        self.reaction_scalings = reaction_scalings
        self.clean()
        self.check_consistency()
        return

    def clean(self):
        if type(self.atom_IDs) == list:
            self.atom_IDs = np.array(self.atom_IDs).flatten()
        if type(self.reaction_kinds) == list:
            self.reaction_kinds = np.array(self.reaction_kinds).flatten()
        if type(self.diffusion_steps) == list:
            self.diffusion_steps = np.array(self.diffusion_steps).flatten()
        if type(self.reactive_steps) == list:
            self.reactive_steps = np.array(self.reactive_steps).flatten()
        if type(self.times) == list:
            self.times = np.array(self.times).flatten()
        return

    def check_consistency(self):

        # check number of atoms
        atoms = [_.shape[0] if _ is not None else 0 for _ in [self.atom_IDs]]
        atoms += [
            _.shape[1] if _ is not None else 0
            for _ in [self.molecule_IDs, self.molecule_kinds]
        ]
        if len(set(atoms)) != 1:
            raise ValueError(
                "Inconsistent number of atoms between SystemState attributes."
            )
        atoms = atoms[0]

        # check number of reactions
        reactions = [_.shape[0] if _ is not None else 0 for _ in [self.reaction_kinds]]
        reactions += [
            _.shape[1] if _ is not None else 0
            for _ in [self.reaction_selections, self.reaction_scalings]
        ]
        if len(set(reactions)) != 1:
            raise ValueError(
                "Inconsistent number of reactions between SystemState attributes."
            )
        reactions = reactions[0]

        # check number of steps
        steps = [
            _.shape[0] if _ is not None else 0
            for _ in [
                self.diffusion_steps,
                self.reactive_steps,
                self.times,
                self.molecule_IDs,
                self.molecule_kinds,
                self.reaction_selections,
                self.reaction_scalings,
            ]
        ]
        if len(set(steps)) != 1:
            raise ValueError(
                "Inconsistent number of steps between SystemState attributes."
            )
        steps = steps[0]

        # check final step molecule IDs and kinds match
        if self.molecule_IDs is not None:
            unique_IDs = np.unique(self.molecule_IDs[-1,:])
            ID_indices = [np.argwhere(self.molecule_IDs[-1,:] == ID)[0] for ID in unique_IDs]
            unique_kinds = np.unique(self.molecule_kinds[-1,:])
            kind_indices = [np.argwhere(self.molecule_kinds[-1,:] == kind)[0] for kind in unique_kinds]
            if np.all(ID_indices == kind_indices) is False:
                raise ValueError(
                    "Inconsistent molecule IDs and kinds in final step of SystemState."
                )

        return

    def check_atoms_list(self, atoms_list):

        # can we proceed?
        if self.atom_IDs is None:
            raise ValueError("SystemState.atom_IDs is None")

        # prep
        IDs_molecules = list(
            zip(
                deepcopy(self.atom_IDs),
                deepcopy(self.molecule_IDs[-1, :]),
                deepcopy(self.molecule_kinds[-1, :]),
            )
        )
        IDs_molecules.sort(key=lambda x: x[0])
        atoms_list = deepcopy(atoms_list)
        atoms_list.sort(key=lambda x: x.ID)

        # check atom IDs
        if not np.all(list(zip(*IDs_molecules))[0] == [_.ID for _ in atoms_list]):
            raise ValueError("SystemState.atom_IDs does not match atoms_list")

        # check molecule IDs
        if None not in [_.molecule_ID for _ in atoms_list]:
            if not np.all(
                list(zip(*IDs_molecules))[1] == [_.molecule_ID for _ in atoms_list]
            ):
                raise ValueError("SystemState.molecule_IDs does not match atoms_list")

        # check molecule kinds
        if None not in [_.kind for _ in atoms_list]:
            if not np.all(
                list(zip(*IDs_molecules))[2] == [_.molecule_kind for _ in atoms_list]
            ):
                raise ValueError("SystemState.molecule_kinds does not match atoms_list")

        return

    def get_molecule_kind(self, molecule_ID):
        return self.molecule_kinds[-1,np.argwhere(self.molecule_IDs[-1] == molecule_ID)[0]][0]

    def assemble_reaction_scaling_df(self):
        return pd.DataFrame(
            data=self.reaction_scalings,
            index=self.reactive_steps,
            columns=self.reaction_kinds,
        )

    def assemble_progression_df(self, species_kinds):
        _dict = {}
        stacked_molecules = np.stack([self.molecule_IDs, self.molecule_kinds], axis=2)
        ID_kind_counts = [
            np.unique(np.unique(step, axis=0)[:, 1], return_counts=True)
            for step in stacked_molecules
        ]
        _dict.update({
            kind: [
                    step[1][np.argwhere(step[0] == kind)][0][0] if kind in step[0] else 0
                    for step in ID_kind_counts
                ]
            for kind in species_kinds
        })
        _dict.update({
            "time": self.times
        })
        _dict.update({
            reaction_kind: self.reaction_selections[:,idx]
            for idx,reaction_kind in enumerate(self.reaction_kinds)
        })
        return pd.DataFrame(_dict, index=self.reactive_steps)

    def update_molecules(self, molecules_list):
        _map = {atom_ID: idx for idx,atom_ID in enumerate(self.atom_IDs)}
        self.molecule_IDs = np.append(self.molecule_IDs, [np.zeros_like(self.atom_IDs)], axis=0)
        self.molecule_kinds = np.append(self.molecule_kinds, [np.zeros_like(self.atom_IDs)], axis=0)
        for molecule in molecules_list:
            for atom in molecule.atoms_list:
                self.molecule_IDs[-1,_map[atom.ID]] = molecule.ID
                self.molecule_kinds[-1,_map[atom.ID]] = molecule.kind
        return
    
    def update_reactions(self, reaction_selections, reaction_scaling_df):
        self.reaction_selections = np.append(self.reaction_selections, [np.zeros_like(self.reaction_kinds)], axis=0)
        self.reaction_scalings = np.append(self.reaction_scalings, [np.zeros_like(self.reaction_kinds)], axis=0)
        for idx,reaction_kind in enumerate(self.reaction_kinds):
            self.reaction_selections[-1,idx] = reaction_selections.count(reaction_kind)
            self.reaction_scalings[-1,idx] = reaction_scaling_df[reaction_kind][-1]
        return

    def update_steps(self, diffusion_cycle, dt):
        self.diffusion_steps = np.append(self.diffusion_steps, [diffusion_cycle])
        self.reactive_steps = np.append(self.reactive_steps, [self.reactive_steps[-1] + 1])
        self.times = np.append(self.times, [self.times[-1] + dt])
        return

    def write_to_json(self, filename=None):
        write_dict = {}
        if filename is None:
            filename = self.filename
        for k,v in self.__dict__.items():
            if type(v) == np.ndarray:
                write_dict[k] = v.tolist()
            else:
                write_dict[k] = v
        with open(self.filename, "w") as f:
            f.write(json.dumps(write_dict, indent=4))
        return

    def read_data_from_json(self, filename=None):
        if filename is None:
            filename = self.filename
        with open(filename, "r") as f:
            data = json.load(f)
        for k,v in data.items():
            if k == "filename":
                continue
            setattr(self, k, np.array(v))
        self.clean()
        self.check_consistency()
        return 


def read_hkmcmd_json(filename):
    with open(filename, "r") as json_file:
        data_dict = json.load(json_file)
    if data_dict["species"] is not None:
        data_dict["species"] = [Molecule(**item) for item in data_dict["species"]]
    if data_dict["reactions"] is not None:
        data_dict["reactions"] = [Reaction(**item) for item in data_dict["reactions"]]
    return data_dict


def write_hkmcmd_json(data_dict, filename):
    if data_dict["species"] is not None:
        data_dict["species"] = [item.make_jsonable() for item in data_dict["species"]]
    if data_dict["reactions"] is not None:
        data_dict["reactions"] = [
            item.make_jsonable() for item in data_dict["reactions"]
        ]
    with open(filename, "w") as json_file:
        json.dump(data_dict, json_file, indent=4, default=str)
    return


class SystemData:

    def __init__(self, name, prefix, filename_json=None):
        self.name = name
        self.prefix = prefix
        self.filename_json = filename_json
        self.hkmcmd = None
        self.scaling_reaction = None
        self.scaling_diffusion = None
        self.lammps = None
        self.species = None
        self.reactions = None
        self.MD_initial = None
        self.MD_cycling = None
        if self.filename_json is None:
            self.filename_json = self.name + ".json"
        return

    def read_json(self):
        if not os.path.exists(self.filename_json):
            raise ValueError(
                f"JSON file {self.filename_json} does not exist. Cannot read system data."
            )
        for k, v in read_hkmcmd_json(self.filename_json).items():
            setattr(self, k, v)
        return

    def write_json(self):
        write_hkmcmd_json(
            {
                k: v
                for k, v in self.__dict__.items()
                if k not in ["name", "prefix", "filename_json", "filename_excel"]
            },
            self.filename_json,
        )
        return

    def clean(self):

        # sort the species list by kind
        self.species.sort(key=lambda x: x.kind)

        # reset the Atom and IntraMode IDs in the molecule
        for molecule in self.species:
            molecule.clean_IDs()

        # check lammps data
        if self.lammps is not None:

            # check number of atom types
            lammps_atom_types = self.lammps.get("atom_types", None)
            species_atom_types = sorted(
                set(
                    utility.flatten_nested_list(
                        [
                            [atom.lammps_type for atom in molecule.atoms]
                            for molecule in self.species
                        ]
                    )
                )
            )
            if lammps_atom_types is not None and lammps_atom_types != len(
                species_atom_types
            ):
                raise ValueError(
                    f"Number of atom types in LAMMPS data ({lammps_atom_types}) does not match number of atom types in species ({species_atom_types})."
                )

            # check number of bond types
            lammps_bond_types = self.lammps.get("bond_types", None)
            species_bond_types = sorted(
                set(
                    utility.flatten_nested_list(
                        [
                            [bond.kind for bond in molecule.bonds]
                            for molecule in self.species if molecule.bonds is not None
                        ]
                    )
                )
            )
            if lammps_bond_types is not None and lammps_bond_types != len(
                species_bond_types
            ):
                raise ValueError(
                    f"Number of bond types in LAMMPS data ({lammps_bond_types}) does not match number of bond types in species ({species_bond_types})."
                )

            # check number of angle types
            lammps_angle_types = self.lammps.get("angle_types", None)
            species_angle_types = sorted(
                set(
                    utility.flatten_nested_list(
                        [
                            [angle.kind for angle in molecule.angles]
                            for molecule in self.species if molecule.angles is not None
                        ]
                    )
                )
            )
            if lammps_angle_types is not None and lammps_angle_types != len(
                species_angle_types
            ):
                raise ValueError(
                    f"Number of angle types in LAMMPS data ({lammps_angle_types}) does not match number of angle types in species ({species_angle_types})."
                )

            # check number of dihedral types
            lammps_dihedral_types = self.lammps.get("dihedral_types", None)
            species_dihedral_types = sorted(
                set(
                    utility.flatten_nested_list(
                        [
                            [dihedral.kind for dihedral in molecule.dihedrals]
                            for molecule in self.species if molecule.dihedrals is not None
                        ]
                    )
                )
            )
            if lammps_dihedral_types is not None and lammps_dihedral_types != len(
                species_dihedral_types
            ):
                raise ValueError(
                    f"Number of dihedral types in LAMMPS data ({lammps_dihedral_types}) does not match number of dihedral types in species ({species_dihedral_types})."
                )

            # check number of improper types
            lammps_improper_types = self.lammps.get("improper_types", None)
            species_improper_types = sorted(
                set(
                    utility.flatten_nested_list(
                        [
                            [improper.kind for improper in molecule.impropers]
                            for molecule in self.species if molecule.impropers is not None
                        ]
                    )
                )
            )
            if lammps_improper_types is not None and lammps_improper_types != len(
                species_improper_types
            ):
                raise ValueError(
                    f"Number of improper types in LAMMPS data ({lammps_improper_types}) does not match number of improper types in species ({species_improper_types})."
                )

        return

    def prepare_dict_for_LammpsInitWriter(self, key):
        """
        thermo_freq=100, #
        coords_freq=100, #
        avg_calculate_every=50, #
        avg_number_of_steps=10, #
        avg_stepsize=5, #
        units="lj", #
        atom_style="full", #
        dimension=3, #
        newton="on", #
        pair_style="lj/cut 3.0", #
        bond_style="harmonic", #
        angle_style="harmonic", #
        dihedral_style="opls", #
        improper_style="cvff", #
        run_name=["equil1"], #
        run_style=["npt"], #
        run_stepsize=[1000000], #
        run_temperature=[[298.0, 298.0, 100.0]], go from string to list of floats
        run_pressure_volume=[[1.0, 1.0, 100.0]], # ?
        run_steps=[1.0], #
        thermo_keywords=["temp", "press", "ke", "pe"], # one string to list of strings
        neigh_modify="every 1 delay 10 check yes one 10000", #
        write_trajectories=True, #
        write_intermediate_restarts=True, #
        write_final_data=True, #
        write_final_restarts=True, #
        """
        keys = [
            "thermo_freq",
            "coords_freq",
            "avg_calculate_every",
            "avg_number_of_steps",
            "avg_stepsize",
            "units",
            "atom_style",
            "dimension",
            "newton",
            "pair_style",
            "bond_style",
            "angle_style",
            "dihedral_style",
            "improper_style",
            "run_name",
            "run_style",
            "run_stepsize",
            "run_temperature",
            "run_pressure_volume",
            "run_steps",
            "thermo_keywords",
            "neigh_modify",
            "write_trajectories",
            "write_intermediate_restarts",
            "write_final_data",
            "write_final_restarts",
        ]
        dict_ = {
            "prefix": self.prefix,
            "settings_file_name": f"{self.name}.in.settings",
            "data_file_name": f"{self.prefix}.in.data",
        }
        attr = getattr(self, key)
        dict_.update({k: attr[k] for k in keys})
        return dict_

    @property
    def atomic_masses(self):
        atoms = utility.flatten_nested_list(
            [molecule.atoms for molecule in self.species]
        )
        masses = sorted(
            set([(atom.lammps_type, atom.mass) for atom in atoms]), key=lambda x: x[0]
        )
        if None in [_[1] for _ in masses]:
            raise ValueError("Masses must be defined for all atoms.")
        return masses

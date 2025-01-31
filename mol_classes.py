import numpy as np


class AtomList:
    """Class for all arrays of all atom attributes.

    Attributes
    -----------

    ids: numpy array
        List of IDs of each atom during simulation

    lammps_type: numpy array
        List of atom-type (as used in LAMMPS) of each atom

    taffi_type: numpy array
        List of TAFFI type of each atom

    element: numpy array
        List of element type of each atom

    mass: numpy array
        List of atom masses 

    charge: numpy array
        List of atom charges
    
    mol_id: numpy array
        List of IDs of molecule the atom belongs to (as used in LAMMPS)

    mol_type: numpy array
        List of the molecule hash type (as defined in TAFFI scripts)

    x: numpy array
        List of x-coordinates of atoms

    y: numpy array
        List of y-coordinates of atoms

    z: numpy array
        List of z-coordinates of atoms

    vx: numpy array
        List of velocity component along x-axis of atoms

    vy: numpy array
        List of velocity component along y-axis of atoms

    vz: numpy array
        List of velocity component along z-axis of atoms

    fx: numpy array
        List of force component along x-axis of atoms

    fy: numpy array
        List of force component along y-axis of atoms

    fz: numpy array
        List of force component along z-axis of atoms

    q1: numpy array
        List of first component of quaternions for all atoms

    q2: numpy array
        List of second component of quaternions for all atoms

    q3: numpy array
        List of third component of quaternions for all atoms

    q4: numpy array
        List of fourth component of quaternions for all atoms

    Methods
    -----------

    append(ids=[], lammps_type=[], taffi_type=[], element=[], mass=[], charge=[], mol_id=[], mol_type=[],\
     x=[], y=[], z=[], vx=[], vy=[], vz=[], fx=[], fy=[], fz=[], q1=[], q2=[], q3=[], q4=[])
        Appends entries to each of the attributes, i.e. adds values of new atom attributes to the existing\
        list 

    get_idx(lammps_type=[], taffi_type=[], element=[], mass=[], charge=[], mol_id=[], mol_type=[])
        For a non-empty list "l" of an attributes "a1", the method returns the index in the list where the 
        atoms have values of a1 attribute in the input list l1. For example, this is useful to parse indices
        of atoms of type 1 and 2, or indices of Carbon and Hydrogen atoms

    get_idx_group(lammps_type=None, taffi_type=None, element=None, mass=None, charge=None, mol_id=None,\
     mol_type=None)
        For a given attribute, groups indices based on same value of attribute. Returns dictionary with\
        keys as attribute values (for example., element types) and objects as the indices of atoms with\
        that value (for example., indices of C atoms)

    del_idx(idx,reassign_ids=1,reassign_lammps_type=1)
        Deletes entries from all atom attributes at the indices in "idx" list. This is useful if you want\
        to remove certain atoms from a simulation box. It will also reassign the new IDs and lammps types \
        by default, so that they are conrinuous, i.e. 1,2,3... .


    Errors
    -----------     
    
    "Error! Mismatch in the length of AtomList attributes. Exiting..."
        Self-explanatory


    Additional notes
    -----------

    If one wishes to add new attributes (eg., atom shape):
    Use the same format as for other attributes and modify all the methods below.

    Future work
    -----------     

    """

    # Initialize
    def __init__(
        self,
        ids=[],
        lammps_type=[],
        taffi_type=[],
        element=[],
        mass=[],
        charge=[],
        mol_id=[],
        mol_type=[],
        x=[],
        y=[],
        z=[],
        vx=[],
        vy=[],
        vz=[],
        fx=[],
        fy=[],
        fz=[],
        q1=[],
        q2=[],
        q3=[],
        q4=[],
    ):
        self.ids = np.array(ids)
        self.lammps_type = np.array(lammps_type)
        self.taffi_type = np.array(taffi_type)
        self.element = np.array(element)
        self.mass = np.array(mass)
        self.charge = np.array(charge)
        self.mol_id = np.array(mol_id)
        self.mol_type = np.array(mol_type)
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.vx = np.array(vx)
        self.vy = np.array(vy)
        self.vz = np.array(vz)
        self.fx = np.array(fx)
        self.fy = np.array(fy)
        self.fz = np.array(fz)
        self.q1 = np.array(q1)
        self.q2 = np.array(q2)
        self.q3 = np.array(q3)
        self.q4 = np.array(q4)

        # Checks
        all_keys = [
            "ids",
            "lammps_type",
            "taffi_type",
            "element",
            "mass",
            "charge",
            "mol_id",
            "mol_type",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "fx",
            "fy",
            "fz",
            "q1",
            "q2",
            "q3",
            "q4",
        ]
        Ls = {_: len(self.__dict__[_]) for _ in all_keys if len(self.__dict__[_]) != 0}
        if len(set([Ls[k] for k in Ls.keys()])) > 1:
            print("Error! Mismatch in the length of AtomList attributes. Exiting...")
            print(
                "Length of each attribute is: {}".format(
                    ", ".join(["{}: {}".format(k, Ls[k]) for k in Ls.keys()])
                )
            )
            quit()

    # Append info for new atoms
    def append(
        self,
        ids=[],
        lammps_type=[],
        taffi_type=[],
        element=[],
        mass=[],
        charge=[],
        mol_id=[],
        mol_type=[],
        x=[],
        y=[],
        z=[],
        vx=[],
        vy=[],
        vz=[],
        fx=[],
        fy=[],
        fz=[],
        q1=[],
        q2=[],
        q3=[],
        q4=[],
    ):
        self.ids = np.array(list(self.ids) + ids)
        self.lammps_type = np.array(list(self.lammps_type) + list(lammps_type))
        self.taffi_type = np.array(list(self.taffi_type) + list(taffi_type))
        self.element = np.array(list(self.element) + list(element))
        self.mass = np.array(list(self.mass) + list(mass))
        self.charge = np.array(list(self.charge) + list(charge))
        self.mol_id = np.array(list(self.mol_id) + list(mol_id))
        self.mol_type = np.array(list(self.mol_type) + list(mol_type))
        self.x = np.array(list(self.x) + list(x))
        self.y = np.array(list(self.y) + list(y))
        self.z = np.array(list(self.z) + list(z))
        self.vx = np.array(list(self.vx) + list(vx))
        self.vy = np.array(list(self.vy) + list(vy))
        self.vz = np.array(list(self.vz) + list(vz))
        self.fx = np.array(list(self.fx) + list(fx))
        self.fy = np.array(list(self.fy) + list(fy))
        self.fz = np.array(list(self.fz) + list(fz))
        self.q1 = np.array(list(self.q1) + list(q1))
        self.q2 = np.array(list(self.q2) + list(q2))
        self.q3 = np.array(list(self.q3) + list(q3))
        self.q4 = np.array(list(self.q4) + list(q4))

    # Get ids based on a specific value of an attribute
    def get_idx(
        self,
        ids=None,
        lammps_type=None,
        taffi_type=None,
        element=None,
        mass=None,
        charge=None,
        mol_id=None,
        mol_type=None,
    ):
        if ids is not None:
            return [_ for _, v in enumerate(self.ids) if v in ids]
        if lammps_type is not None:
            return [_ for _, v in enumerate(self.lammps_type) if v in lammps_type]
        if taffi_type is not None:
            return [_ for _, v in enumerate(self.taffi_type) if v in taffi_type]
        if element is not None:
            return [_ for _, v in enumerate(self.element) if v in element]
        if mass is not None:
            return [_ for _, v in enumerate(self.mass) if v in mass]
        if charge is not None:
            return [_ for _, v in enumerate(self.charge) if v in charge]
        if mol_id is not None:
            return [_ for _, v in enumerate(self.mol_id) if v in mol_id]
        if mol_type is not None:
            return [_ for _, v in enumerate(self.mol_type) if v in mol_type]

    # Get idx grouped according to an attribute
    def get_idx_group(
        self,
        lammps_type=None,
        taffi_type=None,
        element=None,
        mass=None,
        charge=None,
        mol_id=None,
        mol_type=None,
    ):
        if lammps_type is not None:
            return {
                i: [_ for _, v in enumerate(self.lammps_type) if v == i]
                for i in set(self.lammps_type)
            }
        if taffi_type is not None:
            return {
                i: [_ for _, v in enumerate(self.taffi_type) if v == i]
                for i in set(self.taffi_type)
            }
        if element is not None:
            return {
                i: [_ for _, v in enumerate(self.element) if v == i]
                for i in set(self.element)
            }
        if mass is not None:
            return {
                i: [_ for _, v in enumerate(self.mass) if v == i]
                for i in set(self.mass)
            }
        if charge is not None:
            return {
                i: [_ for _, v in enumerate(self.charge) if v == i]
                for i in set(self.charge)
            }
        if mol_id is not None:
            return {
                i: [_ for _, v in enumerate(self.mol_id) if v == i]
                for i in set(self.mol_id)
            }
        if mol_type is not None:
            return {
                i: [_ for _, v in enumerate(self.mol_type) if v == i]
                for i in set(self.mol_type)
            }

    # Delete entries from all attributes at the supplied indices
    def del_idx(self, idx, reassign_ids=True, reassign_lammps_type=True):
        self.taffi_type = np.array(
            [v for _, v in enumerate(self.taffi_type) if _ not in idx]
        )
        self.element = np.array([v for _, v in enumerate(self.element) if _ not in idx])
        self.mass = np.array([v for _, v in enumerate(self.mass) if _ not in idx])
        self.charge = np.array([v for _, v in enumerate(self.charge) if _ not in idx])
        self.mol_id = np.array([v for _, v in enumerate(self.mol_id) if _ not in idx])
        self.mol_type = np.array(
            [v for _, v in enumerate(self.mol_type) if _ not in idx]
        )
        self.x = np.array([v for _, v in enumerate(self.x) if _ not in idx])
        self.y = np.array([v for _, v in enumerate(self.y) if _ not in idx])
        self.z = np.array([v for _, v in enumerate(self.z) if _ not in idx])
        self.vx = np.array([v for _, v in enumerate(self.vx) if _ not in idx])
        self.vy = np.array([v for _, v in enumerate(self.vy) if _ not in idx])
        self.vz = np.array([v for _, v in enumerate(self.vz) if _ not in idx])
        self.fx = np.array([v for _, v in enumerate(self.fx) if _ not in idx])
        self.fy = np.array([v for _, v in enumerate(self.fy) if _ not in idx])
        self.fz = np.array([v for _, v in enumerate(self.fz) if _ not in idx])
        self.q1 = np.array([v for _, v in enumerate(self.q1) if _ not in idx])
        self.q2 = np.array([v for _, v in enumerate(self.q2) if _ not in idx])
        self.q3 = np.array([v for _, v in enumerate(self.q3) if _ not in idx])
        self.q4 = np.array([v for _, v in enumerate(self.q4) if _ not in idx])
        # Reassign continuous ids from 1,2... onwards, not just delete entries.
        if reassign_ids:
            self.ids = [v for _, v in enumerate(self.ids) if _ not in idx]
            self.old2new_ids = {v: _ + 1 for _, v in enumerate(self.ids)}
            self.new2old_ids = {_ + 1: v for _, v in enumerate(self.ids)}
            self.ids = np.array(list(range(1, len(self.ids) + 1)))
        else:
            self.ids = np.array([v for _, v in enumerate(self.ids) if _ not in idx])

        # Reassigning need to be done for type as well
        if reassign_lammps_type:
            self.lammps_type = [
                v for _, v in enumerate(self.lammps_type) if _ not in idx
            ]
            self.old2new_lammps_type = {
                v: _ + 1 for _, v in enumerate(sorted(set(self.lammps_type)))
            }
            self.new2old_lammps_type = {
                self.old2new_lammps_type[_]: _ for _ in self.old2new_lammps_type.keys()
            }
            self.lammps_type = np.array(
                [self.old2new_lammps_type[v] for v in self.lammps_type]
            )
        else:
            self.lammps_type = np.array(
                [v for _, v in enumerate(self.lammps_type) if _ not in idx]
            )


class IntraModeList:
    """Class for all arrays of attributes related to a kind of intramolecular model. For example,
     bond/angle/dihedral/improper.

    Attributes
    -----------

    ids: numpy array
        List of IDs (as used in LAMMPS) of each mode during simulation

    lammps_type: numpy array
        List of mode-type (as used in LAMMPS) of each mode

    atom_ids: numpy array
        List of lists of IDs of atoms forming the modes

    Methods
    -----------

    append(ids=[], lammps_type=[], atom_ids=[])
        Appends entries to each of the attributes, i.e. adds values of new mode attributes to the existing\
        list 

    get_idx(lammps_type=[], atom_ids=[])
        For a non-empty list "l" of an attributes "a1", the method returns the index in the list where the 
        modes have values of a1 attribute in the input list l1. For example, this is useful to parse indices
        of modes of type 1 and 2, or indices of modes formed by specific atom(s).

    get_idx_group(lammps_type=None, atom_ids=None)
        For a given attribute, groups indices based on same value of attribute. Returns dictionary with\
        keys as attribute values (for example., mode types) and objects as the indices of modes with\
        that value (for example., indices of C-H mode)

    get_atom_ids(lammps_type=[], atom_ids=[])
        Same as get_idx, but instead of returning corresponding indices in list, it returns IDs of atoms\
        associatd with those modes. For instance, you wanted IDs of all atoms forming mode-type 1 you can\
        use this command. Alternately, you can get indices from get_idx and then parse the atom IDs at\
        those indices and flatten the array.

    get_atom_ids_group(lammps_type=None, atom_ids=None)
        Same as get_idx_group, but instead of returning corresponding indices in list, it returns IDs of\
        atoms associatd with those modes.       

    del_idx(idx,reassign_ids=1,reassign_atom_ids=1,reassign_lammps_type=1,old2new_ids={})
        Deletes entries from all mode attributes at the indices in "idx" list. This is useful if you want\
        to remove certain bonds from a simulation box. It will also reassign the new IDs by default, so\
        that they are conrinuous, i.e. 1,2,3... . It also reassigns the attributes lammps_type and atom IDs.\
        A dictionary of old atom IDs to new atom IDs (generated by the del_idx method in AtomList class)\
        can be supplied with old2new_id.  

    del_by_atom_ids(self,atom_ids,reassign_ids=1,reassign_atom_ids=1,reassign_lammps_type=1,old2new_ids={})
        Same as del_idx, but instead of deleting specific indices, user can supply the atom IDs which are\
        being removed and the method will remove entries from all attributes associated with bonds contain\
        ing these atoms.

    get_bonded_ids()
        Returns a dictionary of ids of bonded atoms. The keys of the dictionary are ids of atoms and objects\
         are the atom ids bonded to them through the mode of interest.

    Errors  
    -----------     

    "Error! Mismatch in the length of IntraModeList attributes. Exiting..."
        Self-explanatory


    Additional notes
    -----------

    If one wishes to add new attributes (eg., taffi type for bonds):
    Use the same format as for other attributes and modify all the methods below.

    Future work
    -----------     
    """

    def __init__(self, ids=[], lammps_type=[], atom_ids=[]):
        self.ids = np.array(ids)
        self.lammps_type = np.array(lammps_type)
        self.atom_ids = np.array(atom_ids)

        # Checks
        all_keys = ["ids", "lammps_type", "atom_ids"]
        Ls = {_: len(self.__dict__[_]) for _ in all_keys if len(self.__dict__[_]) != 0}
        if len(set([Ls[k] for k in Ls.keys()])) > 1:
            print(
                "Error! Mismatch in the length of IntraModeList attributes. Exiting..."
            )
            print(
                "Length of each attribute is: {}".format(
                    ", ".join(["{}: {}".format(k, Ls[k]) for k in Ls.keys()])
                )
            )
            quit()

    # Append info for new atoms
    def append(self, ids=[], lammps_type=[], atom_ids=[]):
        self.ids = np.array(list(self.ids) + ids)
        self.lammps_type = np.array(list(self.lammps_type) + lammps_type)
        self.atom_ids = np.array(self.atom_ids.tolist() + atom_ids)

    # Get idx based on a specific value of an attribute
    def get_idx(
        self, lammps_type=[], atom_ids=[]
    ):  # For atom_ids supply a list of indices directly
        if lammps_type != []:
            return [_ for _, v in enumerate(self.lammps_type) if v in lammps_type]
        if atom_ids != []:
            return sorted(
                set(
                    [
                        i
                        for i, v in enumerate(self.atom_ids)
                        if sum([1 for _ in v if _ in atom_ids])
                    ]
                )
            )

    # Get idx grouped according to an attribute
    def get_idx_group(self, lammps_type=None, x=None, atom_ids=None):
        if lammps_type != None:
            return {
                i: [_ for _, v in enumerate(self.lammps_type) if v == i]
                for i in set(self.lammps_type)
            }

    # Get atom IDs forming the bonds based on a specific value of an attribute
    def get_atom_ids(self, lammps_type=None):
        if lammps_type != None:
            return sorted(
                set(
                    [
                        _
                        for i, v in enumerate(self.lammps_type)
                        if v in lammps_type
                        for _ in self.atom_ids[i]
                    ]
                )
            )

    # Get atom IDs grouped according to an attribute
    def get_atom_ids_group(self, lammps_type=None):
        if lammps_type != None:
            return {
                i: sorted(
                    set(
                        [
                            _
                            for j, v in enumerate(self.lammps_type)
                            if v == i
                            for _ in self.atom_ids[j]
                        ]
                    )
                )
                for i in set(self.lammps_type)
            }

    # Delete entries from all attributes at the supplied indices
    def del_idx(
        self,
        idx,
        reassign_ids=True,
        reassign_atom_ids=True,
        reassign_lammps_type=True,
        old2new_ids={},
    ):
        # Reassign continuous ids from 1,2... onwards, not just delete entries.
        if reassign_ids:
            self.ids = [v for _, v in enumerate(self.ids) if _ not in idx]
            self.old2new_ids = {v: _ + 1 for _, v in enumerate(self.ids)}
            self.new2old_ids = {_ + 1: v for _, v in enumerate(self.ids)}
            self.ids = np.array(list(range(1, len(self.ids) + 1)))
        else:
            self.ids = np.array([v for _, v in enumerate(self.ids) if _ not in idx])

        # Reassign respective atom ids too if needed, and they can be supplied from the AtomList dictionary
        if reassign_atom_ids:
            if old2new_ids == {}:
                keep_atoms = sorted(
                    set(
                        [
                            _
                            for i, v in enumerate(self.atom_ids)
                            if i not in idx
                            for _ in v
                        ]
                    )
                )
                old2new_ids = {v: _ + 1 for _, v in enumerate(keep_atoms)}
            self.atom_ids = np.array(
                [
                    [old2new_ids[_] for _ in v]
                    for i, v in enumerate(self.atom_ids)
                    if i not in idx
                ]
            )
        else:
            self.atom_ids = np.array(
                [v for _, v in enumerate(self.atom_ids) if _ not in idx]
            )

        # Reassigning need to be done for type as well
        if reassign_lammps_type:
            self.lammps_type = [
                v for _, v in enumerate(self.lammps_type) if _ not in idx
            ]
            self.old2new_lammps_type = {
                v: _ + 1 for _, v in enumerate(sorted(set(self.lammps_type)))
            }
            self.new2old_lammps_type = {
                self.old2new_lammps_type[_]: _ for _ in self.old2new_lammps_type.keys()
            }
            self.lammps_type = np.array(
                [self.old2new_lammps_type[v] for v in self.lammps_type]
            )
        else:
            self.lammps_type = np.array(
                [v for _, v in enumerate(self.lammps_type) if _ not in idx]
            )

    # Delete entries from all attributes if the mode is formed by the atoms with indices supplied. Basically, remove all modes containing the atoms to be removed.
    def del_by_atom_ids(
        self,
        atom_ids,
        reassign_ids=True,
        reassign_atom_ids=True,
        reassign_lammps_type=True,
        old2new_ids={},
    ):
        idx = self.get_idx(atom_ids=atom_ids)
        self.del_idx(
            idx,
            reassign_ids=reassign_ids,
            reassign_atom_ids=reassign_atom_ids,
            reassign_lammps_type=reassign_lammps_type,
            old2new_ids=old2new_ids,
        )

    # Get a dictionary of ids of bonded atoms. The keys of the dictionary are ids of atoms and objects are the atom ids bonded to them
    def get_bonded_ids(self):
        bonded_ids = {}
        for atom_ids in self.atom_ids:
            for a in atom_ids:
                if a not in bonded_ids.keys():
                    bonded_ids[a] = []
                bonded_ids[a] += [a2 for a2 in atom_ids if a2 != a]

        return bonded_ids

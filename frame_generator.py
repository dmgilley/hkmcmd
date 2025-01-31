#!/bin/env python
# Author: Lin (lin1209@purdue.edu)
import sys, argparse, subprocess, os, time, math, shutil
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
from hybrid_mdmc.mol_classes import AtomList


def main(argv):

    for atom, timestep, box in frame_generator("plumed.0.lammpstrj", end=1):
        print(atom.x)
        print(timestep)


def frame_generator(
    name, start=0, end=-1, every=1, unwrap=False, adj_list=None, return_prop=False
):
    """Parser for LAMMPS trajectory files.


    Parameters
    ----------
    name: str
        The LAMMPS trajectory file to parse.

    start, end: int, optional
        Frames to start/end parsinf from each trajectory.
        (default: 1, last_frame)

    every: int, optional
        Every N frames will be parsed where N is this variable.
        (default: 1 = parse all)

    unwrap: boolean, optional
        When this flag is present, the geometries are unwrapped.
        Requires adj_list.
        (default: False)

    adj_list: list of lists, optional
        Adjacency list used for unwrapping the geometry.
        (default: None)

    return_prop: boolean, optional
        Returns the prop dictionary in addition to the default AtomList.
        (default: False)


    Returns
    -------
    atom: instance of AtomList.

    timestep: current frame.

    box: dimensions of the simulation cell.


    Additional Notes
    ----------------
    Attributes that are supported by lammps and AtomList
    :id, lammps_type, mass, charge, molecule id, coordinates (x,y,z), velocities (vx,vy,vz), forces (fx,fy,fz)
    This function use yield to save memory
    All properties (including those that supported in AtomList class) are stored in prop dictionary
    Default is to return AtomList, but if those omitted properties are desired,
    can use return_prop flag so that it returns prop dictionary in addition to original return.

    Usage example:
    --------------------------------------------------
    for atom,timestep,box in frame_generator(trjname):
        print(box) # print every frame's box
    --------------------------------------------------


    """

    # attributes that are supported by lammps and AtomList
    # key: lammps keyword, value: AtomList attribute
    candidates = {
        "id": "ids",
        "mol": "mol_id",
        "type": "lammps_type",
        "mass": "mass",
        "q": "charge",
        "xu": "x",
        "yu": "y",
        "zu": "z",
        "x": "x",
        "y": "y",
        "z": "z",
        "vx": "vx",
        "vy": "vy",
        "vz": "vz",
        "fx": "fx",
        "fy": "fy",
        "fz": "fz",
    }

    # Parse Trajectories
    frame = -1  # Frame counter (total number of frames in the trajectory)
    frame_count = -1  # Frame counter (number of parsed frames in the trajectory)
    frame_flag = 0  # Flag for marking the start of a parsed frame
    atom_flag = 0  # Flag for marking the start of a parsed Atom data block
    N_atom_flag = 0  # Flag for marking the place the number of atoms should be updated
    atom_count = 0  # Atom counter for each frame
    box_flag = 0  # Flag for marking the start of the box parse
    box_count = -1  # Line counter for keeping track of the box dimensions.

    # Open the trajectory file for reading
    with open(name, "r") as f:

        # Iterate over the lines of the original trajectory file
        for lines in f:

            fields = lines.split()

            # Find the start of each frame and check if it is included in the user-requested range
            if len(fields) == 2 and fields[1] == "TIMESTEP":
                frame += 1
                if (
                    frame >= start
                    and (frame <= end or end == -1)
                    and (frame - start) % every == 0
                ):
                    frame_flag = 1
                    frame_count += 1
                elif frame > end:
                    break
            # Parse commands for when a user-requested frame is being parsed
            if frame_flag == 1:

                # Header parse commands
                if atom_flag == 0 and N_atom_flag == 0 and box_flag == 0:
                    if len(fields) > 2 and fields[1] == "ATOMS":
                        atom_flag = 1
                        # read in all properties
                        prop = {_: [-1 for k in range(N_atoms)] for _ in fields[2:]}
                        ind_dict = {
                            _: fields.index(_) - 2 for _ in fields[2:]
                        }  # key: property name in lammpstrj, value: field index
                        # if frame_count == 0:
                        #   print("The following properties are found in lammpstrj file: {}".format(' '.join(list(ind_dict.keys()))))
                        continue
                    if len(fields) > 2 and fields[1] == "NUMBER":
                        N_atom_flag = 1
                        continue

                    if len(fields) > 2 and fields[1] == "BOX":
                        box_flag = 1
                        continue

                    if len(fields) == 1:
                        timestep = fields[0]
                        continue

                # Update the number of atoms in each frame
                if N_atom_flag == 1:

                    # Intialize total geometry of the molecules being parsed in this frame
                    # Note: from here forward the N_current acts as a counter of the number of atoms that have been parsed from the trajectory.
                    N_atoms = int(fields[0])
                    prop = {
                        candidates[_]: [-1 for k in range(N_atoms)] for _ in candidates
                    }
                    N_atom_flag = 0
                    continue

                # Read in box dimensions
                if box_flag == 1:
                    # initilize box array
                    if box_count == -1:
                        # cubic  box only has 2 columns
                        if len(fields) == 2:
                            box = np.zeros([3, 2])
                        # crystal has 3 columns
                        else:
                            box = np.zeros([3, 3])

                    box_count += 1
                    box[box_count] = [float(_) for _ in fields]

                    # After all box data has been parsed, save the box_lengths/2 to temporary variables for unwrapping coordinates and reset flags/counters
                    if box_count == 2:
                        box_count = -1
                        box_flag = 0
                    continue

                # Parse relevant atoms
                if atom_flag == 1:
                    for _ in prop:
                        if _ in ["id", "mol", "type"]:
                            prop[_][atom_count] = int(fields[ind_dict[_]])
                        else:
                            prop[_][atom_count] = float(fields[ind_dict[_]])
                    atom_count += 1

                    # Reset flags once all atoms have been parsed
                    if atom_count == N_atoms:

                        frame_flag = 0
                        atom_flag = 0
                        atom_count = 0

                        # Sort based on ids
                        prop["id"], sort_ind = list(
                            zip(
                                *sorted(
                                    [
                                        (k, count_k)
                                        for count_k, k in enumerate(prop["id"])
                                    ]
                                )
                            )
                        )
                        for _ in prop:
                            prop[_] = np.array([prop[_][k] for k in sort_ind])

                        # Populate atom with prop dictionary
                        # properties not supported by atomlis class will not be parsed
                        # if those omitted properties are still desired, use return_prop flag to return original prop dictionary
                        # initilize atomlist
                        atom = AtomList(ids=prop["id"])
                        for _ in prop:
                            if _ in list(candidates.keys()):
                                atom.__dict__[candidates[_]] = prop[_]

                        # Upwrap the geometry
                        if unwrap is True:
                            atom = unwrap_atomlist(atom, adj_list, box)

                        if return_prop:
                            yield atom, timestep, box, prop
                        else:
                            yield atom, timestep, box


def unwrap_atomlist(atomlist, adj_list, box):
    N_atoms = len(atomlist.ids)
    geo = np.zeros([N_atoms, 3])
    for i in range(N_atoms):
        geo[i] = np.array([atomlist.x[i], atomlist.y[i], atomlist.z[i]])
    geo = unwrap_geo(geo, adj_list, box)
    for i in range(N_atoms):
        atomlist.x[i] = geo[i][0]
        atomlist.y[i] = geo[i][1]
        atomlist.z[i] = geo[i][2]

    return atomlist


# Description: Performed the periodic boundary unwrap of the geometry
def unwrap_geo(geo, adj_list, box):
    # Unwrap the molecules using the adjacency matrix
    # Loops over the individual atoms and if they haven't been unwrapped yet, performs a walk
    # of the molecular graphs unwrapping based on the bonds.
    box = np.array(box).transpose()
    if len(box) == 3:  # crystal
        # calculate the actual triclinic cell boundaries and dimensions and the transformation matrix to transform from triclinic cell to orthogonal cell
        blo, bhi, sides, M = triclinic(
            deepcopy(box[0, :]), deepcopy(box[1, :]), deepcopy(box[2, :])
        )
        crystal = [blo, bhi, sides, M]
        # Convert coordinates in the frame of triclinic cell
        geo_final = convert_triclinic(geo, blo, M)
        L = np.repeat(np.array([sides]), geo.shape[0], axis=0)
        nL2, pL2 = -0.5 * L[0, :], 0.5 * L[0, :]
    else:
        crystal = []
        geo_final = deepcopy(geo)
        L = np.repeat(np.array([box[1, :] - box[0, :]]), geo.shape[0], axis=0)
        nL2, pL2 = -0.5 * L[0, :], 0.5 * L[0, :]

    if adj_list == [[]]:  # Just a single bead
        return geo, crystal
    # Apply minimum image convension to wrap the coordinates
    unwrapped = []
    for count_i, i in enumerate(geo):
        # Skip if this atom has already been unwrapped
        if count_i in unwrapped:
            continue

        # Proceed with a walk of the molecular graph
        # The molecular graph is cumulatively built up in the "unwrap" list and is initially seeded with the current atom
        else:
            unwrap = [count_i]  # list of indices to unwrap (next loop)
            unwrapped += [
                count_i
            ]  # list of indices that have already been unwrapped (first index is left in place)
            for j in unwrap:

                # new holds the index in geo_final of bonded atoms to j that need to be unwrapped
                new = [k for k in adj_list[j] if k not in unwrapped]

                # unwrap the new atoms
                for k in new:
                    unwrapped += [k]
                    dgeo = geo_final[k, :] - geo_final[j, :]

                    check = dgeo < nL2
                    while (check).any():
                        geo_final[k, :][check] += L[k, check]
                        dgeo = geo_final[k, :] - geo_final[j, :]
                        check = dgeo < nL2

                    check = dgeo > pL2
                    while (check).any():
                        geo_final[k, :][check] -= L[k, check]
                        dgeo = geo_final[k, :] - geo_final[j, :]
                        check = dgeo > pL2

                # append the just unwrapped atoms to the molecular graph so that their connections can be looped over and unwrapped.
                unwrap += new

    # Convert back to orthogonal coordinate frame
    if len(box) == 3:
        geo_final = convert_orthogonal(geo_final, blo, M)
    return geo_final


def triclinic(blo, bhi, tilt):
    # The function calculates the triclinic cell properties,
    # The side lengths and the angles between them
    # It returns the side lengths and the transformation matrix
    # to transform from triclinic to orthogonal
    xy, xz, yz = tilt
    # The next four lines are necessary if you are reading the dimensions of the box
    # from lammps, it can be discarded if you are readibng from data file
    blo[0] = blo[0] - min(0, xy, xz, xy + xz)
    bhi[0] = bhi[0] - max(0, xy, xz, xy + xz)
    blo[1] = blo[1] - min(0, yz)
    bhi[1] = bhi[1] - max(0, yz)

    # Calculates everything
    lx, ly, lz = bhi - blo
    a = lx
    b = math.sqrt(ly**2 + xy**2)
    c = math.sqrt(lz**2 + xz**2 + yz**2)
    M = [[1, xy / b, xz / c], [0, ly / b, yz / c], [0, 0, lz / c]]

    return [blo, bhi, np.array([a, b, c]), np.array(M)]


def convert_triclinic(traj, blo, M):
    # Converts from orthodgonal to triclinic
    traj_triclinic = np.zeros(traj.shape)
    for a in range(traj.shape[0]):
        traj[a, :] = traj[a, :] - blo
        traj_triclinic[a, :] = np.linalg.solve(M, traj[a, :])
    return traj_triclinic


def convert_orthogonal(traj_triclinic, blo, M):
    # Converts from triclinic to orthogonal
    traj = np.zeros(traj_triclinic.shape)
    for a in range(traj.shape[0]):
        traj[a, :] = np.dot(M, traj_triclinic[a, :])
        traj[a, :] = traj[a, :] + blo
    return traj


if __name__ == "__main__":
    main(sys.argv[1:])



============
Installation
============

Requirements
============

* Python_ 3.13 or newer
* NumPy_ 2.2 or newer
* SciPy_ 1.15 or newer
* Pandas_ 2.2 or newer
* Sckit-learn_ 1.6 or newer
* Matplotlib_ 3.10 or newer

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _Pandas: https://pandas.pydata.org/
.. _Scikit-learn: https://scikit-learn.org/stable/
.. _Matplotlib: https://matplotlib.org/

Installation From Source
========================

Clone the source into your directory of choice::

    $ cd MY_DIRECTORY/
    $ git clone https://github.com/Savoie-Research-Group/hkmcmd.git

Then, use pip to install just like a package from pypi

::

    $ pip install .

or, if you would like to install in editable mode for tweaking

::

    $ pip install -e .

Make sure all the tests are passing

::

    $ python -m unittest discover -s tests

Finally, try running the example notebooks (see below).

==================
Naming Conventions
==================
The naming conventions are constructed such that it is (hopefully) simple to run multiple replicates of a given system. To accomplish this, two names are used throughout, "SYSTEM" and "PREFIX." The files that describe the HkMCMD and MD settings are prepended with SYSTEM (SYSTEM.json and SYSTEM.in.settings, see below), whereas individual replicates are designated with PREFIX. As explained below, SYSTEM.json and SYSTEM.in.settings must be created manually/interactively. All other necessary files can be created with provided scripts. This naming convention thus allows for creating the two system files by hand, followed by automated creation of individual replicate files, to avoid tedious repeating of file creation. For example, if running two replicates of a methane combustion, the names could be...

Replictate 1


::

    SYSTEM: "methane_combustion"
    PREFIX: "methane_combustion.replicate1"


Replictate 2

::

    SYSTEM: "methane_combustion"
    PREFIX: "methane_combustion.replicate2"


If desired, SYSTEM and PREFIX can be set to the same. For example,

::

    SYSTEM: "SEI_investigation"
    PREFIX: "SEI_investigation"

is a perfectly acceptable option for running a single replicate of HkMCMD to simulate a Solid Electrolyte Interphase.

=====
Files
=====

general HkMCMD
==============

1. SYSTEM.json

    * user-created prior to simulation
    * main settings file that holds all information needed to perform HkMCMD

2. PREFIX.system_state.json

    * created with hkmcmd.initialize.py, updated at the end of every call to hkmcmd.run.hkmcmd.py
    * contains information regarding the state of the system at each reactive step
    * e.g. the molecule assignment and molecule type of each atom at every reactive step

3. PREFIX.summary.txt

    * created with hkmcmd.summarize.py
    * summarizes the system state at each reactive step
    * e.g. instead of listing every atom's molecule ID and molecule type, it lists the total number of each type of molecule
    * allows for fairly detailed analysis while significantly decreasing the size of file that must be transfered around and read by the analysis scripts

Molecular Dynamics
==================

1. PREFIX.in.init

    * LAMMPS formatted input script detailing the MD run
    * created with hkmcmd.initialize.py, and overwritten every time hkmcmd.run.hkmcmd.py is called

2. PREFIX.in.data

    * LAMMPS formatted data file containing particle-specific information (position, molecule ID, bonds, etc.)
    * created with hkmcmd.initialize.py, and *overwritten every time hkmcmd.run.hkmcmd.py is called*

3. PREFIX.end.data

    * LAMMPS formatted data file containing particle-specific information (position, molecule ID, bonds, etc.)
    * created by LAMMPS at the end of an MD simulation

4. PREFIX.diffusion.lammpstrj

    * LAMMPS "dump" file containing particle information at specified steps (position, velocity, etc.)
    * created by LAMMPS during MD simulation

5. various others not used by HkMCMD, but potentially useful to diagnostics

    * **NOTE:** all LAMMPS files used by/created from the initial MD run (i.e. prior to the HkMCMD loop) are copied as PREFIX_prep.* for diagnostic purposes

Diffusion Scaling
=================

1. PREFIX.difusion.txt

    * calculated diffusion rates between all system voxels
    * generated with hkmcmd.diffusion.py
    * can be appended on each diffusive cycle

2. PREFIX.diffusion.log

    * log file for tracking calculation of diffusion rates
    * generated with hkmcmd.diffusion.py

3. PREFIX.direct_transition_rates.txt

    * direct transition rates between voxels
    * generated with hkmcmd.diffusion.py

4. PREFIX.mvabfa.txt

    * "molecular voxel assignment by frame array" file; holds the voxel ID in which each molecule resides per frame
    * gerenated with hkmcmd.diffusion.py

5. PREFIX.msd.txt

    * mean (squared) displacement of each molecule time
    * generated with hkmcmd.MSD.py

Quick Start
===========

HkMCMD is built as a wrapper that operates around an MD engine. Current functionality supports the LAMMPS MD engine. A bash script is used to call appropriate programs in the appropriate order. The order of operations are as follows...
( input_file(s) > script > output_file(s) )

1. manual creation of initial MD force field file

    N/A > manual creation > SYSTEM.in.settings

2. manual creation of the HkMCMD settings file (this can be done using the helper notebook "create.ipynb")

    N/A > manual creation *or* /examples/kmctoy/create.ipynb > SYSTEM.json

3. create LAMMPS input files for the initial MD run and initialize the system state file

    SYSTEM.json > hkmcmd.initialize.py > PREFIX.in.data
                                         PREFIX.in.init
                                         PREFIX.system_state.json

4. run the initial MD

    SYSTEM.in.settings > MD engine > PREFIX.end.data
    PREFIX.in.data                   PREFIX.diffusion.lammpstrj
    PREFIX.in.init

5. calculate diffusion rates (optional)

    SYSTEM.json                > hkmcmd.diffusion.py > PREFIX.diffusion.log
    PREFIX.end.data                                    PREFIX.mvabfa.txt
    PREFIX.diffusion.lammpstrj                         PREFIX.direct_transition_rates.txt
                                                       PREFIX.diffusion.txt

6. calculate MSD (optional)

        SYSTEM.json                > hkmcmd.MSD.py > PREFIX.msd.txt
        PREFIX.end.data
        PREFIX.diffusion.lammpstrj



7. begin HkMCMD loop

    7a. perform kMC reaction selections

                            SYSTEM.json > hkmcmd.run.hkmcmd.py > PREFIX.system_state.json
               PREFIX.system_state.json                          PREFIX.in.date
                        PREFIX.end.data                          PREFIX.in.init
        PREFIX.diffusion.txt (optional)

    7b. perform MD

        SYSTEM.in.settings > MD engine > PREFIX.end.data
            PREFIX.in.data               PREFIX.diffusion.lammpstr
            PREFIX.in.init

    7c. repeat step 7 until "done"


8. create a summary file (optional)

        SYSTEM.json              > hkmcmd.summarize.py > PREFIX.summary.txt
        PREFIX.system_state.json


/examples/kmctoy/
=================

This example walks through running a simple HkMCMD simulation. The system consists of condensed-phase LJ particles undergoing reversible dimerization, 2A <-> A_2. Reaction scaling and diffusion scaling are NOT included in this example. This simulation can be directly executed with a SLURM submission scheduler after editing the kmctoy.sh submission file to align with an appropriate HPC cluster. An expected output file is provided for comparison, however perfect agreement will not be achieved due to the stochastic nature of the algorithm. An interactive jupyter notebook is provided for post-run analysis.

Included in this directory are the following files:

Input Files
-----------

* kmctoy.json
* kmctoy.in.settings
* kmctoy.sh

Output Files
------------

* kmctoy_ref.summary.txt
* kmctoy_ref.pdf (reference plot)

Jupyter Notebooks
-----------------

* create.ipynb (create the necessary files to run HkMCMD)
* analysis.ipynb (plot the results)

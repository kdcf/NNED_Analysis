# NNED_Analysis
Computes the normalized number of electrons displaced (NNED) metric between a Hartree-Fock and DFT density. Interfaces with the PySCF package.

Use either script with a working copy of the free and open PySCF package. User inputs are controlled near the top of the enclosed python script, and include:

ATOM --> The XYZ coordinates

CHARGE --> The net system charge

SPIN --> Number of alpha electrons minus number of beta electrons (not multiplicity)

FUNCTIONAL --> The desired XC functional to be used for NNED analysis

Scripts and their outputs:

nned_analysis.py --> Conducts basic NNED analysis, prints the NNED metric.
nned_analysis_w_cubes.py --> Conducts basic NNED analysis and prints significant difference-density natural orbitals (NED > 0.0001) to cube files for visualization.

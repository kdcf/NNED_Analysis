# NNED_Analysis
Computes the normalized number of electrons displaced (NNED) metric between a Hartree-Fock and DFT density. Interfaces with the PySCF package.

Use with a working copy of PySCF. User inputs are controlled near the top of the enclosed python script, and include:

ATOM --> The XYZ coordinates
CHARGE --> The net system charge
SPIN --> Number of alpha electrons minus number of beta electrons (not multiplicity)
FUNCTIONAL --> The desired XC functional to be used for NNED analysis

"""
Difference Density Analysis: NNED metric
=========================================
Implements Equations 1 and 2 from the Methods section:

    ΔP = P_KS - P_HF                                     (Eq. 1)

    U† S^(1/2) ΔP S^(1/2) U = δ                          (Eq. 2)

The eigenvalues δ represent electron occupation number shifts between
the HF and KS-DFT SCF solutions. The sum of positive eigenvalues gives
the Number of Electrons Displaced (NED), and normalizing by the total
electron count gives the NNED.

These routines assume unrestricted orbitals.

Usage: edit the MOLECULE, BASIS, and FUNCTIONAL sections below.
"""

import numpy as np
from pyscf import gto, scf, dft

# USER INPUT
ATOM = """
B        0.0000000000      0.0000000000     0.0000
N        0.0000000000      0.0000000000      1.336
"""
BASIS    = "cc-pVTZ"
CHARGE   = 0
SPIN     = 2          # Nalpha - Nbeta (0 = singlet)
FUNCTIONAL = "PW91"   # any PySCF-supported XC string

def build_mol(atom, basis, charge, spin):
    mol = gto.Mole()
    mol.symmetry = False
    mol.atom   = atom
    mol.basis  = basis
    mol.charge = charge
    mol.spin   = spin
    mol.verbose = 3
    mol.build()
    return mol


def run_hf(mol):
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-8
    mf.kernel()
    assert mf.converged, "HF did not converge!"
    return mf


def run_ks(mol, functional, dm0):
    mf = dft.UKS(mol)
    mf.conv_tol = 1e-8
    mf.xc = functional
    mf.grids.atom_grid = (99, 590) # Use 99 radial shells and 590 angular points
    mf.grids.prune = None
    mf.kernel(dm0=dm0)
    assert mf.converged, f"KS-DFT ({functional}) did not converge!"
    return mf


def ao_density_matrix(mf):
    """Return the 1-particle density matrix in the AO basis."""
    dm = mf.make_rdm1()

    return dm[0],dm[1]


def overlap_matrix(mol):
    """Return the AO overlap matrix S."""
    return mol.intor("int1e_ovlp")


def difference_density_nned(P_ks, P_hf, S):
    """
    Compute eigenvalues of the symmetrically orthogonalized ΔP
    and derive NED / NNED.

    Implements Eq. 2:  U† S^(1/2) ΔP S^(1/2) U = δ

    Parameters
    ----------
    P_ks, P_hf : ndarray (nao, nao)   density matrices in AO basis
    S          : ndarray (nao, nao)   AO overlap matrix
    n_elec     : int                  total number of electrons

    Returns
    -------
    delta      : ndarray  eigenvalues of the orthogonalised ΔP
    ned        : float    number of electrons displaced (sum of δ > 0)
    """
    dP = P_ks - P_hf                          # Eq. 1

    # S^(1/2) via symmetric diagonalization
    s_evals, s_evecs = np.linalg.eigh(S)
    S_half = s_evecs @ np.diag(np.sqrt(s_evals)) @ s_evecs.T

    # Symmetrically orthogonalized ΔP  (Eq. 2)
    dP_orth = S_half @ dP @ S_half

    # Eigenvalue decomposition
    #delta, U = np.linalg.eigvalsh(dP_orth)
    delta, U = np.linalg.eigh(dP_orth)

    # NED = sum of positive eigenvalues
    ned  = float(np.sum(delta[delta > 0]))

    return delta, U, ned

def write_ddno_cube(mol, S, C_ao, delta, spin_label, out_dir="."):
    """ Writes Cubes for visualization """
    from pyscf.tools import cubegen
    import os

    os.makedirs(out_dir, exist_ok=True)

    nao = S.shape[0]
    assert C_ao.shape == (nao, nao), "C_ao must be (nao, nao)"

    for i in range(nao):
        coeff = C_ao[:, i]   # AO coefficients for DDNO i
        d     = delta[i]
        if(abs(d) > 1.e-4): # print only significant ones
            sign  = "pos" if d >= 0 else "neg"
            fname = os.path.join(out_dir, f"ddno_{spin_label}_{i:04d}_{sign}_delta{d:+.4f}.cube")

            # cubegen.orbital expects a (nao,) coefficient vector
            cubegen.orbital(mol, fname, coeff, margin=8.0)
            print(f"  Wrote {fname}  (δ = {d:+.6f})")

def ao_coeffs_from_orth_eigvecs(S, U):
    """
    Back-transform orthonormal eigenvectors U (in S^(1/2) basis) to AO basis.

        C_AO = S^(-1/2) U

    Parameters
    ----------
    S : ndarray (nao, nao)   AO overlap matrix
    U : ndarray (nao, nao)   eigenvectors from eigh( S^(1/2) ΔP S^(1/2) )

    Returns
    -------
    C_ao : ndarray (nao, nao)   columns are DDNOs expanded in the AO basis
    """
    s_evals, s_evecs = np.linalg.eigh(S)
    S_half_inv = s_evecs @ np.diag(1.0 / np.sqrt(s_evals)) @ s_evecs.T
    return S_half_inv @ U

# MAIN
if __name__ == "__main__":
    mol = build_mol(ATOM, BASIS, CHARGE, SPIN)
    n_elec = mol.nelectron

    print("\n" + "="*60)
    print("  Running HF ...")
    print("="*60)
    mf_hf = run_hf(mol)
    P_hf_a, P_hf_b  = ao_density_matrix(mf_hf)
    dm_hf = mf_hf.make_rdm1()

    print("\n" + "="*60)
    print(f"  Running KS-DFT  [ xc = {FUNCTIONAL} ] ...")
    print("="*60)
    mf_ks = run_ks(mol, FUNCTIONAL, dm0=dm_hf)
    P_ks_a, P_ks_b  = ao_density_matrix(mf_ks)

    S = overlap_matrix(mol)

    delta_a, U_a, ned_a = difference_density_nned(P_ks_a, P_hf_a, S)
    delta_b, U_b, ned_b = difference_density_nned(P_ks_b, P_hf_b, S)

    ned_total  = ned_a + ned_b
    nned_total = ned_total / n_elec

    # printing results
    print("\n" + "="*60)
    print("  DIFFERENCE DENSITY ANALYSIS")
    print(f"  Molecule : {ATOM.strip()}")
    print(f"  Basis    : {BASIS}")
    print(f"  XC       : {FUNCTIONAL}")
    print(f"  N_elec   : {n_elec}")
    print("="*60)

    print("\n  Alpha Eigenvalue spectrum of S^(1/2) ΔP S^(1/2)  (δ):")
    print(f"  {'Index':>6}  {'δ':>14}")
    print("  " + "-"*24)
    for i, d in enumerate(delta_a):
        marker = " <-- δ+" if d > 0 else (" <-- δ-" if d < 0 else "")
        print(f"  {i:>6}  {d:>14.8f}{marker}")
    print("\n  Beta  Eigenvalue spectrum of S^(1/2) ΔP S^(1/2)  (δ):")
    print(f"  {'Index':>6}  {'δ':>14}")
    print("  " + "-"*24)
    for i, d in enumerate(delta_b):
        marker = " <-- δ+" if d > 0 else (" <-- δ-" if d < 0 else "")
        print(f"  {i:>6}  {d:>14.8f}{marker}")

    print(f"\n  Sum of positive eigenvalues (NED)  = {ned_total:.8f}")
    print(f"  Total electrons                    = {n_elec}")
    print(f"  NNED = NED / N_elec               = {nned_total:.8f}")
    print("="*60 + "\n")

    C_ao_a = ao_coeffs_from_orth_eigvecs(S, U_a)
    C_ao_b = ao_coeffs_from_orth_eigvecs(S, U_b)

    write_ddno_cube(mol, S, C_ao_a, delta_a, spin_label="alpha", out_dir="ddno_cubes")
    write_ddno_cube(mol, S, C_ao_b, delta_b, spin_label="beta",  out_dir="ddno_cubes")

    print(f"\n  All cube files written to ./ddno_cubes/")
    print("="*60 + "\n")


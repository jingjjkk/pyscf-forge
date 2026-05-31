"""Shared Z-vector solver for SASF delta_A orbital response gradient.

Replaces 3*N_atom CPHF linear solves with ONE Z-vector solve + dot products.

Theory: The orbital response of energy E is dE/dR = direct + orbital.
  dE_orbital = Tr(M @ kappa) where kappa = kappa_anti + kappa_sym
  H @ kappa_anti = -(gfix + H @ kappa_sym)
  => solve H^T @ z = M_vec (once), then
     dE_orbital = z^T @ (-(gfix + H @ kappa_sym)) + Tr(M @ kappa_sym)
"""

import numpy as np
from pyscf import lib

from ._block_analytic_hf import (
    _canonical_roks_pairs,
    _pack_roks_canonical_residual,
    _anti_mo_from_roks_canonical_vec,
    _roks_general_orbital_action_hf,
)
from ._direct import _full_jk_deriv_atom


# ---------------------------------------------------------------------------
#  Hessian builder
# ---------------------------------------------------------------------------

def build_roks_hessian(tdobj):
    """Build the full ROKS orbital Hessian in the canonical variable space.

    Returns
    -------
    hmat : (nvar, nvar) ndarray
    pairs : list of (p, q, name) tuples
    nmo : int
    """
    nmo = tdobj._scf.mo_coeff.shape[1]
    pairs = _canonical_roks_pairs(tdobj)
    nvar = len(pairs)
    eye = np.eye(nvar)
    hmat = np.column_stack([
        _roks_general_orbital_action_hf(
            tdobj, pairs,
            _anti_mo_from_roks_canonical_vec(nmo, pairs, eye[i]),
        )
        for i in range(nvar)
    ])
    return hmat, pairs, nmo


# ---------------------------------------------------------------------------
#  M-vector packing
# ---------------------------------------------------------------------------

def pack_mvec(m_full, pairs):
    """Pack anti-symmetric part of M-matrix into canonical pair vector.

    For each pair i = (p, q, name), mvec[i] = M[p,q] - M[q,p].
    """
    r = m_full - m_full.T
    return np.array([r[p, q] for p, q, _name in pairs])


# ---------------------------------------------------------------------------
#  Z-vector solve
# ---------------------------------------------------------------------------

def solve_zvec(hmat, mvec):
    """Solve H^T @ z = mvec for the Z-vector."""
    return np.linalg.solve(hmat.T, mvec)


# ---------------------------------------------------------------------------
#  Perturbation RHS builder
# ---------------------------------------------------------------------------

def _perturbation_rhs(tdobj, ia, xyz, pairs, eri1, hcore_deriv, s1):
    mol = tdobj.mol
    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    offsetdic = mol.offset_nr_by_atom()
    p0, p1 = offsetdic[ia][2:]

    dm_a = mo_coeff[:, mo_occ > 0] @ mo_coeff[:, mo_occ > 0].T
    dm_b = mo_coeff[:, mo_occ == 2] @ mo_coeff[:, mo_occ == 2].T

    j1a, k1a = _full_jk_deriv_atom(mol, dm_a, ia, eri1=eri1)
    j1b, k1b = _full_jk_deriv_atom(mol, dm_b, ia, eri1=eri1)

    dfock_a = hcore_deriv(ia)[xyz] + j1a[xyz] + j1b[xyz] - k1a[xyz]
    dfock_b = hcore_deriv(ia)[xyz] + j1a[xyz] + j1b[xyz] - k1b[xyz]

    gfix = _pack_roks_canonical_residual(
        pairs,
        mo_coeff.T @ dfock_a @ mo_coeff,
        mo_coeff.T @ dfock_b @ mo_coeff,
    )

    s1ao = np.zeros((mol.nao, mol.nao))
    s1ao[p0:p1] += s1[xyz, p0:p1]
    s1ao[:, p0:p1] += s1[xyz, p0:p1].T
    ksym = -0.5 * (mo_coeff.T @ s1ao @ mo_coeff)

    h_resp = _roks_general_orbital_action_hf(tdobj, pairs, ksym)

    return gfix, ksym, h_resp


# ---------------------------------------------------------------------------
#  Orbital response gradient via Z-vector
# ---------------------------------------------------------------------------

def zvec_orbital_grad(td_grad, tdobj, hmat, pairs, m_full, zvec, atmlst=None):
    """Compute orbital response gradient using pre-solved Z-vector.

    Parameters
    ----------
    td_grad : TDDFT gradient object
    tdobj : TDDFT instance
    hmat : (nvar, nvar) ndarray — orbital Hessian (for verification)
    pairs : list — canonical pair list
    m_full : (nmo, nmo) ndarray — total M-matrix
    zvec : (nvar,) ndarray — solved Z-vector
    atmlst : optional atom list

    Returns
    -------
    de_orbital : (n_atom, 3) ndarray
    """
    mol = tdobj.mol
    mf = tdobj._scf

    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)

    eri1 = mol.intor('int2e_ip1', comp=3)
    hcore_deriv = mf.nuc_grad_method().hcore_generator(mol)
    s1 = mf.nuc_grad_method().get_ovlp(mol)

    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            gfix, ksym, h_resp = _perturbation_rhs(
                tdobj, ia, xyz, pairs, eri1, hcore_deriv, s1,
            )
            rhs = -(gfix + h_resp)
            de[k, xyz] = zvec @ rhs + np.trace(m_full @ ksym)
    return de

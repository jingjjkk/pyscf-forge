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

def make_roks_hessian_action(tdobj, pairs=None):
    """Build a matrix-free canonical ROKS Hessian action.

    The action is algebraically equivalent to
    ``_roks_general_orbital_action_hf`` but delegates the linear Fock response
    to ``mf.gen_response``.  The outer ROKS packing/projection is kept local
    because PySCF's UCPHF layout does not represent the spatial CV constraint.
    """
    mf = tdobj._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = mo_coeff.shape[1]
    if pairs is None:
        pairs = _canonical_roks_pairs(tdobj)

    dm_a = mo_coeff[:, mo_occ > 0] @ mo_coeff[:, mo_occ > 0].T
    dm_b = mo_coeff[:, mo_occ == 2] @ mo_coeff[:, mo_occ == 2].T
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, (dm_a, dm_b), hermi=1)
    fock_a = hcore + vj[0] + vj[1] - vk[0]
    fock_b = hcore + vj[0] + vj[1] - vk[1]
    fmo_a = mo_coeff.T @ fock_a @ mo_coeff
    fmo_b = mo_coeff.T @ fock_b @ mo_coeff
    occ_a = np.zeros_like(mo_occ, dtype=float)
    occ_b = np.zeros_like(mo_occ, dtype=float)
    occ_a[mo_occ > 0] = 1.0
    occ_b[mo_occ == 2] = 1.0
    vresp = mf.gen_response(hermi=1)

    def action_one(vec):
        kappa = _anti_mo_from_roks_canonical_vec(nmo, pairs, vec)
        ddm_mo_a = kappa * occ_a[None, :] + occ_a[:, None] * kappa.T
        ddm_mo_b = kappa * occ_b[None, :] + occ_b[:, None] * kappa.T
        ddm_a = mo_coeff @ ddm_mo_a @ mo_coeff.T
        ddm_b = mo_coeff @ ddm_mo_b @ mo_coeff.T
        vfock_a, vfock_b = vresp(np.stack((ddm_a, ddm_b)))

        dfmo_a = kappa.T @ fmo_a + fmo_a @ kappa
        dfmo_a += mo_coeff.T @ vfock_a @ mo_coeff
        dfmo_b = kappa.T @ fmo_b + fmo_b @ kappa
        dfmo_b += mo_coeff.T @ vfock_b @ mo_coeff
        return _pack_roks_canonical_residual(pairs, dfmo_a, dfmo_b)

    def action(vec):
        vec = np.asarray(vec)
        if vec.ndim == 1:
            return action_one(vec)
        return np.asarray([action_one(v) for v in vec])

    return action, pairs, nmo


def make_roks_hessian_transpose_action(tdobj, pairs=None):
    """Build the adjoint action for the packed canonical ROKS Hessian."""
    mf = tdobj._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = mo_coeff.shape[1]
    if pairs is None:
        pairs = _canonical_roks_pairs(tdobj)

    dm_a = mo_coeff[:, mo_occ > 0] @ mo_coeff[:, mo_occ > 0].T
    dm_b = mo_coeff[:, mo_occ == 2] @ mo_coeff[:, mo_occ == 2].T
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, (dm_a, dm_b), hermi=1)
    fock_a = hcore + vj[0] + vj[1] - vk[0]
    fock_b = hcore + vj[0] + vj[1] - vk[1]
    fmo_a = mo_coeff.T @ fock_a @ mo_coeff
    fmo_b = mo_coeff.T @ fock_b @ mo_coeff
    occ_a = np.zeros_like(mo_occ, dtype=float)
    occ_b = np.zeros_like(mo_occ, dtype=float)
    occ_a[mo_occ > 0] = 1.0
    occ_b[mo_occ == 2] = 1.0
    vresp = mf.gen_response(hermi=1)

    def unpack_adjoint_source(vec):
        g_alpha = np.zeros((nmo, nmo))
        g_beta = np.zeros_like(g_alpha)
        for val, (p, q, name) in zip(vec, pairs):
            if name in ('cc', 'oo', 'vv'):
                g_alpha[p, q] += 0.5 * val
                g_beta[p, q] += 0.5 * val
            elif name == 'co':
                g_beta[p, q] += val
            elif name == 'cv':
                g_alpha[p, q] += val
                g_beta[p, q] += val
            elif name == 'ov':
                g_alpha[p, q] += val
            else:
                raise RuntimeError('Unknown ROKS canonical pair type %s' % name)
        return g_alpha, g_beta

    def pack_adjoint_kappa(grad_kappa):
        return np.array([
            grad_kappa[p, q] - grad_kappa[q, p]
            for p, q, _name in pairs
        ])

    def action_one(vec):
        g_alpha, g_beta = unpack_adjoint_source(vec)

        grad_kappa = fmo_a @ (g_alpha + g_alpha.T)
        grad_kappa += fmo_b @ (g_beta + g_beta.T)

        q_alpha = mo_coeff @ g_alpha @ mo_coeff.T
        q_beta = mo_coeff @ g_beta @ mo_coeff.T
        q_alpha = 0.5 * (q_alpha + q_alpha.T)
        q_beta = 0.5 * (q_beta + q_beta.T)
        v_alpha, v_beta = vresp(np.stack((q_alpha, q_beta)))
        u_alpha = mo_coeff.T @ v_alpha @ mo_coeff
        u_beta = mo_coeff.T @ v_beta @ mo_coeff

        grad_kappa += u_alpha * occ_a[None, :]
        grad_kappa += u_alpha.T * occ_a[None, :]
        grad_kappa += u_beta * occ_b[None, :]
        grad_kappa += u_beta.T * occ_b[None, :]
        return pack_adjoint_kappa(grad_kappa)

    def action(vec):
        vec = np.asarray(vec)
        if vec.ndim == 1:
            return action_one(vec)
        return np.asarray([action_one(v) for v in vec])

    return action, pairs, nmo


def build_roks_hessian_reference(tdobj):
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


def build_roks_hessian(tdobj):
    """Build the full ROKS orbital Hessian using the matrix-free action."""
    action, pairs, nmo = make_roks_hessian_action(tdobj)
    nvar = len(pairs)
    eye = np.eye(nvar)
    hmat = np.column_stack([action(eye[i]) for i in range(nvar)])
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


def _roks_canonical_precond_diag(tdobj, pairs, level_shift=0.0):
    """Orbital-energy diagonal preconditioner for canonical ROKS pairs."""
    mf = tdobj._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm_a = mo_coeff[:, mo_occ > 0] @ mo_coeff[:, mo_occ > 0].T
    dm_b = mo_coeff[:, mo_occ == 2] @ mo_coeff[:, mo_occ == 2].T
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, (dm_a, dm_b), hermi=1)
    fock_a = hcore + vj[0] + vj[1] - vk[0]
    fock_b = hcore + vj[0] + vj[1] - vk[1]
    eps_a = np.diag(mo_coeff.T @ fock_a @ mo_coeff)
    eps_b = np.diag(mo_coeff.T @ fock_b @ mo_coeff)
    eps_c = 0.5 * (eps_a + eps_b)

    diag = np.empty(len(pairs))
    for i, (p, q, name) in enumerate(pairs):
        if name in ('cc', 'oo', 'vv'):
            diag[i] = eps_c[p] - eps_c[q]
        elif name == 'co':
            diag[i] = eps_b[p] - eps_b[q]
        elif name == 'cv':
            diag[i] = eps_a[p] - eps_a[q] + eps_b[p] - eps_b[q]
        elif name == 'ov':
            diag[i] = eps_a[p] - eps_a[q]
        else:
            raise RuntimeError('Unknown ROKS canonical pair type %s' % name)

    if level_shift:
        diag = diag + level_shift
    small = np.abs(diag) < 1e-8
    if np.any(small):
        diag = diag.copy()
        diag[small] = np.where(diag[small] < 0, -1e-8, 1e-8)
    return diag


def solve_zvec_krylov(action, pairs, tdobj, mvec, tol=1e-12, max_cycle=None,
                      level_shift=0.0, lindep=1e-22, verbose=0):
    """Solve ``action(z) = mvec`` with the PySCF CPHF Krylov pattern."""
    diag = _roks_canonical_precond_diag(tdobj, pairs, level_shift=level_shift)
    zbase = mvec / diag
    if max_cycle is None:
        max_cycle = len(mvec)

    def aop(z):
        z = np.asarray(z)
        if z.ndim == 1:
            return action(z) / diag - z
        return np.asarray([action(zi) / diag - zi for zi in z])

    zvec = lib.krylov(
        aop, zbase, tol=tol, max_cycle=max_cycle,
        lindep=lindep, hermi=False, verbose=verbose,
    )
    return np.asarray(zvec).reshape(-1)


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

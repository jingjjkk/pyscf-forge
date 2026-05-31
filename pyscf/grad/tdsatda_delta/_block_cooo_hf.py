"""SASF CO-OO block analytic gradient (HF/ROKS, canonical response)."""

import numpy as np
from pyscf import lib

from ._blocks import (
    _hybrid_coefficients,
    _mo_pair_dm,
    _sasf_orbitals,
    make_sasf_blocks,
)
from ._q_rhs import (
    _add_eri_term_q,
    _add_fock_response_q,
    _add_fock_term,
)
from ._block_analytic_hf import (
    _roks_canonical_response_kappas_hf,
)
from ._direct import (
    _add_j_bilinear_ip1,
    _full_spin_fock_derivs_by_atom,
    _general_eri,
)


def _spin_coefficients(si):
    if si <= 0.5:
        raise NotImplementedError('Si > 1/2 required for SASF CO-OO block')
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1
    chi = 1.0 / np.sqrt(2 * si * (2 * si - 1))
    return zeta, chi


def _cooo_t_beta_alpha(tdobj, xy):
    """Return pair-level beta CO and alpha OC coefficient matrices."""
    b = make_sasf_blocks(tdobj, xy)
    zeta, chi = _spin_coefficients(b.si)
    tr_oo = float(np.trace(b.x_oo))
    t_beta_co = 2 * chi * tr_oo * b.x_co
    t_alpha_oc = -2 * zeta * lib.einsum('iu,wu->wi', b.x_co, b.x_oo)
    return t_beta_co, t_alpha_oc


def cooo_block_energy(tdobj, xy):
    """Full off-diagonal ``CO-OO + OO-CO`` HF correction energy."""
    b = make_sasf_blocks(tdobj, xy)
    zeta, _ = _spin_coefficients(b.si)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    mf = tdobj._scf
    fock = mf.get_fock()
    t_beta_co, t_alpha_oc = _cooo_t_beta_alpha(tdobj, xy)

    e = float(lib.einsum('iu,iu', t_beta_co, orbcs.T @ fock.fockb @ orbos))
    e += float(lib.einsum('wi,wi', t_alpha_oc, orbos.T @ fock.focka @ orbcs))

    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    if hybrid:
        e += _cooo_hfx_energy(tdobj, xy, coeff=-2 * hyb * zeta)
        if omega != 0:
            e += _cooo_hfx_energy(
                tdobj, xy, coeff=-2 * (alpha - hyb) * zeta, omega=omega
            )
    return e


def _cooo_hfx_energy(tdobj, xy, coeff, omega=None):
    b = make_sasf_blocks(tdobj, xy)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    eri = _general_eri(
        tdobj.mol, [orbos, orbos, orbos, orbcs], omega=omega
    ).reshape(len(osidx), len(osidx), len(osidx), len(csidx))
    return coeff * float(lib.einsum('iu,wv,uvwi->', b.x_co, b.x_oo, eri))


def cooo_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
    """Central FD of the fixed-amplitude CO-OO pair block energy."""
    from pyscf.sftda.sasf import TDA_SASF
    from ._fd import _make_displaced_mf

    mol0 = td_grad.mol
    coords0 = mol0.atom_coords()
    if atmlst is None:
        atmlst = range(mol0.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))

    def energy_at(coords):
        mol = mol0.copy()
        mol.set_geom_(coords, unit='Bohr')
        mf = _make_displaced_mf(td_grad.base._scf, mol)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Displaced ROKS/ROHF reference did not converge')
        td = TDA_SASF(
            mf, collinear_samples=td_grad.base.collinear_samples,
            remove=td_grad.base.remove,
        )
        td.nstates = td_grad.base.nstates
        td.verbose = 0
        return cooo_block_energy(td, xy)

    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            cp = coords0.copy()
            cm = coords0.copy()
            cp[ia, xyz] += step
            cm[ia, xyz] -= step
            de[k, xyz] = (energy_at(cp) - energy_at(cm)) / (2 * step)
    return de


def cooo_fock_probe_densities(tdobj, xy):
    """Return ``(P_alpha, P_beta)`` for the CO-OO pair Fock terms."""
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    t_beta_co, t_alpha_oc = _cooo_t_beta_alpha(tdobj, xy)
    p_beta = _mo_pair_dm(orbcs, t_beta_co, orbos)
    p_alpha = _mo_pair_dm(orbos, t_alpha_oc, orbcs)
    return p_alpha, p_beta


def cooo_direct_grad_fock(td_grad, tdobj, xy, atmlst=None):
    """Fixed-AO direct skeleton derivative of the alpha/beta Fock terms."""
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    p_alpha, p_beta = cooo_fock_probe_densities(tdobj, xy)
    eri1 = tdobj.mol.intor('int2e_ip1', comp=3)
    for k, ia in enumerate(atmlst):
        f1a, f1b = _full_spin_fock_derivs_by_atom(
            td_grad, tdobj, ia, eri1=eri1
        )
        de[k] += lib.einsum('pq,xpq->x', p_alpha, f1a)
        de[k] += lib.einsum('pq,xpq->x', p_beta, f1b)
    return de


def cooo_direct_grad_hfx(td_grad, tdobj, xy, atmlst=None):
    """Direct HFX skeleton derivative of the CO-OO pair ERI term."""
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    if not hybrid:
        return de

    b = make_sasf_blocks(tdobj, xy)
    zeta, _ = _spin_coefficients(b.si)
    _, _, _, orbcs, orbos, _ = _sasf_orbitals(tdobj)
    offsetdic = tdobj.mol.offset_nr_by_atom()

    def add_with_scale(scale, omega=None):
        for i in range(b.x_co.shape[0]):
            for w in range(b.x_oo.shape[0]):
                t_uv = np.outer(b.x_co[i], b.x_oo[w])
                dm_l = _mo_pair_dm(orbos, t_uv, orbos)
                dm_r = np.outer(orbos[:, w], orbcs[:, i])
                _add_j_bilinear_ip1(
                    de, td_grad, tdobj.mol, dm_l, dm_r,
                    atmlst, offsetdic, scale=scale, omega=omega
                )

    add_with_scale(-2 * hyb * zeta)
    if omega != 0:
        add_with_scale(-2 * (alpha - hyb) * zeta, omega=omega)
    return de


def cooo_direct_grad(td_grad, tdobj, xy, atmlst=None):
    return (
        cooo_direct_grad_fock(td_grad, tdobj, xy, atmlst=atmlst) +
        cooo_direct_grad_hfx(td_grad, tdobj, xy, atmlst=atmlst)
    )


def cooo_m_matrix_fock(tdobj, xy):
    mf = tdobj._scf
    mo = mf.mo_coeff
    nmo = mo.shape[1]
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    fock = mf.get_fock()
    focka_mo = mo.T @ fock.focka @ mo
    fockb_mo = mo.T @ fock.fockb @ mo
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    p_alpha = np.zeros((mf.mol.nao, mf.mol.nao))
    p_beta = np.zeros_like(p_alpha)
    t_beta_co, t_alpha_oc = _cooo_t_beta_alpha(tdobj, xy)
    _add_fock_term(
        q_alpha, q_beta, p_alpha, p_beta, mo, focka_mo, fockb_mo,
        csidx, osidx, t_beta_co, 'beta'
    )
    _add_fock_term(
        q_alpha, q_beta, p_alpha, p_beta, mo, focka_mo, fockb_mo,
        osidx, csidx, t_alpha_oc, 'alpha'
    )
    _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta)
    return q_alpha + q_beta


def cooo_m_matrix_hfx(tdobj, xy):
    mf = tdobj._scf
    nmo = mf.mo_coeff.shape[1]
    b = make_sasf_blocks(tdobj, xy)
    zeta, _ = _spin_coefficients(b.si)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    if not hybrid:
        return q_alpha
    coeff = lib.einsum('iu,wv->uvwi', b.x_co, b.x_oo)
    _add_eri_term_q(
        tdobj, q_alpha, q_beta,
        [orbos, orbos, orbos, orbcs],
        [osidx, osidx, osidx, csidx],
        ['beta', 'beta', 'beta', 'alpha'],
        coeff, scale=-2 * hyb * zeta,
    )
    if omega != 0:
        _add_eri_term_q(
            tdobj, q_alpha, q_beta,
            [orbos, orbos, orbos, orbcs],
            [osidx, osidx, osidx, csidx],
            ['beta', 'beta', 'beta', 'alpha'],
            coeff, scale=-2 * (alpha - hyb) * zeta, omega=omega,
        )
    return q_alpha + q_beta


def cooo_m_matrix(tdobj, xy):
    return cooo_m_matrix_fock(tdobj, xy) + cooo_m_matrix_hfx(tdobj, xy)


def cooo_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = cooo_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(
        td_grad, tdobj, atmlst=atmlst
    )
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        de[k] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


def cooo_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = cooo_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = cooo_canonical_response_grad(
        td_grad, tdobj, xy, atmlst=atmlst
    )
    return direct + orbital, {
        'direct': direct,
        'orbital': orbital,
        'kappas': kappas,
    }

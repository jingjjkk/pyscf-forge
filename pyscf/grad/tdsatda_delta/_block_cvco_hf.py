"""SASF CV-CO block analytic gradient (HF/ROKS, canonical response)."""

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
    _full_beta_fock_deriv_by_atom,
    _general_eri,
)


def _eta(si):
    if si <= 0.5:
        raise NotImplementedError('Si > 1/2 required for SASF CV-CO block')
    return np.sqrt((2 * si + 1) / (2 * si)) - 1


def cvco_block_energy(tdobj, xy):
    """Full off-diagonal ``CV-CO + CO-CV`` HF correction energy."""
    b = make_sasf_blocks(tdobj, xy)
    eta = _eta(b.si)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    mf = tdobj._scf
    fockb = mf.get_fock().fockb

    t_vo = 2 * eta * lib.einsum('ia,iv->av', b.x_cv, b.x_co)
    e = float(lib.einsum('av,av', t_vo, orbvs.T @ fockb @ orbos))

    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    if hybrid:
        e += _cvco_hfx_energy(tdobj, xy, coeff=-2 * hyb * eta)
        if omega != 0:
            e += _cvco_hfx_energy(
                tdobj, xy, coeff=-2 * (alpha - hyb) * eta, omega=omega
            )
    return e


def _cvco_hfx_energy(tdobj, xy, coeff, omega=None):
    b = make_sasf_blocks(tdobj, xy)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    eri = _general_eri(
        tdobj.mol, [orbvs, orbos, orbcs, orbcs], omega=omega
    ).reshape(len(vsidx), len(osidx), len(csidx), len(csidx))
    return coeff * float(lib.einsum('ia,jv,avji->', b.x_cv, b.x_co, eri))


def cvco_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
    """Central FD of the fixed-amplitude CV-CO pair block energy."""
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
        return cvco_block_energy(td, xy)

    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            cp = coords0.copy()
            cm = coords0.copy()
            cp[ia, xyz] += step
            cm[ia, xyz] -= step
            de[k, xyz] = (energy_at(cp) - energy_at(cm)) / (2 * step)
    return de


def cvco_fock_probe_density(tdobj, xy):
    """AO beta probe density for the full CV-CO + CO-CV Fock term."""
    b = make_sasf_blocks(tdobj, xy)
    eta = _eta(b.si)
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    t_vo = 2 * eta * lib.einsum('ia,iv->av', b.x_cv, b.x_co)
    return _mo_pair_dm(orbvs, t_vo, orbos)


def cvco_direct_grad_fock(td_grad, tdobj, xy, atmlst=None):
    """Fixed-AO direct skeleton derivative of the beta Fock term."""
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    p_beta = cvco_fock_probe_density(tdobj, xy)
    eri1 = tdobj.mol.intor('int2e_ip1', comp=3)
    for k, ia in enumerate(atmlst):
        f1b = _full_beta_fock_deriv_by_atom(td_grad, tdobj, ia, eri1=eri1)
        de[k] += lib.einsum('pq,xpq->x', p_beta, f1b)
    return de


def cvco_direct_grad_hfx(td_grad, tdobj, xy, atmlst=None):
    """Direct HFX skeleton derivative of the CV-CO pair ERI term."""
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    if not hybrid:
        return de

    b = make_sasf_blocks(tdobj, xy)
    eta = _eta(b.si)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    offsetdic = tdobj.mol.offset_nr_by_atom()

    def add_with_scale(scale, omega=None):
        if omega is None or omega == 0:
            context = None
        else:
            context = tdobj.mol.with_range_coulomb(omega)
        if context is None:
            for i in range(b.x_cv.shape[0]):
                for j in range(b.x_co.shape[0]):
                    t_av = np.outer(b.x_cv[i], b.x_co[j])
                    dm_l = _mo_pair_dm(orbvs, t_av, orbos)
                    dm_r = np.outer(orbcs[:, j], orbcs[:, i])
                    _add_j_bilinear_ip1(
                        de, td_grad, tdobj.mol, dm_l, dm_r,
                        atmlst, offsetdic, scale=scale
                    )
        else:
            with context:
                for i in range(b.x_cv.shape[0]):
                    for j in range(b.x_co.shape[0]):
                        t_av = np.outer(b.x_cv[i], b.x_co[j])
                        dm_l = _mo_pair_dm(orbvs, t_av, orbos)
                        dm_r = np.outer(orbcs[:, j], orbcs[:, i])
                        _add_j_bilinear_ip1(
                            de, td_grad, tdobj.mol, dm_l, dm_r,
                            atmlst, offsetdic, scale=scale
                        )

    add_with_scale(-2 * hyb * eta)
    if omega != 0:
        add_with_scale(-2 * (alpha - hyb) * eta, omega=omega)
    return de


def cvco_direct_grad(td_grad, tdobj, xy, atmlst=None):
    return (
        cvco_direct_grad_fock(td_grad, tdobj, xy, atmlst=atmlst) +
        cvco_direct_grad_hfx(td_grad, tdobj, xy, atmlst=atmlst)
    )


def cvco_m_matrix_fock(tdobj, xy):
    mf = tdobj._scf
    mo = mf.mo_coeff
    nmo = mo.shape[1]
    b = make_sasf_blocks(tdobj, xy)
    eta = _eta(b.si)
    csidx, osidx, vsidx, _, _, _ = _sasf_orbitals(tdobj)
    fock = mf.get_fock()
    focka_mo = mo.T @ fock.focka @ mo
    fockb_mo = mo.T @ fock.fockb @ mo
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    p_alpha = np.zeros((mf.mol.nao, mf.mol.nao))
    p_beta = np.zeros_like(p_alpha)
    t_vo = 2 * eta * lib.einsum('ia,iv->av', b.x_cv, b.x_co)
    _add_fock_term(
        q_alpha, q_beta, p_alpha, p_beta, mo, focka_mo, fockb_mo,
        vsidx, osidx, t_vo, 'beta'
    )
    _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta)
    return q_alpha + q_beta


def cvco_m_matrix_hfx(tdobj, xy):
    mf = tdobj._scf
    nmo = mf.mo_coeff.shape[1]
    b = make_sasf_blocks(tdobj, xy)
    eta = _eta(b.si)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    if not hybrid:
        return q_alpha
    coeff = lib.einsum('ia,jv->avji', b.x_cv, b.x_co)
    _add_eri_term_q(
        tdobj, q_alpha, q_beta,
        [orbvs, orbos, orbcs, orbcs],
        [vsidx, osidx, csidx, csidx],
        ['beta', 'beta', 'alpha', 'alpha'],
        coeff, scale=-2 * hyb * eta,
    )
    if omega != 0:
        _add_eri_term_q(
            tdobj, q_alpha, q_beta,
            [orbvs, orbos, orbcs, orbcs],
            [vsidx, osidx, csidx, csidx],
            ['beta', 'beta', 'alpha', 'alpha'],
            coeff, scale=-2 * (alpha - hyb) * eta, omega=omega,
        )
    return q_alpha + q_beta


def cvco_m_matrix(tdobj, xy):
    return cvco_m_matrix_fock(tdobj, xy) + cvco_m_matrix_hfx(tdobj, xy)


def cvco_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = cvco_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(
        td_grad, tdobj, atmlst=atmlst
    )
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        de[k] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


def cvco_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = cvco_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = cvco_canonical_response_grad(
        td_grad, tdobj, xy, atmlst=atmlst
    )
    return direct + orbital, {
        'direct': direct,
        'orbital': orbital,
        'kappas': kappas,
    }

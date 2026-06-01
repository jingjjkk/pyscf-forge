"""SASF OV-OO block analytic gradient (HF/ROKS, canonical response).
Mirrors _block_cooo_hf.py with C↔V substitution."""
import numpy as np
from pyscf import lib

from ._blocks import (
    _hybrid_coefficients, _mo_pair_dm, _sasf_orbitals, make_sasf_blocks,
)
from ._fock_basis import (
    alpha_from_0z,
    beta_from_0z,
    fock0z_from_alpha_beta,
    make_fock_basis,
)
from ._q_rhs import (
    _add_eri_term_q, _add_fock_response_q, _add_fock_term,
)
from ._block_analytic_hf import (
    _roks_canonical_response_kappas_hf,
)
from ._direct import (
    _add_j_bilinear_ip1,
    _full_spin_fock_derivs_by_atom,
    _general_eri,
)


def _spin_coeffs(si):
    if si <= 0.5: raise NotImplementedError('Si > 1/2 required')
    zeta = np.sqrt(2*si/(2*si-1)) - 1
    chi = 1.0/np.sqrt(2*si*(2*si-1))
    return zeta, chi


def _ovoo_t_beta_vo_alpha_vo(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy)
    zeta, chi = _spin_coeffs(b.si)
    tr_oo = float(np.trace(b.x_oo))
    t_beta_vo = 2*zeta*lib.einsum('ua,uv->av', b.x_ov, b.x_oo)
    t_alpha_vo = -2*chi*tr_oo*b.x_ov.T
    return t_beta_vo, t_alpha_vo


# ===== Energy =====

def ovoo_block_energy(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy)
    zeta, _ = _spin_coeffs(b.si)
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    mf = tdobj._scf
    fbasis = make_fock_basis(mf)
    t_beta_vo, t_alpha_vo = _ovoo_t_beta_vo_alpha_vo(tdobj, xy)
    e  = float(lib.einsum(
        'au,au', t_beta_vo,
        orbvs.T @ beta_from_0z(fbasis.fock0, fbasis.fockz) @ orbos
    ))
    e += float(lib.einsum(
        'aw,aw', t_alpha_vo,
        orbvs.T @ alpha_from_0z(fbasis.fock0, fbasis.fockz) @ orbos
    ))
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    if hybrid:
        e += _ovoo_hfx_energy(tdobj, xy, coeff=-2*hyb*zeta)
        if omega != 0:
            e += _ovoo_hfx_energy(tdobj, xy, coeff=-2*(alpha-hyb)*zeta, omega=omega)
    return e


def _ovoo_hfx_energy(tdobj, xy, coeff, omega=None):
    b = make_sasf_blocks(tdobj, xy)
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    nv, no_ = orbvs.shape[1], orbos.shape[1]
    eri = _general_eri(
        tdobj.mol, [orbvs, orbos, orbos, orbos], omega=omega
    ).reshape(nv, no_, no_, no_)
    return coeff * float(lib.einsum('ua,wv,avwu->', b.x_ov, b.x_oo, eri))


# ===== FD reference =====

def ovoo_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
    from pyscf.sftda.sasf import TDA_SASF
    from ._fd import _make_displaced_mf
    mol0 = td_grad.mol; coords0 = mol0.atom_coords()
    if atmlst is None: atmlst = range(mol0.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    def energy_at(coords):
        mol = mol0.copy(); mol.set_geom_(coords, unit='Bohr')
        mf = _make_displaced_mf(td_grad.base._scf, mol); mf.kernel()
        if not mf.converged: raise RuntimeError('displaced ROKS not converged')
        td = TDA_SASF(mf, collinear_samples=td_grad.base.collinear_samples,
                      remove=td_grad.base.remove)
        td.nstates = td_grad.base.nstates; td.verbose = 0
        return ovoo_block_energy(td, xy)
    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            cp, cm = coords0.copy(), coords0.copy()
            cp[ia, xyz] += step; cm[ia, xyz] -= step
            de[k, xyz] = (energy_at(cp) - energy_at(cm)) / (2*step)
    return de


# ===== Direct gradient =====

def ovoo_fock_probe_densities(tdobj, xy):
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    t_beta_vo, t_alpha_vo = _ovoo_t_beta_vo_alpha_vo(tdobj, xy)
    p_beta = _mo_pair_dm(orbvs, t_beta_vo, orbos)
    p_alpha = _mo_pair_dm(orbvs, t_alpha_vo, orbos)
    return p_alpha, p_beta


def ovoo_direct_grad_fock(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    p_alpha, p_beta = ovoo_fock_probe_densities(tdobj, xy)
    eri1 = tdobj.mol.intor('int2e_ip1', comp=3)
    for k, ia in enumerate(atmlst):
        f1a, f1b = _full_spin_fock_derivs_by_atom(td_grad, tdobj, ia, eri1=eri1)
        f10, f1z = fock0z_from_alpha_beta(f1a, f1b)
        de[k] += lib.einsum(
            'pq,xpq->x', p_alpha, alpha_from_0z(f10, f1z)
        )
        de[k] += lib.einsum(
            'pq,xpq->x', p_beta, beta_from_0z(f10, f1z)
        )
    return de


def ovoo_direct_grad_hfx(td_grad, tdobj, xy, atmlst=None):
    hybrid, hyb, omega, alpha_coeff = _hybrid_coefficients(tdobj._scf)
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    if not hybrid: return de

    b = make_sasf_blocks(tdobj, xy)
    zeta, _ = _spin_coeffs(b.si)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    offsetdic = tdobj.mol.offset_nr_by_atom()

    def add_with_scale(scale, omega=None):
        for u in range(b.x_ov.shape[0]):
            d1 = orbvs @ b.x_ov[u, :]
            for w in range(b.x_oo.shape[0]):
                d2 = orbos @ b.x_oo[w, :]
                dm_l = np.outer(d1, d2)
                dm_r = np.outer(orbos[:, w], orbos[:, u])
                _add_j_bilinear_ip1(de, td_grad, tdobj.mol, dm_l, dm_r,
                                    atmlst, offsetdic, scale=scale,
                                    omega=omega)

    add_with_scale(-2 * hyb * zeta)
    if omega != 0:
        add_with_scale(-2 * (alpha_coeff - hyb) * zeta, omega=omega)
    return de


def ovoo_direct_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    return ovoo_direct_grad_fock(td_grad, tdobj, xy, atmlst) + \
           ovoo_direct_grad_hfx(td_grad, tdobj, xy, atmlst)


# ===== M-matrix =====

def ovoo_m_matrix_fock(tdobj, xy):
    mf = tdobj._scf; mo = mf.mo_coeff; nmo = mo.shape[1]
    csidx, osidx, vsidx, _, _, _ = _sasf_orbitals(tdobj)
    fbasis = make_fock_basis(mf, mo)
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    p_a = np.zeros((mf.mol.nao, mf.mol.nao)); p_b = np.zeros_like(p_a)
    t_beta_vo, t_alpha_vo = _ovoo_t_beta_vo_alpha_vo(tdobj, xy)
    _add_fock_term(q_a, q_b, p_a, p_b, mo,
                   fbasis.fock0_mo, fbasis.fockz_mo,
                   vsidx, osidx, t_beta_vo, 'beta')
    _add_fock_term(q_a, q_b, p_a, p_b, mo,
                   fbasis.fock0_mo, fbasis.fockz_mo,
                   vsidx, osidx, t_alpha_vo, 'alpha')
    _add_fock_response_q(tdobj, q_a, q_b, p_a, p_b)
    return q_a + q_b


def ovoo_m_matrix_hfx(tdobj, xy):
    mf = tdobj._scf; nmo = mf.mo_coeff.shape[1]
    b = make_sasf_blocks(tdobj, xy)
    zeta, _ = _spin_coeffs(b.si)
    csidx, osidx, vsidx, _, orbos, orbvs = _sasf_orbitals(tdobj)
    hybrid, hyb, omega, alpha_coeff = _hybrid_coefficients(mf)
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    if not hybrid: return q_a
    coeff = lib.einsum('ua,wv->avwu', b.x_ov, b.x_oo)
    _add_eri_term_q(tdobj, q_a, q_b, [orbvs, orbos, orbos, orbos],
                    [vsidx, osidx, osidx, osidx],
                    ['beta', 'beta', 'beta', 'beta'], coeff, scale=-2*hyb*zeta)
    if omega != 0:
        _add_eri_term_q(tdobj, q_a, q_b, [orbvs, orbos, orbos, orbos],
                        [vsidx, osidx, osidx, osidx],
                        ['beta', 'beta', 'beta', 'beta'], coeff,
                        scale=-2*(alpha_coeff-hyb)*zeta, omega=omega)
    return q_a + q_b


def ovoo_m_matrix(tdobj, xy):
    return ovoo_m_matrix_fock(tdobj, xy) + ovoo_m_matrix_hfx(tdobj, xy)


# ===== Orbital response =====

def ovoo_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = ovoo_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(td_grad, tdobj, atmlst=atmlst)
    de = np.zeros((len(atmlst), 3))
    for kk, ia in enumerate(atmlst):
        de[kk] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


def ovoo_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = ovoo_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = ovoo_canonical_response_grad(td_grad, tdobj, xy, atmlst=atmlst)
    return direct + orbital, {
        'direct': direct, 'orbital': orbital, 'kappas': kappas,
    }

"""SASF CV-OV block analytic gradient (HF/ROKS, canonical response).
Mirrors _block_cvco_hf.py with appropriate index substitutions."""
import numpy as np
from pyscf import lib

from ._blocks import (
    _hybrid_coefficients, _mo_pair_dm, _sasf_orbitals, make_sasf_blocks,
)
from ._fock_basis import (
    alpha_from_0z,
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


def _eta(si):
    if si <= 0.5: raise NotImplementedError('Si > 1/2 required')
    return np.sqrt((2*si+1)/(2*si)) - 1


# ===== Energy =====

def cvov_block_energy(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy); eta = _eta(b.si)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    mf = tdobj._scf
    fbasis = make_fock_basis(mf)
    t_oc = -2*eta*lib.einsum('ia,va->vi', b.x_cv, b.x_ov)
    e = float(lib.einsum(
        'vi,vi', t_oc, orbos.T @ alpha_from_0z(
            fbasis.fock0, fbasis.fockz
        ) @ orbcs
    ))
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    if hybrid:
        e += _cvov_hfx_energy(tdobj, xy, coeff=-2*hyb*eta)
        if omega != 0:
            e += _cvov_hfx_energy(tdobj, xy, coeff=-2*(alpha-hyb)*eta, omega=omega)
    return e


def _cvov_hfx_energy(tdobj, xy, coeff, omega=None):
    b = make_sasf_blocks(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    nv, no_, nc = len(b.x_cv[0]), len(b.x_ov), len(b.x_cv)
    eri = _general_eri(
        tdobj.mol, [orbvs, orbvs, orbos, orbcs], omega=omega
    ).reshape(nv, nv, no_, nc)
    return coeff * float(lib.einsum('ia,vb,abvi->', b.x_cv, b.x_ov, eri))


# ===== FD reference =====

def cvov_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
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
        return cvov_block_energy(td, xy)
    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            cp, cm = coords0.copy(), coords0.copy()
            cp[ia, xyz] += step; cm[ia, xyz] -= step
            de[k, xyz] = (energy_at(cp) - energy_at(cm)) / (2*step)
    return de


# ===== Direct gradient =====

def cvov_fock_probe_density(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy); eta = _eta(b.si)
    _, _, _, orbcs, orbos, _ = _sasf_orbitals(tdobj)
    t_oc = -2*eta*lib.einsum('ia,va->vi', b.x_cv, b.x_ov)
    return _mo_pair_dm(orbos, t_oc, orbcs)


def cvov_direct_grad_fock(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    p_alpha = cvov_fock_probe_density(tdobj, xy)
    eri1 = tdobj.mol.intor('int2e_ip1', comp=3)
    for k, ia in enumerate(atmlst):
        f1a, f1b = _full_spin_fock_derivs_by_atom(
            td_grad, tdobj, ia, eri1=eri1
        )
        f10, f1z = fock0z_from_alpha_beta(f1a, f1b)
        de[k] += lib.einsum(
            'pq,xpq->x', p_alpha, alpha_from_0z(f10, f1z)
        )
    return de


def cvov_direct_grad_hfx(td_grad, tdobj, xy, atmlst=None):
    hybrid, hyb, omega, alpha_coeff = _hybrid_coefficients(tdobj._scf)
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    if not hybrid: return de

    b = make_sasf_blocks(tdobj, xy); eta = _eta(b.si)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    offsetdic = tdobj.mol.offset_nr_by_atom()

    def add_with_scale(scale, omega=None):
        for i in range(b.x_cv.shape[0]):
            d1 = orbvs @ b.x_cv[i, :]
            for v in range(b.x_ov.shape[0]):
                d2 = orbvs @ b.x_ov[v, :]
                dm_l = np.outer(d1, d2)
                dm_r = np.outer(orbos[:, v], orbcs[:, i])
                _add_j_bilinear_ip1(de, td_grad, tdobj.mol, dm_l, dm_r,
                                    atmlst, offsetdic, scale=scale,
                                    omega=omega)

    add_with_scale(-2 * hyb * eta)
    if omega != 0:
        add_with_scale(-2 * (alpha_coeff - hyb) * eta, omega=omega)
    return de


def cvov_direct_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    return cvov_direct_grad_fock(td_grad, tdobj, xy, atmlst) + \
           cvov_direct_grad_hfx(td_grad, tdobj, xy, atmlst)


# ===== M-matrix =====

def cvov_m_matrix_fock(tdobj, xy):
    mf = tdobj._scf; mo = mf.mo_coeff; nmo = mo.shape[1]
    b = make_sasf_blocks(tdobj, xy); eta = _eta(b.si)
    csidx, osidx, vsidx, _, _, _ = _sasf_orbitals(tdobj)
    fbasis = make_fock_basis(mf, mo)
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    p_a = np.zeros((mf.mol.nao, mf.mol.nao)); p_b = np.zeros_like(p_a)
    t_oc = -2*eta*lib.einsum('ia,va->vi', b.x_cv, b.x_ov)
    _add_fock_term(q_a, q_b, p_a, p_b, mo,
                   fbasis.fock0_mo, fbasis.fockz_mo,
                   osidx, csidx, t_oc, 'alpha')
    _add_fock_response_q(tdobj, q_a, q_b, p_a, p_b)
    return q_a + q_b


def cvov_m_matrix_hfx(tdobj, xy):
    mf = tdobj._scf; nmo = mf.mo_coeff.shape[1]
    b = make_sasf_blocks(tdobj, xy); eta = _eta(b.si)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    hybrid, hyb, omega, alpha_coeff = _hybrid_coefficients(mf)
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    if not hybrid: return q_a
    coeff = lib.einsum('ia,vb->abvi', b.x_cv, b.x_ov)
    _add_eri_term_q(tdobj, q_a, q_b, [orbvs, orbvs, orbos, orbcs],
                    [vsidx, vsidx, osidx, csidx],
                    ['beta', 'beta', 'beta', 'alpha'], coeff, scale=-2*hyb*eta)
    if omega != 0:
        _add_eri_term_q(tdobj, q_a, q_b, [orbvs, orbvs, orbos, orbcs],
                        [vsidx, vsidx, osidx, csidx],
                        ['beta', 'beta', 'beta', 'alpha'], coeff,
                        scale=-2*(alpha_coeff-hyb)*eta, omega=omega)
    return q_a + q_b


def cvov_m_matrix(tdobj, xy):
    return cvov_m_matrix_fock(tdobj, xy) + cvov_m_matrix_hfx(tdobj, xy)


# ===== Orbital response =====

def cvov_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = cvov_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(td_grad, tdobj, atmlst=atmlst)
    de = np.zeros((len(atmlst), 3))
    for kk, ia in enumerate(atmlst):
        de[kk] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


def cvov_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = cvov_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = cvov_canonical_response_grad(td_grad, tdobj, xy, atmlst=atmlst)
    return direct + orbital, {
        'direct': direct, 'orbital': orbital, 'kappas': kappas,
    }

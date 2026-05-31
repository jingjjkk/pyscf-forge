"""
SASF CO-CO block analytic gradient (HF/ROKS, canonical CPHF).
Uses verified Q-based M construction (same as _block_analytic_hf.py pattern).
"""
import numpy as np
from pyscf import lib

from ._blocks import (
    _hybrid_coefficients, _mo_pair_dm, _sasf_orbitals, make_sasf_blocks,
)
from ._fock_basis import make_fock_basis
from ._q_rhs import _add_eri_term_q, _add_fock_term, _add_fock_response_q
from ._block_analytic_hf import (
    _add_k_bilinear_ip1,
    _roks_canonical_response_kappas_hf,
)
from ._direct import _add_j_bilinear_ip1, _general_eri


# ===== Energy =====

def coco_block_energy(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy); _check_si(b.si)
    _, _, _, orbcs, orbos, _ = _sasf_orbitals(tdobj)
    fbasis = make_fock_basis(tdobj._scf)
    c_f = 2.0/(2*b.si-1)
    t_cc = lib.einsum('iu,ju->ji', b.x_co, b.x_co)
    focks_cc = -orbcs.T @ fbasis.fockz @ orbcs
    e = c_f * float(lib.einsum('ji,ji', t_cc, focks_cc))
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if hybrid:
        c_hfx = hyb/(2*b.si-1)
        e += _coco_hfx_energy(tdobj, xy, coeff=-c_hfx)
        if omega != 0:
            e += _coco_hfx_energy(tdobj, xy, coeff=-(alpha-hyb)/(2*b.si-1), omega=omega)
    return e


def _coco_hfx_energy(tdobj, xy, coeff, omega=None):
    _, _, _, orbcs, orbos, _ = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy)
    ncs, nos = orbcs.shape[1], orbos.shape[1]
    eri = _general_eri(
        tdobj._scf.mol, [orbos, orbcs, orbcs, orbos], omega=omega
    ).reshape(nos, ncs, ncs, nos)
    return coeff * float(lib.einsum('iu,jv,uijv', b.x_co, b.x_co, eri))


def _check_si(si):
    if si <= 0.5: raise NotImplementedError('Si > 1/2 required')


# ===== FD reference =====

def coco_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
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
        return coco_block_energy(td, xy)
    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            cp, cm = coords0.copy(), coords0.copy()
            cp[ia, xyz] += step; cm[ia, xyz] -= step
            de[k, xyz] = (energy_at(cp) - energy_at(cm))/(2*step)
    return de


# ===== Direct gradient =====

def coco_fock_probe_density(tdobj, xy):
    _, _, _, orbcs, _, _ = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy)
    c_eff = 1.0/(2*b.si-1)
    t_cc = lib.einsum('iu,ju->ji', b.x_co, b.x_co)*c_eff
    return _mo_pair_dm(orbcs, t_cc, orbcs)


def coco_open_density(tdobj):
    _, _, _, _, orbos, _ = _sasf_orbitals(tdobj)
    return orbos @ orbos.T


def coco_hfx_density_d(tdobj, xy):
    _, _, _, orbcs, orbos, _ = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy)
    return orbos @ b.x_co.T @ orbcs.T


def coco_hfx_density_d_sym(tdobj, xy):
    d = coco_hfx_density_d(tdobj, xy); return 0.5*(d + d.T)


def coco_direct_grad_fock(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    _add_k_bilinear_ip1(de, td_grad, tdobj.mol,
                        coco_fock_probe_density(tdobj, xy),
                        coco_open_density(tdobj),
                        atmlst, tdobj.mol.offset_nr_by_atom(), scale=1.0)
    return de


def coco_direct_grad_hfx(td_grad, tdobj, xy, atmlst=None):
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    if not hybrid: return de
    mol = tdobj.mol; b = make_sasf_blocks(tdobj, xy)
    dmat = coco_hfx_density_d(tdobj, xy)
    dsym = coco_hfx_density_d_sym(tdobj, xy)
    offsetdic = mol.offset_nr_by_atom()
    _add_j_bilinear_ip1(de, td_grad, mol, dsym, dmat, atmlst, offsetdic,
                        scale=-hyb/(2*b.si-1))
    if omega != 0:
        _add_j_bilinear_ip1(
            de, td_grad, mol, dsym, dmat, atmlst, offsetdic,
            scale=-(alpha-hyb)/(2*b.si-1), omega=omega,
        )
    return de


def coco_direct_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    return coco_direct_grad_fock(td_grad, tdobj, xy, atmlst) + \
           coco_direct_grad_hfx(td_grad, tdobj, xy, atmlst)


# ===== M-matrix (Q-based, verified against _block_analytic_hf pattern) =====

def coco_m_matrix_fock(tdobj, xy):
    """M_Fock from _add_fock_term + _add_fock_response_q (same as CV-CV pattern)."""
    mf = tdobj._scf; mo = mf.mo_coeff; nmo = mo.shape[1]
    _, _, _, orbcs, _, _ = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy); _check_si(b.si)
    fbasis = make_fock_basis(mf, mo)
    csidx = np.where(mf.mo_occ == 2)[0]
    scale = 2.0/(2*b.si-1)
    t_cc = lib.einsum('iu,ju->ji', b.x_co, b.x_co)*scale
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    p_a = np.zeros((mf.mol.nao, mf.mol.nao)); p_b = np.zeros_like(p_a)
    _add_fock_term(q_a, q_b, p_a, p_b, mo,
                   fbasis.fock0_mo, fbasis.fockz_mo,
                   csidx, csidx, t_cc, 'spin')
    _add_fock_response_q(tdobj, q_a, q_b, p_a, p_b)
    return q_a + q_b


def coco_m_matrix_hfx(tdobj, xy):
    """M_HFX from _add_eri_term_q (Q-based, verified against bin pattern)."""
    mf = tdobj._scf; mo = mf.mo_coeff; nmo = mo.shape[1]
    csidx = np.where(mf.mo_occ == 2)[0]; osidx = np.where(mf.mo_occ == 1)[0]
    _, _, _, orbcs, orbos, _ = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy); _check_si(b.si)
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    if not hybrid: return q_a
    eta = lib.einsum('iu,jv->uijv', b.x_co, b.x_co)
    sc = -hyb/(2*b.si-1)
    _add_eri_term_q(tdobj, q_a, q_b, [orbos, orbcs, orbcs, orbos],
                    [osidx, csidx, csidx, osidx],
                    ['beta', 'alpha', 'alpha', 'beta'], eta, scale=sc)
    if omega != 0:
        sc2 = -(alpha-hyb)/(2*b.si-1)
        _add_eri_term_q(tdobj, q_a, q_b, [orbos, orbcs, orbcs, orbos],
                        [osidx, csidx, csidx, osidx],
                        ['beta', 'alpha', 'alpha', 'beta'], eta, scale=sc2, omega=omega)
    return q_a + q_b


def coco_m_matrix(tdobj, xy):
    return coco_m_matrix_fock(tdobj, xy) + coco_m_matrix_hfx(tdobj, xy)


# ===== Orbital response =====

def coco_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = coco_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(td_grad, tdobj, atmlst=atmlst)
    de = np.zeros((len(atmlst), 3))
    for kk, ia in enumerate(atmlst):
        de[kk] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


# ===== Assembly =====

def coco_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = coco_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = coco_canonical_response_grad(
        td_grad, tdobj, xy, atmlst=atmlst)
    return direct + orbital, {
        'direct': direct, 'orbital': orbital, 'kappas': kappas,
    }

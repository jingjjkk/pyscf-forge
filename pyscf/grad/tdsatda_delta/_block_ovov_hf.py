"""
SASF OV-OV block analytic gradient (HF/ROKS, canonical CPHF).
Mirrors _block_coco_hf.py with C↔V substitution.
"""
import numpy as np
from pyscf import lib

from ._blocks import (
    _hybrid_coefficients, _mo_pair_dm, _sasf_orbitals, make_sasf_blocks,
)
from ._q_rhs import _add_eri_term_q, _add_fock_term, _add_fock_response_q
from ._block_analytic_hf import (
    _add_k_bilinear_ip1,
    _roks_canonical_response_kappas_hf,
)
from ._block_coco_hf import _check_si
from ._direct import _add_j_bilinear_ip1, _general_eri


# ===== Energy =====

def ovov_block_energy(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy); _check_si(b.si)
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    fock = tdobj._scf.get_fock(); focks = 0.5*(fock.fockb - fock.focka)
    c_f = 2.0/(2*b.si-1)
    t_vv = lib.einsum('ua,ub->ab', b.x_ov, b.x_ov)
    focks_vv = orbvs.T @ focks @ orbvs
    e = c_f * float(lib.einsum('ab,ab', t_vv, focks_vv))
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if hybrid:
        c_hfx = hyb/(2*b.si-1)
        e += _ovov_hfx_energy(tdobj, xy, coeff=-c_hfx)
        if omega != 0:
            e += _ovov_hfx_energy(tdobj, xy, coeff=-(alpha-hyb)/(2*b.si-1), omega=omega)
    return e


def _ovov_hfx_energy(tdobj, xy, coeff, omega=None):
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy)
    nos, nvs = orbos.shape[1], orbvs.shape[1]
    eri = _general_eri(
        tdobj._scf.mol, [orbvs, orbos, orbos, orbvs], omega=omega
    ).reshape(nvs, nos, nos, nvs)
    return coeff * float(lib.einsum('ua,vb,auvb', b.x_ov, b.x_ov, eri))


# ===== FD reference =====

def ovov_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
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
        return ovov_block_energy(td, xy)
    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            cp, cm = coords0.copy(), coords0.copy()
            cp[ia, xyz] += step; cm[ia, xyz] -= step
            de[k, xyz] = (energy_at(cp) - energy_at(cm))/(2*step)
    return de


# ===== Direct gradient =====

def ovov_fock_probe_density(tdobj, xy):
    _, _, _, _, _, orbvs = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy)
    c_eff = 1.0/(2*b.si-1)
    t_vv = lib.einsum('ua,ub->ab', b.x_ov, b.x_ov)*c_eff
    return _mo_pair_dm(orbvs, t_vv, orbvs)


def ovov_open_density(tdobj):
    """D^O = C_O C_O^T."""
    _, _, _, _, orbos, _ = _sasf_orbitals(tdobj)
    return orbos @ orbos.T


def ovov_hfx_density_d(tdobj, xy):
    """D = C_V X_OV^T C_O^T."""
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy)
    return orbvs @ b.x_ov.T @ orbos.T


def ovov_hfx_density_d_sym(tdobj, xy):
    d = ovov_hfx_density_d(tdobj, xy); return 0.5*(d + d.T)


def ovov_direct_grad_fock(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    _add_k_bilinear_ip1(de, td_grad, tdobj.mol,
                        ovov_fock_probe_density(tdobj, xy),
                        ovov_open_density(tdobj),
                        atmlst, tdobj.mol.offset_nr_by_atom(), scale=1.0)
    return de


def ovov_direct_grad_hfx(td_grad, tdobj, xy, atmlst=None):
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    if not hybrid: return de
    mol = tdobj.mol; b = make_sasf_blocks(tdobj, xy)
    dmat = ovov_hfx_density_d(tdobj, xy)
    dsym = ovov_hfx_density_d_sym(tdobj, xy)
    offsetdic = mol.offset_nr_by_atom()
    _add_j_bilinear_ip1(de, td_grad, mol, dsym, dmat, atmlst, offsetdic,
                        scale=-hyb/(2*b.si-1))
    if omega != 0:
        _add_j_bilinear_ip1(
            de, td_grad, mol, dsym, dmat, atmlst, offsetdic,
            scale=-(alpha-hyb)/(2*b.si-1), omega=omega,
        )
    return de


def ovov_direct_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    return ovov_direct_grad_fock(td_grad, tdobj, xy, atmlst) + \
           ovov_direct_grad_hfx(td_grad, tdobj, xy, atmlst)


# ===== M-matrix =====

def ovov_m_matrix_fock(tdobj, xy):
    """M_Fock from Q-based construction, VV block."""
    mf = tdobj._scf; mo = mf.mo_coeff; nmo = mo.shape[1]
    b = make_sasf_blocks(tdobj, xy); _check_si(b.si)
    fock = mf.get_fock()
    focka_mo = mo.T @ fock.focka @ mo
    fockb_mo = mo.T @ fock.fockb @ mo
    vsidx = np.where(mf.mo_occ == 0)[0]
    scale = 2.0/(2*b.si-1)
    t_vv = lib.einsum('ua,ub->ab', b.x_ov, b.x_ov)*scale
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    p_a = np.zeros((mf.mol.nao, mf.mol.nao)); p_b = np.zeros_like(p_a)
    _add_fock_term(q_a, q_b, p_a, p_b, mo, focka_mo, fockb_mo,
                   vsidx, vsidx, t_vv, 'spin')
    _add_fock_response_q(tdobj, q_a, q_b, p_a, p_b)
    return q_a + q_b


def ovov_m_matrix_hfx(tdobj, xy):
    """M_HFX from Q-based ERI construction, V,O,O,V."""
    mf = tdobj._scf; mo = mf.mo_coeff; nmo = mo.shape[1]
    csidx = np.where(mf.mo_occ == 2)[0]
    osidx = np.where(mf.mo_occ == 1)[0]
    vsidx = np.where(mf.mo_occ == 0)[0]
    _, _, _, _, orbos, orbvs = _sasf_orbitals(tdobj)
    b = make_sasf_blocks(tdobj, xy); _check_si(b.si)
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    if not hybrid: return q_a
    eta = lib.einsum('ua,vb->auvb', b.x_ov, b.x_ov)
    sc = -hyb/(2*b.si-1)
    _add_eri_term_q(tdobj, q_a, q_b, [orbvs, orbos, orbos, orbvs],
                    [vsidx, osidx, osidx, vsidx],
                    ['beta', 'beta', 'beta', 'beta'], eta, scale=sc)
    if omega != 0:
        sc2 = -(alpha-hyb)/(2*b.si-1)
        _add_eri_term_q(tdobj, q_a, q_b, [orbvs, orbos, orbos, orbvs],
                        [vsidx, osidx, osidx, vsidx],
                        ['beta', 'beta', 'beta', 'beta'], eta, scale=sc2, omega=omega)
    return q_a + q_b


def ovov_m_matrix(tdobj, xy):
    return ovov_m_matrix_fock(tdobj, xy) + ovov_m_matrix_hfx(tdobj, xy)


# ===== Orbital response =====

def ovov_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = ovov_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(td_grad, tdobj, atmlst=atmlst)
    de = np.zeros((len(atmlst), 3))
    for kk, ia in enumerate(atmlst):
        de[kk] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


# ===== Assembly =====

def ovov_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = ovov_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = ovov_canonical_response_grad(
        td_grad, tdobj, xy, atmlst=atmlst)
    return direct + orbital, {
        'direct': direct, 'orbital': orbital, 'kappas': kappas,
    }

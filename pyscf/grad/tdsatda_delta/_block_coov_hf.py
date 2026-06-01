"""SASF CO-OV block analytic gradient (HF/ROKS, canonical response).
Pure HFX block: ΔA_iu,bv = hyb/(2Si-1)[(ui|vb) - (ub|vi)], no Fock."""
import numpy as np
from pyscf import lib

from ._blocks import (
    _hybrid_coefficients, _sasf_orbitals, make_sasf_blocks,
)
from ._q_rhs import _add_eri_term_q
from ._block_analytic_hf import (
    _roks_canonical_response_kappas_hf,
)
from ._direct import _add_j_bilinear_ip1, _general_eri


# ===== Energy =====

def coov_block_energy(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy)
    if b.si <= 0.5: raise NotImplementedError('Si > 1/2')
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    e = 0.0
    if hybrid:
        c_hfx = hyb/(2*b.si-1)
        e += _coov_hfx_energy(tdobj, xy, coeff=c_hfx)
        if omega != 0:
            c_rsh = (alpha-hyb)/(2*b.si-1)
            e += _coov_hfx_energy(tdobj, xy, coeff=c_rsh, omega=omega)
    return 2.0 * e


def _coov_hfx_energy(tdobj, xy, coeff, omega=None):
    b = make_sasf_blocks(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    nco, nos_, nvs = len(b.x_co), len(b.x_ov), orbvs.shape[1]
    eri1 = _general_eri(tdobj.mol, [orbos, orbcs, orbos, orbvs], omega=omega).reshape(nos_, nco, nos_, nvs)
    eri2 = _general_eri(tdobj.mol, [orbos, orbvs, orbos, orbcs], omega=omega).reshape(nos_, nvs, nos_, nco)
    e  = coeff * float(lib.einsum('iu,vb,uivb->', b.x_co, b.x_ov, eri1))
    e -= coeff * float(lib.einsum('iu,vb,ubvi->', b.x_co, b.x_ov, eri2))
    return e


# ===== FD reference =====

def coov_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
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
        return coov_block_energy(td, xy)
    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            cp, cm = coords0.copy(), coords0.copy()
            cp[ia, xyz] += step; cm[ia, xyz] -= step
            de[k, xyz] = (energy_at(cp) - energy_at(cm)) / (2*step)
    return de


# ===== Direct gradient =====

def coov_direct_grad_hfx(td_grad, tdobj, xy, atmlst=None):
    hybrid, hyb, omega, alpha_coeff = _hybrid_coefficients(tdobj._scf)
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst); de = np.zeros((len(atmlst), 3))
    if not hybrid: return de
    b = make_sasf_blocks(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    offsetdic = tdobj.mol.offset_nr_by_atom()
    c = hyb/(2*b.si-1)

    def add_with_scale(scale):
        for u in range(b.x_co.shape[1]):
            d1_c = orbcs @ b.x_co[:, u]
            d1_v = orbvs @ b.x_ov[u, :]
            for v in range(b.x_ov.shape[0]):
                d2_c = orbcs @ b.x_co[:, v]
                d2_v = orbvs @ b.x_ov[v, :]
                dm_inner = np.outer(orbos[:, u], d1_c)
                dm_outer = np.outer(orbos[:, v], d2_v)
                _add_j_bilinear_ip1(de, td_grad, tdobj.mol,
                    dm_inner, dm_outer, atmlst, offsetdic, scale=scale,
                    omega=omega)
                dm_inner2 = np.outer(orbos[:, u], d2_v)
                dm_outer2 = np.outer(orbos[:, v], d1_c)
                _add_j_bilinear_ip1(de, td_grad, tdobj.mol,
                    dm_inner2, dm_outer2, atmlst, offsetdic, scale=-scale,
                    omega=omega)

    add_with_scale(c)
    if omega != 0:
        add_with_scale((alpha_coeff-hyb)/(2*b.si-1))
    return de


def coov_direct_grad(td_grad, tdobj, xy, atmlst=None):
    return 2.0 * coov_direct_grad_hfx(td_grad, tdobj, xy, atmlst=atmlst)


# ===== M-matrix =====

def coov_m_matrix_hfx(tdobj, xy):
    mf = tdobj._scf; nmo = mf.mo_coeff.shape[1]
    b = make_sasf_blocks(tdobj, xy)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    hybrid, hyb, omega, alpha_coeff = _hybrid_coefficients(mf)
    q_a = np.zeros((nmo, nmo)); q_b = np.zeros_like(q_a)
    if not hybrid: return q_a
    c = hyb/(2*b.si-1)
    ct1 = lib.einsum('iu,vb->uivb', b.x_co, b.x_ov)
    _add_eri_term_q(tdobj, q_a, q_b, [orbos, orbcs, orbos, orbvs],
                    [osidx, csidx, osidx, vsidx],
                    ['beta', 'alpha', 'beta', 'beta'], ct1, scale=c)
    ct2 = lib.einsum('iu,vb->ubvi', b.x_co, b.x_ov)
    _add_eri_term_q(tdobj, q_a, q_b, [orbos, orbvs, orbos, orbcs],
                    [osidx, vsidx, osidx, csidx],
                    ['beta', 'beta', 'beta', 'alpha'], ct2, scale=-c)
    if omega != 0:
        c2 = (alpha_coeff-hyb)/(2*b.si-1)
        _add_eri_term_q(tdobj, q_a, q_b, [orbos, orbcs, orbos, orbvs],
                        [osidx, csidx, osidx, vsidx],
                        ['beta', 'alpha', 'beta', 'beta'], ct1, scale=c2, omega=omega)
        _add_eri_term_q(tdobj, q_a, q_b, [orbos, orbvs, orbos, orbcs],
                        [osidx, vsidx, osidx, csidx],
                        ['beta', 'beta', 'beta', 'alpha'], ct2, scale=-c2, omega=omega)
    return 2.0 * (q_a + q_b)


coov_m_matrix = coov_m_matrix_hfx


# ===== Orbital response =====

def coov_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None: atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = coov_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(td_grad, tdobj, atmlst=atmlst)
    de = np.zeros((len(atmlst), 3))
    for kk, ia in enumerate(atmlst):
        de[kk] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


def coov_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = coov_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = coov_canonical_response_grad(td_grad, tdobj, xy, atmlst=atmlst)
    return direct + orbital, {
        'direct': direct, 'orbital': orbital, 'kappas': kappas,
    }

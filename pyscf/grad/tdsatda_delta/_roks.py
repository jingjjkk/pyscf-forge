from functools import reduce

import numpy as np

from pyscf import lib
from pyscf import scf
from pyscf.lib import logger

from ._blocks import (
    roks_spaces,
    pack_roks_kappa,
    unpack_roks_kappa,
)


def _sasf_hf_fock_for_orbs(tdobj, mo_coeff_alpha, mo_coeff_beta):
    '''HF alpha/beta Fock matrices for spin-separated orbital probes.'''
    mf = tdobj._scf
    mol = mf.mol
    mo_occ = mf.mo_occ
    occa = mo_occ > 0
    occb = mo_occ == 2
    dm_a = mo_coeff_alpha[:, occa] @ mo_coeff_alpha[:, occa].conj().T
    dm_b = mo_coeff_beta[:, occb] @ mo_coeff_beta[:, occb].conj().T
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, (dm_a, dm_b), hermi=0)
    focka = hcore + vj[0] + vj[1] - vk[0]
    fockb = hcore + vj[0] + vj[1] - vk[1]
    return focka, fockb


def make_roks_hessian_action_hf(tdobj):
    '''Build the HF ROKS orbital Hessian action for CO/CV/OV rotations.'''
    mf_roks = tdobj._scf
    mf_resp = tdobj.sftda._scf
    mo_coeff = mf_roks.mo_coeff
    csidx, osidx, vsidx = roks_spaces(tdobj)
    nc, no, nv = len(csidx), len(osidx), len(vsidx)
    orb_c = mo_coeff[:, csidx]
    orb_o = mo_coeff[:, osidx]
    orb_v = mo_coeff[:, vsidx]
    c_cov = np.hstack((orb_c, orb_o, orb_v))
    focka, fockb = _sasf_hf_fock_for_orbs(tdobj, mo_coeff, mo_coeff)
    fmo_a = reduce(np.dot, (c_cov.T, focka, c_cov))
    fmo_b = reduce(np.dot, (c_cov.T, fockb, c_cov))
    vresp = mf_resp.gen_response(hermi=1)
    sl_c = slice(0, nc)
    sl_o = slice(nc, nc + no)
    sl_v = slice(nc + no, nc + no + nv)

    def action(vec):
        zco, zcv, zov = unpack_roks_kappa(vec, nc, no, nv)
        kappa = np.zeros((nc + no + nv, nc + no + nv))
        kappa[sl_o, sl_c] = zco
        kappa[sl_c, sl_o] = -zco.T
        kappa[sl_v, sl_c] = zcv
        kappa[sl_c, sl_v] = -zcv.T
        kappa[sl_v, sl_o] = zov
        kappa[sl_o, sl_v] = -zov.T

        explicit_a = np.dot(fmo_a, kappa) - np.dot(kappa, fmo_a)
        explicit_b = np.dot(fmo_b, kappa) - np.dot(kappa, fmo_b)

        dma = reduce(np.dot, (orb_v, zcv, orb_c.T))
        dma += reduce(np.dot, (orb_v, zov, orb_o.T))
        dmb = reduce(np.dot, (orb_o, zco, orb_c.T))
        dmb += reduce(np.dot, (orb_v, zcv, orb_c.T))
        v1a, v1b = vresp(np.stack((dma + dma.T, dmb + dmb.T)))
        v1mo_a = reduce(np.dot, (c_cov.T, v1a, c_cov))
        v1mo_b = reduce(np.dot, (c_cov.T, v1b, c_cov))

        out_co = explicit_b[sl_o, sl_c] + v1mo_b[sl_o, sl_c]
        out_cv = (
            explicit_a[sl_v, sl_c] + explicit_b[sl_v, sl_c] +
            v1mo_a[sl_v, sl_c] + v1mo_b[sl_v, sl_c]
        )
        out_ov = explicit_a[sl_v, sl_o] + v1mo_a[sl_v, sl_o]
        return pack_roks_kappa(out_co, out_cv, out_ov)

    return action, (nc, no, nv)


def _roks_hessian_diag(tdobj, dims, level_shift=0):
    mf = tdobj.sftda._scf
    mo_energy = mf.mo_energy
    csidx, osidx, vsidx = roks_spaces(tdobj)
    gap_co = mo_energy[1][osidx, None] - mo_energy[1][csidx]
    gap_cv = (mo_energy[0][vsidx, None] - mo_energy[0][csidx] +
              mo_energy[1][vsidx, None] - mo_energy[1][csidx])
    gap_ov = mo_energy[0][vsidx, None] - mo_energy[0][osidx]
    diag = pack_roks_kappa(gap_co, gap_cv, gap_ov)
    if level_shift:
        diag = diag + level_shift
    small = np.abs(diag) < 1e-8
    if np.any(small):
        diag = diag.copy()
        diag[small] = np.where(diag[small] < 0, -1e-8, 1e-8)
    return diag


def solve_roks_z_hf(td_grad, tdobj, rhs, verbose=logger.WARN):
    '''Solve the HF ROKS Z-vector equation in the CO/CV/OV space.'''
    action, dims = make_roks_hessian_action_hf(tdobj)
    diag = _roks_hessian_diag(tdobj, dims)
    h1base = -rhs / diag

    def aop(z):
        z = np.asarray(z)
        if z.ndim == 1:
            return action(z) / diag - z
        return np.asarray([action(zi) / diag - zi for zi in z])

    log = logger.new_logger(td_grad, verbose)
    z = lib.krylov(
        aop, h1base, tol=td_grad.cphf_conv_tol,
        max_cycle=td_grad.cphf_max_cycle, hermi=False, verbose=log,
    )
    z = np.asarray(z).reshape(-1)
    residual = action(z) + rhs
    log.debug('ROKS Z-vector residual max %.6g norm %.6g',
              np.max(np.abs(residual)), np.linalg.norm(residual))
    return z, dims


def roks_z_to_uks_vo(tdobj, zvec, dims):
    '''Map a solved ROKS spatial Z-vector to alpha/beta UKS VO blocks.'''
    nc, no, nv = dims
    zco, zcv, zov = unpack_roks_kappa(zvec, nc, no, nv)
    mf = tdobj.sftda._scf
    mo_occ = mf.mo_occ
    csidx, osidx, vsidx = roks_spaces(tdobj)
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    z1a = np.zeros((len(viridxa), len(occidxa)))
    z1b = np.zeros((len(viridxb), len(occidxb)))

    row_v_a = [np.where(viridxa == v)[0][0] for v in vsidx]
    col_c_a = [np.where(occidxa == c)[0][0] for c in csidx]
    col_o_a = [np.where(occidxa == o)[0][0] for o in osidx]
    row_o_b = [np.where(viridxb == o)[0][0] for o in osidx]
    row_v_b = [np.where(viridxb == v)[0][0] for v in vsidx]
    col_c_b = [np.where(occidxb == c)[0][0] for c in csidx]

    z1a[np.ix_(row_v_a, col_c_a)] = zcv
    z1a[np.ix_(row_v_a, col_o_a)] = zov
    z1b[np.ix_(row_o_b, col_c_b)] = zco
    z1b[np.ix_(row_v_b, col_c_b)] = zcv
    return z1a, z1b


def _hf_fock_for_roks_orbs(tdobj, mo_coeff, mol=None):
    '''HF alpha/beta Fock matrices for a spatial ROKS orbital set.'''
    mf = tdobj._scf
    if mol is None:
        mol = mf.mol
    mo_occ = mf.mo_occ
    dm_a = mo_coeff[:, mo_occ > 0] @ mo_coeff[:, mo_occ > 0].T
    dm_b = mo_coeff[:, mo_occ == 2] @ mo_coeff[:, mo_occ == 2].T
    hcore = mf.get_hcore(mol)
    vj, vk = scf.hf.get_jk(mol, (dm_a, dm_b), hermi=1)
    focka = hcore + vj[0] + vj[1] - vk[0]
    fockb = hcore + vj[0] + vj[1] - vk[1]
    return focka, fockb


def roks_brillouin_residual_hf(tdobj, mo_coeff=None, mol=None):
    '''ROKS Brillouin residual packed as O<-C, V<-C, V<-O blocks.'''
    mf = tdobj._scf
    if mol is None:
        mol = mf.mol
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    csidx, osidx, vsidx = roks_spaces(tdobj)
    focka, fockb = _hf_fock_for_roks_orbs(tdobj, mo_coeff, mol=mol)
    fmo_a = reduce(np.dot, (mo_coeff.T, focka, mo_coeff))
    fmo_b = reduce(np.dot, (mo_coeff.T, fockb, mo_coeff))
    rco = fmo_b[np.ix_(osidx, csidx)]
    rcv = fmo_a[np.ix_(vsidx, csidx)] + fmo_b[np.ix_(vsidx, csidx)]
    rov = fmo_a[np.ix_(vsidx, osidx)]
    return pack_roks_kappa(rco, rcv, rov)

#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Nuclear gradients for SA-SF-TDA.

This module provides the PySCF-style gradient interface for
``pyscf.sftda.satda.SATDA``.  The finite-difference backend is kept as a
reference interface and validation tool.  The HF ``deltaS=-1`` analytical path
is a development implementation checked against finite differences in the
regression tests.
'''

from dataclasses import dataclass
from functools import reduce

import numpy as np

from pyscf import ao2mo
from pyscf import dft
from pyscf import lib
from pyscf import scf
from pyscf.dft import numint
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.lib import logger
from pyscf.scf import ucphf


def _normalized_x(tdobj, root):
    x = np.asarray(tdobj.xy[root][0]).ravel()
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise RuntimeError('SA-SF-TDA root has near-zero amplitude norm')
    return x / norm


def _amplitude_overlap(x_ref, tdobj, root):
    x = _normalized_x(tdobj, root)
    return abs(np.vdot(x_ref, x))


@dataclass
class SATDASFBlocks:
    csidx: np.ndarray
    osidx: np.ndarray
    vsidx: np.ndarray
    x_co: np.ndarray
    x_cv: np.ndarray
    x_oo: np.ndarray
    x_ov: np.ndarray
    si: float


@dataclass
class SATDAFockCoefficients:
    '''Coefficient matrices for the HF-equivalent SATDA Fock-like terms.'''

    t_s_cc: np.ndarray
    t_s_vv: np.ndarray
    t_s_cv: np.ndarray
    t_b_vo: np.ndarray
    t_b_co: np.ndarray
    t_a_oc: np.ndarray
    t_a_vo: np.ndarray
    trace_oo: float
    si: float


def make_satda_sf_blocks(tdobj, xy):
    if getattr(tdobj, 'deltaS', None) != -1:
        raise NotImplementedError(
            'The analytical SATDA gradient currently supports only deltaS=-1'
        )

    mf = tdobj._scf
    mo_occ = mf.mo_occ
    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    ncs = len(csidx)
    nos = len(osidx)
    x = np.asarray(xy[0])
    if x.shape != (ncs + nos, nos + len(vsidx)):
        raise ValueError(
            'SATDA X amplitude shape %s incompatible with C/O/V dimensions '
            '(%d, %d)' % (x.shape, ncs + nos, nos + len(vsidx))
        )
    return SATDASFBlocks(
        csidx=csidx,
        osidx=osidx,
        vsidx=vsidx,
        x_co=x[:ncs, :nos],
        x_cv=x[:ncs, nos:],
        x_oo=x[ncs:, :nos],
        x_ov=x[ncs:, nos:],
        si=(mf.mol.nelec[0] - mf.mol.nelec[1]) * 0.5,
    )


def _satda_orbitals(tdobj):
    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    return csidx, osidx, vsidx, orbcs, orbos, orbvs


def _mo_pair_dm(c_left, mat, c_right):
    '''AO matrix for a general MO pair coefficient matrix.'''
    return c_left @ mat @ c_right.conj().T


def _hybrid_coefficients(mf):
    if isinstance(mf, dft.KohnShamDFT):
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
        hybrid = mf._numint.libxc.is_hybrid_xc(mf.xc)
        return hybrid, hyb, omega, alpha
    return True, 1.0, 0.0, 0.0


def satda_fock_coefficients(tdobj, xy):
    '''Build coefficient matrices for the HF-equivalent SATDA Fock terms.'''
    b = make_satda_sf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SATDA spin adaptation requires Si > 1/2')

    tr_oo = float(np.trace(b.x_oo))
    eta = np.sqrt((2 * si + 1) / (2 * si)) - 1
    gamma = np.sqrt((2 * si + 1) / (2 * si - 1))
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1
    chi = 1.0 / np.sqrt(2 * si * (2 * si - 1))

    t_s_cc = (
        lib.einsum('ia,ja->ji', b.x_cv, b.x_cv) / si
        + lib.einsum('iu,ju->ji', b.x_co, b.x_co) * 2 / (2 * si - 1)
    )
    t_s_vv = (
        lib.einsum('ia,ib->ab', b.x_cv, b.x_cv) / si
        + lib.einsum('ua,ub->ab', b.x_ov, b.x_ov) * 2 / (2 * si - 1)
    )
    t_s_cv = (gamma * (1 + 1 / si)) * tr_oo * b.x_cv

    t_b_vo = (
        2 * eta * lib.einsum('ia,iv->av', b.x_cv, b.x_co)
        + 2 * zeta * lib.einsum('ua,uv->av', b.x_ov, b.x_oo)
    )
    t_b_co = 2 * chi * tr_oo * b.x_co

    t_a_oc = (
        -2 * eta * lib.einsum('ia,va->vi', b.x_cv, b.x_ov)
        - 2 * zeta * lib.einsum('iu,vu->vi', b.x_co, b.x_oo)
    )
    t_a_vo = -2 * chi * tr_oo * b.x_ov.T

    return SATDAFockCoefficients(
        t_s_cc=t_s_cc,
        t_s_vv=t_s_vv,
        t_s_cv=t_s_cv,
        t_b_vo=t_b_vo,
        t_b_co=t_b_co,
        t_a_oc=t_a_oc,
        t_a_vo=t_a_vo,
        trace_oo=tr_oo,
        si=si,
    )


def satda_fock_probe_densities(tdobj, xy):
    '''Return AO probe densities for the HF-equivalent Fock-like terms.'''
    coeff = satda_fock_coefficients(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _satda_orbitals(tdobj)

    dm_s = np.zeros((tdobj.mol.nao, tdobj.mol.nao))
    dm_s += _mo_pair_dm(orbcs, coeff.t_s_cc, orbcs)
    dm_s += _mo_pair_dm(orbvs, coeff.t_s_vv, orbvs)
    dm_s += _mo_pair_dm(orbcs, coeff.t_s_cv, orbvs)

    dm_b = np.zeros_like(dm_s)
    dm_b += _mo_pair_dm(orbvs, coeff.t_b_vo, orbos)
    dm_b += _mo_pair_dm(orbcs, coeff.t_b_co, orbos)

    dm_a = np.zeros_like(dm_s)
    dm_a += _mo_pair_dm(orbos, coeff.t_a_oc, orbcs)
    dm_a += _mo_pair_dm(orbvs, coeff.t_a_vo, orbos)

    return dm_a - 0.5 * dm_s, dm_b + 0.5 * dm_s


def _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, left_idx, right_idx, coeff_mat, spin):
    coeff_mat = np.asarray(coeff_mat)
    if coeff_mat.size == 0:
        return

    left_idx = np.asarray(left_idx)
    right_idx = np.asarray(right_idx)
    c_left = mo_coeff[:, left_idx]
    c_right = mo_coeff[:, right_idx]
    dm = _mo_pair_dm(c_left, coeff_mat, c_right)

    def add_projection(q, fock_mo, scale):
        if scale == 0:
            return
        q[:, left_idx] += scale * (fock_mo[:, right_idx] @ coeff_mat.T)
        q[:, right_idx] += scale * (fock_mo[:, left_idx] @ coeff_mat)

    if spin == 'alpha':
        add_projection(q_alpha, focka_mo, 1.0)
        p_alpha += dm
    elif spin == 'beta':
        add_projection(q_beta, fockb_mo, 1.0)
        p_beta += dm
    elif spin == 'spin':
        add_projection(q_beta, fockb_mo, 0.5)
        add_projection(q_alpha, focka_mo, -0.5)
        p_beta += 0.5 * dm
        p_alpha -= 0.5 * dm
    else:
        raise ValueError('Unknown Fock term spin label %s' % spin)


def _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta):
    '''Add Fock-density response contribution to Q_alpha/Q_beta.'''
    mf = tdobj._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ > 0)[0]
    occidxb = np.where(mo_occ == 2)[0]

    if isinstance(mf, dft.KohnShamDFT) and mf._numint._xc_type(mf.xc) != 'HF':
        umf = _as_spin_unrestricted_reference(mf)
        vresp = umf.gen_response(hermi=0)
        va, vb = vresp(np.asarray((p_alpha.T, p_beta.T)))
    else:
        p_tot = p_alpha + p_beta
        vj = mf.get_j(mol, p_tot.T, hermi=0)
        vk_a = mf.get_k(mol, p_alpha.T, hermi=0)
        vk_b = mf.get_k(mol, p_beta.T, hermi=0)
        va = vj - vk_a
        vb = vj - vk_b
    q_alpha[:, occidxa] += mo_coeff.conj().T @ (va + va.T) @ mo_coeff[:, occidxa]
    q_beta[:, occidxb] += mo_coeff.conj().T @ (vb + vb.T) @ mo_coeff[:, occidxb]


def satda_delta_fock_q(tdobj, xy, with_response=True):
    '''Unconstrained MO coefficient derivative Q for Fock-like terms.'''
    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    p_alpha = np.zeros((mf.mol.nao, mf.mol.nao))
    p_beta = np.zeros_like(p_alpha)

    coeff = satda_fock_coefficients(tdobj, xy)
    csidx, osidx, vsidx, _, _, _ = _satda_orbitals(tdobj)
    fock = mf.get_fock()
    focka_mo = mo_coeff.conj().T @ fock.focka @ mo_coeff
    fockb_mo = mo_coeff.conj().T @ fock.fockb @ mo_coeff

    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, csidx, csidx,
                   coeff.t_s_cc, 'spin')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, vsidx, vsidx,
                   coeff.t_s_vv, 'spin')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, csidx, vsidx,
                   coeff.t_s_cv, 'spin')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, vsidx, osidx,
                   coeff.t_b_vo, 'beta')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, csidx, osidx,
                   coeff.t_b_co, 'beta')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, osidx, csidx,
                   coeff.t_a_oc, 'alpha')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   focka_mo, fockb_mo, vsidx, osidx,
                   coeff.t_a_vo, 'alpha')

    if with_response:
        _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta)
    return q_alpha, q_beta


def _satda_hf_exchange_energy_with_coeff(tdobj, xy, coeff, omega=None):
    mf = tdobj._scf
    mol = mf.mol
    b = make_satda_sf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SATDA spin adaptation requires Si > 1/2')
    _, _, _, orbcs, orbos, orbvs = _satda_orbitals(tdobj)

    eta = np.sqrt((2 * si + 1) / (2 * si)) - 1
    gamma = np.sqrt((2 * si + 1) / (2 * si - 1))
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1

    def general(orbs):
        if omega is None or omega == 0:
            return ao2mo.general(mol, orbs, compact=False)
        with mol.with_range_coulomb(omega):
            return ao2mo.general(mol, orbs, compact=False)

    e = 0.0
    eri = general([orbos, orbcs, orbcs, orbos]).reshape(
        len(b.osidx), len(b.csidx), len(b.csidx), len(b.osidx)
    )
    e -= lib.einsum('iu,jv,uijv', b.x_co, b.x_co, eri) * coeff / (2 * si - 1)

    eri = general([orbvs, orbos, orbos, orbvs]).reshape(
        len(b.vsidx), len(b.osidx), len(b.osidx), len(b.vsidx)
    )
    e -= lib.einsum('ua,vb,auvb', b.x_ov, b.x_ov, eri) * coeff / (2 * si - 1)

    eri = general([orbvs, orbos, orbcs, orbcs]).reshape(
        len(b.vsidx), len(b.osidx), len(b.csidx), len(b.csidx)
    )
    e -= 2 * coeff * eta * lib.einsum('ia,jv,avji', b.x_cv, b.x_co, eri)

    eri = general([orbvs, orbvs, orbos, orbcs]).reshape(
        len(b.vsidx), len(b.vsidx), len(b.osidx), len(b.csidx)
    )
    e -= 2 * coeff * eta * lib.einsum('ia,vb,abvi', b.x_cv, b.x_ov, eri)

    eri1 = general([orbos, orbcs, orbos, orbvs]).reshape(
        len(b.osidx), len(b.csidx), len(b.osidx), len(b.vsidx)
    )
    eri2 = general([orbos, orbvs, orbos, orbcs]).reshape(
        len(b.osidx), len(b.vsidx), len(b.osidx), len(b.csidx)
    )
    e += 2 * coeff * lib.einsum('iu,vb,uivb', b.x_co, b.x_ov, eri1) / (2 * si - 1)
    e -= 2 * coeff * lib.einsum('iu,vb,ubvi', b.x_co, b.x_ov, eri2) / (2 * si - 1)

    eri = general([orbvs, orbos, orbos, orbcs]).reshape(
        len(b.vsidx), len(b.osidx), len(b.osidx), len(b.csidx)
    )
    e -= 2 * coeff * (gamma - 1) * lib.einsum('ia,wv,avwi', b.x_cv, b.x_oo, eri)

    eri = general([orbos, orbos, orbos, orbcs]).reshape(
        len(b.osidx), len(b.osidx), len(b.osidx), len(b.csidx)
    )
    e -= 2 * coeff * zeta * lib.einsum('iu,wv,uvwi', b.x_co, b.x_oo, eri)

    eri = general([orbvs, orbos, orbos, orbos]).reshape(
        len(b.vsidx), len(b.osidx), len(b.osidx), len(b.osidx)
    )
    e -= 2 * coeff * zeta * lib.einsum('ua,wv,avwu', b.x_ov, b.x_oo, eri)
    return float(e)


def satda_hf_exchange_coefficient_energy(tdobj, xy):
    mf = tdobj._scf
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    if not hybrid:
        return 0.0
    e = _satda_hf_exchange_energy_with_coeff(tdobj, xy, hyb)
    if omega != 0:
        e += _satda_hf_exchange_energy_with_coeff(
            tdobj, xy, alpha - hyb, omega=omega
        )
    return e


def _add_eri_term_q(tdobj, q_alpha, q_beta, orb_sets, idx_sets,
                    spin_sets, coeff_tensor, scale=1.0, omega=None):
    coeff_tensor = np.asarray(coeff_tensor)
    if coeff_tensor.size == 0 or scale == 0:
        return

    mol = tdobj._scf.mol
    mo_coeff = tdobj._scf.mo_coeff
    nmo = mo_coeff.shape[1]
    dims = coeff_tensor.shape

    def general(orbs):
        if omega is None or omega == 0:
            return ao2mo.general(mol, orbs, compact=False)
        with mol.with_range_coulomb(omega):
            return ao2mo.general(mol, orbs, compact=False)

    for pos in range(4):
        orbs = list(orb_sets)
        orbs[pos] = mo_coeff
        eri = general(orbs).reshape(
            *(dims[:pos] + (nmo,) + dims[pos + 1:])
        )
        if pos == 0:
            contrib = lib.einsum('pqtu,rqtu->rp', coeff_tensor, eri) * scale
        elif pos == 1:
            contrib = lib.einsum('pqtu,prtu->rq', coeff_tensor, eri) * scale
        elif pos == 2:
            contrib = lib.einsum('pqtu,pqru->rt', coeff_tensor, eri) * scale
        else:
            contrib = lib.einsum('pqtu,pqtr->ru', coeff_tensor, eri) * scale

        target = q_alpha if spin_sets[pos] == 'alpha' else q_beta
        target[:, idx_sets[pos]] += contrib


def _satda_delta_hf_exchange_q_with_coeff(tdobj, xy, coeff=1.0, omega=None):
    mf = tdobj._scf
    b = make_satda_sf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SATDA spin adaptation requires Si > 1/2')

    csidx, osidx, vsidx, orbcs, orbos, orbvs = _satda_orbitals(tdobj)
    nmo = mf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)

    eta = np.sqrt((2 * si + 1) / (2 * si)) - 1
    gamma = np.sqrt((2 * si + 1) / (2 * si - 1))
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1

    def add(orb_sets, idx_sets, spin_sets, tensor, scale):
        _add_eri_term_q(tdobj, q_alpha, q_beta, orb_sets, idx_sets,
                        spin_sets, tensor, scale=coeff * scale, omega=omega)

    add((orbos, orbcs, orbcs, orbos),
        (osidx, csidx, csidx, osidx),
        ('beta', 'alpha', 'alpha', 'beta'),
        lib.einsum('iu,jv->uijv', b.x_co, b.x_co),
        -1.0 / (2 * si - 1))

    add((orbvs, orbos, orbos, orbvs),
        (vsidx, osidx, osidx, vsidx),
        ('beta', 'beta', 'beta', 'beta'),
        lib.einsum('ua,vb->auvb', b.x_ov, b.x_ov),
        -1.0 / (2 * si - 1))

    add((orbvs, orbos, orbcs, orbcs),
        (vsidx, osidx, csidx, csidx),
        ('beta', 'beta', 'alpha', 'alpha'),
        lib.einsum('ia,jv->avji', b.x_cv, b.x_co),
        -2 * eta)

    add((orbvs, orbvs, orbos, orbcs),
        (vsidx, vsidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('ia,vb->abvi', b.x_cv, b.x_ov),
        -2 * eta)

    add((orbos, orbcs, orbos, orbvs),
        (osidx, csidx, osidx, vsidx),
        ('beta', 'alpha', 'beta', 'beta'),
        lib.einsum('iu,vb->uivb', b.x_co, b.x_ov),
        2.0 / (2 * si - 1))

    add((orbos, orbvs, orbos, orbcs),
        (osidx, vsidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('iu,vb->ubvi', b.x_co, b.x_ov),
        -2.0 / (2 * si - 1))

    add((orbvs, orbos, orbos, orbcs),
        (vsidx, osidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('ia,wv->avwi', b.x_cv, b.x_oo),
        -2 * (gamma - 1))

    add((orbos, orbos, orbos, orbcs),
        (osidx, osidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('iu,wv->uvwi', b.x_co, b.x_oo),
        -2 * zeta)

    add((orbvs, orbos, orbos, orbos),
        (vsidx, osidx, osidx, osidx),
        ('beta', 'beta', 'beta', 'beta'),
        lib.einsum('ua,wv->avwu', b.x_ov, b.x_oo),
        -2 * zeta)
    return q_alpha, q_beta


def satda_delta_hf_exchange_q(tdobj, xy):
    '''Unconstrained MO coefficient derivative Q for HF exchange-like terms.'''
    mf = tdobj._scf
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    nmo = mf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    if not hybrid:
        return q_alpha, q_beta

    qa, qb = _satda_delta_hf_exchange_q_with_coeff(tdobj, xy, hyb)
    q_alpha += qa
    q_beta += qb
    if omega != 0:
        qa, qb = _satda_delta_hf_exchange_q_with_coeff(
            tdobj, xy, alpha - hyb, omega=omega
        )
        q_alpha += qa
        q_beta += qb
    return q_alpha, q_beta


def satda_delta_q(tdobj, xy, include_fock=True, include_hf=True,
                  fock_response=True):
    '''Full unconstrained MO derivative Q for the HF-equivalent SATDA terms.'''
    mf = tdobj._scf
    nmo = mf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)

    if include_fock:
        qa, qb = satda_delta_fock_q(tdobj, xy, with_response=fock_response)
        q_alpha += qa
        q_beta += qb
    if include_hf:
        qa, qb = satda_delta_hf_exchange_q(tdobj, xy)
        q_alpha += qa
        q_beta += qb
    return q_alpha, q_beta


def _satda_hf_fock_for_orbs(tdobj, mo_coeff_alpha, mo_coeff_beta):
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


def _satda_hf_energy_for_orbs(tdobj, xy, mo_coeff_alpha, mo_coeff_beta,
                              include_delta=True):
    '''HF Rayleigh quotient in the SF-base plus spin-adaptation form.'''
    mf = tdobj._scf
    mol = mf.mol
    mo_occ = mf.mo_occ
    b = make_satda_sf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SATDA spin adaptation requires Si > 1/2')

    csidx = b.csidx
    osidx = b.osidx
    vsidx = b.vsidx
    ncs = len(csidx)
    nos = len(osidx)
    nvs = len(vsidx)

    ca_c = mo_coeff_alpha[:, csidx]
    ca_o = mo_coeff_alpha[:, osidx]
    ca_v = mo_coeff_alpha[:, vsidx]
    cb_c = mo_coeff_beta[:, csidx]
    cb_o = mo_coeff_beta[:, osidx]
    cb_v = mo_coeff_beta[:, vsidx]
    orboa = np.hstack((ca_c, ca_o))
    orbvb = np.hstack((cb_o, cb_v))

    focka, fockb = _satda_hf_fock_for_orbs(
        tdobj, mo_coeff_alpha, mo_coeff_beta
    )
    x = np.asarray(xy[0])

    fockv = orbvb.conj().T @ fockb @ orbvb
    focko = orboa.conj().T @ focka @ orboa
    e = lib.einsum('ia,ab,ib', x, fockv, x)
    e -= lib.einsum('ia,ji,ja', x, focko, x)

    eri = ao2mo.general(mol, [orboa, orboa, orbvb, orbvb], compact=False)
    eri = eri.reshape(ncs + nos, ncs + nos, nos + nvs, nos + nvs)
    e -= lib.einsum('ia,jb,ijba', x, x, eri)
    if not include_delta:
        return float(e)

    x_co = b.x_co
    x_cv = b.x_cv
    x_oo = b.x_oo
    x_ov = b.x_ov
    tr_oo = np.trace(x_oo)
    eta = np.sqrt((2 * si + 1) / (2 * si)) - 1
    gamma = np.sqrt((2 * si + 1) / (2 * si - 1))
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1
    chi = 1.0 / np.sqrt(2 * si * (2 * si - 1))

    focks_cc = 0.5 * (cb_c.conj().T @ fockb @ cb_c
                      - ca_c.conj().T @ focka @ ca_c)
    focks_vv = 0.5 * (cb_v.conj().T @ fockb @ cb_v
                      - ca_v.conj().T @ focka @ ca_v)
    focks_cv = 0.5 * (cb_c.conj().T @ fockb @ cb_v
                      - ca_c.conj().T @ focka @ ca_v)
    fockb_vo = cb_v.conj().T @ fockb @ cb_o
    fockb_co = cb_c.conj().T @ fockb @ cb_o
    focka_oc = ca_o.conj().T @ focka @ ca_c
    focka_vo = ca_v.conj().T @ focka @ ca_o

    e += lib.einsum('ia,ja,ji', x_cv, x_cv, focks_cc) / si
    e += lib.einsum('ia,ib,ab', x_cv, x_cv, focks_vv) / si
    e += lib.einsum('iu,ju,ji', x_co, x_co, focks_cc) * 2 / (2 * si - 1)
    e += lib.einsum('ua,ub,ab', x_ov, x_ov, focks_vv) * 2 / (2 * si - 1)
    e += 2 * eta * lib.einsum('ia,iv,av', x_cv, x_co, fockb_vo)
    e -= 2 * eta * lib.einsum('ia,va,vi', x_cv, x_ov, focka_oc)
    e += gamma * (1 + 1 / si) * tr_oo * lib.einsum('ia,ia', x_cv, focks_cv)
    e += 2 * chi * tr_oo * lib.einsum('iu,iu', x_co, fockb_co)
    e -= 2 * zeta * lib.einsum('iu,vu,vi', x_co, x_oo, focka_oc)
    e -= 2 * chi * tr_oo * lib.einsum('ua,au', x_ov, focka_vo)
    e += 2 * zeta * lib.einsum('ua,uv,av', x_ov, x_oo, fockb_vo)

    eri = ao2mo.general(mol, [cb_o, ca_c, ca_c, cb_o], compact=False)
    eri = eri.reshape(nos, ncs, ncs, nos)
    e -= lib.einsum('iu,jv,uijv', x_co, x_co, eri) / (2 * si - 1)

    eri = ao2mo.general(mol, [cb_v, cb_o, cb_o, cb_v], compact=False)
    eri = eri.reshape(nvs, nos, nos, nvs)
    e -= lib.einsum('ua,vb,auvb', x_ov, x_ov, eri) / (2 * si - 1)

    eri = ao2mo.general(mol, [cb_v, cb_o, ca_c, ca_c], compact=False)
    eri = eri.reshape(nvs, nos, ncs, ncs)
    e -= 2 * eta * lib.einsum('ia,jv,avji', x_cv, x_co, eri)

    eri = ao2mo.general(mol, [cb_v, cb_v, cb_o, ca_c], compact=False)
    eri = eri.reshape(nvs, nvs, nos, ncs)
    e -= 2 * eta * lib.einsum('ia,vb,abvi', x_cv, x_ov, eri)

    eri1 = ao2mo.general(mol, [cb_o, ca_c, cb_o, cb_v], compact=False)
    eri1 = eri1.reshape(nos, ncs, nos, nvs)
    eri2 = ao2mo.general(mol, [cb_o, cb_v, cb_o, ca_c], compact=False)
    eri2 = eri2.reshape(nos, nvs, nos, ncs)
    e += 2 * lib.einsum('iu,vb,uivb', x_co, x_ov, eri1) / (2 * si - 1)
    e -= 2 * lib.einsum('iu,vb,ubvi', x_co, x_ov, eri2) / (2 * si - 1)

    eri = ao2mo.general(mol, [cb_v, cb_o, cb_o, ca_c], compact=False)
    eri = eri.reshape(nvs, nos, nos, ncs)
    e -= 2 * (gamma - 1) * lib.einsum('ia,wv,avwi', x_cv, x_oo, eri)

    eri = ao2mo.general(mol, [cb_o, cb_o, cb_o, ca_c], compact=False)
    eri = eri.reshape(nos, nos, nos, ncs)
    e -= 2 * zeta * lib.einsum('iu,wv,uvwi', x_co, x_oo, eri)

    eri = ao2mo.general(mol, [cb_v, cb_o, cb_o, cb_o], compact=False)
    eri = eri.reshape(nvs, nos, nos, nos)
    e -= 2 * zeta * lib.einsum('ua,wv,avwu', x_ov, x_oo, eri)
    return float(e)


def _copy_scf_settings(mf_ref, mf):
    mf.verbose = 0
    for key in (
        'max_memory', 'conv_tol', 'conv_tol_grad', 'max_cycle', 'diis_space',
        'diis_start_cycle', 'level_shift', 'damp', 'direct_scf',
    ):
        if hasattr(mf_ref, key) and hasattr(mf, key):
            setattr(mf, key, getattr(mf_ref, key))

    if isinstance(mf_ref, dft.KohnShamDFT):
        mf.grids.level = mf_ref.grids.level
        mf.grids.prune = mf_ref.grids.prune
        mf.grids.radi_method = mf_ref.grids.radi_method
        mf.grids.becke_scheme = mf_ref.grids.becke_scheme
        mf.small_rho_cutoff = mf_ref.small_rho_cutoff
    return mf


def _make_displaced_mf(mf_ref, mol):
    if isinstance(mf_ref, (dft.roks.ROKS, dft.rks_symm.SymAdaptedROKS)):
        mf = mol.ROKS(xc=mf_ref.xc)
    elif isinstance(mf_ref, (scf.rohf.ROHF, scf.hf_symm.SymAdaptedROHF)):
        mf = mol.ROHF()
    else:
        raise NotImplementedError(
            'SATDA finite-difference gradients currently support ROKS/ROHF '
            'references only'
        )
    return _copy_scf_settings(mf_ref, mf)


def _as_spin_unrestricted_reference(mf_ref):
    if isinstance(mf_ref, dft.KohnShamDFT):
        mf = mf_ref.to_uks()
    else:
        mf = mf_ref.to_uhf()
    mf.verbose = 0
    return mf


def _as_dm_stack(dm):
    dm = np.asarray(dm)
    if dm.ndim == 2:
        dm = dm.reshape(1, *dm.shape)
    return dm


def _as_v1_stack(v1):
    v1 = np.asarray(v1)
    if v1.ndim == 3:
        v1 = v1.reshape(1, *v1.shape)
    return v1


def _satda_sf_lda_block_matrix(si):
    a = np.sqrt((2 * si + 1) / (2 * si))
    b = np.sqrt(2 * si / (2 * si - 1))
    c = np.sqrt((2 * si + 1) / (2 * si - 1))
    d = 1.0 / (2 * si - 1)
    return np.asarray((
        (1 + d, a, b, 1.0),
        (a, 1.0, c, a),
        (b, c, 1.0, b),
        (1.0, a, b, 1 + d),
    ))


def _satda_sf_gga_block_matrices(si):
    a = np.sqrt((2 * si + 1) / (2 * si))
    b = np.sqrt(2 * si / (2 * si - 1))
    c = np.sqrt((2 * si + 1) / (2 * si - 1))
    d = 1.0 / (2 * si - 1)
    e = 2 * si / (2 * si - 1)
    m0 = np.asarray((
        (1.0, a, b, e),
        (a, 1.0, c, a),
        (b, c, 1.0, b),
        (e, a, b, 1.0),
    ))
    m1 = np.asarray((
        (d, 0.0, 0.0, -d),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (-d, 0.0, 0.0, d),
    ))
    return m0, m1


# ``fxc_ref`` is built with the ROKS half-density convention used by
# ``cache_xc_kernel(..., spin=1)``.  The SATDA spin-flip block gradient carries
# the corresponding quarter factor, while the sigma-vector contraction itself
# keeps the unscaled matrix in ``satda.py``.
SATDA_SF_LDA_XC_GRAD_SCALE = 0.25


def _satda_sf_transition_blocks(tdobj, xy):
    b = make_satda_sf_blocks(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _satda_orbitals(tdobj)
    blocks = (
        (b.osidx, b.csidx, _mo_pair_dm(orbos, b.x_co.T, orbcs), b.x_co.T),
        (b.vsidx, b.csidx, _mo_pair_dm(orbvs, b.x_cv.T, orbcs), b.x_cv.T),
        (b.osidx, b.osidx, _mo_pair_dm(orbos, b.x_oo.T, orbos), b.x_oo.T),
        (b.vsidx, b.osidx, _mo_pair_dm(orbvs, b.x_ov.T, orbos), b.x_ov.T),
    )
    return b, blocks


def _satda_sf_lda_fxc_ref(tdobj, max_memory=2000):
    mf = tdobj._scf
    ni = mf._numint
    fxc = ni.cache_xc_kernel(
        mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 1,
        max_memory=max_memory,
    )[2]
    return 0.5 * (fxc[0, :, 0] - fxc[0, :, 1]
                  - fxc[1, :, 0] + fxc[1, :, 1])


def _satda_sf_lda_apply_fxc_ref(tdobj, dms, fxc_ref, max_memory=2000):
    mf = tdobj._scf
    dms = np.asarray(dms)
    if dms.ndim == 2:
        dms = dms.reshape(1, *dms.shape)
    return mf._numint.nr_rks_fxc(
        mf.mol, mf.grids, mf.xc, None, dms, 0, 0,
        None, None, fxc_ref, max_memory=max_memory,
    )


def _satda_sf_gga_apply_fxc1_ref(tdobj, dms, fxc_ref, max_memory=2000):
    from pyscf.sftda.satda import nr_rks_fxc1_gga

    mf = tdobj._scf
    dms = np.asarray(dms)
    if dms.ndim == 2:
        dms = dms.reshape(1, *dms.shape)
    return nr_rks_fxc1_gga(
        mf._numint, mf.mol, mf.grids, mf.xc, dms, fxc_ref,
        max_memory=max_memory,
    )


def _satda_sf_gga_primitives(ni, mol, ao, mask, ao_loc, dms):
    shls_slice = (0, mol.nbas)
    prim = []
    for dm in dms:
        c0 = numint._dot_ao_dm(mol, ao[0], dm, mask, shls_slice, ao_loc)
        rho0 = numint._contract_rho(ao[0], c0)
        r_grad = np.asarray([
            numint._contract_rho(ao[i], c0) for i in range(1, 4)
        ])
        c_grad = [
            numint._dot_ao_dm(mol, ao[i], dm, mask, shls_slice, ao_loc)
            for i in range(1, 4)
        ]
        l_grad = np.asarray([
            numint._contract_rho(ao[0], c_grad[i]) for i in range(3)
        ])
        tau = np.empty((3, 3, rho0.size))
        for i in range(3):
            for j in range(3):
                tau[i, j] = numint._contract_rho(ao[j + 1], c_grad[i])
        prim.append((rho0, l_grad, r_grad, tau))
    return prim


def _satda_sf_gga_u_from_primitive(fxc, primitive):
    rho0, l_grad, r_grad, tau = primitive
    ngrids = rho0.size
    u = np.zeros((4, 4, ngrids))
    u[0, 0] = fxc[0, 0] * rho0
    u[0, 0] += lib.einsum('ig,ig->g', fxc[1:4, 0], l_grad)
    u[0, 0] += lib.einsum('jg,jg->g', fxc[0, 1:4], r_grad)
    u[0, 0] += lib.einsum('ijg,ijg->g', fxc[1:4, 1:4], tau)
    u[1:4, 0] += fxc[1:4, 0] * rho0
    u[1:4, 0] += lib.einsum('ijg,jg->ig', fxc[1:4, 1:4], r_grad)
    u[0, 1:4] += fxc[0, 1:4] * rho0
    u[0, 1:4] += lib.einsum('ijg,ig->jg', fxc[1:4, 1:4], l_grad)
    u[1:4, 1:4] += fxc[1:4, 1:4] * rho0
    return u


def _satda_sf_gga_primitive_bilinear_coeff(prim_l, prim_r):
    rho_l, l_l, r_l, tau_l = prim_l
    rho_r, l_r, r_r, tau_r = prim_r
    coeff = np.empty((4, 4, rho_l.size))
    coeff[0, 0] = rho_l * rho_r
    coeff[1:4, 0] = l_r * rho_l + r_l * rho_r
    coeff[0, 1:4] = r_r * rho_l + l_l * rho_r
    coeff[1:4, 1:4] = (
        lib.einsum('g,ijg->ijg', rho_l, tau_r)
        + lib.einsum('ig,jg->ijg', r_l, r_r)
        + lib.einsum('jg,ig->ijg', l_l, l_r)
        + lib.einsum('ijg,g->ijg', tau_l, rho_r)
    )
    return coeff


def _satda_sf_ao_deriv_component(ao, idx, coord):
    if idx == 0:
        return ao[1 + coord]
    if idx == 1:
        return (ao[4], ao[5], ao[6])[coord]
    if idx == 2:
        return (ao[5], ao[7], ao[8])[coord]
    if idx == 3:
        return (ao[6], ao[8], ao[9])[coord]
    raise ValueError('Invalid AO derivative index %d' % idx)


def _satda_sf_gga_eval_k1_mat_deriv(mol, ao, u, mask, ao_loc):
    shls_slice = (0, mol.nbas)
    vmat = np.zeros((4, mol.nao_nr(), mol.nao_nr()))
    for i in range(4):
        for j in range(4):
            if not np.any(u[i, j]):
                continue
            aow = numint._scale_ao(ao[j], u[i, j])
            vmat[0] += numint._dot_ao_ao(
                mol, ao[i], aow, mask, shls_slice, ao_loc
            )
            for x in range(3):
                vmat[x + 1] += numint._dot_ao_ao(
                    mol, _satda_sf_ao_deriv_component(ao, i, x), aow,
                    mask, shls_slice, ao_loc,
                )
    return vmat


def _satda_sf_lda_ref_density_mats(tdobj, xy, with_deriv=False,
                                   max_memory=2000):
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype != 'LDA':
        raise NotImplementedError('SATDA XC analytical gradient currently '
                                  'supports only LDA kernels')

    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    mat = _satda_sf_lda_block_matrix(b.si)
    nao = mol.nao_nr()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    vmat_a = np.zeros((4, nao, nao))
    vmat_b = np.zeros_like(vmat_a)

    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=max_memory):
        ao0 = ao[0]
        rho0 = ni.eval_rho2(
            mol, ao0, mf.mo_coeff, mf.mo_occ, mask, xctype,
            with_lapl=False,
        ) * 0.5
        rho = (rho0, rho0)
        kxc = ni.eval_xc_eff(
            mf.xc, rho, deriv=3, xctype=xctype, spin=1,
        )[3]
        kref_a = 0.5 * (
            kxc[0, 0, 0, 0, 0, 0] - kxc[0, 0, 1, 0, 0, 0]
            - kxc[1, 0, 0, 0, 0, 0] + kxc[1, 0, 1, 0, 0, 0]
        )
        kref_b = 0.5 * (
            kxc[0, 0, 0, 0, 1, 0] - kxc[0, 0, 1, 0, 1, 0]
            - kxc[1, 0, 0, 0, 1, 0] + kxc[1, 0, 1, 0, 1, 0]
        )
        rho_blocks = np.asarray([
            ni.eval_rho(mol, ao0, dm, mask, xctype, hermi=0,
                        with_lapl=False)
            for dm in dms
        ])
        rho_pair = lib.einsum('bl,bg,lg->g', mat, rho_blocks, rho_blocks)
        rho_pair *= SATDA_SF_LDA_XC_GRAD_SCALE
        wv_a = (kref_a * rho_pair * weight).reshape(1, -1)
        wv_b = (kref_b * rho_pair * weight).reshape(1, -1)
        tdrks_grad._lda_eval_mat_(
            mol, vmat_a, ao, wv_a, mask, shls_slice, ao_loc
        )
        tdrks_grad._lda_eval_mat_(
            mol, vmat_b, ao, wv_b, mask, shls_slice, ao_loc
        )

    if with_deriv:
        vmat_a[1:] *= -1
        vmat_b[1:] *= -1
        return vmat_a, vmat_b
    return vmat_a[0], vmat_b[0]


def _satda_sf_gga_ref_density_mats(tdobj, xy, with_deriv=False,
                                   max_memory=2000):
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if ni._xc_type(mf.xc) != 'GGA':
        raise NotImplementedError('SATDA/GGA reference response requested '
                                  'for a non-GGA functional')

    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    m0, m1 = _satda_sf_gga_block_matrices(b.si)
    nao = mol.nao_nr()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    vmat_a = np.zeros((4, nao, nao))
    vmat_b = np.zeros_like(vmat_a)
    ao_deriv = 2

    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, ao_deriv, max_memory=max_memory):
        rho0 = ni.eval_rho2(
            mol, ao[:4], mf.mo_coeff, mf.mo_occ, mask, 'GGA',
            with_lapl=False,
        ) * 0.5
        rho = (rho0, rho0)
        kxc = ni.eval_xc_eff(
            mf.xc, rho, deriv=3, xctype='GGA', spin=1,
        )[3]
        kref_a = 0.5 * (
            kxc[0, :, 0, :, 0, :, :] - kxc[0, :, 1, :, 0, :, :]
            - kxc[1, :, 0, :, 0, :, :] + kxc[1, :, 1, :, 0, :, :]
        )
        kref_b = 0.5 * (
            kxc[0, :, 0, :, 1, :, :] - kxc[0, :, 1, :, 1, :, :]
            - kxc[1, :, 0, :, 1, :, :] + kxc[1, :, 1, :, 1, :, :]
        )
        rho_blocks = np.asarray([
            ni.eval_rho(mol, ao[:4], dm, mask, 'GGA', hermi=0,
                        with_lapl=False)
            for dm in dms
        ])
        prim = _satda_sf_gga_primitives(ni, mol, ao, mask, ao_loc, dms)

        coeff0 = lib.einsum('bl,bxg,lyg->xyg', m0, rho_blocks, rho_blocks)
        coeff1 = np.zeros_like(coeff0)
        for ib in range(4):
            for il in range(4):
                if m1[ib, il] != 0:
                    coeff1 += m1[ib, il] * _satda_sf_gga_primitive_bilinear_coeff(
                        prim[ib], prim[il]
                    )
        coeff = (coeff0 + coeff1) * SATDA_SF_LDA_XC_GRAD_SCALE
        wv_a = lib.einsum('xyg,xyzg,g->zg', coeff, kref_a, weight)
        wv_b = lib.einsum('xyg,xyzg,g->zg', coeff, kref_b, weight)
        tdrks_grad._gga_eval_mat_(
            mol, vmat_a, ao, wv_a, mask, shls_slice, ao_loc
        )
        tdrks_grad._gga_eval_mat_(
            mol, vmat_b, ao, wv_b, mask, shls_slice, ao_loc
        )

    if with_deriv:
        vmat_a[1:] *= -1
        vmat_b[1:] *= -1
        return vmat_a, vmat_b
    return vmat_a[0], vmat_b[0]


def _satda_sf_lda_transition_deriv_mats(tdobj, xy, fxc_ref,
                                        max_memory=2000):
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    mat = _satda_sf_lda_block_matrix(b.si)
    nao = mol.nao_nr()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    fxc_ref = np.asarray(fxc_ref).reshape(-1, np.asarray(fxc_ref).shape[-1])
    src = np.zeros((4, 4, nao, nao))

    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        ao0 = ao[0]
        fxc_blk = fxc_ref[:, p0:p1]
        for iblk, dm in enumerate(dms):
            rho = ni.eval_rho(
                mol, ao0, dm, mask, xctype, hermi=0, with_lapl=False
            )
            wv = fxc_blk * rho.reshape(1, -1) * weight
            tdrks_grad._lda_eval_mat_(
                mol, src[iblk], ao, wv, mask, shls_slice, ao_loc
            )

    src[:, 1:] *= -1
    out = np.zeros_like(src)
    for iblk in range(4):
        out[iblk] = (
            SATDA_SF_LDA_XC_GRAD_SCALE
            * lib.einsum('l,lxpq->xpq', mat[iblk], src)
        )
    return out


def _satda_sf_gga_transition_deriv_mats(tdobj, xy, fxc_ref,
                                        max_memory=2000):
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    m0, m1 = _satda_sf_gga_block_matrices(b.si)
    nao = mol.nao_nr()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    src0 = np.zeros((4, 4, nao, nao))
    src1 = np.zeros_like(src0)

    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, 2, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        fxc_blk = fxc_ref[:, :, p0:p1]
        prim = _satda_sf_gga_primitives(ni, mol, ao, mask, ao_loc, dms)
        for iblk, dm in enumerate(dms):
            rho = ni.eval_rho(
                mol, ao[:4], dm, mask, 'GGA', hermi=0, with_lapl=False
            )
            wv = lib.einsum('xg,xyg,g->yg', rho, fxc_blk, weight)
            tdrks_grad._gga_eval_mat_(
                mol, src0[iblk], ao, wv, mask, shls_slice, ao_loc
            )
            u = _satda_sf_gga_u_from_primitive(fxc_blk * weight, prim[iblk])
            src1[iblk] += _satda_sf_gga_eval_k1_mat_deriv(
                mol, ao, u, mask, ao_loc
            )

    src0[:, 1:] *= -1
    src1[:, 1:] *= -1
    out = np.zeros_like(src0)
    for iblk in range(4):
        out[iblk] = SATDA_SF_LDA_XC_GRAD_SCALE * (
            lib.einsum('l,lxpq->xpq', m0[iblk], src0)
            + lib.einsum('l,lxpq->xpq', m1[iblk], src1)
        )
    return out


def _satda_sf_lda_xc_q(tdobj, xy, max_memory=2000):
    mf = tdobj._scf
    if mf._numint._xc_type(mf.xc) != 'LDA':
        nmo = mf.mo_coeff.shape[1]
        return np.zeros((nmo, nmo)), np.zeros((nmo, nmo))

    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    mat = _satda_sf_lda_block_matrix(b.si)
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)

    fxc_ref = _satda_sf_lda_fxc_ref(tdobj, max_memory=max_memory)
    vsrc = _satda_sf_lda_apply_fxc_ref(
        tdobj, [blk[2] for blk in blocks], fxc_ref, max_memory=max_memory
    )
    vblocks = np.asarray([
        2.0 * SATDA_SF_LDA_XC_GRAD_SCALE
        * lib.einsum('l,lpq->pq', mat[iblk], vsrc)
        for iblk in range(4)
    ])

    for vmat, (target_idx, source_idx, dm, amp_t_s) in zip(vblocks, blocks):
        vmo = mo_coeff.conj().T @ vmat @ mo_coeff
        q_beta[:, target_idx] += vmo[:, source_idx] @ amp_t_s.T
        q_alpha[:, source_idx] += vmo[:, target_idx] @ amp_t_s

    wa, wb = _satda_sf_lda_ref_density_mats(
        tdobj, xy, with_deriv=False, max_memory=max_memory
    )
    occidxa = np.where(mf.mo_occ > 0)[0]
    occidxb = np.where(mf.mo_occ == 2)[0]
    q_alpha[:, occidxa] += mo_coeff.conj().T @ (wa + wa.T) @ mo_coeff[:, occidxa]
    q_beta[:, occidxb] += mo_coeff.conj().T @ (wb + wb.T) @ mo_coeff[:, occidxb]
    return q_alpha, q_beta


def _satda_sf_gga_xc_q(tdobj, xy, max_memory=2000):
    mf = tdobj._scf
    if mf._numint._xc_type(mf.xc) != 'GGA':
        nmo = mf.mo_coeff.shape[1]
        return np.zeros((nmo, nmo)), np.zeros((nmo, nmo))

    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    m0, m1 = _satda_sf_gga_block_matrices(b.si)
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)

    fxc_ref = _satda_sf_lda_fxc_ref(tdobj, max_memory=max_memory)
    dms = [blk[2] for blk in blocks]
    v0 = _satda_sf_lda_apply_fxc_ref(
        tdobj, dms, fxc_ref, max_memory=max_memory
    )
    v1 = _satda_sf_gga_apply_fxc1_ref(
        tdobj, dms, fxc_ref, max_memory=max_memory
    )
    vblocks = np.asarray([
        2.0 * SATDA_SF_LDA_XC_GRAD_SCALE
        * (lib.einsum('l,lpq->pq', m0[iblk], v0)
           + lib.einsum('l,lpq->pq', m1[iblk], v1))
        for iblk in range(4)
    ])

    for vmat, (target_idx, source_idx, dm, amp_t_s) in zip(vblocks, blocks):
        vmo = mo_coeff.conj().T @ vmat @ mo_coeff
        q_beta[:, target_idx] += vmo[:, source_idx] @ amp_t_s.T
        q_alpha[:, source_idx] += vmo[:, target_idx] @ amp_t_s

    wa, wb = _satda_sf_gga_ref_density_mats(
        tdobj, xy, with_deriv=False, max_memory=max_memory
    )
    occidxa = np.where(mf.mo_occ > 0)[0]
    occidxb = np.where(mf.mo_occ == 2)[0]
    q_alpha[:, occidxa] += mo_coeff.conj().T @ (wa + wa.T) @ mo_coeff[:, occidxa]
    q_beta[:, occidxb] += mo_coeff.conj().T @ (wb + wb.T) @ mo_coeff[:, occidxb]
    return q_alpha, q_beta


def _satda_sf_lda_xc_direct_de(td_grad, tdobj, xy, atmlst, offsetdic,
                               oo0a, oo0b, max_memory=2000):
    mf = tdobj._scf
    if mf._numint._xc_type(mf.xc) != 'LDA':
        return np.zeros((len(tuple(atmlst)), 3))

    atmlst = tuple(atmlst)
    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    fxc_ref = _satda_sf_lda_fxc_ref(tdobj, max_memory=max_memory)
    trans_der = _satda_sf_lda_transition_deriv_mats(
        tdobj, xy, fxc_ref, max_memory=max_memory
    )
    wa_der, wb_der = _satda_sf_lda_ref_density_mats(
        tdobj, xy, with_deriv=True, max_memory=max_memory
    )

    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        for fder, dm in zip(trans_der[:, 1:], dms):
            de[k] += lib.einsum('xpq,pq->x', fder[:, p0:p1], dm[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', fder[:, p0:p1], dm.T[p0:p1]) * 2
        de[k] += lib.einsum('xpq,pq->x', wa_der[1:, p0:p1], oo0a[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wa_der[1:, p0:p1], oo0a.T[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wb_der[1:, p0:p1], oo0b[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wb_der[1:, p0:p1], oo0b.T[p0:p1])
    return de


def _satda_sf_gga_xc_direct_de(td_grad, tdobj, xy, atmlst, offsetdic,
                               oo0a, oo0b, max_memory=2000):
    mf = tdobj._scf
    if mf._numint._xc_type(mf.xc) != 'GGA':
        return np.zeros((len(tuple(atmlst)), 3))

    atmlst = tuple(atmlst)
    b, blocks = _satda_sf_transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    fxc_ref = _satda_sf_lda_fxc_ref(tdobj, max_memory=max_memory)
    trans_der = _satda_sf_gga_transition_deriv_mats(
        tdobj, xy, fxc_ref, max_memory=max_memory
    )
    wa_der, wb_der = _satda_sf_gga_ref_density_mats(
        tdobj, xy, with_deriv=True, max_memory=max_memory
    )

    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        for fder, dm in zip(trans_der[:, 1:], dms):
            de[k] += lib.einsum('xpq,pq->x', fder[:, p0:p1], dm[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', fder[:, p0:p1], dm.T[p0:p1]) * 2
        de[k] += lib.einsum('xpq,pq->x', wa_der[1:, p0:p1], oo0a[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wa_der[1:, p0:p1], oo0a.T[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wb_der[1:, p0:p1], oo0b[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wb_der[1:, p0:p1], oo0b.T[p0:p1])
    return de


def _contract_uks_lda_vxc_deriv(td_grad, mf, dmoo=None, max_memory=2000):
    mol = td_grad.mol
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    else:
        raise NotImplementedError('Only LDA/GGA XC AO derivatives are available '
                                  'in the SATDA experimental DFT path')

    nao = mf.mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    vxc1 = np.zeros((2, 4, nao, nao))
    if dmoo is None:
        f1dm = None
    else:
        dmoo = np.asarray(dmoo)
        f1dm = np.zeros((2, 4, nao, nao))

    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, ao_deriv, max_memory=max_memory):
        ao0 = ao[0] if xctype == 'LDA' else ao[:4]
        rho = (
            ni.eval_rho2(mol, ao0, mf.mo_coeff[0], mf.mo_occ[0], mask,
                         xctype, with_lapl=False),
            ni.eval_rho2(mol, ao0, mf.mo_coeff[1], mf.mo_occ[1], mask,
                         xctype, with_lapl=False),
        )
        vxc, fxc = ni.eval_xc_eff(
            mf.xc, rho, deriv=2, xctype=xctype, spin=1,
        )[1:3]
        fmat_(
            mol, vxc1[0], ao, vxc[0] * weight, mask, shls_slice, ao_loc
        )
        fmat_(
            mol, vxc1[1], ao, vxc[1] * weight, mask, shls_slice, ao_loc
        )

        if dmoo is not None:
            rho2 = np.asarray((
                ni.eval_rho(mol, ao0, dmoo[0], mask, xctype, hermi=1,
                            with_lapl=False),
                ni.eval_rho(mol, ao0, dmoo[1], mask, xctype, hermi=1,
                            with_lapl=False),
            ))
            if xctype == 'LDA':
                rho2 = rho2[:, np.newaxis]
            wv = lib.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(
                mol, f1dm[0], ao, wv[0], mask, shls_slice, ao_loc
            )
            fmat_(
                mol, f1dm[1], ao, wv[1], mask, shls_slice, ao_loc
            )

    vxc1[:, 1:] *= -1
    if f1dm is not None:
        f1dm[:, 1:] *= -1
    return vxc1, f1dm


def _add_j_bilinear_ip1(de, td_grad, mol, dm_l, dm_r, atmlst, offsetdic,
                        scale=1.0, omega=None):
    '''Add direct ERI derivative for scale * sum L[pq] R[tu] (pq|tu).'''
    if scale == 0:
        return

    dm_l = _as_dm_stack(dm_l)
    dm_r = _as_dm_stack(dm_r)
    if len(dm_l) != len(dm_r):
        raise ValueError('Bilinear density stacks have different lengths')
    if len(dm_l) == 0 or not np.any(dm_l) or not np.any(dm_r):
        return

    vj_r = _as_v1_stack(td_grad.get_j(mol, dm_r, hermi=0, omega=omega))
    vj_l = _as_v1_stack(td_grad.get_j(mol, dm_l, hermi=0, omega=omega))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        term = lib.einsum('nxpq,npq->x', vj_r[:, :, p0:p1], dm_l[:, p0:p1])
        term += lib.einsum('nxpq,nqp->x', vj_r[:, :, p0:p1], dm_l[:, :, p0:p1])
        term += lib.einsum('nxpq,npq->x', vj_l[:, :, p0:p1], dm_r[:, p0:p1])
        term += lib.einsum('nxpq,nqp->x', vj_l[:, :, p0:p1], dm_r[:, :, p0:p1])
        de[k] += scale * term


def _add_j_bilinear_ip1_batches(de, td_grad, mol, dm_l, dm_r, atmlst,
                                offsetdic, scale=1.0, omega=None,
                                blksize=64):
    if not dm_l:
        return
    for p0 in range(0, len(dm_l), blksize):
        p1 = min(p0 + blksize, len(dm_l))
        _add_j_bilinear_ip1(
            de, td_grad, mol, np.asarray(dm_l[p0:p1]),
            np.asarray(dm_r[p0:p1]), atmlst, offsetdic,
            scale=scale, omega=omega,
        )


def _satda_delta_hf_exchange_direct_with_coeff(
        de, td_grad, tdobj, xy, atmlst, offsetdic, coeff=1.0, omega=None):
    '''Direct AO ERI derivative for the HF exchange-like SATDA terms.'''
    b = make_satda_sf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SATDA spin adaptation requires Si > 1/2')

    mol = td_grad.mol
    _, _, _, orbcs, orbos, orbvs = _satda_orbitals(tdobj)
    ncs = len(b.csidx)
    nos = len(b.osidx)

    eta = np.sqrt((2 * si + 1) / (2 * si)) - 1
    gamma = np.sqrt((2 * si + 1) / (2 * si - 1))
    zeta = np.sqrt(2 * si / (2 * si - 1)) - 1

    def add(dm_l, dm_r, scale):
        _add_j_bilinear_ip1(
            de, td_grad, mol, dm_l, dm_r, atmlst, offsetdic,
            scale=coeff * scale, omega=omega,
        )

    def add_batches(dm_l, dm_r, scale):
        _add_j_bilinear_ip1_batches(
            de, td_grad, mol, dm_l, dm_r, atmlst, offsetdic,
            scale=coeff * scale, omega=omega,
        )

    add(_mo_pair_dm(orbos, b.x_co.T, orbcs),
        _mo_pair_dm(orbcs, b.x_co, orbos),
        -1.0 / (2 * si - 1))

    add(_mo_pair_dm(orbvs, b.x_ov.T, orbos),
        _mo_pair_dm(orbos, b.x_ov, orbvs),
        -1.0 / (2 * si - 1))

    dm_l = []
    dm_r = []
    for i in range(ncs):
        for j in range(ncs):
            dm_l.append(_mo_pair_dm(orbvs, np.outer(b.x_cv[i], b.x_co[j]), orbos))
            dm_r.append(np.outer(orbcs[:, j], orbcs[:, i].conj()))
    add_batches(dm_l, dm_r, -2 * eta)

    dm_l = []
    dm_r = []
    for i in range(ncs):
        for v in range(nos):
            dm_l.append(_mo_pair_dm(orbvs, np.outer(b.x_cv[i], b.x_ov[v]), orbvs))
            dm_r.append(np.outer(orbos[:, v], orbcs[:, i].conj()))
    add_batches(dm_l, dm_r, -2 * eta)

    add(_mo_pair_dm(orbos, b.x_co.T, orbcs),
        _mo_pair_dm(orbos, b.x_ov, orbvs),
        2.0 / (2 * si - 1))

    dm_l = []
    dm_r = []
    for i in range(ncs):
        for v in range(nos):
            dm_l.append(_mo_pair_dm(orbos, np.outer(b.x_co[i], b.x_ov[v]), orbvs))
            dm_r.append(np.outer(orbos[:, v], orbcs[:, i].conj()))
    add_batches(dm_l, dm_r, -2.0 / (2 * si - 1))

    dm_l = []
    dm_r = []
    for i in range(ncs):
        for w in range(nos):
            dm_l.append(_mo_pair_dm(orbvs, np.outer(b.x_cv[i], b.x_oo[w]), orbos))
            dm_r.append(np.outer(orbos[:, w], orbcs[:, i].conj()))
    add_batches(dm_l, dm_r, -2 * (gamma - 1))

    dm_l = []
    dm_r = []
    for i in range(ncs):
        for w in range(nos):
            dm_l.append(_mo_pair_dm(orbos, np.outer(b.x_co[i], b.x_oo[w]), orbos))
            dm_r.append(np.outer(orbos[:, w], orbcs[:, i].conj()))
    add_batches(dm_l, dm_r, -2 * zeta)

    dm_l = []
    dm_r = []
    for u in range(nos):
        for w in range(nos):
            dm_l.append(_mo_pair_dm(orbvs, np.outer(b.x_ov[u], b.x_oo[w]), orbos))
            dm_r.append(np.outer(orbos[:, w], orbos[:, u].conj()))
    add_batches(dm_l, dm_r, -2 * zeta)


def satda_delta_hf_exchange_direct_de(td_grad, tdobj, xy, atmlst, offsetdic):
    '''Direct AO ERI derivative for all HF exchange-like SATDA blocks.'''
    mf = tdobj._scf
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    de = np.zeros((len(tuple(atmlst)), 3))
    if not hybrid:
        return de

    _satda_delta_hf_exchange_direct_with_coeff(
        de, td_grad, tdobj, xy, atmlst, offsetdic, coeff=hyb
    )
    if omega != 0:
        _satda_delta_hf_exchange_direct_with_coeff(
            de, td_grad, tdobj, xy, atmlst, offsetdic,
            coeff=alpha - hyb, omega=omega,
        )
    return de


def grad_elec_hf_experimental(td_grad, x_y, atmlst=None,
                              max_memory=2000, verbose=logger.INFO):
    '''Development electronic gradient for SATDA deltaS=-1.

    HF uses the equivalence between the ROKS-native SATDA sigma vector and the
    SF-base plus spin-adaptation decomposition.  LDA adds the SATDA-specific
    block XC kernel contractions on top of the same spin-unrestricted CPHF
    layout.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    tdobj = td_grad.base
    mf_ref = tdobj._scf
    mf = _as_spin_unrestricted_reference(tdobj._scf)
    xctype_ref = (mf_ref._numint._xc_type(mf_ref.xc)
                  if isinstance(mf_ref, dft.KohnShamDFT) else 'HF')
    if xctype_ref not in ('HF', 'LDA', 'GGA'):
        raise NotImplementedError(
            'SATDA analytical gradients with XC kernels currently support '
            'only LDA/GGA.  Use finite_diff for MGGA functionals.'
        )
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf_ref)

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    x = np.asarray(x_y[0])
    y = np.zeros((noccb, nvira))

    dvva = lib.einsum('ia,ib->ab', y, y)
    dvvb = lib.einsum('ia,ib->ab', x, x)
    dooa = -lib.einsum('ia,ja->ij', x, x)
    doob = -lib.einsum('ia,ja->ij', y, y)

    dmzooa = reduce(np.dot, (orboa, dooa, orboa.T))
    dmzooa += reduce(np.dot, (orbva, dvva, orbva.T))
    dmzoob = reduce(np.dot, (orbob, doob, orbob.T))
    dmzoob += reduce(np.dot, (orbvb, dvvb, orbvb.T))

    dmx = reduce(np.dot, (orbvb, x.T, orboa.T))
    dmy = reduce(np.dot, (orbob, y, orbva.T))
    dmt = dmx + dmy

    vresp = mf.gen_response(hermi=1)
    veff0doo = vresp(np.asarray((dmzooa, dmzoob)))
    if hybrid:
        vk1 = mf.get_k(mol, dmt, hermi=0) * hyb
        if omega != 0:
            vk1 += mf.get_k(mol, dmt, hermi=0, omega=omega) * (alpha - hyb)
        veff0mo = reduce(np.dot, (mo_coeff[1].T, -vk1, mo_coeff[0]))
    else:
        vk1 = np.zeros_like(dmt)
        veff0mo = np.zeros((nmob, nmoa))

    wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa))
    wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob))
    wvoa += lib.einsum('ac,ka->ck', veff0mo[noccb:, nocca:], x)
    wvoa -= lib.einsum('jk,jc->ck', veff0mo[:noccb, :nocca], y)
    wvob += lib.einsum('ac,ka->ck', veff0mo.T[nocca:, noccb:], y)
    wvob -= lib.einsum('jk,jc->ck', veff0mo.T[:nocca, :noccb], x)

    q_delta_a, q_delta_b = satda_delta_q(tdobj, x_y)
    if xctype_ref == 'LDA':
        q_xc_a, q_xc_b = _satda_sf_lda_xc_q(
            tdobj, x_y, max_memory=max_memory
        )
        q_delta_a += q_xc_a
        q_delta_b += q_xc_b
    elif xctype_ref == 'GGA':
        q_xc_a, q_xc_b = _satda_sf_gga_xc_q(
            tdobj, x_y, max_memory=max_memory
        )
        q_delta_a += q_xc_a
        q_delta_b += q_xc_b
    r_delta_a = q_delta_a - q_delta_a.T
    r_delta_b = q_delta_b - q_delta_b.T
    wvoa += r_delta_a[np.ix_(viridxa, occidxa)]
    wvob += r_delta_b[np.ix_(viridxb, occidxb)]

    def fvind(z):
        za = z[0, :nvira * nocca].reshape(nvira, nocca)
        zb = z[0, nvira * nocca:].reshape(nvirb, noccb)
        dma = reduce(np.dot, (orbva, za, orboa.T))
        dmb = reduce(np.dot, (orbvb, zb, orbob.T))
        dm1 = np.stack((dma + dma.T, dmb + dmb.T))
        v1 = vresp(dm1)
        v1a = reduce(np.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(np.dot, (orbvb.T, v1[1], orbob))
        return np.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(
        fvind, mo_energy, mo_occ, (wvoa, wvob),
        max_cycle=td_grad.cphf_max_cycle,
        tol=td_grad.cphf_conv_tol,
    )[0]
    time1 = log.timer('SATDA/HF Z-vector using UCPHF solver', *time0)

    z1ao = np.empty((2, nao, nao))
    z1ao[0] = reduce(np.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(np.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)))

    im0a = np.zeros((nmoa, nmoa))
    im0b = np.zeros((nmob, nmob))
    im0a[:nocca, :nocca] = reduce(np.dot, (orboa.T, veff0doo[0] + veff[0], orboa))
    im0b[:noccb, :noccb] = reduce(np.dot, (orbob.T, veff0doo[1] + veff[1], orbob))
    im0a[:nocca, :nocca] += lib.einsum('al,ka->lk', veff0mo[noccb:, :nocca], x)
    im0b[:noccb, :noccb] += lib.einsum('al,ka->lk', veff0mo.T[nocca:, :noccb], y)
    im0a[nocca:, nocca:] = lib.einsum('jd,jc->dc', veff0mo[:noccb, nocca:], y)
    im0b[noccb:, noccb:] = lib.einsum('jd,jc->dc', veff0mo.T[:nocca, noccb:], x)
    im0a[:nocca, nocca:] = lib.einsum('jk,jc->kc', veff0mo[:noccb, :nocca], y) * 2
    im0b[:noccb, noccb:] = lib.einsum('jk,jc->kc', veff0mo.T[:nocca, :noccb], x) * 2
    im0a += (q_delta_a + q_delta_a.T) * 0.5
    im0b += (q_delta_b + q_delta_b.T) * 0.5

    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[nocca:, :nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:, :noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca, nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb, noccb:] = mo_energy[1][noccb:]
    dm1a = np.zeros((nmoa, nmoa))
    dm1b = np.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = dooa
    dm1b[:noccb, :noccb] = doob
    dm1a[nocca:, nocca:] = dvva
    dm1b[noccb:, noccb:] = dvvb
    dm1a[nocca:, :nocca] = z1a * 2
    dm1b[noccb:, :noccb] = z1b * 2
    dm1a[:nocca, :nocca] += np.eye(nocca)
    dm1b[:noccb, :noccb] += np.eye(noccb)
    im0a = reduce(np.dot, (mo_coeff[0], im0a + zeta_a * dm1a, mo_coeff[0].T))
    im0b = reduce(np.dot, (mo_coeff[1], im0b + zeta_b * dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    mf_grad = tdobj._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1dooa = 4 * z1ao[0] + 2 * dmzooa
    dmz1doob = 4 * z1ao[1] + 2 * dmzoob
    dm_probe_a, dm_probe_b = satda_fock_probe_densities(tdobj, x_y)
    dmz1dooa_direct = dmz1dooa + 2 * dm_probe_a
    dmz1doob_direct = dmz1doob + 2 * dm_probe_b
    oo0a = reduce(np.dot, (orboa, orboa.T))
    oo0b = reduce(np.dot, (orbob, orbob.T))
    as_dm1 = oo0a + oo0b + (dmz1dooa_direct + dmz1doob_direct) * 0.5

    dm = (oo0a, dmz1dooa_direct + dmz1dooa_direct.T,
          oo0b, dmz1doob_direct + dmz1doob_direct.T)
    if hybrid:
        vj, vk = td_grad.get_jk(mol, dm, hermi=1)
        vj = vj.reshape(2, 2, 3, nao, nao)
        vk = vk.reshape(2, 2, 3, nao, nao) * hyb
        if omega != 0:
            vk += td_grad.get_k(
                mol, dm, hermi=1, omega=omega
            ).reshape(2, 2, 3, nao, nao) * (alpha - hyb)
        veff1 = vj[0] + vj[1] - vk
        vk1 = -td_grad.get_k(mol, (dmt, dmt.T)) * hyb
        if omega != 0:
            vk1 += -td_grad.get_k(
                mol, (dmt, dmt.T), omega=omega
            ) * (alpha - hyb)
    else:
        vj = td_grad.get_j(mol, dm, hermi=1).reshape(2, 2, 3, nao, nao)
        veff1 = vj[0] + vj[1]
        veff1 = np.stack((veff1, veff1))
        vk1 = np.zeros((2, 3, nao, nao))
    if xctype_ref in ('LDA', 'GGA'):
        vxc1, f1dm = _contract_uks_lda_vxc_deriv(
            td_grad, mf,
            dmoo=(dmz1dooa_direct + dmz1dooa_direct.T,
                  dmz1doob_direct + dmz1doob_direct.T),
            max_memory=max_memory,
        )
        veff1[:, 0] += vxc1[:, 1:]
        veff1[:, 1] += f1dm[:, 1:]
    veff1a, veff1b = veff1
    time1 = log.timer('SATDA/HF 2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    offsetdic = mol.offset_nr_by_atom()
    de = np.zeros((len(atmlst), 3))
    de += satda_delta_hf_exchange_direct_de(
        td_grad, tdobj, x_y, atmlst, offsetdic
    )
    if xctype_ref == 'LDA':
        de += _satda_sf_lda_xc_direct_de(
            td_grad, tdobj, x_y, atmlst, offsetdic, oo0a, oo0b,
            max_memory=max_memory,
        )
    elif xctype_ref == 'GGA':
        de += _satda_sf_gga_xc_direct_de(
            td_grad, tdobj, x_y, atmlst, offsetdic, oo0a, oo0b,
            max_memory=max_memory,
        )

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)
        de[k] += lib.einsum('xpq,pq->x', h1ao, as_dm1)
        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], oo0a[p0:p1]) * 2
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], oo0b[p0:p1]) * 2

        de[k] -= lib.einsum('xpq,pq->x', s1[:, p0:p1], im0[p0:p1])
        de[k] -= lib.einsum('xqp,pq->x', s1[:, p0:p1], im0[:, p0:p1])

        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], dmz1dooa_direct[p0:p1]) * .5
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], dmz1doob_direct[p0:p1]) * .5
        de[k] += lib.einsum('xpq,qp->x', veff1a[0, :, p0:p1], dmz1dooa_direct[:, p0:p1]) * .5
        de[k] += lib.einsum('xpq,qp->x', veff1b[0, :, p0:p1], dmz1doob_direct[:, p0:p1]) * .5
        de[k] += lib.einsum('xij,ij->x', veff1a[1, :, p0:p1], oo0a[p0:p1]) * .5
        de[k] += lib.einsum('xij,ij->x', veff1b[1, :, p0:p1], oo0b[p0:p1]) * .5

        if hybrid:
            de[k] += lib.einsum('xpq,pq->x', vk1[0, :, p0:p1], dmt[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', vk1[1, :, p0:p1], dmt.T[p0:p1]) * 2
        de[k] += td_grad.extra_force(ia, locals())

    log.timer('SATDA electronic nuclear gradients', *time0)
    return de


class Gradients(rhf_grad.GradientsBase):
    '''Nuclear gradients for :class:`pyscf.sftda.satda.SATDA`.

    The returned gradient is for the total excited-state energy

        E_ref + omega_SATDA

    matching the convention of PySCF TD gradient objects.  Root tracking is
    performed by overlap of the flattened SATDA X amplitudes with the reference
    root, which supports both ``deltaS=-1`` and ``deltaS=0`` amplitude layouts.
    '''

    _keys = rhf_grad.GradientsBase._keys | {
        'state', 'step', 'nstates', 'root_overlap_tol', 'method',
        'cphf_max_cycle', 'cphf_conv_tol',
    }

    def __init__(self, td):
        rhf_grad.GradientsBase.__init__(self, td)
        self.state = 1
        self.step = 1e-3
        self.nstates = td.nstates
        self.root_overlap_tol = 0.7
        self.method = 'finite_diff'
        self.cphf_max_cycle = 50
        self.cphf_conv_tol = 1e-8

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** SATDA gradients for %s ********',
                 self.base.__class__)
        log.info('State ID = %d', self.state)
        log.info('step = %.6g Bohr', self.step)
        log.info('nstates = %d', self.nstates)
        log.info('root_overlap_tol = %.6g', self.root_overlap_tol)
        log.info('method = %s', self.method)
        log.info('cphf_conv_tol = %.6g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('unit = Eh/Bohr')
        if self.method == 'finite_diff':
            log.warn('Using central finite differences of E_ref + omega.')
        elif self.method == 'analytic_experimental':
            log.warn('method="analytic_experimental" is an HF-only development '
                     'path for SATDA deltaS=-1.')
        return self

    def _run_td_at(self, coords_bohr, x_ref=None):
        mol = self.mol.copy()
        mol.set_geom_(coords_bohr, unit='Bohr')

        mf = _make_displaced_mf(self.base._scf, mol)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Displaced ROKS/ROHF reference did not converge')

        td = self.base.__class__(mf)
        td.deltaS = self.base.deltaS
        td.nstates = self.nstates
        td.conv_tol = self.base.conv_tol
        td.lindep = self.base.lindep
        td.max_cycle = self.base.max_cycle
        td.max_memory = self.base.max_memory
        td.verbose = 0
        td.kernel()

        if x_ref is None:
            root = self.state - 1
            overlap = 1.0
        else:
            overlaps = np.array([_amplitude_overlap(x_ref, td, i)
                                 for i in range(len(td.e))])
            root = int(np.argmax(overlaps))
            overlap = overlaps[root]
            if overlap < self.root_overlap_tol:
                raise RuntimeError(
                    'State tracking failed: best overlap %.6f below threshold '
                    '%.6f' % (overlap, self.root_overlap_tol)
                )

        if not np.asarray(td.converged)[root]:
            raise RuntimeError(
                'Displaced SATDA tracked root %d did not converge' % (root + 1)
            )

        return mf.e_tot + td.e[root], root, overlap

    def _kernel_finite_diff(self, atmlst):
        coords0 = self.mol.atom_coords()
        x_ref = _normalized_x(self.base, self.state - 1)
        atmlst = tuple(atmlst)
        de = np.zeros((len(atmlst), 3))

        for k, ia in enumerate(atmlst):
            for xyz in range(3):
                coords_p = coords0.copy()
                coords_m = coords0.copy()
                coords_p[ia, xyz] += self.step
                coords_m[ia, xyz] -= self.step

                e_p, root_p, ovlp_p = self._run_td_at(coords_p, x_ref)
                e_m, root_m, ovlp_m = self._run_td_at(coords_m, x_ref)
                de[k, xyz] = (e_p - e_m) / (2 * self.step)

                logger.debug(
                    self,
                    'atom %d xyz %d roots %+d/%+d overlaps %.6f/%.6f',
                    ia, xyz, root_p + 1, root_m + 1, ovlp_p, ovlp_m,
                )
        return de

    def _kernel_analytic(self, xy, atmlst):
        raise NotImplementedError(
            'Full SATDA analytical nuclear gradients are not complete yet.  '
            'Use method="analytic_experimental" for the HF-only deltaS=-1 '
            'development implementation, or method="finite_diff" for central '
            'finite differences of E_ref + omega.'
        )

    def _kernel_analytic_experimental(self, xy, atmlst):
        if getattr(self.base, 'deltaS', None) != -1:
            raise NotImplementedError(
                'The experimental analytical SATDA gradient currently supports '
                'only deltaS=-1.'
            )
        mf = self.base._scf
        xctype = (mf._numint._xc_type(mf.xc)
                  if isinstance(mf, dft.KohnShamDFT) else 'HF')
        if xctype not in ('HF', 'LDA', 'GGA'):
            raise NotImplementedError(
                'SATDA analytical gradients with XC kernels currently support '
                'only LDA/GGA in the experimental deltaS=-1 path.'
            )
        make_satda_sf_blocks(self.base, xy)
        if xctype == 'HF':
            e_probe = _satda_hf_energy_for_orbs(
                self.base, xy, mf.mo_coeff, mf.mo_coeff
            )
            if abs(e_probe - self.base.e[self.state - 1]) > 1e-7:
                raise RuntimeError(
                    'Internal SATDA/HF orbital-RHS energy check failed: %.12g vs %.12g'
                    % (e_probe, self.base.e[self.state - 1])
                )
            de = grad_elec_hf_experimental(
                self, xy, atmlst=atmlst, max_memory=self.max_memory,
                verbose=self.verbose,
            )
            de += self.base._scf.nuc_grad_method().grad_nuc(atmlst=atmlst)
            return de

        de = grad_elec_hf_experimental(
            self, xy, atmlst=atmlst, max_memory=self.max_memory,
            verbose=self.verbose,
        )
        xy0 = (np.zeros_like(np.asarray(xy[0])), 0)
        de0 = grad_elec_hf_experimental(
            self, xy0, atmlst=atmlst, max_memory=self.max_memory,
            verbose=self.verbose,
        )
        g0 = self.base._scf.nuc_grad_method().set(
            verbose=0,
        ).kernel(atmlst=atmlst)
        return g0 + de - de0

    def kernel(self, state=None, atmlst=None, step=None, method=None):
        if state is not None:
            self.state = state
        if step is not None:
            self.step = step
        if method is not None:
            self.method = method
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
            atmlst = range(self.mol.natm)

        if self.state == 0:
            logger.warn(self, 'state=0 requested; returning ground-state gradient')
            return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

        if self.base.xy is None:
            self.base.run()
        if self.state < 1 or self.state > len(self.base.xy):
            raise ValueError('state must be in [1, %d]' % len(self.base.xy))

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        xy = self.base.xy[self.state - 1]
        if self.method == 'finite_diff':
            de = self._kernel_finite_diff(atmlst)
        elif self.method == 'analytic':
            de = self._kernel_analytic(xy, atmlst)
        elif self.method == 'analytic_experimental':
            de = self._kernel_analytic_experimental(xy, atmlst)
        else:
            raise ValueError('Unknown SATDA gradient method %s' % self.method)

        self.de = de
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '-------------- SATDA gradients '
                        'for state %d ----------', self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '--------------------------------------------')


Grad = Gradients

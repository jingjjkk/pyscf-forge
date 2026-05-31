"""LDA XC block-kernel contributions for SATDA ``deltaS=-1`` gradients."""

import numpy as np

from pyscf import lib
from pyscf.grad import tdrks as tdrks_grad

from ._blocks import _mo_pair_dm, _sasf_orbitals, make_sasf_blocks


SATDA_LDA_XC_GRAD_SCALE = 0.25


def _as_lda(tdobj):
    mf = tdobj._scf
    return hasattr(mf, '_numint') and mf._numint._xc_type(mf.xc) == 'LDA'


def _lda_block_matrix(si):
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


def _transition_blocks(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy)
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    blocks = (
        (b.osidx, b.csidx, _mo_pair_dm(orbos, b.x_co.T, orbcs), b.x_co.T),
        (b.vsidx, b.csidx, _mo_pair_dm(orbvs, b.x_cv.T, orbcs), b.x_cv.T),
        (b.osidx, b.osidx, _mo_pair_dm(orbos, b.x_oo.T, orbos), b.x_oo.T),
        (b.vsidx, b.osidx, _mo_pair_dm(orbvs, b.x_ov.T, orbos), b.x_ov.T),
    )
    return b, blocks


def lda_fxc_ref(tdobj, max_memory=2000):
    mf = tdobj._scf
    fxc = mf._numint.cache_xc_kernel(
        mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 1,
        max_memory=max_memory,
    )[2]
    return 0.5 * (
        fxc[0, :, 0] - fxc[0, :, 1]
        - fxc[1, :, 0] + fxc[1, :, 1]
    )


def lda_apply_fxc_ref(tdobj, dms, fxc_ref, max_memory=2000):
    mf = tdobj._scf
    dms = np.asarray(dms)
    if dms.ndim == 2:
        dms = dms.reshape(1, *dms.shape)
    return mf._numint.nr_rks_fxc(
        mf.mol, mf.grids, mf.xc, None, dms, 0, 0,
        None, None, fxc_ref, max_memory=max_memory,
    )


def lda_xc_energy(tdobj, xy, max_memory=2000):
    """LDA SATDA block-kernel scalar energy for fixed amplitudes."""
    if not _as_lda(tdobj):
        return 0.0

    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    b, blocks = _transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    mat = _lda_block_matrix(b.si)
    fxc_ref = np.asarray(lda_fxc_ref(tdobj, max_memory=max_memory))
    ngrids = fxc_ref.shape[-1]
    fxc_ref = fxc_ref.reshape(-1, ngrids)[0]

    value = 0.0
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, mol.nao_nr(), 1, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        ao0 = ao[0]
        rho_blocks = np.asarray([
            ni.eval_rho(mol, ao0, dm, mask, 'LDA', hermi=0,
                        with_lapl=False)
            for dm in dms
        ])
        value += lib.einsum(
            'bl,bg,lg,g,g->',
            mat, rho_blocks, rho_blocks, fxc_ref[p0:p1], weight,
        )
    return float(SATDA_LDA_XC_GRAD_SCALE * value)


def lda_ref_density_mats(tdobj, xy, with_deriv=False, max_memory=2000):
    """Reference-density derivative term from the LDA third derivative."""
    nao = tdobj.mol.nao_nr()
    if not _as_lda(tdobj):
        out = np.zeros((4, nao, nao)) if with_deriv else np.zeros((nao, nao))
        return out.copy(), out.copy()

    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    b, blocks = _transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    mat = _lda_block_matrix(b.si)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    vmat_a = np.zeros((4, nao, nao))
    vmat_b = np.zeros_like(vmat_a)

    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=max_memory):
        ao0 = ao[0]
        rho0 = ni.eval_rho2(
            mol, ao0, mf.mo_coeff, mf.mo_occ, mask, 'LDA',
            with_lapl=False,
        ) * 0.5
        rho = (rho0, rho0)
        kxc = ni.eval_xc_eff(
            mf.xc, rho, deriv=3, xctype='LDA', spin=1,
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
            ni.eval_rho(mol, ao0, dm, mask, 'LDA', hermi=0,
                        with_lapl=False)
            for dm in dms
        ])
        rho_pair = lib.einsum('bl,bg,lg->g', mat, rho_blocks, rho_blocks)
        rho_pair *= SATDA_LDA_XC_GRAD_SCALE
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


def lda_transition_deriv_mats(tdobj, xy, fxc_ref, max_memory=2000):
    """AO/grid derivative matrices for LDA block transition densities."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    b, blocks = _transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    mat = _lda_block_matrix(b.si)
    nao = mol.nao_nr()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    fxc_ref = np.asarray(fxc_ref)
    fxc_ref = fxc_ref.reshape(-1, fxc_ref.shape[-1])
    src = np.zeros((4, 4, nao, nao))

    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        ao0 = ao[0]
        fxc_blk = fxc_ref[:, p0:p1]
        for iblk, dm in enumerate(dms):
            rho = ni.eval_rho(
                mol, ao0, dm, mask, 'LDA', hermi=0, with_lapl=False
            )
            wv = fxc_blk * rho.reshape(1, -1) * weight
            tdrks_grad._lda_eval_mat_(
                mol, src[iblk], ao, wv, mask, shls_slice, ao_loc
            )

    src[:, 1:] *= -1
    out = np.zeros_like(src)
    for iblk in range(4):
        out[iblk] = (
            SATDA_LDA_XC_GRAD_SCALE
            * lib.einsum('l,lxpq->xpq', mat[iblk], src)
        )
    return out


def lda_xc_q(tdobj, xy, max_memory=2000):
    """Unconstrained MO derivative ``Q_alpha/Q_beta`` for LDA XC blocks."""
    mf = tdobj._scf
    nmo = mf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    if not _as_lda(tdobj):
        return q_alpha, q_beta

    b, blocks = _transition_blocks(tdobj, xy)
    mat = _lda_block_matrix(b.si)
    mo_coeff = mf.mo_coeff

    fxc_ref = lda_fxc_ref(tdobj, max_memory=max_memory)
    vsrc = lda_apply_fxc_ref(
        tdobj, [blk[2] for blk in blocks], fxc_ref,
        max_memory=max_memory,
    )
    vblocks = np.asarray([
        2.0 * SATDA_LDA_XC_GRAD_SCALE
        * lib.einsum('l,lpq->pq', mat[iblk], vsrc)
        for iblk in range(4)
    ])

    for vmat, (target_idx, source_idx, dm, amp_t_s) in zip(vblocks, blocks):
        vmo = mo_coeff.conj().T @ vmat @ mo_coeff
        q_beta[:, target_idx] += vmo[:, source_idx] @ amp_t_s.T
        q_alpha[:, source_idx] += vmo[:, target_idx] @ amp_t_s

    wa, wb = lda_ref_density_mats(
        tdobj, xy, with_deriv=False, max_memory=max_memory
    )
    occidxa = np.where(mf.mo_occ > 0)[0]
    occidxb = np.where(mf.mo_occ == 2)[0]
    q_alpha[:, occidxa] += mo_coeff.conj().T @ (wa + wa.T) @ mo_coeff[:, occidxa]
    q_beta[:, occidxb] += mo_coeff.conj().T @ (wb + wb.T) @ mo_coeff[:, occidxb]
    return q_alpha, q_beta


def lda_xc_m_matrix(tdobj, xy, max_memory=2000):
    q_alpha, q_beta = lda_xc_q(tdobj, xy, max_memory=max_memory)
    return q_alpha + q_beta


def lda_xc_direct_de(td_grad, tdobj, xy, atmlst=None, max_memory=2000):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    if not _as_lda(tdobj):
        return de

    mf = tdobj._scf
    mol = tdobj.mol
    b, blocks = _transition_blocks(tdobj, xy)
    dms = [blk[2] for blk in blocks]
    fxc_ref = lda_fxc_ref(tdobj, max_memory=max_memory)
    trans_der = lda_transition_deriv_mats(
        tdobj, xy, fxc_ref, max_memory=max_memory
    )
    wa_der, wb_der = lda_ref_density_mats(
        tdobj, xy, with_deriv=True, max_memory=max_memory
    )
    mo = mf.mo_coeff
    oo0a = mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
    oo0b = mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
    offsetdic = mol.offset_nr_by_atom()

    for k, ia in enumerate(atmlst):
        p0, p1 = offsetdic[ia][2:]
        for fder, dm in zip(trans_der[:, 1:], dms):
            de[k] += lib.einsum('xpq,pq->x', fder[:, p0:p1], dm[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', fder[:, p0:p1], dm.T[p0:p1]) * 2
        de[k] += lib.einsum('xpq,pq->x', wa_der[1:, p0:p1], oo0a[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wa_der[1:, p0:p1], oo0a.T[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wb_der[1:, p0:p1], oo0b[p0:p1])
        de[k] += lib.einsum('xpq,pq->x', wb_der[1:, p0:p1], oo0b.T[p0:p1])
    return de

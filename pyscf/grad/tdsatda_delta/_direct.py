import numpy as np

from pyscf import ao2mo
from pyscf import dft
from pyscf import lib

from ._blocks import (
    _sasf_orbitals,
    _hybrid_coefficients,
    _mo_pair_dm,
    make_sasf_blocks,
)
from ._fock_coeff import sasf_fock_probe_densities


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


def _general_eri(mol, orb_sets, omega=None):
    if omega is None or omega == 0:
        return ao2mo.general(mol, orb_sets, compact=False)
    with mol.with_range_coulomb(omega):
        return ao2mo.general(mol, orb_sets, compact=False)


def _full_jk_deriv_atom(mol, dm, ia, eri1=None):
    '''Full nuclear derivative of J/K matrices for a fixed AO density.'''
    nao = mol.nao
    p0, p1 = mol.offset_nr_by_atom()[ia][2:]
    if eri1 is None:
        eri1 = mol.intor('int2e_ip1', comp=3)
    sub = eri1[:, p0:p1]

    d_j = np.zeros((3, nao, nao))
    d_k = np.zeros((3, nao, nao))

    d_j[:, p0:p1, :] += lib.einsum('xajkl,lk->xaj', sub, dm)
    d_j[:, :, p0:p1] += lib.einsum('xailk,lk->xia', sub, dm)
    d_j += lib.einsum('xalij,la->xij', sub, dm[:, p0:p1])
    d_j += lib.einsum('xakji,ak->xij', sub, dm[p0:p1, :])

    d_k[:, p0:p1, :] += lib.einsum('xaklj,kl->xaj', sub, dm)
    d_k += lib.einsum('xaijl,al->xij', sub, dm[p0:p1, :])
    d_k += lib.einsum('xajik,ka->xij', sub, dm[:, p0:p1])
    d_k[:, :, p0:p1] += lib.einsum('xalki,kl->xia', sub, dm)
    return -d_j, -d_k


def _full_spin_fock_derivs_by_atom(td_grad, tdobj, ia, eri1=None):
    mol = tdobj.mol
    mf = tdobj._scf
    mo = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm_a = mo[:, mo_occ > 0] @ mo[:, mo_occ > 0].T
    dm_b = mo[:, mo_occ == 2] @ mo[:, mo_occ == 2].T
    h1 = mf.nuc_grad_method().hcore_generator(mol)(ia)
    j1a, k1a = _full_jk_deriv_atom(mol, dm_a, ia, eri1=eri1)
    j1b, k1b = _full_jk_deriv_atom(mol, dm_b, ia, eri1=eri1)
    if (isinstance(mf, dft.KohnShamDFT)
            and mf._numint._xc_type(mf.xc) != 'HF'):
        hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
        if omega != 0:
            raise NotImplementedError(
                'Range-separated LDA Fock derivatives are not implemented'
            )
        f1a = h1 + j1a + j1b
        f1b = h1 + j1a + j1b
        if hybrid:
            f1a -= hyb * k1a
            f1b -= hyb * k1b
        v1a, v1b = _full_lda_vxc_deriv_atom(
            mf, dm_a, dm_b, ia, max_memory=getattr(td_grad, 'max_memory', 2000)
        )
        f1a += v1a
        f1b += v1b
    else:
        f1a = h1 + j1a + j1b - k1a
        f1b = h1 + j1a + j1b - k1b
    return f1a, f1b


def _full_lda_vxc_deriv_atom(mf, dm_a, dm_b, ia, max_memory=2000):
    """Full nuclear derivative of LDA ``v_xc`` for fixed AO densities.

    The returned matrices follow the same full-matrix convention as
    :func:`_full_jk_deriv_atom`: all AO centers belonging to ``ia`` are
    differentiated and the sign is the nuclear derivative, not the electron
    coordinate derivative.
    """
    mol = mf.mol
    ni = mf._numint
    if ni._xc_type(mf.xc) != 'LDA':
        raise NotImplementedError('Only LDA vxc derivatives are implemented')

    nao = mol.nao_nr()
    p0, p1 = mol.offset_nr_by_atom()[ia][2:]
    v1 = np.zeros((2, 3, nao, nao))
    ao_loc = mol.ao_loc_nr()

    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, 1, max_memory=max_memory):
        ao0 = ao[0]
        rho_a = ni.eval_rho(
            mol, ao0, dm_a, mask, 'LDA', hermi=1, with_lapl=False
        )
        rho_b = ni.eval_rho(
            mol, ao0, dm_b, mask, 'LDA', hermi=1, with_lapl=False
        )
        vxc, fxc = ni.eval_xc_eff(
            mf.xc, (rho_a, rho_b), deriv=2, xctype='LDA', spin=1,
        )[1:3]

        for xyz in range(3):
            aoa = ao[xyz + 1][:, p0:p1]
            for ispin in range(2):
                wv = weight * vxc[ispin, 0]
                aow = ao0 * wv[:, None]
                v1[ispin, xyz, p0:p1] -= aoa.T @ aow
                v1[ispin, xyz, :, p0:p1] -= aow.T @ aoa

            drho_a = -_rho_deriv_atom_component(ao0, aoa, dm_a, p0, p1)
            drho_b = -_rho_deriv_atom_component(ao0, aoa, dm_b, p0, p1)
            for ispin in range(2):
                wv = weight * (
                    fxc[ispin, 0, 0, 0] * drho_a
                    + fxc[ispin, 0, 1, 0] * drho_b
                )
                aow = ao0 * wv[:, None]
                v1[ispin, xyz] += ao0.T @ aow

    return v1[0], v1[1]


def _rho_deriv_atom_component(ao0, aoa, dm, p0, p1):
    left = lib.einsum('gu,uv,gv->g', aoa, dm[p0:p1], ao0)
    right = lib.einsum('gu,uv,gv->g', ao0, dm[:, p0:p1], aoa)
    return left + right


def _full_alpha_fock_deriv_by_atom(td_grad, tdobj, ia, eri1=None):
    return _full_spin_fock_derivs_by_atom(
        td_grad, tdobj, ia, eri1=eri1
    )[0]


def _full_beta_fock_deriv_by_atom(td_grad, tdobj, ia, eri1=None):
    return _full_spin_fock_derivs_by_atom(
        td_grad, tdobj, ia, eri1=eri1
    )[1]


def sasf_delta_fock_direct_de(td_grad, tdobj, xy, atmlst=None):
    """Direct skeleton derivative of all SATDA Fock-like delta terms."""
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    dm_a, dm_b = sasf_fock_probe_densities(tdobj, xy)
    eri1 = tdobj.mol.intor('int2e_ip1', comp=3)
    for k, ia in enumerate(atmlst):
        f1a, f1b = _full_spin_fock_derivs_by_atom(
            td_grad, tdobj, ia, eri1=eri1
        )
        de[k] += lib.einsum('pq,xpq->x', dm_a, f1a)
        de[k] += lib.einsum('pq,xpq->x', dm_b, f1b)
    return de


def _add_j_bilinear_ip1(de, td_grad, mol, dm_l, dm_r, atmlst, offsetdic,
                        scale=1.0, omega=None):
    '''Add direct ERI derivative for ``scale * sum L[pq] R[tu] (pq|tu)``.'''
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


def _sasf_delta_hf_exchange_direct_with_coeff(
        de, td_grad, tdobj, xy, atmlst, offsetdic, coeff=1.0, omega=None):
    '''Direct AO ERI derivative for SASF HF exchange-like ``delta_A``.'''
    b = make_sasf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SASF spin adaptation requires Si > 1/2')

    mol = td_grad.mol
    _, _, _, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
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


def sasf_delta_hf_exchange_direct_de(td_grad, tdobj, xy, atmlst, offsetdic):
    '''Direct AO ERI derivative for all SASF HF exchange-like blocks.'''
    mf = tdobj._scf
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    de = np.zeros((len(tuple(atmlst)), 3))
    if not hybrid:
        return de

    _sasf_delta_hf_exchange_direct_with_coeff(
        de, td_grad, tdobj, xy, atmlst, offsetdic, coeff=hyb
    )
    if omega != 0:
        _sasf_delta_hf_exchange_direct_with_coeff(
            de, td_grad, tdobj, xy, atmlst, offsetdic,
            coeff=alpha - hyb, omega=omega,
        )
    return de

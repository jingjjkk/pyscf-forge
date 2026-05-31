from dataclasses import dataclass

import numpy as np

from pyscf import lib

from ._blocks import (
    _hybrid_coefficients,
    _mo_pair_dm,
    _sasf_orbitals,
    make_sasf_blocks,
    pack_roks_kappa,
    roks_spaces,
)
from ._fock_coeff import (
    sasf_fock_coefficients,
    sasf_fock_coefficient_energy,
)
from ._fock_basis import (
    fock_by_spin,
    make_fock_basis,
)
from ._exchange import sasf_hf_exchange_coefficient_energy
from ._q_rhs import (
    _add_eri_term_q,
    _add_fock_response_q,
    _add_fock_term,
    sasf_delta_q,
    sasf_delta_roks_rhs,
)
from ._direct import _full_jk_deriv_atom, _general_eri


@dataclass
class SASFBlockContribution:
    name: str
    kind: str
    energy: float
    q_alpha: np.ndarray
    q_beta: np.ndarray
    probe_alpha: object = None
    probe_beta: object = None

    def roks_rhs(self, tdobj):
        csidx, osidx, vsidx = roks_spaces(tdobj)
        r_alpha = self.q_alpha - self.q_alpha.T
        r_beta = self.q_beta - self.q_beta.T
        rco = r_alpha[np.ix_(osidx, csidx)] + r_beta[np.ix_(osidx, csidx)]
        rcv = r_alpha[np.ix_(vsidx, csidx)] + r_beta[np.ix_(vsidx, csidx)]
        rov = r_alpha[np.ix_(vsidx, osidx)] + r_beta[np.ix_(vsidx, osidx)]
        return pack_roks_kappa(rco, rcv, rov)


def _empty_q(tdobj):
    nmo = tdobj._scf.mo_coeff.shape[1]
    return np.zeros((nmo, nmo)), np.zeros((nmo, nmo))


def _empty_probe(tdobj):
    nao = tdobj.mol.nao
    return np.zeros((nao, nao)), np.zeros((nao, nao))


def _fock_block(tdobj, name, left_idx, right_idx, coeff_mat, spin,
                fock0_mo, fockz_mo, with_response=True):
    q_alpha, q_beta = _empty_q(tdobj)
    p_alpha, p_beta = _empty_probe(tdobj)
    mo_coeff = tdobj._scf.mo_coeff
    _add_fock_term(
        q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
        fock0_mo, fockz_mo, left_idx, right_idx, coeff_mat, spin,
    )
    if with_response:
        _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta)

    left_idx = np.asarray(left_idx)
    right_idx = np.asarray(right_idx)
    fblock = fock_by_spin(
        spin, fock0_mo, fockz_mo
    )[np.ix_(left_idx, right_idx)]
    energy = float(lib.einsum('pq,pq', coeff_mat, fblock))

    return SASFBlockContribution(
        name=name, kind='fock', energy=energy,
        q_alpha=q_alpha, q_beta=q_beta,
        probe_alpha=p_alpha, probe_beta=p_beta,
    )


def fock_like_blocks(tdobj, xy, with_response=True):
    coeff = sasf_fock_coefficients(tdobj, xy)
    csidx, osidx, vsidx, _, _, _ = _sasf_orbitals(tdobj)
    mo_coeff = tdobj._scf.mo_coeff
    fbasis = make_fock_basis(tdobj._scf, mo_coeff)
    return [
        _fock_block(tdobj, 'fock:S_CC', csidx, csidx, coeff.t_s_cc, 'spin',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
        _fock_block(tdobj, 'fock:S_VV', vsidx, vsidx, coeff.t_s_vv, 'spin',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
        _fock_block(tdobj, 'fock:S_CV', csidx, vsidx, coeff.t_s_cv, 'spin',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
        _fock_block(tdobj, 'fock:B_VO', vsidx, osidx, coeff.t_b_vo, 'beta',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
        _fock_block(tdobj, 'fock:B_CO', csidx, osidx, coeff.t_b_co, 'beta',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
        _fock_block(tdobj, 'fock:A_OC', osidx, csidx, coeff.t_a_oc, 'alpha',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
        _fock_block(tdobj, 'fock:A_VO', vsidx, osidx, coeff.t_a_vo, 'alpha',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
    ]


def cvcv_fock_blocks(tdobj, xy, with_response=True):
    '''Return the two Fock-like pieces that form the SASF CV-CV block.

    In ``sasf.get_a_sasf`` this block is

        A_CVCV = (delta_ij F_s(ab) + delta_ab F_s(ji)) / S_i

    so its Rayleigh quotient is represented by two spin-Fock contractions:
    ``T_cv_cc * F_s(CC)`` and ``T_cv_vv * F_s(VV)``.
    '''
    b = make_sasf_blocks(tdobj, xy)
    if b.si <= 0.5:
        raise NotImplementedError('SASF spin adaptation requires Si > 1/2')
    csidx, _, vsidx, _, _, _ = _sasf_orbitals(tdobj)
    mo_coeff = tdobj._scf.mo_coeff
    fbasis = make_fock_basis(tdobj._scf, mo_coeff)
    t_cc = lib.einsum('ia,ja->ji', b.x_cv, b.x_cv) / b.si
    t_vv = lib.einsum('ia,ib->ab', b.x_cv, b.x_cv) / b.si
    return [
        _fock_block(tdobj, 'fock:CVCV_CC', csidx, csidx, t_cc, 'spin',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
        _fock_block(tdobj, 'fock:CVCV_VV', vsidx, vsidx, t_vv, 'spin',
                    fbasis.fock0_mo, fbasis.fockz_mo, with_response=with_response),
    ]


def cvcv_block_energy(tdobj, xy):
    return sum(block.energy for block in cvcv_fock_blocks(
        tdobj, xy, with_response=False
    ))


def cvcv_probe_density(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy)
    if b.si <= 0.5:
        raise NotImplementedError('SASF spin adaptation requires Si > 1/2')
    _, _, _, orbcs, _, orbvs = _sasf_orbitals(tdobj)
    t_cc = lib.einsum('ia,ja->ji', b.x_cv, b.x_cv) / b.si
    t_vv = lib.einsum('ia,ib->ab', b.x_cv, b.x_cv) / b.si
    return _mo_pair_dm(orbcs, t_cc, orbcs) + _mo_pair_dm(orbvs, t_vv, orbvs)


def cvcv_open_density(tdobj):
    _, _, _, _, orbos, _ = _sasf_orbitals(tdobj)
    return orbos @ orbos.T


def cvcv_m_matrix(tdobj, xy):
    '''Unconstrained MO-basis coefficient matrix for the CV-CV block.'''
    mf = tdobj._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    p_ao = cvcv_probe_density(tdobj, xy)
    d_open = cvcv_open_density(tdobj)
    k_open = mf.get_k(mol, d_open, hermi=1)
    k_probe = mf.get_k(mol, p_ao, hermi=1)
    f_mo = mo_coeff.T @ k_open @ mo_coeff
    g_mo = mo_coeff.T @ k_probe @ mo_coeff
    nmo = mo_coeff.shape[1]
    p_mo = np.zeros((nmo, nmo))
    o_mo = np.zeros((nmo, nmo))
    b = make_sasf_blocks(tdobj, xy)
    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    p_mo[np.ix_(csidx, csidx)] = lib.einsum('ia,ja->ji', b.x_cv, b.x_cv) / b.si
    p_mo[np.ix_(vsidx, vsidx)] = lib.einsum('ia,ib->ab', b.x_cv, b.x_cv) / b.si
    o_mo[np.ix_(osidx, osidx)] = np.eye(len(osidx))
    return f_mo @ p_mo + g_mo @ o_mo


def _cvcv_m_matrix_q_based(tdobj, xy):
    '''Q-builder version of :func:`cvcv_m_matrix` for regression checks.'''
    q_alpha, q_beta = _empty_q(tdobj)
    for block in cvcv_fock_blocks(tdobj, xy, with_response=True):
        q_alpha += block.q_alpha
        q_beta += block.q_beta
    return q_alpha + q_beta


def cvcv_roks_rhs(tdobj, xy):
    '''Non-redundant CO/CV/OV RHS of the CV-CV block.

    This is locally correct as an orbital-rotation derivative, but it is not a
    complete gradient route for ``cvcv_block_fd_gradient`` because that FD
    benchmark uses canonical MO arrays and therefore includes redundant
    CC/OO/VV canonical-gauge rotations.
    '''
    m = cvcv_m_matrix(tdobj, xy)
    csidx, osidx, vsidx = roks_spaces(tdobj)
    r = m - m.T
    return pack_roks_kappa(
        r[np.ix_(osidx, csidx)],
        r[np.ix_(vsidx, csidx)],
        r[np.ix_(vsidx, osidx)],
    )


def _add_k_bilinear_ip1(de, td_grad, mol, dm_l, dm_r, atmlst, offsetdic,
                        scale=1.0):
    if scale == 0:
        return
    vk_r = td_grad.get_k(mol, dm_r, hermi=0)
    vk_rt = td_grad.get_k(mol, dm_r.T, hermi=0)
    vk_l = td_grad.get_k(mol, dm_l, hermi=0)
    vk_lt = td_grad.get_k(mol, dm_l.T, hermi=0)
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        term = lib.einsum('xpq,pq->x', vk_r[:, p0:p1], dm_l[p0:p1])
        term += lib.einsum('xqp,pq->x', vk_rt[:, p0:p1], dm_l[:, p0:p1])
        term += lib.einsum('xpq,pq->x', vk_l[:, p0:p1], dm_r[p0:p1])
        term += lib.einsum('xqp,pq->x', vk_lt[:, p0:p1], dm_r[:, p0:p1])
        de[k] += scale * term


def cvcv_direct_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    _add_k_bilinear_ip1(
        de, td_grad, tdobj.mol,
        cvcv_probe_density(tdobj, xy), cvcv_open_density(tdobj),
        atmlst, tdobj.mol.offset_nr_by_atom(), scale=0.5,
    )
    return de


def _canonical_roks_pairs(tdobj):
    csidx, osidx, vsidx = roks_spaces(tdobj)
    pairs = []
    for idx, name in ((csidx, 'cc'), (osidx, 'oo'), (vsidx, 'vv')):
        for p in range(1, len(idx)):
            for q in range(p):
                pairs.append((idx[p], idx[q], name))
    for o in osidx:
        for c in csidx:
            pairs.append((o, c, 'co'))
    for v in vsidx:
        for c in csidx:
            pairs.append((v, c, 'cv'))
    for v in vsidx:
        for o in osidx:
            pairs.append((v, o, 'ov'))
    return pairs


def _pack_roks_canonical_residual(pairs, fmo_a, fmo_b):
    fmo_c = 0.5 * (fmo_a + fmo_b)
    out = []
    for p, q, name in pairs:
        if name in ('cc', 'oo', 'vv'):
            out.append(fmo_c[p, q])
        elif name == 'co':
            out.append(fmo_b[p, q])
        elif name == 'cv':
            out.append(fmo_a[p, q] + fmo_b[p, q])
        elif name == 'ov':
            out.append(fmo_a[p, q])
        else:
            raise RuntimeError('Unknown ROKS canonical pair type %s' % name)
    return np.asarray(out)


def _anti_mo_from_roks_canonical_vec(nmo, pairs, vec):
    kappa = np.zeros((nmo, nmo))
    for val, (p, q, name) in zip(vec, pairs):
        kappa[p, q] = val
        kappa[q, p] = -val
    return kappa


def _roks_general_orbital_action_hf(tdobj, pairs, kappa):
    mf = tdobj._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm_a = mo_coeff[:, mo_occ > 0] @ mo_coeff[:, mo_occ > 0].T
    dm_b = mo_coeff[:, mo_occ == 2] @ mo_coeff[:, mo_occ == 2].T
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(mol, (dm_a, dm_b), hermi=1)
    fock_a = hcore + vj[0] + vj[1] - vk[0]
    fock_b = hcore + vj[0] + vj[1] - vk[1]
    fmo_a = mo_coeff.T @ fock_a @ mo_coeff
    fmo_b = mo_coeff.T @ fock_b @ mo_coeff

    occ_a = np.zeros_like(mo_occ, dtype=float)
    occ_b = np.zeros_like(mo_occ, dtype=float)
    occ_a[mo_occ > 0] = 1.0
    occ_b[mo_occ == 2] = 1.0
    n_a = np.diag(occ_a)
    n_b = np.diag(occ_b)
    ddm_a = mo_coeff @ (kappa @ n_a + n_a @ kappa.T) @ mo_coeff.T
    ddm_b = mo_coeff @ (kappa @ n_b + n_b @ kappa.T) @ mo_coeff.T
    vj1, vk1 = mf.get_jk(mol, (ddm_a, ddm_b), hermi=1)
    vfock_a = vj1[0] + vj1[1] - vk1[0]
    vfock_b = vj1[0] + vj1[1] - vk1[1]

    dfmo_a = kappa.T @ fmo_a + fmo_a @ kappa
    dfmo_a += mo_coeff.T @ vfock_a @ mo_coeff
    dfmo_b = kappa.T @ fmo_b + fmo_b @ kappa
    dfmo_b += mo_coeff.T @ vfock_b @ mo_coeff
    return _pack_roks_canonical_residual(pairs, dfmo_a, dfmo_b)


def _roks_canonical_response_kappas_hf(td_grad, tdobj, atmlst=None):
    '''Solve dense first-order canonical ROKS orbital response matrices.

    The returned matrices are raw MO coefficient derivatives
    ``C^[x] = C K^[x]`` in the moving AO-array convention.  They contain the
    symmetric overlap response, the non-redundant CO/CV/OV response, and the
    redundant CC/OO/VV canonical-gauge rotations.
    '''
    mol = tdobj.mol
    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = mo_coeff.shape[1]
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)

    pairs = _canonical_roks_pairs(tdobj)
    nvar = len(pairs)
    eye = np.eye(nvar)
    hmat = np.column_stack([
        _roks_general_orbital_action_hf(
            tdobj, pairs,
            _anti_mo_from_roks_canonical_vec(nmo, pairs, eye[i]),
        )
        for i in range(nvar)
    ])

    hcore_deriv = mf.nuc_grad_method().hcore_generator(mol)
    s1 = mf.nuc_grad_method().get_ovlp(mol)
    dm_a = mo_coeff[:, mo_occ > 0] @ mo_coeff[:, mo_occ > 0].T
    dm_b = mo_coeff[:, mo_occ == 2] @ mo_coeff[:, mo_occ == 2].T
    offsetdic = mol.offset_nr_by_atom()
    eri1 = mol.intor('int2e_ip1', comp=3)

    kappas = {}
    for ia in atmlst:
        p0, p1 = offsetdic[ia][2:]
        j1a, k1a = _full_jk_deriv_atom(mol, dm_a, ia, eri1=eri1)
        j1b, k1b = _full_jk_deriv_atom(mol, dm_b, ia, eri1=eri1)
        katom = np.zeros((3, nmo, nmo))
        for xyz in range(3):
            dfock_a = hcore_deriv(ia)[xyz] + j1a[xyz] + j1b[xyz] - k1a[xyz]
            dfock_b = hcore_deriv(ia)[xyz] + j1a[xyz] + j1b[xyz] - k1b[xyz]
            gfix = _pack_roks_canonical_residual(
                pairs,
                mo_coeff.T @ dfock_a @ mo_coeff,
                mo_coeff.T @ dfock_b @ mo_coeff,
            )
            s1ao = np.zeros((mol.nao, mol.nao))
            s1ao[p0:p1] += s1[xyz, p0:p1]
            s1ao[:, p0:p1] += s1[xyz, p0:p1].T
            ksym = -0.5 * (mo_coeff.T @ s1ao @ mo_coeff)
            rhs = -(gfix + _roks_general_orbital_action_hf(
                tdobj, pairs, ksym
            ))
            avec = np.linalg.solve(hmat, rhs)
            katom[xyz] = (
                _anti_mo_from_roks_canonical_vec(nmo, pairs, avec) + ksym
            )
        kappas[ia] = katom
    return kappas


def cvcv_canonical_response_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = cvcv_m_matrix(tdobj, xy)
    kappas = _roks_canonical_response_kappas_hf(
        td_grad, tdobj, atmlst=atmlst
    )
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        de[k] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


def cvcv_analytic_grad(td_grad, tdobj, xy, atmlst=None, verbose=0):
    direct = cvcv_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    orbital, kappas = cvcv_canonical_response_grad(
        td_grad, tdobj, xy, atmlst=atmlst
    )
    return direct + orbital, {
        'direct': direct,
        'orbital': orbital,
        'kappas': kappas,
    }


def cvcv_block_matrix(tdobj):
    '''Materialize only the SASF CV-CV correction block.

    This helper is for small diagnostics.  It mirrors the corresponding
    assignment in ``pyscf.sftda.sasf.get_a_sasf`` without materializing the
    other SASF correction blocks.
    '''
    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    ncs, nos, nvs = len(csidx), len(osidx), len(vsidx)
    si = (mf.mol.nelec[0] - mf.mol.nelec[1]) * 0.5
    orbcs = mo_coeff[:, csidx]
    orbvs = mo_coeff[:, vsidx]
    fbasis = make_fock_basis(mf)
    focksc = -orbcs.T @ fbasis.fockz @ orbcs
    focksv = -orbvs.T @ fbasis.fockz @ orbvs
    a = np.zeros((ncs + nos, nos + nvs, ncs + nos, nos + nvs))
    a_cvcv = (
        lib.einsum('ij,ab->iajb', np.eye(ncs), focksv) +
        lib.einsum('ab,ji->iajb', np.eye(nvs), focksc)
    ) / si
    a[:ncs, nos:, :ncs, nos:] = a_cvcv
    return a


def cvcv_block_matrix_energy(tdobj, xy):
    x = np.asarray(xy[0])
    a_cvcv = cvcv_block_matrix(tdobj)
    return float(lib.einsum('ia,iajb,jb', x, a_cvcv, x))


def cvcv_block_fd_gradient(td_grad, xy, atmlst=None, step=2e-4):
    '''Central finite difference of the fixed-amplitude CV-CV block energy.

    The displaced geometries rerun the HF ROKS reference but reuse the input
    amplitude array in the displaced C/O/V slots.  This is a diagnostic for
    ``X0.T @ A_CVCV^[x] @ X0`` and is not the derivative of an independent
    eigenvalue.
    '''
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
        return cvcv_block_energy(td, xy)

    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            coords_p = coords0.copy()
            coords_m = coords0.copy()
            coords_p[ia, xyz] += step
            coords_m[ia, xyz] -= step
            de[k, xyz] = (energy_at(coords_p) - energy_at(coords_m)) / (2 * step)
    return de


def _general(mol, orb_sets, omega=None):
    return _general_eri(mol, orb_sets, omega=omega)


def _eri_block(tdobj, name, orb_sets, idx_sets, spin_sets, tensor, scale,
               omega=None):
    q_alpha, q_beta = _empty_q(tdobj)
    _add_eri_term_q(
        tdobj, q_alpha, q_beta, orb_sets, idx_sets, spin_sets,
        tensor, scale=scale, omega=omega,
    )
    eri = _general(tdobj.mol, orb_sets, omega=omega).reshape(tensor.shape)
    energy = float(scale * lib.einsum('pqrs,pqrs', tensor, eri))
    return SASFBlockContribution(
        name=name, kind='hfx', energy=energy,
        q_alpha=q_alpha, q_beta=q_beta,
    )


def _hfx_blocks_with_coeff(tdobj, xy, coeff=1.0, omega=None, suffix=''):
    b = make_sasf_blocks(tdobj, xy)
    if b.si <= 0.5:
        raise NotImplementedError('SASF spin adaptation requires Si > 1/2')
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    eta = np.sqrt((2 * b.si + 1) / (2 * b.si)) - 1
    gamma = np.sqrt((2 * b.si + 1) / (2 * b.si - 1))
    zeta = np.sqrt(2 * b.si / (2 * b.si - 1)) - 1

    blocks = []

    def add(name, orb_sets, idx_sets, spin_sets, tensor, scale):
        blocks.append(_eri_block(
            tdobj, name + suffix, orb_sets, idx_sets, spin_sets,
            tensor, coeff * scale, omega=omega,
        ))

    add('hfx:COCO',
        (orbos, orbcs, orbcs, orbos),
        (osidx, csidx, csidx, osidx),
        ('beta', 'alpha', 'alpha', 'beta'),
        lib.einsum('iu,jv->uijv', b.x_co, b.x_co),
        -1.0 / (2 * b.si - 1))
    add('hfx:OVOV',
        (orbvs, orbos, orbos, orbvs),
        (vsidx, osidx, osidx, vsidx),
        ('beta', 'beta', 'beta', 'beta'),
        lib.einsum('ua,vb->auvb', b.x_ov, b.x_ov),
        -1.0 / (2 * b.si - 1))
    add('hfx:CVCO',
        (orbvs, orbos, orbcs, orbcs),
        (vsidx, osidx, csidx, csidx),
        ('beta', 'beta', 'alpha', 'alpha'),
        lib.einsum('ia,jv->avji', b.x_cv, b.x_co),
        -2 * eta)
    add('hfx:CVOV',
        (orbvs, orbvs, orbos, orbcs),
        (vsidx, vsidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('ia,vb->abvi', b.x_cv, b.x_ov),
        -2 * eta)
    add('hfx:COOV_1',
        (orbos, orbcs, orbos, orbvs),
        (osidx, csidx, osidx, vsidx),
        ('beta', 'alpha', 'beta', 'beta'),
        lib.einsum('iu,vb->uivb', b.x_co, b.x_ov),
        2.0 / (2 * b.si - 1))
    add('hfx:COOV_2',
        (orbos, orbvs, orbos, orbcs),
        (osidx, vsidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('iu,vb->ubvi', b.x_co, b.x_ov),
        -2.0 / (2 * b.si - 1))
    add('hfx:CVOO',
        (orbvs, orbos, orbos, orbcs),
        (vsidx, osidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('ia,wv->avwi', b.x_cv, b.x_oo),
        -2 * (gamma - 1))
    add('hfx:COOO',
        (orbos, orbos, orbos, orbcs),
        (osidx, osidx, osidx, csidx),
        ('beta', 'beta', 'beta', 'alpha'),
        lib.einsum('iu,wv->uvwi', b.x_co, b.x_oo),
        -2 * zeta)
    add('hfx:OVOO',
        (orbvs, orbos, orbos, orbos),
        (vsidx, osidx, osidx, osidx),
        ('beta', 'beta', 'beta', 'beta'),
        lib.einsum('ua,wv->avwu', b.x_ov, b.x_oo),
        -2 * zeta)
    return blocks


def hf_exchange_like_blocks(tdobj, xy):
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if not hybrid:
        return []
    blocks = _hfx_blocks_with_coeff(tdobj, xy, coeff=hyb)
    if omega != 0:
        blocks.extend(_hfx_blocks_with_coeff(
            tdobj, xy, coeff=alpha - hyb, omega=omega, suffix=':rsh'
        ))
    return blocks


def sasf_hf_block_contributions(tdobj, xy, fock_response=True):
    return (
        fock_like_blocks(tdobj, xy, with_response=fock_response) +
        hf_exchange_like_blocks(tdobj, xy)
    )


def sum_block_q(blocks, tdobj):
    q_alpha, q_beta = _empty_q(tdobj)
    for block in blocks:
        q_alpha += block.q_alpha
        q_beta += block.q_beta
    return q_alpha, q_beta


def sum_block_roks_rhs(blocks, tdobj):
    rhs = None
    for block in blocks:
        rb = block.roks_rhs(tdobj)
        rhs = rb.copy() if rhs is None else rhs + rb
    if rhs is None:
        dims = roks_spaces(tdobj)
        return np.zeros(len(pack_roks_kappa(
            np.zeros((len(dims[1]), len(dims[0]))),
            np.zeros((len(dims[2]), len(dims[0]))),
            np.zeros((len(dims[2]), len(dims[1]))),
        )))
    return rhs


def block_consistency_report(tdobj, xy, fock_response=True):
    blocks = sasf_hf_block_contributions(
        tdobj, xy, fock_response=fock_response
    )
    e_fock_blocks = sum(b.energy for b in blocks if b.kind == 'fock')
    e_hfx_blocks = sum(b.energy for b in blocks if b.kind == 'hfx')
    q_alpha, q_beta = sum_block_q(blocks, tdobj)
    q_ref_a, q_ref_b = sasf_delta_q(
        tdobj, xy, include_fock=True, include_hf=True,
        fock_response=fock_response,
    )
    rhs_blocks = sum_block_roks_rhs(blocks, tdobj)
    rhs_ref = sasf_delta_roks_rhs(
        tdobj, xy, include_fock=True, include_hf=True,
        fock_response=fock_response,
    )
    return {
        'blocks': blocks,
        'energy_fock_blocks': e_fock_blocks,
        'energy_hfx_blocks': e_hfx_blocks,
        'energy_fock_ref': sasf_fock_coefficient_energy(tdobj, xy),
        'energy_hfx_ref': sasf_hf_exchange_coefficient_energy(tdobj, xy),
        'q_alpha_maxdiff': np.max(np.abs(q_alpha - q_ref_a)),
        'q_beta_maxdiff': np.max(np.abs(q_beta - q_ref_b)),
        'rhs_roks_maxdiff': np.max(np.abs(rhs_blocks - rhs_ref)),
    }

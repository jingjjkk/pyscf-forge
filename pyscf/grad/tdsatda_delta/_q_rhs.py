import numpy as np

from pyscf import ao2mo
from pyscf import lib

from ._blocks import (
    _sasf_orbitals,
    _hybrid_coefficients,
    _mo_pair_dm,
    make_sasf_blocks,
    roks_spaces,
    pack_roks_kappa,
)
from ._fock_coeff import sasf_fock_coefficients
from ._fock_basis import (
    alpha_from_0z,
    beta_from_0z,
    make_fock_basis,
)


def _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fock0_mo, fockz_mo, left_idx, right_idx, coeff_mat, spin):
    '''Accumulate projection and probe-density pieces for one Fock term.'''
    coeff_mat = np.asarray(coeff_mat)
    if coeff_mat.size == 0:
        return

    left_idx = np.asarray(left_idx)
    right_idx = np.asarray(right_idx)
    c_left = mo_coeff[:, left_idx]
    c_right = mo_coeff[:, right_idx]
    dm = _mo_pair_dm(c_left, coeff_mat, c_right)
    fock_alpha_mo = alpha_from_0z(fock0_mo, fockz_mo)
    fock_beta_mo = beta_from_0z(fock0_mo, fockz_mo)

    def add_projection(q, fock_mo, scale):
        if scale == 0:
            return
        q[:, left_idx] += scale * (fock_mo[:, right_idx] @ coeff_mat.T)
        q[:, right_idx] += scale * (fock_mo[:, left_idx] @ coeff_mat)

    if spin == 'alpha':
        add_projection(q_alpha, fock_alpha_mo, 1.0)
        p_alpha += dm
    elif spin == 'beta':
        add_projection(q_beta, fock_beta_mo, 1.0)
        p_beta += dm
    elif spin == 'spin':
        add_projection(q_beta, fock_beta_mo, 0.5)
        add_projection(q_alpha, fock_alpha_mo, -0.5)
        p_beta += 0.5 * dm
        p_alpha -= 0.5 * dm
    else:
        raise ValueError('Unknown Fock term spin label %s' % spin)


def _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta):
    '''Add HF Fock-density response contribution to ``Q_alpha/Q_beta``.'''
    mf = tdobj._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ > 0)[0]
    occidxb = np.where(mo_occ == 2)[0]

    p_tot = p_alpha + p_beta
    vj = mf.get_j(mol, p_tot.T, hermi=0)
    vk_a = mf.get_k(mol, p_alpha.T, hermi=0)
    vk_b = mf.get_k(mol, p_beta.T, hermi=0)
    va = vj - vk_a
    vb = vj - vk_b
    q_alpha[:, occidxa] += mo_coeff.conj().T @ (va + va.T) @ mo_coeff[:, occidxa]
    q_beta[:, occidxb] += mo_coeff.conj().T @ (vb + vb.T) @ mo_coeff[:, occidxb]


def sasf_delta_fock_q(tdobj, xy, with_response=True):
    '''Unconstrained MO coefficient derivative Q for SASF Fock-like terms.'''
    mf = tdobj._scf
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    p_alpha = np.zeros((mf.mol.nao, mf.mol.nao))
    p_beta = np.zeros_like(p_alpha)

    coeff = sasf_fock_coefficients(tdobj, xy)
    csidx, osidx, vsidx, _, _, _ = _sasf_orbitals(tdobj)
    fbasis = make_fock_basis(mf, mo_coeff)

    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fbasis.fock0_mo, fbasis.fockz_mo, csidx, csidx,
                   coeff.t_s_cc, 'spin')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fbasis.fock0_mo, fbasis.fockz_mo, vsidx, vsidx,
                   coeff.t_s_vv, 'spin')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fbasis.fock0_mo, fbasis.fockz_mo, csidx, vsidx,
                   coeff.t_s_cv, 'spin')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fbasis.fock0_mo, fbasis.fockz_mo, vsidx, osidx,
                   coeff.t_b_vo, 'beta')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fbasis.fock0_mo, fbasis.fockz_mo, csidx, osidx,
                   coeff.t_b_co, 'beta')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fbasis.fock0_mo, fbasis.fockz_mo, osidx, csidx,
                   coeff.t_a_oc, 'alpha')
    _add_fock_term(q_alpha, q_beta, p_alpha, p_beta, mo_coeff,
                   fbasis.fock0_mo, fbasis.fockz_mo, vsidx, osidx,
                   coeff.t_a_vo, 'alpha')

    if with_response:
        _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta)
    return q_alpha, q_beta


def _add_eri_term_q(tdobj, q_alpha, q_beta, orb_sets, idx_sets,
                    spin_sets, coeff_tensor, scale=1.0, omega=None):
    '''Accumulate projection RHS for one four-index ERI contraction.'''
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


def _sasf_delta_hf_exchange_q_with_coeff(tdobj, xy, coeff=1.0, omega=None):
    mf = tdobj._scf
    b = make_sasf_blocks(tdobj, xy)
    si = b.si
    if si <= 0.5:
        raise NotImplementedError('SASF spin adaptation requires Si > 1/2')

    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
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


def sasf_delta_hf_exchange_q(tdobj, xy):
    '''Unconstrained MO coefficient derivative Q for SASF HF-like terms.'''
    mf = tdobj._scf
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    nmo = mf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    if not hybrid:
        return q_alpha, q_beta

    qa, qb = _sasf_delta_hf_exchange_q_with_coeff(tdobj, xy, hyb)
    q_alpha += qa
    q_beta += qb
    if omega != 0:
        qa, qb = _sasf_delta_hf_exchange_q_with_coeff(
            tdobj, xy, alpha - hyb, omega=omega
        )
        q_alpha += qa
        q_beta += qb
    return q_alpha, q_beta


def sasf_delta_q(tdobj, xy, include_fock=True, include_hf=True,
                 fock_response=True):
    '''Full unconstrained MO derivative Q for SASF ``delta_A`` terms.'''
    mf = tdobj._scf
    nmo = mf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)

    if include_fock:
        qa, qb = sasf_delta_fock_q(tdobj, xy, with_response=fock_response)
        q_alpha += qa
        q_beta += qb
    if include_hf:
        qa, qb = sasf_delta_hf_exchange_q(tdobj, xy)
        q_alpha += qa
        q_beta += qb
    return q_alpha, q_beta


def sasf_delta_rhs(tdobj, xy, include_fock=True, include_hf=True,
                   fock_response=True):
    '''Return full SASF delta_A orbital derivative in VO layout.

    PySCF TD-gradient Z-vector equations use half of this antisymmetrized
    orbital derivative as the ``ucphf.solve`` RHS.
    '''
    mf = tdobj._scf
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ > 0)[0]
    occidxb = np.where(mo_occ == 2)[0]
    viridxa = np.where(mo_occ == 0)[0]
    viridxb = np.where(mo_occ < 2)[0]
    q_alpha, q_beta = sasf_delta_q(
        tdobj, xy, include_fock=include_fock, include_hf=include_hf,
        fock_response=fock_response,
    )

    r_alpha = q_alpha - q_alpha.T
    r_beta = q_beta - q_beta.T
    return (
        r_alpha[np.ix_(viridxa, occidxa)],
        r_beta[np.ix_(viridxb, occidxb)],
    )


def sasf_delta_roks_rhs(tdobj, xy, include_fock=True, include_hf=True,
                        fock_response=True, fock_scale=1.0,
                        hf_scale=1.0):
    '''SASF delta_A RHS in the physical ROKS spatial rotation variables.

    In the one-kappa ROKS variable space the RHS uses ``q - q.T`` directly.
    The ``0.5 * (q - q.T)`` convention belongs to the doubled UKS variable
    space and must not be used here.
    '''
    csidx, osidx, vsidx = roks_spaces(tdobj)
    nmo = tdobj._scf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    if include_fock:
        qa, qb = sasf_delta_q(
            tdobj, xy, include_fock=True, include_hf=False,
            fock_response=fock_response,
        )
        q_alpha += fock_scale * qa
        q_beta += fock_scale * qb
    if include_hf:
        qa, qb = sasf_delta_q(
            tdobj, xy, include_fock=False, include_hf=True,
            fock_response=fock_response,
        )
        q_alpha += hf_scale * qa
        q_beta += hf_scale * qb

    r_alpha = q_alpha - q_alpha.T
    r_beta = q_beta - q_beta.T
    rco = r_alpha[np.ix_(osidx, csidx)] + r_beta[np.ix_(osidx, csidx)]
    rcv = r_alpha[np.ix_(vsidx, csidx)] + r_beta[np.ix_(vsidx, csidx)]
    rov = r_alpha[np.ix_(vsidx, osidx)] + r_beta[np.ix_(vsidx, osidx)]
    return pack_roks_kappa(rco, rcv, rov)

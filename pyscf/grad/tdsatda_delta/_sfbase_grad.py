"""Fixed-amplitude ordinary SF-TDA block ledger for SATDA gradients.

This module defines the non-delta part

    X^T A_SF X

for a SATDA ``deltaS=-1`` amplitude ``X``.  The scalar reference is the
ordinary spin-flip TDA action evaluated on the UKS view of the ROKS reference.
No ``tduks_sf`` gradient assembly is used here.
"""

import numpy as np

from pyscf import lib
from pyscf.sftda.uks_sf import TDA_SF

from ._blocks import make_sasf_blocks
from ._blocks import _hybrid_coefficients
from ._blocks import _mo_pair_dm
from ._blocks import _sasf_orbitals
from ._fd import _make_displaced_mf
from ._direct import _add_j_bilinear_ip1
from ._direct import _full_spin_fock_derivs_by_atom
from ._direct import _general_eri
from ._fock_basis import make_fock_basis
from ._q_rhs import _add_eri_term_q
from ._q_rhs import _add_fock_response_q
from ._q_rhs import _add_fock_term
from ._block_analytic_hf import _roks_canonical_response_kappas_hf
from ._zvec_solver import (
    build_roks_hessian,
    make_roks_hessian_transpose_action,
    pack_mvec,
    solve_zvec,
    solve_zvec_krylov,
    zvec_orbital_grad,
)


_SFBASE_BLOCK_NAMES = (
    'CO', 'CV', 'OO', 'OV',
)


def _copy_td_settings(src, dst):
    dst.deltaS = src.deltaS
    dst.nstates = src.nstates
    dst.conv_tol = src.conv_tol
    dst.lindep = src.lindep
    dst.max_cycle = src.max_cycle
    dst.max_memory = src.max_memory
    dst.verbose = 0
    return dst


def _displaced_td_like(tdobj, coords_bohr):
    from pyscf.sftda.satda import SATDA

    mol = tdobj.mol.copy()
    mol.set_geom_(coords_bohr, unit='Bohr')
    mf = _make_displaced_mf(tdobj._scf, mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('Displaced ROKS/ROHF reference did not converge')
    return _copy_td_settings(tdobj, SATDA(mf).set(deltaS=tdobj.deltaS))


def _zero_block_like(blocks):
    return {
        'CO': np.zeros_like(blocks.x_co),
        'CV': np.zeros_like(blocks.x_cv),
        'OO': np.zeros_like(blocks.x_oo),
        'OV': np.zeros_like(blocks.x_ov),
    }


def split_sfbase_blocks(tdobj, xy):
    """Return ``CO/CV/OO/OV`` views of the SATDA ``deltaS=-1`` amplitude."""
    b = make_sasf_blocks(tdobj, xy)
    return {
        'CO': b.x_co,
        'CV': b.x_cv,
        'OO': b.x_oo,
        'OV': b.x_ov,
    }


def _ordinary_sf_action(tdobj, x):
    """Return ``A_SF x`` for ordinary SF-TDA on the UKS view of the ROKS ref."""
    mf_uks = tdobj._scf.to_uks()
    sf = TDA_SF(mf_uks, extype=1, collinear_samples=-1).set(
        nstates=1, verbose=0,
    )
    vind, _hdiag = sf.gen_vind()
    ax = vind(np.asarray(x).reshape(1, -1)).reshape(np.asarray(x).shape)
    return ax


def satda_sfbase_action(tdobj, xy):
    """Return the ordinary SF-TDA action ``A_SF X`` in SATDA block shape."""
    return _ordinary_sf_action(tdobj, np.asarray(xy[0]))


def satda_sfbase_energy(tdobj, xy):
    """Fixed-amplitude ordinary SF-base scalar ``X^T A_SF X``."""
    x = np.asarray(xy[0])
    ax = satda_sfbase_action(tdobj, xy)
    return float(np.vdot(x, ax).real)


def satda_sfbase_block_energy(tdobj, xy, left, right):
    """Fixed-amplitude block scalar ``X_left^T A_SF(left,right) X_right``."""
    left = left.upper()
    right = right.upper()
    if left not in _SFBASE_BLOCK_NAMES or right not in _SFBASE_BLOCK_NAMES:
        raise ValueError('Unknown SF-base block %s-%s' % (left, right))

    blocks = split_sfbase_blocks(tdobj, xy)
    xr_blocks = _zero_block_like(make_sasf_blocks(tdobj, xy))
    xr_blocks[right] = blocks[right]
    xr = np.block([
        [xr_blocks['CO'], xr_blocks['CV']],
        [xr_blocks['OO'], xr_blocks['OV']],
    ])
    axr = _ordinary_sf_action(tdobj, xr)
    ax_blocks = split_sfbase_blocks(tdobj, (axr, 0))
    return float(np.vdot(blocks[left], ax_blocks[left]).real)


def satda_sfbase_block_energies(tdobj, xy):
    """Return the full 16-block ordinary SF-base energy ledger."""
    return {
        '%s-%s' % (left, right): satda_sfbase_block_energy(
            tdobj, xy, left, right
        )
        for left in _SFBASE_BLOCK_NAMES
        for right in _SFBASE_BLOCK_NAMES
    }


def satda_sfbase_block_sum_energy(tdobj, xy):
    """Return the sum of all 16 ordinary SF-base block energies."""
    return float(sum(satda_sfbase_block_energies(tdobj, xy).values()))


def satda_sfbase_finite_diff(tdobj, xy, atmlst=None, step=2e-4,
                             left=None, right=None):
    """Relaxed-SCF, fixed-amplitude FD for ordinary SF-base energy.

    If ``left`` and ``right`` are supplied, finite-difference one block.
    Otherwise finite-difference the full 16-block sum.
    """
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    coords0 = tdobj.mol.atom_coords()
    de = np.zeros((len(atmlst), 3))

    def energy_at(coords):
        td1 = _displaced_td_like(tdobj, coords)
        if left is None and right is None:
            return satda_sfbase_block_sum_energy(td1, xy)
        if left is None or right is None:
            raise ValueError('left and right must be provided together')
        return satda_sfbase_block_energy(td1, xy, left, right)

    for k, ia in enumerate(atmlst):
        for xyz in range(3):
            coords_p = coords0.copy()
            coords_m = coords0.copy()
            coords_p[ia, xyz] += step
            coords_m[ia, xyz] -= step
            de[k, xyz] = (energy_at(coords_p) - energy_at(coords_m)) / (2 * step)
    return de


def satda_sfbase_ledger_report(tdobj, xy):
    """Return consistency diagnostics for the ordinary SF-base energy ledger."""
    full = satda_sfbase_energy(tdobj, xy)
    parts = satda_sfbase_block_energies(tdobj, xy)
    block_sum = float(sum(parts.values()))
    return {
        'energy_action': full,
        'energy_block_sum': block_sum,
        'energy_diff': block_sum - full,
        'block_energies': parts,
    }


def _sfbase_check_si(si):
    if si <= 0.5:
        raise NotImplementedError('SATDA deltaS=-1 requires Si > 1/2')


def _sfbase_block_data(tdobj, xy, name):
    name = name.upper()
    blocks = split_sfbase_blocks(tdobj, xy)
    if name not in blocks:
        raise ValueError('Unknown SF-base block %s' % name)
    csidx, osidx, vsidx, orbcs, orbos, orbvs = _sasf_orbitals(tdobj)
    row_key, col_key = name
    idx = {'C': csidx, 'O': osidx, 'V': vsidx}
    orb = {'C': orbcs, 'O': orbos, 'V': orbvs}
    return {
        'name': name,
        'x': blocks[name],
        'row_key': row_key,
        'col_key': col_key,
        'row_idx': idx[row_key],
        'col_idx': idx[col_key],
        'row_orb': orb[row_key],
        'col_orb': orb[col_key],
    }


def _sfbase_validate_pair(tdobj, xy, left, right):
    b = make_sasf_blocks(tdobj, xy)
    _sfbase_check_si(b.si)
    return _sfbase_block_data(tdobj, xy, left), _sfbase_block_data(tdobj, xy, right)


def _sfbase_hfx_energy(tdobj, xy, left, right, coeff, omega=None):
    ldat, rdat = _sfbase_validate_pair(tdobj, xy, left, right)
    eri = _general_eri(
        tdobj.mol,
        (ldat['col_orb'], rdat['col_orb'],
         rdat['row_orb'], ldat['row_orb']),
        omega=omega,
    ).reshape(
        len(ldat['col_idx']), len(rdat['col_idx']),
        len(rdat['row_idx']), len(ldat['row_idx']),
    )
    tensor = lib.einsum('ia,jb->abji', ldat['x'], rdat['x'])
    return coeff * float(lib.einsum('abji,abji', tensor, eri))


def satda_sfbase_block_explicit_energy(tdobj, xy, left, right):
    """Explicit ordinary SF-base block scalar.

    The formula is the ordinary UKS SF-TDA ``a->b`` block:
    beta virtual Fock on the right index, alpha occupied Fock on the left
    index, and the spin-flip HF exchange response.
    """
    ldat, rdat = _sfbase_validate_pair(tdobj, xy, left, right)
    fbasis = make_fock_basis(tdobj._scf)
    fock_beta = fbasis.fock0 - fbasis.fockz
    fock_alpha = fbasis.fock0 + fbasis.fockz

    energy = 0.0
    if ldat['row_key'] == rdat['row_key']:
        p_cols = lib.einsum('ia,ib->ab', ldat['x'], rdat['x'])
        f_cols = (
            ldat['col_orb'].T @ fock_beta @ rdat['col_orb']
        )
        energy += float(lib.einsum('ab,ab', p_cols, f_cols))

    if ldat['col_key'] == rdat['col_key']:
        p_rows = lib.einsum('ja,ia->ji', rdat['x'], ldat['x'])
        f_rows = (
            rdat['row_orb'].T @ fock_alpha @ ldat['row_orb']
        )
        energy -= float(lib.einsum('ji,ji', p_rows, f_rows))

    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if hybrid:
        energy += _sfbase_hfx_energy(tdobj, xy, left, right, coeff=-hyb)
        if omega != 0:
            energy += _sfbase_hfx_energy(
                tdobj, xy, left, right, coeff=-(alpha - hyb), omega=omega
            )
    return energy


def satda_sfbase_block_energy_check(tdobj, xy, left, right):
    """Return action-ledger and explicit ordinary SF-base block energies."""
    ledger = satda_sfbase_block_energy(tdobj, xy, left, right)
    explicit = satda_sfbase_block_explicit_energy(tdobj, xy, left, right)
    return {
        'ledger': ledger,
        'explicit': explicit,
        'diff': explicit - ledger,
    }


def satda_sfbase_block_direct_fock(td_grad, tdobj, xy, left, right,
                                   atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    ldat, rdat = _sfbase_validate_pair(tdobj, xy, left, right)
    de = np.zeros((len(atmlst), 3))
    eri1 = tdobj.mol.intor('int2e_ip1', comp=3)

    dm_cols = None
    if ldat['row_key'] == rdat['row_key']:
        p_cols = lib.einsum('ia,ib->ab', ldat['x'], rdat['x'])
        dm_cols = _mo_pair_dm(ldat['col_orb'], p_cols, rdat['col_orb'])

    dm_rows = None
    if ldat['col_key'] == rdat['col_key']:
        p_rows = lib.einsum('ja,ia->ji', rdat['x'], ldat['x'])
        dm_rows = _mo_pair_dm(rdat['row_orb'], p_rows, ldat['row_orb'])

    for k, ia in enumerate(atmlst):
        f1a, f1b = _full_spin_fock_derivs_by_atom(
            td_grad, tdobj, ia, eri1=eri1
        )
        if dm_cols is not None:
            de[k] += lib.einsum('pq,xpq->x', dm_cols, f1b)
        if dm_rows is not None:
            de[k] -= lib.einsum('pq,xpq->x', dm_rows, f1a)
    return de


def satda_sfbase_block_direct_hfx(td_grad, tdobj, xy, left, right,
                                  atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    hybrid, hyb, omega, alpha = _hybrid_coefficients(tdobj._scf)
    if not hybrid:
        return de

    ldat, rdat = _sfbase_validate_pair(tdobj, xy, left, right)
    offsetdic = tdobj.mol.offset_nr_by_atom()

    def add_with_coeff(coeff, omega=None):
        for i in range(len(ldat['row_idx'])):
            for j in range(len(rdat['row_idx'])):
                dm_l = _mo_pair_dm(
                    ldat['col_orb'],
                    np.outer(ldat['x'][i], rdat['x'][j]),
                    rdat['col_orb'],
                )
                dm_r = np.outer(
                    rdat['row_orb'][:, j],
                    ldat['row_orb'][:, i].conj(),
                )
                _add_j_bilinear_ip1(
                    de, td_grad, tdobj.mol, dm_l, dm_r,
                    atmlst, offsetdic, scale=coeff, omega=omega,
                )

    add_with_coeff(-hyb)
    if omega != 0:
        add_with_coeff(-(alpha - hyb), omega=omega)
    return de


def satda_sfbase_block_direct_grad(td_grad, tdobj, xy, left, right,
                                   atmlst=None):
    return (
        satda_sfbase_block_direct_fock(
            td_grad, tdobj, xy, left, right, atmlst=atmlst
        ) +
        satda_sfbase_block_direct_hfx(
            td_grad, tdobj, xy, left, right, atmlst=atmlst
        )
    )


def satda_sfbase_block_m_fock(tdobj, xy, left, right):
    mf = tdobj._scf
    mo = mf.mo_coeff
    nmo = mo.shape[1]
    ldat, rdat = _sfbase_validate_pair(tdobj, xy, left, right)
    fbasis = make_fock_basis(mf, mo)

    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    p_alpha = np.zeros((mf.mol.nao, mf.mol.nao))
    p_beta = np.zeros_like(p_alpha)

    if ldat['row_key'] == rdat['row_key']:
        p_cols = lib.einsum('ia,ib->ab', ldat['x'], rdat['x'])
        _add_fock_term(
            q_alpha, q_beta, p_alpha, p_beta, mo,
            fbasis.fock0_mo, fbasis.fockz_mo,
            ldat['col_idx'], rdat['col_idx'], p_cols, 'beta',
        )

    if ldat['col_key'] == rdat['col_key']:
        p_rows = lib.einsum('ja,ia->ji', rdat['x'], ldat['x'])
        _add_fock_term(
            q_alpha, q_beta, p_alpha, p_beta, mo,
            fbasis.fock0_mo, fbasis.fockz_mo,
            rdat['row_idx'], ldat['row_idx'], -p_rows, 'alpha',
        )

    _add_fock_response_q(tdobj, q_alpha, q_beta, p_alpha, p_beta)
    return q_alpha + q_beta


def satda_sfbase_block_m_hfx(tdobj, xy, left, right):
    mf = tdobj._scf
    nmo = mf.mo_coeff.shape[1]
    q_alpha = np.zeros((nmo, nmo))
    q_beta = np.zeros_like(q_alpha)
    hybrid, hyb, omega, alpha = _hybrid_coefficients(mf)
    if not hybrid:
        return q_alpha

    ldat, rdat = _sfbase_validate_pair(tdobj, xy, left, right)
    tensor = lib.einsum('ia,jb->abji', ldat['x'], rdat['x'])

    def add_with_coeff(coeff, omega=None):
        _add_eri_term_q(
            tdobj, q_alpha, q_beta,
            (ldat['col_orb'], rdat['col_orb'],
             rdat['row_orb'], ldat['row_orb']),
            (ldat['col_idx'], rdat['col_idx'],
             rdat['row_idx'], ldat['row_idx']),
            ('beta', 'beta', 'beta', 'beta'),
            tensor, scale=coeff, omega=omega,
        )

    add_with_coeff(-hyb)
    if omega != 0:
        add_with_coeff(-(alpha - hyb), omega=omega)
    return q_alpha + q_beta


def satda_sfbase_block_m_matrix(tdobj, xy, left, right):
    return (
        satda_sfbase_block_m_fock(tdobj, xy, left, right) +
        satda_sfbase_block_m_hfx(tdobj, xy, left, right)
    )


def satda_sfbase_block_orbital_grad(td_grad, tdobj, xy, left, right,
                                    atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    m = satda_sfbase_block_m_matrix(tdobj, xy, left, right)
    kappas = _roks_canonical_response_kappas_hf(
        td_grad, tdobj, atmlst=atmlst
    )
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        de[k] = lib.einsum('pq,xpq->x', m, kappas[ia])
    return de, kappas


def satda_sfbase_block_analytic_grad(td_grad, tdobj, xy, left, right,
                                     atmlst=None):
    left = left.upper()
    right = right.upper()
    direct = satda_sfbase_block_direct_grad(
        td_grad, tdobj, xy, left, right, atmlst=atmlst
    )
    orbital, kappas = satda_sfbase_block_orbital_grad(
        td_grad, tdobj, xy, left, right, atmlst=atmlst
    )
    return direct + orbital, {
        'direct': direct,
        'orbital': orbital,
        'kappas': kappas,
    }


def _sfbase_cvcv_p_mats(tdobj, xy):
    b = make_sasf_blocks(tdobj, xy)
    _sfbase_check_si(b.si)
    p_vv = lib.einsum('ia,ib->ab', b.x_cv, b.x_cv)
    p_cc = lib.einsum('ia,ja->ij', b.x_cv, b.x_cv)
    return b, p_vv, p_cc


def satda_sfbase_cvcv_explicit_energy(tdobj, xy):
    """Explicit fixed-amplitude ordinary SF-base ``CV-CV`` scalar.

    This is the code-level reference for the analytic ``CV-CV`` block.  It
    mirrors only the ``zs_cv -> v1mo_cv`` terms in ``SATDA.gen_vind_sf``.
    """
    return satda_sfbase_block_explicit_energy(tdobj, xy, 'CV', 'CV')


def _sfbase_cvcv_hfx_energy(tdobj, xy, coeff, omega=None):
    return _sfbase_hfx_energy(tdobj, xy, 'CV', 'CV', coeff, omega=omega)


def satda_sfbase_cvcv_energy_check(tdobj, xy):
    """Return action-ledger and explicit ``CV-CV`` energies for diagnostics."""
    return satda_sfbase_block_energy_check(tdobj, xy, 'CV', 'CV')


def satda_sfbase_cvcv_direct_fock(td_grad, tdobj, xy, atmlst=None):
    """Direct skeleton derivative of the ``CV-CV`` Fock projection terms."""
    return satda_sfbase_block_direct_fock(
        td_grad, tdobj, xy, 'CV', 'CV', atmlst=atmlst
    )


def satda_sfbase_cvcv_direct_hfx(td_grad, tdobj, xy, atmlst=None):
    """Direct ERI skeleton derivative of the ``CV-CV`` response term."""
    return satda_sfbase_block_direct_hfx(
        td_grad, tdobj, xy, 'CV', 'CV', atmlst=atmlst
    )


def satda_sfbase_cvcv_direct_grad(td_grad, tdobj, xy, atmlst=None):
    return (
        satda_sfbase_cvcv_direct_fock(td_grad, tdobj, xy, atmlst=atmlst) +
        satda_sfbase_cvcv_direct_hfx(td_grad, tdobj, xy, atmlst=atmlst)
    )


def satda_sfbase_cvcv_m_fock(tdobj, xy):
    """Unconstrained MO coefficient derivative for CVCV Fock terms."""
    return satda_sfbase_block_m_fock(tdobj, xy, 'CV', 'CV')


def satda_sfbase_cvcv_m_hfx(tdobj, xy):
    """Unconstrained MO coefficient derivative for CVCV HF response."""
    return satda_sfbase_block_m_hfx(tdobj, xy, 'CV', 'CV')


def satda_sfbase_cvcv_m_matrix(tdobj, xy):
    return satda_sfbase_block_m_matrix(tdobj, xy, 'CV', 'CV')


def satda_sfbase_cvcv_orbital_grad(td_grad, tdobj, xy, atmlst=None):
    return satda_sfbase_block_orbital_grad(
        td_grad, tdobj, xy, 'CV', 'CV', atmlst=atmlst
    )


def satda_sfbase_cvcv_analytic_grad(td_grad, tdobj, xy, atmlst=None):
    """Analytic fixed-amplitude CVCV contribution to ``X^T A_SF^[x] X``."""
    return satda_sfbase_block_analytic_grad(
        td_grad, tdobj, xy, 'CV', 'CV', atmlst=atmlst
    )


def _sfbase_total_direct_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    for left in _SFBASE_BLOCK_NAMES:
        for right in _SFBASE_BLOCK_NAMES:
            de += satda_sfbase_block_direct_grad(
                td_grad, tdobj, xy, left, right, atmlst=atmlst
            )
    return de


def _sfbase_total_m_matrix(tdobj, xy):
    m = None
    for left in _SFBASE_BLOCK_NAMES:
        for right in _SFBASE_BLOCK_NAMES:
            mb = satda_sfbase_block_m_matrix(tdobj, xy, left, right)
            m = mb if m is None else m + mb
    return m


def satda_sfbase_gradient_zvec(td_grad, tdobj, xy, atmlst=None,
                               hessian_solver='krylov'):
    """Analytic ordinary SF-base gradient via one ROKS Z-vector solve."""
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)

    m_total = _sfbase_total_m_matrix(tdobj, xy)
    de_direct = _sfbase_total_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)

    if hessian_solver == 'dense':
        hmat, pairs, nmo = build_roks_hessian(tdobj)
        mvec = pack_mvec(m_total, pairs)
        zvec = solve_zvec(hmat, mvec)
        zvec_residual = np.max(np.abs(hmat.T @ zvec - mvec))
    elif hessian_solver == 'krylov':
        action_t, pairs, nmo = make_roks_hessian_transpose_action(tdobj)
        mvec = pack_mvec(m_total, pairs)
        zvec = solve_zvec_krylov(
            action_t, pairs, tdobj, mvec,
            tol=min(getattr(td_grad, 'cphf_conv_tol', 1e-9), 1e-12),
            max_cycle=max(getattr(td_grad, 'cphf_max_cycle', 50), len(mvec)),
            verbose=getattr(td_grad, 'verbose', 0),
        )
        zvec_residual = np.max(np.abs(action_t(zvec) - mvec))
        if zvec_residual > 1e-8:
            hmat, pairs, nmo = build_roks_hessian(tdobj)
            mvec = pack_mvec(m_total, pairs)
            zvec = solve_zvec(hmat, mvec)
            hessian_solver = 'dense_fallback'
            zvec_residual = np.max(np.abs(hmat.T @ zvec - mvec))
        else:
            hmat = None
    else:
        raise ValueError('Unknown hessian_solver %s' % hessian_solver)

    de_orbital = zvec_orbital_grad(
        td_grad, tdobj, hmat, pairs, m_total, zvec, atmlst=atmlst,
    )
    return de_direct + de_orbital, {
        'de_direct': de_direct,
        'de_orbital': de_orbital,
        'm_total': m_total,
        'zvec': zvec,
        'hmat': hmat,
        'pairs': pairs,
        'hessian_solver': hessian_solver,
        'zvec_residual': zvec_residual,
    }

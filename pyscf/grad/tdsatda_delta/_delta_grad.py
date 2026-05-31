"""SATDA total delta_A correction gradient — sum of all 9 non-zero blocks.

Two routes:
  - satda_delta_gradient         : CPHF (each block solves per-atom kappas)
  - satda_delta_gradient_zvec    : Z-vector (one H^T @ z = M_vec solve total)

The z-vector route replaces 9 * 3 * N_atom CPHF solves with ONE linear solve.
"""

import numpy as np

from ._fock_coeff import sasf_fock_coefficient_energy
from ._exchange import sasf_hf_exchange_coefficient_energy
from ._block_analytic_hf import (
    cvcv_m_matrix, cvcv_direct_grad, cvcv_analytic_grad,
)
from ._block_coco_hf import (
    coco_m_matrix, coco_direct_grad, coco_analytic_grad,
)
from ._block_ovov_hf import (
    ovov_m_matrix, ovov_direct_grad, ovov_analytic_grad,
)
from ._block_cvco_hf import (
    cvco_m_matrix, cvco_direct_grad, cvco_analytic_grad,
)
from ._block_cvov_hf import (
    cvov_m_matrix, cvov_direct_grad, cvov_analytic_grad,
)
from ._block_cooo_hf import (
    cooo_m_matrix, cooo_direct_grad, cooo_analytic_grad,
)
from ._block_ovoo_hf import (
    ovoo_m_matrix, ovoo_direct_grad, ovoo_analytic_grad,
)
from ._block_coov_hf import (
    coov_m_matrix, coov_direct_grad, coov_analytic_grad,
)
from ._block_cvoo_hf import (
    cvoo_m_matrix, cvoo_direct_grad, cvoo_analytic_grad,
)

from ._zvec_solver import (
    build_roks_hessian,
    make_roks_hessian_transpose_action,
    pack_mvec,
    roks_canonical_response_kappas,
    solve_zvec,
    solve_zvec_krylov,
    zvec_orbital_grad,
)
from ._direct import (
    sasf_delta_fock_direct_de,
    sasf_delta_hf_exchange_direct_de,
)
from ._xc_lda import (
    lda_xc_energy,
    lda_xc_direct_de,
    lda_xc_m_matrix,
)


def _xc_type(tdobj):
    mf = tdobj._scf
    if hasattr(mf, '_numint'):
        return mf._numint._xc_type(mf.xc)
    return 'HF'


def _assert_total_delta_gradient_supported(tdobj):
    if _xc_type(tdobj) != 'HF':
        raise NotImplementedError(
            'Full SATDA delta DFT gradient is not enabled yet.  The LDA '
            'block XC M-matrix and direct skeleton helpers are implemented '
            'and locally FD-verified, but the DFT nuclear perturbation RHS '
            'for the ROKS Z-vector equation still needs a separate '
            'calibrated implementation.'
        )


def _assert_cpks_delta_gradient_supported(tdobj):
    xctype = _xc_type(tdobj)
    if xctype not in ('HF', 'LDA'):
        raise NotImplementedError(
            'The forward SATDA delta CPKS route currently supports only '
            'HF and LDA references'
        )


def _total_delta_energy(tdobj, xy):
    return (
        sasf_fock_coefficient_energy(tdobj, xy)
        + sasf_hf_exchange_coefficient_energy(tdobj, xy)
        + lda_xc_energy(tdobj, xy)
    )


def _total_m_matrix(tdobj, xy):
    return (
        cvcv_m_matrix(tdobj, xy) +
        coco_m_matrix(tdobj, xy) +
        ovov_m_matrix(tdobj, xy) +
        cvco_m_matrix(tdobj, xy) +
        cvov_m_matrix(tdobj, xy) +
        cooo_m_matrix(tdobj, xy) +
        ovoo_m_matrix(tdobj, xy) +
        coov_m_matrix(tdobj, xy) +
        cvoo_m_matrix(tdobj, xy) +
        lda_xc_m_matrix(tdobj, xy)
    )


def _total_direct_grad(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    de = np.zeros((len(atmlst), 3))
    de += cvcv_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += coco_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += ovov_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += cvco_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += cvov_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += cooo_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += ovoo_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += coov_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += cvoo_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)
    de += lda_xc_direct_de(td_grad, tdobj, xy, atmlst=atmlst)
    return de


def _total_direct_grad_cpks(td_grad, tdobj, xy, atmlst=None):
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    return (
        sasf_delta_fock_direct_de(td_grad, tdobj, xy, atmlst=atmlst)
        + sasf_delta_hf_exchange_direct_de(
            td_grad, tdobj, xy, atmlst, tdobj.mol.offset_nr_by_atom())
        + lda_xc_direct_de(td_grad, tdobj, xy, atmlst=atmlst)
    )


# ---------------------------------------------------------------------------
#  CPHF route (legacy — 9 separate per-block CPHF solves)
# ---------------------------------------------------------------------------

def sasf_delta_gradient(td_grad, tdobj, xy, atmlst=None, verbose=0):
    """Total SASF spin-adaptation correction gradient — CPHF route.

    Each of the 9 blocks calls the shared CPHF solver independently,
    resulting in 9 * 3 * N_atom linear solves.
    """
    _assert_total_delta_gradient_supported(tdobj)
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)

    blocks = [
        ('CV-CV', cvcv_analytic_grad),
        ('CO-CO', coco_analytic_grad),
        ('OV-OV', ovov_analytic_grad),
        ('CV-CO', cvco_analytic_grad),
        ('CV-OV', cvov_analytic_grad),
        ('CO-OO', cooo_analytic_grad),
        ('OV-OO', ovoo_analytic_grad),
        ('CO-OV', coov_analytic_grad),
        ('CV-OO', cvoo_analytic_grad),
    ]

    de = np.zeros((len(atmlst), 3))
    parts = {}
    for name, grad_fn in blocks:
        d, _ = grad_fn(td_grad, tdobj, xy, atmlst=atmlst)
        de += d
        parts[name] = d
    return de, parts


def sasf_delta_gradient_cpks(td_grad, tdobj, xy, atmlst=None):
    """Total SATDA delta_A gradient via forward ROKS CPKS equations."""
    _assert_cpks_delta_gradient_supported(tdobj)
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)

    m_total = _total_m_matrix(tdobj, xy)
    de_direct = _total_direct_grad_cpks(
        td_grad, tdobj, xy, atmlst=atmlst
    )
    kappas, response = roks_canonical_response_kappas(
        td_grad, tdobj, atmlst=atmlst
    )
    de_orbital = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        de_orbital[k] = np.einsum('pq,xpq->x', m_total, kappas[ia])

    return de_direct + de_orbital, {
        'de_direct': de_direct,
        'de_orbital': de_orbital,
        'm_total': m_total,
        'kappas': kappas,
        'response': response,
    }


# ---------------------------------------------------------------------------
#  Z-vector route (ONE linear solve total)
# ---------------------------------------------------------------------------

def sasf_delta_gradient_zvec(td_grad, tdobj, xy, atmlst=None,
                             hessian_solver='krylov'):
    """Total SASF delta_A gradient via Z-vector method.

    One solve of H^T @ z = M_vec replaces 9 * 3 * N_atom CPHF solves.

    Returns
    -------
    de : (natm, 3) ndarray — total delta_A nuclear gradient
    details : dict with keys:
        de_direct, de_orbital, m_total, zvec, hmat, pairs
    """
    _assert_total_delta_gradient_supported(tdobj)
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)

    # 1. Sum M-matrices from all 9 blocks
    m_total = _total_m_matrix(tdobj, xy)

    # 2. Sum direct (skeleton) gradients
    de_direct = _total_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)

    # 3. Pack M-vector and solve Z-vector
    if hessian_solver == 'dense':
        hmat, pairs, nmo = build_roks_hessian(tdobj)
        mvec = pack_mvec(m_total, pairs)
        zvec = solve_zvec(hmat, mvec)
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

    # 4. Orbital response via Z-vector
    de_orbital = zvec_orbital_grad(
        td_grad, tdobj, hmat, pairs, m_total, zvec, atmlst=atmlst,
    )

    de = de_direct + de_orbital
    return de, {
        'de_direct': de_direct,
        'de_orbital': de_orbital,
        'm_total': m_total,
        'zvec': zvec,
        'hmat': hmat,
        'pairs': pairs,
        'hessian_solver': hessian_solver,
        'zvec_residual': (
            np.max(np.abs(hmat.T @ zvec - mvec))
            if hessian_solver == 'dense' else zvec_residual
        ),
    }


satda_delta_gradient = sasf_delta_gradient
satda_delta_gradient_zvec = sasf_delta_gradient_zvec
satda_delta_gradient_cpks = sasf_delta_gradient_cpks

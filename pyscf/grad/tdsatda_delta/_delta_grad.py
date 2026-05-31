"""SATDA total delta_A correction gradient — sum of all 9 non-zero blocks.

Two routes:
  - satda_delta_gradient         : CPHF (each block solves per-atom kappas)
  - satda_delta_gradient_zvec    : Z-vector (one H^T @ z = M_vec solve total)

The z-vector route replaces 9 * 3 * N_atom CPHF solves with ONE linear solve.
"""

import numpy as np

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
    pack_mvec,
    solve_zvec,
    zvec_orbital_grad,
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
        cvoo_m_matrix(tdobj, xy)
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
    return de


# ---------------------------------------------------------------------------
#  CPHF route (legacy — 9 separate per-block CPHF solves)
# ---------------------------------------------------------------------------

def sasf_delta_gradient(td_grad, tdobj, xy, atmlst=None, verbose=0):
    """Total SASF spin-adaptation correction gradient — CPHF route.

    Each of the 9 blocks calls the shared CPHF solver independently,
    resulting in 9 * 3 * N_atom linear solves.
    """
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


# ---------------------------------------------------------------------------
#  Z-vector route (ONE linear solve total)
# ---------------------------------------------------------------------------

def sasf_delta_gradient_zvec(td_grad, tdobj, xy, atmlst=None):
    """Total SASF delta_A gradient via Z-vector method.

    One solve of H^T @ z = M_vec replaces 9 * 3 * N_atom CPHF solves.

    Returns
    -------
    de : (natm, 3) ndarray — total delta_A nuclear gradient
    details : dict with keys:
        de_direct, de_orbital, m_total, zvec, hmat, pairs
    """
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)

    # 1. Sum M-matrices from all 9 blocks
    m_total = _total_m_matrix(tdobj, xy)

    # 2. Sum direct (skeleton) gradients
    de_direct = _total_direct_grad(td_grad, tdobj, xy, atmlst=atmlst)

    # 3. Build Hessian once, pack M-vector, solve Z-vector
    hmat, pairs, nmo = build_roks_hessian(tdobj)
    mvec = pack_mvec(m_total, pairs)
    zvec = solve_zvec(hmat, mvec)

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
    }


satda_delta_gradient = sasf_delta_gradient
satda_delta_gradient_zvec = sasf_delta_gradient_zvec

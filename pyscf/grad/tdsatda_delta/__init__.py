"""Block-resolved HF delta-gradient helpers for SATDA.

This package is a direct migration of the validated SASF HF delta-gradient
blocks.  The public SATDA names are aliases for the migrated implementation so
that the first migration step does not change the underlying equations.
"""

from ._delta_grad import (
    satda_delta_block_energies,
    satda_delta_block_energy,
    satda_delta_finite_diff,
    satda_delta_gradient_reference,
    sasf_delta_gradient,
    sasf_delta_gradient_cpks,
    sasf_delta_gradient_zvec,
)
from ._grad import Grad, Gradients
from ._sfbase_grad import (
    satda_sfbase_action,
    satda_sfbase_block_analytic_grad,
    satda_sfbase_block_energies,
    satda_sfbase_block_energy,
    satda_sfbase_block_energy_check,
    satda_sfbase_block_explicit_energy,
    satda_sfbase_block_m_matrix,
    satda_sfbase_block_sum_energy,
    satda_sfbase_cvcv_analytic_grad,
    satda_sfbase_cvcv_direct_grad,
    satda_sfbase_cvcv_energy_check,
    satda_sfbase_cvcv_explicit_energy,
    satda_sfbase_cvcv_m_matrix,
    satda_sfbase_energy,
    satda_sfbase_finite_diff,
    satda_sfbase_gradient_zvec,
    satda_sfbase_ledger_report,
)

satda_delta_gradient = sasf_delta_gradient
satda_delta_gradient_cpks = sasf_delta_gradient_cpks
satda_delta_gradient_zvec = sasf_delta_gradient_zvec

__all__ = [
    "sasf_delta_gradient",
    "sasf_delta_gradient_cpks",
    "sasf_delta_gradient_zvec",
    "satda_delta_gradient",
    "satda_delta_gradient_cpks",
    "satda_delta_gradient_zvec",
    "satda_delta_block_energy",
    "satda_delta_block_energies",
    "satda_delta_finite_diff",
    "satda_delta_gradient_reference",
    "satda_sfbase_action",
    "satda_sfbase_energy",
    "satda_sfbase_block_analytic_grad",
    "satda_sfbase_block_energy",
    "satda_sfbase_block_energy_check",
    "satda_sfbase_block_energies",
    "satda_sfbase_block_explicit_energy",
    "satda_sfbase_block_m_matrix",
    "satda_sfbase_block_sum_energy",
    "satda_sfbase_cvcv_analytic_grad",
    "satda_sfbase_cvcv_direct_grad",
    "satda_sfbase_cvcv_energy_check",
    "satda_sfbase_cvcv_explicit_energy",
    "satda_sfbase_cvcv_m_matrix",
    "satda_sfbase_finite_diff",
    "satda_sfbase_ledger_report",
    "satda_sfbase_gradient_zvec",
    "Grad",
    "Gradients",
]

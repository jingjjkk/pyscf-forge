"""Block-resolved HF delta-gradient helpers for SATDA.

This package is a direct migration of the validated SASF HF delta-gradient
blocks.  The public SATDA names are aliases for the migrated implementation so
that the first migration step does not change the underlying equations.
"""

from ._delta_grad import (
    sasf_delta_gradient,
    sasf_delta_gradient_cpks,
    sasf_delta_gradient_zvec,
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
]

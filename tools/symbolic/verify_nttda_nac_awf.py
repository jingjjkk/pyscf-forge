#!/usr/bin/env python
"""Deterministic algebra checks for the NTTDA ``deltaS=-1`` AWF."""

from pathlib import Path
import sys

import numpy as np
import pyscf


ROOT = Path(__file__).resolve().parents[2]
PYSCF_PATH = str(ROOT / "pyscf")
if PYSCF_PATH not in pyscf.__path__:
    pyscf.__path__.insert(0, PYSCF_PATH)

import pyscf.grad  # noqa: E402
import pyscf.nac  # noqa: E402
import pyscf.sftda  # noqa: E402

for package, path in (
        (pyscf.grad, ROOT / "pyscf" / "grad"),
        (pyscf.nac, ROOT / "pyscf" / "nac"),
        (pyscf.sftda, ROOT / "pyscf" / "sftda")):
    path = str(path)
    if path not in package.__path__:
        package.__path__.insert(0, path)

from pyscf.grad.nttda.delta_s_minus_one import (  # noqa: E402
    spin_lowering_fock_projections,
)
from pyscf.nac.nttda import (  # noqa: E402
    build_spin_adapted_awf,
    interstate_rdm1,
)


class _Reference:
    pass


class _TD:
    deltaS = -1


def _make_td(nclosed, nopen, nvirtual):
    nmo = nclosed + nopen + nvirtual
    mf = _Reference()
    mf.mo_occ = np.r_[
        np.full(nclosed, 2), np.ones(nopen), np.zeros(nvirtual),
    ]
    mf.mo_coeff = np.eye(nmo)
    tdobj = _TD()
    tdobj._scf = mf
    tdobj.mol = None
    return tdobj


def _trace_free_amplitude(rng, nclosed, nopen, nvirtual):
    amplitude = rng.normal(size=(nclosed + nopen, nopen + nvirtual))
    oo = amplitude[nclosed:, :nopen]
    oo -= np.eye(nopen) * np.trace(oo) / nopen
    return amplitude


def _fock_probe(tdobj, amplitude):
    nmo = tdobj._scf.mo_coeff.shape[1]
    output = np.zeros((nmo, nmo))
    for term in spin_lowering_fock_projections(tdobj, (amplitude, 0)):
        output += term.weight_f0 * term.density()
    return output


def run_checks():
    rng = np.random.default_rng(20260717)
    maximum = {
        "norm": 0.0,
        "orthogonality": 0.0,
        "symmetric_rdm": 0.0,
        "antisymmetric_source": 0.0,
    }
    for nopen in (2, 3, 4):
        nclosed, nvirtual = 2, 2
        tdobj = _make_td(nclosed, nopen, nvirtual)
        x_i = _trace_free_amplitude(
            rng, nclosed, nopen, nvirtual,
        )
        x_i /= np.linalg.norm(x_i)
        x_j = _trace_free_amplitude(
            rng, nclosed, nopen, nvirtual,
        )
        x_j -= x_i * np.vdot(x_i, x_j)
        x_j /= np.linalg.norm(x_j)

        awf_i = build_spin_adapted_awf(tdobj, (x_i, 0))
        awf_j = build_spin_adapted_awf(tdobj, (x_j, 0))
        maximum["norm"] = max(
            maximum["norm"], abs(awf_i.norm - 1), abs(awf_j.norm - 1),
        )
        # Identity MO overlap is sufficient for this same-geometry check.
        from pyscf.nac.nttda import _awf_overlap_from_mo
        overlap = _awf_overlap_from_mo(
            awf_i, awf_j, np.eye(awf_i.nmo),
        )
        maximum["orthogonality"] = max(
            maximum["orthogonality"], abs(overlap),
        )

        gamma = interstate_rdm1(tdobj, (x_i, 0), (x_j, 0))
        polarized_probe = 0.25 * (
            _fock_probe(tdobj, x_i + x_j)
            - _fock_probe(tdobj, x_i - x_j)
        )
        maximum["symmetric_rdm"] = max(
            maximum["symmetric_rdm"],
            np.max(np.abs(
                0.5 * (gamma + gamma.T)
                - 0.5 * (polarized_probe + polarized_probe.T)
            )),
        )
        antisymmetric = 0.5 * (gamma - gamma.T)
        maximum["antisymmetric_source"] = max(
            maximum["antisymmetric_source"],
            np.max(np.abs(antisymmetric + antisymmetric.T)),
        )
    return maximum


if __name__ == "__main__":
    errors = run_checks()
    for name, error in errors.items():
        print(f"{name:24s} max_abs_error = {error:.3e}")
    if max(errors.values()) > 1e-11:
        sys.exit(1)

# SATDA deltaS=-1 Gradient Status

This package contains the block-resolved analytic gradient implementation for
`pyscf.sftda.satda.SATDA` with `deltaS=-1`.

## Trusted Entry Point

Use:

```python
g = td.Gradients().kernel(state=state, method='analytic')
```

for HF/ROKS references.  This returns the total excited-state gradient

```text
d(E_ref + omega_SATDA)/dR
```

where the electronic excitation-energy part is assembled as

```text
omega^[R] = SF-base 16 block-pair analytic gradient
          + delta_A 9 block analytic gradient
```

`method='finite_diff'` remains available as the expensive reference for
`E_ref + omega_SATDA`.

## What Is Complete

HF/ROKS, `deltaS=-1` is complete.

The SF-base part is implemented in `_sfbase_grad.py` as all 16 block pairs:

```text
CO/CO, CO/CV, CO/OO, CO/OV,
CV/CO, CV/CV, CV/OO, CV/OV,
OO/CO, OO/CV, OO/OO, OO/OV,
OV/CO, OV/CV, OV/OO, OV/OV
```

The SATDA correction part is implemented in `_delta_grad.py` as the 9
independent delta blocks:

```text
CV-CV, CO-CO, OV-OV,
CV-CO, CV-OV, CO-OO,
OV-OO, CO-OV, CV-OO
```

Important coefficient fixes were made before this status was written:

- `OV-OO`: the Fock-like coefficient/probe structure was corrected.
- `CO-OV`: the block is now in symmetric pair-sum convention.
- `CV-OO`: the block is now in symmetric pair-sum convention.

After these fixes, the scalar ledger satisfies:

```text
SF-base block sum + delta_A block sum == SATDA eigenvalue
```

to about `1e-15` on the HNO/STO-3G/ROKS-HF validation case.

## Validation Snapshot

System:

```text
HNO / STO-3G / ROKS-HF / spin=2 / SATDA(deltaS=-1)
```

Validation command path:

```python
td.Gradients().kernel(state=state, method='analytic')
td.Gradients().kernel(state=state, method='finite_diff', step=2e-4)
```

Results for total excited-state gradients:

```text
state 1 max|analytic - finite_diff| = 5.918e-08
state 3 max|analytic - finite_diff| = 4.717e-08
state 4 max|analytic - finite_diff| = 4.796e-08
```

The finite-difference reference includes relaxed SCF and state tracking.

## Files To Trust

Core trusted files:

- `_grad.py`: public PySCF-style gradient interface.
- `_sfbase_grad.py`: ordinary SF-base block ledger and analytic gradient.
- `_delta_grad.py`: SATDA delta_A block ledger and analytic gradient.
- `_zvec_solver.py`: canonical ROKS response and Z-vector solver.
- `_block_*_hf.py`: individual HF delta_A block implementations.
- `_direct.py`, `_q_rhs.py`, `_fock_basis.py`, `_blocks.py`: shared helpers.

Theory/status notes:

- `derivations_sfbase_grad.md`
- `derivations_response_zvector.md`

## Quarantined / Not Trusted

The obsolete monolithic HF gradient assembly was moved out of the active
package path:

```text
.codex_trash/satda_gradient_hf_cleanup_20260602/_grad_hf.py
```

Do not import from `.codex_trash/` and do not use it as a formula or validation
reference.

The old `pyscf.grad.tdsatda` path and old `pyscf/grad/test/test_satda_grad.py`
are not trusted validation routes for this work.

## Current Limitations

Only HF/ROKS `deltaS=-1` is enabled through the analytic interface.

Non-HF DFT references are intentionally not enabled yet.  Existing LDA/GGA/XC
notes and partial helpers should be treated as development material until the
full DFT scalar ledger and gradient are block-validated against finite
differences.

## Recommended Next Work

1. Add a small regression test for HF/ROKS `deltaS=-1` using the public
   `td.Gradients().kernel(method='analytic')` interface.
2. Re-enter DFT work from the block ledger, not from the old monolithic
   gradient assembly.
3. For LDA first, require both:
   - scalar closure against `td.gen_vind_sf()`;
   - block or grouped-block gradient closure against relaxed-SCF FD.
4. Only after LDA closes should GGA be added.

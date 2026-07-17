# Symbolic and tiny-system checks

These scripts support formula-to-code validation for the NTTDA derivative
implementations.  They are deterministic developer checks, not production
calculation entrypoints.

- `verify_nttda_nac_awf.py`: checks the spin-adapted `deltaS=-1` AWF metric,
  transition-density symmetry, and the connection source used by NTTDA NAC.

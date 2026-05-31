'''Finite-difference support helpers for SASF gradients.'''

from pyscf import dft


def _copy_roks_settings(mf_ref, mf):
    mf.verbose = 0
    mf.max_memory = mf_ref.max_memory
    mf.conv_tol = mf_ref.conv_tol
    mf.conv_tol_grad = mf_ref.conv_tol_grad
    mf.max_cycle = mf_ref.max_cycle
    mf.diis_space = mf_ref.diis_space
    mf.diis_start_cycle = mf_ref.diis_start_cycle
    mf.level_shift = mf_ref.level_shift
    mf.damp = mf_ref.damp
    if isinstance(mf_ref, dft.KohnShamDFT):
        mf.grids.level = mf_ref.grids.level
        mf.grids.prune = mf_ref.grids.prune
        mf.grids.radi_method = mf_ref.grids.radi_method
        mf.grids.becke_scheme = mf_ref.grids.becke_scheme
        mf.small_rho_cutoff = mf_ref.small_rho_cutoff
    return mf


def _make_displaced_mf(mf_ref, mol):
    if isinstance(mf_ref, dft.roks.ROKS):
        mf = mol.ROKS(xc=mf_ref.xc)
    elif isinstance(mf_ref, dft.rks_symm.SymAdaptedROKS):
        mf = mol.ROKS(xc=mf_ref.xc)
    elif mf_ref.__class__.__name__.endswith('ROHF'):
        mf = mol.ROHF()
    else:
        raise NotImplementedError(
            'TDA_SASF finite-difference gradients currently support ROKS/ROHF '
            'references only'
        )
    return _copy_roks_settings(mf_ref, mf)

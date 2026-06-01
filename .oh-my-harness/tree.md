# Tree

Use this file for navigation only. Verify implementation details by reading source files directly.

- Source: `git ls-files --cached --others --exclude-standard`
- Entries: 374

```text
./
├── .agents/
│   └── skills/
│       ├── harness/
│       │   ├── agents/
│       │   │   └── openai.yaml
│       │   ├── refs/
│       │   │   ├── local-review.md
│       │   │   ├── visual-display.md
│       │   │   └── writing-plan.md
│       │   ├── scripts/
│       │   │   └── review-wait.mjs
│       │   └── SKILL.md
│       ├── receiving-code-review/
│       │   └── SKILL.md
│       ├── systematic-debugging/
│       │   ├── condition-based-waiting-example.ts
│       │   ├── condition-based-waiting.md
│       │   ├── CREATION-LOG.md
│       │   ├── defense-in-depth.md
│       │   ├── find-polluter.sh
│       │   ├── LICENSE.upstream
│       │   ├── root-cause-tracing.md
│       │   ├── SKILL.md
│       │   ├── test-academic.md
│       │   ├── test-pressure-1.md
│       │   ├── test-pressure-2.md
│       │   └── test-pressure-3.md
│       └── tdd/
│           ├── LICENSE.upstream
│           ├── mocking.md
│           ├── SKILL.md
│           └── tests.md
├── .codex/
│   └── hooks.json
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── feature-transfer.md
│   ├── PULL_REQUEST_TEMPLATE/
│   │   ├── implementation.md
│   │   └── research.md
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── lint.yml
│   │   ├── publish.yml
│   │   ├── run_ci.sh
│   │   └── test.sh
│   ├── pr-review-comment.md
│   └── writing-plan.md
├── .oh-my-harness/
│   └── hooks/
│       └── tree.mjs
├── doc/
│   └── mcpdft/
│       └── README.md
├── docs/
│   └── specs/
│       ├── agent-workflow.md
│       └── review-guidelines.md
├── examples/
│   ├── afqmc/
│   │   ├── staging/
│   │   │   ├── 00-stage-cisd.py
│   │   │   ├── 01-run-staged.py
│   │   │   ├── 02-run-staged-multi-device.py
│   │   │   └── README.md
│   │   ├── 00-rhf.py
│   │   ├── 01-cisd.py
│   │   ├── 02-rhf-fp.py
│   │   ├── 03-lno-rhfafqmc.py
│   │   ├── 04-ucisd.py
│   │   ├── 05-cisd-fp.py
│   │   └── README.md
│   ├── csf_fci/
│   │   ├── 01-csf_fci.py
│   │   └── 02-csf_symm_fci.py
│   ├── dft2/
│   │   └── 24-reparameterize_xc_functional.py
│   ├── dsrg_mrpt2/
│   │   ├── 01-simple.py
│   │   └── 02-state_average.py
│   ├── geomopt/
│   │   ├── 00-mcpdft.py
│   │   └── 01-multi_state.py
│   ├── grad/
│   │   ├── 01-spin_flip_tddft_grad.py
│   │   ├── 02-satda_finite_diff_grad.py
│   │   ├── 20_dft_corrected_casci_grad.py
│   │   ├── 21_fomo_casci_grad.py
│   │   └── 22_fomo_dft_corrected_casci_grad.py
│   ├── lno/
│   │   ├── 00-lnoccsd(t)_pm.py
│   │   ├── 01-lnoccsd(t)_iao.py
│   │   ├── 02-ulnoccsd(t)_pm.py
│   │   └── 03-klnoccsd(t)_pm.py
│   ├── mcdcft/
│   │   ├── 01_h2_potential_curve.py
│   │   ├── 02_h2_potential_curve_restart.py
│   │   ├── 03_DC24_single_point.py
│   │   └── 04_functional_parameter.py
│   ├── mcpdft/
│   │   └── 03-metaGGA_functionals.py
│   ├── mcscf/
│   │   ├── 81_dft_corrected_casci_embedding.py
│   │   ├── 82_fomo_casci.py
│   │   └── 83_fomo_dft_corrected_casci.py
│   ├── msdft/
│   │   └── 01-simple-noci.py
│   ├── nac/
│   │   └── 01-cmspdft_nac.py
│   ├── occri/
│   │   ├── isdfx/
│   │   │   ├── 01-basic_usage.py
│   │   │   └── 02-kpoint_scaling.py
│   │   ├── 01-simple_gamma_point.py
│   │   └── 02-kpoint_calculations.py
│   ├── pbc/
│   │   └── 22-kpoints_khf_stagger.py
│   ├── pprpa/
│   │   ├── 01-pprpa_total_energy.py
│   │   ├── 02-pprpa_excitation_energy.py
│   │   ├── 03-hhrpa_excitation_energy.py
│   │   ├── 04-gamma_pprpa_excitation_energy.py
│   │   └── 05-gamma_hhrpa_excitation_energy.py
│   ├── prop/
│   │   ├── 00-dipole_moment.py
│   │   ├── 01-excited_state_dipole_moment.py
│   │   ├── 02-pdft_transition_dipole_moment.py
│   │   └── 03-lpdft-dipole_moment.py
│   ├── pv/
│   │   └── 00.energy-PV.py
│   ├── pwscf/
│   │   ├── al.py
│   │   ├── kccsd.py
│   │   ├── kmp2.py
│   │   ├── kpt_symm.py
│   │   ├── li.py
│   │   ├── README.md
│   │   ├── set_meshes.py
│   │   └── sg15.py
│   ├── scf/
│   │   └── 73-m3soscf.py
│   └── sftda/
│       ├── 01-spin_flip_tddft.py
│       ├── 02-spin_flip_tddft_roks.py
│       └── 03-spin_square_and_fosc.py
├── pyscf/
│   ├── afqmc/
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── levels.py
│   │   │   ├── ops.py
│   │   │   ├── system.py
│   │   │   └── typing.py
│   │   ├── ham/
│   │   │   ├── __init__.py
│   │   │   ├── chol.py
│   │   │   └── hubbard.py
│   │   ├── meas/
│   │   │   ├── __init__.py
│   │   │   ├── auto.py
│   │   │   ├── cis.py
│   │   │   ├── cisd.py
│   │   │   ├── eom_cisd.py
│   │   │   ├── eom_t_cisd.py
│   │   │   ├── gcisd.py
│   │   │   ├── ghf.py
│   │   │   ├── multi_ghf.py
│   │   │   ├── pt2ccsd.py
│   │   │   ├── rhf.py
│   │   │   ├── ucisd.py
│   │   │   └── uhf.py
│   │   ├── prop/
│   │   │   ├── __init__.py
│   │   │   ├── afqmc_fp.py
│   │   │   ├── afqmc.py
│   │   │   ├── blocks.py
│   │   │   ├── chol_afqmc_ops_fp.py
│   │   │   ├── chol_afqmc_ops.py
│   │   │   ├── types.py
│   │   │   └── utils.py
│   │   ├── test/
│   │   │   ├── _import_boundary_helper.py
│   │   │   ├── _sharding_helper.py
│   │   │   ├── conftest.py
│   │   │   ├── test_afqmc_fp.py
│   │   │   ├── test_afqmc.py
│   │   │   ├── test_import.py
│   │   │   ├── test_lnoafqmc.py
│   │   │   └── test_sharding.py
│   │   ├── trial/
│   │   │   ├── __init__.py
│   │   │   ├── auto.py
│   │   │   ├── cis.py
│   │   │   ├── cisd.py
│   │   │   ├── eom_cisd.py
│   │   │   ├── eom_t_cisd.py
│   │   │   ├── gcisd.py
│   │   │   ├── ghf.py
│   │   │   ├── multi_ghf.py
│   │   │   ├── pt2ccsd.py
│   │   │   ├── rhf.py
│   │   │   ├── ucisd.py
│   │   │   └── uhf.py
│   │   ├── __init__.py
│   │   ├── afqmc.py
│   │   ├── config.py
│   │   ├── driver.py
│   │   ├── lnoafqmc.py
│   │   ├── runtime_layout.py
│   │   ├── runtime_provenance.py
│   │   ├── setup_fp.py
│   │   ├── setup.py
│   │   ├── sharding.py
│   │   ├── staging.py
│   │   ├── stat_utils.py
│   │   ├── testing.py
│   │   └── walkers.py
│   ├── csf_fci/
│   │   ├── test/
│   │   │   ├── test_csf_symm.py
│   │   │   ├── test_csf.py
│   │   │   ├── test_csfstring.py
│   │   │   └── test_spin_op.py
│   │   ├── __init__.py
│   │   ├── csdstring.py
│   │   ├── csf_symm.py
│   │   ├── csf.py
│   │   ├── csfstring.py
│   │   └── spin_op.py
│   ├── dft2/
│   │   ├── test/
│   │   │   ├── dm_h4.npy
│   │   │   ├── test_grad_metagga_mcpdft.py
│   │   │   ├── test_libxc.py
│   │   │   ├── test_lpdft.py
│   │   │   └── test_mgga.py
│   │   ├── __init__.py
│   │   └── libxc.py
│   ├── dsrg_mrpt2/
│   │   ├── test/
│   │   │   └── test_dsrg_mrpt2.py
│   │   ├── __init__.py
│   │   └── dsrg_mrpt2.py
│   ├── grad/
│   │   ├── tdsatda_delta/
│   │   │   ├── __init__.py
│   │   │   ├── _block_analytic_hf.py
│   │   │   ├── _block_coco_hf.py
│   │   │   ├── _block_cooo_hf.py
│   │   │   ├── _block_coov_hf.py
│   │   │   ├── _block_cvco_hf.py
│   │   │   ├── _block_cvoo_hf.py
│   │   │   ├── _block_cvov_hf.py
│   │   │   ├── _block_ovoo_hf.py
│   │   │   ├── _block_ovov_hf.py
│   │   │   ├── _blocks.py
│   │   │   ├── _delta_grad.py
│   │   │   ├── _direct.py
│   │   │   ├── _exchange.py
│   │   │   ├── _fd.py
│   │   │   ├── _fock_basis.py
│   │   │   ├── _fock_coeff.py
│   │   │   ├── _grad.py
│   │   │   ├── _q_rhs.py
│   │   │   ├── _roks.py
│   │   │   ├── _sfbase_grad.py
│   │   │   ├── _xc_lda.py
│   │   │   ├── _zvec_solver.py
│   │   │   ├── derivations_response_zvector.md
│   │   │   ├── derivations_sfbase_grad.md
│   │   │   └── README.md
│   │   ├── test/
│   │   │   ├── test_grad_dft_corrected_casci.py
│   │   │   ├── test_sftda_grad.py
│   │   │   └── test_sftddft_grad.py
│   │   ├── derivations_xc_functional.md
│   │   ├── dft_corrected_casci.py
│   │   └── tduks_sf.py
│   ├── lib/
│   │   ├── csf/
│   │   │   └── csfstring.c
│   │   ├── dft/
│   │   │   └── libxc_itrf2.c
│   │   ├── dsrg/
│   │   │   └── dsrg_helper.c
│   │   ├── lno/
│   │   │   ├── ccsd_t.c
│   │   │   └── uccsd_t.c
│   │   ├── occri/
│   │   │   ├── occri.c
│   │   │   └── occri.h
│   │   ├── pwscf/
│   │   │   ├── CMakeLists.txt
│   │   │   └── pwscf.c
│   │   ├── sfnoci/
│   │   │   ├── CMakeLists.txt
│   │   │   └── SFNOCI_contract.c
│   │   └── CMakeLists.txt
│   ├── lno/
│   │   ├── test/
│   │   │   ├── test_lnoccsd.py
│   │   │   ├── test_makelnordm1.py
│   │   │   ├── test_ulnoccsd_t.py
│   │   │   └── test_ulnoccsd.py
│   │   ├── __init__.py
│   │   ├── domain.py
│   │   ├── lno.py
│   │   ├── lnoccsd_t.py
│   │   ├── lnoccsd.py
│   │   ├── make_lno_rdm1.py
│   │   ├── tools.py
│   │   ├── ulno.py
│   │   ├── ulnoccsd_t_slow.py
│   │   ├── ulnoccsd_t.py
│   │   └── ulnoccsd.py
│   ├── lrdf/
│   │   ├── grad/
│   │   │   ├── __init__.py
│   │   │   └── rhf.py
│   │   ├── hessian/
│   │   │   ├── __init__.py
│   │   │   └── rhf.py
│   │   ├── test/
│   │   │   ├── test_df_grad.py
│   │   │   ├── test_df_hess.py
│   │   │   └── test_lrdf.py
│   │   ├── __init__.py
│   │   └── lrdf.py
│   ├── mcdcft/
│   │   ├── test/
│   │   │   └── test_energy_h2.py
│   │   ├── __init__.py
│   │   ├── dcfnal.py
│   │   └── mcdcft.py
│   ├── mcscf/
│   │   ├── test/
│   │   │   └── test_mcscf_dft_corrected_casci.py
│   │   └── dft_corrected_casci.py
│   ├── msdft/
│   │   ├── tests/
│   │   │   └── test_noci.py
│   │   ├── __init__.py
│   │   └── noci.py
│   ├── nac/
│   │   ├── test/
│   │   │   └── test_tdsatda.py
│   │   ├── __init__.py
│   │   └── derivation_nac.md
│   ├── occri/
│   │   ├── isdfx/
│   │   │   ├── __init__.py
│   │   │   ├── interpolation.py
│   │   │   ├── isdfx_k_kpts.py
│   │   │   └── utils.py
│   │   ├── test/
│   │   │   ├── test_isdfx_energy.py
│   │   │   ├── test_isdfx.py
│   │   │   ├── test_occri.py
│   │   │   ├── test_performance.py
│   │   │   └── test_regression.py
│   │   ├── __init__.py
│   │   ├── .gitignore
│   │   ├── occri_k_kpts.py
│   │   └── utils.py
│   ├── pbc/
│   │   ├── lno/
│   │   │   ├── test/
│   │   │   │   ├── test_klnoccsd.py
│   │   │   │   └── test_makeklnordm1.py
│   │   │   ├── __init__.py
│   │   │   ├── klno.py
│   │   │   ├── klnoccsd.py
│   │   │   ├── make_lno_rdm1.py
│   │   │   └── tools.py
│   │   ├── pwscf/
│   │   │   ├── ao2mo/
│   │   │   │   ├── __init__.py
│   │   │   │   └── molint.py
│   │   │   ├── test/
│   │   │   │   ├── test_gto_vs_pw.py
│   │   │   │   ├── test_hf_and_ks.py
│   │   │   │   ├── test_kpt_symm.py
│   │   │   │   ├── test_krccsd.py
│   │   │   │   ├── test_krhf_krmp2.py
│   │   │   │   ├── test_krmp2.py
│   │   │   │   ├── test_kuhf_kump2.py
│   │   │   │   ├── test_kump2.py
│   │   │   │   ├── test_ncpp_cell.py
│   │   │   │   ├── test_proj.py
│   │   │   │   └── test_pwcpw.py
│   │   │   ├── __init__.py
│   │   │   ├── chkfile.py
│   │   │   ├── jk.py
│   │   │   ├── kccsd_rhf.py
│   │   │   ├── khf.py
│   │   │   ├── kmp2.py
│   │   │   ├── kpt_symm.py
│   │   │   ├── krks.py
│   │   │   ├── kuhf.py
│   │   │   ├── kuks.py
│   │   │   ├── kump2.py
│   │   │   ├── ncpp_cell.py
│   │   │   ├── pseudo.py
│   │   │   ├── pw_helper.py
│   │   │   ├── smearing.py
│   │   │   └── upf.py
│   │   └── scf/
│   │       ├── test/
│   │       │   └── test_khf_stagger.py
│   │       └── khf_stagger.py
│   ├── pprpa/
│   │   ├── tests/
│   │   │   └── test_rpprpa.py
│   │   ├── __init__.py
│   │   ├── rpprpa_davidson.py
│   │   ├── rpprpa_direct.py
│   │   └── upprpa_direct.py
│   ├── prop/
│   │   ├── dip_moment/
│   │   │   ├── test/
│   │   │   │   ├── h2co_tpbe66_631g_edipole_num.npy
│   │   │   │   ├── test_cmspdft_pdm.py
│   │   │   │   ├── test_lpdft_pdm.py
│   │   │   │   ├── test_mcpdft_pdm.py
│   │   │   │   └── test_sapdft_pdm.py
│   │   │   ├── lpdft.py
│   │   │   ├── mcpdft.py
│   │   │   └── mspdft.py
│   │   └── trans_dip_moment/
│   │       ├── test/
│   │       │   └── test_cmspdft_tdm.py
│   │       └── mspdft.py
│   ├── pv/
│   │   ├── test/
│   │   │   └── test_energy-pv.py
│   │   └── energy.py
│   ├── scf/
│   │   └── fomoscf.py
│   ├── sfnoci/
│   │   ├── test/
│   │   │   └── test_sfnoci.py
│   │   ├── direct_sfnoci.py
│   │   └── sfnoci.py
│   ├── sftda/
│   │   ├── test/
│   │   │   ├── test_sftda_roks.py
│   │   │   ├── test_sftda.py
│   │   │   ├── test_sftddft_roks.py
│   │   │   └── test_sftddft.py
│   │   ├── __init__.py
│   │   ├── derivations_satda.md
│   │   ├── numint2c_sftd.py
│   │   ├── SASFTDA_Rev(2).pdf
│   │   ├── satda.py
│   │   ├── scf_genrep_sftd.py
│   │   ├── uhf_sf.py
│   │   └── uks_sf.py
│   ├── soscf/
│   │   ├── test/
│   │   │   └── test_m3soscf.py
│   │   ├── m3soscf.py
│   │   └── sigma_utils.py
│   ├── tdscf/
│   │   ├── test/
│   │   │   ├── test_krylov.py
│   │   │   └── test_ris.py
│   │   ├── _krylov_tools.py
│   │   ├── math_helper.py
│   │   ├── parameter.py
│   │   ├── ris.py
│   │   └── spectralib.py
│   └── tools/
│       ├── test/
│       │   └── test_trexio.py
│       └── trexio.py
├── .flake8
├── .gitignore
├── .ruff.toml
├── AGENTS.md
├── CHANGELOG
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── NOTICE
├── pyproject.toml
├── README.md
└── setup.py
```

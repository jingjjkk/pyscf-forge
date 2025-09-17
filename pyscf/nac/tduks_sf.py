import copy
import time
start_time = time.time()
from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.dft import numint
from pyscf.dft import numint2c
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.sftda.numint2c_sftd import cache_xc_kernel_sf
from pyscf import grad

def get_Hellmann_Feymann(td_grad, x_y_I, x_y_J, atmlst=None, max_memory=6000, verbose=logger.INFO):
    '''
    Compute the Hellmann-Feynman (HF) contribution to non-adiabatic coupling vectors (NACVs)
    between two spin-flip excited states, using the auxiliary wavefunction formalism.

    This term is also known as the "CI term" in NAC literature. It represents the direct
    derivative coupling arising from the parametric dependence of the Hamiltonian on
    nuclear coordinates, evaluated via the Hellmann-Feynman theorem.

    The returned NACV is post-corrected with ETF (energy-based translational fix) to ensure
    translational invariance and physical consistency.

    Args:
        td_grad (pyscf.grad.tdrks.TDGradients or similar):
            Gradient object associated with SF-TDA or SF-TDDFT calculation.
            Must contain converged excitation vectors and molecular information.
        x_y_I (tuple of numpy.ndarray):
            Excitation coefficients (X, Y) for the I-th excited state from SF-TDDFT.
        x_y_J (tuple of numpy.ndarray):
            Excitation coefficients (X, Y) for the J-th excited state.
        atmlst (list of int, optional):
            List of atom indices to compute gradients for. If None, all atoms are included.
            Default: None.
        max_memory (float, optional):
            Maximum memory (in MB) allowed for intermediate arrays. Default: 6000.
        verbose (int or pyscf.lib.logger.Logger, optional):
            Verbosity level. Default: logger.INFO.

    Returns:
        numpy.ndarray:
            Hellmann-Feynman contribution to the NAC vector between states I and J.
            Shape: (natm, 3), in atomic units.
            The vector is ETF-corrected to remove translational/rotational artifacts.

    Notes:
        - This function assumes the use of spin-flip (SF) excitation vectors, avilable for both TDA and TDDFT.
        - For full NAC, combine this term with the "wavefunction overlap term".
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0]>0)[0]
    occidxb = np.where(mo_occ[1]>0)[0]
    viridxa = np.where(mo_occ[0]==0)[0]
    viridxb = np.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]

    nmoa = nocca + nvira
    nmob = noccb + nvirb

    if td_grad.base.extype==0 or 1:
        
        (x_ab_I, x_ba_I), (y_ab_I, y_ba_I) = x_y_I
        (x_ab_J, x_ba_J), (y_ab_J, y_ba_J) = x_y_J
        xpy_ab_I = (x_ab_I + y_ab_I).T
        xpy_ba_I = (x_ba_I + y_ba_I).T
        xmy_ab_I = (x_ab_I - y_ab_I).T
        xmy_ba_I = (x_ba_I - y_ba_I).T
        xpy_ab_J = (x_ab_J + y_ab_J).T
        xpy_ba_J = (x_ba_J + y_ba_J).T
        xmy_ab_J = (x_ab_J - y_ab_J).T
        xmy_ba_J = (x_ba_J - y_ba_J).T
    

        dvv_a_IJ = np.einsum('ai,bi->ab', xpy_ab_I, xpy_ab_J) + np.einsum('ai,bi->ab', xmy_ab_I, xmy_ab_J) # T_IJ^{ab \alpha \beta}*2
        dvv_b_IJ = np.einsum('ai,bi->ab', xpy_ba_I, xpy_ba_J) + np.einsum('ai,bi->ab', xmy_ba_I, xmy_ba_J) # T_IJ^{ab \beta \alpha}*2
        doo_a_IJ =-np.einsum('ai,aj->ij', xpy_ba_I, xpy_ba_J) - np.einsum('ai,aj->ij', xmy_ba_I, xmy_ba_J) # T_IJ^{ij \alpha \beta}*2
        doo_b_IJ =-np.einsum('ai,aj->ij', xpy_ab_I, xpy_ab_J) - np.einsum('ai,aj->ij', xmy_ab_I, xmy_ab_J) # T_JI^{ij \beta \alpha}*2
        dvv_a_JI = np.einsum('ai,bi->ab', xpy_ab_J, xpy_ab_I) + np.einsum('ai,bi->ab', xmy_ab_J, xmy_ab_I) # T_JI^{ab \alpha \beta}*2
        dvv_b_JI = np.einsum('ai,bi->ab', xpy_ba_J, xpy_ba_I) + np.einsum('ai,bi->ab', xmy_ba_J, xmy_ba_I) # T_JI^{ab \beta \alpha}*2
        doo_a_JI =-np.einsum('ai,aj->ij', xpy_ba_J, xpy_ba_I) - np.einsum('ai,aj->ij', xmy_ba_J, xmy_ba_I) # T_JI^{ij \alpha \beta}*2
        doo_b_JI =-np.einsum('ai,aj->ij', xpy_ab_J, xpy_ab_I) - np.einsum('ai,aj->ij', xmy_ab_J, xmy_ab_I) # T_JI^{ij \beta \alpha}*2
        

        dmxpy_ab_I = reduce(np.dot, (orbva, xpy_ab_I, orbob.T)) # ua ai iv -> uv -> (X+Y)I_{uv \alpha \beta}
        dmxpy_ba_I = reduce(np.dot, (orbvb, xpy_ba_I, orboa.T)) # ua ai iv -> uv -> (X+Y)I_{uv \beta \alpha}
        dmxpy_ab_J = reduce(np.dot, (orbva, xpy_ab_J, orbob.T)) # ua ai iv -> uv -> (X+Y)J_{uv \alpha \beta}
        dmxpy_ba_J = reduce(np.dot, (orbvb, xpy_ba_J, orboa.T)) # ua ai iv -> uv -> (X+Y)J_{uv \beta \alpha}
        dmxmy_ab_I = reduce(np.dot, (orbva, xmy_ab_I, orbob.T)) # ua ai iv -> uv -> (X-Y)I_{uv \alpha \beta}
        dmxmy_ba_I = reduce(np.dot, (orbvb, xmy_ba_I, orboa.T)) # ua ai iv -> uv -> (X-Y)I_{uv \beta \alpha}
        dmxmy_ab_J = reduce(np.dot, (orbva, xmy_ab_J, orbob.T)) # ua ai iv -> uv -> (X-Y)J_{uv \alpha \beta}
        dmxmy_ba_J = reduce(np.dot, (orbvb, xmy_ba_J, orboa.T)) # ua ai iv -> uv -> (X-Y)J_{uv \beta \alpha}


        dmzoo_a_IJ = reduce(np.dot, (orboa, doo_a_IJ, orboa.T)) 
        dmzoo_b_IJ = reduce(np.dot, (orbob, doo_b_IJ, orbob.T)) 
        dmzoo_a_JI = reduce(np.dot, (orboa, doo_a_JI, orboa.T)) 
        dmzoo_b_JI = reduce(np.dot, (orbob, doo_b_JI, orbob.T)) 
        dmzoo_a_IJ+= reduce(np.dot, (orbva, dvv_a_IJ, orbva.T))
        dmzoo_b_IJ+= reduce(np.dot, (orbvb, dvv_b_IJ, orbvb.T)) 
        dmzoo_a_JI+= reduce(np.dot, (orbva, dvv_a_JI, orbva.T)) 
        dmzoo_b_JI+= reduce(np.dot, (orbvb, dvv_b_JI, orbvb.T)) 

    
        dmzoo_a = (dmzoo_a_IJ + dmzoo_a_JI)*0.5
        dmzoo_b = (dmzoo_b_IJ + dmzoo_b_JI)*0.5
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        # used by mcfun.
        rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        mo_coeff, mo_occ, spin=1)

        f1vo_I, f1oo_IJ, vxc1, k1ao = \
                _contract_xc_kernel(td_grad, mf.xc, ((dmxpy_ab_I,dmxpy_ba_I),(dmxmy_ab_I,dmxmy_ba_I)),((dmxpy_ab_J,dmxpy_ba_J),(dmxmy_ab_J,dmxmy_ba_J)),
                                    (dmzoo_a_IJ,dmzoo_b_IJ), True, True, max_memory)
        f1vo_J, f1oo_JI, _, _ = \
                _contract_xc_kernel(td_grad, mf.xc, ((dmxpy_ab_J,dmxpy_ba_J),(dmxmy_ab_J,dmxmy_ba_J)),((dmxpy_ab_I,dmxpy_ba_I),(dmxmy_ab_I,dmxmy_ba_I)),
                                    (dmzoo_a_JI,dmzoo_b_JI), False, False, max_memory)
        k1ao_xpy, k1ao_xmy = k1ao

        # f1vo, (2,2,4,nao,nao), (X+Y) and (X-Y) with fxc_sf
        # f1oo, (2,4,nao,nao), 2T with fxc_sc
        # vxc1, ao with v1^{\sigma}
        # k1ao_xpy，(2,2,4,nao,nao), (X+Y)(X+Y) and (X-Y)(X-Y) with gxc

        if abs(hyb) > 1e-10:
            dm = (dmzoo_a, dmxpy_ba_I+dmxpy_ab_I.T,dmxpy_ba_J+dmxpy_ab_J.T, dmxmy_ba_I-dmxmy_ab_I.T,dmxmy_ba_J-dmxmy_ab_J.T,
                  dmzoo_b, dmxpy_ab_I+dmxpy_ba_I.T,dmxpy_ab_J+dmxpy_ba_J.T, dmxmy_ab_I-dmxmy_ba_I.T, dmxmy_ab_J-dmxmy_ba_J.T) 
            vj, vk = mf.get_jk(mol, dm, hermi=0)
            vk *= hyb
            if abs(omega) > 1e-10:
                vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
            vj = vj.reshape(2,5,nao,nao)
            vk = vk.reshape(2,5,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] - vk[:,0]+ 0.5*(f1oo_IJ[:,0]+f1oo_JI[:,0])
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff_I = - vk[:,1] + f1vo_I[0,:,0]
            veff_J = - vk[:,2] + f1vo_J[0,:,0]
            veff0mop_ba_I = reduce(np.dot, (mo_coeff[1].T, veff_I[0], mo_coeff[0]))
            veff0mop_ab_I = reduce(np.dot, (mo_coeff[0].T, veff_I[1], mo_coeff[1]))
            veff0mop_ba_J = reduce(np.dot, (mo_coeff[1].T, veff_J[0], mo_coeff[0]))
            veff0mop_ab_J = reduce(np.dot, (mo_coeff[0].T, veff_J[1], mo_coeff[1]))
            

            wvoa += np.einsum('ca,ci->ai', veff0mop_ba_I[noccb:,nocca:], xpy_ba_J)
            wvoa += np.einsum('ca,ci->ai', veff0mop_ba_J[noccb:,nocca:], xpy_ba_I)
            wvob += np.einsum('ca,ci->ai', veff0mop_ab_I[nocca:,noccb:], xpy_ab_J)
            wvob += np.einsum('ca,ci->ai', veff0mop_ab_J[nocca:,noccb:], xpy_ab_I)
           

            wvoa -= np.einsum('il,al->ai', veff0mop_ab_I[:nocca,:noccb], xpy_ab_J)
            wvoa -= np.einsum('il,al->ai', veff0mop_ab_J[:nocca,:noccb], xpy_ab_I)
            wvob -= np.einsum('il,al->ai', veff0mop_ba_I[:noccb,:nocca], xpy_ba_J)
            wvob -= np.einsum('il,al->ai', veff0mop_ba_J[:noccb,:nocca], xpy_ba_I)

            

            veff_I = -vk[:,3] + f1vo_I[1,:,0]
            veff_J = -vk[:,4] + f1vo_J[1,:,0]

            veff0mom_ba_I = reduce(np.dot, (mo_coeff[1].T, veff_I[0], mo_coeff[0]))
            veff0mom_ab_I = reduce(np.dot, (mo_coeff[0].T, veff_I[1], mo_coeff[1]))
            veff0mom_ba_J = reduce(np.dot, (mo_coeff[1].T, veff_J[0], mo_coeff[0]))
            veff0mom_ab_J = reduce(np.dot, (mo_coeff[0].T, veff_J[1], mo_coeff[1]))

            wvoa += np.einsum('ca,ci->ai', veff0mom_ba_I[noccb:,nocca:], xmy_ba_J)
            wvoa += np.einsum('ca,ci->ai', veff0mom_ba_J[noccb:,nocca:], xmy_ba_I)
            wvob += np.einsum('ca,ci->ai', veff0mom_ab_I[nocca:,noccb:], xmy_ab_J)
            wvob += np.einsum('ca,ci->ai', veff0mom_ab_J[nocca:,noccb:], xmy_ab_I)

            

            wvoa -= np.einsum('il,al->ai', veff0mom_ab_I[:nocca,:noccb], xmy_ab_J)
            wvoa -= np.einsum('il,al->ai', veff0mom_ab_J[:nocca,:noccb], xmy_ab_I)
            wvob -= np.einsum('il,al->ai', veff0mom_ba_I[:noccb,:nocca], xmy_ba_J)
            wvob -= np.einsum('il,al->ai', veff0mom_ba_J[:noccb,:nocca], xmy_ba_I)

        

        else:
            dm = (dmzoo_a, dmxpy_ba_I+dmxpy_ab_I.T,dmxpy_ba_J+dmxpy_ab_J.T, dmxmy_ba_I-dmxmy_ab_I.T, dmxmy_ba_J-dmxmy_ab_J.T,    
                  dmzoo_b, dmxpy_ab_I+dmxpy_ba_I.T,dmxpy_ab_J+dmxpy_ba_J.T, dmxmy_ab_I-dmxmy_ba_I.T, dmxmy_ab_J-dmxmy_ba_J.T)
            vj = mf.get_j(mol, dm, hermi=0).reshape(2,5,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] + 0.5*(f1oo_IJ[:,0]+ f1oo_JI[:,0])
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff_I = f1vo_I[0,:,0]
            veff_J = f1vo_J[0,:,0]
            veff0mop_ba_I = reduce(np.dot, (mo_coeff[1].T, veff_I[0], mo_coeff[0]))
            veff0mop_ab_I = reduce(np.dot, (mo_coeff[0].T, veff_I[1], mo_coeff[1]))
            veff0mop_ba_J = reduce(np.dot, (mo_coeff[1].T, veff_J[0], mo_coeff[0]))
            veff0mop_ab_J = reduce(np.dot, (mo_coeff[0].T, veff_J[1], mo_coeff[1]))

            

            wvoa += np.einsum('ca,ci->ai', veff0mop_ba_I[noccb:,nocca:], xpy_ba_J)
            wvoa += np.einsum('ca,ci->ai', veff0mop_ba_J[noccb:,nocca:], xpy_ba_I)
            wvob += np.einsum('ca,ci->ai', veff0mop_ab_I[nocca:,noccb:], xpy_ab_J)
            wvob += np.einsum('ca,ci->ai', veff0mop_ab_J[nocca:,noccb:], xpy_ab_I)


            wvoa -= np.einsum('il,al->ai', veff0mop_ab_I[:nocca,:noccb], xpy_ab_J)
            wvoa -= np.einsum('il,al->ai', veff0mop_ab_J[:nocca,:noccb], xpy_ab_I)
            wvob -= np.einsum('il,al->ai', veff0mop_ba_I[:noccb,:nocca], xpy_ba_J)
            wvob -= np.einsum('il,al->ai', veff0mop_ba_J[:noccb,:nocca], xpy_ba_I)

          

            veff_I = f1vo_I[1,:,0]
            veff_J = f1vo_J[1,:,0]
            veff0mom_ba_I = reduce(np.dot, (mo_coeff[1].T, veff_I[0], mo_coeff[0]))
            veff0mom_ab_I = reduce(np.dot, (mo_coeff[0].T, veff_I[1], mo_coeff[1]))
            veff0mom_ba_J = reduce(np.dot, (mo_coeff[1].T, veff_J[0], mo_coeff[0]))
            veff0mom_ab_J = reduce(np.dot, (mo_coeff[0].T, veff_J[1], mo_coeff[1]))
      

            wvoa += np.einsum('ca,ci->ai', veff0mom_ba_I[noccb:,nocca:], xmy_ba_J) 
            wvoa += np.einsum('ca,ci->ai', veff0mom_ba_J[noccb:,nocca:], xmy_ba_I)
            wvob += np.einsum('ca,ci->ai', veff0mom_ab_I[nocca:,noccb:], xmy_ab_J)
            wvob += np.einsum('ca,ci->ai', veff0mom_ab_J[nocca:,noccb:], xmy_ab_I)


            wvoa -= np.einsum('il,al->ai', veff0mom_ab_I[:nocca,:noccb], xmy_ab_J) 
            wvoa -= np.einsum('il,al->ai', veff0mom_ab_J[:nocca,:noccb], xmy_ab_I)
            wvob -= np.einsum('il,al->ai', veff0mom_ba_I[:noccb,:nocca], xmy_ba_J)
            wvob -= np.einsum('il,al->ai', veff0mom_ba_J[:noccb,:nocca], xmy_ba_I)


    vresp = mf.gen_response(hermi=1)

    def fvind(x):
        dm1 = np.empty((2,nao,nao))
        x_a = x[0,:nvira*nocca].reshape(nvira,nocca)
        x_b = x[0,nvira*nocca:].reshape(nvirb,noccb)
        dm_a = reduce(np.dot, (orbva, x_a, orboa.T))
        dm_b = reduce(np.dot, (orbvb, x_b, orbob.T))
        dm1[0] = (dm_a + dm_a.T).real
        dm1[1] = (dm_b + dm_b.T).real

        v1 = vresp(dm1)
        v1a = reduce(np.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(np.dot, (orbvb.T, v1[1], orbob))
        return np.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa,wvob), #z-vector for UCPHF
                           max_cycle=td_grad.cphf_max_cycle,
                           tol=td_grad.cphf_conv_tol)[0]

    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = np.zeros((2,nao,nao))
    z1ao[0] += reduce(np.dot, (orbva, z1a, orboa.T))
    z1ao[1] += reduce(np.dot, (orbvb, z1b, orbob.T))

    veff = vresp((z1ao+z1ao.transpose(0,2,1))*0.5)

    im0a = np.zeros((nmoa,nmoa))
    im0b = np.zeros((nmob,nmob))

    im0a[:nocca,:nocca] = reduce(np.dot, (orboa.T, veff0doo[0]+veff[0], orboa)) *.5
    im0b[:noccb,:noccb] = reduce(np.dot, (orbob.T, veff0doo[1]+veff[1], orbob)) *.5

    im0a[:nocca,:nocca] += np.einsum('aj,ai->ij', veff0mop_ba_I[noccb:,:nocca], xpy_ba_J) *0.25
    im0a[:nocca,:nocca] += np.einsum('aj,ai->ij', veff0mop_ba_J[noccb:,:nocca], xpy_ba_I) *0.25

    im0b[:noccb,:noccb] += np.einsum('aj,ai->ij', veff0mop_ab_I[nocca:,:noccb], xpy_ab_J) *0.25
    im0b[:noccb,:noccb] += np.einsum('aj,ai->ij', veff0mop_ab_J[nocca:,:noccb], xpy_ab_I) *0.25

    im0a[:nocca,:nocca] += np.einsum('aj,ai->ij', veff0mom_ba_I[noccb:,:nocca], xmy_ba_J) *0.25
    im0a[:nocca,:nocca] += np.einsum('aj,ai->ij', veff0mom_ba_J[noccb:,:nocca], xmy_ba_I) *0.25

    im0b[:noccb,:noccb] += np.einsum('aj,ai->ij', veff0mom_ab_I[nocca:,:noccb], xmy_ab_J) *0.25
    im0b[:noccb,:noccb] += np.einsum('aj,ai->ij', veff0mom_ab_J[nocca:,:noccb], xmy_ab_I) *0.25

    im0a[nocca:,nocca:]  = np.einsum('bi,ai->ab', veff0mop_ab_I[nocca:,:noccb], xpy_ab_J) *0.25
    im0a[nocca:,nocca:] += np.einsum('bi,ai->ab', veff0mop_ab_J[nocca:,:noccb], xpy_ab_I) *0.25

    im0b[noccb:,noccb:]  = np.einsum('bi,ai->ab', veff0mop_ba_I[noccb:,:nocca], xpy_ba_J) *0.25
    im0b[noccb:,noccb:] += np.einsum('bi,ai->ab', veff0mop_ba_J[noccb:,:nocca], xpy_ba_I) *0.25

    im0a[nocca:,nocca:] += np.einsum('bi,ai->ab', veff0mom_ab_I[nocca:,:noccb], xmy_ab_J) *0.25
    im0a[nocca:,nocca:] += np.einsum('bi,ai->ab', veff0mom_ab_J[nocca:,:noccb], xmy_ab_I) *0.25

    im0b[noccb:,noccb:] += np.einsum('bi,ai->ab', veff0mom_ba_I[noccb:,:nocca], xmy_ba_J) *0.25
    im0b[noccb:,noccb:] += np.einsum('bi,ai->ab', veff0mom_ba_J[noccb:,:nocca], xmy_ba_I) *0.25

    im0a[nocca:,:nocca]  = np.einsum('il,al->ai', veff0mop_ab_I[:nocca,:noccb], xpy_ab_J)*0.5
    im0a[nocca:,:nocca] += np.einsum('il,al->ai', veff0mop_ab_J[:nocca,:noccb], xpy_ab_I)*0.5

    im0b[noccb:,:noccb]  = np.einsum('il,al->ai', veff0mop_ba_I[:noccb,:nocca], xpy_ba_J)*0.5
    im0b[noccb:,:noccb] += np.einsum('il,al->ai', veff0mop_ba_J[:noccb,:nocca], xpy_ba_I)*0.5

    im0a[nocca:,:nocca] += np.einsum('il,al->ai', veff0mom_ab_I[:nocca,:noccb], xmy_ab_J)*0.5
    im0a[nocca:,:nocca] += np.einsum('il,al->ai', veff0mom_ab_J[:nocca,:noccb], xmy_ab_I)*0.5

    im0b[noccb:,:noccb] += np.einsum('il,al->ai', veff0mom_ba_I[:noccb,:nocca], xmy_ba_J)*0.5
    im0b[noccb:,:noccb] += np.einsum('il,al->ai', veff0mom_ba_J[:noccb,:nocca], xmy_ba_I)*0.5


    

    zeta_a = (mo_energy[0][:,None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:,None] + mo_energy[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca,nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb,noccb:] = mo_energy[1][noccb:]

    dm1a = np.zeros((nmoa,nmoa))
    dm1b = np.zeros((nmob,nmob))
    dm1a[:nocca,:nocca] = (doo_a_IJ+doo_a_JI) * 0.25
    dm1b[:noccb,:noccb] = (doo_b_IJ+doo_b_JI) * 0.25
    dm1a[nocca:,nocca:] = (dvv_a_IJ+dvv_a_JI) * 0.25
    dm1b[noccb:,noccb:] = (dvv_b_IJ+dvv_b_JI) * 0.25

    dm1a[nocca:,:nocca] = z1a *.5
    dm1b[noccb:,:noccb] = z1b *.5


    im0a = reduce(np.dot, (mo_coeff[0], im0a+zeta_a*dm1a, mo_coeff[0].T))
    im0b = reduce(np.dot, (mo_coeff[1], im0b+zeta_b*dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = mf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)

    # -mol.intor('int1e_ipovlp', comp=3)
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo_a = z1ao[0] + dmzoo_a_IJ
    dmz1doo_b = z1ao[1] + dmzoo_b_IJ
    oo0a = reduce(np.dot, (orboa, orboa.T))
    oo0b = reduce(np.dot, (orbob, orbob.T))

    as_dm1 =  (dmz1doo_a + dmz1doo_b) * .5

    if abs(hyb) > 1e-10:
        dm = (oo0a, dmz1doo_a+dmz1doo_a.T, dmxpy_ba_I+dmxpy_ab_I.T, dmxpy_ba_J+dmxpy_ab_J.T , dmxmy_ba_I-dmxmy_ab_I.T, dmxmy_ba_J-dmxmy_ab_J.T,
              oo0b, dmz1doo_b+dmz1doo_b.T, dmxpy_ab_I+dmxpy_ba_I.T, dmxpy_ab_J+dmxpy_ba_J.T,  dmxmy_ab_I-dmxmy_ba_I.T, dmxmy_ab_J-dmxmy_ba_J.T)
        vj, vk = td_grad.get_jk(mol, dm)
        vj = vj.reshape(2,6,3,nao,nao)
        vk = vk.reshape(2,6,3,nao,nao) * hyb
        vj[:,2:6] *= 0.0
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dm).reshape(2,6,3,nao,nao) * (alpha-hyb)

        veff1 = np.zeros((2,6,3,nao,nao))
        veff1[:,:2] = vj[0,:2] + vj[1,:2] - vk[:,:2]
    else:
        dm = (oo0a, dmz1doo_a+dmz1doo_a.T, dmxpy_ba_I+dmxpy_ab_I.T,dmxpy_ba_J+dmxpy_ab_J.T, 
              oo0b, dmz1doo_b+dmz1doo_b.T, dmxpy_ab_I+dmxpy_ba_I.T,dmxpy_ab_J+dmxpy_ba_J.T,)
        vj = td_grad.get_j(mol, dm).reshape(2,4,3,nao,nao)
        vj[:,2:4] *= 0.0
        veff1 = np.zeros((2,6,3,nao,nao))
        veff1[:,:4] = vj[0] + vj[1]

    fxcz1 = _contract_xc_kernel_z(td_grad, mf.xc, z1ao, max_memory)

    veff1[:,0] += vxc1[:,1:]
    veff1[:,1] += (f1oo_IJ[:,1:] + fxcz1[:,1:])*2
    veff1[0,1] += (k1ao_xpy[0,0,1:] + k1ao_xpy[0,1,1:] + k1ao_xpy[1,0,1:] + k1ao_xpy[1,1,1:]
                  +k1ao_xmy[0,0,1:] + k1ao_xmy[0,1,1:] + k1ao_xmy[1,0,1:] + k1ao_xmy[1,1,1:])*2
    veff1[1,1] += (k1ao_xpy[0,0,1:] + k1ao_xpy[0,1,1:] - k1ao_xpy[1,0,1:] - k1ao_xpy[1,1,1:]
                  +k1ao_xmy[0,0,1:] + k1ao_xmy[0,1,1:] - k1ao_xmy[1,0,1:] - k1ao_xmy[1,1,1:])*2

    veff1[:,2] += f1vo_I[0,:,1:]
    veff1[:,3] += f1vo_J[0,:,1:]
    veff1[:,4] += f1vo_I[1,:,1:]
    veff1[:,5] += f1vo_J[1,:,1:]
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    nac = np.zeros((len(atmlst),3))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)
        nac[k] = np.einsum('xpq,pq->x', h1ao, as_dm1)
        

        nac[k] += np.einsum('xpq,pq->x', veff1a[0,:,p0:p1], dmz1doo_a[p0:p1]) *.5
        nac[k] += np.einsum('xpq,pq->x', veff1b[0,:,p0:p1], dmz1doo_b[p0:p1]) *.5
        nac[k] += np.einsum('xpq,qp->x', veff1a[0,:,p0:p1], dmz1doo_a[:,p0:p1]) *.5
        nac[k] += np.einsum('xpq,qp->x', veff1b[0,:,p0:p1], dmz1doo_b[:,p0:p1]) *.5

        nac[k] -= np.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        nac[k] -= np.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        nac[k] += np.einsum('xij,ij->x', veff1a[1,:,p0:p1], oo0a[p0:p1])*0.5
        nac[k] += np.einsum('xij,ij->x', veff1b[1,:,p0:p1], oo0b[p0:p1])*0.5

        nac[k] += np.einsum('xij,ij->x', veff1b[2,:,p0:p1], dmxpy_ab_J[p0:p1,:])*0.5
        nac[k] += np.einsum('xij,ij->x', veff1b[3,:,p0:p1], dmxpy_ab_I[p0:p1,:])*0.5

        nac[k] += np.einsum('xij,ij->x', veff1a[2,:,p0:p1], dmxpy_ba_J[p0:p1,:])*0.5
        nac[k] += np.einsum('xij,ij->x', veff1a[3,:,p0:p1], dmxpy_ba_I[p0:p1,:])*0.5

        nac[k] += np.einsum('xji,ij->x', veff1b[2,:,p0:p1], dmxpy_ab_J[:,p0:p1])*0.5
        nac[k] += np.einsum('xji,ij->x', veff1b[3,:,p0:p1], dmxpy_ab_I[:,p0:p1])*0.5

        nac[k] += np.einsum('xji,ij->x', veff1a[2,:,p0:p1], dmxpy_ba_J[:,p0:p1])*0.5
        nac[k] += np.einsum('xji,ij->x', veff1a[3,:,p0:p1], dmxpy_ba_I[:,p0:p1])*0.5

        nac[k] += np.einsum('xij,ij->x', veff1b[4,:,p0:p1], dmxmy_ab_J[p0:p1,:])*0.5
        nac[k] += np.einsum('xij,ij->x', veff1b[5,:,p0:p1], dmxmy_ab_I[p0:p1,:])*0.5

        nac[k] += np.einsum('xij,ij->x', veff1a[4,:,p0:p1], dmxmy_ba_J[p0:p1,:])*0.5
        nac[k] += np.einsum('xij,ij->x', veff1a[5,:,p0:p1], dmxmy_ba_I[p0:p1,:])*0.5

        nac[k] += np.einsum('xji,ij->x', veff1b[4,:,p0:p1], dmxmy_ab_J[:,p0:p1])*0.5
        nac[k] += np.einsum('xji,ij->x', veff1b[5,:,p0:p1], dmxmy_ab_I[:,p0:p1])*0.5

        nac[k] += np.einsum('xji,ij->x', veff1a[4,:,p0:p1], dmxmy_ba_J[:,p0:p1])*0.5
        nac[k] += np.einsum('xji,ij->x', veff1a[5,:,p0:p1], dmxmy_ba_I[:,p0:p1])*0.5

        if abs(hyb) > 1e-10:
            nac[k] -= np.einsum('xij,ij->x', vk[1,2,:,p0:p1], dmxpy_ab_J[p0:p1,:])*0.5
            nac[k] -= np.einsum('xij,ij->x', vk[1,3,:,p0:p1], dmxpy_ab_I[p0:p1,:])*0.5

            nac[k] -= np.einsum('xij,ij->x', vk[0,2,:,p0:p1], dmxpy_ba_J[p0:p1,:])*0.5
            nac[k] -= np.einsum('xij,ij->x', vk[0,3,:,p0:p1], dmxpy_ba_I[p0:p1,:])*0.5

            nac[k] -= np.einsum('xji,ij->x', vk[0,2,:,p0:p1], dmxpy_ab_J[:,p0:p1])*0.5
            nac[k] -= np.einsum('xji,ij->x', vk[0,3,:,p0:p1], dmxpy_ab_I[:,p0:p1])*0.5
            nac[k] -= np.einsum('xji,ij->x', vk[1,2,:,p0:p1], dmxpy_ba_J[:,p0:p1])*0.5
            nac[k] -= np.einsum('xji,ij->x', vk[1,3,:,p0:p1], dmxpy_ba_I[:,p0:p1])*0.5

            nac[k] -= np.einsum('xij,ij->x', vk[1,4,:,p0:p1], dmxmy_ab_J[p0:p1,:])*0.5
            nac[k] -= np.einsum('xij,ij->x', vk[1,5,:,p0:p1], dmxmy_ab_I[p0:p1,:])*0.5

            nac[k] -= np.einsum('xij,ij->x', vk[0,4,:,p0:p1], dmxmy_ba_J[p0:p1,:])*0.5
            nac[k] -= np.einsum('xij,ij->x', vk[0,5,:,p0:p1], dmxmy_ba_I[p0:p1,:])*0.5
            nac[k] += np.einsum('xji,ij->x', vk[0,4,:,p0:p1], dmxmy_ab_J[:,p0:p1])*0.5
            nac[k] += np.einsum('xji,ij->x', vk[0,5,:,p0:p1], dmxmy_ab_I[:,p0:p1])*0.5
            nac[k] += np.einsum('xji,ij->x', vk[1,4,:,p0:p1], dmxmy_ba_J[:,p0:p1])*0.5
            nac[k] += np.einsum('xji,ij->x', vk[1,5,:,p0:p1], dmxmy_ba_I[:,p0:p1])*0.5

        
    log.timer('TDUKS nuclear gradients', *time0)
    return nac


def _contract_xc_kernel(td_grad, xc_code, dmvo_I, dmvo_J, dmoo=None, with_vxc=True,
                        with_kxc=True, max_memory=6000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    f1vo = np.zeros((2,2,4,nao,nao))
    deriv = 2

    if dmoo is not None:
        f1oo = np.zeros((2,4,nao,nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((2,4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao_xpy = np.zeros((2,2,4,nao,nao))
        k1ao_xmy = np.zeros((2,2,4,nao,nao))
        deriv = 3
    else:
        k1ao_xpy = k1ao_xmy = None

    nimc = numint2c.NumInt2C()
    nimc.collinear = 'mcol'
    nimc.collinear_samples=td_grad.base.collinear_samples

    fxc_sf,kxc_sf = cache_xc_kernel_sf(nimc,mol,mf.grids,mf.xc,mo_coeff,mo_occ,deriv=3,spin=1)[2:]
    p0,p1=0,0

    if xctype == 'LDA':
        def lda_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[0], wv)
            for k in range(4):
                vmat[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)

        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0 = p1
            p1+= weight.shape[0]
            s_s = fxc_sf[...,p0:p1] * weight

            rho1_ab = ni.eval_rho(mol, ao[0], dmvo_I[0][0], mask, xctype)
            rho1_ba = ni.eval_rho(mol, ao[0], dmvo_I[0][1], mask, xctype)
            lda_sum_(f1vo[0][1], ao, (rho1_ab+rho1_ba)*s_s*2, mask)
            lda_sum_(f1vo[0][0], ao, (rho1_ba+rho1_ab)*s_s*2, mask)

            if with_kxc:
                rho1_ab_J = ni.eval_rho(mol, ao[0], dmvo_J[0][0], mask, xctype)
                rho1_ba_J = ni.eval_rho(mol, ao[0], dmvo_J[0][1], mask, xctype)

                s_s_n = kxc_sf[:,:,0][...,p0:p1] * weight
                s_s_s = kxc_sf[:,:,1][...,p0:p1] * weight
                
                lda_sum_(k1ao_xpy[0][0], ao, s_s_n*2*rho1_ab*(rho1_ab_J+rho1_ba_J), mask)
                lda_sum_(k1ao_xpy[0][1], ao, s_s_n*2*rho1_ba*(rho1_ba_J+rho1_ab_J), mask)
                lda_sum_(k1ao_xpy[1][0], ao, s_s_s*2*rho1_ab*(rho1_ab_J+rho1_ba_J), mask)
                lda_sum_(k1ao_xpy[1][1], ao, s_s_s*2*rho1_ba*(rho1_ba_J+rho1_ab_J), mask)

            rho1_ab = ni.eval_rho(mol, ao[0], dmvo_I[1][0], mask, xctype)
            rho1_ba = ni.eval_rho(mol, ao[0], dmvo_I[1][1], mask, xctype)

            lda_sum_(f1vo[1][1], ao, (rho1_ab-rho1_ba)*s_s*2, mask)
            lda_sum_(f1vo[1][0], ao, (rho1_ba-rho1_ab)*s_s*2, mask)

            if with_kxc:
                rho1_ab_J = ni.eval_rho(mol, ao[0], dmvo_J[1][0], mask, xctype)
                rho1_ba_J = ni.eval_rho(mol, ao[0], dmvo_J[1][1], mask, xctype)
                
                s_s_n = kxc_sf[:,:,0][...,p0:p1] * weight # Re-slicing for clarity
                s_s_s = kxc_sf[:,:,1][...,p0:p1] * weight
                
                lda_sum_(k1ao_xmy[0][0], ao, s_s_n*2*rho1_ab*(rho1_ab_J-rho1_ba_J), mask)
                lda_sum_(k1ao_xmy[0][1], ao, s_s_n*2*rho1_ba*(rho1_ba_J-rho1_ab_J), mask)
                lda_sum_(k1ao_xmy[1][0], ao, s_s_s*2*rho1_ab*(rho1_ab_J-rho1_ba_J), mask)
                lda_sum_(k1ao_xmy[1][1], ao, s_s_s*2*rho1_ba*(rho1_ba_J-rho1_ab_J), mask)

            rho = (ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]
            u_u, u_d, d_d = fxc[0].T * weight
            if dmoo is not None:
                rho2a = ni.eval_rho(mol, ao[0], dmoo[0], mask, xctype, hermi=1)
                rho2b = ni.eval_rho(mol, ao[0], dmoo[1], mask, xctype, hermi=1)
                lda_sum_(f1oo[0], ao, u_u*rho2a+u_d*rho2b, mask)
                lda_sum_(f1oo[1], ao, u_d*rho2a+d_d*rho2b, mask)
            if with_vxc:
                vrho = vxc[0].T * weight
                lda_sum_(v1ao[0], ao, vrho[0], mask)
                lda_sum_(v1ao[1], ao, vrho[1], mask)

    elif xctype == 'GGA':
        def gga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T
            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv, mask, ao_loc)

        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0 = p1
            p1+= weight.shape[0]

            rho1_ab_I = ni.eval_rho(mol, ao, dmvo_I[0][0], mask, xctype)
            rho1_ba_I = ni.eval_rho(mol, ao, dmvo_I[0][1], mask, xctype)
            wv_sf = uks_sf_gga_wv1((rho1_ab_I,rho1_ba_I),fxc_sf[...,p0:p1],weight)
            gga_sum_(f1vo[0][1], ao, wv_sf[0]+wv_sf[1], mask)
            gga_sum_(f1vo[0][0], ao, wv_sf[1]+wv_sf[0], mask)

            if with_kxc:
                rho1_ab_J = ni.eval_rho(mol, ao, dmvo_J[0][0], mask, xctype)
                rho1_ba_J = ni.eval_rho(mol, ao, dmvo_J[0][1], mask, xctype)
                
                # Correcting the logic by direct einsum, as requested:
                rho_sum_J = rho1_ab_J + rho1_ba_J
                kxc_sf_slice = kxc_sf[...,p0:p1]
                gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab_I, rho_sum_J, kxc_sf_slice, optimize=True)
                gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba_I, rho_sum_J, kxc_sf_slice, optimize=True)
                # Apply scaling factors and weight
                gv_ab[0,1:] *= 2.0; gv_ab[1,1:] *= 2.0
                gv_ba[0,1:] *= 2.0; gv_ba[1,1:] *= 2.0
                gv_ab *= weight; gv_ba *= weight
                
                gga_sum_(k1ao_xpy[0][0], ao, gv_ab[0], mask)
                gga_sum_(k1ao_xpy[0][1], ao, gv_ba[0], mask) # Bug fix: original was gv_sf[1][0]
                gga_sum_(k1ao_xpy[1][0], ao, gv_ab[1], mask) # Bug fix: original was gv_sf[0][1]
                gga_sum_(k1ao_xpy[1][1], ao, gv_ba[1], mask) # Bug fix: original was gv_sf[1][1]

            rho1_ab_I = ni.eval_rho(mol, ao, dmvo_I[1][0], mask, xctype)
            rho1_ba_I = ni.eval_rho(mol, ao, dmvo_I[1][1], mask, xctype)
            wv_sf = uks_sf_gga_wv1((rho1_ab_I,rho1_ba_I),fxc_sf[...,p0:p1],weight)
            gga_sum_(f1vo[1][1], ao, wv_sf[0]-wv_sf[1], mask)
            gga_sum_(f1vo[1][0], ao, wv_sf[1]-wv_sf[0], mask)

            if with_kxc:
                rho1_ab_J = ni.eval_rho(mol, ao, dmvo_J[1][0], mask, xctype)
                rho1_ba_J = ni.eval_rho(mol, ao, dmvo_J[1][1], mask, xctype)
               
                rho_diff_J = rho1_ab_J - rho1_ba_J
                kxc_sf_slice = kxc_sf[...,p0:p1]
                gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab_I, rho_diff_J, kxc_sf_slice, optimize=True)
                gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba_I, -rho_diff_J, kxc_sf_slice, optimize=True)
                # Apply scaling factors and weight
                gv_ab[:,1:] *= 2.0; gv_ba[:,1:] *= 2.0
                gv_ab *= weight; gv_ba *= weight
                
                gga_sum_(k1ao_xmy[0][0], ao, gv_ab[0], mask)
                gga_sum_(k1ao_xmy[0][1], ao, gv_ba[0], mask)
                gga_sum_(k1ao_xmy[1][0], ao, gv_ab[1], mask)
                gga_sum_(k1ao_xmy[1][1], ao, gv_ba[1], mask)

            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]
            if dmoo is not None:
                rho2 = (ni.eval_rho(mol, ao, dmoo[0], mask, xctype, hermi=1),
                        ni.eval_rho(mol, ao, dmoo[1], mask, xctype, hermi=1))
                wv = numint._uks_gga_wv1(rho, rho2, vxc, fxc, weight)
                gga_sum_(f1oo[0], ao, wv[0], mask)
                gga_sum_(f1oo[1], ao, wv[1], mask)
            if with_vxc:
                wv = numint._uks_gga_wv0(rho, vxc, weight)
                gga_sum_(v1ao[0], ao, wv[0], mask)
                gga_sum_(v1ao[1], ao, wv[1], mask)

    elif xctype == 'MGGA':
        def mgga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

            # Tau part
            aow = numint._scale_ao(ao[1], wv[4], aow)
            tmp += numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[2], wv[4], aow)
            tmp += numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[3], wv[4], aow)
            tmp += numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T

            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv[:4], mask, ao_loc)
            rks_grad._tau_grad_dot_(vmat[1:], mol, ao, wv[4]*2, mask, ao_loc, True)

        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0 = p1
            p1+= weight.shape[0]
            ngrid=weight.shape[-1]
            
            # --- State I Densities ---
            rho1_ab_I_tmp = ni.eval_rho(mol, ao, dmvo_I[0][0], mask, xctype)
            rho1_ba_I_tmp = ni.eval_rho(mol, ao, dmvo_I[0][1], mask, xctype)
            rho1_ab_I = np.empty((5, ngrid)); rho1_ba_I = np.empty((5, ngrid))
            rho1_ab_I[:4] = rho1_ab_I_tmp[:4]; rho1_ba_I[:4] = rho1_ba_I_tmp[:4]
            rho1_ab_I[4] = rho1_ab_I_tmp[5]; rho1_ba_I[4] = rho1_ba_I_tmp[5]
            
            wv_sf = uks_sf_mgga_wv1((rho1_ab_I,rho1_ba_I), fxc_sf[...,p0:p1],weight)
            mgga_sum_(f1vo[0][1], ao, wv_sf[0]+wv_sf[1], mask)
            mgga_sum_(f1vo[0][0], ao, wv_sf[1]+wv_sf[0], mask)

            if with_kxc:
                # --- State J Densities ---
                rho1_ab_J_tmp = ni.eval_rho(mol, ao, dmvo_J[0][0], mask, xctype)
                rho1_ba_J_tmp = ni.eval_rho(mol, ao, dmvo_J[0][1], mask, xctype)
                rho1_ab_J = np.empty((5, ngrid)); rho1_ba_J = np.empty((5, ngrid))
                rho1_ab_J[:4] = rho1_ab_J_tmp[:4]; rho1_ba_J[:4] = rho1_ba_J_tmp[:4]
                rho1_ab_J[4] = rho1_ab_J_tmp[5]; rho1_ba_J[4] = rho1_ba_J_tmp[5]
                
                # Direct einsum for NAC
                rho_sum_J = rho1_ab_J + rho1_ba_J
                kxc_sf_slice = kxc_sf[...,p0:p1]
                gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab_I, rho_sum_J, kxc_sf_slice, optimize=True)
                gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba_I, rho_sum_J, kxc_sf_slice, optimize=True)
                # Apply scaling and weight
                gv_ab[:,1:4] *= 2.0; gv_ab[:,4] *= 0.5
                gv_ba[:,1:4] *= 2.0; gv_ba[:,4] *= 0.5
                gv_ab *= weight; gv_ba *= weight
                
                mgga_sum_(k1ao_xpy[0][0], ao, gv_ab[0], mask)
                mgga_sum_(k1ao_xpy[0][1], ao, gv_ba[0], mask)
                mgga_sum_(k1ao_xpy[1][0], ao, gv_ab[1], mask)
                mgga_sum_(k1ao_xpy[1][1], ao, gv_ba[1], mask)

            
            rho1_ab_I_y_tmp = ni.eval_rho(mol, ao, dmvo_I[1][0], mask, xctype)
            rho1_ba_I_y_tmp = ni.eval_rho(mol, ao, dmvo_I[1][1], mask, xctype)
            rho1_ab_I_y = np.empty((5, ngrid)); rho1_ba_I_y = np.empty((5, ngrid))
            rho1_ab_I_y[:4] = rho1_ab_I_y_tmp[:4]; rho1_ba_I_y[:4] = rho1_ba_I_y_tmp[:4]
            rho1_ab_I_y[4] = rho1_ab_I_y_tmp[5]; rho1_ba_I_y[4] = rho1_ba_I_y_tmp[5]
            
            wv_sf = uks_sf_mgga_wv1((rho1_ab_I_y,rho1_ba_I_y), fxc_sf[...,p0:p1],weight)
            mgga_sum_(f1vo[1][1], ao, wv_sf[0]-wv_sf[1], mask)
            mgga_sum_(f1vo[1][0], ao, wv_sf[1]-wv_sf[0], mask)

            if with_kxc:
                # --- State J Densities (imaginary part) ---
                rho1_ab_J_y_tmp = ni.eval_rho(mol, ao, dmvo_J[1][0], mask, xctype)
                rho1_ba_J_y_tmp = ni.eval_rho(mol, ao, dmvo_J[1][1], mask, xctype)
                rho1_ab_J_y = np.empty((5, ngrid)); rho1_ba_J_y = np.empty((5, ngrid))
                rho1_ab_J_y[:4] = rho1_ab_J_y_tmp[:4]; rho1_ba_J_y[:4] = rho1_ba_J_y_tmp[:4]
                rho1_ab_J_y[4] = rho1_ab_J_y_tmp[5]; rho1_ba_J_y[4] = rho1_ba_J_y_tmp[5]

                # Direct einsum for NAC
                rho_diff_J = rho1_ab_J_y - rho1_ba_J_y
                kxc_sf_slice = kxc_sf[...,p0:p1]
                gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab_I_y, rho_diff_J, kxc_sf_slice, optimize=True)
                gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba_I_y, -rho_diff_J, kxc_sf_slice, optimize=True)
                # Apply scaling and weight
                gv_ab[:,1:4] *= 2.0; gv_ab[:,4] *= 0.5
                gv_ba[:,1:4] *= 2.0; gv_ba[:,4] *= 0.5
                gv_ab *= weight; gv_ba *= weight

                mgga_sum_(k1ao_xmy[0][0], ao, gv_ab[0], mask)
                mgga_sum_(k1ao_xmy[0][1], ao, gv_ba[0], mask)
                mgga_sum_(k1ao_xmy[1][0], ao, gv_ab[1], mask)
                mgga_sum_(k1ao_xmy[1][1], ao, gv_ba[1], mask)

            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]

            if dmoo is not None:
                rho2 = (ni.eval_rho(mol, ao, dmoo[0], mask, xctype, hermi=1),
                        ni.eval_rho(mol, ao, dmoo[1], mask, xctype, hermi=1))
                wv_tmp = numint._uks_mgga_wv1(rho, rho2, vxc, fxc, weight)
                wv = np.empty((2,5,ngrid))
                wv[0][:4] = wv_tmp[0][:4]; wv[0][4]  = wv_tmp[0][5]
                wv[1][:4] = wv_tmp[1][:4]; wv[1][4]  = wv_tmp[1][5]
                mgga_sum_(f1oo[0], ao, wv[0], mask)
                mgga_sum_(f1oo[1], ao, wv[1], mask)

            if with_vxc:
                wv_tmp = numint._uks_mgga_wv0(rho, vxc, weight)
                wv = np.empty((2,5,ngrid))
                wv[0][:4] = wv_tmp[0][:4]; wv[0][4]  = wv_tmp[0][5]
                wv[1][:4] = wv_tmp[1][:4]; wv[1][4]  = wv_tmp[1][5]
                mgga_sum_(v1ao[0], ao, wv[0], mask)
                mgga_sum_(v1ao[1], ao, wv[1], mask)

    else:
        f1vo = np.zeros((2,2,4,nao,nao))
        f1oo = np.zeros((2,4,nao,nao))
        v1ao = np.zeros((2,4,nao,nao))
        k1ao_xpy = np.zeros((2,2,4,nao,nao))
        k1ao_xmy = np.zeros((2,2,4,nao,nao))
        

    f1vo[:,:,1:] *= -1
    if f1oo is not None: f1oo[:,1:] *= -1
    if v1ao is not None: v1ao[:,1:] *= -1
    if with_kxc:
        k1ao_xpy[:,:,1:] *= -1
        k1ao_xmy[:,:,1:] *= -1
    return f1vo, f1oo, v1ao, (k1ao_xpy,k1ao_xmy)

def uks_sf_gga_wv1(rho1, fxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    wv_ab, wv_ba = np.empty((2,4,ngrid))
    wv_ab = np.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    wv_ba = np.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    wv_ab[1:] *=2.0
    wv_ba[1:] *=2.0
    return wv_ab*weight, wv_ba*weight

def uks_sf_gga_wv2_p(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,4,ngrid))
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)
    gv_ab[0,1:] *=2.0
    gv_ab[1,1:] *=2.0
    gv_ba[0,1:] *=2.0
    gv_ba[1,1:] *=2.0
    return gv_ab*weight, gv_ba*weight

def uks_sf_gga_wv2_m(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,5,ngrid))
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)
    gv_ab[:,1:] *=2.0
    gv_ba[:,1:] *=2.0
    return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv1(rho1, fxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    wv_ab, wv_ba = np.empty((2,5,ngrid))
    wv_ab = np.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    wv_ba = np.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    wv_ab[1:4] *=2.0
    wv_ba[1:4] *=2.0
    wv_ab[4] *= 0.5
    wv_ba[4] *= 0.5
    return wv_ab*weight, wv_ba*weight

def uks_sf_mgga_wv2_p(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,5,ngrid))
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)
    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv2_m(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,5,ngrid))
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)
    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight


def _contract_xc_kernel_z(td_grad, xc_code, dmvo, max_memory=2000):
    mol = td_grad.base._scf.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmvo = [(dmvo[0]+dmvo[0].T)*.5,
            (dmvo[1]+dmvo[1].T)*.5]

    f1vo = np.zeros((2,4,nao,nao))
    deriv = 2

    if xctype == 'LDA':
        def lda_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[0], wv)
            for k in range(4):
                vmat[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)

        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = (ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:3]
            u_u, u_d, d_d = fxc[0].T * weight
            rho1a = ni.eval_rho(mol, ao[0], dmvo[0], mask, xctype, hermi=1)
            rho1b = ni.eval_rho(mol, ao[0], dmvo[1], mask, xctype, hermi=1)

            lda_sum_(f1vo[0], ao, u_u*rho1a+u_d*rho1b, mask)
            lda_sum_(f1vo[1], ao, u_d*rho1a+d_d*rho1b, mask)

    elif xctype == 'GGA':
        def gga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T
            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv, mask, ao_loc)
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:3]

            rho1 = (ni.eval_rho(mol, ao, dmvo[0], mask, xctype, hermi=1),
                    ni.eval_rho(mol, ao, dmvo[1], mask, xctype, hermi=1))
            wv = numint._uks_gga_wv1(rho, rho1, vxc, fxc, weight)
            gga_sum_(f1vo[0], ao, wv[0], mask)
            gga_sum_(f1vo[1], ao, wv[1], mask)

    elif xctype == 'MGGA':
        def mgga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

            aow = numint._scale_ao(ao[1], wv[5], aow)
            tmp += numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[2], wv[5], aow)
            tmp += numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[3], wv[5], aow)
            tmp += numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T

            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv[:4], mask, ao_loc)
            rks_grad._tau_grad_dot_(vmat[1:], mol, ao, wv[5]*2, mask, ao_loc, True)

        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]

            rho1 = (ni.eval_rho(mol, ao, dmvo[0], mask, xctype, hermi=1),
                    ni.eval_rho(mol, ao, dmvo[1], mask, xctype, hermi=1))
            wv = numint._uks_mgga_wv1(rho, rho1, vxc, fxc, weight)
            mgga_sum_(f1vo[0], ao, wv[0], mask)
            mgga_sum_(f1vo[1], ao, wv[1], mask)

            vxc = fxc = rho = rho1 = None

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    f1vo[:,1:] *= -1
    return f1vo

def nac_csf(td_grad, x_y_I,x_y_J, atmlst=None):
    '''
    Compute the CSF (Configuration State Function) contribution to non-adiabatic coupling vectors (NACVs)
    between two spin-flip excited states.

    This term arises from the explicit nuclear coordinate dependence of the electronic wavefunctions
    (i.e., the derivative of the CI coefficients), and is necessary for reconstructing the full NACV
    consistent with finite-difference calculations.

    Important: This term breaks translational invariance and is NOT ETF-corrected.
    Including it will make the total NACV inconsistent with momentum conservation,
    but essential for matching finite-difference benchmarks.

    Args:
        td_grad:
            Gradient object associated with SF-TDA or SF-TDDFT calculation.
            Must contain molecular structure and orbital information.
    Returns:
        numpy.ndarray:
            CSF contribution to the NAC vector between states I and J.
            Shape: (natm, 3), in atomic units.
            Not ETF-corrected — use with caution in dynamics simulations.

    Notes:
        - This term + `get_Hellmann_Feynman` = Full NACV (matches finite difference).
        - The name "CSF" refers to the fact that this term originates from the derivative
          of the CI coefficients in the configuration state function basis.
    '''
    mol = td_grad.mol
    mf = td_grad.base._scf

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0]>0)[0]
    occidxb = np.where(mo_occ[1]>0)[0]
    viridxa = np.where(mo_occ[0]==0)[0]
    viridxb = np.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]

    nmoa = nocca + nvira
    nmob = noccb + nvirb

    if td_grad.base.extype==0 or 1:
        
        (x_ab_I, x_ba_I), (y_ab_I, y_ba_I) = x_y_I
        (x_ab_J, x_ba_J), (y_ab_J, y_ba_J) = x_y_J

        x_ab_I = (x_ab_I+y_ab_I).T
        x_ba_I = (x_ba_I+y_ba_I).T
        x_ab_J = (x_ab_J+y_ab_J).T
        x_ba_J = (x_ba_J+y_ba_J).T

        dvv_a_IJ = np.einsum('ai,bi->ab', x_ab_I, x_ab_J) + np.einsum('ai,bi->ab', x_ab_I, x_ab_J) # T^{ab \alpha \beta}*2
        
        
        dvv_b_IJ = np.einsum('ai,bi->ab', x_ba_I, x_ba_J) + np.einsum('ai,bi->ab', x_ba_I, x_ba_J) # T^{ab \beta \alpha}*2
        

        doo_b_IJ = np.einsum('ai,aj->ij', x_ab_I, x_ab_J) + np.einsum('ai,aj->ij', x_ab_I, x_ab_J) # T^{ij \alpha \beta}*2
        doo_b_JI = np.einsum('ai,aj->ij', x_ab_J, x_ab_I) + np.einsum('ai,aj->ij', x_ab_J, x_ab_I)

        doo_a_IJ = np.einsum('ai,aj->ij', x_ba_I, x_ba_J) + np.einsum('ai,aj->ij', x_ba_I, x_ba_J) # T^{ij \beta \alpha}*2

        dmzoo_a_IJ = reduce(np.dot, (orboa, doo_a_IJ, orboa.T))*0.5 # \sum_{\sigma ab} 2*Tab \sigma C_{au} C_{bu}
         
        dmzoo_b_IJ = reduce(np.dot, (orbob, doo_b_IJ, orbob.T))*0.5 # \sum_{\sigma ab} 2*Tij \sigma C_{iu} C_{iu}
        
        dmzoo_a_IJ+= reduce(np.dot, (orbva, dvv_a_IJ, orbva.T))*0.5
        
        dmzoo_b_IJ+= reduce(np.dot, (orbvb, dvv_b_IJ, orbvb.T))*0.5
        
        mf_grad = td_grad.base._scf.nuc_grad_method()    
        s1 = mf_grad.get_ovlp(mol)
        if atmlst is None:
            atmlst = range(mol.natm)
        offsetdic = mol.offset_nr_by_atom()
        nac_csf = np.zeros((len(atmlst),3))
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            nac_csf[k] -= np.einsum('xpq,pq->x', s1[:,p0:p1], dmzoo_a_IJ[p0:p1])*0.5
            nac_csf[k] -= np.einsum('xpq,pq->x', s1[:,p0:p1], dmzoo_b_IJ[p0:p1])*0.5
            nac_csf[k] += np.einsum('xqp,pq->x', s1[:,p0:p1], dmzoo_a_IJ[:,p0:p1])*0.5
            nac_csf[k] += np.einsum('xqp,pq->x', s1[:,p0:p1], dmzoo_b_IJ[:,p0:p1])*0.5
    return nac_csf

def as_scanner(GradientsClass):
    '''
    Convert a NonAdiabaticCouplings class into a geometry scanner.

    The returned Scanner class automatically re-runs SCF and TDDFT when molecular
    geometry changes, then recomputes NACs — ideal for trajectory or geometry optimization.

    Usage:
        nac_obj = NonAdiabaticCouplings(td)
        nac_scanner = nac_obj.as_scanner()
        nac_vec = nac_scanner(new_mol)  # auto-recalculates if geometry changed

    Args:
        GradientsClass (class): A PySCF gradient class (must inherit from grad.Gradients).

    Returns:
        class: A Scanner subclass that supports `__call__(mol)` for automatic recomputation.
    '''
    
    
    class Scanner(GradientsClass):
        def __init__(self, td_object):
            
            super().__init__(td_object)

        def __call__(self, mol):
            '''
            Compute NACs at the given molecular geometry.

            If the geometry differs from the previous one, automatically:
                1. Updates the SCF object
                2. Re-runs SCF and TDDFT
                3. Recomputes NACs

            Args:
                mol (pyscf.gto.Mole): New molecular geometry.

            Returns:
                numpy.ndarray: NAC vector in atomic units, shape (natm, 3).
            '''
            
            td_obj = self.base
            mf_obj = td_obj._scf
            
            
            if np.allclose(mf_obj.mol.atom_coords(), mol.atom_coords()):
                
                return self.kernel()       
           
            lib.logger.info(self, 'New geometry detected. Recalculating SCF and TDDFT.')
                   
            mf_obj.mol = mol         
            mf_obj.kernel()            
            td_obj.kernel()
            return self.kernel()

    return Scanner



class NonAdiabaticCouplings(grad.tdrhf.Gradients):
    '''
    Non-adiabatic coupling (NAC) calculator for spin-flip TDDFT/TDA excited states.

    Computes derivative couplings between two electronic states (I and J) using
    either Hellmann-Feynman term only (ETF-corrected, for dynamics) or full NAC
    (including CSF term, for finite-difference validation).

    Supports automatic phase tracking, energy-difference normalization, and geometry scanning.

    Attributes:
        state_I (int): Index of the first excited state (1-based).
        state_J (int): Index of the second excited state.
        ediff (bool): If True, divide NAC by energy difference (E_J - E_I).
        use_etfs (bool): If True, use only Hellmann-Feynman term (ETF-corrected); else include CSF term.
        x_y_I_prev, x_y_J_prev: Previous CI vectors for phase tracking.
    '''
    def __init__(self, td):
        super().__init__(td)
        self.state_I = None
        self.state_J = None
        self.ediff = False
        self.use_etfs = False
        
        # For phase tracking across geometries
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        
        self._keys = self._keys.union({'state_I', 'state_J', 'ediff', 'use_etfs'})

    def dump_flags(self, verbose=None):
        
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('State I = %d', self.state_I)
        log.info('State J = %d', self.state_J)
        log.info('ediff = %s', self.ediff)
        log.info('use_etfs = %s', self.use_etfs)
        return self

    
    def _vector_dot(self, vec1, vec2):
        '''
        Compute symplectic inner product between two SF-TDDFT excitation vectors:
            <vec1 | vec2> = X1^T * X2 - Y1^T * Y2

        Args:
            vec1, vec2 (tuple): Each is ((X_alpha, X_beta), (Y_alpha, Y_beta)),
        Returns:
            float: Inner product value.
        '''
        x1, y1 = vec1
        x2, y2 = vec2
        
        
        dot_x = 0.0
        
        if isinstance(x1[0], np.ndarray) and isinstance(x2[0], np.ndarray):
            dot_x += np.dot(x1[0].ravel(), x2[0].ravel())
        if isinstance(x1[1], np.ndarray) and isinstance(x2[1], np.ndarray):
            dot_x += np.dot(x1[1].ravel(), x2[1].ravel())
            
        
        dot_y = 0.0
        
        if isinstance(y1[0], np.ndarray) and isinstance(y2[0], np.ndarray):
            dot_y += np.dot(y1[0].ravel(), y2[0].ravel())
        if isinstance(y1[1], np.ndarray) and isinstance(y2[1], np.ndarray):
            dot_y += np.dot(y1[1].ravel(), y2[1].ravel())
            
        return dot_x - dot_y

    def _scale_vector(self, vec, factor):
        '''
        Multiply an excitation vector by a scalar factor.

        Args:
            vec (tuple): ((x_a, x_b), (y_a, y_b))
            factor (float): Scaling factor.

        Returns:
            tuple: Scaled vector in same format.
        '''
        x, y = vec
        
        
        x_a_new = x[0] * factor if isinstance(x[0], np.ndarray) else 0
        x_b_new = x[1] * factor if isinstance(x[1], np.ndarray) else 0
        
        
        y_a_new = y[0] * factor if isinstance(y[0], np.ndarray) else 0
        y_b_new = y[1] * factor if isinstance(y[1], np.ndarray) else 0
        
        return ((x_a_new, x_b_new), (y_a_new, y_b_new))

    def compute_nac(self, state_I=None, state_J=None, atmlst=None,
                    ediff=None, use_etfs=None):
        '''
        Compute non-adiabatic coupling vector between two excited states.

        Performs automatic phase correction to ensure smooth sign changes across geometries.
        Optionally includes CSF term and energy-difference normalization.

        Args:
            state_I, state_J (int): State indices (1-based; 0=ground state).
            atmlst (list of int): Atom indices to compute gradients for. Default: all atoms.
            ediff (bool): If True, normalize NAC by (E_J - E_I). Default: class setting.
            use_etfs (bool): If True, use only Hellmann-Feynman term. Default: class setting.

        Returns:
            numpy.ndarray: NAC vector in atomic units, shape (natm, 3).
        '''
        state_I = state_I if state_I is not None else self.state_I
        state_J = state_J if state_J is not None else self.state_J
        

        if state_I is None or state_J is None:
            raise RuntimeError("state_I and state_J must be specified")

       
        x_y_I = self.base.xy[state_I - 1]
        x_y_J = self.base.xy[state_J - 1]
        e_I = self.base.e[state_I - 1]
        e_J = self.base.e[state_J - 1]
        
       
        #Phase correction
        if self.x_y_I_prev is not None:
            overlap_I = self._vector_dot(x_y_I, self.x_y_I_prev)
            if overlap_I < 0:
                logger.debug(self, 'Flipping sign of state I due to phase change.')
                x_y_I = self._scale_vector(x_y_I, -1.0)
        self.x_y_I_prev = copy.deepcopy(x_y_I)

        
        if self.x_y_J_prev is not None:
            overlap_J = self._vector_dot(x_y_J, self.x_y_J_prev)
            if overlap_J < 0:
                logger.debug(self, 'Flipping sign of state J due to phase change.')
                x_y_J = self._scale_vector(x_y_J, -1.0)
        self.x_y_J_prev = copy.deepcopy(x_y_J)
        

        
        hf_term = get_Hellmann_Feymann(self, x_y_I, x_y_J, atmlst=atmlst)
        if not use_etfs:
            csf_term = nac_csf(self, x_y_I, x_y_J, atmlst) * (e_J - e_I)
            nac = hf_term + csf_term
        else:
            nac = hf_term

        
        if ediff:
            delta_e = e_J - e_I
            if abs(delta_e) < 1e-10:
                logger.warn(self, f"Energy difference is very small: {delta_e}. NAC not divided.")
            else:
                nac = nac / delta_e
      
        return nac
    
    def kernel(self, state_I=None, state_J=None, atmlst=None,
               ediff=None, use_etfs=None):
        
        if state_I is not None: self.state_I = state_I
        if state_J is not None: self.state_J = state_J
        if atmlst is not None: self.atmlst = atmlst
        if ediff is not None: self.ediff = ediff
        if use_etfs is not None: self.use_etfs = use_etfs

        self.nac = self.compute_nac(
            state_I=self.state_I,
            state_J=self.state_J,
            atmlst=self.atmlst,
            ediff=self.ediff,
            use_etfs=self.use_etfs
        )
        return self.nac

    # Reset phase tracking history. Call this before starting a new trajectory
    def reset_phase(self):
        self.x_y_I_prev = None
        self.x_y_J_prev = None
        return self
    as_scanner=as_scanner

NAC = NonAdiabaticCouplings

from pyscf import sftda

sftda.uks_sf.TDA_SF.NAC = sftda.uks_sf.TDDFT_SF.NAC = lib.class_as_method(NonAdiabaticCouplings)

NAC = as_scanner(NonAdiabaticCouplings)


if __name__ == '__main__':
    from pyscf import gto, scf, tdscf
    from pyscf import dft
    from pyscf import sftda, grad
    from pyscf.grad import tduks_sf  # this import is necessary.
    
    try:
        import mcfun  # mcfun>=0.2.5 must be used.
    except ImportError:
        mcfun = None
    from pyscf.sftda.tools_td import transition_analyze 

    mol = gto.Mole()
    mol.atom = '''
    C    0.000000    0.000000    0.000000
    O    1.215000    0.000000    0.000000
    H    0.582000    0.940000    0.000000
    H    0.582000   -0.940000    0.000000
    '''
    mol.basis = '6-31g'
    mol.spin = 2  
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'b3lyp' 
    mf.kernel()

    # TDA_SF object
    mftd1 = sftda.uks_sf.TDA_SF(mf)

    mftd1.max_space = 4000 #necessary
    mftd1.nstates = 4  # the number of excited states
    mftd1.extype = 1  
    mftd1.collinear_samples = 200
    
    mftd1.kernel()
    

    print(transition_analyze(mf, mftd1, mftd1.e[0], mftd1.xy[2], tdtype='TDA'))  #spin analysis
    print(transition_analyze(mf, mftd1, mftd1.e[2], mftd1.xy[2], tdtype='TDA'))

    # nac object
    nac_grad = NAC(mftd1)
    nac_grad.state_I = 1  # S0 
    nac_grad.state_J = 3  # S1

    nac_grad.use_etfs = True  # whether to use ETF correction
    nac_grad.ediff = True     # whether divide by energy difference

    
    nac = nac_grad.kernel()

   
    print("\nNon-Adiabatic Coupling (NAC)with ETF between S0 and S1:")
    print("NAC shape:", nac.shape)  # should be (n_atoms, 3)
    for i, atom in enumerate(mol._atom):
        print(f"Atom {i + 1} ({atom[0]}): {nac[i]}")
    
    nac_grad.use_etfs = False 
    nac = nac_grad.kernel()
    print("\nNon-Adiabatic Coupling (NAC)without ETF between S0 and S1:")
    for i, atom in enumerate(mol._atom):
        print(f"Atom {i + 1} ({atom[0]}): {nac[i]}")
    
    
    #scanner test
    coords = mol.atom_coords()
    coords[0][1] += 0.1  
    coords[1][1] -= 0.1  
    mol.set_geom_(coords, inplace=True) 
    nac_vector_2 = nac_grad(mol)  
    print(nac_vector_2)

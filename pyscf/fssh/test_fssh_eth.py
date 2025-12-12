'''
测试脚本，使用python -m pyscf.fssh.test_fssh_azo -1运行
'''
import os
print(f"[DEBUG] OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"[DEBUG] MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")

import argparse
import numpy as np
# ... (所有其他的 import) ...
import random
from scipy.special import laguerre
from pyscf.fssh.fssh_sf import FSSH_SF
from pyscf.geomopt import geometric_solver
def wignerfunc(mu, temp):
    mu = np.array(mu).item()
    
    if temp < 1e-6:
        # T = 0 K: only n = 0 is populated
        n = 0
        while True:
            q = random.uniform(0, 1) * 10.0 - 5.0
            p = random.uniform(0, 1) * 10.0 - 5.0
            rho2 = 2 * (q ** 2 + p ** 2)
            w = np.exp(-0.5 * rho2)  # n=0: (-1)^0 * L_0(rho2)=1
            r = random.uniform(0, 1)
            if r < w < 1:
                break
        return float(q), float(p)
    
    else:
        # T > 0 K: same as original code
        max_pop = 0.9999
        ex = mu / (0.69503 * temp)
        pop = 0.0
        lvl_pop = []
        n = -1
        while True:
            n += 1
            pop += float(np.exp(-ex * n) * (1 - np.exp(-ex)))
            lvl_pop.append(pop)
            if pop >= max_pop:
                break

        while True:
            random_state = random.uniform(0, pop)
            n = -1
            for i in lvl_pop:
                n += 1
                if random_state <= i:
                    break

            if n > 150:
                print('Sampled vibrational state is higher than 150, adjusted to 150')
                n = 150
            # NO break here — same as original

            q = random.uniform(0, 1) * 10.0 - 5.0
            p = random.uniform(0, 1) * 10.0 - 5.0
            rho2 = 2 * (q ** 2 + p ** 2)
            w = (-1) ** n * laguerre(n)(rho2) * np.exp(-0.5 * rho2)
            r = random.uniform(0, 1)
            if r < w < 1:
                break

        return float(q), float(p)
def wigner(temp, freqs, xyz, vib):
    ## This function is based on SHARC wigner.py
    ## This function does Wigner sampling for structure and velocity
    ## This function calls wignerfunc to find update coefficient
    ## This function returns initial condition as [[atom x y z v(x) v(y) v(z)],...]

    nfreq = len(freqs)
    natom = len(xyz)
    # rmass = sample['rmass']
    FS2AUTIME = 41.34137 
    mu_to_hartree = 4.55633518e-6  # 1 cm-1  = h*c/Eh = 4.55633518e-6 au
    ma_to_amu = 1822.88852  # 1 g/mol = 1/Na*me*1000 = 1822.88852 amu
    bohr_to_angstrom = 0.529177249  # 1 Bohr  = 0.529177249 Angstrom

    q_p = np.array([wignerfunc(i, temp) for i in freqs])  # generates update coordinates and momenta pairs Q and P

    q = q_p[:, 0].reshape((nfreq, 1))  # first column is Q

    q *= 1 / np.sqrt(freqs * mu_to_hartree * ma_to_amu)  # convert coordinates from m to Bohr'* ma_to_amu'
    qvib = np.array([np.ones((natom, 3)) * i for i in q])  # generate identity array to expand Q
    qvib = np.sum(vib * qvib, axis=0)  # sum sampled structure over all modes
    newc = (xyz + qvib) * bohr_to_angstrom  # cartesian coordinates in Angstrom

    p = q_p[:, 1].reshape((nfreq, 1))  # second column is P
    p *= np.sqrt(freqs * mu_to_hartree/ ma_to_amu)  # convert velocity from m/s to Bohr/au / ma_to_amu
    pvib = np.array([np.ones((natom, 3)) * i for i in p])  # generate identity array to expand P
    velo = np.sum(vib * pvib, axis=0)  # sum sampled velocity over all modes in Bohr/au
    velo = velo/1.889726
    velo = velo*(FS2AUTIME*1000)
    initcond = np.concatenate((newc, velo), axis=1)

    return initcond
def main(traj_idx):
    from pyscf import gto, dft, sftda, hessian
    import pyscf
    from pyscf.hessian import thermo
    # --- 在这里放入你所有的设置和运行代码 ---

    # *** 核心修改：使用 traj_idx 来命名输出 ***
    output_dir = f"traj_{traj_idx}" # 为每条轨迹创建独立的文件夹

    # 1. 定义分子 (这部分不变)
    atom ='''
C       -1.333526     0.151557     0.002045
H       -0.936384    -0.804118     0.494306
H       -2.333962    -0.014210    -0.531314
C       -0.597312     1.438604    -0.002665
H       -0.796505     2.277230     0.698398
H        0.245907     1.677226    -0.653921 
'''
    print("--- Calculating S0 Ground State Properties for Sampling ---")
    # 使用 RKS (自旋=0) 来描述 S0 基态
    mol = pyscf.M(atom=atom, basis='6-31g**', unit='A', spin=0)
    mf = dft.RKS(mol,xc='b3lyp').run(conv_tol=1e-6) # 优化S0几何构型
    mol_s0 = geometric_solver.optimize(mf) 
    print("Equilibrium coordinates (Angstrom):") 
    print(mol_s0.atom_coords(unit='a')) 
    mf_s0 = dft.RKS(mol_s0, xc='b3lyp').run() 
    
    # 对优化好的 S0 构型进行频率分析
    hess_s0 = mf_s0.Hessian().kernel()
    thermo_data_s0 = thermo.harmonic_analysis(mol_s0, hess_s0)
    
    # 提取 S0 的数据用于 Wigner 采样
    s0_equilibrium_coords_bohr = mol_s0.atom_coords(unit='Bohr')
    s0_freqs = thermo_data_s0['freq_wavenumber']
    s0_masses = mol_s0.atom_mass_list()
    s0_norm_modes_non_weighted = thermo_data_s0['norm_mode'] 
    
    print("--- S0 Calculation Finished ---")
    
    # ==============================================================================
    # >>>>>>>>>> 步骤 2: 使用 S0 性质进行 Wigner 采样 <<<<<<<<<<<<<<<
    # ==============================================================================
    print(f"--- Generating Initial Conditions via Wigner Sampling on S0 PES ---")
    temperature = 300
    
    
    # 过滤掉 S0 的平动和转动零频
    valid_indices = np.where(s0_freqs > 1e-1)[0]
    
    initial_condition = wigner(
        temperature,
        s0_freqs[valid_indices].reshape(-1, 1),
        s0_equilibrium_coords_bohr,
        s0_norm_modes_non_weighted[valid_indices]
    )
    
    initial_positions = initial_condition[:, 0:3]  # in Angstrom
    
    initial_velocities = initial_condition[:, 3:6] # in A/ps
    sampled_atom_string = ""
    
    atom_symbols = [mol_s0.atom_pure_symbol(i) for i in range(mol_s0.natm)]
    for i in range(mol_s0.natm):
        x, y, z = initial_positions[i]
        sampled_atom_string += f"{atom_symbols[i]} {x:10.6f} {y:10.6f} {z:10.6f}\n"

    # 使用采样出的构型和 T1 参考态来构建 FSSH 的计算对象
    mol_t1 = pyscf.M(atom=sampled_atom_string, basis='6-31G**', unit='A', spin=2)
    mf_t1 = dft.UKS(mol_t1, xc='b3lyp').run() # 只需要跑一次单点能
    
    sftd = sftda.uks_sf.TDA_SF(mf_t1)
    sftd.extype = 1
    sftd.collinear_samples = 50
    sftd.max_space = 1000
    sftd.nstates = 8
    sftd.kernel()
    
    # 3. 准备动力学
    # **将输出目录传递给 FSSH_SF**
    fssh_sim = FSSH_SF(sftd, states=[1, 2], nsteps=300, dt=0.5, 
                       output_dir=output_dir) # 500步 ~ 100 fs
    fssh_sim.cur_state = 2

    # **为每条轨迹生成不同的随机初速度**
    
    initial_coeffs = np.array([0.0, 1.0], dtype=complex)
    
    # 4. 运行
    print(f"--- Starting Trajectory {traj_idx} ---")
    fssh_sim.kernel(position=mol_t1.atom_coords(), 
                    velocity=initial_velocities, 
                    coefficient=initial_coeffs)
    print(f"--- Finished Trajectory {traj_idx} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single FSSH-SF trajectory for ethylene.")
    parser.add_argument("traj_idx", type=int, help="The index of the trajectory.")
    args = parser.parse_args()
    
    main(args.traj_idx)

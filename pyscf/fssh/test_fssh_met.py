#!/usr/bin/env python3
import argparse
import os
import numpy as np

# ========= 1. 设置命令行参数和线程数 =========
parser = argparse.ArgumentParser()
parser.add_argument('--traj_id', type=int, default=0, help='Trajectory ID for parallel runs')
parser.add_argument('--steps', type=int, default=2000, help='Number of steps to run')
args = parser.parse_args()

# 设置随机种子（每条轨迹不同）
np.random.seed(12345 + args.traj_id)  # 基础种子 + 轨迹ID

# 设置PySCF线程数（从环境变量读取，或默认4）
n_threads = int(os.environ.get('OMP_NUM_THREADS', 4))
from pyscf import lib
lib.num_threads(n_threads)
print(f"=== Trajectory {args.traj_id}: Using {n_threads} threads ===")


from pyscf import gto, dft
from pyscf.sftda import uks_sf 
from pyscf.sftda.uhf_sf import get_ab_sf
from fssh_sf import FSSH_SF_NACV
from tools import extract_state
from pyscf.sftda.tools_td import transition_analyze

# --- 1. 分子和电子结构设置 ---
mol = gto.Mole()
mol.atom = '''
C   0.000000    0.658300    0.000000
N   0.000000   -0.603300    0.000000
H   0.930000    1.248300    0.000000
H  -0.930000    1.248300    0.000000
H   0.000000   -1.213300    0.000000
'''
mol.basis = 'cc-pVDZ'
mol.spin = 2
mol.build()

print("--- Running Methanimine Simulation ---")

mf = dft.UKS(mol)
mf.xc = 'svwn'

mf.kernel()


# 5. 创建TDDFT_SF对象并手动填充结果
mftd1 = uks_sf.TDA_SF(mf)
mftd1.extype = 1
mftd1.max_space = 400
mftd1.collinear_samples =50
mftd1.nstates = 10
mftd1.kernel()
S = extract_state(mf, mftd1, Smin=0.0, Smax=0.7)
target1 = S[0]
target2 = S[1]
E1 =mf.e_tot+ mftd1.e[target1]
E2 =mf.e_tot+ mftd1.e[target2]
print(transition_analyze(mf, mftd1, mftd1.e[target1], mftd1.xy[target1], tdtype='TDDFT'))  #spin analysis
print(transition_analyze(mf, mftd1, mftd1.e[target2], mftd1.xy[target2], tdtype='TDDFT'))
# 2. 定义 FSSH 模拟参数
active_states = [target1+1, target2+1]


output_dir = f'methanimine_traj_{args.traj_id}'
os.makedirs(output_dir, exist_ok=True)  # 自动创建目录

fssh_simulation = FSSH_SF_NACV(
    sftda_obj=mftd1,
    states=active_states,
    nac_options={'use_etfs': True, 'ediff': True},
    dt=0.2,
    cphf_options={'max_cycle': 200, 'conv_tol': 1e-6},
    nsteps=args.steps,  # 使用命令行传入的步数
    output_dir=output_dir  # 每条轨迹独立目录
)

# --- 4. 设置初始条件 ---
initial_pos = mol.atom_coords()
initial_vel = (np.random.rand(*initial_pos.shape) - 0.5) * 0.01
initial_coeffs = np.array([0.0, 1.0], dtype=complex) # 初始在S1

# --- 5. 运行模拟 ---
fssh_simulation.kernel(
    position=initial_pos,
    velocity=initial_vel,
    coefficient=initial_coeffs
)
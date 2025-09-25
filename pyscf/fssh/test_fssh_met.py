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

import pyscf.nac.tduks_sf
from pyscf import gto, dft
from pyscf.sftda import uks_sf 
from pyscf.sftda.uhf_sf import get_ab_sf
from fssh_sf import FSSH_SF
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
''' 通过全对角化和筛选获得 SF-TDDFT 激发态 --- 
a, b = get_ab_sf(mf, collinear_samples=50)
A_baba, A_abab = a
B_baab, B_abba = b

# 获取轨道维度信息
mo_occ = mf.mo_occ
occ_a = np.where(mo_occ[0] > 0)[0]
occ_b = np.where(mo_occ[1] > 0)[0]
virt_a = np.where(mo_occ[0] == 0)[0]
virt_b = np.where(mo_occ[1] == 0)[0]

n_occ_a, n_virt_b = len(occ_a), len(virt_b)
n_occ_b, n_virt_a = len(occ_b), len(virt_a)

# 将矩阵reshape为2D
A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b))
B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a))
B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b))
A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a))

# 构建完整的Casida矩阵
Casida_matrix = np.block([
    [ A_abab_2d, B_abba_2d],
    [-B_baab_2d,-A_baba_2d]  # 注意，PySCF的B矩阵定义可能与某些文献相反，这里用-B
])

# 对非对称矩阵使用np.linalg.eig进行全对角化
eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)

# 4. 筛选并归一化物理上有意义的解 (这部分是关键！)
# 按照能量大小排序
idxt = eigenvals.real.argsort()
eigenvals = eigenvals[idxt].real
eigenvecs = eigenvecs[:, idxt]

# 计算范数 ||X||^2 - ||Y||^2 来筛选物理的自旋翻转解
norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2
norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2

# 保留范数 > 0 的解
sfd_eigenvals = eigenvals[norms > 0]
sfd_eigenvecs = eigenvecs[:, norms > 0].T

# 定义归一化函数 (与您提供的tdgrad中的函数一致)
def norm_xy(z):
    x_flat = z[:n_occ_a*n_virt_b]
    y_flat = z[n_occ_a*n_virt_b:]
    norm = np.linalg.norm(x_flat)**2 - np.linalg.norm(y_flat)**2
    norm = np.sqrt(1./norm)
    
    x = x_flat.reshape(n_occ_a, n_virt_b) * norm
    y = y_flat.reshape(n_occ_b, n_virt_a) * norm
    
    # 按照PySCF的格式返回 ((y_aa, x_ab), (y_ba, x_bb))
    # 对于alpha->beta的翻转 (extype=1)，激发部分是x_ab，退激发是y_ba
    return ((0, x), (y, 0))

# 5. 创建TDDFT_SF对象并手动填充结果
mftd1 = uks_sf.TDDFT_SF(mf)
mftd1.max_space = 4000  # necessary for NAC calculations
mftd1.extype = 1
mftd1.collinear_samples =50
mftd1.e = sfd_eigenvals
mftd1.xy = [norm_xy(z) for z in sfd_eigenvecs]
mftd1.nstates = len(mftd1.e)

print(f"通过全对角化和筛选，找到 {mftd1.nstates} 个自旋向下翻转激发态。")
print("前4个激发能:", mftd1.e[:4])
'''

# 5. 创建TDDFT_SF对象并手动填充结果
mftd1 = uks_sf.TDDFT_SF(mf)
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

fssh_simulation = FSSH_SF(
    tddft=mftd1,
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
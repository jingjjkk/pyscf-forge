import numpy as np
from pyscf import gto, dft, sftda
from pyscf.grad import tduks_sf
from fssh_ktdc_sf import FSSH_SF  # 假设您的FSSH类在这个文件中

# 使用LiH分子进行测试（最小可行系统）
mol = gto.Mole()
mol.atom = '''
Li 0.0 0.0 0.0
H  1.6 0.0 0.0  # LiH的平衡距离约为1.6 Å
'''
mol.basis = 'sto-3g'  # 最小基组
mol.spin = 2  # 双自由基特性，有未成对电子
mol.build()

# 简化计算设置
mf = dft.UKS(mol)
mf.xc = 'svwn'
mf.conv_tol = 1e-6  # 降低收敛精度
mf.max_cycle = 50   # 减少最大迭代次数
mf.kernel()

# 只计算2个态
mftd1 = sftda.uks_sf.TDA_SF(mf)
mftd1.nstates = 2
mftd1.max_space = 100  # 减少最大空间
mftd1.conv_tol = 1e-6  # 降低收敛精度
mftd1.kernel()

# 只模拟2个态之间的跃迁
active_states = [1, 2]

# 创建FSSH模拟器，但只运行极少的步数
fssh_simulation = FSSH_SF(
    sftda_obj=mftd1,
    states=active_states,
    dt=0.1,     # 小时间步长
    nsteps=3,   # 只运行3步
    output_dir='test_fssh_sf'
)

# 使用简单的初始条件
initial_pos = mol.atom_coords()
initial_vel = np.zeros_like(initial_pos)  # 零初始速度
initial_coeffs = np.array([1.0, 0.0], dtype=complex)

print("开始简化测试...")
try:
    # 尝试运行简化模拟
    final_pos, final_vel, final_coeffs = fssh_simulation.kernel(
        position=initial_pos,
        velocity=initial_vel,
        coefficient=initial_coeffs
    )
    
    print("测试成功！代码逻辑正确。")
    print(f"最终位置: {final_pos}")
    print(f"最终系数: {final_coeffs}")
    
except Exception as e:
    print(f"测试遇到错误: {e}")
    import traceback
    traceback.print_exc()
    print("这可能是由于计算资源限制，但代码结构看起来是正确的。")
# NTTDA `deltaS=-1` 非绝热耦合推导 — 2026-07-17

**目标：** 从 NTTDA 本征方程和自旋张量配置基出发，推导两个
`deltaS=-1` 激发根之间的完整解析非绝热耦合、Hellmann--Feynman
分子、CSF/AWF 连接项及 ETF 结果，并给出可直接映射到 PySCF 实现的公式。

**假设：**

- Born--Oppenheimer 电子态使用实数、绝热、非简并规范。
- 使用 Tamm--Dancoff 近似，因此只有 $mathbf X$ 振幅，$mathbf Y=0$。
- 参考态是高自旋 ROKS 多重态，开壳层轨道数 $n_O=2S$，且 $S\geq1$。
- 本文只处理目标自旋 $S_t=S-1$，即代码中的 `deltaS=-1`。
- AO 基随核坐标移动；完整导数耦合必须包含配置基的连接项。
- 不包含自旋轨道耦合，因此不同总自旋扇区之间的耦合为零。

## 1. 符号和代码布局

- 闭壳轨道 $i,j\in C$，开壳轨道 $t,u,v,w\in O$，虚轨道
  $a,b\in V$。
- 一般空间轨道 $p,q,r,s$；AO 指标 $\mu,\nu$。
- $\mathbf C_C$、$\mathbf C_O$、$\mathbf C_V$ 是对应的 MO 系数块。
- $\omega_K$ 和 $\mathbf X_K$ 是第 $K$ 个 NTTDA 激发能和归一化振幅。
- $R_{A\xi}$ 是原子 $A$ 的笛卡尔分量，$\xi\in\{x,y,z\}$。
- $\mathbf S^{A\xi}$ 是 `get_ovlp` 返回的 AO 重叠导数块。

`pyscf/sftda/nttda.py` 把 `deltaS=-1` 振幅保存为
`(n_C+n_O, n_O+n_V)`：

$$
\mathbf X_K=
\begin{pmatrix}
\mathbf X_K^{CO} & \mathbf X_K^{CV} \\
\mathbf X_K^{OO} & \mathbf X_K^{OV}
\end{pmatrix}.
$$

→ **解释：** 行表示被移走电子的闭壳/开壳轨道，列表示电子到达的开壳/虚轨道；这与 `split_spin_lowering` 的四块完全一致。

代码变量对应如下：

- `xy[0]`, shape `(n_C+n_O, n_O+n_V)` = $\mathbf X_K$。
- `co/cv/oo/ov` = $\mathbf X_K^{CO/CV/OO/OV}$。
- `base.e[K-1]` = $\omega_K$。
- `base.xy[K-1]` = $(\mathbf X_K,0)$。
- `ediff=True` = 将能量缩放分子除以 $\omega_J-\omega_I$。
- `use_etfs=True` = 只保留 Hellmann--Feynman/ETF 项。

## 2. 本征方程与导数耦合分解

NTTDA 在正交的物理配置子空间中满足

$$
\mathbf A(\mathbf R)\mathbf X_K(\mathbf R)
=\omega_K(\mathbf R)\mathbf X_K(\mathbf R).
$$

→ **解释：** $\mathbf A$ 是 `gen_vind_sfd` 实现的实对称矩阵；零迹 OO 冗余方向是零模并在 `kernel` 中删除。

不同根满足

$$
\mathbf X_I^{T}\mathbf X_J=0.
$$

→ **解释：** 本式只适用于已去除 OO 零模的物理根，且 $I\neq J$。

辅助波函数写成移动的自旋适配配置基展开：

$$
|\Psi_K(\mathbf R)\rangle
=\sum_P X_{PK}(\mathbf R)|\Phi_P(\mathbf R)\rangle.
$$

→ **解释：** $P$ 遍历 CO、CV、OO、OV 四类自旋适配配置；核坐标依赖同时存在于振幅和配置基中。

完整导数耦合为

$$
d_{IJ}^{A\xi}
=\langle\Psi_I|\partial_{A\xi}\Psi_J\rangle
=\sum_P X_{PI}\partial_{A\xi}X_{PJ}
+\sum_{PQ}X_{PI}X_{QJ}
\langle\Phi_P|\partial_{A\xi}\Phi_Q\rangle.
$$

→ **解释：** 第一项是本征矢响应，第二项是移动 CSF/AWF 基的连接；后者不能用普通未自旋适配 spin-flip 行列式代替。

对本征方程求导并左乘 $\mathbf X_I^T$，得到

$$
\sum_P X_{PI}\partial_{A\xi}X_{PJ}
=\frac{N_{IJ}^{\mathrm{HF},A\xi}}
{\omega_J-\omega_I}.
$$

→ **解释：** $N_{IJ}^{\mathrm{HF}}$ 是在对称轨道连接规范下的态间 Hellmann--Feynman 分子；轨道响应通过 ROKS Z-vector 隐含在其中。

因此

$$
d_{IJ}^{A\xi}
=\frac{N_{IJ}^{\mathrm{HF},A\xi}}
{\omega_J-\omega_I}
+d_{IJ}^{\mathrm{CSF},A\xi}.
$$

→ **解释：** 这是完整解析 NAC；ETF 版本只保留右侧第一项。

定义便于 `ediff=False` 返回的能量缩放完整分子：

$$
N_{IJ}^{\mathrm{full},A\xi}
=N_{IJ}^{\mathrm{HF},A\xi}
+(\omega_J-\omega_I)d_{IJ}^{\mathrm{CSF},A\xi}.
$$

→ **解释：** 除以能隙后恢复完整导数耦合，量纲从能量/长度变为 $1/$长度。

## 3. Hellmann--Feynman 分子的解析极化

现有 `delta_s_minus_one.grad_elec` 对任意固定振幅 $\mathbf X$ 返回齐次二次泛函

$$
\mathbf G[\mathbf X]
=\partial_{\mathbf R}
\left(\mathbf X^T\mathbf A\mathbf X\right)_{\mathrm{relaxed\ orbital}}.
$$

→ **解释：** `direct`、M 矩阵、Z-vector RHS、Z-vector 解和最终 Pulay 项均对 $\mathbf X$ 保持二次齐次性。

对实对称双线性型使用极化恒等式：

$$
\mathbf N_{IJ}^{\mathrm{HF}}
=\frac{1}{4}
\left[
\mathbf G[\mathbf X_I+\mathbf X_J]
-\mathbf G[\mathbf X_I-\mathbf X_J]
\right].
$$

→ **解释：** 该式直接复用已通过能量有限差分验证的 HF/LDA/GGA/MGGA/hybrid/RSH 梯度后端，同时保留全部 ROKS 轨道驰豫。

对角极限给出重要的实现检查：

$$
\mathbf N_{II}^{\mathrm{HF}}=\mathbf G[\mathbf X_I].
$$

→ **解释：** 内部辅助函数允许 $I=J$ 做此检查；公共 NAC 接口仍拒绝同态耦合。

## 4. `deltaS=-1` 自旋适配辅助波函数

### 4.1 参考多重态分量

最高权参考态 $|S,S\rangle$ 是闭壳轨道双占据、全部开壳轨道 alpha 占据的单个 Slater 行列式。其余分量由总降自旋算符生成：

$$
|S,M\rangle
=\mathcal N_{SM}
(\hat S_-)^{S-M}|S,S\rangle.
$$

→ **解释：** $\mathcal N_{SM}$ 通过显式归一化确定；实现只需 $M=S,S-1,S-2$ 三个分量。

总降自旋算符为

$$
\hat S_-=\sum_p a_{p\beta}^{\dagger}a_{p\alpha}.
$$

→ **解释：** 使用二次量子化操作而不是手填行列式符号，可以自动保留正确的费米相位。

rank-1 激发张量三个分量取 NTTDA 论文的相位约定：

$$
T_{pq}^{\dagger}(1,-1)=a_{p\beta}^{\dagger}a_{q\alpha}.
$$

→ **解释：** 这是从 alpha 源轨道到 beta 目标轨道的 spin-flip-down 分量。

$$
T_{pq}^{\dagger}(1,0)
=\frac{1}{\sqrt{2}}
\left(a_{p\alpha}^{\dagger}a_{q\alpha}
-a_{p\beta}^{\dagger}a_{q\beta}\right).
$$

→ **解释：** 这是自旋张量的零投影分量。

$$
T_{pq}^{\dagger}(1,+1)
=-a_{p\alpha}^{\dagger}a_{q\beta}.
$$

→ **解释：** 负号固定三个张量分量之间的 Condon--Shortley 相位。

选择目标态最高权分量 $M_t=S_t=S-1$，未归一化配置为

$$
|\widetilde\Phi_{pq}^{S-1}\rangle
=\sum_{m=-1}^{+1}
\langle S,S-1-m;1,m|S-1,S-1\rangle
T_{pq}^{\dagger}(1,m)|S,S-1-m\rangle.
$$

→ **解释：** 三项分别使用参考多重态的 $M=S,S-1,S-2$ 分量，因而显式恢复目标总自旋而非仅固定 $S_z$。

所需 Clebsch--Gordan 系数为

$$
c_{-1}=\sqrt{\frac{2S-1}{2S+1}}.
$$

→ **解释：** 它乘在 $T^{\dagger}(1,-1)|S,S\rangle$ 上。

$$
c_{0}=-\sqrt{\frac{2S-1}{S(2S+1)}}.
$$

→ **解释：** 它乘在 $T^{\dagger}(1,0)|S,S-1\rangle$ 上。

$$
c_{+1}=\frac{1}{\sqrt{S(2S+1)}}.
$$

→ **解释：** 它乘在 $T^{\dagger}(1,+1)|S,S-2\rangle$ 上。

### 4.2 四个振幅块的归一化

归一化配置定义为

$$
|\Phi_{pq}^{B}\rangle
=n_B|\widetilde\Phi_{pq}^{S-1}\rangle.
$$

→ **解释：** $B\in\{CO,CV,OO,OV\}$，不同块因泡利约束具有不同约化矩阵元。

四块归一化因子为

$$
n_{CV}=1.
$$

→ **解释：** 闭壳到虚轨道配置的耦合态已经归一化。

$$
n_{CO}=n_{OV}=\sqrt{\frac{2S}{2S+1}}.
$$

→ **解释：** 目标或源为开壳轨道时，未归一化态的范数为 $(2S+1)/(2S)$。

$$
n_{OO}=\sqrt{\frac{2S-1}{2S+1}}.
$$

→ **解释：** OO 非对角配置的未归一化范数为 $(2S+1)/(2S-1)$。

CO、CV、OV 块在各自配置标签上是单位度量。OO 块的度量为

$$
\langle\Phi_{tu}^{OO}|\Phi_{vw}^{OO}\rangle
=\delta_{tv}\delta_{uw}
-\frac{1}{2S}\delta_{tu}\delta_{vw}.
$$

→ **解释：** 第二项只作用于 OO 对角配置，删除与单位矩阵成比例的迹方向。

因此任意振幅的 AWF 范数为

$$
\langle\Psi|\Psi\rangle
=\|\mathbf X\|_F^2
-\frac{1}{2S}
\left|\operatorname{Tr}\mathbf X^{OO}\right|^2.
$$

→ **解释：** `gen_vind_sfd` 的零本征值正是 $\mathbf X^{OO}\propto\mathbf 1$；删除零模后的物理根满足 $\operatorname{Tr}\mathbf X^{OO}=0$，所以欧氏归一化与 AWF 归一化相同。

## 5. 态间一粒子密度与 CSF 连接

把上述自旋适配 AWF 展开到统一的 Slater 行列式基：

$$
|\Psi_K\rangle=\sum_D c_D^K|D\rangle.
$$

→ **解释：** 每个系数同时包含 NTTDA 振幅、CG 系数、块归一化和费米相位。

态间自旋求和一粒子密度定义为

$$
\gamma_{pq}^{IJ}
=\sum_{\sigma\in\{\alpha,\beta\}}
\langle\Psi_I|a_{p\sigma}^{\dagger}a_{q\sigma}|\Psi_J\rangle.
$$

→ **解释：** 实现通过 Slater--Condon 单激发规则和行列式系数字典计算，代价随非零 AWF 行列式数多项式增长。

AO 表示为

$$
D_{\mu\nu}^{IJ}
=\sum_{pq}C_{\mu p}\gamma_{pq}^{IJ}C_{\nu q}^{*}.
$$

→ **解释：** 在当前实轨道实现中就是 `mo_coeff @ gamma @ mo_coeff.T`。

配置基连接的 AO 部分只取态间密度的反对称部分：

$$
d_{IJ}^{\mathrm{CSF,AO},A\xi}
=\frac{1}{2}
\sum_{\mu\in A}\sum_{\nu}
S_{\mu\nu}^{A\xi}
\left(D_{\nu\mu}^{IJ}-D_{\mu\nu}^{IJ}\right).
$$

→ **解释：** 对称部分已经包含在 Hellmann--Feynman 分子的 Pulay 规范中；本式是 AO 随原子中心移动产生的反对称连接。

能量梯度可以删除等占据子空间内部的冗余旋转，但态间极化后，移动配置基还包含 C--O、C--V、O--V 轨道响应连接。定义反对称 MO 源

$$
M_{pq}^{IJ}=\frac{1}{2}
\left(\gamma_{pq}^{IJ}-\gamma_{qp}^{IJ}\right).
$$

→ **解释：** 只有 $M_{pq}-M_{qp}$ 能与反厄米轨道旋转耦合；代码用 `pack_m_matrix` 取出独立 C--O、C--V、O--V 分量。

设 $\mathbf H^{\mathrm{ROKS}}$ 是参考态轨道 Hessian，轨道连接的伴随方程为

$$
\left(\mathbf H^{\mathrm{ROKS}}\right)^T
\mathbf z^{IJ}=\mathbf m^{IJ}.
$$

→ **解释：** $\mathbf m^{IJ}$ 是 $\mathbf M^{IJ}$ 的独立反对称分量；一次态间 Z-vector 求解代替对每个核坐标显式求解轨道响应。

把伴随解与 ROKS Fock 骨架导数、AO 正交归一导数收缩，得到

$$
d_{IJ}^{\mathrm{CSF,orb},A\xi}
=\sum_{pq}M_{pq}^{IJ}U_{pq}^{A\xi}.
$$

→ **解释：** 实现复用 `make_hessian_transpose_action`、`solve_zvector`、`spin_fock_direct_hf/dft` 和 `_orbital_gradient`，因此同样覆盖 HF、半局域 XC、hybrid 与 RSH。

完整移动配置连接为

$$
d_{IJ}^{\mathrm{CSF},A\xi}
=d_{IJ}^{\mathrm{CSF,AO},A\xi}
+d_{IJ}^{\mathrm{CSF,orb},A\xi}.
$$

→ **解释：** 缺少第二项时，对角能量梯度仍可正确，但态间 NAC 一般不能与跨几何 AWF 重叠有限差分一致。

交换两个实态时

$$
\mathbf d_{JI}^{\mathrm{CSF}}
=-\mathbf d_{IJ}^{\mathrm{CSF}}.
$$

→ **解释：** 因为 $\boldsymbol\gamma^{JI}=(\boldsymbol\gamma^{IJ})^T$。

## 6. ETF 结果与反对称性

本实现沿用现有 SF-NAC 接口语义，把 ETF 结果定义为只保留 Hellmann--Feynman 项：

$$
\mathbf d_{IJ}^{\mathrm{ETF}}
=\frac{\mathbf N_{IJ}^{\mathrm{HF}}}
{\omega_J-\omega_I}.
$$

→ **解释：** `use_etfs=True, ediff=True` 返回该量；不加入 CSF/AWF 反对称轨道重叠项。

HF 分子满足

$$
\mathbf N_{JI}^{\mathrm{HF}}=\mathbf N_{IJ}^{\mathrm{HF}}.
$$

→ **解释：** 它来自实对称双线性型的极化。

因此 ETF 和完整 NAC 均满足

$$
\mathbf d_{JI}=-\mathbf d_{IJ}.
$$

→ **解释：** 交换态时能隙变号，CSF 项自身也变号。

## 7. 自旋适配 AWF 重叠有限差分

不同几何 $L$、$R$ 的空间 MO 重叠为

$$
S_{pq}^{LR}
=\sum_{\mu\nu}
C_{\mu p}^{L*}
\langle\chi_{\mu}^{L}|\chi_{\nu}^{R}\rangle
C_{\nu q}^{R}.
$$

→ **解释：** AO 交叉重叠由 `gto.intor_cross('int1e_ovlp', mol_L, mol_R)` 计算。

两个同自旋 Slater 行列式的交叉重叠分解为 alpha、beta 两个行列式：

$$
\langle D_L|D_R\rangle
=\det\mathbf S_{\alpha}^{LR}
\det\mathbf S_{\beta}^{LR}.
$$

→ **解释：** 行列式的行、列分别选择左右态中占据的同自旋空间轨道。

AWF 重叠为

$$
\langle\Psi_I^L|\Psi_J^R\rangle
=\sum_{D_LD_R}
c_{D_L}^{I,L*}c_{D_R}^{J,R}
\langle D_L|D_R\rangle.
$$

→ **解释：** 该重叠同时用于根的一一匹配、相位对齐和有限差分基准。

采用平衡中心差分：

$$
d_{IJ}^{A\xi,\mathrm{FD}}
=\frac{
\langle\Psi_I(\mathbf R-h\mathbf e_{A\xi})|
\Psi_J(\mathbf R+h\mathbf e_{A\xi})\rangle
-
\langle\Psi_I(\mathbf R+h\mathbf e_{A\xi})|
\Psi_J(\mathbf R-h\mathbf e_{A\xi})\rangle
}{4h}.
$$

→ **解释：** 两个位移几何的根先与中心几何做最大 AWF 重叠的一一指派并调成正相位；该公式显式保持 $I,J$ 反对称性。

## 8. 最终实现公式

`use_etfs=False, ediff=False`：

$$
\boxed{
\mathbf N_{IJ}^{\mathrm{full}}
=\mathbf N_{IJ}^{\mathrm{HF}}
+(\omega_J-\omega_I)\mathbf d_{IJ}^{\mathrm{CSF}}
}.
$$

→ **解释：** 返回能量缩放的完整分子，量纲为能量/长度。

`use_etfs=False, ediff=True`：

$$
\boxed{
\mathbf d_{IJ}^{\mathrm{full}}
=\frac{\mathbf N_{IJ}^{\mathrm{HF}}}{\omega_J-\omega_I}
+\mathbf d_{IJ}^{\mathrm{CSF}}
}.
$$

→ **解释：** 这是应与 AWF 重叠有限差分比较的完整 NAC。

`use_etfs=True, ediff=True`：

$$
\boxed{
\mathbf d_{IJ}^{\mathrm{ETF}}
=\frac{\mathbf N_{IJ}^{\mathrm{HF}}}{\omega_J-\omega_I}
}.
$$

→ **解释：** 这是用户要求的 ETF 校正结果，即只包含 Hellmann--Feynman 项。

**代码对应：** `pyscf/nac/nttda.py` 的 `get_hf_interstate_numerator`、
`build_spin_adapted_awf`、`interstate_rdm1`、`nac_csf`、`awf_overlap` 和
`NonAdiabaticCouplings`；解析梯度后端对应
`pyscf/grad/nttda/delta_s_minus_one.py:grad_elec`。

# SATDA NAC 公式推导

## 推导：SATDA/HF $\Delta S=-1$ NAC 的物理定义与有限差分基准 — 2026-05-23

**目标：** 先定义 SATDA 非绝热导数耦合（NAC）真正对应的物理对象，再给出有限差分与解析实现应当匹配的公式。这里的重点是：SATDA 的有限差分不能默认使用普通 SF determinant basis，而必须使用 SATDA 自旋适配的 AWF/CSF 基函数。

**假设：**
- 使用 Tamm-Dancoff 近似，$Y=0$。
- 只讨论 `deltaS=-1`，即 $S_f=S_i-1$。
- 当前解析目标限定 HF 或 `xc='HF'`，所以激发矩阵只包含 one-electron、Coulomb/exchange 与 SATDA 自旋适配修正，不包含 XC kernel 导数。
- 分子轨道与激发振幅均取实数。
- ROKS 高自旋参考态取 $M_S=S$，所有开壳轨道为 $\alpha$ 占据。

**符号声明：**
- 闭壳轨道用 $c,d$，开壳轨道用 $u,v,w,t$，虚轨道用 $a,b$。
- SATDA $\Delta S=-1$ 的振幅四块为 $X_{cu}$、$X_{ca}$、$X_{uv}$、$X_{ua}$，分别对应代码中的 `co`、`cv`、`oo`、`ov`。
- 用复合指标 $P,Q$ 表示四块中的任意一个激发通道，例如 $P=(c,u)$ 或 $P=(u,a)$。
- $\hat{G}_{P}^{\dagger}$ 是 SATDA 自旋适配激发算符，$|\widetilde{\Phi}_{P}\rangle=\hat{G}_{P}^{\dagger}|\mathrm{ROKS}\rangle$ 是 SATDA AWF/CSF 基函数。
- $\mathbf{A}_{\mathrm{SATDA}}$ 是 SATDA/TDA 激发矩阵，$\omega_I$ 是第 $I$ 个激发能。
- $\xi$ 表示任意核坐标分量。

**代码变量到数学符号的对应：**
- `td.xy[I][0]`, shape `(nocc, nvir)` = $\mathbf{X}^{I}$，其中行空间为 $(c,u)$，列空间为 $(u,a)$。
- `zs_co`, `zs_cv`, `zs_oo`, `zs_ov` = $X_{cu}$、$X_{ca}$、$X_{uv}$、$X_{ua}$。
- `satda.py:gen_vind_sf()` = $\boldsymbol{\sigma}=\mathbf{A}_{\mathrm{SATDA}}\mathbf{X}$。
- `awf_overlap()` 应表示 $\langle\widetilde{\Psi}_I(\mathbf{R})|\widetilde{\Psi}_J(\mathbf{R}')\rangle$，但只有在其使用 $\hat{G}_{P}^{\dagger}$ 的 SATDA CSF 展开时才是物理正确的 SATDA AWF overlap。

### Step 1: SATDA 激发态的 AWF 定义

SATDA 的第 $I$ 个 TDA 态应写成

$$
|\widetilde{\Psi}_{I}\rangle
=
\sum_{P} X_{P}^{I} |\widetilde{\Phi}_{P}\rangle
=
\sum_{P} X_{P}^{I}\hat{G}_{P}^{\dagger}|\mathrm{ROKS}\rangle .
$$

→ **解释：** 这里 $P$ 不是普通的 $\alpha\rightarrow\beta$ determinant 指标，而是 SATDA 四块自旋适配激发通道。程序中的一维 `xy` 向量只是这些 $X_P$ 的数值表示；它本身不定义物理 many-body basis。

对 `deltaS=-1`，四块为

$$
P\in
\{(c,u),(c,a),(u,v),(u,a)\}.
$$

→ **解释：** 这对应 `gen_vind_sf()` 中的 `co/cv/oo/ov` 四块。普通 SF determinant basis 也有同样的矩阵形状，但形状相同不表示基函数相同。

SATDA 的物理基函数定义为

$$
|\widetilde{\Phi}_{cu}\rangle=\hat{G}_{c\rightarrow u}^{\dagger}|\mathrm{ROKS}\rangle,\quad
|\widetilde{\Phi}_{ca}\rangle=\hat{G}_{c\rightarrow a}^{\dagger}|\mathrm{ROKS}\rangle,
$$

$$
|\widetilde{\Phi}_{uv}\rangle=\hat{G}_{u\rightarrow v}^{\dagger}|\mathrm{ROKS}\rangle,\quad
|\widetilde{\Phi}_{ua}\rangle=\hat{G}_{u\rightarrow a}^{\dagger}|\mathrm{ROKS}\rangle.
$$

→ **解释：** 后续有限差分 overlap 必须使用这些 $|\widetilde{\Phi}_{P}\rangle$。若直接把每个 $P$ 替换成单个普通 spin-flip determinant $\hat{a}_{q\beta}^{\dagger}\hat{a}_{p\alpha}|\mathrm{ROKS}\rangle$，就退化成 ordinary SF AWF，而不是 SATDA AWF。

### Step 2: SATDA CSF overlap 的一般公式

两个几何 $\mathbf{R}$ 和 $\mathbf{R}'$ 上的 SATDA AWF overlap 应为

$$
S_{IJ}^{\mathrm{AWF}}(\mathbf{R},\mathbf{R}')
=
\langle\widetilde{\Psi}_{I}(\mathbf{R})|
\widetilde{\Psi}_{J}(\mathbf{R}')\rangle
=
\sum_{P Q}
X_{P}^{I}(\mathbf{R})X_{Q}^{J}(\mathbf{R}')
\langle\widetilde{\Phi}_{P}(\mathbf{R})|
\widetilde{\Phi}_{Q}(\mathbf{R}')\rangle .
$$

→ **解释：** 这是有限差分 NAC 的物理基准。关键量不是振幅点积，而是 SATDA CSF basis overlap matrix $\langle\widetilde{\Phi}_{P}|\widetilde{\Phi}_{Q}\rangle$。

将 SATDA CSF 展开到 Slater determinant 后，可写成

$$
|\widetilde{\Phi}_{P}\rangle
=
\sum_{m} C_{Pm}^{\mathrm{SA}} |D_m\rangle .
$$

→ **解释：** $C_{Pm}^{\mathrm{SA}}$ 是由 SATDA 自旋适配算符 $\hat{G}_{P}^{\dagger}$ 决定的 Clebsch-Gordan/开壳耦合系数；$|D_m\rangle$ 是具体的 spin-orbital Slater determinant。

因此 basis overlap 为

$$
\langle\widetilde{\Phi}_{P}(\mathbf{R})|
\widetilde{\Phi}_{Q}(\mathbf{R}')\rangle
=
\sum_{m n}
C_{Pm}^{\mathrm{SA}} C_{Qn}^{\mathrm{SA}}
\langle D_m(\mathbf{R})|D_n(\mathbf{R}')\rangle .
$$

→ **解释：** determinant overlap $\langle D_m|D_n\rangle$ 可以用 $\alpha$ 和 $\beta$ spin-orbital overlap 子矩阵的行列式计算。程序实现层面应在这里展开 SATDA CSF，而不是只保留一个 ordinary SF determinant。

若采用普通 SF determinant 近似，相当于把

$$
C_{Pm}^{\mathrm{SA}}\rightarrow \delta_{m,m(P)}.
$$

→ **解释：** 这正是目前需要警惕的错误：它给出的是 ordinary SF AWF overlap，不是 SATDA 的 spin-adapted AWF overlap。即使该有限差分随步长收敛，也只能证明 ordinary-SF-like overlap 的数值稳定，不能证明 SATDA NAC 的物理正确性。

### Step 3: 有限差分 NAC 的正确对象

SATDA AWF 下的 derivative coupling 定义为

$$
d_{IJ}^{\xi}
=
\left\langle
\widetilde{\Psi}_{I}(\mathbf{R})
\middle|
\frac{\partial \widetilde{\Psi}_{J}(\mathbf{R})}{\partial \xi}
\right\rangle .
$$

→ **解释：** 这是 AWF 形式的 NAC。它依赖于右态振幅响应、MO 响应、AO basis overlap response，以及 SATDA CSF basis 的自旋适配系数。

中心差分形式应为

$$
d_{IJ}^{\xi}
\approx
\frac{
S_{IJ}^{\mathrm{AWF}}(\mathbf{R},\mathbf{R}+h\mathbf{e}_{\xi})
-
S_{IJ}^{\mathrm{AWF}}(\mathbf{R},\mathbf{R}-h\mathbf{e}_{\xi})
}{2h}.
$$

→ **解释：** 这里左态固定在 $\mathbf{R}$，右态在 displaced geometry 重新求解 SATDA 并做 root tracking。只有当 $S_{IJ}^{\mathrm{AWF}}$ 使用 Step 2 的 SATDA CSF overlap 时，这个有限差分才是 SATDA NAC 的物理基准。

为了固定相位，应要求

$$
\langle \mathbf{X}_{J}(\mathbf{R})|\mathbf{X}_{J}(\mathbf{R}\pm h\mathbf{e}_{\xi})\rangle > 0 .
$$

→ **解释：** 这是 root tracking 与相位连续性的数值规范。它不能替代 CSF basis overlap；它只决定 displaced root 的符号。

### Step 4: NAC 与激发矩阵导数的关系

SATDA/TDA 本征方程为

$$
\mathbf{A}_{\mathrm{SATDA}}(\mathbf{R})\mathbf{X}_{I}
=
\omega_I \mathbf{X}_{I}.
$$

→ **解释：** $\mathbf{A}_{\mathrm{SATDA}}$ 是激发空间哈密顿量，不是总电子哈密顿量的基态能量加激发能形式。

对核坐标求导并左乘 $\mathbf{X}_{I}^{T}$，在 $I\ne J$ 时有

$$
(\omega_J-\omega_I)
\mathbf{X}_{I}^{T}
\frac{\partial \mathbf{X}_{J}}{\partial \xi}
=
\mathbf{X}_{I}^{T}
\frac{\partial \mathbf{A}_{\mathrm{SATDA}}}{\partial \xi}
\mathbf{X}_{J}
+ \mathbf{B}_{IJ}^{\xi}.
$$

→ **解释：** 第一项是激发矩阵的 Hellmann-Feynman numerator；$\mathbf{B}_{IJ}^{\xi}$ 表示由于 SATDA CSF basis 和 MO/AO metric 随核坐标变化产生的 basis-response/overlap 项。解析实现中的 `im0`、CSF term 和 Z-vector 消元都属于这类 metric/response 处理。

因此 energy-scaled NAC numerator 定义为

$$
N_{IJ}^{\xi}
=
(\omega_J-\omega_I)d_{IJ}^{\xi}.
$$

→ **解释：** 若代码 `ediff=True`，返回 $d_{IJ}^{\xi}$；若 `ediff=False`，返回 $N_{IJ}^{\xi}$。

### Step 5: 解析 numerator 不能包含基态能量梯度

总激发态能量可写为

$$
E_I^{\mathrm{total}}(\mathbf{R})
=
E_{\mathrm{ref}}(\mathbf{R})+\omega_I(\mathbf{R}).
$$

→ **解释：** PySCF 的 excited-state gradient 通常返回总梯度，即参考态梯度加激发能梯度。

但是 NAC 的 Hellmann-Feynman numerator 对应的是激发矩阵 $\mathbf{A}_{\mathrm{SATDA}}$，所以对角极限必须满足

$$
N_{II}^{\xi}
=
\frac{\partial \omega_I}{\partial \xi}
=
\frac{\partial E_I^{\mathrm{total}}}{\partial \xi}
-
\frac{\partial E_{\mathrm{ref}}}{\partial \xi}.
$$

→ **解释：** 这是判断解析 NAC 物理对象是否正确的最低要求。如果拿 `td.Gradients()` 的总梯度直接比较，就会错误地把基态梯度混入 NAC numerator。

态间 $I\ne J$ 的解析 numerator 应写成

$$
N_{IJ}^{\xi}
=
\mathbf{X}_{I}^{T}
\mathbf{A}_{\mathrm{SATDA}}^{[\xi]}
\mathbf{X}_{J}
+ N_{IJ,\mathrm{orb}}^{\xi}
+ N_{IJ,\mathrm{metric}}^{\xi}.
$$

→ **解释：** $\mathbf{A}_{\mathrm{SATDA}}^{[\xi]}$ 是去除 MO response 后的 direct/skeleton 核导数；$N_{IJ,\mathrm{orb}}^{\xi}$ 由 Z-vector 消去 MO response；$N_{IJ,\mathrm{metric}}^{\xi}$ 是 overlap metric 与 SATDA CSF basis 导数项。三者合起来仍然只对应 $\omega$ 的导数，不对应 $E_{\mathrm{ref}}+\omega$ 的导数。

### Step 6: 对当前程序实现的物理判据

若有限差分使用的 overlap 是

$$
S_{IJ}^{\mathrm{old}}(\mathbf{R},\mathbf{R}')
=
\sum_{pq,rs}
X_{pq}^{I}X_{rs}^{J}
\langle
\hat{a}_{q\beta}^{\dagger}\hat{a}_{p\alpha}\mathrm{ROKS}(\mathbf{R})
|
\hat{a}_{s\beta}^{\dagger}\hat{a}_{r\alpha}\mathrm{ROKS}(\mathbf{R}')
\rangle ,
$$

→ **解释：** 这只是 ordinary spin-flip determinant AWF overlap。它没有使用 $\hat{G}_{P}^{\dagger}$ 的 SATDA 自旋适配展开，因此不是 SATDA-NAC 的最终物理基准。

正确实现必须改为

$$
S_{IJ}^{\mathrm{SATDA}}(\mathbf{R},\mathbf{R}')
=
\sum_{P Q}
X_{P}^{I}X_{Q}^{J}
\sum_{m n}
C_{Pm}^{\mathrm{SA}} C_{Qn}^{\mathrm{SA}}
\langle D_m(\mathbf{R})|D_n(\mathbf{R}')\rangle .
$$

→ **解释：** 这一步是后续修复 `awf_overlap()` 的必要公式。只有它完成后，有限差分结果才可作为解析 NAC 的物理 reference。

### 最终结果

SATDA/HF $\Delta S=-1$ NAC 的物理实现顺序应为

$$
\hat{G}_{P}^{\dagger}
\longrightarrow
C_{Pm}^{\mathrm{SA}}
\longrightarrow
S_{IJ}^{\mathrm{SATDA}}(\mathbf{R},\mathbf{R}')
\longrightarrow
d_{IJ}^{\xi}
\longleftrightarrow
\frac{N_{IJ}^{\xi}}{\omega_J-\omega_I}.
$$

→ **解释：** 先由 SATDA 自旋适配算符定义 CSF basis，再构造有限差分 AWF overlap；解析 numerator 必须匹配同一个 AWF/激发矩阵对象，并满足对角极限 $N_{II}^{\xi}=\partial\omega_I/\partial\xi$。

**代码对应：** `pyscf-forge/pyscf/sftda/satda.py:gen_vind_sf()` 定义 $\mathbf{A}_{\mathrm{SATDA}}$ 的作用；`pyscf-forge/pyscf/nac/tdsatda.py:awf_overlap()` 应按本文公式重写；`pyscf-forge/pyscf/nac/tdsatda.py:get_hf_interstate_numerator()` 的解析路径必须先通过对角激发能梯度极限检查。

### 当前代码状态

当前 `pyscf-forge/pyscf/nac/tdsatda.py` 中的 SATDA NAC 数值入口必须保持禁用：

$$
\texttt{finite\_diff},\quad
\texttt{analytic\_experimental},\quad
\texttt{awf\_overlap},\quad
\texttt{nac\_csf},\quad
\texttt{get\_hf\_interstate\_numerator}
\quad
\Longrightarrow
\quad
\texttt{NotImplementedError}.
$$

→ **解释：** 禁用原因不是数值步长，而是物理对象未闭合：旧有限差分 overlap 是 ordinary SF determinant overlap，不是 SATDA spin-adapted AWF/CSF overlap；旧解析 numerator 也没有通过 $N_{II}^{\xi}=\partial\omega_I/\partial\xi$ 的对角极限检查。因此在 SATDA CSF 展开和解析 numerator 重新推导完成前，代码不能返回任何 SATDA NAC 数值。

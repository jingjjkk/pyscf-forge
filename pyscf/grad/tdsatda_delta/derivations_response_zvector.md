## 推导：用 `gen_response` 构造 ROKS z-vector Hessian action — 2026-05-31

**目标：** 判断当前 `tdsatda_delta` 的 ROKS z-vector 方程能否用 PySCF 的 `gen_response` 构造，而不是手动组装完整 Hessian 矩阵。

**假设：**
- 使用实轨道、TDA、ROKS 参考态。
- 当前讨论的是 HF/SATDA delta correction 的轨道响应 z-vector，不是 SATDA 激发空间本征方程。
- 空间轨道转动写作反对称矩阵 $\boldsymbol{\kappa}$。

**符号声明：**
- $C_{\mu p}$ = 空间 MO 系数。
- $n_p^{\alpha}$、$n_p^{\beta}$ = ROKS 参考态中 alpha/beta 占据数。
- $\mathbf{F}^{\alpha}$、$\mathbf{F}^{\beta}$ = alpha/beta Fock 矩阵。
- $\mathbf{V}_{\mathrm{resp}}^{\sigma}[\delta\mathbf{D}^{\alpha},\delta\mathbf{D}^{\beta}]$ = `mf.gen_response(hermi=1)` 返回的有效势响应。
- $R_{pq}$ = ROKS Brillouin residual 在空间转动变量上的投影。

**代码变量到数学符号的对应：**
- `_anti_mo_from_roks_canonical_vec(..., vec)` = 从 packed canonical pair vector 构造 $\boldsymbol{\kappa}$。
- `_pack_roks_canonical_residual(pairs, fmo_a, fmo_b)` = 将 spin-resolved MO residual 投影到 ROKS 空间转动变量。
- `mf.gen_response(hermi=1)` = $\mathbf{V}_{\mathrm{resp}}[\delta\mathbf{D}]$。
- `_roks_general_orbital_action_hf` = 当前手写的 $H\boldsymbol{\kappa}$。

### Step 1: ROKS 空间转动导致的密度响应

对一个空间轨道转动 $\boldsymbol{\kappa}$，spin-resolved 密度一阶变化为

$$
\delta \mathbf{D}^{\sigma}
=
\mathbf{C}
\left(
\boldsymbol{\kappa}\mathbf{n}^{\sigma}
+ \mathbf{n}^{\sigma}\boldsymbol{\kappa}^{T}
\right)
\mathbf{C}^{T}.
$$

→ **解释：** 这是 `C -> C(I+\kappa)` 下占据投影矩阵 $\mathbf{n}^{\sigma}$ 的一阶变化；ROKS 只有一个空间 $\boldsymbol{\kappa}$，但 $\mathbf{n}^{\alpha}$ 和 $\mathbf{n}^{\beta}$ 不同，所以 CO/CV/OV 三类转动进入不同 spin 通道。

其中 CO、CV、OV 的 spin 归属为

$$
\begin{aligned}
\mathrm{CO}:&\quad \delta\mathbf{D}^{\alpha}=0,\quad \delta\mathbf{D}^{\beta}\ne 0,\\
\mathrm{CV}:&\quad \delta\mathbf{D}^{\alpha}\ne 0,\quad \delta\mathbf{D}^{\beta}\ne 0,\\
\mathrm{OV}:&\quad \delta\mathbf{D}^{\alpha}\ne 0,\quad \delta\mathbf{D}^{\beta}=0.
\end{aligned}
$$

→ **解释：** core 在 alpha/beta 都占据，open 只在 alpha 占据，virtual 两个 spin 都不占据；因此同一个空间转动在不同自旋通道中的物理含义不同。

### Step 2: `gen_response` 能提供的部分

给定上面的 $\delta\mathbf{D}^{\alpha},\delta\mathbf{D}^{\beta}$，PySCF 的
`mf.gen_response(hermi=1)` 可以提供

$$
\delta \mathbf{V}^{\sigma}
=
\mathbf{V}_{\mathrm{resp}}^{\sigma}
\left[
\delta\mathbf{D}^{\alpha},\delta\mathbf{D}^{\beta}
\right].
$$

→ **解释：** HF 情况下这是 Coulomb/exchange 响应；DFT 情况下还应包含 XC kernel 响应。也就是说，`gen_response` 可以替换手写的 `get_jk` 线性响应部分。

但完整的 MO residual 变化还包含 Fock 对易子项：

$$
\delta \mathbf{F}_{\mathrm{MO}}^{\sigma}
=
\boldsymbol{\kappa}^{T}\mathbf{F}_{\mathrm{MO}}^{\sigma}
+ \mathbf{F}_{\mathrm{MO}}^{\sigma}\boldsymbol{\kappa}
+ \mathbf{C}^{T}\delta\mathbf{V}^{\sigma}\mathbf{C}.
$$

→ **解释：** 前两项来自 MO 基本身的转动，不是密度响应核；`gen_response` 不会自动给出这部分，所以外层仍需显式加上 Fock 对易子。

### Step 3: ROKS 变量投影

完整 action 不是直接返回 alpha/beta VO 块，而是投影到 ROKS 空间变量：

$$
H\boldsymbol{\kappa}
=
\mathcal{P}_{\mathrm{ROKS}}
\left[
\delta \mathbf{F}_{\mathrm{MO}}^{\alpha},
\delta \mathbf{F}_{\mathrm{MO}}^{\beta}
\right].
$$

→ **解释：** $\mathcal{P}_{\mathrm{ROKS}}$ 对 CO 使用 beta residual，对 OV 使用 alpha residual，对 CV 使用 alpha+beta residual；这一步正是普通 UKS `ucphf.solve` 不能直接替代的原因，因为 UKS 把 $\kappa_{\mathrm{CV}}^{\alpha}$ 和 $\kappa_{\mathrm{CV}}^{\beta}$ 当成两个独立变量。

当前 canonical 实现还包含 CC/OO/VV 规范转动：

$$
\mathcal{P}_{\mathrm{canon}}:
\{\mathrm{CC},\mathrm{OO},\mathrm{VV},\mathrm{CO},\mathrm{CV},\mathrm{OV}\}.
$$

→ **解释：** CC/OO/VV 对密度响应为零，但对 canonical MO 标签的导数可通过 Fock 对易子确定；如果保留当前 canonical displaced-SCF 对比逻辑，新的 `gen_response` action 也必须保留这些 pair，而不能只解 CO/CV/OV。

### Step 4: z-vector 方程

若 $M_{pq}$ 是 SATDA delta correction 对 MO 系数的 orbital derivative 系数，则 packed RHS 为

$$
m_I
=
\left(M_{pq}-M_{qp}\right)_{I}.
$$

→ **解释：** $I$ 表示 packed canonical pair；这是当前 `pack_mvec` 的定义。

z-vector 方程为

$$
H^{T}\mathbf{z}=\mathbf{m}.
$$

→ **解释：** 如果 packed ROKS Hessian 在当前度量下严格对称，可以用同一个 action 迭代求解；否则需要显式 adjoint action 或继续小体系 materialize $H$ 后使用 $H^{T}$。因此替换前必须测试 $u^{T}H v$ 与 $v^{T}H u$。

核坐标扰动下，重叠诱导的对称轨道响应写作

$$
\boldsymbol{\kappa}_{\mathrm{sym}}^{[x]}
=
-\frac{1}{2}\mathbf{C}^{T}\mathbf{S}^{[x]}\mathbf{C}.
$$

→ **解释：** 这一项不是 CPHF 解变量，而是移动 AO 基下的正交归一条件直接给出的对称部分。

因此 orbital response 梯度可写为

$$
E_{\mathrm{orb}}^{[x]}
=
\mathbf{z}^{T}
\left[
-\mathbf{g}_{\mathrm{fix}}^{[x]}
-H\boldsymbol{\kappa}_{\mathrm{sym}}^{[x]}
\right]
+\mathrm{Tr}\left[
\mathbf{M}\boldsymbol{\kappa}_{\mathrm{sym}}^{[x]}
\right].
$$

→ **解释：** `gen_response` 可以用于计算 $H\boldsymbol{\kappa}_{\mathrm{sym}}^{[x]}$，但 $\mathbf{g}_{\mathrm{fix}}^{[x]}$ 仍来自显式核导数，包括 $h^{[x]}$、ERI skeleton derivative 以及未来 DFT 的 grid/XC skeleton derivative。

### 最终结果

可以用 `gen_response` 构造 z-vector Hessian action，但不能只调用 `gen_response` 就得到完整 z-vector 方程。可替换的是

$$
\delta\mathbf{D}\longmapsto
\delta\mathbf{V}_{\mathrm{eff}}
$$

→ **解释：** 这对应当前手写 Hessian 中的 J/K 或未来 XC kernel 响应部分。

仍需手写或保留的外层为

$$
\boldsymbol{\kappa}
\longmapsto
\delta\mathbf{D}^{\alpha/\beta}
\longmapsto
\delta\mathbf{V}^{\alpha/\beta}
\longmapsto
\delta\mathbf{F}_{\mathrm{MO}}^{\alpha/\beta}
\longmapsto
\mathcal{P}_{\mathrm{ROKS}}
\left[
\delta\mathbf{F}_{\mathrm{MO}}^{\alpha},
\delta\mathbf{F}_{\mathrm{MO}}^{\beta}
\right].
$$

→ **解释：** 这个包装层定义了 ROKS 的 CO/CV/OV 物理变量和 canonical CC/OO/VV 规范变量；它不能由普通 UKS `ucphf.solve` 自动处理。

**代码对应：** `pyscf/grad/tdsatda_delta/_zvec_solver.py:build_roks_hessian`、`pyscf/grad/tdsatda_delta/_block_analytic_hf.py:_roks_general_orbital_action_hf`、`pyscf/grad/tdsatda_delta/_roks.py:make_roks_hessian_action_hf`

## 推导：为什么不能直接用 `satda.py:gen_vind_sf` 作为 z-vector Hessian — 2026-05-31

**目标：** 判断 `pyscf/sftda/satda.py` 中的 `gen_vind_sf` 是否可以直接用来构造梯度 z-vector 方程。

**假设：**
- `gen_vind_sf` 对应 SATDA $\Delta S=-1$ 的激发态 TDA 工作方程。
- 梯度 z-vector 方程对应 ROKS 基态轨道响应方程。

**符号声明：**
- $\mathbf{X}$ = SATDA 激发振幅。
- $\boldsymbol{\kappa}$ = ROKS 空间轨道转动。
- $\mathbf{A}_{\mathrm{SATDA}}$ = SATDA 激发空间矩阵。
- $\mathbf{H}_{\mathrm{orb}}$ = ROKS 轨道 Hessian。

**代码变量到数学符号的对应：**
- `satda.py:gen_vind_sf` 返回的 `vind` = $\mathbf{A}_{\mathrm{SATDA}}\mathbf{X}$。
- `tdsatda_delta/_zvec_solver.py:build_roks_hessian` = materialize $\mathbf{H}_{\mathrm{orb}}$。
- `tdsatda_delta/_block_analytic_hf.py:_roks_general_orbital_action_hf` = $\mathbf{H}_{\mathrm{orb}}\boldsymbol{\kappa}$。

### Step 1: `gen_vind_sf` 的变量空间

`gen_vind_sf` 的输入振幅可以写为

$$
\mathbf{X}
=
\begin{pmatrix}
\mathbf{X}_{\mathrm{CO}} & \mathbf{X}_{\mathrm{CV}} \\
\mathbf{X}_{\mathrm{OO}} & \mathbf{X}_{\mathrm{OV}}
\end{pmatrix},
$$

其中行空间为 $\mathrm{C}\oplus\mathrm{O}$，列空间为 $\mathrm{O}\oplus\mathrm{V}$。

→ **解释：** 这是 $\Delta S=-1$ spin-flip 激发空间；open 轨道在 beta 通道是 virtual，因此列空间包含 O 和 V。这里的 OO 是激发振幅块，不是 ROKS 空间轨道的 open-open 反对称规范转动。

`gen_vind_sf` 计算的是

$$
\mathbf{Y}
=
\mathbf{A}_{\mathrm{SATDA}}\mathbf{X}.
$$

→ **解释：** 这个算符用于求 SATDA 激发能和激发矢量；它是激发空间的响应矩阵，不是基态 SCF 稳定性/CPHF 轨道 Hessian。

### Step 2: z-vector 的变量空间

梯度 z-vector 需要的变量是 ROKS 空间轨道转动

$$
\boldsymbol{\kappa}
=
\boldsymbol{\kappa}_{\mathrm{CO}}
+\boldsymbol{\kappa}_{\mathrm{CV}}
+\boldsymbol{\kappa}_{\mathrm{OV}}
+\boldsymbol{\kappa}_{\mathrm{CC}}
+\boldsymbol{\kappa}_{\mathrm{OO}}
+\boldsymbol{\kappa}_{\mathrm{VV}},
$$

其中每个块都满足整体反对称条件 $\kappa_{pq}=-\kappa_{qp}$。

→ **解释：** CO/CV/OV 是非冗余 ROKS 轨道响应；CC/OO/VV 是 canonical gauge 转动。这个空间不是 `gen_vind_sf` 的 $(\mathrm{C}\oplus\mathrm{O})\rightarrow(\mathrm{O}\oplus\mathrm{V})$ 激发空间。

z-vector 方程是

$$
\mathbf{H}_{\mathrm{orb}}^{T}\mathbf{z}
=
\mathbf{m},
$$

→ **解释：** $\mathbf{H}_{\mathrm{orb}}$ 是基态 ROKS Brillouin residual 对空间轨道转动的一阶导数；右端 $\mathbf{m}$ 才来自 SATDA delta correction 的 orbital derivative。

### Step 3: 两个算符虽然相似，但不能互换

`gen_vind_sf` 中的算符结构可抽象为

$$
\mathbf{A}_{\mathrm{SATDA}}
=
\mathbf{A}_{\mathrm{Fock}}^{\mathrm{spin-flip}}
+\mathbf{A}_{\mathrm{resp}}^{\mathrm{spin-flip}},
$$

→ **解释：** 这些项按照 SATDA Table 的 CO/CV/OO/OV 激发块组合，包括 spin-adaptation 系数和 spin-flip transition-density response。

而 ROKS 轨道 Hessian action 为

$$
\mathbf{H}_{\mathrm{orb}}\boldsymbol{\kappa}
=
\mathcal{P}_{\mathrm{ROKS}}
\left[
\boldsymbol{\kappa}^{T}\mathbf{F}^{\alpha}
+\mathbf{F}^{\alpha}\boldsymbol{\kappa}
+\mathbf{C}^{T}\mathbf{V}_{\mathrm{resp}}^{\alpha}[\delta\mathbf{D}]\mathbf{C},
\boldsymbol{\kappa}^{T}\mathbf{F}^{\beta}
+\mathbf{F}^{\beta}\boldsymbol{\kappa}
+\mathbf{C}^{T}\mathbf{V}_{\mathrm{resp}}^{\beta}[\delta\mathbf{D}]\mathbf{C}
\right].
$$

→ **解释：** 这里的 response density 是由空间轨道转动产生的基态密度变化，不是 SATDA spin-flip transition density；最后还要用 ROKS 的 CO/CV/OV/canonical 投影。

因此一般有

$$
\mathbf{A}_{\mathrm{SATDA}}
\ne
\mathbf{H}_{\mathrm{orb}}.
$$

→ **解释：** 即使二者都使用 C/O/V 分块和 `fock0/fockz`，它们的变量空间、密度扰动、spin 投影和物理含义都不同。

### 最终结果

`gen_vind_sf` 不能直接作为梯度 z-vector 方程的 Hessian action。它可以借鉴的部分是：

$$
\texttt{gen_rohf_response_sf},\quad
\mathbf{F}^{0}/\mathbf{F}^{z}\text{ 的构造方式},\quad
\text{SATDA functional block 组合规则}.
$$

→ **解释：** 这些对后续 DFT/XC 梯度非常有参考价值；但 z-vector 方程仍应构造 ROKS 轨道 Hessian action，即 $\boldsymbol{\kappa}\rightarrow\delta\mathbf{D}\rightarrow\delta\mathbf{V}\rightarrow\mathcal{P}_{\mathrm{ROKS}}(\delta\mathbf{F}_{\mathrm{MO}})$。

如果要复用 `gen_vind_sf` 的代码，合理方式不是直接调用 `vind(z)`，而是抽出或仿写其 response-kernel 和 `fock0/fockz` block 组织，让它们服务于 ROKS orbital Hessian。

**代码对应：** `pyscf/sftda/satda.py:gen_vind_sf`、`pyscf/grad/tdsatda_delta/_zvec_solver.py:build_roks_hessian`

## 推导：`satda.py:gen_rohf_response_sf` 能否直接作为 z-vector Hessian — 2026-05-31

**目标：** 判断 `pyscf/sftda/satda.py:gen_rohf_response_sf` 是否可以替代 ground-state `mf.gen_response` 来构造梯度 z-vector 方程。

**假设：**
- `gen_rohf_response_sf` 用于 SATDA 的 $\Delta S=-1$ 激发空间矩阵-向量乘。
- 梯度 z-vector 方程约束的是参考态 ROKS 轨道响应，而不是激发振幅响应。

**符号声明：**
- $\mathbf{x}_{\mathrm{CO}}$、$\mathbf{x}_{\mathrm{CV}}$、$\mathbf{x}_{\mathrm{OO}}$、$\mathbf{x}_{\mathrm{OV}}$ = SATDA 激发空间振幅块。
- $\boldsymbol{\kappa}$ = ground-state ROKS 空间轨道转动。
- $\mathcal{A}_{\mathrm{SATDA}}$ = SATDA/TDA 激发空间 Hessian。
- $\mathcal{H}_{\mathrm{ROKS}}$ = ground-state ROKS orbital Hessian。

**代码变量到数学符号的对应：**
- `gen_rohf_response_sf(...): vind(dms_co, dms_cv, dms_oo, dms_ov)` = SATDA 激发空间 effective-potential action。
- `satda.py:gen_vind_sf` = 构造 $\mathcal{A}_{\mathrm{SATDA}}\mathbf{x}$。
- `_zvec_solver.py:build_roks_hessian` = 构造 $\mathcal{H}_{\mathrm{ROKS}}$。

### Step 1: `gen_rohf_response_sf` 的变量空间

`gen_rohf_response_sf` 中输入密度由 SATDA 振幅生成：

$$
\begin{aligned}
\mathbf{D}_{\mathrm{CO}} &= C_{\mathrm{O}}\mathbf{x}_{\mathrm{CO}}^{T}C_{\mathrm{C}}^{T},\\
\mathbf{D}_{\mathrm{CV}} &= C_{\mathrm{V}}\mathbf{x}_{\mathrm{CV}}^{T}C_{\mathrm{C}}^{T},\\
\mathbf{D}_{\mathrm{OO}} &= C_{\mathrm{O}}\mathbf{x}_{\mathrm{OO}}^{T}C_{\mathrm{O}}^{T},\\
\mathbf{D}_{\mathrm{OV}} &= C_{\mathrm{V}}\mathbf{x}_{\mathrm{OV}}^{T}C_{\mathrm{O}}^{T}.
\end{aligned}
$$

→ **解释：** 这些是 SATDA 激发空间 transition density，不是参考态密度对空间轨道转动 $\boldsymbol{\kappa}$ 的响应。

因此它构造的是

$$
\mathbf{x}
\longmapsto
\mathcal{A}_{\mathrm{SATDA}}\mathbf{x}.
$$

→ **解释：** 这正是 `satda.py:gen_vind_sf` 需要的 action；其中包含 SATDA 特有的 spin-adapted block 组合系数。

### Step 2: 梯度 z-vector 方程的变量空间

梯度 z-vector 方程需要的是参考态 orbital Hessian：

$$
\boldsymbol{\kappa}
\longmapsto
\mathcal{H}_{\mathrm{ROKS}}\boldsymbol{\kappa}
=
\mathcal{P}_{\mathrm{ROKS}}
\left[
\delta\mathbf{F}_{\mathrm{MO}}^{\alpha},
\delta\mathbf{F}_{\mathrm{MO}}^{\beta}
\right].
$$

→ **解释：** 这里的输入是 ground-state orbital rotation；输出是 Brillouin residual 的变化。它不是 SATDA 激发矩阵作用。

对于一个 $\boldsymbol{\kappa}$，密度变化是

$$
\delta \mathbf{D}^{\sigma}
=
\mathbf{C}
\left(
\boldsymbol{\kappa}\mathbf{n}^{\sigma}
+\mathbf{n}^{\sigma}\boldsymbol{\kappa}^{T}
\right)
\mathbf{C}^{T}.
$$

→ **解释：** 这个密度响应由参考态占据矩阵决定；它不等价于 `gen_rohf_response_sf` 的 `dms_co/dms_cv/dms_oo/dms_ov` transition density 组合。

### Step 3: 二者不能直接互换

若直接把 `gen_rohf_response_sf` 当作 z-vector Hessian，会把

$$
\mathcal{H}_{\mathrm{ROKS}}
\quad\text{误替换为}\quad
\mathcal{A}_{\mathrm{SATDA}}.
$$

→ **解释：** 这两个算符的物理对象不同：前者是 ground-state SCF stationarity 的二阶导数，后者是激发能工作方程矩阵。即使两者都含有 $f_0/f_z$ 和类似的 CO/CV/OO/OV 块，也不能在 z-vector 方程中直接替代。

### 最终结果

`gen_rohf_response_sf` 不能直接作为当前梯度 z-vector 方程的 Hessian action。它能安全复用的部分有两类：

$$
\begin{aligned}
&\text{1. 从 `gen_rohf_response_sf` 取得 DFT 下更一致的 } \mathbf{F}^{z}
\text{，用于 } \mathbf{F}^{0}/\mathbf{F}^{z}\text{ 表示；}\\
&\text{2. 在推导和实现 SATDA } A^{[x]}\text{ 的 direct/M-matrix 块时，复用其 block 组合规则。}
\end{aligned}
$$

→ **解释：** z-vector Hessian 仍应由 ground-state ROKS orbital-response action 构造；其中的有效势响应层可以用普通 `mf.gen_response` 或未来写一个 ROKS orbital-response wrapper。`gen_rohf_response_sf` 更适合作为 SATDA 激发空间和 DFT block 组合的基准，而不是 orbital z-vector solver 的直接替代。

**代码对应：** `pyscf/sftda/satda.py:gen_rohf_response_sf`、`pyscf/sftda/satda.py:gen_vind_sf`、`pyscf/grad/tdsatda_delta/_zvec_solver.py:build_roks_hessian`

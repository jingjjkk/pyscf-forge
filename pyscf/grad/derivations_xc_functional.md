## 推导：SASF/HF 梯度加入 XC 泛函项的结构差异 — 2026-05-31

**目标：** 从 `pyscf-forge/pyscf/sftda/SASFTDA_Rev(2).pdf` 和 `pyscf-forge/pyscf/sftda/satda.py` 的 `deltaS=-1` 工作方程出发，说明相对于 HF exact-exchange 情形，SASF 梯度的 XC 泛函项新增哪些对象、哪些矩阵元组合，以及为什么不能简单复用普通 SF-TDDFT 的 `_contract_xc_kernel`。

**假设：**
- 只讨论当前已经完成 HF 解析梯度的同一物理情形：ROKS/ROHF 型 open-shell 参考态、TDA、`deltaS=-1`、实轨道。
- 先推 LDA/GGA 的二阶响应结构；MGGA 与 NLC 暂不作为第一阶段实现目标。
- 记 $S$ 为初态自旋量子数，且 `deltaS=-1` 要求 $S \ge 1$。
- 下列公式只描述 SASF 修正的 XC 结构；普通 SF-TDA 基础梯度仍由已验证的 SF 梯度路径承担。

**符号声明：**
- $C,O,V$ 分别表示 core、open、virtual 空间。
- $D^{X}$ 表示由某个 block 振幅构造的 AO pair density，例如 $D^{CO}$、$D^{CV}$、$D^{OO}$、$D^{OV}$。
- $\mathcal{K}_{0}^{\mathrm{Ref}}[D]$ 表示普通 XC kernel action，对应代码 `vref0 = nr_rks_fxc(..., fxc_ref)`。
- $\mathcal{K}_{1}^{\mathrm{Ref}}[D]$ 表示 index-crossed XC kernel action，对应代码 `vref1`；LDA 下可由同一个 `nr_rks_fxc` 计算，GGA 下需要 `nr_rks_fxc1_gga`。
- $\mathcal{J}[D]$ 和 $\mathcal{K}^{\mathrm{HF}}[D]$ 分别表示 Coulomb 和 HF exchange action。
- $a_{\mathrm{x}}$ 表示 hybrid exact-exchange 系数，range-separated 部分按同样结构加上 $(\alpha-a_{\mathrm{x}})$ 的长程 action。

**代码变量到数学符号的对应：**
- `fxc_ref` = $f_{\mathrm{xc}}^{\mathrm{Ref}}$。
- `vref0_co`, `vref0_cv`, `vref0_oo`, `vref0_ov` = $\mathcal{R}_{0}^{CO}$、$\mathcal{R}_{0}^{CV}$、$\mathcal{R}_{0}^{OO}$、$\mathcal{R}_{0}^{OV}$。
- `vref1_co`, `vref1_ov` = $\mathcal{R}_{1}^{CO}$、$\mathcal{R}_{1}^{OV}$。
- `fockz` = $F^{z}$，代码中返回 `0.5 * delta`。
- `fock0` = $F^{0}=F^{\alpha}-F^{z}$。

### Step 1: XC reference kernel 的自旋组合

`satda.py` 对 `deltaS=-1` 使用的 XC kernel 不是普通 spin-flip 的单一横向 kernel，而是

$$
f_{\mathrm{xc}}^{\mathrm{Ref}}
= \frac{1}{2}
\left(
f_{\alpha\alpha}
- f_{\alpha\beta}
- f_{\beta\alpha}
+ f_{\beta\beta}
\right).
$$

→ **解释：** 这正是 `satda.py:305-307` 中 `fxc_ref` 的定义。它是 spin-difference channel 的 kernel，来自 PDF 中将一般自旋矩阵元统一写成 $K^{\mathrm{Ref}}$ 的假设，而不是普通 SF 梯度里直接使用的 transverse noncollinear kernel。

对任意 AO density $D$，LDA 下普通 action 可写成

$$
\left[\mathcal{K}_{0}^{\mathrm{Ref}}[D]\right]_{\mu\nu}
=
\int
\chi_{\mu}(\mathbf{r})
\chi_{\nu}(\mathbf{r})
f_{\mathrm{xc}}^{\mathrm{Ref}}(\mathbf{r})
\rho_{D}(\mathbf{r})
d\mathbf{r},
$$

→ **解释：** 这里 $\rho_{D}(\mathbf{r})=\sum_{\lambda\sigma}D_{\lambda\sigma}\chi_{\lambda}(\mathbf{r})\chi_{\sigma}(\mathbf{r})$。这对应 `ni.nr_rks_fxc(..., fxc_ref)` 的 LDA 情形。

GGA 下 density feature 变为

$$
\mathbf{r}_{D}(\mathbf{r})
=
\left(
\rho_{D},
\partial_{x}\rho_{D},
\partial_{y}\rho_{D},
\partial_{z}\rho_{D}
\right),
$$

→ **解释：** GGA 的 kernel 是 $4\times4$ feature kernel，`satda.py:nr_rks_fxc1_gga` 里显式使用 `ao[0]` 和 `ao[1:4]` 形成这些 feature。

因此 GGA 的普通 action 是 feature contraction：

$$
\left[\mathcal{K}_{0}^{\mathrm{Ref}}[D]\right]_{\mu\nu}
=
\sum_{A,B=0}^{3}
\int
\Phi_{\mu\nu}^{A}(\mathbf{r})
f_{\mathrm{xc},AB}^{\mathrm{Ref}}(\mathbf{r})
r_{D}^{B}(\mathbf{r})
d\mathbf{r}.
$$

→ **解释：** $\Phi_{\mu\nu}^{A}$ 表示 AO pair 的第 $A$ 个 density feature。$A=0$ 是 $\chi_{\mu}\chi_{\nu}$，$A=1,2,3$ 是 AO pair gradient feature。实际 PySCF 程序中这由 `nr_rks_fxc` 或自定义 `nr_rks_fxc1_gga` 完成。

### Step 2: 相对于 HF，多出两类 reference kernel action

PDF 中 `Sf=Si-1` 表将所有非 HF 的二电子部分写成 $K^{\mathrm{Ref}}$。代码实现上它分成两种 action：

$$
\mathcal{R}_{0}^{X}
=
\mathcal{K}_{0}^{\mathrm{Ref}}[D^{X}]
- a_{\mathrm{x}}\mathcal{K}^{\mathrm{HF}}[D^{X}],
\quad
X\in\{CO,CV,OO,OV\},
$$

→ **解释：** 这对应 `vref0`。当 `xctype != 'HF'` 时第一项来自 `nr_rks_fxc(..., fxc_ref)`；当 hybrid 存在时，代码再减去 exact exchange `vk`。

第二类是 index-crossed action：

$$
\mathcal{R}_{1}^{X}
=
\mathcal{K}_{1}^{\mathrm{Ref}}[D^{X}]
- a_{\mathrm{x}}\mathcal{J}[D^{X}],
\quad
X\in\{CO,OV\}.
$$

→ **解释：** 这对应 `vref1`。HF 极限下它退化为 $-\mathcal{J}[D]$，而 DFT 情形还包含 $\mathcal{K}_{1}^{\mathrm{Ref}}[D]$。这就是 DFT 相比 HF 最容易漏掉的项之一。

在 LDA 中，$\mathcal{K}_{1}^{\mathrm{Ref}}$ 与 $\mathcal{K}_{0}^{\mathrm{Ref}}$ 可由同一个 scalar-density contraction 实现：

$$
\left[\mathcal{K}_{1}^{\mathrm{Ref}}[D]\right]_{\mu\nu}
=
\int
\chi_{\mu}(\mathbf{r})
\chi_{\nu}(\mathbf{r})
f_{\mathrm{xc}}^{\mathrm{Ref}}(\mathbf{r})
\rho_{D}(\mathbf{r})
d\mathbf{r}.
$$

→ **解释：** 这解释了 `satda.py` 中 LDA 的 `vref1 = ni.nr_rks_fxc(...)`。

但在 GGA 中，index-crossed action 需要区分 left/right AO derivative：

$$
\left[\mathcal{K}_{1}^{\mathrm{Ref}}[D]\right]_{\mu\nu}
=
\sum_{A,B=0}^{3}
\int
\Psi_{\mu\nu}^{A}(\mathbf{r})
f_{\mathrm{xc},AB}^{\mathrm{Ref}}(\mathbf{r})
\widetilde{r}_{D}^{B}(\mathbf{r})
d\mathbf{r}.
$$

→ **解释：** $\Psi$ 和 $\widetilde{r}$ 表示 index-crossed 后的 AO derivative 排布；它不再等价于普通 `nr_rks_fxc` 的矩阵构造。因此 `satda.py` 对 GGA 使用 `nr_rks_fxc1_gga`，这也是不能直接复用普通 SF `_contract_xc_kernel` 的核心原因。

### Step 3: `deltaS=-1` 响应方程中的 XC block 组合

定义

$$
c_{1}=\sqrt{\frac{2S+1}{2S}},
\quad
c_{2}=\sqrt{\frac{2S}{2S-1}},
\quad
c_{3}=\sqrt{\frac{2S+1}{2S-1}},
\quad
c_{4}=\frac{2S}{2S-1},
\quad
c_{5}=\frac{1}{2S-1}.
$$

→ **解释：** 这些系数来自 PDF 的 `Sf=Si-1` block 表，也逐项出现在 `satda.py:348-371`。

`satda.py` 中 XC/HF kernel action 对四个 block 的 AO response 是

$$
V^{CO}
=
\mathcal{R}_{0}^{CO}
 + c_{5}\mathcal{R}_{1}^{CO}
 + c_{1}\mathcal{R}_{0}^{CV}
 + c_{2}\mathcal{R}_{0}^{OO}
 + c_{4}\mathcal{R}_{0}^{OV}
-c_{5}\mathcal{R}_{1}^{OV},
$$

→ **解释：** 这是代码 `v1ao_co += ...` 的数学翻译。

$$
V^{CV}
=
c_{1}\mathcal{R}_{0}^{CO}
 + \mathcal{R}_{0}^{CV}
 + c_{3}\mathcal{R}_{0}^{OO}
 + c_{1}\mathcal{R}_{0}^{OV},
$$

→ **解释：** 这是代码 `v1ao_cv += ...` 的数学翻译。

$$
V^{OO}
=
c_{2}\mathcal{R}_{0}^{CO}
 + c_{3}\mathcal{R}_{0}^{CV}
 + \mathcal{R}_{0}^{OO}
 + c_{2}\mathcal{R}_{0}^{OV},
$$

→ **解释：** 这是代码 `v1ao_oo += ...` 的数学翻译。

$$
V^{OV}
=
c_{4}\mathcal{R}_{0}^{CO}
-c_{5}\mathcal{R}_{1}^{CO}
 + c_{1}\mathcal{R}_{0}^{CV}
 + c_{2}\mathcal{R}_{0}^{OO}
 + \mathcal{R}_{0}^{OV}
 + c_{5}\mathcal{R}_{1}^{OV}.
$$

→ **解释：** 这是代码 `v1ao_ov += ...` 的数学翻译。注意 $\mathcal{R}_{1}$ 只进入 CO 和 OV 输出，而不是所有 block。

### Step 4: DFT 还改变 Fock-like 一电子块

代码返回

$$
F^{z}
=
\frac{1}{2}
\left(
\mathcal{K}_{0}^{\mathrm{Ref}}[D^{OO}]
-a_{\mathrm{x}}\mathcal{K}^{\mathrm{HF}}[D^{OO}]
\right),
$$

→ **解释：** 这对应 `delta = nr_rks_fxc(..., dmoo, fxc_ref)`、`delta -= get_k(..., dmoo) * hyb`、`return ..., 0.5 * delta`。

并定义

$$
F^{0}=F^{\alpha}-F^{z}.
$$

→ **解释：** 这对应 `gen_vind_sf` 中的 `fock0 = focka - fockz`。因此 DFT 不只是给二电子 kernel 多加一个 $f_{\mathrm{xc}}$；它还把 PDF 表中的 $f^{0}$、$f^{z}$ 分离出来，改变所有 Fock-like block 的一电子部分。

例如程序中的几个 Fock block 是

$$
F_{COCO}^{(0)} = O^{\dagger}(F^{0}-F^{z})O,
\quad
F_{COCO}^{(1)} = C^{\dagger}(F^{0}+F^{z})C,
\quad
F_{COCO}^{(2)} = C^{\dagger}F^{z}C.
$$

→ **解释：** 这对应 `fock_coco0`、`fock_coco1`、`fock_coco2`。HF-only 实现中这些组合被 exact-exchange ERI 表达式吸收；DFT 下必须显式保留 $F^{z}$ 的 XC 部分。

### Step 5: 梯度层面相对于 HF 新增的项

当前 HF block 梯度每个 block 分为

$$
\Omega_{B}^{[x]}
=
\Omega_{B,\mathrm{direct}}^{[x]}
+
\mathrm{Tr}\left[
M_{B} \kappa^{[x]}
\right].
$$

→ **解释：** 这就是现在 `_delta_grad.py` 的 block direct skeleton 加 orbital response 结构。

加入 XC 后，同一结构仍成立，但每个 block 的 $M_{B}$ 和 direct skeleton 都多出 XC 项：

$$
M_{B}
=
M_{B}^{\mathrm{HF}}
+
M_{B}^{\mathrm{XC},0}
+
M_{B}^{\mathrm{XC},1}
+
M_{B}^{F^{z}}.
$$

→ **解释：** $M_{B}^{\mathrm{XC},0}$ 来自 $\mathcal{K}_{0}^{\mathrm{Ref}}$，$M_{B}^{\mathrm{XC},1}$ 来自 index-crossed $\mathcal{K}_{1}^{\mathrm{Ref}}$，$M_{B}^{F^{z}}$ 来自 $F^{z}$ 对 open-shell density 的依赖。

direct skeleton 也对应分解为

$$
\Omega_{B,\mathrm{direct}}^{[x]}
=
\Omega_{B,\mathrm{direct}}^{\mathrm{HF},[x]}
+
\Omega_{B,\mathrm{grid/AO}}^{\mathrm{XC},[x]}
+
\Omega_{B,k_{\mathrm{xc}}}^{\mathrm{XC},[x]}.
$$

→ **解释：** 第一项是已完成 HF direct skeleton。第二项来自 AO basis/grid weight/feature 的显式核导数。第三项来自 $f_{\mathrm{xc}}$ 随参考态 density 改变而产生的 $k_{\mathrm{xc}}$ 项；GGA 还包含 density gradient feature 的导数。

### 最终结果

相对于 HF，`deltaS=-1` 的 SASF DFT 泛函贡献至少新增三类对象：

$$
\boxed{
f_{\mathrm{xc}}^{\mathrm{Ref}}
=
\frac{1}{2}
\left(
f_{\alpha\alpha}
-f_{\alpha\beta}
-f_{\beta\alpha}
+f_{\beta\beta}
\right)
}
$$

→ **解释：** 这是 SATDA/SASF 使用的 reference spin kernel，不是普通 SF-TDDFT 可直接复用的 kernel。

$$
\boxed{
\mathcal{K}_{0}^{\mathrm{Ref}}[D]
\quad\text{and}\quad
\mathcal{K}_{1}^{\mathrm{Ref}}[D]
}
$$

→ **解释：** $\mathcal{K}_{0}$ 是普通 kernel action，$\mathcal{K}_{1}$ 是 index-crossed action；GGA 下 $\mathcal{K}_{1}$ 需要 `nr_rks_fxc1_gga` 型实现。

$$
\boxed{
F^{z}
=
\frac{1}{2}
\left(
\mathcal{K}_{0}^{\mathrm{Ref}}[D^{OO}]
-a_{\mathrm{x}}\mathcal{K}^{\mathrm{HF}}[D^{OO}]
\right),
\quad
F^{0}=F^{\alpha}-F^{z}
}
$$

→ **解释：** DFT 不仅增加二电子 kernel，还修改 Fock-like block 的一电子组合。实现时必须让每个 block 的 energy、M-matrix、direct skeleton 和 Z-vector Hessian 都使用同一套 $F^{0}$、$F^{z}$、$\mathcal{K}_{0}$、$\mathcal{K}_{1}$ 分账。

**代码对应：**
- `pyscf-forge/pyscf/sftda/satda.py:297-382`：`gen_rohf_response_sf`，`deltaS=-1` 的 DFT/HF response kernel。
- `pyscf-forge/pyscf/sftda/satda.py:33-86`：`nr_rks_fxc1_gga`，GGA index-crossed kernel action。
- `pyscf-forge/pyscf/sftda/SASFTDA_Rev(2).pdf`：第 3 节 `Sf = Si - 1` 表，以及 $K^{\mathrm{Ref}}$ 的定义。

### 对后续实现的直接约束

- 第一版 DFT 不能从 `_contract_xc_kernel` 直接复制，因为 `_contract_xc_kernel` 没有 SASF block-dependent 的 $\mathcal{K}_{0}/\mathcal{K}_{1}$ 组合。
- LDA 最小实现应先写公共 `xc_ref_kernel_action_lda(tdobj, dms, crossed=False)`，其中 `crossed=False` 对应 $\mathcal{K}_{0}$，`crossed=True` 对应 $\mathcal{K}_{1}$。LDA 两者可共用底层 `nr_rks_fxc`。
- GGA 需要单独实现 crossed action，即 `satda.py:nr_rks_fxc1_gga` 对应的梯度版本。
- Z-vector Hessian 后续也必须改成 DFT ROKS Hessian：HF 的 dense Hessian action 只能覆盖 exact exchange，DFT 时要把上述 $V^{CO},V^{CV},V^{OO},V^{OV}$ 的 XC action 加进去。

## 补充：SATDA XC 梯度更接近 collinear TDDFT gradient，而不是 SF noncollinear kernel — 2026-05-31

**目标：** 对比 `pyscf-forge/pyscf/grad/tdsatda.py`、`pyscf/pyscf/grad/tdrks.py`、`pyscf/pyscf/grad/tduks.py` 与 `pyscf-forge-sasf/pyscf/grad/tduks_sf.py` 的 XC 梯度结构，明确后续 SASF DFT 实现应该参考哪一类代码。

**假设：**
- 仍然只讨论 `deltaS=-1` 的 SATDA/SASF collinear block 组合。
- `tdsatda.py` 中已有 LDA/GGA 实验实现可作为结构参考，但具体系数和分账仍需由 block-level FD/解析测试锁定。

### Step 1: `tduks_sf` 与 SATDA 的物理 kernel 不同

普通 SF-TDDFT 的 noncollinear 梯度把 spin-flip 激发解释为横向自旋密度扰动，核心对象类似

$$
\delta m_{x},\quad \delta m_{y},
\quad
f_{xx}+f_{yy}.
$$

→ **解释：** `tduks_sf._contract_xc_kernel` 中的注释和 contraction 体现了 transverse noncollinear kernel 结构，例如对 spin-flip transition density 使用横向自旋通道的组合。

SATDA/SASF 的 `deltaS=-1` 工作方程不是这样组织的。它先在 collinear ROKS/UKS 参考上构造四类 block density：

$$
D^{CO},\quad D^{CV},\quad D^{OO},\quad D^{OV},
$$

→ **解释：** 这些就是 `tdsatda.py:_satda_sf_transition_blocks` 返回的四个 AO pair density。

然后用 collinear spin-resolved functional derivatives 组合出 reference channel：

$$
f_{\mathrm{xc}}^{\mathrm{Ref}}
=
\frac{1}{2}
\left(
f_{\alpha\alpha}
-f_{\alpha\beta}
-f_{\beta\alpha}
+f_{\beta\beta}
\right).
$$

→ **解释：** 这与前一节的 `satda.py:gen_rohf_response_sf` 一致，也与 `tdsatda.py:_satda_sf_lda_fxc_ref` 一致。它是 collinear spin-kernel 的线性组合，不是 noncollinear transverse kernel 的直接调用。

### Step 2: 更合适的程序参考是 `tdrks/tduks._contract_xc_kernel`

`tdrks.py` 和 `tduks.py` 的 `_contract_xc_kernel` 在梯度中完成四类任务：

$$
\begin{aligned}
F^{VO}_{\mathrm{xc}} &\leftarrow f_{\mathrm{xc}}[\rho^{VO}],\\
F^{OO}_{\mathrm{xc}} &\leftarrow f_{\mathrm{xc}}[\rho^{OO}],\\
V^{[x]}_{\mathrm{xc}} &\leftarrow v_{\mathrm{xc}}^{[x]},\\
K^{[x]}_{\mathrm{xc}} &\leftarrow k_{\mathrm{xc}}[\rho^{VO},\rho^{VO}].
\end{aligned}
$$

→ **解释：** 这对应 PySCF 原生 TDDFT 梯度中的 `f1vo`、`f1oo`、`vxc1`、`k1ao`。这些对象正好是核导数、Z-vector、im0/metric 分账需要的 XC building blocks。

`tdsatda.py` 采用的结构与它们同类，而不是与 `tduks_sf` 同类：

$$
\begin{aligned}
\texttt{\_satda\_sf\_lda/gga\_xc\_q}
&\leftrightarrow M_{\mathrm{xc}},\\
\texttt{\_satda\_sf\_lda/gga\_transition\_deriv\_mats}
&\leftrightarrow f_{\mathrm{xc}}^{[x]}[\text{transition blocks}],\\
\texttt{\_satda\_sf\_lda/gga\_ref\_density\_mats}
&\leftrightarrow k_{\mathrm{xc}}[\text{block densities},\text{block densities}],\\
\texttt{\_contract\_uks\_lda\_vxc\_deriv}
&\leftrightarrow v_{\mathrm{xc}}^{[x]}+f_{\mathrm{xc}}[\rho^{OO}/\rho^{Z}].
\end{aligned}
$$

→ **解释：** 这些函数都是 collinear AO/grid derivative contraction。它们只是把普通 TDDFT 的单个 transition density 换成 SATDA 的四个 block densities，再用 spin-adaptation coefficient matrix 组合。

### Step 3: SATDA 的 block coefficient matrix 是核心差异

`tdsatda.py` 对 LDA 使用一个 block matrix：

$$
\mathbf{M}^{\mathrm{LDA}}(S)
=
\begin{pmatrix}
1 & c_{1} & c_{2} & c_{4}\\
c_{1} & 1 & c_{3} & c_{1}\\
c_{2} & c_{3} & 1 & c_{2}\\
c_{4} & c_{1} & c_{2} & 1
\end{pmatrix}
+
\begin{pmatrix}
c_{5} & 0 & 0 & -c_{5}\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
-c_{5} & 0 & 0 & c_{5}
\end{pmatrix}.
$$

→ **解释：** 这对应 `tdsatda.py:_satda_sf_lda_block_matrix` 和 `_satda_sf_gga_block_matrices` 中的 `m0/m1` 结构。第一矩阵来自普通 $\mathcal{K}_{0}^{\mathrm{Ref}}$ block coupling，第二矩阵来自 crossed $\mathcal{K}_{1}^{\mathrm{Ref}}$，只耦合 CO/OV。

因此 XC energy/direct/M 的基本型不是

$$
\rho_{\mathrm{SF}} f_{\mathrm{xc}}^{\perp} \rho_{\mathrm{SF}},
$$

→ **解释：** 这是 noncollinear SF 的思路，不适合 SATDA/SASF 的 collinear block 组合。

而是

$$
\Omega_{\mathrm{xc}}
=
\frac{1}{4}
\sum_{B,L\in\{CO,CV,OO,OV\}}
M_{BL}(S)
\int
\mathbf{r}_{B}^{T}(\mathbf{r})
f_{\mathrm{xc}}^{\mathrm{Ref}}(\mathbf{r})
\mathbf{r}_{L}(\mathbf{r})
d\mathbf{r}
$$

→ **解释：** $\mathbf{r}_{B}$ 是 block density feature。前面的 $\frac{1}{4}$ 对应 `tdsatda.py` 的 `SATDA_SF_LDA_XC_GRAD_SCALE = 0.25`。这个系数来自 ROKS half-density convention 和梯度中二次 block contraction 的规范化；实现时必须通过 block FD 锁定，不能经验调整。

### Step 4: 对 SASF DFT 实现路线的修正

第一版实现应优先抽象以下 collinear building blocks：

$$
\begin{aligned}
\mathrm{apply\_fxc\_ref}(D_{B}) &\rightarrow \mathcal{K}_{0}^{\mathrm{Ref}}[D_{B}],\\
\mathrm{apply\_fxc1\_ref}(D_{B}) &\rightarrow \mathcal{K}_{1}^{\mathrm{Ref}}[D_{B}],\\
\mathrm{transition\_deriv\_mats}(D_{B}) &\rightarrow \left(\mathcal{K}_{0/1}^{\mathrm{Ref}}[D_{B}]\right)^{[x]}_{\mathrm{AO/grid}},\\
\mathrm{ref\_density\_mats}(D_{B},D_{L}) &\rightarrow k_{\mathrm{xc}}^{\mathrm{Ref}}[D_{B},D_{L}].
\end{aligned}
$$

→ **解释：** 这正是 `tdsatda.py` 当前 LDA/GGA 实验实现的层次，也正好对应 `tdrks/tduks._contract_xc_kernel` 的梯度分账。

### 最终结果

后续 SASF DFT 梯度不应以 `tduks_sf._contract_xc_kernel` 为主参考，而应以如下关系为主：

$$
\boxed{
\text{SASF/SATDA XC gradient}
\approx
\text{collinear } \texttt{tdrks/tduks.\_contract\_xc\_kernel}
+
\text{SASF block coefficient matrices}
}
$$

→ **解释：** `tduks_sf` 体现的是 noncollinear spin-flip 横向核；SATDA/SASF 的新工作方程体现的是 collinear spin-resolved kernel 的 block 组合。正确实现应复用 `tdrks/tduks` 的 AO/grid derivative 思路，再用 `satda.py/tdsatda.py` 的 block matrix 组合替换普通 TDDFT 的单 transition-density contraction。

**代码对应：**
- `pyscf-forge/pyscf/grad/tdsatda.py:_satda_sf_lda_fxc_ref`
- `pyscf-forge/pyscf/grad/tdsatda.py:_satda_sf_lda_apply_fxc_ref`
- `pyscf-forge/pyscf/grad/tdsatda.py:_satda_sf_gga_apply_fxc1_ref`
- `pyscf-forge/pyscf/grad/tdsatda.py:_satda_sf_lda_xc_q`
- `pyscf-forge/pyscf/grad/tdsatda.py:_satda_sf_gga_xc_q`
- `pyscf/pyscf/grad/tdrks.py:_contract_xc_kernel`
- `pyscf/pyscf/grad/tduks.py:_contract_xc_kernel`

## 补充：HF 极限下 `sasf.py` 与 `satda.py` 的 Fock 变量基变换 — 2026-05-31

**目标：** 说明 `Zhao 和 Li 2025`/`pyscf-forge-sasf/pyscf/sftda/sasf.py` 与 `SASFTDA_Rev(2).pdf`/`pyscf-forge/pyscf/sftda/satda.py` 在 HF 极限下不是两套不同矩阵元，而是同一公式在不同 Fock 变量基下的表示。

**假设：**
- 只讨论 HF exact-exchange 极限，因此 $F^{\alpha}$ 与 $F^{\beta}$ 是通常的 ROKS spin Fock 矩阵。
- `satda.py` 的 `deltaS=-1` 路径使用 `fock0` 与 `fockz`。
- `sasf.py` 使用 `focka`、`fockb` 和 `focks = 0.5 * (fockb - focka)`。

### Step 1: 两套变量的线性关系

定义

$$
F^{0}=\frac{1}{2}\left(F^{\alpha}+F^{\beta}\right),
\quad
F^{z}=\frac{1}{2}\left(F^{\alpha}-F^{\beta}\right).
$$

→ **解释：** 这是 SATDA PDF 中把矩阵元写成 $f^{0}$ 与 $f^{z}$ 的基。它只是对 $\alpha/\beta$ Fock 的线性变换。

因此反变换是

$$
F^{\alpha}=F^{0}+F^{z},
\quad
F^{\beta}=F^{0}-F^{z}.
$$

→ **解释：** `satda.py` 中 `fock0 = focka - fockz`。在 HF 极限若 `fockz = 0.5 * (focka - fockb)`，则 `fock0 = 0.5 * (focka + fockb)`。

`sasf.py` 中的 spin Fock 是

$$
F^{S}_{\mathrm{sasf}}
=\frac{1}{2}\left(F^{\beta}-F^{\alpha}\right)
=-F^{z}.
$$

→ **解释：** 这对应 `sasf.py:get_a_sasf` 中 `focks = 0.5 * (fock.fockb - fock.focka)`。因此凡是 `sasf.py` 中出现 `focks` 的地方，翻译到 SATDA PDF 的 $f^{z}$ 记号时要带负号。

### Step 2: `satda.py` Fock block 还原为 `sasf.py` 的 alpha/beta 形式

`satda.py` 的几个基本组合满足

$$
F^{0}-F^{z}=F^{\beta},
\quad
F^{0}+F^{z}=F^{\alpha}.
$$

→ **解释：** 这立即解释 `satda.py:gen_vind_sf` 中 `fock0 - fockz` 与 `fock0 + fockz` 的物理含义。

例如

$$
O^{\dagger}(F^{0}-F^{z})O
=
O^{\dagger}F^{\beta}O,
$$

→ **解释：** 这对应 SATDA 的 `fock_coco0`，在 `sasf.py` 的基下就是 beta Fock 的 open-open block。

又如

$$
C^{\dagger}(F^{0}+F^{z})C
=
C^{\dagger}F^{\alpha}C,
$$

→ **解释：** 这对应 SATDA 的 `fock_coco1`，在 `sasf.py` 的基下就是 alpha Fock 的 core-core block。

而

$$
C^{\dagger}F^{z}C
=
-C^{\dagger}F^{S}_{\mathrm{sasf}}C.
$$

→ **解释：** 这对应 SATDA 的 `fock_coco2`，也是比较 CO-CO/CV-CV/OV-OV 等块时最容易出现符号错觉的地方。

### Step 3: 为什么 HF 极限一致

HF 极限下，SATDA 的 Fock-like 项用 $F^{0}$ 与 $F^{z}$ 表示；SASF 老实现用 $F^{\alpha}$、$F^{\beta}$ 与 $F^{S}_{\mathrm{sasf}}$ 表示。由于两组变量之间是可逆线性变换：

$$
\left(F^{\alpha},F^{\beta}\right)
\longleftrightarrow
\left(F^{0},F^{z}\right),
$$

→ **解释：** 只要所有 block 的系数、转置和 full off-diagonal pair factor 一致，两套写法必然给出同一个 HF 矩阵。

换句话说，

$$
A_{\mathrm{SASF}}^{\mathrm{HF}}
\left[F^{\alpha},F^{\beta},F^{S}_{\mathrm{sasf}}\right]
=
A_{\mathrm{SATDA}}^{\mathrm{HF}}
\left[F^{0},F^{z}\right],
\quad
F^{S}_{\mathrm{sasf}}=-F^{z}.
$$

→ **解释：** 这就是 `pyscf-forge-sasf/pyscf/sftda/sasf.py` 与 `pyscf-forge/pyscf/sftda/satda.py` 在 HF 极限下一致的根本原因。

### 最终结果

后续实现 DFT 时，最安全的理解方式是：

$$
\boxed{
\text{HF 部分可以在 }(F^{\alpha},F^{\beta})\text{ 基或 }(F^{0},F^{z})\text{ 基中等价表示。}
}
$$

→ **解释：** 这说明老 SASF HF 解析梯度和新 SATDA HF 工作方程之间没有理论冲突。

但 DFT 下更自然的变量是

$$
\boxed{
F^{0}=F^{\alpha}-F^{z},
\quad
F^{z}=
\frac{1}{2}
\left(
\mathcal{K}_{0}^{\mathrm{Ref}}[D^{OO}]
-a_{\mathrm{x}}\mathcal{K}^{\mathrm{HF}}[D^{OO}]
\right)
}
$$

→ **解释：** 这是 `satda.py` 的实现变量。由于 $F^{z}$ 包含 XC reference kernel action，DFT 实现应优先沿用 SATDA 的 $F^{0}/F^{z}$ 分账，再在 HF 极限检查它是否退化到 `sasf.py` 的 $F^{\alpha}/F^{\beta}/F^{S}_{\mathrm{sasf}}$ 表示。

**代码对应：**
- `pyscf-forge-sasf/pyscf/sftda/sasf.py:get_a_sasf` 中 `focks = 0.5 * (fock.fockb - fock.focka)`。
- `pyscf-forge/pyscf/sftda/satda.py:gen_vind_sf` 中 `fockz`、`fock0 = focka - fockz` 以及 `fock0 ± fockz` 的 block 组合。

---

## 推导：复用 `_contract_xc_kernel` 实现 SATDA XC 梯度 — 2026-05-31

**目标：** 分析 `pyscf/pyscf/grad/tdrks.py:_contract_xc_kernel`（RKS 版）和 `pyscf/pyscf/grad/tduks.py:_contract_xc_kernel`（UKS 版）的网格循环结构，明确哪些部分可被 SATDA XC 梯度直接复用、哪些必须替换，并给出修改后的完整 grid loop 骨架。

**假设：**
- 仅考虑 `deltaS=-1`、ROKS/ROHF 参考态、TDA。
- 先讨论 LDA/GGA 的二阶响应；MGGA 与 NLC 不作为当前目标。
- 记 $S$ 为初态自旋量子数，$S \ge 1$。

**符号声明：**
- $B, L \in \{\text{CO}, \text{CV}, \text{OO}, \text{OV}\}$ 遍历四个激发块。
- $D^{B}$ 为 AO 基 block pair density，例如 $D^{\text{CO}}$、$D^{\text{CV}}$ 等。
- $f_{\mathrm{xc}}^{\mathrm{Ref}}$ 为 SATDA reference spin kernel（见 §Step 1 定义）。
- $\mathcal{F}[\mathbf{wv}, \text{ao}]$ 表示 `_lda_eval_mat_` / `_gga_eval_mat_` 将网格上的 weighted kernel action 组装到 AO 基梯度矩阵的操作。
- $\tilde{D}^{B}$ 表示对 block 密度做 0.5 倍对称化: $\tilde{D}^{B} = \frac{1}{2}(D^{B} + D^{B,\dagger})$ (hermi=1 处理)。

### Step 1: 可复用的工具函数

`tdrks.py:356-379` 定义了三个工具函数，它们是纯矩阵组装操作，不包含任何核选择的逻辑：

$$
\mathcal{F}_{\mathrm{LDA}}[\mathbf{w}, \text{ao}]
= \sum_{k=0}^{3} \int \text{ao}[k]^{\mu} \,
\bigl(\text{ao}[0]^{\nu} \cdot w^{k}\bigr) \,
d\mathbf{r}
$$

→ **解释：** 这是 `_lda_eval_mat_`。输入是 (mol, vmat, ao, wv, mask, ...)，将每个网格点上的 `wv[k]` (=feature 0 到 3) 与 AO 函数值和导数相乘，累加到 vmat 中。LDA 只有 `ao[0]`（AO 值），GGA 增加 `ao[1:4]`（梯度）。

这三个函数——`_lda_eval_mat_`、`_gga_eval_mat_`、`_mgga_eval_mat_`——直接 import 即可，无需修改。SATDA 的 grid loop 输出完全兼容这些组装器的输入签名。

### Step 2: 可复用的网格循环骨架

`tdrks._contract_xc_kernel` (line 279-306, singlet 分支) 和 `tduks._contract_xc_kernel` (line 339-374) 的核心循环可抽象为：

```
for ao, mask, weight, coords in block_loop(mol, grids, nao, ao_deriv):
    rho = eval_rho2(mol, ao0, mo_coeff, mo_occ, ...)      # ① 参考态密度
    vxc, fxc, kxc = eval_xc_eff(xc_code, rho, deriv)       # ② XC 势/核/超核
    rho1 = eval_rho(mol, ao0, dmvo, ...)                   # ③ 扰动密度
    wv = einsum('yg,xyg,g->xg', rho1, fxc, weight)         # ④ fxc 作用
    fmat_(mol, f1vo, ao, wv, ...)                          # ⑤ 组装到 AO
```

→ **解释：** 这五步中，① 和 ⑤ 完全可复用：`eval_rho2` 对 ROKS 参考态同样有效，`_lda_eval_mat_` 签名不变。② 的参数变但调用方式不变（`eval_xc_eff` 总需完整 fxc 矩阵来构造 `fxc_ref`）。③ 和 ④ 需要修改以适应多块密度和 `fxc_ref` 核。

### Step 3: 必须修改的三处差异

#### 差异 A: 核选择 — $f_{\mathrm{xc}}^{\mathrm{Ref}}$ vs $f_{\mathrm{xc}}^{s/t}$

`tdrks.py` 使用（line 286, 317-321）:
- Singlet: `fxc` (full, shape `(x,y,g)`) 直接使用
- Triplet: `fxc_t = fxc[:,:,0] - fxc[:,:,1]; fxc_t = fxc_t[0] - fxc_t[1]`

`tduks.py` 使用（line 347, 354）:
- 通用: `fxc` 全量 shape `(a,x,b,y,g)`，通过 `einsum('axg,axbyg,g->byg', rho1, fxc, weight)` 处理 spin 指标

SATDA 需要（`satda.py:305-307`）:

$$
f_{\mathrm{xc}}^{\mathrm{Ref}}
= \frac{1}{2} \bigl(
f_{\alpha\alpha} - f_{\alpha\beta} - f_{\beta\alpha} + f_{\beta\beta}
\bigr)
$$

→ **解释：** 这是 spin-difference channel，与 singlet (`f_αα + f_αβ`) 或 triplet (`f_αα - f_αβ`) 都不同。代码上等价于：

```python
fxc_ref = 0.5 * (fxc[0, :, 0] - fxc[0, :, 1] - fxc[1, :, 0] + fxc[1, :, 1])
```

因此必须用 ROKS 的完整 spin-resolved `fxc`（shape `(2, x, 2, y, g)`）来构造，之后 `einsum` 只用 `fxc_ref` 做收缩：

$$
\mathbf{wv}^{B} = \sum_{y} \rho_{y}^{B} \, (f_{\mathrm{xc}}^{\mathrm{Ref}})_{xy} \cdot \mathrm{weight}
$$

#### 差异 B: 多块密度替代单 dmvo

tdrks 输入一个 `dmvo`（shape `(nao,nao)`）。SATDA 输入四个 block densities:

$$
\{ D^{\text{CO}}, D^{\text{CV}}, D^{\text{OO}}, D^{\text{OV}} \}
$$

每个 $D^{B}$ 在网格上计算其 density feature $\mathbf{r}^{B}$：

$$
\rho^{B}_{0} = \sum_{\mu\nu} D_{\mu\nu}^{B} \, \chi_{\mu} \chi_{\nu},
\qquad
\rho^{B}_{x} = \nabla_{x} \rho^{B}_{0} \quad (\text{GGA})
$$

→ **解释：** 在 grid loop 内，不是只调用一次 `eval_rho`，而是对四个 block 各调用一次。LDA 下所有 block 的 `eval_rho` 都可用同一个 `fmat_` (`_lda_eval_mat_`)。GGA 下需要注意 `ao_deriv=2` 获取梯度。

#### 差异 C: 块系数矩阵 $M_{0}(S)$, $M_{1}(S)$ 组合

这是 SATDA 与普通 TDDFT 的核心差异。fxc 作用后得到四个 AO 基 proto-response $\mathbf{W}^{B}$（每个网格点上的 weighted fxc action），它们需要按块系数矩阵组合后才成为最终的梯度的 AO 基矩阵元。

定义系数（来自 `satda.py:361-368` 和 `derivations_xc_functional.md` §Step 3）:

$$
\begin{aligned}
c_{1} &= \sqrt{\frac{2S+1}{2S}}, &
c_{2} &= \sqrt{\frac{2S}{2S-1}}, \\
c_{3} &= \sqrt{\frac{2S+1}{2S-1}}, &
c_{4} &= \frac{2S}{2S-1}, \quad
c_{5} = \frac{1}{2S-1}.
\end{aligned}
$$

$K_{0}^{\mathrm{Ref}}$ 的块耦合矩阵（对称，所有块间互相耦合）:

$$
M_{0}(S) =
\begin{pmatrix}
1 & c_{1} & c_{2} & c_{4} \\
c_{1} & 1 & c_{3} & c_{1} \\
c_{2} & c_{3} & 1 & c_{2} \\
c_{4} & c_{1} & c_{2} & 1
\end{pmatrix}_{(\text{CO, CV, OO, OV}) \times (\text{CO, CV, OO, OV})}
$$

$K_{1}^{\mathrm{Ref}}$ 的块耦合矩阵（仅 CO 与 OV 之间的 crossed coupling）:

$$
M_{1}(S) =
\begin{pmatrix}
c_{5} & 0 & 0 & -c_{5} \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
-c_{5} & 0 & 0 & c_{5}
\end{pmatrix}
$$

→ **解释：** $M_{1}$ 非零元仅出现在 CO↔CO、CO↔OV、OV↔CO、OV↔OV 位置，这是因为 `satda.py:361-368` 中 crossed action (`vref1`) 只进入 CO 和 OV 的组装。

在每个网格点上，四个 block 的输出由 $M_{0}$ 和 $M_{1}$ 线性组合:

$$
\mathbf{W}_{\text{final}}^{i}
= \sum_{j} \bigl( M_{0}^{ij} \, \mathbf{W}_{0}^{j} + M_{1}^{ij} \, \mathbf{W}_{1}^{j} \bigr)
$$

其中 $\mathbf{W}_{0}^{j}$ 是 $D^{j}$ 经 `fxc_ref` 的普通 kernel action，$\mathbf{W}_{1}^{j}$ 是 crossed kernel action（LDA 下与 $\mathbf{W}_{0}^{j}$ 在 LDA 层面相同但组装方式不同，GGA 下需要 `nr_rks_fxc1_gga`）。

### Step 4: $K_{0}^{\mathrm{Ref}}$ 与 $K_{1}^{\mathrm{Ref}}$ 在 LDA 和 GGA 下的差异化处理

在进入完整 grid loop 之前，必须先区分两种 kernel action 在 LDA 和 GGA 下的不同行为：

#### LDA: $K_{0}^{\mathrm{Ref}}$ 与 $K_{1}^{\mathrm{Ref}}$ 的 grid-weighted 向量相同

LDA 下两者对同一个 block 密度 $D^{B}$ 的网格输出相同：

$$
\left[\mathcal{K}_{0}^{\mathrm{Ref}}[D^{B}]\right](\mathbf{r})
= f_{\mathrm{xc}}^{\mathrm{Ref}}(\mathbf{r}) \cdot \rho_{D^{B}}(\mathbf{r})
= \left[\mathcal{K}_{1}^{\mathrm{Ref}}[D^{B}]\right](\mathbf{r})
$$

→ **解释：** `satda.py:334-336` 中 LDA 的 `vref0` 和 `vref1` 都调用 `ni.nr_rks_fxc(..., fxc_ref)`。区别仅在于：
- `K_{0}^{\mathrm{Ref}}` 作用于 4 个 block (CO, CV, OO, OV)，通过 `dms0` 传入
- `K_{1}^{\mathrm{Ref}}` 作用于 2 个 block (CO, OV)，通过 `dms1` 传入

因此 LDA grid loop 中只需计算一份 `wv[kj]`，同时用于 $M_{0}$ 和 $M_{1}$ 的组合。两者的差异体现在：$M_{0}$ 耦合全部 4 个 block，$M_{1}$ 仅耦合 CO↔OV。

#### GGA: $K_{0}^{\mathrm{Ref}}$ 与 $K_{1}^{\mathrm{Ref}}$ 的 grid-weighted 向量不同

GGA 下对同一个 $D^{B}$，两个 action 产生不同的 $U$ 矩阵：

$$
U_{AB}^{(0)} = f_{\mathrm{xc},AB}^{\mathrm{Ref}} \cdot r_{B}
\quad\text{vs}\quad
U_{AB}^{(1)} = \sum_{C} f_{\mathrm{xc},AC}^{\mathrm{Ref}} \cdot \widetilde{r}_{C}
$$

以及不同的 AO 基组装：

$$
V_{\mu\nu}^{(0)} = \sum_{A} \Phi_{\mu\nu}^{A} \cdot U_{A*}
\quad\text{vs}\quad
V_{\mu\nu}^{(1)} = \sum_{A} \Psi_{\mu\nu}^{A} \cdot U_{A*}
$$

→ **解释：** `satda.py:241-243` 中 GGA 的 `vref0` 用 `ni.nr_rks_fxc`，`vref1` 用 `nr_rks_fxc1_gga`。两者的 `fxc_ref` 相同，但 density feature 的构造（$\Phi$ vs $\Psi$，$r$ vs $\widetilde{r}$）不同。梯度实现中必须分别计算 `wv_0[kj]` 和 `wv_1[kj]`。

#### 减法模式也不同：$K^{\mathrm{HF}}$ vs $\mathcal{J}$

两种 action 在 hybrid 泛函下的减法模式不同（来自 `satda.py:346-353`）：

$$
\mathcal{R}_{0}^{X} = \mathcal{K}_{0}^{\mathrm{Ref}}[D^{X}] - a_{\mathrm{x}} \mathcal{K}^{\mathrm{HF}}[D^{X}],
\qquad X\in\{\text{CO, CV, OO, OV}\}
$$

$$
\mathcal{R}_{1}^{X} = \mathcal{K}_{1}^{\mathrm{Ref}}[D^{X}] - a_{\mathrm{x}} \mathcal{J}[D^{X}],
\qquad X\in\{\text{CO, OV}\}
$$

→ **解释：** $\mathcal{R}_{0}$ 减去的是 exact exchange (K)，$\mathcal{R}_{1}$ 减去的是 Coulomb (J)。在梯度中，这意味着：
- $\mathcal{R}_{0}$ 的 direct gradient 需要额外加 `_add_k_bilinear_ip1`（交换导数）乘以 $a_{\mathrm{x}}$
- $\mathcal{R}_{1}$ 的 direct gradient 需要额外加 `_add_j_bilinear_ip1`（库仑导数）乘以 $a_{\mathrm{x}}$

### Step 5: 修改后的完整 grid loop 骨架（LDA 版本）

综合上面三处修改及 $K_{0}$/$K_{1}$ 的区分，SATDA 的 XC 梯度 grid loop 为:

```
keys = ['co','cv','oo','ov']
f1 = {k: zeros((4,nao,nao)) for k in keys}       # K_0^Ref 梯度输出
f1x = {k: zeros((4,nao,nao)) for k in keys}      # K_1^Ref 梯度输出(CO,OV only)
vxc1 = zeros((4,nao,nao))

for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv=1):
    ao0 = ao                                          # LDA

    # ① 参考态密度 → fxc 全量 (shape (2,x,2,y,g))
    rho_spin = eval_roks_spin_rho2(mol, ao0, mo_coeff, mo_occ, mask, 'LDA')
    _, fxc, kxc = ni.eval_xc_eff(xc_code, rho_spin, deriv=2+with_kxc)[1:]

    # ★ 差异 A: 构造 fxc_ref (spin-difference channel)
    fxc_ref = 0.5*(fxc[0,:,0] - fxc[0,:,1] - fxc[1,:,0] + fxc[1,:,1])
    # LDA: shape (1,1,g)

    # ★ 差异 B: 对四个 block 密度各做 eval_rho
    wv = {}   # K_0^Ref 的 weighted response（全 4 块）
    for k, dm_b in block_dms.items():
        rho_b = ni.eval_rho(mol, ao0, dm_b, mask, 'LDA', hermi=1)
        rho_b = rho_b[newaxis] * 2                     # LDA: (1,g)
        # LDA: K_0 与 K_1 的 wv 相同 → 计算一份即可
        wv[k] = einsum('yg,xyg,g->xg', rho_b, fxc_ref, weight)

    # ★ 差异 C: 分别用 M0(S) 和 M1(S) 组合
    # M0: 耦合全部 4 blocks, M1: 仅耦合 CO↔OV
    for i, ki in enumerate(keys):
        wv_i_0 = zeros_like(wv[ki])
        wv_i_1 = zeros_like(wv[ki])
        for j, kj in enumerate(keys):
            wv_i_0 += M0[i,j] * wv[kj]                  # K_0^Ref: all blocks
            if ki in ['co','ov']:
                wv_i_1 += M1[i,j] * wv[kj]              # K_1^Ref: CO,OV only
        fmat_(mol, f1[ki],  ao, wv_i_0, mask, ...)     # 复用 _lda_eval_mat_
        fmat_(mol, f1x[ki], ao, wv_i_1, mask, ...)     # 单独存 K_1 部分

    # vxc1: Fock-like 骨架
    vxc_ref = 0.5 * (vxc[0] - vxc[1])
    fmat_(mol, vxc1, ao, vxc_ref * weight, mask, ...)

for k in keys: f1[k][1:]*=-1; f1x[k][1:]*=-1
vxc1[1:] *= -1
```

→ **解释：** `eval_roks_spin_rho2` 表示沿用 `cache_xc_kernel(..., mo_coeff, mo_occ, spin=1)` 的 spin-resolved 参考密度约定，而不是把同一个标量密度复制到 alpha/beta 两个通道。LDA 下 `wv` 对于 $K_{0}$ 和 $K_{1}$ 数值相同，所以只计算一份。但输出分存为 `f1` 和 `f1x` 两套，因为后续 gradient assembly 中它们的减法模式不同：`f1[kj]` 用于组装 $\mathcal{R}_{0}$（需减 `_add_k_bilinear_ip1`），`f1x[kj]` 用于组装 $\mathcal{R}_{1}$（需减 `_add_j_bilinear_ip1`）。

### Step 6: GGA 下需要两套独立的 wv 向量

GGA 的 grid loop 必须分别计算 `wv_0` ($K_{0}^{\mathrm{Ref}}$) 和 `wv_1` ($K_{1}^{\mathrm{Ref}}$):

1. `ao_deriv=2`，`fmat_` 换为 `_gga_eval_mat_`
2. **$K_{0}^{\mathrm{Ref}}$ 的 `wv_0[kj]`**：按普通 `tdrks._contract_xc_kernel` 的方式计算，`fxc_ref` shape $(4,4,g)$，`rho_b` shape $(4,g)$，`einsum('yg,xyg,g->xg', rho_b, fxc_ref, weight)` → $(4,g)$
3. **$K_{1}^{\mathrm{Ref}}$ 的 `wv_1[kj]`**：需要内联实现 `nr_rks_fxc1_gga` 的 $U$ 矩阵构造（`satda.py:69-78`），并且在组装到 AO 基时，对于 $K_{1}$ 使用的是 crossed derivative convention（$\Psi_{\mu\nu}^{A}$ 而非 $\Phi_{\mu\nu}^{A}$）:

   ```
   # K_1^Ref crossed U matrix (per block density D^j, per grid chunk):
   rho0  = contract_rho(ao[0], c0)
   L_grad = contract_rho(ao[0], c_grad[i])   # left derivatives
   R_grad = contract_rho(ao[j], c0)          # right derivatives
   tau   = contract_rho(ao[j], c_grad[i])

   U = zeros((4,4,g))
   U[0,0] = fxc_ref[0,0]*rho0 + Σ fxc_ref[i,0]*L_i + Σ fxc_ref[0,j]*R_j + Σ fxc_ref[i,j]*tau_ij
   U[1:4,0] = fxc_ref[1:4,0]*rho0 + Σ fxc_ref[1:4,j]*R_j
   U[0,1:4] = fxc_ref[0,1:4]*rho0 + Σ fxc_ref[i,1:4]*L_i
   U[1:4,1:4] = fxc_ref[1:4,1:4]*rho0

   # 组装到 AO (crossed convention, 与 _gga_eval_mat_ 不同!)
   aow = scale_ao_sparse(ao, U[i,:,:])
   v_chunk = dot_ao_ao_sparse(ao[i], aow, ...)
   ```

→ **解释：** 这是 `satda.py:nr_rks_fxc1_gga` (line 33-85) 的逻辑直接植入 grid loop。与普通 `_gga_eval_mat_` 的区别在于 density feature 的构造方式和 AO 组装方式都不同，无法复用 `_gga_eval_mat_`。

### Step 7: `_contract_xc_kernel` 的 HF 交换部分不需要复用

`tdrks._contract_xc_kernel` 只处理纯 XC 的 grid 部分。HF 交换（`tdrks.py:85-103` 中的 `get_jk` 调用）在 SATDA 中已有独立的处理路径（`tdsatda_delta/_exchange.py` 的 9 项 ERI 和 `_add_eri_term_q`）。

当泛函是 hybrid 时，HF 交换乘以 `hyb` 系数的部分继续用现有 ERI 路径，XC kernel 路径中不重复处理。但 $\mathcal{R}_{0}$ 需要附加 `_add_k_bilinear_ip1`（交换导数），$\mathcal{R}_{1}$ 需要附加 `_add_j_bilinear_ip1`（库仑导数），均乘以 `hyb`。

### 复用总结

| 组件 | 来源 | 是否可直接复用 | 备注 |
|---|---|---|---|
| `block_loop` | `ni` (numint) | 是 | 签名不变 |
| `eval_rho2` | `ni` | 是 | ROKS 参考态兼容 |
| `eval_rho` | `ni` | 是 | 对每个 block 密度调用 |
| `eval_xc_eff` | `ni` | 是 | 需要完整 fxc/kxc 来构造 `fxc_ref` |
| `_lda_eval_mat_` | `tdrks` | 是 | 用于 $K_{0}^{\mathrm{Ref}}$ 的 AO 组装 |
| `_gga_eval_mat_` | `tdrks` | 是 | 用于 $K_{0}^{\mathrm{Ref}}$ 的 GGA AO 组装 |
| `_mgga_eval_mat_` | `tdrks` | 是 | MGGA 暂未实现 |
| **`fxc_s` / `fxc_t` 选择** | `tdrks:_contract_xc_kernel` | **否** | 替换为 `fxc_ref` = spin-difference |
| **单 dmvo** | `tdrks:_contract_xc_kernel` | **否** | 替换为 4 个 block densities |
| $K_{0}^{\mathrm{Ref}}$ 块组合 | 无 | **否** | 新增 $M_{0}(S)$ 矩阵 |
| $K_{1}^{\mathrm{Ref}}$ 块组合 | 无 | **否** | 新增 $M_{1}(S)$ 矩阵，仅耦合 CO↔OV |
| $\mathcal{R}_{0}$ hybrid 减法 | `tdsatda_delta/_direct.py` | 是 | `_add_k_bilinear_ip1` (交换导数) |
| $\mathcal{R}_{1}$ hybrid 减法 | `tdsatda_delta/_direct.py` | 是 | `_add_j_bilinear_ip1` (库仑导数) |
| **GGA $K_{1}^{\mathrm{Ref}}$ AO 组装** | `satda:nr_rks_fxc1_gga` | **需内联改写** | crossed derivative convention 不同，无法复用 `_gga_eval_mat_` |
| LDA $K_{1}^{\mathrm{Ref}}$ 组装 | `_lda_eval_mat_` (`tdrks`) | 是 | LDA 下 $K_{0}$ 与 $K_{1}$ 的 AO 组装相同 |

**代码对应：**
- `pyscf/pyscf/grad/tdrks.py:_contract_xc_kernel` — grid loop 模板（line 279-306 singlet, line 308-348 triplet）
- `pyscf/pyscf/grad/tdrks.py:_lda_eval_mat_` — LDA 组装器 (line 356-360)
- `pyscf/pyscf/grad/tdrks.py:_gga_eval_mat_` — GGA 组装器 (line 362-368)
- `pyscf/pyscf/grad/tduks.py:_contract_xc_kernel` — UKS 版 grid loop (line 339-374)
- `pyscf-forge/pyscf/sftda/satda.py:gen_rohf_response_sf` — `fxc_ref` 定义和 block 系数 (line 284-382)
- `pyscf-forge/pyscf/sftda/satda.py:nr_rks_fxc1_gga` — GGA crossed kernel action (line 33-85)
- `pyscf-forge/pyscf/grad/tdsatda_delta/_exchange.py` — HF 交换项（hybrid 时复用）
- `pyscf-forge/pyscf/grad/tdsatda_delta/_direct.py` — ERI 直接导数（hybrid 时复用）

## 推导：SATDA `deltaS=-1` LDA XC 分块梯度的第一阶段实现对象 — 2026-05-31

**目标：** 从 SATDA 的 LDA block-kernel 能量泛函推导当前 `tdsatda_delta/_xc_lda.py` 中实现并测试的两个局部解析对象：轨道旋转导数矩阵 $M_{\mathrm{xc}}$ 与 frozen-orbital direct skeleton 梯度。

**假设：**
- 只讨论 LDA，不包含 GGA feature-gradient 与 crossed GGA kernel。
- 当前对象是 `delta_A` 修正中的 SATDA block XC 部分，不是完整 TDDFT 总梯度入口。
- 分子轨道、振幅和 AO 基均取实数。
- direct skeleton 测试采用 frozen MO coefficient 与 frozen excitation amplitude，即只微分 AO/grid/reference-density 显式核。

**符号声明：**
- $B,L\in\{CO,CV,OO,OV\}$ 为四个 SATDA transition block。
- $D^{B}_{\mu\nu}$ 为 block $B$ 的 AO pair density。
- $\rho_{B}(\mathbf{r})=\sum_{\mu\nu}D^{B}_{\mu\nu}\chi_{\mu}(\mathbf{r})\chi_{\nu}(\mathbf{r})$。
- $f_{\mathrm{xc}}^{\mathrm{Ref}}$ 为 spin-difference reference kernel。
- $M_{BL}(S)$ 为 `deltaS=-1` 的 LDA block coefficient matrix。
- $s_{\mathrm{LDA}}=\frac{1}{4}$ 为当前 half-density convention 下的 overall factor。

**代码变量到数学符号的对应：**
- `_lda_block_matrix(si)` = $M_{BL}(S)$。
- `_transition_blocks(tdobj, xy)` = $\{D^{CO},D^{CV},D^{OO},D^{OV}\}$ 及其 MO target/source index。
- `lda_xc_energy()` = $\Omega_{\mathrm{xc}}^{\mathrm{LDA}}$。
- `lda_xc_q()` / `lda_xc_m_matrix()` = $Q_{\alpha},Q_{\beta}$ 与 $M_{\mathrm{xc}}=Q_{\alpha}+Q_{\beta}$。
- `lda_xc_direct_de()` = frozen-orbital direct skeleton。

### Step 1: LDA block-kernel 能量

当前 LDA block XC 能量写成

$$
\Omega_{\mathrm{xc}}^{\mathrm{LDA}}
=s_{\mathrm{LDA}}
\sum_{B,L}
M_{BL}(S)
\int
\rho_{B}(\mathbf{r})
f_{\mathrm{xc}}^{\mathrm{Ref}}(\mathbf{r})
\rho_{L}(\mathbf{r})
d\mathbf{r}.
$$

→ **解释：** 这是 `lda_xc_energy()` 的 grid 形式。`_lda_block_matrix(si)` 给出 $M_{BL}$，`lda_fxc_ref()` 给出 $f_{\mathrm{xc}}^{\mathrm{Ref}}$，`ni.eval_rho(..., hermi=0)` 给出非对称 pair density 的 $\rho_{B}$。

### Step 2: 对 block density 的一阶变分给出 `Q`

对某个 block $B$ 的 density 变分，有

$$
\delta\Omega_{\mathrm{xc}}^{\mathrm{LDA}}
=
2s_{\mathrm{LDA}}
\sum_{B,L}
M_{BL}(S)
\int
\delta\rho_{B}(\mathbf{r})
f_{\mathrm{xc}}^{\mathrm{Ref}}(\mathbf{r})
\rho_{L}(\mathbf{r})
d\mathbf{r}
+\delta\Omega_{\mathrm{ref}}.
$$

→ **解释：** $M_{BL}$ 对称，LDA kernel action 对左右 density 对称，所以 block density 变分产生因子 $2$。代码中这对应 `vblocks = 2.0 * SATDA_LDA_XC_GRAD_SCALE * einsum(mat[iblk], vsrc)`。

把 kernel action 记为

$$
\left[V_{B}^{\mathrm{xc}}\right]_{\mu\nu}
=
2s_{\mathrm{LDA}}
\sum_{L}
M_{BL}(S)
\int
\chi_{\mu}(\mathbf{r})\chi_{\nu}(\mathbf{r})
f_{\mathrm{xc}}^{\mathrm{Ref}}(\mathbf{r})
\rho_{L}(\mathbf{r})
d\mathbf{r}.
$$

→ **解释：** 这就是 `lda_apply_fxc_ref()` 后按 block matrix 组合出的 `vblocks[iblk]`。

例如若 $D^{B}=C_{T}X_{B}^{T}C_{S}^{T}$，则轨道系数变分产生

$$
\delta\Omega_{B}
=
\operatorname{Tr}
\left[
V_{B}^{\mathrm{xc}}
\delta C_{T}X_{B}^{T}C_{S}^{T}
\right]
+
\operatorname{Tr}
\left[
V_{B}^{\mathrm{xc}}
C_{T}X_{B}^{T}\delta C_{S}^{T}
\right].
$$

→ **解释：** 第一项把贡献放进 target spin channel，第二项把贡献放进 source spin channel。`lda_xc_q()` 中 `q_beta[:, target_idx]` 与 `q_alpha[:, source_idx]` 正是这两个项的 MO 形式。

此外，$f_{\mathrm{xc}}^{\mathrm{Ref}}$ 本身依赖参考态 half-density。其变分给出

$$
\delta\Omega_{\mathrm{ref}}
=
\operatorname{Tr}
\left[
W_{\alpha}\delta D_{\alpha}^{0}
\right]
+
\operatorname{Tr}
\left[
W_{\beta}\delta D_{\beta}^{0}
\right].
$$

→ **解释：** $W_{\alpha}$ 和 $W_{\beta}$ 来自 LDA 三阶导数 $k_{\mathrm{xc}}$。代码中它们是 `lda_ref_density_mats(..., with_deriv=False)` 返回的 `wa, wb`，随后加入占据列的 `q_alpha/q_beta`。

### Step 3: 轨道旋转导数矩阵

把 $Q_{\alpha}$ 与 $Q_{\beta}$ 合并到空间轨道变量，当前 helper 使用

$$
M_{\mathrm{xc}}=Q_{\alpha}+Q_{\beta}.
$$

→ **解释：** 这与已有 HF block helper 的 spatial M convention 一致。测试中对每个 canonical ROKS 反对称变量 $\kappa_{pq}$ 验证

$$
\frac{
\Omega_{\mathrm{xc}}(C e^{+\epsilon\kappa_{pq}})
-
\Omega_{\mathrm{xc}}(C e^{-\epsilon\kappa_{pq}})
}{
2\epsilon
}
=
\left[M_{\mathrm{xc}}-M_{\mathrm{xc}}^{T}\right]_{pq}.
$$

→ **解释：** 这就是新增测试 `test_migrated_lda_xc_m_matrix_matches_orbital_fd`。它不使用总梯度误差作为依据，而是直接验证 $M_{\mathrm{xc}}$ 的定义。

### Step 4: frozen-orbital direct skeleton

固定 MO coefficient 与振幅时，核坐标显式导数为

$$
\Omega_{\mathrm{xc,direct}}^{[x]}
=
s_{\mathrm{LDA}}
\sum_{B,L}M_{BL}(S)
\left[
\int
\rho_{B}^{[x]} f_{\mathrm{xc}}^{\mathrm{Ref}}\rho_{L}
+
\rho_{B} f_{\mathrm{xc}}^{\mathrm{Ref}}\rho_{L}^{[x]}
d\mathbf{r}
\right]
+\Omega_{k_{\mathrm{xc}}}^{[x]}.
$$

→ **解释：** 前两项来自 AO/grid 显式导数，对应 `lda_transition_deriv_mats()`；最后一项来自 reference density 改变导致 $f_{\mathrm{xc}}^{\mathrm{Ref}}$ 的核导数，对应 `lda_ref_density_mats(..., with_deriv=True)`。

在原子分块 AO 导数 convention 下，代码按该原子 AO 行块收缩：

$$
\frac{\partial\Omega_{\mathrm{xc,direct}}}{\partial R_{A,x}}
=
\sum_{B}
\left[
2\operatorname{Tr}_{A}
\left(
V_{B}^{[x]}D^{B}
\right)
+
2\operatorname{Tr}_{A}
\left(
V_{B}^{[x]}(D^{B})^{T}
\right)
\right]
+\operatorname{Tr}_{A}(W_{\alpha}^{[x]}D_{\alpha}^{0})
+\operatorname{Tr}_{A}(W_{\beta}^{[x]}D_{\beta}^{0})
+\mathrm{transpose\ terms}.
$$

→ **解释：** 这就是 `lda_xc_direct_de()` 中对 `trans_der[:, 1:]`、`wa_der[1:]`、`wb_der[1:]` 的收缩。新增测试 `test_migrated_lda_xc_direct_matches_frozen_fd` 用 frozen MO coefficient 的核位移 FD 验证该对象。

### 最终结果

当前第一阶段 LDA 实现已经锁定两个局部解析对象：

$$
\boxed{
M_{\mathrm{xc}}^{\mathrm{LDA}}
=
Q_{\alpha}^{\mathrm{LDA}}
+
Q_{\beta}^{\mathrm{LDA}}
}
$$

→ **解释：** 该对象通过 canonical ROKS orbital-rotation finite difference 验证。

$$
\boxed{
\Omega_{\mathrm{xc,direct}}^{\mathrm{LDA},[x]}
}
$$

→ **解释：** 该对象通过 frozen-orbital/frozen-amplitude nuclear finite difference 验证。

尚未由本节完成的是完整 DFT 总梯度中的 ROKS z-vector 核扰动 RHS：

$$
\mathbf{g}_{\mathrm{fix}}^{[x]}
=
\operatorname{pack}
\left[
C^{T}
\left(
h^{[x]}+J^{[x]}+v_{\mathrm{xc}}^{[x]}-\alpha_{\mathrm{x}}K^{[x]}
\right)
C
\right].
$$

→ **解释：** 当前 `_zvec_solver._perturbation_rhs()` 仍是 HF AO integral derivative convention。完整 LDA 总梯度闭合前必须单独校准这一项，不能把本节的两个局部闭合测试误读成完整 DFT 梯度已经完成。

**代码对应：**
- `pyscf/grad/tdsatda_delta/_xc_lda.py:lda_xc_energy`
- `pyscf/grad/tdsatda_delta/_xc_lda.py:lda_xc_q`
- `pyscf/grad/tdsatda_delta/_xc_lda.py:lda_xc_direct_de`
- `pyscf/grad/test/test_satda_grad.py:test_migrated_lda_xc_m_matrix_matches_orbital_fd`
- `pyscf/grad/test/test_satda_grad.py:test_migrated_lda_xc_direct_matches_frozen_fd`

## 推导：普通 SF-base 的 CV-CV 解析梯度 — 2026-06-01

**目标：** 对 SATDA `deltaS=-1` 振幅 $X$ 的普通 SF-base 分块标量
$E_{\mathrm{CV,CV}} = X_{\mathrm{CV}}^{T} A_{\mathrm{SF,CV,CV}} X_{\mathrm{CV}}$
推导 HF/ROKS 下的固定振幅核梯度。

**假设：**
- HF 极限，实轨道，TDA，固定 SATDA 振幅 $X_{ia}$。
- 这里的 $A_{\mathrm{SF}}$ 是普通 UKS spin-flip TDA 的 action 投影到 SATDA 的
  C/O/V block，不是 `SATDA.gen_vind_sf` 中含 $1/S_i$ spin-adaptation 项的
  SATDA 自身 CV-CV 块。
- $i,j$ 为 core 轨道，$a,b$ 为 virtual 轨道。

**代码变量到数学符号的对应：**
- `b.x_cv`, shape `(ncore, nvir)` = $X_{ia}$。
- `p_vv = einsum('ia,ib->ab', X, X)` = $P^{VV}_{ab}$。
- `p_cc = einsum('ia,ja->ij', X, X)` = $P^{CC}_{ij}$。
- `fbasis.fock0 - fbasis.fockz` = $F^{\beta}$。
- `fbasis.fock0 + fbasis.fockz` = $F^{\alpha}$。

### Step 1: CV-CV 能量标量

普通 SF-TDA 的 CV-CV action 为

$$
(A_{\mathrm{SF}} X)_{ia}
= \sum_b F^{\beta}_{ab} X_{ib}
- \sum_j F^{\alpha}_{ji} X_{ja}
- \sum_{jb} (ab|ji) X_{jb}.
$$

→ **解释：** 前两项是普通 spin-flip 的 beta-virtual 与 alpha-occupied
Fock 投影；最后一项是 HF exchange response。这里不含 SATDA 自身
`gen_vind_sf` 中的 $F^z/S_i$ 项。

因此固定振幅分块标量为

$$
E_{\mathrm{CV,CV}}
= \sum_{ab} P^{VV}_{ab} F^{\beta}_{ab}
- \sum_{ij} P^{CC}_{ij} F^{\alpha}_{ij}
- \sum_{iajb} X_{ia} X_{jb} (ab|ji).
$$

→ **解释：** 对 $i$ 或 $a$ 的 Kronecker delta 收缩分别给出
$P^{VV}$ 和 $P^{CC}$；ERI 项保留四指标形式。

### Step 2: Direct Skeleton

固定 AO 密度下的显式核导数为

$$
E_{\mathrm{dir}}^{[x]}
= \operatorname{Tr}\left[D^{VV} F_{\beta}^{[x]}\right]
- \operatorname{Tr}\left[D^{CC} F_{\alpha}^{[x]}\right]
- \sum_{iajb} X_{ia}X_{jb}(ab|ji)^{[x]}.
$$

→ **解释：** $D^{VV}_{\mu\nu}=C_{\mu a}P^{VV}_{ab}C_{\nu b}$，
$D^{CC}_{\mu\nu}=C_{\mu i}P^{CC}_{ij}C_{\nu j}$。ERI direct 项在代码中
按每个 $i,j$ 分解为
$D^{VV,ij}_{\mu\nu}=C_{\mu a}X_{ia}X_{jb}C_{\nu b}$ 和
$D^{CC,ji}_{\lambda\sigma}=C_{\lambda j}C_{\sigma i}$ 的 J 型双线性收缩。

### Step 3: Orbital Response M 矩阵

将 $E_{\mathrm{CV,CV}}$ 对 MO 系数的一阶导数写成

$$
\delta E_{\mathrm{CV,CV}}
= \sum_{pq} M_{pq}\kappa_{pq}
 + E_{\mathrm{dir}}^{[x]}.
$$

→ **解释：** 代码用 `_add_fock_term + _add_fock_response_q` 生成 Fock 部分
$M$，用 `_add_eri_term_q` 生成 ERI response 部分 $M$，再与 ROKS canonical
orbital response $\kappa^{[x]}$ 收缩。

最终梯度为

$$
E_{\mathrm{CV,CV}}^{[x]}
= E_{\mathrm{dir}}^{[x]}
+ \sum_{pq} M_{pq}\kappa_{pq}^{[x]}.
$$

→ **解释：** $\kappa^{[x]}$ 由 ROKS canonical CPKS 方程给出，包含非冗余
CO/CV/OV 转动和冗余 canonical gauge 转动；这与固定振幅、canonical displaced-SCF
FD 基准一致。

**代码对应：** `pyscf/grad/tdsatda_delta/_sfbase_grad.py` 中
`satda_sfbase_cvcv_explicit_energy`、
`satda_sfbase_cvcv_direct_grad`、
`satda_sfbase_cvcv_m_matrix` 和
`satda_sfbase_cvcv_analytic_grad`。

## 推导：普通 SF-base 的通用 block-pair 公式 — 2026-06-01

**目标：** 将上面的 CV-CV 公式推广到任意 block pair
$L\!R$，其中 block 名为 `CO/CV/OO/OV`，包括 `OO-OO`。

**假设：**
- 与上一节相同：HF、实轨道、固定 SATDA 振幅。
- `OO-OO` 使用同一普通 SF-TDA 公式；其 open-open canonical gauge 由 ROKS
  canonical CPKS 的 redundant `oo` pair 处理。

**符号声明：**
- 左 block 为 $X^{L}_{ia}$，右 block 为 $X^{R}_{jb}$。
- $i,j$ 属于左/右 alpha-occupied row 空间，可能是 C 或 O。
- $a,b$ 属于左/右 beta-virtual column 空间，可能是 O 或 V。

### Step 1: 通用 block 能量

普通 SF-TDA action 的 block-pair 标量为

$$
E_{L,R}
= \delta_{\mathrm{row}(L),\mathrm{row}(R)}
  \sum_{ab}\left(\sum_i X^L_{ia}X^R_{ib}\right)F^{\beta}_{ab}
- \delta_{\mathrm{col}(L),\mathrm{col}(R)}
  \sum_{ji}\left(\sum_a X^R_{ja}X^L_{ia}\right)F^{\alpha}_{ji}
- \sum_{iajb} X^L_{ia}X^R_{jb}(ab|ji).
$$

→ **解释：** beta Fock 只在左右 block 的 row 空间相同时非零；alpha Fock
只在左右 block 的 column 空间相同时非零；exchange response 是通用四指标项。

### Step 2: 通用 direct skeleton

固定 AO 密度的显式核导数为

$$
E_{L,R,\mathrm{dir}}^{[x]}
= \delta_{\mathrm{row}(L),\mathrm{row}(R)}
\operatorname{Tr}\left[D^{ab}_{L,R}F_{\beta}^{[x]}\right]
- \delta_{\mathrm{col}(L),\mathrm{col}(R)}
\operatorname{Tr}\left[D^{ji}_{R,L}F_{\alpha}^{[x]}\right]
- \sum_{iajb} X^L_{ia}X^R_{jb}(ab|ji)^{[x]}.
$$

→ **解释：** 前两项用 spin Fock 的完整 AO 骨架导数；最后一项使用通用 J 型
bilinear ERI derivative，避免为每个 block 手写不同指标排列。

### Step 3: 通用 M 矩阵

通用 orbital response 项仍写成

$$
E_{L,R,\mathrm{orb}}^{[x]}
= \sum_{pq} M^{L,R}_{pq}\kappa_{pq}^{[x]}.
$$

→ **解释：** `satda_sfbase_block_m_fock` 由 `_add_fock_term` 和
`_add_fock_response_q` 生成 Fock 与 Fock-response 的 $M$；
`satda_sfbase_block_m_hfx` 由 `_add_eri_term_q` 生成四指标 exchange response 的
$M$。二者相加后与 canonical ROKS CPKS 的 $\kappa^{[x]}$ 收缩。

**代码对应：** `pyscf/grad/tdsatda_delta/_sfbase_grad.py` 中
`satda_sfbase_block_explicit_energy`、
`satda_sfbase_block_direct_grad`、
`satda_sfbase_block_m_matrix` 和
`satda_sfbase_block_analytic_grad`。

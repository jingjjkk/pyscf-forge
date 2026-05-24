# SASF TDA 公式推导 — 2026-05-22

**目标：** 将 SASFTDA_Rev(2).pdf 中的理论与 `satda.py` 代码逐行对应，覆盖 $\Delta S = -1$ (spin-flip, $S_f = S_i - 1$) 和 $\Delta S = 0$ (spin-conserving, $S_f = S_i$) 两种情形的完整推导。

**代码对应：** `pyscf-forge/pyscf/sftda/satda.py` — `SATDA` 类及其 `gen_rohf_response_sf/sc`、`gen_vind_sf/sc` 函数。

---

## 1. ROKS 基态与轨道划分

### 1.1 自旋量子数

分子总自旋 $S$ 由 $\alpha$ 和 $\beta$ 电子数之差确定：

$$
S = \frac{N_{\alpha} - N_{\beta}}{2}
$$

**代码对应：** `satda.py:161`, `satda.py:296`, `satda.py:391`, `satda.py:510`
```python
s = (mol.nelec[0] - mol.nelec[1]) * 0.5
```

### 1.2 轨道空间划分 (ROKS 基态)

基于 ROKS 占据数 $n_p^{\text{ROKS}} \in \{2, 1, 0\}$ 将分子轨道分为三类：

| 轨道类型 | 占据数 | 指标 | 数目 | 代码变量 |
|---|---|---|---|---|
| 闭壳 (core) | $n_p = 2$ | $c,d \in \text{cs}$ | $N_{cs}$ | `csidx`, `orbcs` |
| 开壳 (open) | $n_p = 1$ | $u,v,w \in \text{os}$ | $N_{os}$ | `osidx`, `orbos` |
| 虚轨道 (virtual) | $n_p = 0$ | $a,b \in \text{vs}$ | $N_{vs}$ | `vsidx`, `orbvs` |

基态为 $S_z = S$ 的高自旋态，所有开壳轨道被 $\alpha$ 电子占据：

$$
|\text{ROKS}\rangle = |\underbrace{c_1\bar{c}_1 \dots c_{N_{cs}}\bar{c}_{N_{cs}}}_{\text{闭壳}} \underbrace{u_1 \dots u_{N_{os}}}_{\text{开壳(全 $\alpha$)}}\rangle
$$

其中 $N_{os} = 2S$.

**代码对应：** `satda.py:394-402`, `satda.py:513-521`
```python
csidx = np.where(mo_occ == 2)[0]   # 闭壳
osidx = np.where(mo_occ == 1)[0]   # 开壳
vsidx = np.where(mo_occ == 0)[0]   # 虚轨道
orbcs = mo_coeff[:, csidx]
orbos = mo_coeff[:, osidx]
orbvs = mo_coeff[:, vsidx]
```

### 1.3 ROKS Fock 矩阵

ROKS 的有效单粒子算符为（PDF §1）：

$$
\mathbf{F}^{\text{ROKS}} = \mathbf{h} + \mathbf{J}[\mathbf{D}_{\text{闭壳}} + \mathbf{D}_{\text{开壳}}] + \mathbf{V}_{xc}[\mathbf{D}_{\text{总}}]
$$

其中 $\mathbf{F}^{\alpha}$ 和 $\mathbf{F}^{\beta}$ 由 ROKS 耦合系数决定。代码中用 `mf.get_fock()` 直接获取:

**代码对应：** `satda.py:410-412`, `satda.py:530-531`
```python
fock = mf.get_fock()
focka = fock.focka   # F_α: α 自旋 Fock 矩阵
fockb = fock.fockb   # F_β: β 自旋 Fock 矩阵
```

---

## 2. $\Delta S = -1$: 自旋翻转 (Spin-Flip, $S_f = S_i - 1$)

### 2.1 激发算符与激发空间 (PDF §2.1)

对于 $\Delta S = -1$，自旋匹配的激发算符为（PDF §2.1, Eq. 1-2）：

$$
\mathbb{G}_{c \to u}^{\dagger} |\text{ROKS}\rangle = \frac{1}{\sqrt{2S}} \left(\hat{a}_{u\alpha}^{\dagger} \hat{a}_{c\beta} + \sum_{v=1}^{2S} \hat{a}_{u\beta}^{\dagger} \hat{a}_{v\alpha} \hat{a}_{v\beta}^{\dagger} \hat{a}_{c\beta}\right) |\text{ROKS}\rangle
$$

$$
\mathbb{G}_{u \to a}^{\dagger} |\text{ROKS}\rangle = \frac{1}{\sqrt{2S}} \left(\hat{a}_{a\alpha}^{\dagger} \hat{a}_{u\beta} + \sum_{v=1}^{2S} \hat{a}_{a\beta}^{\dagger} \hat{a}_{v\alpha} \hat{a}_{v\beta}^{\dagger} \hat{a}_{u\beta}\right) |\text{ROKS}\rangle
$$

这些算符产生 $S_f = S-1$ 的自旋纯态。

在 Tamm-Dancoff 近似下，激发空间由四种子块组成：

$$
\mathbf{X} = \begin{pmatrix}
\mathbf{X}_{co} & \mathbf{X}_{cv} \\
\mathbf{X}_{oo} & \mathbf{X}_{ov}
\end{pmatrix}
$$

其中:
- $\mathbf{X}_{co}$: $c \to u$ (闭壳 $\to$ 开壳) — shape $(N_{cs}, N_{os})$
- $\mathbf{X}_{cv}$: $c \to a$ (闭壳 $\to$ 虚轨道) — shape $(N_{cs}, N_{vs})$
- $\mathbf{X}_{oo}$: $u \to v$ (开壳 $\to$ 开壳) — shape $(N_{os}, N_{os})$
- $\mathbf{X}_{ov}$: $u \to a$ (开壳 $\to$ 虚轨道) — shape $(N_{os}, N_{vs})$

**代码对应：** `satda.py:522-523,564-569`
```python
nocc = ncs + nos
nvir = nos + nvs
zs = np.asarray(zs).reshape(-1, nocc, nvir)
zs_co = zs[:, :ncs, :nos]      # c→u
zs_cv = zs[:, :ncs, nos:]       # c→a
zs_oo = zs[:, ncs:, :nos]       # u→v
zs_ov = zs[:, ncs:, nos:]       # u→a
```

### 2.2 TDA 本征方程 (PDF §2.2)

自旋翻转 TDA 的本征值问题为：

$$
\mathbf{A} \cdot \mathbf{X} = \omega \mathbf{X}
$$

其中 $\mathbf{A}$ 矩阵具有块结构：

$$
\mathbf{A} = \begin{pmatrix}
\mathbf{A}_{co,co} & \mathbf{A}_{co,cv} & \mathbf{A}_{co,oo} & \mathbf{A}_{co,ov} \\
\mathbf{A}_{cv,co} & \mathbf{A}_{cv,cv} & \mathbf{A}_{cv,oo} & \mathbf{A}_{cv,ov} \\
\mathbf{A}_{oo,co} & \mathbf{A}_{oo,cv} & \mathbf{A}_{oo,oo} & \mathbf{A}_{oo,ov} \\
\mathbf{A}_{ov,co} & \mathbf{A}_{ov,cv} & \mathbf{A}_{ov,oo} & \mathbf{A}_{ov,ov}
\end{pmatrix}
$$

### 2.3 $\mathbf{A}$ 矩阵的 Fock + 响应函数分解 (PDF §2.3)

$\mathbf{A}$ 矩阵可以分解为纯 Fock 部分和响应函数部分：

$$
\mathbf{A} = \mathbf{A}^{\text{Fock}} + \mathbf{A}^{\text{Resp}}
$$

其中 $\mathbf{A}^{\text{Resp}}$ 进一步分解（对于 DFT）：

$$
\mathbf{A}^{\text{Resp}} = \mathbf{A}^{\text{Coulomb}} + \mathbf{A}^{\text{HFex}} + \mathbf{A}^{\text{XC}}
$$

$\mathbf{A}^{\text{Fock}}$ 由 Fock 矩阵元之差给出；$\mathbf{A}^{\text{XC}}$ 由交换相关核的积分给出：

$$
A_{ia,jb}^{\text{XC}} = \int d\mathbf{r} d\mathbf{r}' \, \phi_i(\mathbf{r}) \phi_a(\mathbf{r}) \, f_{xc}(\mathbf{r},\mathbf{r}') \, \phi_j(\mathbf{r}') \phi_b(\mathbf{r}')
$$

代码通过 **sigma 向量** 方式计算 $\boldsymbol{\sigma} = \mathbf{A} \cdot \mathbf{X}$，避免显式构建 $\mathbf{A}$ 矩阵。

### 2.4 响应函数 `gen_rohf_response_sf` (PDF §2.3, §2.4)

**代码对应：** `satda.py:284-382`

#### 2.4.1 XC 核 `fxc_ref` 的构造

ROKS 的密度响应 XC 核为（PDF Eq. 11-12）：

$$
f_{xc}^{\text{ref}}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) - f_{xc}^{\alpha\beta}(r) - f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r) \Big]
$$

这是在 collinear 表示下 `fxc_d0[0, :, 0]` 和 `fxc_d0[:, 0, :]` 等的组合。

**代码对应：** `satda.py:305-307` (deltaS=-1 版本)、`satda.py:170-171` (deltaS=0 版本)
```python
fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])
```

对于 `gen_rohf_response_sf`（deltaS=-1），仅需 $f_{xc}^{\text{ref}}$，而对 `gen_rohf_response_sc`（deltaS=0）还需要额外的 UKS 核分量（见 §3.3）。

#### 2.4.2 密度矩阵构造

对于每个 sigma 向量计算，先将激发振幅转为 AO 基密度矩阵：

$$
D_{\mu\nu}^{co} = \sum_{c,u} X_{c u} \, C_{\nu c}^{*} C_{\mu u}
$$

$$
D_{\mu\nu}^{cv} = \sum_{c,a} X_{c a} \, C_{\nu c}^{*} C_{\mu a}
$$

$$
D_{\mu\nu}^{oo} = \sum_{u,v} X_{u v} \, C_{\nu u}^{*} C_{\mu v} \quad (= \text{开壳} \to \text{开壳})
$$

$$
D_{\mu\nu}^{ov} = \sum_{u,a} X_{u a} \, C_{\nu u}^{*} C_{\mu a}
$$

**代码对应：** `satda.py:570-573`
```python
dms_co = lib.einsum('xov,pv,qo->xpq', zs_co, orbos, orbcs.conj())  # X_{c,u}: col v=os, row o=cs
dms_cv = lib.einsum('xov,pv,qo->xpq', zs_cv, orbvs, orbcs.conj())
dms_oo = lib.einsum('xov,pv,qo->xpq', zs_oo, orbos, orbos.conj())
dms_ov = lib.einsum('xov,pv,qo->xpq', zs_ov, orbvs, orbos.conj())
```

理解 einsum 的指标映射：`xov` = 态指标×行指标×列指标, `pv` = AO×列指标(MO), `qo` = AO×行指标(MO) → `xpq` = 态指标×AO×AO.

对于 $D_{co}$: $p \leftrightarrow$ virtual index (`orbos`, $u$), $q \leftrightarrow$ occupied index (`orbcs` conjugate, $c$):
$$
D_{\mu\nu}^{co} = \sum_{cu} X_{cu} \, C_{\mu u} \, C_{\nu c}^{*}
$$

#### 2.4.3 XC 响应计算

$\mathbf{A}^{\text{XC}}$ 的作用通过 numerical integration 计算:

对于完整 4-block 密度 $D^0 = [D^{co}, D^{cv}, D^{oo}, D^{ov}]$:
$$
V_{\mu\nu}^{\text{ref0}} = \int d\mathbf{r} \, \phi_{\mu}(\mathbf{r}) \phi_{\nu}(\mathbf{r}) \, f_{xc}^{\text{ref}}(\mathbf{r}) \, \sum_{pq} D_{pq} \phi_p(\mathbf{r}) \phi_q(\mathbf{r})
$$

对于简化 2-block 密度 $D^1 = [D^{co}, D^{ov}]$:
$$
V_{\mu\nu}^{\text{ref1}} = \int d\mathbf{r} \, \phi_{\mu}(\mathbf{r}) \phi_{\nu}(\mathbf{r}) \, f_{xc}^{\text{ref}}(\mathbf{r}) \, \sum_{pq} D_{pq} \phi_p(\mathbf{r}) \phi_q(\mathbf{r})
$$

**代码对应：** `satda.py:330-331,334-340`
```python
dms0 = np.concatenate((dms_co, dms_cv, dms_oo, dms_ov), axis=0)
dms1 = np.concatenate((dms_co, dms_ov), axis=0)
vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, ...)
vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, ...)  # LDA
# 或 nr_rks_fxc1_gga(...) / nr_rks_fxc1_mgga(...) for GGA/MGGA
```

#### 2.4.4 HF 交换响应

对于 hybrid 泛函，需要加上 HF 交换项减去 XC 核中的交换:

$$
V_{\mu\nu}^{\text{HF}} = -c_{hyb} \sum_{\lambda\sigma} (\mu\lambda|\nu\sigma) D_{\lambda\sigma}
$$

即 `mf.get_k(mol, dms0, hermi) * hyb` (带有 range-separated 修正).

**代码对应：** `satda.py:346-352`
```python
if hybrid:
    vk = mf.get_k(mol, dms0, hermi) * hyb
    vj = mf.get_j(mol, dms1, hermi) * hyb
    if omega != 0:
        vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
        vj += mf.get_j(mol, dms1, hermi, omega=omega) * (alpha - hyb)
    vref0 -= vk
    vref1 -= vj
```

注意：此处对 $D^0$（4-block）用 exchange $K$，对 $D^1$（2-block）用 Coulomb $J$。这是因为在 UKS 的 spin-flip TDA 中，自旋翻转激发只有 exchange 型耦合（α→β 方向），无 Coulomb 耦合。而 $D^1$ 的 $J$ 来自基态 Fock 矩阵的自旋适配修正。

#### 2.4.5 自旋适配系数组装 (PDF Eq. 20-24)

将 AO 基的 $V^{\text{ref0,1}}$ 按自旋适配系数组合到各激发子块。对于 $\Delta S = -1$，系数矩阵为：

$$
\begin{pmatrix}
\mathbf{V}_{co} \\
\mathbf{V}_{cv} \\
\mathbf{V}_{oo} \\
\mathbf{V}_{ov}
\end{pmatrix}
=
\mathbf{C}_{\Delta S=-1}
\begin{pmatrix}
\mathbf{V}_{co}^{\text{ref0}} \\
\mathbf{V}_{cv}^{\text{ref0}} \\
\mathbf{V}_{oo}^{\text{ref0}} \\
\mathbf{V}_{ov}^{\text{ref0}} \\
\mathbf{V}_{co}^{\text{ref1}} \\
\mathbf{V}_{ov}^{\text{ref1}}
\end{pmatrix}
$$

其中系数矩阵 $\mathbf{C}_{\Delta S=-1}$ 的元素（PDF Eq. 20-24）为：

| | $V_{co}^{\text{ref0}}$ | $V_{cv}^{\text{ref0}}$ | $V_{oo}^{\text{ref0}}$ | $V_{ov}^{\text{ref0}}$ | $V_{co}^{\text{ref1}}$ | $V_{ov}^{\text{ref1}}$ |
|---|---|---|---|---|---|---|
| $V_{co}$ | $1$ | $\sqrt{\frac{2S+1}{2S}}$ | $\sqrt{\frac{2S}{2S-1}}$ | $\frac{2S}{2S-1}$ | $\frac{1}{2S-1}$ | $-\frac{1}{2S-1}$ |
| $V_{cv}$ | $\sqrt{\frac{2S+1}{2S}}$ | $1$ | $\sqrt{\frac{2S+1}{2S-1}}$ | $\sqrt{\frac{2S+1}{2S}}$ | $0$ | $0$ |
| $V_{oo}$ | $\sqrt{\frac{2S}{2S-1}}$ | $\sqrt{\frac{2S+1}{2S-1}}$ | $1$ | $\sqrt{\frac{2S}{2S-1}}$ | $0$ | $0$ |
| $V_{ov}$ | $\frac{2S}{2S-1}$ | $\sqrt{\frac{2S+1}{2S}}$ | $\sqrt{\frac{2S}{2S-1}}$ | $1$ | $-\frac{1}{2S-1}$ | $\frac{1}{2S-1}$ |

**代码对应：** `satda.py:361-368`

逐个验证：

**$V_{co}$ (line 361-362):**
```python
v1ao_co += vref0_co + vref1_co/(2*s - 1) + np.sqrt((2*s + 1)/2/s)*vref0_cv
v1ao_co += np.sqrt(2*s/(2*s - 1))*vref0_oo + 2*s/(2*s-1)*vref0_ov - vref1_ov/(2*s-1)
```
→ 系数: $(1, \sqrt{\frac{2S+1}{2S}}, \sqrt{\frac{2S}{2S-1}}, \frac{2S}{2S-1}, \frac{1}{2S-1}, -\frac{1}{2S-1})$ ✓

**$V_{cv}$ (line 363-364):**
```python
v1ao_cv += vref0_co*np.sqrt((2*s+1)/2/s) + vref0_cv + np.sqrt((2*s+1)/(2*s-1))*vref0_oo
v1ao_cv += np.sqrt((2*s+1)/2/s)*vref0_ov
```
→ 系数: $(\sqrt{\frac{2S+1}{2S}}, 1, \sqrt{\frac{2S+1}{2S-1}}, \sqrt{\frac{2S+1}{2S}}, 0, 0)$ ✓

**$V_{oo}$ (line 365-366):**
```python
v1ao_oo += np.sqrt(2*s/(2*s-1))*vref0_co + np.sqrt((2*s+1)/(2*s-1))*vref0_cv
v1ao_oo += vref0_oo + np.sqrt(2*s/(2*s-1))*vref0_ov
```
→ 系数: $(\sqrt{\frac{2S}{2S-1}}, \sqrt{\frac{2S+1}{2S-1}}, 1, \sqrt{\frac{2S}{2S-1}}, 0, 0)$ ✓

**$V_{ov}$ (line 367-368):**
```python
v1ao_ov += 2*s/(2*s-1)*vref0_co - vref1_co/(2*s-1) + np.sqrt((2*s+1)/2/s)*vref0_cv
v1ao_ov += np.sqrt(2*s/(2*s-1))*vref0_oo + vref0_ov + vref1_ov/(2*s-1)
```
→ 系数: $(\frac{2S}{2S-1}, \sqrt{\frac{2S+1}{2S}}, \sqrt{\frac{2S}{2S-1}}, 1, -\frac{1}{2S-1}, \frac{1}{2S-1})$ ✓

#### 2.4.6 Fockz: 开壳密度响应 (PDF §2.5)

`fockz = 0.5 * delta` 是开壳 Fock 矩阵的响应修正:

$$
\mathbf{F}^z = \frac{1}{2} \left[ \mathbf{K}^{\text{ref}}[\mathbf{D}_{oo}] - c_{hyb} \mathbf{K}^{\text{HF}}[\mathbf{D}_{oo}] \right]
$$

其中 $\mathbf{D}_{oo} = \sum_{u \in os} \phi_u \phi_u^{\dagger}$ 为开壳轨道密度。

**代码对应：** `satda.py:372-382`
```python
orbos = mo_coeff[:, np.where(mo_occ == 1)[0]]
dmoo = orbos @ orbos.T
if xctype != 'HF':
    delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, ...)
else:
    delta = np.zeros_like(dmoo)
if hybrid:
    delta -= mf.get_k(mol, dmoo, 1) * hyb
    if omega != 0:
        delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
return vind, 0.5 * delta
```

### 2.5 `gen_vind_sf`: 完整 Sigma 向量 (PDF §2.6)

**代码对应：** `satda.py:503-617`

#### 2.5.1 Fock 矩阵的分块定义

在得到 `vresp` (XC+HF 响应) 和 `fockz` 后，定义有效 Fock 矩阵:

$$
\mathbf{F}^0 = \mathbf{F}^{\alpha} - \mathbf{F}^z
$$

**代码对应：** `satda.py:532`
```python
fock0 = focka - fockz
```

然后在各轨道子空间投影:

**代码对应：** `satda.py:534-552`

| 名称 | 定义 | MO 基投影 |
|---|---|---|
| $F_{co}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{os}$ |
| $F_{co}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{cs}^{\dagger} (F^0 + F^z) \mathbf{C}_{cs}$ |
| $F_{co}^{2}$ | $F^z$ | $\mathbf{C}_{cs}^{\dagger} F^z \mathbf{C}_{cs}$ |
| $F_{cv}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ |
| $F_{cooo}^{0}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{cs}$ |
| $F_{cooo}^{1}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{cs}$ |
| $F_{cv}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{vs}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ |
| $F_{cv}^{2}$ | $F^z$ | $\mathbf{C}_{vs}^{\dagger} F^z \mathbf{C}_{vs}$ |
| $F_{cv}^{3}$ | $F^z$ | $\mathbf{C}_{cs}^{\dagger} F^z \mathbf{C}_{cs}$ |
| $F_{cvoo}$ | $F^z$ | $\mathbf{C}_{vs}^{\dagger} F^z \mathbf{C}_{cs}$ |
| $F_{c\tilde{v}}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{cs}$ (同 $F_{cooo}^{0}$) |
| $F_{ooo}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{os}$ (同 $F_{co}^{0}$) |
| $F_{ooo}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{os}$ |
| $F_{ooov}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ (同 $F_{cv}$) |
| $F_{ooov}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{vs}$ |
| $F_{ov}^{0}$ | $F^0 - F^z$ | $\mathbf{C}_{vs}^{\dagger} (F^0 - F^z) \mathbf{C}_{vs}$ (同 $F_{cv}^{0}$) |
| $F_{ov}^{1}$ | $F^0 + F^z$ | $\mathbf{C}_{os}^{\dagger} (F^0 + F^z) \mathbf{C}_{os}$ (同 $F_{ooo}^{1}$) |
| $F_{ov}^{2}$ | $F^z$ | $\mathbf{C}_{vs}^{\dagger} F^z \mathbf{C}_{vs}$ (同 $F_{cv}^{2}$) |

**代码对应：** `satda.py:534-552`
```python
fock_coco0 = orbos.T @ (fock0 - fockz) @ orbos
fock_coco1 = orbcs.T @ (fock0 + fockz) @ orbcs
fock_coco2 = orbcs.T @ fockz @ orbcs
fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
fock_cooo0 = orbos.T @ (fock0 + fockz) @ orbcs
fock_cooo1 = orbos.T @ (fock0 - fockz) @ orbcs
fock_cvcv0 = orbvs.T @ (fock0 - fockz) @ orbvs
fock_cvcv1 = fock_coco1
fock_cvcv2 = orbvs.T @ fockz @ orbvs
fock_cvcv3 = fock_coco2
fock_cvoo = orbvs.T @ fockz @ orbcs
fock_cvov = fock_cooo0
fock_oooo0 = fock_coco0
fock_oooo1 = orbos.T @ (fock0 + fockz) @ orbos
fock_ooov0 = fock_cocv
fock_ooov1 = orbos.T @ (fock0 + fockz) @ orbvs
fock_ovov0 = fock_cvcv0
fock_ovov1 = fock_oooo1
fock_ovov2 = fock_cvcv2
```

#### 2.5.2 对角元 (preconditioner diagonal, PDF §2.6)

对于 TDA 迭代对角化的预处理器，需要 $\mathbf{A}$ 的对角近似:

**co 块 (line 555-556):**

$$
h_{\text{diag}}^{co} = \varepsilon_{u}^{F^0-F^z} \otimes \mathbf{1}_{c} - \mathbf{1}_{u} \otimes \varepsilon_{c}^{F^0+F^z} - \frac{2}{2S-1} \cdot \mathbf{1}_{u} \otimes \varepsilon_{c}^{F^z}
$$

即:
$$
h_{\text{diag}}^{co}[i, u] = \varepsilon_u^{(0)} - \varepsilon_c^{(1)} - \frac{2}{2S-1} \varepsilon_c^{(2)}
$$

其中 $\varepsilon^{(0)} = \text{diag}(F_{co}^{0})$, $\varepsilon^{(1)} = \text{diag}(F_{co}^{1})$, $\varepsilon^{(2)} = \text{diag}(F_{co}^{2})$.

**cv 块 (line 557-558):**

$$
h_{\text{diag}}^{cv} = \varepsilon_{a}^{F^0-F^z} \otimes \mathbf{1}_{c} - \mathbf{1}_{a} \otimes \varepsilon_{c}^{F^0+F^z} - \frac{1}{S} \varepsilon_{a}^{F^z} \otimes \mathbf{1}_{c} + \frac{1}{S} \cdot \mathbf{1}_{a} \otimes \varepsilon_{c}^{F^z}
$$

**oo 块 (line 559):**

$$
h_{\text{diag}}^{oo} = \varepsilon_{v}^{F^0-F^z} \otimes \mathbf{1}_{u} - \mathbf{1}_{v} \otimes \varepsilon_{u}^{F^0+F^z}
$$

**ov 块 (line 560-561):**

$$
h_{\text{diag}}^{ov} = \varepsilon_{a}^{F^0-F^z} \otimes \mathbf{1}_{u} - \mathbf{1}_{a} \otimes \varepsilon_{u}^{F^0+F^z} - \frac{2}{2S-1} \varepsilon_{a}^{F^z} \otimes \mathbf{1}_{u}
$$

**代码对应：** `satda.py:555-561`
```python
hdiag_co = fock_coco0.diagonal()[None, :] - fock_coco1.diagonal()[:, None]
hdiag_co -= fock_coco2.diagonal()[:, None] * 2 / (2 * s - 1)
hdiag_cv = fock_cvcv0.diagonal()[None, :] - fock_cvcv1.diagonal()[:, None]
hdiag_cv -= fock_cvcv2.diagonal()[None, :] / s + fock_cvcv3.diagonal()[:, None] / s
hdiag_oo = fock_oooo0.diagonal()[None, :] - fock_oooo1.diagonal()[:, None]
hdiag_ov = fock_ovov0.diagonal()[None, :] - fock_ovov1.diagonal()[:, None]
hdiag_ov -= fock_ovov2.diagonal()[None, :] * 2 / (2 * s - 1)
```

最终 `hdiag = np.block([[hdiag_co, hdiag_cv], [hdiag_oo, hdiag_ov]]).ravel()` (line 562).

#### 2.5.3 Vind 函数: Fock 项叠加 (PDF §2.6)

在计算 AO 基的 XC 响应 `v1ao_co/cv/oo/ov` 后，转为 MO 基并叠加 Fock 矩阵项。

**AO → MO 变换:**

$$
V_{cu}^{\text{MO}} = \sum_{\mu\nu} C_{\mu c}^{*} \, V_{\mu\nu}^{\text{AO}} \, C_{\nu u}
$$

**代码对应：** `satda.py:575-578`
```python
v1mo_co = lib.einsum('xpq,qo,pv->xov', v1ao_co, orbcs, orbos.conj())
v1mo_cv = lib.einsum('xpq,qo,pv->xov', v1ao_cv, orbcs, orbvs.conj())
v1mo_oo = lib.einsum('xpq,qo,pv->xov', v1ao_oo, orbos, orbos.conj())
v1mo_ov = lib.einsum('xpq,qo,pv->xov', v1ao_ov, orbos, orbvs.conj())
```

##### co 块 Fock 项 (PDF Eq. 25)

从自旋适配单粒子能差的推广:

$$
A_{cu, dv}^{\text{Fock}} = \delta_{cd} \, F_{vu}^{F^0-F^z} - \delta_{uv} \, F_{cd}^{F^0+F^z} - \frac{2}{2S-1} \delta_{uv} \, F_{cd}^{F^z}
$$

加上非对角耦合项（激发算符之间的单粒子耦合）:

$$
A_{cu, da}^{\text{Fock,coupling}} = \sqrt{\frac{2S+1}{2S}} \, \delta_{cd} \, F_{ua}^{F^0-F^z} \times X_{da}
$$

$$
A_{cu, vw}^{\text{Fock,coupling}} = -\sqrt{\frac{2S}{2S-1}} \, \delta_{uv} \, F_{wc}^{F^0+F^z} \times X_{vw} \;+\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{vw} \, F_{uc}^{F^0-F^z} \times X_{vc}
$$

**代码对应 (co 块):** `satda.py:580-585`
```python
v1mo_co += lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco0, zs_co)          # δ_cd F^0-F^z_vu X_dv
v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco1, zs_co)          # -δ_uv F^0+F^z_cd X_dv
v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco2, zs_co) * 2/(2s-1) # -2/(2S-1) δ_uv F^z_cd X_dv
v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv, zs_cv) * sqrt((2s+1)/2/s)  # √(...)·δ_cd F^0-F^z_va X_da
v1mo_co -= lib.einsum('uv,wi,xwv->xiu', np.eye(nos), fock_cooo0, zs_oo) * sqrt(2s/(2s-1)) # -√(...)·δ_uv F^0+F^z_wc X_vw
v1mo_co += lib.einsum('vw,ui,xwv->xiu', np.eye(nos), fock_cooo1, zs_oo) / sqrt(2*s*(2*s-1))  # 1/√(...)·δ_vw F^0-F^z_uc X_vc
```

einsum 指标解析:
- `lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco0, zs_co)`:
  - `ij` = δ_cd (闭壳指标)
  - `uv` = F^0-F^z 矩阵在开壳空间中, indices (v,u) → (u,v) 转置
  - `xjv` = zs_co 在 (d→v) 指标
  - `->xiu` = 输出 (c→u)
  - 等价于: $V_{cu} = \sum_{d,v} \delta_{cd} \, F_{vu}^{F^0-F^z} \, X_{dv}$

##### cv 块 Fock 项 (PDF Eq. 26)

$$
A_{ca, db}^{\text{Fock}} = \delta_{cd} \, F_{ab}^{F^0-F^z} - \delta_{ab} \, F_{cd}^{F^0+F^z} - \frac{1}{S} \delta_{cd} \, F_{ab}^{F^z} - \frac{1}{S} \delta_{ab} \, F_{cd}^{F^z}
$$

加上耦合项:

$$
A_{ca, du}^{\text{coupling}} = \sqrt{\frac{2S+1}{2S}} \, \delta_{cd} \, (F_{ua}^{F^0-F^z})^{\dagger} \times X_{du}
$$

$$
A_{ca, vw}^{\text{coupling}} = -\frac{1}{S} \sqrt{\frac{2S+1}{2S-1}} \, \delta_{vw} \, F_{ac}^{F^z} \times X_{vw}
$$

$$
A_{ca, vb}^{\text{coupling}} = -\sqrt{\frac{2S+1}{2S}} \, \delta_{ab} \, F_{cv}^{F^0+F^z} \times X_{vb}
$$

**代码对应 (cv 块):** `satda.py:587-593`
```python
v1mo_cv += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv.T, zs_co) * sqrt((2s+1)/2/s)
v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv0, zs_cv)
v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv1, zs_cv)
v1mo_cv -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv2, zs_cv) / s
v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv3, zs_cv) / s
v1mo_cv -= lib.einsum('vw,ai,xwv->xia', np.eye(nos), fock_cvoo, zs_oo) / s * sqrt((2s+1)/(2s-1))
v1mo_cv -= lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_cvov, zs_ov) * sqrt((2s+1)/2/s)
```

##### oo 块 Fock 项 (PDF Eq. 27)

$$
A_{uv, wt}^{\text{Fock}} = \delta_{uw} \, F_{vt}^{F^0-F^z} - \delta_{vt} \, F_{uw}^{F^0+F^z}
$$

耦合项:

$$
A_{uv, cd}^{\text{coupling}} = -\sqrt{\frac{2S}{2S-1}} \, \delta_{vt} \, (F_{cu}^{F^0+F^z})^{\dagger} \times X_{cd}
\;+\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{ut} \, (F_{cv}^{F^0-F^z})^{\dagger} \times X_{cv}
$$

$$
A_{uv, ca}^{\text{coupling}} = -\frac{1}{S} \sqrt{\frac{2S+1}{2S-1}} \, \delta_{ut} \, (F_{cv}^{F^z})^{\dagger} \times X_{ca}
$$

$$
A_{uv, wa}^{\text{coupling}} = \sqrt{\frac{2S}{2S-1}} \, \delta_{uv} \, F_{ta}^{F^0-F^z} \times X_{wt}
\;-\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{ut} \, F_{va}^{F^0+F^z} \times X_{tv}
$$

**代码对应 (oo 块):** `satda.py:595-601`
```python
v1mo_oo -= lib.einsum('vt,ju,xjv->xut', np.eye(nos), fock_cooo0.T, zs_co) * sqrt(2*s/(2*s-1))
v1mo_oo += lib.einsum('ut,jv,xjv->xut', np.eye(nos), fock_cooo1.T, zs_co) / sqrt(2*s*(2*s-1))
v1mo_oo -= lib.einsum('ut,jb,xjb->xut', np.eye(nos), fock_cvoo.T, zs_cv) / s * sqrt((2*s+1)/(2*s-1))
v1mo_oo += lib.einsum('wu,tv,xwv->xut', np.eye(nos), fock_oooo0, zs_oo)
v1mo_oo -= lib.einsum('tv,wu,xwv->xut', np.eye(nos), fock_oooo1, zs_oo)
v1mo_oo += lib.einsum('uv,tb,xvb->xut', np.eye(nos), fock_ooov0, zs_ov) * sqrt(2*s/(2*s-1))
v1mo_oo -= lib.einsum('tu,vb,xvb->xut', np.eye(nos), fock_ooov1, zs_ov) / sqrt(2*s*(2*s-1))
```

##### ov 块 Fock 项 (PDF Eq. 28)

$$
A_{ua, vb}^{\text{Fock}} = \delta_{uv} \, F_{ab}^{F^0-F^z} - \delta_{ab} \, F_{vu}^{F^0+F^z} - \frac{2}{2S-1} \delta_{uv} \, F_{ab}^{F^z}
$$

耦合项:

$$
A_{ua, cb}^{\text{coupling}} = -\sqrt{\frac{2S+1}{2S}} \, \delta_{ab} \, (F_{cu}^{F^0+F^z})^{\dagger} \times X_{cb}
$$

$$
A_{ua, wv}^{\text{coupling}} = \sqrt{\frac{2S}{2S-1}} \, \delta_{uw} \, (F_{va}^{F^0-F^z})^{\dagger} \times X_{wv}
\;-\; \frac{1}{\sqrt{2S(2S-1)}} \, \delta_{vw} \, (F_{ua}^{F^0+F^z})^{\dagger} \times X_{vw}
$$

**代码对应 (ov 块):** `satda.py:603-608`
```python
v1mo_ov -= lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_cvov.T, zs_cv) * sqrt((2s+1)/2/s)
v1mo_ov += lib.einsum('uw,av,xwv->xua', np.eye(nos), fock_ooov0.T, zs_oo) * sqrt(2*s/(2*s-1))
v1mo_ov -= lib.einsum('vw,au,xwv->xua', np.eye(nos), fock_ooov1.T, zs_oo) / sqrt(2*s*(2*s-1))
v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov0, zs_ov)
v1mo_ov -= lib.einsum('ab,vu,xvb->xua', np.eye(nvs), fock_ovov1, zs_ov)
v1mo_ov -= lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov2, zs_ov) * 2/(2*s-1)
```

最后，将四个子块组装为总复矢量 `v1mo[:, :ncs, :nos] = v1mo_co` 等 (lines 610-614)。

---

## 3. $\Delta S = 0$: 自旋守恒 (Spin-Conserving, $S_f = S_i$)

### 3.1 激发算符 (PDF §3.1)

对于自旋守恒激发 ($S_f = S_i = S$)，需要三类自旋匹配算符除去零激发态（纯 $D_{oo}$）：

**G_co (闭壳→开壳):**

$$
\mathbb{G}_{c \to u}^{\dagger} |\text{ROKS}\rangle = \frac{1}{\sqrt{2S+2}} \left(\hat{a}_{u\alpha}^{\dagger} \hat{a}_{c\alpha} + \hat{a}_{u\beta}^{\dagger} \hat{a}_{c\beta}\right) |\text{ROKS}\rangle
$$

**G_cv (闭壳→虚轨道 + 开壳→虚轨道 的叠加，非 cv0 道):**

两个正交的正交道用于激发:
- "cv" 道: $\mathbb{G}_{c \to a}^{(1)}$ — 对称形式的叠加
- "cv0" 道: $\mathbb{G}_{c \to a}^{(2)}$ — 反对称形式的叠加

**G_ov (开壳→虚轨道, 与 G_co 配对的 partner):**

与前两者配对的组合算符。

此外还有一个特殊的单参数算符 $\mathbb{G}_{oo}$ (开壳→开壳)，只有一个自由度。

### 3.2 激发空间 (PDF §3.2)

激发空间由 5 个矢量组成 (包括 `cv0`):

$$
\mathbf{X} = \begin{pmatrix}
\mathbf{X}_{co} & \mathbf{X}_{cv} \\
\mathbf{X}_{oo} & \mathbf{X}_{ov} \\
& \mathbf{X}_{cv0}
\end{pmatrix}
$$

**代码对应：** `satda.py:403-406`
```python
idx1 = ncs * nos          # co: ncs × nos
idx2 = idx1 + ncs * nvs   # cv: ncs × nvs
idx3 = idx2 + 1           # oo: 1 element (scalar!)
idx4 = idx3 + nos * nvs   # ov: nos × nvs
# cv0: rest, ncs × nvs
```

**关键点：** $oo$ 块仅 1 个元素，因为自旋守恒时只有一个独立的 $D_{oo}$ 算符（对角线）可以激发而不改变自旋。这与 $\Delta S = -1$ 的 $oo$ 块（shape $(N_{os}, N_{os})$）完全不同的物理根源。

### 3.3 响应函数 `gen_rohf_response_sc` (PDF §3.3-3.5)

**代码对应：** `satda.py:149-282`

#### 3.3.1 多个 XC 核的构造

对于 $\Delta S = 0$，仅 $f_{xc}^{\text{ref}}$ 不够，还需要 5 个额外的核分量（PDF Eq. 30-35）:

从 ROKS 密度核:
$$
f_{xc}^{\text{ref}}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) - f_{xc}^{\alpha\beta}(r) - f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r) \Big]
$$

**代码对应：** `satda.py:170-171`
```python
fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])
```

从 UKS 密度核（转为 UKS 以获取非零自旋密度的核）:
$$
f_{xc}^{s}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\alpha\beta}(r) + f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r) \Big]
$$

$$
f_{xc}^{cv0,cv}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\alpha\beta}(r) - f_{xc}^{\beta\alpha}(r) - f_{xc}^{\beta\beta}(r) \Big]
$$

$$
f_{xc}^{cv,cv0}(r) = \frac{1}{2} \Big[ f_{xc}^{\alpha\alpha}(r) - f_{xc}^{\alpha\beta}(r) + f_{xc}^{\beta\alpha}(r) - f_{xc}^{\beta\beta}(r) \Big]
$$

$$
f_{xc}^{cv0,co}(r) = f_{xc}^{\alpha\beta}(r) + f_{xc}^{\beta\beta}(r)
$$

$$
f_{xc}^{cv0,ov}(r) = f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\beta\alpha}(r)
$$

$$
f_{xc}^{co,cv0}(r) = f_{xc}^{\beta\alpha}(r) + f_{xc}^{\beta\beta}(r)
$$

$$
f_{xc}^{ov,cv0}(r) = f_{xc}^{\alpha\alpha}(r) + f_{xc}^{\alpha\beta}(r)
$$

**代码对应：** `satda.py:172-181`
```python
umf = mf.to_uks()
uni = umf._numint
_, _, fxc = uni.cache_xc_kernel(mol, mf.grids, mf.xc, umf.mo_coeff, umf.mo_occ, 1)
fxc_s = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1] + fxc[1, :, 0] + fxc[1, :, 1])
fxc_cv0cv = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1] - fxc[1, :, 0] - fxc[1, :, 1])
fxc_cvcv0 = 0.5 * (fxc[0, :, 0] - fxc[0, :, 1] + fxc[1, :, 0] - fxc[1, :, 1])
fxc_cv0co = fxc[0, :, 1] + fxc[1, :, 1]
fxc_cv0ov = fxc[0, :, 0] + fxc[1, :, 0]
fxc_cocv0 = fxc[1, :, 0] + fxc[1, :, 1]
fxc_ovcv0 = fxc[0, :, 0] + fxc[0, :, 1]
```

#### 3.3.2 Vind 函数: 四项贡献 (PDF §3.4)

响应函数由四部分构成：Coulomb 部分、HF 交换部分、K^Ref 部分、K^CV0 部分。

##### Coulomb 部分 (PDF Eq. 36-38)

$\Delta S = 0$ 时存在 Coulomb 耦合（两个激发算符通过 Coulomb 算符连接产生 $cv0$ 道）:

$$
\mathbf{V}_{co}^{\text{Coul}} = \sqrt{2} \, \mathbf{J}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{ov}^{\text{Coul}} = -\sqrt{2} \, \mathbf{J}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{cv0}^{\text{Coul}} = \sqrt{2} \, \mathbf{J}[\mathbf{D}_{co}] - \sqrt{2} \, \mathbf{J}[\mathbf{D}_{ov}] + 2 \, \mathbf{J}[\mathbf{D}_{cv0}]
$$

**代码对应：** `satda.py:206-214`
```python
dms_j = np.concatenate((dms_co, dms_ov, dms_cv0), axis=0)
vcoul = mf.get_j(mol, dms_j, hermi)
vcoul_co = vcoul[:idx1]
vcoul_ov = vcoul[idx1:idx1+n_ov]
vcoul_cv0 = vcoul[idx1+n_ov:]
v1ao_co += np.sqrt(2) * vcoul_cv0
v1ao_ov -= np.sqrt(2) * vcoul_cv0
v1ao_cv0 += np.sqrt(2) * vcoul_co - np.sqrt(2) * vcoul_ov + 2 * vcoul_cv0
```

##### HF 交换部分 (PDF Eq. 39-42)

对于 hybrid 泛函，HF 交换贡献带有自旋适配系数:

$$
\mathbf{V}_{co}^{\text{HF}} = -\mathbf{K}[\mathbf{D}_{co}] + \mathbf{J}[\mathbf{D}_{co}] - \sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{cv}] - \mathbf{J}[\mathbf{D}_{ov}] - \frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{cv}^{\text{HF}} = -\sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{co}] - \mathbf{K}[\mathbf{D}_{cv}] - \sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{ov}]
$$

$$
\mathbf{V}_{ov}^{\text{HF}} = -\mathbf{J}[\mathbf{D}_{co}] - \sqrt{\frac{S+1}{2S}} \mathbf{K}[\mathbf{D}_{cv}] + \mathbf{J}[\mathbf{D}_{ov}] - \mathbf{K}[\mathbf{D}_{ov}] + \frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{cv0}]
$$

$$
\mathbf{V}_{cv0}^{\text{HF}} = -\frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{co}] + \frac{1}{\sqrt{2}} \mathbf{K}[\mathbf{D}_{ov}] - \mathbf{K}[\mathbf{D}_{cv0}]
$$

**代码对应：** `satda.py:216-233`
```python
dms = np.concatenate((dms_co, dms_cv, dms_ov, dms_cv0), axis=0)
vk = mf.get_k(mol, dms, hermi) * hyb
vj = vcoul[:idx1+n_ov] * hyb
# range-separated correction...
v1ao_co += - vk_co + vj_co - np.sqrt((s+1)/2/s) * vk_cv - vj_ov - np.sqrt(0.5) * vk_cv0
v1ao_cv += - np.sqrt((s+1)/2/s) * vk_co - vk_cv - np.sqrt((s+1)/2/s) * vk_ov
v1ao_ov += - vj_co - np.sqrt((s+1)/2/s) * vk_cv + vj_ov - vk_ov + np.sqrt(0.5) * vk_cv0
v1ao_cv0 += - np.sqrt(0.5) * vk_co + np.sqrt(0.5) * vk_ov - vk_cv0
```

##### K^Ref 部分 (PDF Eq. 43-46)

与 $\Delta S = -1$ 结构相似但系数用 $S$ 而非 $2S$:

$$
\mathbf{V}_{co}^{\text{Ref}} = \mathbf{V}_{co}^{\text{ref0}} - \mathbf{V}_{co}^{\text{ref1}} + \sqrt{\frac{S+1}{2S}} \mathbf{V}_{cv}^{\text{ref0}} + \mathbf{V}_{ov}^{\text{ref1}}
$$

$$
\mathbf{V}_{cv}^{\text{Ref}} = \sqrt{\frac{S+1}{2S}} \mathbf{V}_{co}^{\text{ref0}} + \mathbf{V}_{cv}^{\text{ref0}} + \sqrt{\frac{S+1}{2S}} \mathbf{V}_{ov}^{\text{ref0}}
$$

$$
\mathbf{V}_{ov}^{\text{Ref}} = \mathbf{V}_{co}^{\text{ref1}} + \sqrt{\frac{S+1}{2S}} \mathbf{V}_{cv}^{\text{ref0}} - \mathbf{V}_{ov}^{\text{ref1}} + \mathbf{V}_{ov}^{\text{ref0}}
$$

**代码对应：** `satda.py:235-255`
```python
# vref0 = fxc_ref applied to dms0=(dms_co, dms_cv, dms_ov)
# vref1 = fxc_ref applied to dms1=(dms_co, dms_ov)
v1ao_co += vref0_co - vref1_co + np.sqrt((s+1)/2/s) * vref0_cv + vref1_ov
v1ao_cv += np.sqrt((s+1)/2/s) * vref0_co + vref0_cv + np.sqrt((s+1)/2/s) * vref0_ov
v1ao_ov += vref1_co + np.sqrt((s+1)/2/s) * vref0_cv - vref1_ov + vref0_ov
```

##### K^CV0 部分 (PDF Eq. 47-50)

cv0 道需要额外的 XC 核分量:

$$
\mathbf{V}_{co}^{\text{CV0}} = \frac{1}{\sqrt{2}} \, f_{xc}^{co,cv0} \cdot \mathbf{D}_{cv0}
$$

$$
\mathbf{V}_{cv}^{\text{CV0}} = -\sqrt{\frac{S+1}{S}} \, f_{xc}^{cv,cv0} \cdot \mathbf{D}_{cv0}
$$

$$
\mathbf{V}_{ov}^{\text{CV0}} = -\frac{1}{\sqrt{2}} \, f_{xc}^{ov,cv0} \cdot \mathbf{D}_{cv0}
$$

$$
\mathbf{V}_{cv0}^{\text{CV0}} = \frac{1}{\sqrt{2}} \, f_{xc}^{cv0,co} \cdot \mathbf{D}_{co} - \sqrt{\frac{S+1}{S}} \, f_{xc}^{cv0,cv} \cdot \mathbf{D}_{cv}
- \frac{1}{\sqrt{2}} \, f_{xc}^{cv0,ov} \cdot \mathbf{D}_{ov} + f_{xc}^{s} \cdot \mathbf{D}_{cv0}
$$

**代码对应：** `satda.py:257-269`
```python
v_cocv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_cocv0, ...)
v_cvcv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_cvcv0, ...)
v_ovcv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_ovcv0, ...)
v_cv0co = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_co, 0, hermi, None, None, fxc_cv0co, ...)
v_cv0cv = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv, 0, hermi, None, None, fxc_cv0cv, ...)
v_cv0ov = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_ov, 0, hermi, None, None, fxc_cv0ov, ...)
v_cv0cv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_s, ...)
v1ao_co += np.sqrt(0.5) * v_cocv0
v1ao_cv -= np.sqrt((s+1)/s) * v_cvcv0
v1ao_ov -= np.sqrt(0.5) * v_ovcv0
v1ao_cv0 += np.sqrt(0.5)*v_cv0co - np.sqrt((s+1)/s)*v_cv0cv - np.sqrt(0.5)*v_cv0ov + v_cv0cv0
```

### 3.4 `gen_vind_sc`: 完整 Sigma 向量 (PDF §3.6)

**代码对应：** `satda.py:384-501`

#### 3.4.1 Fock 矩阵分块

基于 $\mathbf{F}^0 = \mathbf{F}^{\alpha} - \mathbf{F}^z$，投影到各轨道子空间:

**代码对应：** `satda.py:413-432`
```python
fock0 = focka - fockz
fock_coco1 = orbos.T @ (fock0 - fockz) @ orbos      # o→o with F^0-F^z
fock_coco2 = orbcs.T @ (fock0 - fockz) @ orbcs       # c→c with F^0-F^z
fock_cocv  = orbos.T @ (fock0 - fockz) @ orbvs       # o→v with F^0-F^z
fock_cvcv1 = orbvs.T @ (fock0 - fockz/s) @ orbvs     # v→v with F^0-F^z/s
fock_cvcv2 = orbcs.T @ (fock0 + fockz/s) @ orbcs     # c→c with F^0+F^z/s
fock_cocv0 = orbos.T @ fockb @ orbvs                 # o→v with F^β (用于 cv0 道)
fock_cvov  = orbos.T @ (fock0 + fockz) @ orbcs       # o→c with F^0+F^z
fock_cvcv01 = 0.5 * orbvs.T @ (focka - fockb) @ orbvs  # (F^α - F^β)/2 在 v 空间
fock_cvcv02 = 0.5 * orbcs.T @ (focka - fockb) @ orbcs  # (F^α - F^β)/2 在 c 空间
fock_ovov1 = orbvs.T @ (fock0 + fockz) @ orbvs       # v→v with F^0+F^z
fock_ovov2 = orbos.T @ (fock0 + fockz) @ orbos       # o→o with F^0+F^z
fock_ovcv0 = orbcs.T @ focka @ orbos                 # c→o with F^α
fock_cv0cv01 = 0.5 * orbvs.T @ (focka + fockb) @ orbvs  # (F^α+F^β)/2 在 v 空间 (cv0 道对角)
fock_cv0cv02 = 0.5 * orbcs.T @ (focka + fockb) @ orbcs  # (F^α+F^β)/2 在 c 空间 (cv0 道对角)
fock_cooo   = orbos.T @ (fock0 - fockz) @ orbcs      # o→c with F^0-F^z
fock_cvoo   = orbvs.T @ fockz @ orbcs                # v→c with F^z
fock_ovoo   = orbvs.T @ (fock0 + fockz) @ orbos      # v→o with F^0+F^z
fock_cv0oo  = 0.5 * orbvs.T @ (focka + fockb) @ orbcs  # v→c with (F^α+F^β)/2
```

#### 3.4.2 对角元 (PDF §3.6)

对于 5 个激发道:

**代码对应：** `satda.py:435-440`
```python
hdiag_co = (fock_coco1.diagonal()[None,:] - fock_coco2.diagonal()[:,None]).ravel()
hdiag_cv = (fock_cvcv1.diagonal()[None,:] - fock_cvcv2.diagonal()[:,None]).ravel()
hdiag_oo = np.array([0.0])                             # oo: 标量, 对角元为 0
hdiag_ov = (fock_ovov1.diagonal()[None,:] - fock_ovov2.diagonal()[:,None]).ravel()
hdiag_cv0 = (fock_cv0cv01.diagonal()[None,:] - fock_cv0cv02.diagonal()[:,None]).ravel()
```

注意 $oo$ 块对角元为 0，因为该块没有 Fock 矩阵之差，仅靠 XC 响应。

#### 3.4.3 Vind 函数: Fock 项叠加 (PDF §3.6)

与 $\Delta S = -1$ 类似，将 AO 基的响应转为 MO 基后叠加 Fock 项。在此仅给出关键的系数差异。

##### co 块 Fock 项 (line 460-464)

$$
A_{cu, dv}^{\text{Fock}} = \delta_{cd} \, F_{vu}^{F^0-F^z} - \delta_{uv} \, F_{cd}^{F^0-F^z}
$$

耦合项 (含从 $cv$、$oo$、$cv0$ 道的贡献):

$$
A_{cu, da}^{\text{coupling}} = \sqrt{\frac{S+1}{2S}} \, \delta_{cd} \, F_{ua}^{F^0-F^z} \times X_{da}
$$

$$
A_{cu, oo}^{\text{coupling}} = -F_{uc}^{F^0-F^z} \times X_{oo}
$$

$$
A_{cu, da}^{(cv0)\text{coupling}} = \frac{1}{\sqrt{2}} \, \delta_{cd} \, F_{ua}^{F^{\beta}} \times X_{da}
$$

**代码对应：** `satda.py:460-464`
```python
v1mo_co += lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco1, zs_co)
v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco2, zs_co)
v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv, zs_cv) * np.sqrt((s+1)/2/s)
v1mo_co -= np.einsum('ui,xv->xiu', fock_cooo, zs_oo)
v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv0, zs_cv0) * np.sqrt(0.5)
```

##### cv 块 Fock 项 (line 466-472)

$$
A_{ca, db}^{\text{Fock}} = \delta_{cd} \, F_{ab}^{F^0-F^z/s} - \delta_{ab} \, F_{cd}^{F^0+F^z/s}
$$

耦合项 (来自 co, oo, ov, cv0):

$$
A_{ca, du}^{\text{coupling}} = \sqrt{\frac{S+1}{2S}} \, \delta_{cd} \, (F_{ua}^{F^0-F^z})^{\dagger} \times X_{du}
$$

$$
A_{ca, oo}^{\text{coupling}} = \sqrt{\frac{2(S+1)}{S}} \, F_{ac}^{F^z} \times X_{oo}
$$

$$
A_{ca, vb}^{\text{coupling}} = -\sqrt{\frac{S+1}{2S}} \, \delta_{ab} \, F_{cv}^{F^0+F^z} \times X_{vb}
$$

$$
A_{ca, db}^{(cv0)\text{coupling}} = -\sqrt{\frac{S+1}{S}} \, \delta_{cd} \, F_{ab}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
+ \sqrt{\frac{S+1}{S}} \, \delta_{ab} \, F_{cd}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
$$

**代码对应：** `satda.py:466-472`
```python
v1mo_cv += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv.T, zs_co) * np.sqrt((s+1)/2/s)
v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv1, zs_cv)
v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv2, zs_cv)
v1mo_cv += np.einsum('ai,xv->xia', fock_cvoo, zs_oo) * np.sqrt(2*(s+1)/s)
v1mo_cv -= lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_cvov, zs_ov) * np.sqrt((s+1)/2/s)
v1mo_cv -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv01, zs_cv0) * np.sqrt((s+1)/s)
v1mo_cv += lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv02, zs_cv0) * np.sqrt((s+1)/s)
```

##### oo 块 Fock 项 (line 488-492)

$oo$ 仅有与 co、cv、ov、cv0 道的耦合项:

$$
A_{oo, cu} = -F_{cu}^{F^0-F^z} \times X_{cu}
$$

$$
A_{oo, ca} = \sqrt{\frac{2(S+1)}{S}} \, F_{ca}^{F^z} \times X_{ca}
$$

$$
A_{oo, ua} = F_{ua}^{F^0+F^z} \times X_{ua}
$$

$$
A_{oo, ca}^{(cv0)} = -\sqrt{2} \, F_{ca}^{(F^{\alpha}+F^{\beta})/2} \times X_{ca}
$$

**代码对应：** `satda.py:488-492`
```python
v1mo_oo = np.zeros((len(zs),))
v1mo_oo -= lib.einsum('jv,xjv->x', fock_cooo.T, zs_co)
v1mo_oo += lib.einsum('jb,xjb->x', fock_cvoo.T, zs_cv) * np.sqrt(2*(s+1)/s)
v1mo_oo += lib.einsum('vb,xvb->x', fock_ovoo.T, zs_ov)
v1mo_oo -= lib.einsum('jb,xjb->x', fock_cv0oo.T, zs_cv0) * np.sqrt(2)
```

##### ov 块 Fock 项 (line 474-478)

$$
A_{ua, vb}^{\text{Fock}} = \delta_{uv} \, F_{ab}^{F^0+F^z} - \delta_{ab} \, F_{vu}^{F^0+F^z}
$$

耦合项:

$$
A_{ua, cb} = -\sqrt{\frac{S+1}{2S}} \, \delta_{ab} \, (F_{cu}^{F^0+F^z})^{\dagger} \times X_{cb}
$$

$$
A_{ua, oo} = F_{ua}^{F^0+F^z} \times X_{oo}
$$

$$
A_{ua, cb}^{(cv0)} = \frac{1}{\sqrt{2}} \, \delta_{ab} \, F_{cu}^{F^{\alpha}} \times X_{cb}
$$

**代码对应：** `satda.py:474-478`
```python
v1mo_ov -= lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_cvov.T, zs_cv) * np.sqrt((s+1)/2/s)
v1mo_ov += np.einsum('au,xv->xua', fock_ovoo, zs_oo)
v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov1, zs_ov)
v1mo_ov -= lib.einsum('ab,vu,xvb->xua', np.eye(nvs), fock_ovov2, zs_ov)
v1mo_ov += lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_ovcv0, zs_cv0) * np.sqrt(0.5)
```

##### cv0 块 Fock 项 (line 480-486)

cv0 道没有纯 Fock 对角项（已在 $h_{\text{diag}}$ 和 XC 响应中处理），仅有与其他道的耦合:

$$
A_{ca, du}^{(cv0)} = \frac{1}{\sqrt{2}} \, \delta_{cd} \, (F_{ua}^{F^{\beta}})^{\dagger} \times X_{du}
$$

$$
A_{ca, db}^{(cv0)} = -\sqrt{\frac{S+1}{S}} \, \delta_{cd} \, F_{ab}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
$$

$$
A_{ca, db}^{(cv0)} = \sqrt{\frac{S+1}{S}} \, \delta_{ab} \, F_{cd}^{(F^{\alpha}-F^{\beta})/2} \times X_{db}
$$

$$
A_{ca, oo}^{(cv0)} = -\sqrt{2} \, F_{ca}^{(F^{\alpha}+F^{\beta})/2} \times X_{oo}
$$

$$
A_{ca, vb}^{(cv0)} = \frac{1}{\sqrt{2}} \, \delta_{ab} \, (F_{vc}^{F^{\alpha}})^{\dagger} \times X_{vb}
$$

$$
A_{ca, db}^{\text{Fock}} = \delta_{cd} \, F_{ab}^{(F^{\alpha}+F^{\beta})/2} - \delta_{ab} \, F_{cd}^{(F^{\alpha}+F^{\beta})/2}
$$

**代码对应：** `satda.py:480-486`
```python
v1mo_cv0 += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv0.T, zs_co) * np.sqrt(0.5)
v1mo_cv0 -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv01, zs_cv) * np.sqrt((s+1)/s)
v1mo_cv0 += lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv02, zs_cv) * np.sqrt((s+1)/s)
v1mo_cv0 -= np.einsum('ai,xv->xia', fock_cv0oo, zs_oo) * np.sqrt(2)
v1mo_cv0 += lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_ovcv0.T, zs_ov) * np.sqrt(0.5)
v1mo_cv0 += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cv0cv01, zs_cv0)
v1mo_cv0 -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cv0cv02, zs_cv0)
```

---

## 4. Hybrid 泛函与 Range-Separated 修正

对于 hybrid 泛函，HF 交换系数 $c_{\text{hyb}}$ 和 range-separation 参数 ($\omega$, $\alpha$) 分别处理:

$$
\mathbf{K}_{\text{total}} = c_{\text{hyb}} \, \mathbf{K}_{\text{Coulomb}} + (\alpha - c_{\text{hyb}}) \, \mathbf{K}_{\omega}^{\text{SR}}
$$

其中 $\mathbf{K}_{\omega}^{\text{SR}}$ 是短程 exchange (erfc(ωr)/r 型).

**代码对应：** `satda.py:219-223`, `satda.py:347-351`
```python
vk = mf.get_k(mol, dms, hermi) * hyb
if omega != 0:
    vk += mf.get_k(mol, dms, hermi, omega=omega) * (alpha - hyb)
```

---

## 5. GGA 和 Meta-GGA 响应函数

### 5.1 nr_rks_fxc1_gga — GGA 梯度校正 XC 核响应 (PDF Appendix A)

**代码对应：** `satda.py:33-85`

对于 GGA，XC 核 $f_{xc}(r, r')$ 不仅依赖于密度，还依赖于密度梯度。响应函数需要处理:

$$
V_{\mu\nu} = \int \frac{\delta^2 E_{xc}}{\delta \rho(r) \delta \rho(r')} \, \phi_{\mu}(r) \phi_{\nu}(r) \, \delta\rho(r') \, dr \, dr'
$$

作为 $\delta\rho$ 的函数，$V_{\mu\nu}$ 由局部项和梯度项组成:

$$
V_{\mu\nu} = U_{00} \phi_{\mu} \phi_{\nu} \rho_0 + \sum_{i=1}^{3} U_{i0} \phi_{\mu} \phi_{\nu} \nabla_i \rho_0 + \sum_{i=1}^{3} U_{0i} \rho_0 (\nabla_i \phi_{\mu} \phi_{\nu}) + \sum_{i,j=1}^{3} U_{ij} \nabla_i(\phi_{\mu} \phi_{\nu}) \nabla_j \rho_0
$$

其中 $U_{ab}$ 矩阵 (4×4) 组合了 $f_{xc}$ 的各分量和密度梯度:

**代码对应：** `satda.py:69-78` (GGA U matrix)
```python
U = np.zeros((4, 4, weight.size))
U[0, 0] = _fxc[0, 0] * rho0
U[0, 0] += lib.einsum('ig,ig->g', _fxc[1:4, 0], L_grad)    # Σ_i f_{i0} L_i
U[0, 0] += lib.einsum('jg,jg->g', _fxc[0, 1:4], R_grad)    # Σ_j f_{0j} R_j
U[0, 0] += lib.einsum('ijg,ijg->g', _fxc[1:4, 1:4], tau)   # Σ_ij f_{ij} τ_ij
U[1:4, 0] = _fxc[1:4, 0] * rho0
U[1:4, 0] += lib.einsum('ijg,jg->ig', _fxc[1:4, 1:4], R_grad)
U[0, 1:4] = _fxc[0, 1:4] * rho0
U[0, 1:4] += lib.einsum('ijg,ig->jg', _fxc[1:4, 1:4], L_grad)
U[1:4, 1:4] = _fxc[1:4, 1:4] * rho0
```

其中:
- $\rho_0$ = `rho0` — 扰动密度的 0 阶值
- $L_{\text{grad}}$ = 左侧梯度 (将 DM 的梯度作用到 AO 上)
- $R_{\text{grad}}$ = 右侧梯度 (将 AO 的梯度作用到 DM 上)
- $\tau$ = 密度 Hessian 矩阵 (3×3 梯度积)

最终 AO 基的响应矩阵通过稀疏矩阵乘法计算 (line 80-84):

$$
V_{\mu\nu}^{(k)} = \sum_{i=0}^{3} \phi_{\mu}^{(i)} \, U_{i,*} \, \phi_{\nu}^{(*)}
$$

其中 $\phi^{(0)} = \phi$, $\phi^{(1,2,3)} = \nabla \phi$.

### 5.2 nr_rks_fxc1_mgga — Meta-GGA 动能密度响应 (PDF Appendix B)

**代码对应：** `satda.py:88-147`

Meta-GGA 额外依赖动能密度 $\tau(r) = \frac{1}{2} \sum_i |\nabla \phi_i|^2$。$f_{xc}$ 核扩展为 5×5 矩阵（指标 0-3 为密度和梯度, 指标 4 为 $\tau$）.

U 矩阵扩展了含 $\tau$ 的项 (line 128-139):

$$
U_{i0} += \frac{1}{2} f_{i,4} \, L_{\text{grad}}
$$

$$
U_{0i} += \frac{1}{2} f_{4,i} \, R_{\text{grad}}
$$

$$
U_{ij} += \frac{1}{2} f_{i,4} \, R_{\text{grad}} + \frac{1}{2} f_{4,i} \, L_{\text{grad}} + \frac{1}{4} f_{4,4} \, \tau
$$

**代码对应：** `satda.py:128-139`
```python
U[1:4, 0] += 0.5 * _fxc[4, 0].reshape(1, -1) * L_grad
U[1:4, 0] += 0.5 * lib.einsum('jg,ijg->ig', _fxc[4, 1:4], tau)
U[0, 1:4] += 0.5 * _fxc[0, 4].reshape(1, -1) * R_grad
U[0, 1:4] += 0.5 * lib.einsum('ig,ijg->jg', _fxc[1:4, 4], tau)
U[1:4, 1:4] += 0.5 * lib.einsum('ig,jg->ijg', _fxc[1:4, 4], R_grad)
U[1:4, 1:4] += 0.5 * lib.einsum('jg,ig->ijg', _fxc[4, 1:4], L_grad)
U[1:4, 1:4] += 0.25 * _fxc[4, 4].reshape(1, 1, -1) * tau
```

---

## 6. TDA 对角化与迭代算法

### 6.1 Davidson 迭代

**代码对应：** `satda.py:668-680`

$$
\mathbf{X}^{(k+1)} = \text{lr\_eigh}(\text{vind}, \mathbf{X}^{(k)}, \text{precond}, \dots)
$$

使用 PySCF 的 `lr_eigh` (Davidson/Lanczos 型) 对 sigma 向量函数 `vind` 进行迭代对角化。预处理矩阵用 `hdiag`（即 A 矩阵的对角近似）。

### 6.2 初值猜测 (SATDA.init_guess)

**代码对应：** `satda.py:629-636`

取 `hdiag` 最小的几个对角元对应的单位向量作为初值：

```python
n_init = min(nstates + 3, hdiag.size)
idx = np.argsort(hdiag)[:n_init]
x0 = np.zeros((n_init, hdiag.size))
x0[np.arange(n_init), idx] = 1.0
```

### 6.3 本征值分类与 XY 向量

**代码对应：** `satda.py:682-686`

- $\Delta S = 0$: 直接保留为向量 `xy = [(xi, 0) for xi in x1]`
- $\Delta S = -1$: 解为 `xy = [(xi.reshape(nocca, nvirb), 0) for xi in x1]`

reshape 为 $(N_{occ}^{\alpha}, N_{vir}^{\beta})$ ($N_{occ}^{\alpha} = N_{cs} + N_{os}$, $N_{vir}^{\beta} = N_{os} + N_{vs}$) 的矩阵形式，对应激发算符 $\hat{a}_a^{\dagger} \hat{a}_i$ 的系数矩阵。

---

## 7. 公式 → 代码 关键对照总结

| PDF 公式 / 定义 | 代码位置 | 说明 |
|---|---|---|
| $S$ 定义 | `s = (mol.nelec[0] - mol.nelec[1]) * 0.5` | 总自旋量子数 |
| 轨道划分 (cs, os, vs) | `csidx/osidx/vsidx` (line 394-399) | 按 mo_occ 划分 |
| $f_{xc}^{\text{ref}}$ | `fxc_ref` (line 170-171) | ROKS 密度 XC 核 |
| $f_{xc}^{s}$, $f_{xc}^{cv0,cv}$, ... | line 175-181 | UKS 核分量 (仅 ΔS=0) |
| $\mathbf{A}^{\text{Resp}}$ (ΔS=-1) | `gen_rohf_response_sf` (line 284-382) | XC + HF 响应函数 |
| $\mathbf{A}^{\text{Resp}}$ (ΔS=0) | `gen_rohf_response_sc` (line 149-282) | + Coulomb + 多核 XC |
| $\mathbf{F}^z$ | `fockz = 0.5*delta` (line 282, 382) | 开壳密度响应 |
| $\mathbf{A}^{\text{Fock}}$ (ΔS=-1) | `gen_vind_sf` (line 503-617) | 自旋适配 Fock 项 |
| $\mathbf{A}^{\text{Fock}}$ (ΔS=0) | `gen_vind_sc` (line 384-501) | 含 cv0 道 |
| GGA U 矩阵 | `nr_rks_fxc1_gga` (line 33-85) | 梯度校正 XC 响应 |
| MGGA U 矩阵 | `nr_rks_fxc1_mgga` (line 88-147) | 动能密度 XC 响应 |
| TDA 对角化 | `SATDA.kernel()` (line 638-693) | lr_eigh Davidson 对角化 |

---

## 附录: 自旋适配系数的物理来源

自旋适配系数来自 Wigner-Eckart 定理。两个自旋态之间的矩阵元可通过约化矩阵元和一个几何因子因式分解:

$$
\langle S', M' | \hat{O} | S, M \rangle = C_{SM, 1q}^{S'M'} \, \langle S' || \hat{O} || S \rangle
$$

其中 $C_{SM, 1q}^{S'M'}$ 是 Clebsch-Gordan 系数。

对于自旋翻转激发 $S_f = S_i - 1$:

| 跃迁类型 | 物理过程 | CG 系数 | 代码中的体现 |
|---|---|---|---|
| co (c→u) | 翻转一个闭壳电子的自旋 | $\frac{1}{\sqrt{2S}}$ | $\sqrt{\frac{2S+1}{2S}}$ 的配对系数 |
| cv (c→a) | 闭壳→虚轨道的自旋翻转 | $\frac{1}{\sqrt{2S(S+1)}}$ | $1/S$ 的 Fock 能差修正 |
| oo (u→v) | 开壳→开壳的保持自旋 | $\sqrt{\frac{2S}{2S-1}}$ | 自旋保持项 |
| ov (u→a) | 开壳→虚轨道的自旋翻转 | $\sqrt{\frac{2S}{2S+1}}$ | $\frac{2S}{2S-1}$ 的主系数 |

对于自旋守恒激发 $S_f = S_i = S$:

| 跃迁类型 | 物理过程 | CG 系数 | 代码中的体现 |
|---|---|---|---|
| co (c→u) | 保持自旋的激发 | $\frac{1}{\sqrt{2S+2}}$ | $\sqrt{\frac{S+1}{2S}}$ |
| cv0 (特殊道) | 纯 Coulomb 耦合 | $\sqrt{2}$ | co↔cv0, ov↔cv0 的 $\sqrt{2}$ 系数 |

这些系数确保激发态是自旋算符 $\hat{S}^2$ 和 $\hat{S}_z$ 的本征态。

---

*推导完成。代码对应: `pyscf-forge/pyscf/sftda/satda.py` (SATDA 类及其响应函数)。*

## 推导：SATDA deltaS=-1 HF 解析梯度 — HF 等价分解 — 2026-05-23

**目标：** 从 SATDA 的 ROKS-native Fock 构造推导其在 HF 极限下的等价 Rayleigh 商分解，并建立 `tdsatda.py` 解析梯度的理论入口。

**假设：**
- 只处理 HF 或 `xc='HF'`，没有 XC kernel 的核导数项。
- $\Delta S=-1$，振幅块为 $X_{co}, X_{cv}, X_{oo}, X_{ov}$。
- 轨道实数，TDA 中 $Y=0$。
- 使用空间轨道 ROKS/ROHF 参考态。

**符号声明：**
- $i,j \in C$（闭壳轨道），$u,v,w \in O$（开壳轨道），$a,b \in V$（虚轨道）。
- $S = (n_{os})/2$ 为参考态总自旋。
- $\mathbf{F}_{\mathrm{z}}$ 是 ROKS/ROHF 的耦合 Fock 矩阵（`fockz`），定义见 Li 等 ROKS 梯度论文。
- $\mathbf{F}^{0} = \mathbf{F}^{\alpha} - \mathbf{F}_{\mathrm{z}}$ 是去耦合后的有效 Fock 矩阵。
- $\mathbf{A}_{\mathrm{SATDA}}^{\mathrm{HF}}$ 是 HF 下 SATDA 的 ROKS-native A 矩阵。
- $\mathbf{A}_{\mathrm{SF}}^{\mathrm{HF}}$ 是普通 SF-TDA 在 HF 下的 A 矩阵。

### Step 1: ROKS-native Fock 矩阵的分块定义

`gen_vind_sf()`（`satda.py` L503-617）使用 ROKS-native 的 Fock 分块构造 sigma 向量。关键 Fock 矩阵（L530-552）为：

$$
\begin{aligned}
\mathbf{F}_{\mathrm{coco}}^{0} &= \mathbf{C}_{o}^{T}(\mathbf{F}^{0} - \mathbf{F}_{\mathrm{z}})\mathbf{C}_{o}, &
\mathbf{F}_{\mathrm{coco}}^{1} &= \mathbf{C}_{c}^{T}(\mathbf{F}^{0} + \mathbf{F}_{\mathrm{z}})\mathbf{C}_{c}, &
\mathbf{F}_{\mathrm{coco}}^{2} &= \mathbf{C}_{c}^{T}\mathbf{F}_{\mathrm{z}}\mathbf{C}_{c}, \\
\mathbf{F}_{\mathrm{cvcv}}^{0} &= \mathbf{C}_{v}^{T}(\mathbf{F}^{0} - \mathbf{F}_{\mathrm{z}})\mathbf{C}_{v}, &
\mathbf{F}_{\mathrm{cvcv}}^{2} &= \mathbf{C}_{v}^{T}\mathbf{F}_{\mathrm{z}}\mathbf{C}_{v}, &
\mathbf{F}_{\mathrm{cvoo}} &= \mathbf{C}_{v}^{T}\mathbf{F}_{\mathrm{z}}\mathbf{C}_{c}, \\
\mathbf{F}_{\mathrm{cocv}} &= \mathbf{C}_{o}^{T}(\mathbf{F}^{0} - \mathbf{F}_{\mathrm{z}})\mathbf{C}_{v}, &
\mathbf{F}_{\mathrm{cooo}}^{0} &= \mathbf{C}_{o}^{T}(\mathbf{F}^{0} + \mathbf{F}_{\mathrm{z}})\mathbf{C}_{c}, &
\mathbf{F}_{\mathrm{cooo}}^{1} &= \mathbf{C}_{o}^{T}(\mathbf{F}^{0} - \mathbf{F}_{\mathrm{z}})\mathbf{C}_{c}.
\end{aligned}
$$

→ **解释：** 这些是 `gen_vind_sf` 中构造 Fock 项 sigma 的全部 MO 子块。它们完全由 ROKS 的 $\mathbf{F}_{\mathrm{z}}$ 和 $\mathbf{F}^{0}$ 决定。

### Step 2: HF 极限下的等价分解

在 HF 极限下（无 XC kernel），SATDA 的 ROKS-native sigma 向量与普通 SF-TDA + 自旋适配修正给出相同的 Rayleigh quotient。即对任意 SATDA 本征矢 $\mathbf{X}$：

$$
\boxed{
\omega_{\mathrm{SATDA}}^{\mathrm{HF}}
= \mathbf{X}^{T}\mathbf{A}_{\mathrm{SF}}^{\mathrm{HF}}\mathbf{X}
+ \mathbf{X}^{T}\Delta\mathbf{A}_{\mathrm{SA}}^{\mathrm{HF}}\mathbf{X}.
}
$$

→ **解释：** 第一项是普通 SF-TDA 的 HF A 矩阵（Fock 差 + exact exchange），第二项是自旋适配修正（Fock-like + HF exchange-like）。这个等价性是代码使用 SF-base 分解路径的理论依据，由 `_satda_hf_energy_for_orbs()` 在每次梯度计算前验证。

自旋适配修正按 Fock-like 和 HF exchange-like 拆分为：

$$
\Delta\mathbf{A}_{\mathrm{SA}}^{\mathrm{HF}}
= \Delta\mathbf{A}^{F} + \Delta\mathbf{A}^{\mathrm{eri}}.
$$

→ **解释：** 纯 HF 下没有 XC kernel 修正，只有这两类。代码中 `satda_delta_fock_q()` 和 `satda_delta_hf_exchange_q()` 分别处理。

**代码对应：** `pyscf-forge/pyscf/sftda/satda.py:gen_vind_sf()` L503-617；`pyscf-forge/pyscf/grad/tdsatda.py:_satda_hf_energy_for_orbs()` L549-659。


## 推导：Fock-like 项的 coefficient matrices 构造 — 2026-05-23

**目标：** 从振幅 $\mathbf{X}$ 的分块和自旋适配系数，构造 7 类 coefficient matrix $T^{s}_{CC}, T^{s}_{VV}, T^{s}_{CV}, T^{\beta}_{VO}, T^{\beta}_{CO}, T^{\alpha}_{OC}, T^{\alpha}_{VO}$。这些矩阵是 $\mathbf{X}^{T}\Delta\mathbf{A}^{F}\mathbf{X}$ 对 Fock 矩阵元的缩合系数。

**符号声明：**
- $\mathbf{X}_{co} \in \mathbb{R}^{n_{cs} \times n_{os}}$，$\mathbf{X}_{cv} \in \mathbb{R}^{n_{cs} \times n_{vs}}$，
  $\mathbf{X}_{oo} \in \mathbb{R}^{n_{os} \times n_{os}}$，$\mathbf{X}_{ov} \in \mathbb{R}^{n_{os} \times n_{vs}}$。
- $\mathrm{tr}_{oo} = \sum_{u} X_{uu}^{oo}$ 为 OO 块的迹。
- $\mathbf{F}^{s} = \frac{1}{2}(\mathbf{F}^{\beta} - \mathbf{F}^{\alpha})$。

**自旋适配系数：**

$$
\eta = \sqrt{\frac{2S+1}{2S}} - 1,\qquad
\gamma = \sqrt{\frac{2S+1}{2S-1}},\qquad
\zeta = \sqrt{\frac{2S}{2S-1}} - 1,\qquad
\chi = \frac{1}{\sqrt{2S(2S-1)}}.
$$

→ **解释：** 这 4 个系数与 SASF 的 `tdsasf.py` 完全相同。它们在 `satda_fock_coefficients()`（L137-182）中计算。

### Step 1: $F^{s}_{CC}$ 的 coefficient — $T^{s}_{CC}$

来自 $CV,CV$ 和 $CO,CO$ 两个对角块对 $F^{s}_{CC}$ 的缩并：

$$
T^{s}_{CC,ji}
= \frac{1}{S}\sum_{a} X^{CV}_{ia} X^{CV}_{ja}
+ \frac{2}{2S-1}\sum_{u} X^{CO}_{iu} X^{CO}_{ju}.
$$

→ **解释：** 第一项来自 $CVCV$ 的 $F^{s}_{ji}$ 项（系数 $1/S$），第二项来自 $COCO$ 的 $F^{s}_{ji}$ 项（系数 $2/(2S-1)$）。

### Step 2: $F^{s}_{VV}$ 的 coefficient — $T^{s}_{VV}$

来自 $CV,CV$ 和 $OV,OV$ 两个对角块：

$$
T^{s}_{VV,ab}
= \frac{1}{S}\sum_{i} X^{CV}_{ia} X^{CV}_{ib}
+ \frac{2}{2S-1}\sum_{u} X^{OV}_{ua} X^{OV}_{ub}.
$$

### Step 3: $F^{s}_{CV}$ 的 coefficient — $T^{s}_{CV}$

仅来自非对角块 $CV,OO$ 及转置块 $OO,CV$：

$$
T^{s}_{CV,ia}
= \frac{\gamma}{S}\left(1 + \frac{1}{S}\right) \mathrm{tr}_{oo} \, X^{CV}_{ia}.
$$

→ **解释：** 这里的系数 $\gamma(1+1/S)$ 来自 `TDA_SASF.gen_vind` 的非对称 CV/OO Fock 耦合约定，与 code 中 `t_s_cv` 一致。因子 2 来自 $CV,OO$ 与转置块的合并。

### Step 4: $F^{\beta}_{VO}$ 的 coefficient — $T^{\beta}_{VO}$

来自 $CV,CO$ 和 $OV,OO$ 块：

$$
T^{\beta}_{VO,av}
= 2\eta \sum_{i} X^{CV}_{ia} X^{CO}_{iv}
+ 2\zeta \sum_{u} X^{OV}_{ua} X^{OO}_{uv}.
$$

### Step 5: $F^{\beta}_{CO}$ 的 coefficient — $T^{\beta}_{CO}$

仅来自 $CO,OO$ 块：

$$
T^{\beta}_{CO,iu}
= 2\chi \, \mathrm{tr}_{oo} \, X^{CO}_{iu}.
$$

### Step 6: $F^{\alpha}_{OC}$ 的 coefficient — $T^{\alpha}_{OC}$

来自 $CV,OV$ 和 $CO,OO$ 块：

$$
T^{\alpha}_{OC,vi}
= -2\eta \sum_{a} X^{CV}_{ia} X^{OV}_{va}
- 2\zeta \sum_{u} X^{CO}_{iu} X^{OO}_{vu}.
$$

### Step 7: $F^{\alpha}_{VO}$ 的 coefficient — $T^{\alpha}_{VO}$

仅来自 $OV,OO$ 块：

$$
T^{\alpha}_{VO,au}
= -2\chi \, \mathrm{tr}_{oo} \, X^{OV}_{ua}.
$$

### Step 8: AO probe density 构造

将 7 类 $T$ 转换为 AO probe density，用于后续的 Fock response 和 skeleton derivative：

$$
\begin{aligned}
\mathbf{P}_{s} &= \mathbf{C}_{c} \mathbf{T}^{s}_{CC} \mathbf{C}_{c}^{T}
+ \mathbf{C}_{v} \mathbf{T}^{s}_{VV} \mathbf{C}_{v}^{T}
+ \mathbf{C}_{c} \mathbf{T}^{s}_{CV} \mathbf{C}_{v}^{T}, \\
\mathbf{P}_{\beta} &= \mathbf{C}_{v} \mathbf{T}^{\beta}_{VO} \mathbf{C}_{o}^{T}
+ \mathbf{C}_{c} \mathbf{T}^{\beta}_{CO} \mathbf{C}_{o}^{T}, \\
\mathbf{P}_{\alpha} &= \mathbf{C}_{o} \mathbf{T}^{\alpha}_{OC} \mathbf{C}_{c}^{T}
+ \mathbf{C}_{v} \mathbf{T}^{\alpha}_{VO} \mathbf{C}_{o}^{T}.
\end{aligned}
$$

由于 $F^{s} = \frac{1}{2}(F^{\beta} - F^{\alpha})$，最终自旋分辨的 probe density 为：

$$
\boxed{
\mathbf{P}^{\alpha}_{\mathrm{probe}} = \mathbf{P}_{\alpha} - \frac{1}{2}\mathbf{P}_{s},
\qquad
\mathbf{P}^{\beta}_{\mathrm{probe}} = \mathbf{P}_{\beta} + \frac{1}{2}\mathbf{P}_{s}.
}
$$

→ **解释：** 这使标量收缩满足 $\mathrm{Tr}[\mathbf{P}_{\alpha}^{\mathrm{probe}} \mathbf{F}^{\alpha}] + \mathrm{Tr}[\mathbf{P}_{\beta}^{\mathrm{probe}} \mathbf{F}^{\beta}] = \sum T^{s} F^{s} + \sum T^{\beta} F^{\beta} + \sum T^{\alpha} F^{\alpha}$。

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:satda_fock_coefficients()` L137-182；`satda_fock_probe_densities()` L185-203。


## 推导：Fock-like RHS 的 projection + response 分解 — 2026-05-23

**目标：** 从 7 类 coefficient matrix 构造未约束 MO 系数导数 $\mathbf{Q}^{\alpha}$、$\mathbf{Q}^{\beta}$，并分解为 MO projection 项和 Fock-density response 项。

**符号声明：**
- $\mathbf{Q}^{\sigma}_{rs} = C_{\lambda r} \, \partial E^{\Delta,F} / \partial C_{\lambda s}$ 是激发能中 Fock-like 修正对 MO 系数 $C_{\lambda s}$ 的缩合导数。
- $\mathbf{F}^{\sigma}_{\mathrm{MO}} = \mathbf{C}^{T} \mathbf{F}^{\sigma}_{\mathrm{AO}} \mathbf{C}$ 是全 MO 空间的 Fock 矩阵。
- $\theta^{\tau}_{s}$ 表示空间轨道 $s$ 是否属于自旋 $\tau$ 的占据空间：$\theta^{\alpha}_{s}=1$ 若 $s \in C \cup O$，$\theta^{\beta}_{s}=1$ 若 $s \in C$。

### Step 1: 通用单自旋 contraction 的 Q 公式

对任意单自旋 coefficient matrix $\mathbf{T}^{\sigma}_{pq}$ 和一电子收缩 $E = \sum_{pq} T^{\sigma}_{pq} F^{\sigma}_{pq}$，其对 MO 系数的缩合导数为：

$$
\boxed{
Q^{(T,\sigma)}_{rs}
= \sum_{q} T^{\sigma}_{sq} F^{\sigma,\mathrm{MO}}_{rq}
+ \sum_{p} T^{\sigma}_{ps} F^{\sigma,\mathrm{MO}}_{pr}
+ \sum_{\tau\in\{\alpha,\beta\}} \theta^{\tau}_{s}
\sum_{pq} T^{\sigma}_{pq} K^{S,\sigma\tau}_{pq,rs}.
}
$$

→ **解释：** 前两项是 projection 贡献（来自 $C_{\mu p}$ 和 $C_{\nu q}$ 的显式导数），第三项是 Fock response 贡献（来自 AO Fock 对密度矩阵的依赖）。$K^{S,\sigma\tau}_{pq,rs} = K^{\sigma\tau}_{pq,rs} + K^{\sigma\tau}_{pq,sr}$ 是对称化的 MO 表象 Fock kernel。

### Step 2: $F^{s}$ contraction 的 Q 公式

对 $F^{s} = \frac{1}{2}(F^{\beta} - F^{\alpha})$ 的收缩，用线性组合得到：

$$
Q^{(T,s)}_{rs}
= \sum_{q} T^{s}_{sq} F^{s,\mathrm{MO}}_{rq}
+ \sum_{p} T^{s}_{ps} F^{s,\mathrm{MO}}_{pr}
+ \frac{1}{2}\sum_{\tau} \theta^{\tau}_{s}
\sum_{pq} T^{s}_{pq}
\left(K^{S,\beta\tau}_{pq,rs} - K^{S,\alpha\tau}_{pq,rs}\right).
$$

→ **解释：** projection 项使用 $F^{s,\mathrm{MO}} = \frac{1}{2}(F^{\beta,\mathrm{MO}} - F^{\alpha,\mathrm{MO}})$。response 项中 $F^{s}$ 对 $\alpha$ 和 $\beta$ 密度的依赖符号相反。

### Step 3: 按 $s$ 所在轨道空间分类

根据 $s$ 的占据情况，response 项简化为：

$$
\begin{aligned}
s \in C &: \theta^{\alpha}_{s}=1,\; \theta^{\beta}_{s}=1, \\
s \in O &: \theta^{\alpha}_{s}=1,\; \theta^{\beta}_{s}=0, \\
s \in V &: \theta^{\alpha}_{s}=0,\; \theta^{\beta}_{s}=0.
\end{aligned}
$$

→ **解释：** virtual 轨道不在参考态密度中，因此没有 Fock response 贡献（$\theta^{s}_{V}=0$），只保留 projection 项。

### Step 4: 代码实现路径

代码中 `_add_fock_term()` 对每个 $T$ 矩阵同时累加 projection 和 probe density：

```python
# projection 部分：
q[:, left_idx] += (fock_mo[:, right_idx] @ coeff_mat.T) * scale
q[:, right_idx] += (fock_mo[:, left_idx] @ coeff_mat) * scale

# probe density 累积（供后续 response）：
p += c_left @ coeff_mat @ c_right.T
```

7 类 $T$ 按 spin 标签分别调用 `_add_fock_term`：

| $T$ 矩阵 | spin 标签 | 作用 |
|-----------|-----------|------|
| $T^{s}_{CC}$ | `'spin'` | $\frac{1}{2}$ 加到 $\beta$ 的 Q，$-\frac{1}{2}$ 加到 $\alpha$ 的 Q |
| $T^{s}_{VV}$ | `'spin'` | 同上 |
| $T^{s}_{CV}$ | `'spin'` | 同上 |
| $T^{\beta}_{VO}$ | `'beta'` | 直接加到 $\beta$ 的 Q |
| $T^{\beta}_{CO}$ | `'beta'` | 同上 |
| $T^{\alpha}_{OC}$ | `'alpha'` | 直接加到 $\alpha$ 的 Q |
| $T^{\alpha}_{VO}$ | `'alpha'` | 同上 |

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:satda_delta_fock_q()` L258-298；`_add_fock_term()` L206-236；`_add_fock_response_q()` L239-255。


## 推导：HF exchange-like $\Delta A$ 的 8 个独立块 — 2026-05-23

**目标：** 列出 HF exchange-like 修正 $\Delta\mathbf{A}^{\mathrm{eri}}$ 的 8 个非零独立块矩阵表达式。对应 `_satda_hf_exchange_energy_with_coeff()` 中每个 ERI 收缩。

**符号声明：**
- $(pq|tu)$ 表示化学记号的 MO 两电子积分。
- $c_{\mathrm{x}}$ 为 exact-exchange 系数（纯 HF 时为 1，hybrid 时为 `hyb`，RSH 时需要额外加 $k_{\omega} = \alpha - c_{\mathrm{x}}$ 的 range-separated 项）。

### Step 1: 对角块 $CO,CO$

$$
\Delta A^{\mathrm{eri}}_{CO,CO}(iu, jv)
= -\frac{1}{2S-1} \, (ui|jv).
$$

→ **解释：** 指标空间为 $O,C,C,O$。该积分有一个 $CO$ 振幅指标与外积对调。

### Step 2: 对角块 $OV,OV$

$$
\Delta A^{\mathrm{eri}}_{OV,OV}(ua, vb)
= -\frac{1}{2S-1} \, (au|vb).
$$

→ **解释：** 指标空间为 $V,O,O,V$。

### Step 3: 非对角块 $CV,CO$（含转置）

$$
\Delta A^{\mathrm{eri}}_{CV,CO}(ia, jv)
= -\eta \, (av|ji).
$$

→ **解释：** 指标空间为 $V,O,C,C$。转置块 $CO,CV$ 在二次型中贡献等量，故后续 RHS 和 skeleton 中出现因子 2。

### Step 4: 非对角块 $CV,OV$（含转置）

$$
\Delta A^{\mathrm{eri}}_{CV,OV}(ia, vb)
= -\eta \, (ab|vi).
$$

→ **解释：** 指标空间为 $V,V,O,C$。

### Step 5: 非对角块 $CO,OV$（含转置）— 两项

$$
\Delta A^{\mathrm{eri}}_{CO,OV}(iu, vb)
= \frac{1}{2S-1}(ui|vb)
- \frac{1}{2S-1}(ub|vi).
$$

→ **解释：** 第一项指标为 $O,C,O,V$，第二项为 $O,V,O,C$。两项符号相反。

### Step 6: 非对角块 $CV,OO$（含转置）

$$
\Delta A^{\mathrm{eri}}_{CV,OO}(ia, wv)
= -(\gamma - 1)(av|wi).
$$

→ **解释：** 指标空间为 $V,O,O,C$。注意系数为 $\gamma - 1$（不是 $\gamma/S$，与 Fock-like 同块不同）。

### Step 7: 非对角块 $CO,OO$（含转置）

$$
\Delta A^{\mathrm{eri}}_{CO,OO}(iu, wv)
= -\zeta \, (uv|wi).
$$

→ **解释：** 指标空间为 $O,O,O,C$。

### Step 8: 非对角块 $OV,OO$（含转置）

$$
\Delta A^{\mathrm{eri}}_{OV,OO}(ua, wv)
= -\zeta \, (av|wu).
$$

→ **解释：** 指标空间为 $V,O,O,O$。

### Step 9: 零块

$$
\Delta A^{\mathrm{eri}}_{CV,CV} = 0,\qquad
\Delta A^{\mathrm{eri}}_{OO,OO} = 0.
$$

### 最终结果

所有非零独立块汇总：

$$
\boxed{
\Delta\mathbf{A}^{\mathrm{eri}} = \Delta A^{\mathrm{eri}}_{CO,CO}
+ \Delta A^{\mathrm{eri}}_{OV,OV}
+ \Delta A^{\mathrm{eri}}_{CV,CO}
+ \Delta A^{\mathrm{eri}}_{CV,OV}
+ \Delta A^{\mathrm{eri}}_{CO,OV}
+ \Delta A^{\mathrm{eri}}_{CV,OO}
+ \Delta A^{\mathrm{eri}}_{CO,OO}
+ \Delta A^{\mathrm{eri}}_{OV,OO}.
}
$$

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:_satda_hf_exchange_energy_with_coeff()` L301-364。


## 推导：HF exchange-like RHS 的 4-projection 展开 — 2026-05-23

**目标：** 对每个 HF exchange-like 独立块，构造其 orbital RHS contribution。将 ERIs 对 MO 系数的缩合导数（4-projection 项）铺到 $\mathbf{Q}^{\alpha}$ 和 $\mathbf{Q}^{\beta}$。

**符号声明：**
- $\mathcal{I}_{pq|tu} = c_{\mathrm{x}}(pq|tu) + k_{\omega}(pq|tu)_{\omega}$ 包含 hybrid 和 RSH 系数。
- $\mathcal{D}_{pq|tu;rs} = C_{\lambda r} \, \partial \mathcal{I}_{pq|tu} / \partial C_{\lambda s}$。

### Step 1: ERI 的 4 项 projection 导数

对固定 MO 系数（暂不涉及 AO Fock response）的 ERI，缩合导数为：

$$
\boxed{
\mathcal{D}_{pq|tu;rs}
= \delta_{ps}\mathcal{I}_{rq|tu}
+ \delta_{qs}\mathcal{I}_{pr|tu}
+ \delta_{ts}\mathcal{I}_{pq|ru}
+ \delta_{us}\mathcal{I}_{pq|tr}.
}
$$

→ **解释：** 四项分别来自 ERI 的四个 MO 指标 $p,q,t,u$ 中每个对 $C_{\lambda s}$ 的导数。$r$ 是左乘的 MO 指标。

### Step 2: 代码化 — `_add_eri_term_q`

对于系数张量 $T_{pqtu}$ 和指标空间 $(P,Q,R,S)$（各自属于 $C,O,V$ 中某块），代码逐条处理四个 projection：

对 `pos=0`（替换第一指标 $p$）：
$$
Q_{rp} \leftarrow \mathrm{scale} \times \sum_{qtu} T_{pqtu} \, \mathcal{I}_{rq|tu}.
$$

其余三个位置类似。`spin_sets` 决定每个 projection contribution 加到 $\mathbf{Q}^{\alpha}$ 还是 $\mathbf{Q}^{\beta}$。

### Step 3: 各块的指标空间和 spin_sets

| 块 | ERI 指标空间 | spin_sets | 系数 |
|---|---|---|---|
| $CO,CO$ | $(O,C,C,O)$ | $(\beta,\alpha,\alpha,\beta)$ | $-\frac{1}{2S-1}$ |
| $OV,OV$ | $(V,O,O,V)$ | $(\beta,\beta,\beta,\beta)$ | $-\frac{1}{2S-1}$ |
| $CV,CO$ | $(V,O,C,C)$ | $(\beta,\beta,\alpha,\alpha)$ | $-2\eta$ |
| $CV,OV$ | $(V,V,O,C)$ | $(\beta,\beta,\beta,\alpha)$ | $-2\eta$ |
| $CO,OV$ (1) | $(O,C,O,V)$ | $(\beta,\alpha,\beta,\beta)$ | $+\frac{2}{2S-1}$ |
| $CO,OV$ (2) | $(O,V,O,C)$ | $(\beta,\beta,\beta,\alpha)$ | $-\frac{2}{2S-1}$ |
| $CV,OO$ | $(V,O,O,C)$ | $(\beta,\beta,\beta,\alpha)$ | $-2(\gamma-1)$ |
| $CO,OO$ | $(O,O,O,C)$ | $(\beta,\beta,\beta,\alpha)$ | $-2\zeta$ |
| $OV,OO$ | $(V,O,O,O)$ | $(\beta,\beta,\beta,\beta)$ | $-2\zeta$ |

→ **解释：** `spin_sets` 决定投影项的去向：`'alpha'` → $\mathbf{Q}^{\alpha}$，`'beta'` → $\mathbf{Q}^{\beta}$。注意各块的系数已含非对角与转置块合并的因子 2。

### Step 4: RHS 反对称化

得到 $\mathbf{Q}^{\alpha,\Delta,\mathrm{eri}}$ 和 $\mathbf{Q}^{\beta,\Delta,\mathrm{eri}}$ 后，取反对称部分：

$$
\mathbf{R}^{\sigma,\Delta,\mathrm{eri}} = \mathbf{Q}^{\sigma,\Delta,\mathrm{eri}} - \left(\mathbf{Q}^{\sigma,\Delta,\mathrm{eri}}\right)^{T}.
$$

再投影到 UCPHF 布局：

$$
W^{\sigma}_{ai} = R^{\sigma}_{ai}, \quad a \in \mathrm{vir}_{\sigma},\; i \in \mathrm{occ}_{\sigma}.
$$

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:_add_eri_term_q()` L380-413；`_satda_delta_hf_exchange_q_with_coeff()` L416-489；`satda_delta_hf_exchange_q()` L492-511。


## 推导：直接 AO 骨架导数 — Fock-like 项 — 2026-05-23

**目标：** 固定 MO 系数和 overlap metric 后，Fock-like skeleton derivative 的计算。将 7 类 $T$ 矩阵收缩的 $\mathbf{P}^{\alpha}_{\mathrm{probe}}$ 和 $\mathbf{P}^{\beta}_{\mathrm{probe}}$ 接入 AO 积分导数。

**上标约定：** $[x]$ 表示 skeleton/direct nuclear derivative — 只对 AO 积分显式求导，不含 MO response 和 overlap metric 项。

### Step 1: AO Fock 的 skeleton derivative

固定 MO 系数时：

$$
F^{\sigma,[x]}_{pq}
= \sum_{\mu\nu} C_{\mu p} F^{\sigma,[x]}_{\mu\nu} C_{\nu q}.
$$

HF 下 AO Fock skeleton derivative 展开为：

$$
F^{\sigma,[x]}_{\mu\nu}
= h^{[x]}_{\mu\nu}
+ \sum_{\lambda\kappa} D_{\lambda\kappa} (\mu\nu|\lambda\kappa)^{[x]}
- c_{\mathrm{x}} \sum_{\lambda\kappa} D^{\sigma}_{\lambda\kappa} (\mu\lambda|\nu\kappa)^{[x]},
$$

其中 $D = D^{\alpha} + D^{\beta}$ 是总密度矩阵。

### Step 2: 将 probe density 合并到 relaxed density

代码中，普通 SF 部分的 relaxed density 为 `dmz1dooa` 和 `dmz1doob`（在 Z-vector 解出后）。SATDA Fock-like probe density 线性叠加到其上：

```python
dmz1dooa_direct = dmz1dooa + 2 * dm_probe_a
dmz1doob_direct = dmz1doob + 2 * dm_probe_b
```

→ **解释：** 因子 2 来自 PySCF 的 `get_jk` 密度约定（`dmz1doo` 在收缩 `h1ao` 和 `veff1` 时已有特定的 Hermitian 对称处理）。合并后的密度传给 `td_grad.get_jk`，自动在 atom loop 中产生正确的 Coulomb + exchange skeleton 贡献。

### Step 3: 一电子项

SATDA Fock-like 修正对一电子积分导数的贡献通过 `as_dm1` 实现：

```python
as_dm1 = oo0a + oo0b + (dmz1dooa_direct + dmz1doob_direct) * 0.5
# atom loop:
de += einsum('xpq,pq->x', h1ao, as_dm1)
```

数学上：

$$
\boxed{
\Omega_{\mathrm{Fock},h}^{[x]}
= \sum_{\mu\nu} \left[
P^{\alpha,\mathrm{probe}}_{\mu\nu} + P^{\beta,\mathrm{probe}}_{\mu\nu}
\right] h^{[x]}_{\mu\nu}.
}
$$

### Step 4: Coulomb 和 exchange 项

通过 `td_grad.get_jk` 产生的 `veff1` 在 atom loop 中与 `dmz1dooa_direct`/`dmz1doob_direct` 收缩，等价于：

$$
\boxed{
\Omega_{\mathrm{Fock},J}^{[x]}
= \sum_{\mu\nu\lambda\kappa}
\left(P^{\alpha,\mathrm{probe}}_{\mu\nu} + P^{\beta,\mathrm{probe}}_{\mu\nu}\right)
D^{\mathrm{tot}}_{\lambda\kappa}
(\mu\nu|\lambda\kappa)^{[x]},
}
$$

$$
\boxed{
\Omega_{\mathrm{Fock},K}^{[x]}
= -c_{\mathrm{x}} \sum_{\mu\nu\lambda\kappa}
\left(
P^{\alpha,\mathrm{probe}}_{\mu\lambda} D^{\alpha}_{\nu\kappa}
+ P^{\beta,\mathrm{probe}}_{\mu\lambda} D^{\beta}_{\nu\kappa}
\right)
(\mu\nu|\lambda\kappa)^{[x]}.
}
$$

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:grad_elec_hf_experimental()` L997-1011（`dmz1dooa_direct`、`dmz1doob_direct`、`as_dm1` 和 `veff1` 构造），L1023-1043（atom loop 中的收缩）。


## 推导：直接 AO 骨架导数 — HF exchange-like 项 — 2026-05-23

**目标：** 对每个 HF exchange-like 独立块，通过 bilinear AO density → `get_j` → 原子切片的路径计算 ERI skeleton derivative。

### Step 1: Bilinear ERI 收缩原理

对形如 $E = \sum_{pqtu} L_{pq} R_{tu} (pq|tu)$ 的标量，固定 MO 系数后的 skeleton derivative 可写成 AO 双线性形式：

$$
E^{[x]}
= \sum_{\mu\nu\lambda\kappa}
L^{\mathrm{AO}}_{\mu\nu} R^{\mathrm{AO}}_{\lambda\kappa}
(\mu\nu|\lambda\kappa)^{[x]},
$$

其中 $L^{\mathrm{AO}}_{\mu\nu} = \sum_{pq} C_{\mu p} L_{pq} C_{\nu q}$，$R^{\mathrm{AO}}_{\lambda\kappa} = \sum_{tu} C_{\lambda t} R_{tu} C_{\kappa u}$。

→ **解释：** 这避免了显式构造 MO 四中心积分导数，从而利用 PySCF 的 `get_j` 核导数 API。

### Step 2: `_add_j_bilinear_ip1` 的双线性收缩

该 helper 对给定 $(\mathbf{L}, \mathbf{R})$ 对，调用 `td_grad.get_j` 得到 Coulomb 型导数势，再在原子循环中收缩：

```python
vj_r = td_grad.get_j(mol, dm_r, hermi=0)  # V_{mu,nu}^x = sum_{lk} R_{lk} (mu nu|lk)^x
vj_l = td_grad.get_j(mol, dm_l, hermi=0)  # V_{lk}^x = sum_{mu,nu} L_{mu,nu} (mu nu|lk)^x
```

→ **解释：** `get_j` 对非对称密度 (`hermi=0`) 返回的导数势有方向性（`(mu nu|lk)^x` 不是对称的），因此需要 row/column 分别收缩 $\mathbf{L}$ 和 $\mathbf{R}$。

### Step 3: 各块的 pair density 构造

| 块 | $\mathbf{L}$ 构造 | $\mathbf{R}$ 构造 | 系数 |
|---|---|---|---|
| $CO,CO$ | $\mathbf{C}_{o} \mathbf{X}_{co}^{T} \mathbf{C}_{c}^{T}$ | $\mathbf{C}_{c} \mathbf{X}_{co} \mathbf{C}_{o}^{T}$ | $-\frac{1}{2S-1}$ |
| $OV,OV$ | $\mathbf{C}_{v} \mathbf{X}_{ov}^{T} \mathbf{C}_{o}^{T}$ | $\mathbf{C}_{o} \mathbf{X}_{ov} \mathbf{C}_{v}^{T}$ | $-\frac{1}{2S-1}$ |
| $CV,CO$ | batch: $\mathbf{C}_{v} (\mathbf{x}_{cv}^{i} \otimes \mathbf{x}_{co}^{j}) \mathbf{C}_{o}^{T}$ | batch: $\mathbf{c}_{c}^{j} \otimes \mathbf{c}_{c}^{i}$ | $-2\eta$ |
| $CV,OV$ | batch: $\mathbf{C}_{v} (\mathbf{x}_{cv}^{i} \otimes \mathbf{x}_{ov}^{v}) \mathbf{C}_{v}^{T}$ | batch: $\mathbf{c}_{o}^{v} \otimes \mathbf{c}_{c}^{i}$ | $-2\eta$ |
| $CO,OV$ (1) | $\mathbf{C}_{o} \mathbf{X}_{co}^{T} \mathbf{C}_{c}^{T}$ | $\mathbf{C}_{o} \mathbf{X}_{ov} \mathbf{C}_{v}^{T}$ | $+\frac{2}{2S-1}$ |
| $CO,OV$ (2) | batch: $\mathbf{C}_{o} (\mathbf{x}_{co}^{i} \otimes \mathbf{x}_{ov}^{v}) \mathbf{C}_{v}^{T}$ | batch: $\mathbf{c}_{o}^{v} \otimes \mathbf{c}_{c}^{i}$ | $-\frac{2}{2S-1}$ |
| $CV,OO$ | batch: $\mathbf{C}_{v} (\mathbf{x}_{cv}^{i} \otimes \mathbf{x}_{oo}^{w}) \mathbf{C}_{o}^{T}$ | batch: $\mathbf{c}_{o}^{w} \otimes \mathbf{c}_{c}^{i}$ | $-2(\gamma-1)$ |
| $CO,OO$ | batch: $\mathbf{C}_{o} (\mathbf{x}_{co}^{i} \otimes \mathbf{x}_{oo}^{w}) \mathbf{C}_{o}^{T}$ | batch: $\mathbf{c}_{o}^{w} \otimes \mathbf{c}_{c}^{i}$ | $-2\zeta$ |
| $OV,OO$ | batch: $\mathbf{C}_{v} (\mathbf{x}_{ov}^{u} \otimes \mathbf{x}_{oo}^{w}) \mathbf{C}_{o}^{T}$ | batch: $\mathbf{c}_{o}^{w} \otimes \mathbf{c}_{o}^{u}$ | $-2\zeta$ |

→ **解释：** `batch` 标记的块不能写成简单的 MO 空间外积（因系数张量 $T_{pqtu}$ 不可分解），故对 shared indices 做循环，每对生成一个 $(\mathbf{L}_m, \mathbf{R}_m)$ 送入 `_add_j_bilinear_ip1_batches`。`blksize=64` 控制每批最多 64 个 density pair。

### Step 4: RSH 处理

若 $\omega \neq 0$，再以系数 $\alpha - c_{\mathrm{x}}$ 和 `omega=omega` 调用同一套 builder：

```python
_satda_delta_hf_exchange_direct_with_coeff(de, td_grad, tdobj, xy, ..., coeff=hyb)
if omega != 0:
    _satda_delta_hf_exchange_direct_with_coeff(de, ..., coeff=alpha-hyb, omega=omega)
```

→ **解释：** RSH 的 range-separated ERI 导数 $(pq|tu)_{\omega}^{[x]}$ 与普通 $(pq|tu)^{[x]}$ 结构相同，只是 Coulomb 算符换成 $\mathrm{erfc}(\omega r_{12})/r_{12}$。PySCF 的 `get_j(omega=omega)` 自动处理。

### 最终结果

总 HF exchange-like skeleton derivative 为：

$$
\boxed{
\Omega_{\mathrm{eri}}^{[x]}
= \sum_{\mathrm{block}\; B} c_B
\sum_{\mu\nu\lambda\kappa}
L^{(B)}_{\mu\nu} R^{(B)}_{\lambda\kappa}
(\mu\nu|\lambda\kappa)^{[x]}.
}
$$

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:_add_j_bilinear_ip1()` L716-738；`_add_j_bilinear_ip1_batches()` L741-752；`_satda_delta_hf_exchange_direct_with_coeff()` L755-843；`satda_delta_hf_exchange_direct_de()` L845-861。


## 推导：Z-vector 方程与 overlap metric 系数 — 2026-05-23

**目标：** 构造总 RHS、解 Z-vector、从收敛的 Z 回填 overlap metric coefficient $B^{S}_{pq}$。

### Step 1: 总 orbital RHS 构造

代码中的 UCPHF RHS 由两部分组成：

$$
\mathbf{W}^{\sigma}_{ai}
= \mathbf{W}^{\sigma,\mathrm{SF}}_{ai}
+ \mathbf{R}^{\sigma,\Delta}_{ai},
\qquad
\mathbf{R}^{\sigma,\Delta} = \left(\mathbf{Q}^{\sigma,\Delta} - (\mathbf{Q}^{\sigma,\Delta})^{T}\right)_{\mathrm{vir}_{\sigma} \times \mathrm{occ}_{\sigma}}.
$$

其中 $\mathbf{Q}^{\sigma,\Delta} = \mathbf{Q}^{\sigma,\Delta,F} + \mathbf{Q}^{\sigma,\Delta,\mathrm{eri}}$ 是 Fock-like 和 HF exchange-like 的总 Q。

→ **解释：** $\mathbf{W}^{\sigma,\mathrm{SF}}$ 是普通 SF-TDA 的 UCPHF RHS（来自 `dmzooa/dmzoob`、`dmt`、`get_jk`、`get_k` 等），与 `tduks_sf.py` 完全相同。SATDA 修正通过加上 $\mathbf{R}^{\sigma,\Delta}$ 完成。

### Step 2: Z-vector 方程

总 Z-vector 方程写为：

$$
\boxed{
\mathcal{H}_{\mathrm{SCF}} \mathbf{Z} = -\mathbf{W}^{\mathrm{total}}.
}
$$

→ **解释：** $\mathcal{H}_{\mathrm{SCF}}$ 是 SCF orbital Hessian（由 `mf.gen_response(hermi=1)` 提供）。所有 SATDA 修正只改变 RHS $\mathbf{W}$，不改变左端 Hessian。因此只需解一次 UCPHF。

代码使用 `ucphf.solve` 求解（L946-950）：

```python
def fvind(z):
    za = z[:nvira*nocca].reshape(nvira, nocca)
    zb = z[nvira*nocca:].reshape(nvirb, noccb)
    dma = orbva @ za @ orboa.T
    dmb = orbvb @ zb @ orbob.T
    dm1 = np.stack((dma + dma.T, dmb + dmb.T))
    v1 = vresp(dm1)
    return np.hstack((orbva.T @ v1[0] @ orboa).ravel(),
                      (orbvb.T @ v1[1] @ orbob).ravel())

z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa, wvob), ...)[0]
```

**注意：** 代码用 `_as_spin_unrestricted_reference(mf)` 将 ROKS/ROHF 转为 UKS/UHF 以适配 PySCF 的 UCPHF solver。ROKS→UKS 的 Hessian 等价性在 Li 等的 ROKS 梯度文献中已有论证。

### Step 3: Z-density 和 relaxed potential

Z-vector 解出后，构造 Z-density 并通过 response kernel 得到 relaxed potential：

```python
z1ao[0] = orbva @ z1a @ orboa.T
z1ao[1] = orbvb @ z1b @ orbob.T
veff = vresp(z1ao + z1ao.transpose(0,2,1))
```

→ **解释：** `veff` 是 Z-vector 诱导的 AO Fock response，将在 `im0` 和 `dmz1doo` 中使用。

### Step 4: Overlap metric coefficient $B^{S}$

定义总 MO coefficient derivative $\mathbf{M}$：

$$
M_{pq}
= Q^{\mathrm{SF}}_{pq}
+ Q^{\Delta}_{pq}
+ Q^{Z}_{pq}.
$$

→ **解释：** $Q^{\mathrm{SF}}$ 由普通 SF 的 Fock + exchange 项给出（即 `veff0doo`、`veff0mo` 在 `im0` 中的贡献）；$Q^{\Delta} = Q^{\Delta,F} + Q^{\Delta,\mathrm{eri}}$ 是 SATDA 修正；$Q^{Z}$ 是 Z-vector Lagrangian 的 MO 系数导数（即 `veff` 和 `zeta` 项）。

Z-vector 方程保证 $\mathbf{M}$ 在独立旋转空间中的反对称部分被消去。重叠导数的系数取对称部分：

$$
\boxed{
B^{S}_{pq}
= -\frac{1}{2}\left(M_{pq} + M_{qp}\right).
}
$$

在 PySCF 的约定中，overlap metric 项以 `de -= s1 * im0` 实现，其中 `im0 = -B^{S}` 的 AO 表示。代码中（L958-989）：

```python
im0a[:nocca,:nocca]  = orboa.T @ (veff0doo[0] + veff[0]) @ orboa  + ...  # SF Fock
im0a += (q_delta_a + q_delta_a.T) * 0.5                                     # SATDA Fock + ERI
...
im0a = mo_coeff[0] @ (im0a + zeta_a * dm1a) @ mo_coeff[0].T                # AO 转换
im0 = im0a + im0b
```

→ **解释：** 关键行是 `im0a += (q_delta_a + q_delta_a.T) * 0.5` — 它直接把 SATDA 总 $\mathbf{Q}^{\Delta}$ 的对称部分加入 overlap coefficient。`zeta_a * dm1a` 项来自占据数和轨道能量的规范化修正。

### Step 5: 总 overlap metric 贡献

$$
\boxed{
\Omega_{S}^{[x]}
= -\sum_{\mu\nu} I_{\mu\nu} S^{[x]}_{\mu\nu},
\qquad
I_{\mu\nu} = \sum_{pq} C_{\mu p} \, I^{\mathrm{MO}}_{pq} \, C_{\nu q},
\qquad
I^{\mathrm{MO}} = -B^{S}.
}
$$

在 atom loop 中对应：

```python
de -= einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
de -= einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])
```

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:grad_elec_hf_experimental()` L920-931（RHS 构造）、L935-950（Z-vector 求解）、L953-989（im0/overlap coefficient 构造）、L1031-1032（s1 收缩）。


## 推导：最终梯度组合公式 — 2026-05-23

**目标：** 将以上所有项汇总为 SATDA deltaS=-1 HF 解析梯度的最终公式。

### Step 1: 总激发态能量

$$
E_{n} = E_{\mathrm{ref}} + \omega_{n},
\qquad
\omega_{n} = \mathbf{X}^{T}\mathbf{A}_{\mathrm{SATDA}}^{\mathrm{HF}}\mathbf{X}.
$$

### Step 2: 总 Lagrangian

$$
\mathcal{L}
= E_{\mathrm{ref}}
+ \mathbf{X}^{T}\mathbf{A}_{\mathrm{SATDA}}^{\mathrm{HF}}\mathbf{X}
- \omega(\mathbf{X}^{T}\mathbf{X} - 1)
+ \mathbf{Z}^{T}\mathbf{g}_{\mathrm{SCF}}
+ \sum_{pq} W_{pq}(S_{pq} - \delta_{pq}).
$$

### Step 3: 核坐标导数

总能量对核坐标 $x$ 的导数分解为四项：

$$
\boxed{
E_{n}^{[x]}
= E_{\mathrm{ref}}^{[x]}
+ \Omega_{\mathrm{dir}}^{[x]}
+ \Omega_{Z}^{[x]}
+ \Omega_{S}^{[x]}.
}
$$

各项展开为：

**（A）参考态梯度：**

$E_{\mathrm{ref}}^{[x]}$ 由 `mf.nuc_grad_method().grad_nuc(atmlst)` 计算（PySCF 的 ROHF/ROKS 解析梯度）。

**（B）直接 skeleton derivative：**

$$
\Omega_{\mathrm{dir}}^{[x]}
= \Omega_{\mathrm{Fock},h}^{[x]}
+ \Omega_{\mathrm{Fock},J}^{[x]}
+ \Omega_{\mathrm{Fock},K}^{[x]}
+ \Omega_{\mathrm{eri}}^{[x]}.
$$

其中前三项包含普通 SF Fock 和 SATDA Fock-like 修正，第四项包含普通 SF exchange 和 SATDA exchange-like 修正。

**（C）Z-vector 直接项：**

$$
\Omega_{Z}^{[x]}
= \mathbf{Z}^{T}\mathbf{g}_{\mathrm{SCF}}^{[x]}.
$$

→ **解释：** 该项由 `z1ao` 和 `veff` 参与 `im0` 和 `dmz1doo` 构造隐式地体现在 atom loop 中，不需要显式调用。

**（D）Overlap metric 项：**

$$
\Omega_{S}^{[x]}
= -\sum_{\mu\nu} I_{\mu\nu} S^{[x]}_{\mu\nu}.
$$

### Step 4: 代码入口

```python
def _kernel_analytic_experimental(self, xy, atmlst):
    # 1. 内部能量检查
    e_probe = _satda_hf_energy_for_orbs(self.base, xy, mf.mo_coeff, mf.mo_coeff)
    # 2. 解析电子梯度（覆盖 direct + Z + overlap 三项）
    de = grad_elec_hf_experimental(self, xy, atmlst=atmlst, ...)
    # 3. 加核排斥梯度
    de += self.base._scf.nuc_grad_method().grad_nuc(atmlst=atmlst)
    return de
```

### Step 5: 总计算图

```
X_co, X_cv, X_oo, X_ov
  │
  ├─→ satda_fock_coefficients() ──→ 7 类 T 矩阵
  │     ├─→ _add_fock_term() ──→ Q_F (projection)
  │     ├─→ satda_fock_probe_densities() ──→ dm_probe_a/b (skeleton)
  │     └─→ _add_fock_response_q() ──→ Q_F,resp
  │
  ├─→ _add_eri_term_q() × 9 ──→ Q_eri (4-projection)
  │
  └─→ _satda_delta_hf_exchange_direct_with_coeff() ──→ bilinear ERI [x]
  
  Q_F + Q_eri ──→ Q_delta ──→ Q - Q^T ──→ W_total ──→ UCPHF ──→ Z
  Q_delta ──→ (Q + Q^T)/2 ──→ im0 (overlap)
  dm_probe_a/b ──→ dmz1doo_direct ──→ skeleton
```

**代码对应：** `pyscf-forge/pyscf/grad/tdsatda.py:grad_elec_hf_experimental()` L864-1046；`Gradients._kernel_analytic_experimental()` L1168-1194。

---

## 推导：SATDA/HF $\Delta S=-1$ 态间 NAC 的 AWF 有限差分与解析双线性化 — 2026-05-23

**目标：** 从 SATDA/HF 的激发能梯度工作方程出发，得到两个 SATDA 态 $I,J$ 之间的非绝热导数耦合（NAC）实现公式，并说明 `pyscf/nac/tdsatda.py` 中有限差分和解析实现的对应关系。

**假设：**
- 只考虑 Tamm-Dancoff 近似，$Y=0$。
- 只考虑 `deltaS=-1`，即 $S_f=S_i-1$ 的 spin-flip 激发。
- 当前解析实现只覆盖 HF 或 `xc='HF'`，不含 DFT XC 核导数。
- 轨道、AO 基函数和激发振幅均取实数。

**符号声明：**
- $X^{I}_{ia}$ 和 $X^{J}_{ia}$ 分别表示态 $I,J$ 的 SATDA spin-flip 振幅，其中 $i$ 属于 $\alpha$ 占据空间，$a$ 属于 $\beta$ 非双占据空间。
- $\omega_I,\omega_J$ 是 SATDA 激发能，$\Delta E_{JI}=\omega_J-\omega_I$。
- $\mathbf{A}$ 是 SATDA/TDA 矩阵；$\mathbf{A}^{[x]}$ 表示去除 MO 响应后的核坐标 $x$ 显式导数。
- $\mathbf{Z}^{IJ}$ 是态间 NAC 的 Z-vector。
- $\mathbf{Q}_{\Delta A}$ 表示 SATDA spin-adaptation 修正项对 MO 旋转的一阶导数矩阵。

**代码变量到数学符号的对应：**
- `x_y_i[0]`, shape `(nocca, nvirb)` = $X^{I}_{ia}$。
- `x_y_j[0]`, shape `(nocca, nvirb)` = $X^{J}_{ia}$。
- `get_hf_interstate_numerator()` = $\langle I|\hat{H}^{[x]}|J\rangle$ 加 orbital-response 消元后的 numerator。
- `awf_overlap()` = 辅助波函数重叠 $\langle \widetilde{\Phi}_I(\mathbf{R})|\widetilde{\Phi}_J(\mathbf{R}')\rangle$。
- `nac_csf()` = AWF determinant basis 的显式 overlap-metric 项。

### Step 1: NAC 与 numerator 的关系

对绝热本征态有

$$
\hat{H}|\Phi_K\rangle = E_K|\Phi_K\rangle.
$$

→ **解释：** 这里 $K$ 可取 $I$ 或 $J$；在 SATDA 中实际使用的是辅助波函数或等价的 EOM/TDA 响应表象。

对核坐标 $x$ 求导并左乘 $\langle\Phi_I|$，在 $I\ne J$ 时得到

$$
\langle\Phi_I|\hat{H}^{[x]}|\Phi_J\rangle
= (E_I-E_J)\langle\Phi_I|\Phi_J^{[x]}\rangle.
$$

→ **解释：** 使用正交归一条件 $\langle\Phi_I|\Phi_J\rangle=0$ 和本征方程消去 $\hat{H}|\Phi_J^{[x]}\rangle$ 项。

因此代码中的两种返回约定为

$$
\mathbf{d}_{IJ}^{x}
= \langle\Phi_I|\Phi_J^{[x]}\rangle,
\qquad
\mathbf{N}_{IJ}^{x}
= \Delta E_{JI}\,\mathbf{d}_{IJ}^{x}.
$$

→ **解释：** `ediff=True` 返回 $\mathbf{d}_{IJ}^{x}$，`ediff=False` 返回 numerator-like quantity $\mathbf{N}_{IJ}^{x}$；这与已有 SF-TDDFT NAC 代码保持一致。

### Step 2: AWF 有限差分参考

SATDA/TDA 辅助波函数写为

$$
|\widetilde{\Phi}_I\rangle
= \sum_{ia} X^{I}_{ia} |\widetilde{\Phi}_{ia}\rangle.
$$

→ **解释：** $|\widetilde{\Phi}_{ia}\rangle$ 是由 ROKS 高自旋参考态生成的 spin-flip determinant/CSF 基函数；`awf_overlap()` 用行列式 overlap 评价其几何变化。

有限差分 NAC 采用固定左态、移动右态的中心差分：

$$
d_{IJ}^{x}
\approx
\frac{
\langle\widetilde{\Phi}_I(\mathbf{R})|
\widetilde{\Phi}_J(\mathbf{R}+h\mathbf{e}_x)\rangle
-
\langle\widetilde{\Phi}_I(\mathbf{R})|
\widetilde{\Phi}_J(\mathbf{R}-h\mathbf{e}_x)\rangle
}{2h}.
$$

→ **解释：** `NonAdiabaticCouplings._kernel_finite_diff()` 对每个原子坐标重跑 ROKS/SATDA，用振幅 overlap 做 root tracking，并通过 `awf_overlap()` 计算左右几何之间的辅助波函数重叠。

### Step 3: 解析项由梯度公式双线性化得到

对单一态 $K$，SATDA/HF 解析梯度中的电子部分可写成二次型泛函

$$
G^{x}(X^{K},X^{K})
=
(X^{K})^{T}\mathbf{A}^{[x]}X^{K}
+ \text{orbital-response and overlap terms}.
$$

→ **解释：** `grad_elec_hf_experimental()` 已经实现了普通 SF 项、SATDA Fock-like probe density、SATDA HF exchange-like direct skeleton derivative 和 Z-vector overlap metric 项。

态间矩阵元用极化恒等式从二次型得到：

$$
G^{x}(X^{I},X^{J})
=
\frac{1}{2}
\left[
G^{x}(X^{I}+X^{J},X^{I}+X^{J})
-G^{x}(X^{I},X^{I})
-G^{x}(X^{J},X^{J})
\right].
$$

→ **解释：** 这就是 `_polarized_tuple()` 与 `_polarized_array()` 的作用；它把已有 gradient helper 中的 SATDA $\Delta A$ 二次项转换为 NAC 所需的态间双线性项。

### Step 4: SATDA 专属 $\Delta A$ 项进入 Z-vector RHS

SATDA 修正对 MO 旋转的导数写成

$$
\mathbf{Q}_{\Delta A}^{IJ}
=
\frac{1}{2}
\left[
\mathbf{Q}_{\Delta A}(X^{I}+X^{J})
-\mathbf{Q}_{\Delta A}(X^{I})
-\mathbf{Q}_{\Delta A}(X^{J})
\right].
$$

→ **解释：** `satda_delta_q()` 原本返回单态二次型的 MO derivative；NAC 中通过 `_bilinear_delta_q()` 得到态间版本。

Z-vector RHS 的 SATDA 增量取反对称部分：

$$
\mathbf{R}_{\Delta A}^{IJ}
= \mathbf{Q}_{\Delta A}^{IJ}
- \left(\mathbf{Q}_{\Delta A}^{IJ}\right)^{T}.
$$

→ **解释：** 代码中 `r_delta_a/b = q_delta_a/b - q_delta_a/b.T`，随后投影到 virtual-occupied 块并加到 `wvoa/wvob`。

### Step 5: SATDA 专属 overlap metric 项

同一个 $\mathbf{Q}_{\Delta A}^{IJ}$ 的对称部分进入 overlap metric coefficient：

$$
\mathbf{B}_{\Delta A}^{IJ}
=
\frac{1}{2}
\left[
\mathbf{Q}_{\Delta A}^{IJ}
+ \left(\mathbf{Q}_{\Delta A}^{IJ}\right)^{T}
\right].
$$

→ **解释：** `get_hf_interstate_numerator()` 中将该项加到 `im0a/im0b`，对应梯度实现里 Z-vector 之后回填 overlap metric 的 SATDA 修正。

### Step 6: 直接 AO skeleton derivative

SATDA Fock-like probe density 的态间形式为

$$
\mathbf{D}_{\mathrm{probe}}^{IJ}
=
\frac{1}{2}
\left[
\mathbf{D}_{\mathrm{probe}}(X^{I}+X^{J})
-\mathbf{D}_{\mathrm{probe}}(X^{I})
-\mathbf{D}_{\mathrm{probe}}(X^{J})
\right].
$$

→ **解释：** `_bilinear_probe_densities()` 对 `satda_fock_probe_densities()` 做极化，得到 $\alpha$ 和 $\beta$ 两个 spin-separated probe density。

HF exchange-like 的直接 AO 积分导数同样双线性化：

$$
\Omega_{\Delta A,\mathrm{eri}}^{[x],IJ}
=
\frac{1}{2}
\left[
\Omega_{\Delta A,\mathrm{eri}}^{[x]}(X^{I}+X^{J})
-\Omega_{\Delta A,\mathrm{eri}}^{[x]}(X^{I})
-\Omega_{\Delta A,\mathrm{eri}}^{[x]}(X^{J})
\right].
$$

→ **解释：** `_bilinear_direct_de()` 复用 `satda_delta_hf_exchange_direct_de()`，因此 gradient 中已拆好的 COCO、OVOV、CVCO、CVOV、COOV、CVOO、COOO、OVOO 等 direct ERI 块可以直接进入 NAC。

### 最终结果

解析实现返回的 numerator 为

$$
\mathbf{N}_{IJ}^{x}
=
\mathbf{N}_{IJ,\mathrm{ordinary\ SF}}^{x}
+ \mathbf{N}_{IJ,\Delta A}^{x}
+ \Delta E_{JI}\,\mathbf{d}_{IJ,\mathrm{basis}}^{x}.
$$

→ **解释：** 前两项由 `get_hf_interstate_numerator()` 计算；最后一项由 `nac_csf()` 给出，在 `use_etfs=False` 时加入，用来匹配 AWF 有限差分参考。

最终 NAC 为

$$
\mathbf{d}_{IJ}^{x}
=
\frac{\mathbf{N}_{IJ}^{x}}{\Delta E_{JI}}.
$$

→ **解释：** `ediff=True` 时执行该除法；若 `ediff=False`，代码保留 numerator 形式，便于与动力学程序中按能隙处理的约定兼容。

**代码对应：** `pyscf-forge/pyscf/nac/tdsatda.py:get_hf_interstate_numerator()`、`awf_overlap()`、`nac_csf()`、`NonAdiabaticCouplings.kernel()`；入口为 `pyscf-forge/pyscf/sftda/satda.py:SATDA.NAC()`。

---

## 推导补记：SATDA/HF NAC 解析路径的对角极限检查 — 2026-05-23

**目标：** 记录当前 `analytic_experimental` NAC 路径被禁用的原因，并给出后续修复必须满足的最小物理校验。

**假设：**
- 仍限定 HF、TDA、`deltaS=-1`。
- `finite_diff` 使用 AWF overlap 中心差分，并已在甲醛 triplet reference 的非零 NAC 分量上检查步长平台。
- 对角极限只用于校验 Hellmann-Feynman numerator，不表示要计算同态 NAC。

**符号声明：**
- $\omega_I$ 是第 $I$ 个 SATDA 激发能。
- $\mathbf{N}_{IJ}^{x}$ 是解析 NAC 中的 Hellmann-Feynman numerator。
- $\mathbf{G}_{I,\mathrm{exc}}^{x}$ 是纯激发能梯度，即总激发态梯度减去基态梯度。

### Step 1: 对角极限应成立

若解析 numerator 的对象确实是纯激发空间哈密顿量，则在 $I=J$ 时必须满足

$$
\mathbf{N}_{II}^{x} = \frac{\partial \omega_I}{\partial x}.
$$

→ **解释：** 这是 Hellmann-Feynman 定理在 TDA 激发矩阵本征值问题上的对角极限；注意右边不是总激发态能量梯度，而是总梯度减去基态梯度。

代码校验中应使用

$$
\mathbf{G}_{I,\mathrm{exc}}^{x}
=
\mathbf{G}_{I,\mathrm{total}}^{x}
-\mathbf{G}_{\mathrm{ref}}^{x}.
$$

→ **解释：** `td.Gradients().kernel(state=I)` 的目标是 $E_{\mathrm{ref}}+\omega_I$，因此直接比较会把基态贡献混入。

### Step 2: 甲醛诊断结果

在甲醛 `STO-3G`、triplet ROKS/HF reference、`state_I=2`、`state_J=3`、O 原子 $z$ 分量上，有限差分 NAC 的步长平台为

$$
d_{23}^{z}(h=5\times 10^{-4})
\approx -0.034904647.
$$

→ **解释：** 该值与 $h=2\times 10^{-4}$ 的结果只差约 $1.8\times 10^{-7}$，因此不是步长过小导致的不稳定大数。

对应能隙为

$$
\Delta E_{32} \approx 0.003639802089.
$$

→ **解释：** 因此有限差分对应的 numerator 量级应为 $-1.27\times 10^{-4}$ Hartree/Bohr，而不是 $10^{-2}$ Hartree/Bohr。

当前旧解析路径给出

$$
\mathbf{N}_{23}^{z} \approx -0.03726804,
\qquad
\frac{\mathbf{N}_{23}^{z}}{\Delta E_{32}} \approx -10.239.
$$

→ **解释：** 这比有限差分 numerator 大约两个数量级，说明错误来自 Hellmann-Feynman numerator 构造。

### Step 3: 分块定位

旧路径中 `get_hf_interstate_numerator()` 的 atom loop 可分为

$$
\mathbf{N}^{x}
=
\mathbf{N}_{h}^{x}
+\mathbf{N}_{S}^{x}
+\mathbf{N}_{\mathrm{veff,dz}}^{x}
+\mathbf{N}_{\mathrm{veff,oo}}^{x}
+\mathbf{N}_{K}^{x}
+\mathbf{N}_{\Delta A,\mathrm{direct}}^{x}.
$$

→ **解释：** 这些分别对应 `hcore/as_dm1`、`overlap/im0`、`veff1` 与 response density、`veff1` 与 reference occupied density、ordinary SF exchange derivative、SATDA direct ERI derivative。

甲醛诊断显示异常主要来自大项之间的错误抵消，而不是 `nac_csf()` 或 SATDA direct ERI 单项：

$$
\mathbf{N}_{h}^{z}\approx -0.39393,\quad
\mathbf{N}_{\mathrm{veff,dz}}^{z}\approx 0.30012,\quad
\mathbf{N}_{\mathrm{veff,oo}}^{z}\approx 0.14150,\quad
\mathbf{N}_{K}^{z}\approx -0.08989.
$$

→ **解释：** 这些量级都远大于有限差分 numerator。把普通 SF-NAC/gradient 的 atom-loop 结构直接套到 SATDA 自旋适配振幅上，不能保证得到纯 SATDA excitation Hamiltonian derivative。

### 最终结论

当前 `analytic_experimental` NAC 路径被禁用，直到满足下面的最小条件：

$$
\mathrm{get\_hf\_interstate\_numerator}(X_I,X_I)
=
\mathbf{G}_{I,\mathrm{total}}
-\mathbf{G}_{\mathrm{ref}}.
$$

→ **解释：** 只有对角极限先成立，态间 $I\ne J$ 的解析 NAC 才有继续验证的意义。

**代码对应：** `pyscf-forge/pyscf/nac/tdsatda.py:NonAdiabaticCouplings._kernel_analytic_experimental()` 当前显式抛出 `NotImplementedError`；有限差分路径 `method="finite_diff"` 保持可用。

# Data Model: UWB 脉冲功率谱密度与 FCC 合规性分析

**Feature**: 002-psd-fcc-compliance
**Date**: 2025-12-17
**Status**: ✅ Complete

本文档定义了 PSD/FCC 合规性可视化特性的 4 个核心实体及其关系。

---

## 实体关系图

```
┌──────────────────────────┐
│  Gaussian Derivative     │
│       Pulse              │
│  - n (阶数)              │
│  - τ (脉冲宽度)          │
│  - fc (中心频率)         │
│  - amplitude             │
└───────────┬──────────────┘
            │ generates
            ↓
┌──────────────────────────┐
│  Power Spectral Density  │
│        (PSD)             │
│  - frequencies           │
│  - psd_values (dBm/MHz)  │
└───────────┬──────────────┘
            │ compares with
            ↓
┌──────────────────────────┐       ┌──────────────────────────┐
│   FCC Indoor Mask        │       │   Compliance Result      │
│  - frequency_bands       │──────>│  - status                │
│  - power_limits          │       │  - violations            │
│  - band_definitions      │       │  - max_excess            │
└──────────────────────────┘       └──────────────────────────┘
            check_compliance()
```

---

## 实体 1: Gaussian Derivative Pulse（高斯导数脉冲）

### 概念定义

高斯导数脉冲是 UWB 系统中常用的脉冲波形，通过对高斯函数进行 n 阶微分得到。不同阶数的脉冲具有不同的频域特性，影响 FCC 合规性。

### 属性

| 属性名 | 类型 | 单位 | 取值范围 | 说明 |
|--------|------|------|----------|------|
| `n` | int | - | {2, 5} | 脉冲阶数（2 阶或 5 阶） |
| `tau` (τ) | float | 秒 (s) | [0.1ns, 2ns] | 脉冲宽度参数，影响频谱展宽 |
| `fc` | float | 赫兹 (Hz) | [0, 15e9] | 中心频率，用于频谱搬移 |
| `amplitude` | float | - | (0, ∞) | 幅度归一化参数，通常归一化为单位能量 |
| `sampling_rate` | float | 赫兹 (Hz) | ≥24e9 | 采样率，满足奈奎斯特定理 |

### 方法

#### `generate_waveform(self) -> np.ndarray`

**功能**：生成脉冲的时域波形。

**输入**：无（使用实体属性）

**输出**：
- `waveform`: np.ndarray，shape (N,)，时域采样点

**算法**：
1. 计算时间轴：t = np.linspace(-5*tau, 5*tau, num_samples)
2. 根据阶数 n 计算归一化系数 A_n
3. 应用公式：
   - n=2: `A_2 * (1 - t²/τ²) * exp(-t²/(2τ²))`
   - n=5: `A_5 * [15t/τ² - 10t³/τ⁴ + t⁵/τ⁶] * exp(-t²/(2τ²))`
4. 如果 fc > 0，进行载波调制：`waveform * cos(2π·fc·t)`

**前置条件**：
- `n` ∈ {2, 5}
- `tau` ∈ [0.1ns, 2ns]
- `sampling_rate` ≥ 2 * max_frequency

**后置条件**：
- `len(waveform)` 基于 `sampling_rate` 和 `tau` 计算
- 能量归一化：`∫|waveform|² dt ≈ 1`（误差 < 1%）

**参考**：
- research.md Section 1.1: 时域公式
- contracts/pulse.md

---

#### `get_time_axis(self) -> np.ndarray`

**功能**：获取时域波形对应的时间轴。

**输入**：无

**输出**：
- `time_axis`: np.ndarray，shape (N,)，单位：秒 (s)

**算法**：
```python
t = np.linspace(-5*tau, 5*tau, num_samples)
```

**前置条件**：无

**后置条件**：
- `len(time_axis) == len(waveform)`
- 时间轴均匀采样，间隔为 1/sampling_rate

---

## 实体 2: Power Spectral Density（功率谱密度）

### 概念定义

功率谱密度描述信号功率在频域的分布，单位为 dBm/MHz。通过 FFT 将时域脉冲转换为频域 PSD，用于与 FCC 掩蔽罩对比。

### 属性

| 属性名 | 类型 | 单位 | 取值范围 | 说明 |
|--------|------|------|----------|------|
| `frequencies` | np.ndarray | 赫兹 (Hz) | [0, 12e9] | 频率轴，单边频谱 |
| `psd_values` | np.ndarray | dBm/MHz | (-∞, +20] | 功率谱密度值（对数刻度） |
| `freq_resolution` | float | 赫兹 (Hz) | ≥10e6 | 频率分辨率（SC-002） |

### 方法

#### `compute_from_pulse(pulse: GaussianDerivativePulse) -> PowerSpectralDensity`

**功能**：从脉冲波形计算 PSD。

**输入**：
- `pulse`: GaussianDerivativePulse 实例

**输出**：
- `psd`: PowerSpectralDensity 实例

**算法**：
1. 获取时域波形：`waveform = pulse.generate_waveform()`
2. 计算 FFT：`fft_result = np.fft.rfft(waveform, n=N_fft)`
3. 计算单边功率谱：`power = |fft_result|² / (N_fft * sampling_rate)`
4. 归一化到 1 MHz 带宽：`psd_linear = power * sampling_rate / 1e6`
5. 转换为 dBm/MHz：`psd_dbm = 10 * log10(psd_linear / 1e-3)`
6. 计算频率轴：`frequencies = np.fft.rfftfreq(N_fft, d=1/sampling_rate)`

**前置条件**：
- `pulse.waveform` 已生成且非空
- `sampling_rate` ≥ 24 GHz（奈奎斯特）

**后置条件**：
- `len(frequencies) == len(psd_values)`
- `freq_resolution = sampling_rate / N_fft ≤ 10 MHz`（SC-002）
- Parseval 定理验证：`∫|waveform|² dt ≈ ∫PSD(f) df`（误差 < 1%）

**参考**：
- research.md Section 1.3: FFT 参数选择
- contracts/psd.md

---

#### `to_db(self) -> np.ndarray`

**功能**：将线性功率谱转换为对数刻度（dBm/MHz）。

**输入**：无（使用实体属性）

**输出**：
- `psd_db`: np.ndarray，单位 dBm/MHz

**算法**：
```python
psd_db = 10 * np.log10(psd_linear / 1e-3)
```

**前置条件**：
- `psd_linear` 已计算且所有值 > 0

**后置条件**：
- `psd_db` 的 shape 与 `psd_linear` 相同

---

## 实体 3: FCC Indoor Mask（FCC 室内掩蔽罩）

### 概念定义

FCC Part 15.209 规定的 UWB 设备室内辐射限制，定义了不同频段的功率密度上限。脉冲 PSD 必须在所有频点低于掩蔽罩才能合规。

### 属性

| 属性名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `band_definitions` | list[tuple] | - | 频段边界列表：[(f_start, f_end, limit), ...] |
| `frequencies` | np.ndarray | Hz | 频率轴（与 PSD 对齐） |
| `mask_values` | np.ndarray | dBm/MHz | 掩蔽罩功率限制值 |

### 频段定义

| 频率范围 | 功率限制 (dBm/MHz) | FCC 标准依据 |
|----------|-------------------|-------------|
| 0.96 - 1.61 GHz | -75.3 | CFR 47 §15.209(a) |
| 1.61 - 3.1 GHz | -53.3 | CFR 47 §15.209(a) |
| 3.1 - 10.6 GHz | -41.3 | CFR 47 §15.209(a) (UWB 合法频段) |
| >10.6 GHz | -51.3 | CFR 47 §15.209(a) |

### 方法

#### `get_fcc_indoor_mask(frequencies: np.ndarray) -> FCC_Indoor_Mask`

**功能**：生成 FCC 室内掩蔽罩。

**输入**：
- `frequencies`: np.ndarray，频率轴（单位：Hz）

**输出**：
- `mask`: FCC_Indoor_Mask 实例

**算法**（使用 `np.piecewise()`）：

```python
def get_limit(f):
    # f 单位：Hz
    if 0.96e9 <= f < 1.61e9:
        return -75.3
    elif 1.61e9 <= f < 3.1e9:
        return -53.3
    elif 3.1e9 <= f < 10.6e9:
        return -41.3
    elif f >= 10.6e9:
        return -51.3
    else:
        return -100.0  # 低于 0.96 GHz 的频率（极低功率）

mask_values = np.piecewise(
    frequencies,
    [
        (frequencies >= 0.96e9) & (frequencies < 1.61e9),
        (frequencies >= 1.61e9) & (frequencies < 3.1e9),
        (frequencies >= 3.1e9) & (frequencies < 10.6e9),
        frequencies >= 10.6e9
    ],
    [-75.3, -53.3, -41.3, -51.3, -100.0]
)
```

**前置条件**：
- `frequencies` 为递增数组
- `frequencies` 覆盖范围 [0, 12 GHz]

**后置条件**：
- `len(mask_values) == len(frequencies)`
- 掩蔽罩在边界频点处阶跃变化（非平滑过渡）
- 数值精度：与 FCC 标准误差 < 0.1 dB（SC-001）

**参考**：
- research.md Section 2.1: 室内辐射掩蔽罩
- contracts/fcc_mask.md

---

#### `get_limit_at_frequency(self, f: float) -> float`

**功能**：获取指定频率处的功率限制。

**输入**：
- `f`: float，频率（单位：Hz）

**输出**：
- `limit`: float，功率限制（单位：dBm/MHz）

**算法**：查表或使用 np.interp()（最近邻插值）

**前置条件**：
- `f` ≥ 0

**后置条件**：
- 返回值为 4 个频段限制值之一

---

#### `plot_mask(self, ax: plt.Axes) -> None`

**功能**：在给定的 matplotlib 坐标轴上绘制掩蔽罩。

**输入**：
- `ax`: matplotlib Axes 对象

**输出**：无（修改 ax 对象）

**算法**：
```python
ax.plot(self.frequencies / 1e9,  # 转换为 GHz
        self.mask_values,
        color="red", linestyle=":", linewidth=2.0,
        label="FCC 室内掩蔽罩")
```

**前置条件**：
- `ax` 为有效的 matplotlib Axes

**后置条件**：
- 掩蔽罩曲线已添加到图表
- 图例标签为 "FCC 室内掩蔽罩"

---

## 实体 4: Compliance Result（合规性结果）

### 概念定义

合规性结果记录脉冲 PSD 与 FCC 掩蔽罩的对比结果，包括合规状态、违规频点列表和最大超限量。

### 属性

| 属性名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `status` | str | - | "Compliant", "Non-compliant", "Marginal Compliant" |
| `violations` | list[tuple] | - | 违规频点列表：[(freq, excess_dB), ...] |
| `max_excess` | float | dB | 最大超限量（正值表示违规） |
| `tolerance` | float | dB | 边界容差（默认 0.1 dB） |

### 方法

#### `check_compliance(psd: PowerSpectralDensity, fcc_mask: FCC_Indoor_Mask, tolerance: float = 0.1) -> ComplianceResult`

**功能**：判断脉冲 PSD 是否符合 FCC 规范。

**输入**：
- `psd`: PowerSpectralDensity 实例
- `fcc_mask`: FCC_Indoor_Mask 实例
- `tolerance`: float，边界容差（单位：dB，默认 0.1）

**输出**：
- `result`: ComplianceResult 实例

**算法**：

```python
def check_compliance(psd, fcc_mask, tolerance=0.1):
    # 计算超限量（dB）
    diff = psd.psd_values - fcc_mask.mask_values
    max_excess = np.max(diff)

    # 判断合规状态
    if max_excess < -tolerance:
        status = "Compliant"
        violations = []
    elif max_excess > tolerance:
        status = "Non-compliant"
        # 找出所有违规频点
        violation_indices = np.where(diff > tolerance)[0]
        violations = [
            (psd.frequencies[i], diff[i])
            for i in violation_indices
        ]
    else:  # -tolerance ≤ max_excess ≤ +tolerance
        status = "Marginal Compliant"
        violations = []

    return ComplianceResult(
        status=status,
        violations=violations,
        max_excess=max_excess,
        tolerance=tolerance
    )
```

**前置条件**：
- `len(psd.psd_values) == len(fcc_mask.mask_values)`
- `psd.frequencies` 与 `fcc_mask.frequencies` 对齐

**后置条件**：
- `status` 为三个状态之一
- 如果 `status == "Non-compliant"`，则 `len(violations) > 0`
- 准确率 100%（SC-003）

**参考**：
- research.md Section 2.3: 合规性判断逻辑
- contracts/compliance.md

---

#### `generate_report(self) -> str`

**功能**：生成合规性报告（纯文本格式）。

**输入**：无（使用实体属性）

**输出**：
- `report`: str，格式化的合规性报告

**算法**：

```python
def generate_report(self):
    report = f"合规性状态: {self.status}\n"
    report += f"最大超限量: {self.max_excess:.2f} dB\n"
    report += f"边界容差: {self.tolerance:.2f} dB\n"

    if self.status == "Non-compliant":
        report += f"\n违规频点数量: {len(self.violations)}\n"
        report += "违规频点列表（前 10 个）:\n"
        for freq, excess in self.violations[:10]:
            report += f"  {freq/1e9:.3f} GHz: 超限 {excess:.2f} dB\n"

    return report
```

**前置条件**：
- `status` 已设置

**后置条件**：
- 返回格式化的字符串

---

## 实体关系

### 1. Pulse → PSD

**关系类型**：一对一（每个脉冲对应一个 PSD）

**关系方法**：
```python
psd = PowerSpectralDensity.compute_from_pulse(pulse)
```

**约束**：
- PSD 的频率分辨率取决于脉冲的采样率和 FFT 点数
- PSD 的频率范围不超过脉冲采样率的一半（奈奎斯特）

### 2. PSD + FCC Mask → Compliance Result

**关系类型**：多对一（多个 PSD 可与同一 FCC Mask 对比）

**关系方法**：
```python
result = ComplianceResult.check_compliance(psd, fcc_mask, tolerance=0.1)
```

**约束**：
- PSD 和 FCC Mask 的频率轴必须对齐（相同长度和频率范围）
- 边界容差 tolerance 可调，默认 0.1 dB

### 3. FCC Mask (独立实体)

**关系类型**：单例（FCC 标准唯一）

**特性**：
- FCC Mask 不依赖具体脉冲设计
- 所有脉冲的 PSD 都与同一 FCC Mask 对比

---

## 数据流图

```
用户输入参数
    ↓
[n, τ, fc]
    ↓
Gaussian Derivative Pulse
    ├─> generate_waveform()
    │       ↓
    │   waveform (时域)
    │       ↓
    └─> compute_psd()
            ↓
        PSD (频域)
            ↓
        ┌───┴───┐
        ↓       ↓
    PSD值   频率轴
        │       │
        └───┬───┘
            ↓
    check_compliance()
            ↓
    ┌───────┴────────┐
    ↓                ↓
Compliance      FCC Mask
  Result         (固定)
    ↓
输出：图表 + 报告
```

---

## 验证标准

### 数据一致性

- [ ] PSD 的频率轴与 FCC Mask 的频率轴长度相同
- [ ] 所有频率值为正数且递增
- [ ] PSD 值在合理范围内（-100 到 +20 dBm/MHz）

### 物理约束

- [ ] 脉冲能量守恒：时域能量 ≈ 频域能量（Parseval 定理）
- [ ] 频率分辨率 ≥10 MHz（SC-002）
- [ ] FCC 掩蔽罩数值与官方标准误差 < 0.1 dB（SC-001）

### 业务规则

- [ ] 合规性判断准确率 100%（SC-003）
- [ ] 5 阶脉冲（fc=6.85 GHz）应判定为"Compliant"
- [ ] 2 阶脉冲（无调制）应判定为"Non-compliant"

---

## 扩展性设计

### 未来可能的扩展

1. **支持更多脉冲阶数**：n ∈ {1, 2, 3, ..., 10}
2. **支持其他 FCC 掩蔽罩**：室外、手持设备、成像系统
3. **支持多国标准**：ETSI（欧洲）、ARIB（日本）
4. **支持参数优化**：自动搜索满足 FCC 的最优 (fc, τ)

### 扩展点

- **Pulse 工厂模式**：支持不同类型脉冲（矩形、三角、chirp）
- **Mask 策略模式**：支持不同国家/地区的掩蔽罩
- **Compliance 规则引擎**：支持自定义合规性判断规则

---

**Data Model 状态**: ✅ 完成
**下一步**: 生成 contracts/ 目录下的 4 个契约文件

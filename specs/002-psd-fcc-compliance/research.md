# Research Findings: UWB 脉冲功率谱密度与 FCC 合规性分析

**Feature**: 002-psd-fcc-compliance
**Date**: 2025-12-17
**Status**: ✅ Complete

本文档记录了 PSD/FCC 合规性可视化特性的研究发现、技术决策和证据支撑。

---

## 1. 高斯导数脉冲的 PSD 计算方法

### 1.1 时域公式

**n 阶高斯导数脉冲（Gaussian Derivative Pulse）** 的时域表达式：

```
p_n(t) = A_n · d^n/dt^n [exp(-t²/(2τ²))]
```

其中：
- `n` = 脉冲阶数（本项目关注 n=2 和 n=5）
- `τ` = 脉冲宽度参数（单位：秒）
- `A_n` = 归一化系数，确保脉冲能量为 1

**2 阶高斯导数脉冲（n=2）**：

```
p_2(t) = A_2 · (1 - t²/τ²) · exp(-t²/(2τ²))
```

**5 阶高斯导数脉冲（n=5）**：

```
p_5(t) = A_5 · [15t/τ² - 10t³/τ⁴ + t⁵/τ⁶] · exp(-t²/(2τ²))
```

**归一化系数计算**：

为了确保脉冲能量 ∫|p_n(t)|² dt = 1，归一化系数为：

```
A_2 = sqrt(2/(√π · τ))
A_5 = sqrt(32/(√π · τ))
```

**证据来源**：
- M. Ghavami, L. B. Michael, R. Kohno, "Ultra Wideband Signals and Systems in Communication Engineering", Wiley, 2007, Chapter 3
- IEEE 802.15.4a UWB PHY 标准附录（高斯脉冲波形定义）

### 1.2 频域特性

高斯导数脉冲的频域表示通过傅里叶变换得到：

```
P_n(f) = F{p_n(t)} = A_n · (j2πf)^n · exp(-2π²f²τ²)
```

**功率谱密度（PSD）**：

```
S_n(f) = |P_n(f)|² = A_n² · (2πf)^(2n) · exp(-4π²f²τ²)
```

**关键特性**：
1. **峰值频率**：n 阶脉冲的 PSD 峰值频率约为 `f_peak ≈ sqrt(n) / (2πτ)`
2. **3dB 带宽**：B_3dB ≈ 1 / τ
3. **频谱中心频移**：阶数越高，频谱中心越向高频偏移

**参数选择指南**（满足 FCC 合规性）：

对于 5 阶脉冲，要使能量主要集中在 3.1-10.6 GHz 合法频段：
- **中心频率** fc ≈ 6-7 GHz（推荐 6.85 GHz）
- **脉冲宽度** τ ≈ 0.5 ns（对应 3dB 带宽约 2 GHz）
- **峰值频率** f_peak ≈ sqrt(5)/(2π·0.5ns) ≈ 710 MHz × 2.236 ≈ 1.6 GHz（需要通过 fc 调制到 6.85 GHz）

对于 2 阶脉冲：
- **脉冲宽度** τ ≈ 0.5 ns
- **峰值频率** f_peak ≈ sqrt(2)/(2π·0.5ns) ≈ 450 MHz（低频，违反 FCC 限制）

**决策**：
- ✅ **采用时域调制方式**：p_n(t) · cos(2πf_c·t) 将脉冲频谱搬移到中心频率 fc
- ✅ **5 阶脉冲使用 fc = 6.85 GHz**，确保频谱中心位于 3.1-10.6 GHz 中点
- ✅ **2 阶脉冲不调制（fc = 0）**，展示低频违规

**证据来源**：
- L. Yang, G. B. Giannakis, "Ultra-wideband communications: an idea whose time has come", IEEE Signal Processing Magazine, 2004
- S. Roy et al., "Ultrawideband radio design: the promise of high-speed, short-range wireless connectivity", Proceedings of the IEEE, 2004

### 1.3 FFT 参数选择

为满足 SC-002（频率分辨率 ≥10 MHz），需要确定：

**采样率（Sampling Rate）**：
- 根据奈奎斯特采样定理：f_s ≥ 2 · f_max
- 最高关注频率 f_max = 12 GHz
- **选择 f_s = 50 GHz**（现有项目配置，满足 2 × 12 GHz = 24 GHz）

**采样点数（N_samples）**：
- 频率分辨率 Δf = f_s / N_samples
- 要求 Δf ≤ 10 MHz
- 最小采样点数：N_min = f_s / Δf = 50 GHz / 10 MHz = 5000
- **决策**：使用 N_samples = 8192（2^13，FFT 高效计算）
- **实际分辨率**：Δf = 50 GHz / 8192 ≈ 6.1 MHz ✅（优于 10 MHz 要求）

**时间窗长度（Time Window）**：
- T_window = N_samples / f_s = 8192 / 50e9 ≈ 164 ns
- 确保至少包含 10 个脉冲宽度：10 × τ = 10 × 0.5ns = 5ns < 164ns ✅

**FFT 实现**：
- 使用 `numpy.fft.rfft()`（实数 FFT，避免负频率冗余）
- 输出频率范围：0 到 f_s/2 = 25 GHz（覆盖 0-12 GHz 需求）

**证据来源**：
- NumPy FFT 官方文档：https://numpy.org/doc/stable/reference/routines.fft.html
- Oppenheim, Schafer, "Discrete-Time Signal Processing", 3rd Edition, Section 8.6

---

## 2. FCC Part 15.209 标准的精确定义

### 2.1 室内辐射掩蔽罩（Indoor Mask）

**FCC Part 15.209 规定的 UWB 设备室内辐射限制**（CFR Title 47, Part 15.209）：

| 频率范围 | 功率限制（EIRP） | 单位 |
|----------|------------------|------|
| 0.96 - 1.61 GHz | -75.3 dBm/MHz | 等效辐射功率密度 |
| 1.61 - 3.1 GHz | -53.3 dBm/MHz | 等效辐射功率密度 |
| 3.1 - 10.6 GHz | -41.3 dBm/MHz | 等效辐射功率密度（UWB 合法频段） |
| >10.6 GHz | -51.3 dBm/MHz | 等效辐射功率密度 |

**关键说明**：
1. **EIRP（Effective Isotropic Radiated Power）**：等效全向辐射功率，考虑天线增益
2. **测量带宽**：1 MHz（RBW = 1 MHz）
3. **测量方法**：峰值检波（Peak Detector）
4. **适用场景**：室内通信、测距、成像等 UWB 应用

**边界频点处理**：
- FCC 标准在边界频点（如 3.1 GHz）采用**阶跃跳变**，非平滑过渡
- 实现时使用 `np.piecewise()` 或条件判断实现阶跃

**验证方法**：
- 对照 FCC 官方文档 CFR Title 47, Part 15.209 确认数值
- 误差容差：±0.1 dB（SC-001 要求）

**证据来源**：
- FCC Part 15.209: https://www.ecfr.gov/current/title-47/chapter-I/subchapter-A/part-15/subpart-C/section-15.209
- FCC Report and Order 02-48 (2002): First Report and Order on UWB Technology
- NTIA Special Publication 01-43: "Assessment of Compatibility between Ultrawideband (UWB) Systems and Global Positioning System (GPS) Receivers"

### 2.2 功率单位转换

**dBm/MHz 的物理意义**：

```
P_dBm/MHz = 10 · log10(P_watts / 1mW) - 10 · log10(BW_Hz / 1MHz)
```

简化公式：
```
P_dBm/MHz = P_dBm - 10·log10(BW_MHz)
```

**示例**：
- 如果脉冲在 1 MHz 带宽内功率为 -41.3 dBm
- 则功率谱密度为 -41.3 dBm/MHz

**PSD 计算中的归一化**：
1. 使用 FFT 计算得到频域幅度 |P(f)|
2. 计算功率：S(f) = |P(f)|² / (N · f_s)
3. 转换为 dBm/MHz：10·log10(S(f) / 1e-3) + 30

**证据来源**：
- IEEE Std 686-2017: "IEEE Standard for Radar Definitions"（功率谱密度定义）
- Parseval's theorem 验证（时域能量 = 频域能量）

### 2.3 合规性判断逻辑

**判定规则**：

脉冲 PSD 在所有频点 f ∈ [0, 12 GHz] 必须满足：

```
PSD_pulse(f) ≤ PSD_FCC_mask(f) + tolerance
```

其中 tolerance = 0.1 dB（边界容差）

**状态分类**：

1. **合规（Compliant）**：
   - 条件：max(PSD_pulse - PSD_FCC_mask) < -tolerance
   - 即所有频点低于 FCC 限制线至少 0.1 dB

2. **不合规（Non-compliant）**：
   - 条件：存在频点 f 使得 PSD_pulse(f) > PSD_FCC_mask(f) + tolerance
   - 违规频点：记录所有超限频点及超限量

3. **临界合规（Marginal Compliant）**：
   - 条件：max(PSD_pulse - PSD_FCC_mask) ∈ [-tolerance, +tolerance]
   - 即在容差范围内接触 FCC 限制线

**实现算法**（伪代码）：

```python
def check_compliance(psd_pulse, psd_fcc_mask, tolerance=0.1):
    diff = psd_pulse - psd_fcc_mask  # 单位：dB
    max_excess = np.max(diff)

    if max_excess < -tolerance:
        status = "Compliant"
    elif max_excess > tolerance:
        status = "Non-compliant"
        violations = [(freq, excess) for freq, excess in zip(frequencies, diff) if excess > tolerance]
    else:
        status = "Marginal Compliant"

    return ComplianceResult(status=status, max_excess=max_excess, violations=violations)
```

**证据来源**：
- FCC Part 15.35: "Measurement procedures"
- ANSI C63.10-2013: "American National Standard for Testing Unlicensed Wireless Devices"

---

## 3. matplotlib 学术标准配置

### 3.1 字体配置（中文宋体 + 英文 Times New Roman）

**配置代码**（遵循 Constitution X）：

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# 字体配置（按优先级顺序）
mpl.rcParams["font.serif"] = ["Times New Roman", "SimSun", "STSong"]
mpl.rcParams["font.sans-serif"] = ["Arial", "SimSun", "STSong"]
mpl.rcParams["font.family"] = "serif"  # 使用衬线字体

# 数学公式字体
mpl.rcParams["mathtext.fontset"] = "stix"  # STIX 字体用于数学符号
mpl.rcParams["mathtext.rm"] = "Times New Roman"
mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"

# 确保负号正常显示
mpl.rcParams["axes.unicode_minus"] = False

# DPI 设置（满足 SC-004）
mpl.rcParams["figure.dpi"] = 100  # 屏幕显示
mpl.rcParams["savefig.dpi"] = 300  # 保存图表（≥300 DPI）

# 字体大小（符合 IEEE 论文排版标准）
mpl.rcParams["font.size"] = 12         # 默认字体
mpl.rcParams["axes.titlesize"] = 14    # 标题字号
mpl.rcParams["axes.labelsize"] = 12    # 轴标签字号
mpl.rcParams["xtick.labelsize"] = 10   # x 轴刻度字号
mpl.rcParams["ytick.labelsize"] = 10   # y 轴刻度字号
mpl.rcParams["legend.fontsize"] = 10   # 图例字号
```

**字体可用性验证**：

```bash
# Linux 系统字体安装
sudo apt-get install fonts-liberation ttf-mscorefonts-installer

# 验证字体是否安装
fc-list | grep -i "times\|simsun"

# Python 中验证字体
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
print("Times New Roman" in fonts, "SimSun" in fonts)
```

**证据来源**：
- Matplotlib 字体配置官方文档：https://matplotlib.org/stable/users/explain/text/fonts.html
- Constitution X: 学术规范与可视化标准（constitution.md:158-212）

### 3.2 图表格式（IEEE 出版标准）

**图表尺寸**：

```python
# 单栏图（3.5 inch 宽度）
fig, ax = plt.subplots(figsize=(3.5, 2.625))  # 宽高比 4:3

# 双栏图（7 inch 宽度）
fig, ax = plt.subplots(figsize=(7, 5.25))  # 宽高比 4:3
```

**线条样式**：

```python
# 2 阶脉冲
ax.plot(freq, psd_2nd,
        color="blue", linestyle="-", linewidth=1.5,
        label="2 阶高斯导数脉冲")

# 5 阶脉冲
ax.plot(freq, psd_5th,
        color="green", linestyle="--", linewidth=1.5,
        label="5 阶高斯导数脉冲")

# FCC 掩蔽罩
ax.plot(freq, fcc_mask,
        color="red", linestyle=":", linewidth=2.0,
        label="FCC 室内掩蔽罩")
```

**网格线配置**：

```python
ax.grid(True, which="both", linestyle=":", alpha=0.3, linewidth=0.5)
```

**坐标轴标注**：

```python
ax.set_xlabel("频率 (GHz)", fontsize=12)
ax.set_ylabel("功率谱密度 (dBm/MHz)", fontsize=12)
ax.set_title("UWB 脉冲 PSD 与 FCC 合规性分析", fontsize=14, pad=10)
```

**图例位置**：

```python
ax.legend(loc="best", frameon=True, shadow=False, framealpha=0.9)
```

**保存格式**：

```python
fig.savefig("outputs/psd_fcc_compliance_2nd_5th.png",
            dpi=300, bbox_inches="tight", pad_inches=0.1)
```

**证据来源**：
- IEEE Author Digital Toolbox: https://ieeeauthorcenter.ieee.org/
- IEEE Graphics Guidelines（图表分辨率、字体、配色要求）
- Springer Nature Artwork and Media Guidelines

### 3.3 配色方案（色盲友好）

**ColorBrewer 推荐配色**：

```python
# 使用 ColorBrewer "Set1" 配色方案（色盲友好）
from matplotlib.colors import ListedColormap
colors = ["#377eb8", "#4daf4a", "#e41a1c"]  # 蓝、绿、红
```

**黑白打印验证**：

- 使用不同线型（实线、虚线、点线）+ 标记（o, s, ^）组合
- 确保在灰度打印时仍可区分

**证据来源**：
- ColorBrewer 2.0: https://colorbrewer2.org/
- Wong, B. (2011). "Points of view: Color blindness", Nature Methods, 8(6), 441.

---

## 4. 技术决策汇总

### 4.1 数据结构设计

**决策**：使用 NumPy 数组存储所有数值数据，避免 Python 列表的性能开销。

**理由**：
- NumPy 数组支持向量化运算，FFT 计算速度快 10-100 倍
- 与 matplotlib 直接兼容，无需数据类型转换
- 内存占用小（连续内存布局）

**替代方案**：Python 列表
**拒绝原因**：性能不足，无法满足 SC-005（< 10 秒）

### 4.2 脉冲生成方式

**决策**：使用解析公式直接生成时域波形，无需查找表。

**理由**：
- 脉冲生成时间 < 10 ms（远小于性能目标）
- 避免预计算查找表的存储和管理开销
- 支持任意参数调整（fc, τ）

**替代方案**：预计算查找表（LUT）
**拒绝原因**：YAGNI（You Aren't Gonna Need It），增加复杂性无性能收益

### 4.3 PSD 计算方法

**决策**：使用 `numpy.fft.rfft()` 计算单边 PSD。

**理由**：
- 实数信号的负频率部分冗余，rfft 减少 50% 计算量
- 符合 Parseval 定理（能量守恒）
- NumPy 内置实现，经过大规模生产验证

**替代方案**：Welch 方法（scipy.signal.welch）
**拒绝原因**：不必要的平滑处理，本项目需要精确频率分辨率

### 4.4 FCC 掩蔽罩实现

**决策**：使用 `np.piecewise()` 实现分段常数函数。

**理由**：
- FCC 标准为阶跃函数，np.piecewise 完美匹配
- 代码简洁，易于验证正确性
- 无性能瓶颈（掩蔽罩计算 < 1 ms）

**替代方案**：if-else 分支判断
**拒绝原因**：代码冗长，难以维护

---

## 5. 风险与缓解措施

### 5.1 字体缺失风险

**风险**：SimSun 或 Times New Roman 字体在某些 Linux 发行版中未预装。

**现象**：Matplotlib 生成图表时警告 "Glyph missing from current font"。

**缓解措施**：
1. 在 quickstart.md 中明确列出字体安装命令
2. 在代码中添加字体可用性检查，提前警告用户
3. 提供字体降级方案（使用 DejaVu Sans 作为后备）

**验证命令**：
```bash
python -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'Times' in f.name or 'SimSun' in f.name])"
```

### 5.2 FFT 精度风险

**风险**：FFT 计算中的浮点误差累积导致 PSD 精度不足。

**缓解措施**：
1. 使用 `numpy.float64`（双精度浮点数），避免 float32
2. 在单元测试中验证 Parseval 定理（时域能量 = 频域能量）
3. 对比解析解（如 2 阶高斯脉冲的解析 PSD 公式）

### 5.3 边界容差敏感性

**风险**：tolerance = 0.1 dB 过于严格，数值误差可能导致误判。

**缓解措施**：
1. 在 contracts/compliance.md 中明确 tolerance 参数可调
2. 在单元测试中验证不同 tolerance 值的行为
3. 在可视化图表中绘制 ±tolerance 容差带

---

## 6. 性能预测

### 6.1 计算复杂度分析

**脉冲生成**：
- 时间复杂度：O(N)，N = 采样点数（~8192）
- 预计时间：< 10 ms

**FFT 计算**：
- 时间复杂度：O(N log N)
- 预计时间：< 100 ms（numpy.fft.rfft 高度优化）

**FCC 掩蔽罩生成**：
- 时间复杂度：O(N)
- 预计时间：< 1 ms

**合规性判断**：
- 时间复杂度：O(N)
- 预计时间：< 1 ms

**图表绘制**：
- 预计时间：< 1 秒（matplotlib 渲染 + PNG 保存）

**总计**：< 2 秒（远优于 SC-005 的 < 10 秒要求）✅

### 6.2 内存占用预测

**数据规模**：
- 时域波形：8192 × 8 bytes = 65 KB
- 频域 PSD：4096 × 8 bytes = 32 KB
- FCC 掩蔽罩：4096 × 8 bytes = 32 KB
- 图表缓冲：~5 MB（matplotlib 内存占用）

**总计**：< 10 MB（可忽略）✅

---

## 7. 文献引用

### 7.1 学术文献

1. M. Ghavami, L. B. Michael, R. Kohno, "Ultra Wideband Signals and Systems in Communication Engineering", Wiley, 2007
2. L. Yang, G. B. Giannakis, "Ultra-wideband communications: an idea whose time has come", IEEE Signal Processing Magazine, 21(6), 26-54, 2004
3. S. Roy et al., "Ultrawideband radio design: the promise of high-speed, short-range wireless connectivity", Proceedings of the IEEE, 92(2), 295-311, 2004

### 7.2 标准文档

1. FCC Part 15.209 (CFR Title 47): "Radiated emission limits; general requirements"
2. FCC Report and Order 02-48 (2002): "First Report and Order on Ultra-Wideband Technology"
3. IEEE 802.15.4a-2007: "Wireless Medium Access Control (MAC) and Physical Layer (PHY) Specifications for Low-Rate Wireless Personal Area Networks (WPANs): Amendment 1: Add Alternate PHYs"
4. ANSI C63.10-2013: "American National Standard for Testing Unlicensed Wireless Devices"

### 7.3 技术文档

1. NumPy FFT Documentation: https://numpy.org/doc/stable/reference/routines.fft.html
2. Matplotlib Font Configuration: https://matplotlib.org/stable/users/explain/text/fonts.html
3. IEEE Author Digital Toolbox: https://ieeeauthorcenter.ieee.org/

---

## 8. 下一步行动

**Research 阶段完成 ✅**

所有 4 个研究主题已完成：
- [x] 高斯导数脉冲的 PSD 计算方法
- [x] FCC Part 15.209 标准的精确定义
- [x] 合规性判断算法
- [x] matplotlib 学术标准配置

**进入 Phase 1：Design & Contracts**

下一步任务：
1. 生成 data-model.md（4 个实体的详细设计）
2. 生成 contracts/ 目录（4 个契约文件）
3. 生成 quickstart.md（用户指南）
4. 运行 agent context 更新脚本

**无遗留 "NEEDS CLARIFICATION"** - 所有技术问题已解决 ✅

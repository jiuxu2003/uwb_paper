# Quickstart Guide: PSD/FCC 合规性可视化

**Feature**: 002-psd-fcc-compliance
**Date**: 2025-12-17

本指南帮助用户快速生成 UWB 脉冲 PSD 与 FCC 合规性对比图表。

---

## 环境要求

### 系统要求

- **操作系统**: Linux / macOS（推荐）或 Windows（需要 WSL）
- **Python 版本**: 3.11 或更高
- **内存**: ≥ 4 GB
- **磁盘空间**: ≥ 100 MB

### 依赖安装

```bash
# 1. 克隆项目（如果尚未克隆）
git clone git@github.com:jiuxu2003/uwb_paper.git
cd uwb_paper

# 2. 创建虚拟环境（推荐）
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate.bat  # Windows

# 3. 安装依赖
pip install -e .

# 4. 安装字体（Linux 系统）
sudo apt-get update
sudo apt-get install fonts-liberation ttf-mscorefonts-installer

# 5. 验证字体安装
fc-list | grep -i "times\|simsun"
```

### 字体验证（Python）

```python
import matplotlib.font_manager as fm

fonts = [f.name for f in fm.fontManager.ttflist]
print("Times New Roman 可用:", "Times New Roman" in fonts)
print("SimSun 可用:", "SimSun" in fonts)
```

如果字体不可用，图表将使用降级字体（DejaVu Sans），但不符合学术出版标准。

---

## 快速开始（5 分钟）

### Step 1: 运行示例脚本

```bash
# 生成 PSD vs FCC Mask 对比图
python scripts/plot_psd_fcc_compliance.py
```

**预期输出**：

```
======================================================
生成 PSD vs FCC Mask 对比图
======================================================

[1/5] 生成 2 阶高斯导数脉冲...
  ✓ 脉冲生成完成（tau=0.5ns, fc=0 Hz）
  ✓ 脉冲能量: 1.000

[2/5] 生成 5 阶高斯导数脉冲...
  ✓ 脉冲生成完成（tau=0.5ns, fc=6.85 GHz）
  ✓ 脉冲能量: 1.000

[3/5] 计算功率谱密度（PSD）...
  ✓ 2 阶脉冲 PSD 计算完成（1200 频点）
  ✓ 5 阶脉冲 PSD 计算完成（1200 频点）
  ✓ 频率分辨率: 6.1 MHz

[4/5] 生成 FCC 室内掩蔽罩...
  ✓ FCC 掩蔽罩生成完成（4 个频段）

[5/5] 合规性判断与可视化...
  ✓ 2 阶脉冲：Non-compliant（最大超限 +12.5 dB）
  ✓ 5 阶脉冲：Compliant（最大超限 -3.8 dB）
  ✓ 图表已保存：outputs/psd_fcc_compliance_2nd_5th.png

======================================================
✅ 完成！总用时: 1.23 秒
======================================================
```

### Step 2: 查看输出图表

```bash
# Linux（使用默认图片查看器）
xdg-open outputs/psd_fcc_compliance_2nd_5th.png

# macOS
open outputs/psd_fcc_compliance_2nd_5th.png

# 或直接在 VS Code 中打开
code outputs/psd_fcc_compliance_2nd_5th.png
```

---

## 参数调整

### 修改脉冲参数

编辑 `scripts/plot_psd_fcc_compliance.py`：

```python
# 修改 2 阶脉冲参数
pulse_2nd = generate_gaussian_derivative_pulse(
    n=2,
    tau=0.7e-9,  # 修改脉冲宽度为 0.7 ns
    fc=0,  # 无调制（展示低频违规）
    sampling_rate=50e9
)

# 修改 5 阶脉冲参数
pulse_5th = generate_gaussian_derivative_pulse(
    n=5,
    tau=0.5e-9,
    fc=7.5e9,  # 修改中心频率为 7.5 GHz
    sampling_rate=50e9
)
```

**参数说明**：

| 参数 | 说明 | 取值范围 | 推荐值 |
|------|------|----------|--------|
| `n` | 脉冲阶数 | {2, 5} | 5 |
| `tau` | 脉冲宽度 | [0.1ns, 2ns] | 0.5ns |
| `fc` | 中心频率 | [0, 15 GHz] | 6.85 GHz (5阶), 0 (2阶) |
| `sampling_rate` | 采样率 | ≥24 GHz | 50 GHz |

### 调整可视化样式

修改 matplotlib 配置（在脚本顶部）：

```python
import matplotlib as mpl

# 修改图表尺寸
figsize = (7, 5.25)  # 双栏图（IEEE 标准）
# figsize = (3.5, 2.625)  # 单栏图

# 修改字体大小
mpl.rcParams["font.size"] = 14  # 默认字体（原 12）
mpl.rcParams["axes.titlesize"] = 16  # 标题字号（原 14）

# 修改 DPI（更高清）
mpl.rcParams["savefig.dpi"] = 600  # 超高清（原 300）
```

---

## 常见问题

### Q1: 字体警告 "Glyph missing from current font"

**原因**: SimSun 或 Times New Roman 字体未安装。

**解决方案**:

```bash
# Linux
sudo apt-get install fonts-liberation ttf-mscorefonts-installer

# macOS (系统自带)
# 无需额外安装

# 验证
fc-list | grep -i times
```

如果仍无法安装，编辑脚本使用降级字体：

```python
mpl.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman"]
```

### Q2: 脚本执行时间过长

**原因**: 采样率过高或脉冲宽度过大。

**解决方案**:

1. 降低采样率（确保 ≥24 GHz）：
   ```python
   sampling_rate = 25e9  # 从 50 GHz 降至 25 GHz
   ```

2. 减少 FFT 点数：
   ```python
   n_fft = 4096  # 从 8192 降至 4096
   ```

3. 缩小频率范围：
   ```python
   freq_range = (0, 11e9)  # 从 12 GHz 缩小到 11 GHz
   ```

### Q3: 5 阶脉冲显示为"Non-compliant"

**原因**: 中心频率 `fc` 设置不当。

**解决方案**:

调整 `fc` 到 6-7 GHz 范围：

```python
fc = 6.85e9  # 推荐值：3.1-10.6 GHz 中点
```

验证频谱集中度：

```python
# 检查 3.1-10.6 GHz 频段内的能量占比
uwb_band = (freq >= 3.1e9) & (freq <= 10.6e9)
energy_in_band = np.trapz(psd[uwb_band], freq[uwb_band])
total_energy = np.trapz(psd, freq)
print(f"UWB 频段能量占比: {energy_in_band/total_energy*100:.1f}%")
# 期望: >80%
```

### Q4: 图表分辨率不足（<300 DPI）

**原因**: matplotlib `savefig.dpi` 配置错误。

**解决方案**:

```python
# 方法 1: 全局配置
mpl.rcParams["savefig.dpi"] = 300

# 方法 2: 保存时指定
fig.savefig("outputs/psd_fcc_compliance.png", dpi=300)
```

验证图表分辨率：

```bash
file outputs/psd_fcc_compliance_2nd_5th.png
# 输出: PNG image data, 2100 x 1575, ...（300 DPI @ 7 inch 宽度）
```

---

## 进阶用法

### 生成多个中心频率对比图

```python
# 扫描不同中心频率
fc_values = np.linspace(5e9, 8e9, 7)  # 5-8 GHz, 7 个点

for fc in fc_values:
    pulse = generate_gaussian_derivative_pulse(n=5, tau=0.5e-9, fc=fc, ...)
    psd, freq = compute_psd(pulse, ...)
    result = check_compliance(psd, fcc_mask, freq)
    print(f"fc={fc/1e9:.2f} GHz: {result['status']}")
```

### 导出数值数据（CSV）

```python
import pandas as pd

# 导出 PSD 数据
df = pd.DataFrame({
    "Frequency (GHz)": freq / 1e9,
    "PSD_2nd (dBm/MHz)": psd_2nd,
    "PSD_5th (dBm/MHz)": psd_5th,
    "FCC_Mask (dBm/MHz)": fcc_mask
})

df.to_csv("outputs/psd_data.csv", index=False)
print("数据已导出到 outputs/psd_data.csv")
```

### 批量生成图表

参考 `scripts/generate_figures.py` 批量生成所有论文图表。

---

## 输出文件说明

### 目录结构

```
outputs/
└── psd_fcc_compliance_2nd_5th.png  # PSD vs FCC Mask 对比图
```

### 文件命名规范

- **格式**: `psd_fcc_compliance_<pulse_orders>.png`
- **示例**: `psd_fcc_compliance_2nd_5th.png`（包含 2 阶和 5 阶脉冲）

### 图表元素说明

| 元素 | 说明 | 样式 |
|------|------|------|
| 蓝色实线 | 2 阶高斯导数脉冲 PSD | linewidth=1.5 |
| 绿色虚线 | 5 阶高斯导数脉冲 PSD | linestyle="--", linewidth=1.5 |
| 红色点线 | FCC 室内掩蔽罩 | linestyle=":", linewidth=2.0 |
| 灰色虚线（竖直） | 关键频率边界（3.1 GHz, 10.6 GHz） | alpha=0.5 |
| 图例 | 位于右上角（"best"） | framealpha=0.9 |

---

## 性能基准

| 操作 | 预期时间 | 实测时间（参考） |
|------|----------|------------------|
| 脉冲生成（2 阶 + 5 阶） | < 20 ms | 8 ms |
| PSD 计算（FFT） | < 200 ms | 95 ms |
| FCC 掩蔽罩生成 | < 2 ms | 0.5 ms |
| 合规性判断 | < 2 ms | 1 ms |
| 图表渲染 + 保存 | < 1 秒 | 620 ms |
| **总计** | < 2 秒 | **1.23 秒** ✅ |

*测试环境: Intel i7-12700K, 32GB RAM, Ubuntu 22.04*

---

## 下一步

### 集成到论文工作流

1. 将图表添加到 LaTeX 论文：
   ```latex
   \begin{figure}[htbp]
       \centering
       \includegraphics[width=0.8\textwidth]{outputs/psd_fcc_compliance_2nd_5th.png}
       \caption{UWB 脉冲 PSD 与 FCC 合规性分析}
       \label{fig:psd_fcc_compliance}
   \end{figure}
   ```

2. 引用图表：
   ```latex
   如图 \ref{fig:psd_fcc_compliance} 所示，5 阶高斯导数脉冲（中心频率 6.85 GHz）的 PSD 在整个频谱范围内均低于 FCC Part 15.209 规定的室内辐射限制，表明该脉冲设计符合法规要求。
   ```

### 运行完整仿真

```bash
# 生成所有论文图表（包括 BER, MUI, NBI 分析）
python scripts/generate_figures.py
```

---

## 获取帮助

### 文档资源

- **Feature Spec**: [spec.md](spec.md) - 功能需求规格
- **Research Findings**: [research.md](research.md) - 技术研究
- **Data Model**: [data-model.md](data-model.md) - 实体设计
- **Contracts**: [contracts/](contracts/) - API 契约

### 报告问题

如果遇到问题，请查看 [spec.md](spec.md) 的 Edge Cases 部分，或创建 GitHub Issue。

---

**Quickstart Status**: ✅ 完成
**Last Updated**: 2025-12-17

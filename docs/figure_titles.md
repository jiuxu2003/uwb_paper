# 图表名称与标题对应表

本文档记录论文中所有图表的文件名与对应标题，便于在论文中正确引用。

## 说明

- 图表文件位于 `outputs/` 目录下
- 所有图表均为 300 DPI 高分辨率 PNG 格式，符合学术出版要求
- 图表中已移除标题，标题应在论文正文或图题（caption）中添加

---

## 1. 波形演示图

### 文件名
`outputs/waveform_demo.png`

### 标题（中文）
TH-UWB 信号时域波形

### 标题（英文）
Time-Domain Waveform of TH-UWB Signal

### 描述
展示TH-PPM调制的超宽带脉冲时域波形，演示帧结构、时隙跳变和脉冲位置调制特性。

---

## 2. 多用户干扰性能分析

### 文件名
`outputs/ber_vs_users.png`

### 标题（中文）
多用户干扰性能分析

### 标题（英文）
Multi-User Interference Performance Analysis

### 描述
展示误码率（BER）随用户数量增加的变化趋势，验证系统在多用户场景下的性能退化特性。纵轴使用对数坐标。

### 测试条件
- SNR = 10 dB
- 用户数量：1, 2, 3, 5, 7, 10
- 比特数：10,000

---

## 3. 窄带干扰抑制性能分析（快速演示版）

### 文件名
`outputs/ber_vs_sir_demo.png`

### 标题（中文）
窄带干扰抑制性能分析（演示）

### 标题（英文）
Narrowband Interference Suppression Performance Analysis (Demo)

### 描述
展示误码率（BER）随信干比（SIR）变化的性能曲线，验证UWB系统的抗窄带干扰能力。纵轴使用对数坐标。此为快速演示版本（1000比特）。

### 测试条件
- SNR = 10 dB
- SIR 范围：-10 dB 到 -40 dB（16个点）
- 干扰频率：1.611 GHz（脉冲峰值频率）
- 比特数：1,000

### 关键结果
- 干扰容限阈值：约 -28 dB（631倍功率差）
- SIR=-26dB 时仍能零误码（398倍干扰）
- SIR=-40dB 时 BER=37.1%（接近随机猜测）

---

## 4. 窄带干扰抑制性能分析（完整版）

### 文件名
`outputs/ber_vs_sir.png`

### 标题（中文）
窄带干扰抑制性能分析

### 标题（英文）
Narrowband Interference Suppression Performance Analysis

### 描述
展示误码率（BER）随信干比（SIR）变化的性能曲线，验证UWB系统的抗窄带干扰能力。纵轴使用对数坐标。此为完整仿真版本（10000比特），适合论文发表。

### 测试条件
- SNR = 10 dB
- SIR 范围：-10 dB 到 -40 dB（16个点）
- 干扰频率：1.611 GHz（脉冲峰值频率）
- 比特数：10,000

### 关键结果
- 干扰容限阈值：约 -28 dB（631倍功率差，BER=0.50%）
- SIR=-26dB 时 BER=0（完美抗干扰）
- SIR=-30dB 时 BER=5.87%
- SIR=-40dB 时 BER=38.72%
- 95% 置信区间已计算

---

## 5. UWB脉冲PSD与FCC合规性分析（2阶导数）

### 文件名
`outputs/psd_fcc_compliance_2nd_order.png`

### 标题（中文）
二阶高斯导数脉冲PSD与FCC室内掩蔽罩对比

### 标题（英文）
2nd-Order Gaussian Derivative Pulse PSD vs. FCC Indoor Mask

### 描述
展示二阶高斯导数脉冲的功率谱密度（PSD）与FCC室内辐射掩蔽罩的对比，验证脉冲设计的FCC合规性。

### 测试条件
- 脉冲类型：二阶高斯导数（Mexican Hat Wavelet）
- 脉冲宽度参数：τ = 0.5 ns
- 频率范围：0-12 GHz

---

## 6. UWB脉冲PSD与FCC合规性分析（5阶导数）

### 文件名
`outputs/psd_fcc_compliance_5th_order.png`

### 标题（中文）
五阶高斯导数脉冲PSD与FCC室内掩蔽罩对比

### 标题（英文）
5th-Order Gaussian Derivative Pulse PSD vs. FCC Indoor Mask

### 描述
展示五阶高斯导数脉冲的功率谱密度（PSD）与FCC室内辐射掩蔽罩的对比，比较不同阶数脉冲的频谱特性。

### 测试条件
- 脉冲类型：五阶高斯导数
- 脉冲宽度参数：τ = 0.5 ns
- 频率范围：0-12 GHz

---

## 7. 脉冲频谱诊断图

### 文件名
`outputs/pulse_spectrum_diagnosis.png`

### 标题（中文）
二阶高斯导数脉冲的时域与频域特性

### 标题（英文）
Time-Domain and Frequency-Domain Characteristics of 2nd-Order Gaussian Derivative Pulse

### 描述
双子图展示：
- **上图**：脉冲时域波形（τ=0.5 ns）
- **下图**：脉冲归一化频谱（0-10 GHz）

标注了峰值频率（1.611 GHz）、当前NBI频率（2.4 GHz）和-3dB带宽。

### 用途
- 频谱分析与诊断
- NBI频率选择依据
- 理解干扰"命中"原理

---

## 图表引用建议

### LaTeX 示例（单图）

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\columnwidth]{outputs/ber_vs_sir.png}
  \caption{窄带干扰抑制性能分析。纵轴为误码率（对数坐标），横轴为信干比。测试条件：SNR=10dB，干扰频率1.611 GHz，10000比特。}
  \label{fig:ber_vs_sir}
\end{figure}
```

### LaTeX 示例（双栏并排）

```latex
\begin{figure*}[htbp]
  \centering
  \subfloat[二阶高斯导数脉冲]{
    \includegraphics[width=0.48\textwidth]{outputs/psd_fcc_compliance_2nd_order.png}
    \label{fig:psd_2nd}
  }
  \hfill
  \subfloat[五阶高斯导数脉冲]{
    \includegraphics[width=0.48\textwidth]{outputs/psd_fcc_compliance_5th_order.png}
    \label{fig:psd_5th}
  }
  \caption{不同阶数高斯导数脉冲的PSD与FCC室内掩蔽罩对比}
  \label{fig:psd_comparison}
\end{figure*}
```

---

## 图表生成脚本

各图表的生成脚本如下：

| 图表文件 | 生成脚本 | 用途 |
|---------|---------|------|
| `waveform_demo.png` | `scripts/demo_waveform.py` | 波形演示 |
| `ber_vs_users.png` | `scripts/run_mui_analysis.py` | 多用户干扰完整仿真 |
| `ber_vs_sir_demo.png` | `scripts/demo_nbi_analysis.py` | NBI快速演示（1000比特） |
| `ber_vs_sir.png` | `scripts/run_nbi_analysis.py` | NBI完整仿真（10000比特） |
| `psd_fcc_compliance_*.png` | `scripts/run_psd_analysis.py` | FCC合规性分析 |
| `pulse_spectrum_diagnosis.png` | `scripts/diagnose_pulse_spectrum.py` | 频谱诊断 |

---

## 注意事项

1. **论文中的图题**：请在论文的图题（caption）中添加完整的标题、描述和测试条件
2. **交叉引用**：使用 `\ref{fig:xxx}` 进行图表交叉引用
3. **分辨率**：所有图表均为 300 DPI，满足 IEEE/Springer/Elsevier 等学术出版社要求
4. **字体**：图表使用 Times New Roman（英文）+ Noto Serif CJK JP（中文）衬线字体
5. **色彩**：所有配色方案均为色盲友好（colorblind-friendly）

---

**文档版本**：1.0
**最后更新**：2025-12-17
**更新人**：Claude (Sonnet 4.5)

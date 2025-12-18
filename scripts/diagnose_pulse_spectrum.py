#!/usr/bin/env python3
"""
诊断脚本：分析UWB脉冲的频谱特性

目的：确定脉冲的中心频率和有效频带，以便选择合适的干扰频率
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pulse import Pulse
from src.config import SystemConfig
from src.visualization import config as viz_config

def main():
    print("=" * 60)
    print("诊断：UWB脉冲频谱分析")
    print("=" * 60)

    # 1. 配置系统参数（与demo_nbi_analysis.py一致）
    print("\n[1/3] 配置系统参数...")
    sys_config = SystemConfig(
        frame_period=100e-9,
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,  # 关键参数
        pulse_amplitude=1.0,
        sampling_rate=50e9,  # 当前采样率
        num_bits=1000,
        random_seed=42,
    )
    print(f"  ✓ 脉冲宽度参数 τ: {sys_config.pulse_tau*1e9:.2f} ns")
    print(f"  ✓ 采样率: {sys_config.sampling_rate/1e9:.1f} GHz")

    # 2. 生成脉冲并计算频谱
    print("\n[2/3] 生成脉冲并计算频谱...")
    pulse = Pulse.generate(sys_config)
    print(f"  ✓ 脉冲持续时间: {pulse.duration*1e9:.2f} ns")
    print(f"  ✓ 采样点数: {len(pulse.waveform)}")

    # FFT计算频谱
    n_fft = len(pulse.waveform)
    # 零填充到更高分辨率
    n_fft_padded = 2048
    fft_result = np.fft.fft(pulse.waveform, n=n_fft_padded)
    frequencies = np.fft.fftfreq(n_fft_padded, d=1/sys_config.sampling_rate)

    # 只取正频率部分
    positive_freq_mask = frequencies >= 0
    frequencies = frequencies[positive_freq_mask]
    fft_magnitude = np.abs(fft_result[positive_freq_mask])

    # 归一化到最大值为1
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)

    # 转换为dB
    fft_db = 20 * np.log10(fft_magnitude + 1e-10)  # 避免log(0)

    # 3. 分析频谱特性
    print("\n[3/3] 分析频谱特性...")

    # 找到峰值频率
    peak_idx = np.argmax(fft_magnitude)
    peak_freq = frequencies[peak_idx]
    print(f"  ✓ 峰值频率: {peak_freq/1e9:.3f} GHz")

    # 找到-3dB带宽
    threshold_3db = np.max(fft_magnitude) / np.sqrt(2)  # -3dB = 0.707
    above_threshold = fft_magnitude > threshold_3db
    freq_above = frequencies[above_threshold]
    if len(freq_above) > 0:
        f_low = freq_above[0]
        f_high = freq_above[-1]
        bandwidth_3db = f_high - f_low
        print(f"  ✓ -3dB 带宽: {bandwidth_3db/1e9:.3f} GHz ({f_low/1e9:.3f} ~ {f_high/1e9:.3f} GHz)")

    # 找到-10dB带宽（90%能量）
    threshold_10db = np.max(fft_magnitude) / np.sqrt(10)  # -10dB = 0.316
    above_threshold_10db = fft_magnitude > threshold_10db
    freq_above_10db = frequencies[above_threshold_10db]
    if len(freq_above_10db) > 0:
        f_low_10db = freq_above_10db[0]
        f_high_10db = freq_above_10db[-1]
        bandwidth_10db = f_high_10db - f_low_10db
        print(f"  ✓ -10dB 带宽: {bandwidth_10db/1e9:.3f} GHz ({f_low_10db/1e9:.3f} ~ {f_high_10db/1e9:.3f} GHz)")

    # 当前干扰频率位置
    current_nbi_freq = 2.4e9
    current_nbi_idx = np.argmin(np.abs(frequencies - current_nbi_freq))
    current_nbi_level = fft_db[current_nbi_idx]
    print(f"\n  [分析] 当前NBI频率 (2.4 GHz) 在频谱中的位置:")
    print(f"    频率: {frequencies[current_nbi_idx]/1e9:.3f} GHz")
    print(f"    相对幅度: {current_nbi_level:.1f} dB (相对于峰值)")

    if current_nbi_level < -10:
        print(f"    ⚠️  警告: 2.4 GHz 位于脉冲频谱的弱区域 ({current_nbi_level:.1f} dB)")
        print(f"    建议: 将NBI频率调整到峰值频率附近 ({peak_freq/1e9:.3f} GHz)")
    else:
        print(f"    ✓ 2.4 GHz 位于脉冲频谱的有效区域")

    # 4. 绘制频谱图
    print("\n[4/4] 绘制频谱图...")
    viz_config.setup_academic_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 子图1: 时域波形
    ax1.plot(pulse.time_axis * 1e9, pulse.waveform, linewidth=1.5)
    ax1.set_xlabel("时间 (ns)")
    ax1.set_ylabel("幅度")
    # ax1.set_title(f"二阶高斯导数脉冲 (τ={sys_config.pulse_tau*1e9:.2f} ns)")  # 标题已移除（用于论文）
    ax1.grid(True, alpha=0.3, linestyle="--")

    # 子图2: 频谱（0-10 GHz）
    freq_max = 10e9  # 显示到10 GHz
    freq_mask = frequencies <= freq_max
    ax2.plot(frequencies[freq_mask]/1e9, fft_db[freq_mask], linewidth=1.5, label="脉冲频谱")

    # 标记峰值频率
    ax2.axvline(peak_freq/1e9, color='red', linestyle='--', alpha=0.7,
                label=f'峰值频率 ({peak_freq/1e9:.3f} GHz)')

    # 标记当前NBI频率
    ax2.axvline(current_nbi_freq/1e9, color='green', linestyle='--', alpha=0.7,
                label=f'当前NBI ({current_nbi_freq/1e9:.1f} GHz, {current_nbi_level:.1f} dB)')

    # 标记-3dB带宽
    if len(freq_above) > 0:
        ax2.axhline(-3, color='orange', linestyle=':', alpha=0.5, label='-3 dB')
        ax2.axvspan(f_low/1e9, f_high/1e9, alpha=0.1, color='orange')

    ax2.set_xlabel("频率 (GHz)")
    ax2.set_ylabel("幅度 (dB)")
    # ax2.set_title("脉冲频谱（归一化）")  # 标题已移除（用于论文）
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(-40, 5)  # 显示-40dB到峰值

    plt.tight_layout()

    output_path = Path("outputs/pulse_spectrum_diagnosis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 频谱图已保存到 {output_path}")

    plt.close()

    # 5. 输出建议
    print("\n" + "=" * 60)
    print("诊断建议:")
    print("-" * 60)
    if current_nbi_level < -10:
        print(f"  1. 当前NBI频率 (2.4 GHz) 偏离脉冲峰值频率 ({peak_freq/1e9:.3f} GHz)")
        print(f"     干扰在频谱中的强度仅为 {current_nbi_level:.1f} dB（相对于峰值）")
        print(f"     建议: 将 nbi_frequency 改为 {peak_freq:.2e} Hz ({peak_freq/1e9:.3f} GHz)")
    else:
        print(f"  1. NBI频率位置合理")

    print(f"\n  2. 采样率建议:")
    print(f"     当前采样率: {sys_config.sampling_rate/1e9:.1f} GHz")
    if f_high_10db > sys_config.sampling_rate / 4:
        suggested_fs = f_high_10db * 4
        print(f"     警告: 脉冲高频成分 ({f_high_10db/1e9:.3f} GHz) 接近Nyquist频率")
        print(f"     建议: 提高采样率到至少 {suggested_fs/1e9:.1f} GHz")
    else:
        print(f"     ✓ 采样率充足（Nyquist频率: {sys_config.sampling_rate/2/1e9:.1f} GHz）")

    print(f"\n  3. 相关接收机对NBI的抑制:")
    print(f"     相关接收机本质上是匹配滤波器，对窄带干扰有天然抑制能力")
    print(f"     即使NBI频率在频谱内，如果NBI带宽远小于信号带宽，")
    print(f"     相关运算的积分效应会显著降低干扰影响（类似于处理增益）")
    print(f"     这可能是BER一直为0的主要原因")

    print("=" * 60)
    print("\n✅ 诊断完成！")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
演示：生成 TH-PPM 信号时域波形

快速验证脚本，生成单用户的 TH-UWB 信号时域波形图（只生成 3 帧用于可视化）。

用法:
    python scripts/demo_waveform.py

输出:
    outputs/waveform_demo.png - 时域波形图（300 DPI，适合论文打印）

参考:
    - quickstart.md Step 1: 生成单用户信号波形图
    - 预期执行时间: < 1 秒
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pulse import Pulse
from src.models.modulation import User
from src.config import SystemConfig
from src.visualization.waveform import plot_waveform


def main():
    """主函数：生成波形图演示"""
    print("=" * 60)
    print("演示：TH-PPM 信号时域波形生成")
    print("=" * 60)

    # 1. 配置系统参数
    print("\n[1/4] 配置系统参数...")
    config = SystemConfig(
        frame_period=100e-9,  # 100 ns 帧周期
        slot_width=10e-9,  # 10 ns 时隙宽度
        num_slots=8,  # 8 个时隙/帧
        ppm_delay=5e-9,  # δ = Tc/2 = 5 ns
        pulse_tau=0.5e-9,  # 脉冲宽度 0.5 ns
        pulse_amplitude=1.0,  # 归一化幅度
        sampling_rate=50e9,  # 50 GHz 采样率
        num_bits=3,  # 只生成 3 帧用于可视化
        random_seed=42,  # 确保可重复性
    )
    print(f"  ✓ 帧周期: {config.frame_period*1e9:.1f} ns")
    print(f"  ✓ 采样率: {config.sampling_rate/1e9:.1f} GHz")
    print(f"  ✓ 帧数: {config.num_bits}")

    # 2. 生成脉冲模板
    print("\n[2/4] 生成脉冲模板...")
    pulse = Pulse.generate(config)
    print(f"  ✓ 脉冲持续时间: {pulse.duration*1e9:.2f} ns")
    print(f"  ✓ 脉冲采样点数: {len(pulse.waveform)}")
    print(f"  ✓ 脉冲能量: {pulse.energy:.2e} J")

    # 3. 创建用户并生成信号
    print("\n[3/4] 创建用户并生成信号...")
    user = User.create(user_id=0, config=config)
    print(f"  ✓ 用户 ID: {user.user_id}")
    print(f"  ✓ 跳时码（前3个）: {user.th_code.code[:3]}")
    print(f"  ✓ 数据比特（前3个）: {user.data_bits[:3]}")

    signal = user.generate_signal(pulse)
    print(f"  ✓ 信号长度: {len(signal)} 采样点")
    print(f"  ✓ 信号功率: {np.mean(signal**2):.6e}")
    print(f"  ✓ 信号峰值: {np.max(np.abs(signal)):.3f}")

    # 4. 可视化并保存
    print("\n[4/4] 生成波形图...")
    time_axis = np.arange(len(signal)) / config.sampling_rate

    plot_waveform(
        time_axis=time_axis,
        signal=signal,
        title=f"TH-UWB 信号时域波形（{config.num_bits} 帧）",
        xlabel="时间 (ns)",
        ylabel="幅度",
        save_path="outputs/waveform_demo.png",
        show=False,  # 无头环境不显示窗口
    )

    # 打印信号特征
    print("\n" + "=" * 60)
    print("波形特征分析:")
    print("-" * 60)
    print(f"  总时长: {time_axis[-1]*1e9:.2f} ns")
    print(f"  采样点数: {len(signal)}")
    print(f"  信号功率: {np.mean(signal**2):.6e}")
    print(f"  信噪比（理论无噪声）: inf dB")
    print(f"  每帧可见 1 个脉冲（位于跳时时隙 + PPM 偏移）")
    print("=" * 60)

    print("\n✅ 演示完成！图表已保存到 outputs/waveform_demo.png")
    print("   提示：可使用图片查看器打开，或在论文中引用（≥300 DPI）")


if __name__ == "__main__":
    main()

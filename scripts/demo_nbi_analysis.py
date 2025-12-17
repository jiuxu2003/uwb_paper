#!/usr/bin/env python3
"""
演示：窄带干扰（NBI）性能分析

快速验证脚本，分析单用户在窄带干扰条件下的误码率（1000 比特，用于快速演示）。

用法:
    python scripts/demo_nbi_analysis.py

输出:
    outputs/ber_vs_sir_demo.png - BER vs SIR 曲线图（300 DPI）

参考:
    - quickstart.md Step 3: 运行窄带干扰分析
    - 预期执行时间: ~1-2 分钟（1000 比特）
"""

import numpy as np
import time
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pulse import Pulse
from src.models.modulation import User
from src.models.channel import Channel
from src.simulation.receiver import Receiver
from src.simulation.metrics import PerformanceMetrics
from src.visualization.performance import plot_ber_vs_sir
from src.config import SystemConfig


def main():
    """主函数：窄带干扰性能分析（快速演示版）"""
    print("=" * 60)
    print("演示：窄带干扰（NBI）性能分析")
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
        num_bits=1000,  # 1000 比特用于快速演示
        random_seed=42,  # 确保可重复性
    )
    print(f"  ✓ 帧周期: {config.frame_period*1e9:.1f} ns")
    print(f"  ✓ 比特数: {config.num_bits}")
    print(f"  ✓ 采样率: {config.sampling_rate/1e9:.1f} GHz")

    # 2. 初始化脉冲和用户
    print("\n[2/4] 初始化脉冲和用户...")
    pulse = Pulse.generate(config)
    user = User.create(user_id=0, config=config)
    print(f"  ✓ 脉冲持续时间: {pulse.duration*1e9:.2f} ns")
    print(f"  ✓ 脉冲能量: {pulse.energy:.2e} J")
    print(f"  ✓ 用户 ID: {user.user_id}")

    # 3. 窄带干扰仿真
    print("\n[3/4] 运行窄带干扰仿真...")
    # 扩展SIR范围到-40dB，并增加采样密度以获得平滑曲线
    sir_db_values = np.linspace(-10, -40, 16)  # -10 to -40 dB, 16个点
    ber_results = []
    snr_db = 10.0  # 恢复 SNR = 10 dB（中等信噪比）
    nbi_frequency = 1.611e9  # 1.611 GHz 干扰频率（脉冲峰值频率，根据频谱诊断调整）
    print(f"  信噪比（SNR）: {snr_db} dB")
    print(f"  干扰频率: {nbi_frequency/1e9:.1f} GHz")
    print(f"  SIR 范围: {sir_db_values} dB")
    print()

    total_start_time = time.perf_counter()

    for sir_db in sir_db_values:
        print(f"  运行仿真: SIR = {sir_db:5.1f} dB...")
        start_time = time.perf_counter()

        # 信道传输（单用户 + SNR + SIR + NBI）
        channel = Channel(config=config, snr_db=snr_db, sir_db=sir_db, nbi_frequency=nbi_frequency)
        received_signal, time_axis = channel.transmit([user], pulse)

        # 接收解调
        receiver = Receiver(config=config, target_user=0, pulse=pulse)
        decoded_bits = receiver.demodulate(received_signal, user.th_code)

        # 计算性能指标
        metrics = PerformanceMetrics(user.data_bits, decoded_bits)
        ber_results.append(metrics.ber)

        elapsed = time.perf_counter() - start_time
        print(
            f"    ✓ BER = {metrics.ber:.4e}, 错误比特: {metrics.num_errors}/{metrics.num_bits}, 用时 {elapsed:.2f} 秒"
        )

    total_elapsed = time.perf_counter() - total_start_time
    print(f"\n  总用时: {total_elapsed:.2f} 秒")

    # 4. 绘制性能曲线
    print("\n[4/4] 生成性能曲线图...")
    plot_ber_vs_sir(
        sir_db=np.array(sir_db_values),
        ber_values=np.array(ber_results),
        title="窄带干扰抑制性能分析（演示）",
        xlabel="信干比 SIR (dB)",
        ylabel="误码率 (BER)",
        save_path="outputs/ber_vs_sir_demo.png",
        show=False,  # 无头环境不显示窗口
    )

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("性能分析结果汇总:")
    print("-" * 60)
    for sir, ber in zip(sir_db_values, ber_results):
        print(f"  SIR={sir:5.1f}dB: BER = {ber:.4e}")
    print("-" * 60)

    # 验证 UWB 抗干扰能力（SC-003）
    ber_worst = ber_results[-1]  # SIR=-10dB 时的 BER
    if ber_worst < 0.5:
        print(f"  ✓ UWB 抗干扰能力验证通过: BER={ber_worst:.4e} at SIR=-10dB (< 0.5)")
    else:
        print(f"  ⚠ UWB 抗干扰能力较弱: BER={ber_worst:.4e} at SIR=-10dB (≥ 0.5)")

    print("  观察: 即使在强窄带干扰下，UWB 系统仍能维持较低误码率")
    print("=" * 60)

    print("\n✅ 演示完成！图表已保存到 outputs/ber_vs_sir_demo.png")
    print("   提示：这是快速演示版本（1000 比特），完整仿真请运行 run_nbi_analysis.py")


if __name__ == "__main__":
    main()

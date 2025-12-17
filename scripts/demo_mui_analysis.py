#!/usr/bin/env python3
"""
演示：多用户干扰（MUI）性能分析

快速验证脚本，分析多用户同时通信时的误码率（1000 比特，用于快速演示）。

用法:
    python scripts/demo_mui_analysis.py

输出:
    outputs/ber_vs_users_demo.png - BER vs 用户数量曲线图（300 DPI）

参考:
    - quickstart.md Step 2: 运行多用户干扰分析
    - 预期执行时间: ~1-2 分钟（1000 比特）
"""

import numpy as np
import time
from src.models.pulse import Pulse
from src.models.modulation import User
from src.models.channel import Channel
from src.simulation.receiver import Receiver
from src.simulation.metrics import PerformanceMetrics
from src.visualization.performance import plot_ber_vs_users
from src.config import SystemConfig


def main():
    """主函数：多用户干扰性能分析（快速演示版）"""
    print("=" * 60)
    print("演示：多用户干扰（MUI）性能分析")
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

    # 2. 初始化脉冲
    print("\n[2/4] 生成脉冲模板...")
    pulse = Pulse.generate(config)
    print(f"  ✓ 脉冲持续时间: {pulse.duration*1e9:.2f} ns")
    print(f"  ✓ 脉冲能量: {pulse.energy:.2e} J")

    # 3. 多用户仿真
    print("\n[3/4] 运行多用户仿真...")
    user_counts = [1, 2, 3, 5]  # 用户数量（快速演示版）
    ber_results = []
    snr_db = 10.0  # 固定 SNR = 10 dB
    print(f"  信噪比（SNR）: {snr_db} dB")
    print(f"  用户数量: {user_counts}")
    print()

    total_start_time = time.perf_counter()

    for K in user_counts:
        print(f"  运行仿真: {K} 个用户...")
        start_time = time.perf_counter()

        # 创建用户
        users = [User.create(user_id=k, config=config) for k in range(K)]

        # 信道传输（多用户信号 + AWGN）
        channel = Channel(config=config, snr_db=snr_db, sir_db=np.inf)
        received_signal, time_axis = channel.transmit(users, pulse)

        # 接收解调（解调第一个用户 user_0 的数据）
        receiver = Receiver(config=config, target_user=0, pulse=pulse)
        decoded_bits = receiver.demodulate(received_signal, users[0].th_code)

        # 计算性能指标
        metrics = PerformanceMetrics(users[0].data_bits, decoded_bits)
        ber_results.append(metrics.ber)

        elapsed = time.perf_counter() - start_time
        print(f"    ✓ BER = {metrics.ber:.4e}, 错误比特: {metrics.num_errors}/{metrics.num_bits}, 用时 {elapsed:.2f} 秒")

    total_elapsed = time.perf_counter() - total_start_time
    print(f"\n  总用时: {total_elapsed:.2f} 秒")

    # 4. 绘制性能曲线
    print("\n[4/4] 生成性能曲线图...")
    plot_ber_vs_users(
        user_counts=np.array(user_counts),
        ber_values=np.array(ber_results),
        snr_db=snr_db,
        title="多用户干扰性能分析（演示）",
        xlabel="用户数量",
        ylabel="误码率 (BER)",
        save_path="outputs/ber_vs_users_demo.png",
        show=False,  # 无头环境不显示窗口
    )

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("性能分析结果汇总:")
    print("-" * 60)
    for K, ber in zip(user_counts, ber_results):
        print(f"  用户数 {K:2d}: BER = {ber:.4e}")
    print("-" * 60)
    print(f"  观察: BER 随用户数量增加而上升（多用户干扰效应）")
    print("=" * 60)

    print("\n✅ 演示完成！图表已保存到 outputs/ber_vs_users_demo.png")
    print("   提示：这是快速演示版本（1000 比特），完整仿真请运行 run_mui_analysis.py")


if __name__ == "__main__":
    main()

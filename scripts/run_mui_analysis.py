#!/usr/bin/env python3
"""
完整仿真：多用户干扰（MUI）性能分析

完整仿真脚本，生成学术论文级别的 BER vs 用户数量性能曲线（10000 比特）。

用法:
    python scripts/run_mui_analysis.py

输出:
    outputs/ber_vs_users.png - BER vs 用户数量曲线图（300 DPI，适合论文打印）

参考:
    - quickstart.md: 完整仿真流程
    - spec.md SC-002: 多用户干扰性能分析目标
    - 预期执行时间: ~5-10 分钟（10000 比特，6 个用户点）
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
from src.simulation.metrics import PerformanceMetrics, SimulationResult
from src.visualization.performance import plot_ber_vs_users
from src.config import SystemConfig


def main():
    """主函数：多用户干扰性能分析（完整版）"""
    print("=" * 60)
    print("完整仿真：多用户干扰（MUI）性能分析")
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
        num_bits=10000,  # 10000 比特用于完整仿真
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
    print(f"  ✓ 脉冲采样点数: {len(pulse.waveform)}")

    # 3. 多用户仿真
    print("\n[3/4] 运行多用户仿真...")
    user_counts = [1, 2, 3, 5, 7, 10]  # 用户数量（完整版）
    ber_results = []
    simulation_results = []  # 保存所有仿真结果
    snr_db = 10.0  # 固定 SNR = 10 dB
    print(f"  信噪比（SNR）: {snr_db} dB")
    print(f"  用户数量: {user_counts}")
    print()

    total_start_time = time.perf_counter()

    for idx, K in enumerate(user_counts, 1):
        print(f"  [{idx}/{len(user_counts)}] 运行仿真: {K} 个用户...")
        start_time = time.perf_counter()

        # 创建用户
        users = [User.create(user_id=k, config=config) for k in range(K)]
        print(f"      - 创建了 {K} 个用户（独立跳时码）")

        # 信道传输（多用户信号 + AWGN）
        channel = Channel(config=config, snr_db=snr_db, sir_db=np.inf)
        received_signal, time_axis = channel.transmit(users, pulse)
        print(f"      - 信道传输完成（信号长度: {len(received_signal)} 采样点）")

        # 接收解调（解调第一个用户 user_0 的数据）
        receiver = Receiver(config=config, target_user=0, pulse=pulse)
        decoded_bits = receiver.demodulate(received_signal, users[0].th_code)
        print(f"      - 解调完成（目标用户: user_0）")

        # 计算性能指标
        metrics = PerformanceMetrics(users[0].data_bits, decoded_bits)
        ber_results.append(metrics.ber)

        # 保存仿真结果
        elapsed = time.perf_counter() - start_time
        result = SimulationResult(
            config=config,
            num_users=K,
            snr_db=snr_db,
            sir_db=np.inf,
            metrics=metrics,
            execution_time=elapsed,
        )
        simulation_results.append(result)

        # 计算置信区间
        ci_lower, ci_upper = metrics.ber_confidence_interval(confidence=0.95)

        print(f"      ✓ BER = {metrics.ber:.4e}, 错误比特: {metrics.num_errors}/{metrics.num_bits}")
        print(f"        95% 置信区间: [{ci_lower:.4e}, {ci_upper:.4e}]")
        print(f"        用时: {elapsed:.2f} 秒")
        print()

    total_elapsed = time.perf_counter() - total_start_time
    print(f"  总用时: {total_elapsed:.2f} 秒")
    print(f"  平均每个用户点: {total_elapsed/len(user_counts):.2f} 秒")

    # 4. 绘制性能曲线
    print("\n[4/4] 生成性能曲线图...")
    plot_ber_vs_users(
        user_counts=np.array(user_counts),
        ber_values=np.array(ber_results),
        snr_db=snr_db,
        title="多用户干扰性能分析",
        xlabel="用户数量",
        ylabel="误码率 (BER)",
        save_path="outputs/ber_vs_users.png",
        show=False,  # 无头环境不显示窗口
        figsize=(8, 6),
    )

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("性能分析结果汇总:")
    print("-" * 60)
    print(f"{'用户数':>6} | {'BER':>12} | {'错误比特':>10} | {'95% 置信区间':>28}")
    print("-" * 60)
    for result in simulation_results:
        ci_lower, ci_upper = result.metrics.ber_confidence_interval(0.95)
        print(
            f"{result.num_users:>6} | "
            f"{result.metrics.ber:>12.4e} | "
            f"{result.metrics.num_errors:>5}/{result.metrics.num_bits:<4} | "
            f"[{ci_lower:.4e}, {ci_upper:.4e}]"
        )
    print("-" * 60)

    # 验证 SC-002: BER 随用户数量单调上升
    is_monotonic = all(
        ber_results[i] <= ber_results[i + 1] for i in range(len(ber_results) - 1)
    )
    if is_monotonic:
        print("  ✓ SC-002 验证通过: BER 随用户数量单调上升")
    else:
        print("  ⚠ SC-002 验证失败: BER 未单调上升（可能需要增加比特数）")

    # 打印性能指标
    ber_single = ber_results[0]
    ber_10users = ber_results[-1]
    print(f"  单用户 BER: {ber_single:.4e}")
    print(f"  10 用户 BER: {ber_10users:.4e}")
    if ber_single > 0:
        print(f"  性能退化: {ber_10users/ber_single:.1f}x")
    else:
        print(f"  性能退化: 无法计算（单用户 BER = 0）")
    print("=" * 60)

    print("\n✅ 完整仿真完成！图表已保存到 outputs/ber_vs_users.png")
    print("   提示：这是完整版本（10000 比特），适合论文发表（≥300 DPI）")


if __name__ == "__main__":
    main()

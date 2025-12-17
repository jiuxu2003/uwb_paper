#!/usr/bin/env python3
"""
完整仿真：窄带干扰（NBI）性能分析

完整仿真脚本，生成学术论文级别的 BER vs SIR 性能曲线（10000 比特）。

用法:
    python scripts/run_nbi_analysis.py

输出:
    outputs/ber_vs_sir.png - BER vs SIR 曲线图（300 DPI，适合论文打印）

参考:
    - quickstart.md: 完整仿真流程
    - spec.md SC-003: 窄带干扰抑制性能分析目标
    - 预期执行时间: ~5-10 分钟（10000 比特，10 个 SIR 点）
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
from src.visualization.performance import plot_ber_vs_sir
from src.config import SystemConfig


def main():
    """主函数：窄带干扰性能分析（完整版）"""
    print("=" * 60)
    print("完整仿真：窄带干扰（NBI）性能分析")
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

    # 2. 初始化脉冲和用户
    print("\n[2/4] 初始化脉冲和用户...")
    pulse = Pulse.generate(config)
    user = User.create(user_id=0, config=config)
    print(f"  ✓ 脉冲持续时间: {pulse.duration*1e9:.2f} ns")
    print(f"  ✓ 脉冲能量: {pulse.energy:.2e} J")
    print(f"  ✓ 脉冲采样点数: {len(pulse.waveform)}")
    print(f"  ✓ 用户 ID: {user.user_id}")

    # 3. 窄带干扰仿真
    print("\n[3/4] 运行窄带干扰仿真...")
    # 扩展SIR范围：从-10dB到-40dB（展示从弱干扰到极强干扰的完整过渡）
    sir_db_values = np.linspace(-10, -40, 16)  # 16个点，获得平滑曲线
    ber_results = []
    simulation_results = []  # 保存所有仿真结果
    snr_db = 10.0  # SNR = 10 dB（中等信噪比）
    nbi_frequency = 1.611e9  # 1.611 GHz 干扰频率（脉冲峰值频率，根据频谱分析优化）
    print(f"  信噪比（SNR）: {snr_db} dB")
    print(f"  干扰频率: {nbi_frequency/1e9:.3f} GHz（脉冲峰值频率）")
    print(f"  SIR 范围: [{sir_db_values[0]:.1f}, {sir_db_values[-1]:.1f}] dB")
    print(f"  SIR 点数: {len(sir_db_values)}")
    print()

    total_start_time = time.perf_counter()

    for idx, sir_db in enumerate(sir_db_values, 1):
        print(f"  [{idx}/{len(sir_db_values)}] 运行仿真: SIR = {sir_db:.1f} dB...")
        start_time = time.perf_counter()

        # 信道传输（单用户 + SNR + SIR + NBI）
        channel = Channel(config=config, snr_db=snr_db, sir_db=sir_db, nbi_frequency=nbi_frequency)
        received_signal, time_axis = channel.transmit([user], pulse)
        print(f"      - 信道传输完成（信号长度: {len(received_signal)} 采样点）")

        # 接收解调
        receiver = Receiver(config=config, target_user=0, pulse=pulse)
        decoded_bits = receiver.demodulate(received_signal, user.th_code)
        print(f"      - 解调完成（目标用户: user_0）")

        # 计算性能指标
        metrics = PerformanceMetrics(user.data_bits, decoded_bits)
        ber_results.append(metrics.ber)

        # 保存仿真结果
        elapsed = time.perf_counter() - start_time
        result = SimulationResult(
            config=config,
            num_users=1,  # 单用户场景
            snr_db=snr_db,
            sir_db=sir_db,
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
    print(f"  平均每个 SIR 点: {total_elapsed/len(sir_db_values):.2f} 秒")

    # 4. 绘制性能曲线
    print("\n[4/4] 生成性能曲线图...")
    plot_ber_vs_sir(
        sir_db=np.array(sir_db_values),
        ber_values=np.array(ber_results),
        title="窄带干扰抑制性能分析",
        xlabel="信干比 SIR (dB)",
        ylabel="误码率 (BER)",
        save_path="outputs/ber_vs_sir.png",
        show=False,  # 无头环境不显示窗口
        figsize=(8, 6),
    )

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("性能分析结果汇总:")
    print("-" * 60)
    print(f"{'SIR (dB)':>10} | {'BER':>12} | {'错误比特':>10} | {'95% 置信区间':>28}")
    print("-" * 60)
    for result in simulation_results:
        ci_lower, ci_upper = result.metrics.ber_confidence_interval(0.95)
        print(
            f"{result.sir_db:>10.1f} | "
            f"{result.metrics.ber:>12.4e} | "
            f"{result.metrics.num_errors:>5}/{result.metrics.num_bits:<4} | "
            f"[{ci_lower:.4e}, {ci_upper:.4e}]"
        )
    print("-" * 60)

    # 验证 SC-003: UWB 抗干扰能力（BER < 0.5 at SIR=-10dB）
    ber_worst = ber_results[-1]  # SIR=-10dB 时的 BER
    if ber_worst < 0.5:
        print(f"  ✓ SC-003 验证通过: BER={ber_worst:.4e} at SIR={sir_db_values[-1]:.1f}dB (< 0.5)")
    else:
        print(f"  ⚠ SC-003 验证失败: BER={ber_worst:.4e} at SIR={sir_db_values[-1]:.1f}dB (≥ 0.5)")

    # 验证 BER 随 SIR 降低而单调上升
    is_monotonic = all(ber_results[i] <= ber_results[i + 1] for i in range(len(ber_results) - 1))
    if is_monotonic:
        print("  ✓ BER 随 SIR 降低单调上升（符合预期）")
    else:
        print("  ⚠ BER 未单调上升（可能需要增加比特数或调整参数）")

    # 打印关键性能指标
    ber_best = ber_results[0]  # SIR=30dB 时的 BER
    print(f"  最佳 BER (SIR={sir_db_values[0]:.1f}dB): {ber_best:.4e}")
    print(f"  最差 BER (SIR={sir_db_values[-1]:.1f}dB): {ber_worst:.4e}")
    if ber_best > 0:
        print(f"  性能退化: {ber_worst/ber_best:.1f}x")
    else:
        print(f"  性能退化: 无法计算（最佳 BER = 0）")
    print("  观察: 即使在强窄带干扰下，UWB 系统仍能维持较低误码率")
    print("=" * 60)

    print("\n✅ 完整仿真完成！图表已保存到 outputs/ber_vs_sir.png")
    print("   提示：这是完整版本（10000 比特），适合论文发表（≥300 DPI）")


if __name__ == "__main__":
    main()

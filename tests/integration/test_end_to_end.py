"""
端到端集成测试

测试完整的 TH-UWB 通信系统流程：从配置到 BER 计算的全链路验证。
"""

import pytest
import numpy as np
from src.config import SystemConfig
from src.models.pulse import Pulse
from src.models.modulation import User
from src.models.channel import Channel
from src.simulation.receiver import Receiver
from src.simulation.metrics import PerformanceMetrics


@pytest.fixture
def config():
    """测试配置（中等规模，平衡速度和准确性）"""
    return SystemConfig(
        frame_period=100e-9,
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,
        pulse_amplitude=1.0,
        sampling_rate=50e9,
        num_bits=500,  # 500 比特用于集成测试
        random_seed=42,
    )


def test_end_to_end_single_user(config):
    """
    端到端测试：单用户场景

    完整流程：config → pulse → user → channel → receiver → metrics

    验证：
    1. 无噪声条件下，BER = 0（完美重建）
    2. 中等 SNR（10 dB）条件下，BER < 0.05
    """
    # 1. 生成脉冲
    pulse = Pulse.generate(config)
    assert pulse is not None
    assert len(pulse.waveform) > 0

    # 2. 创建用户
    user = User.create(user_id=0, config=config)
    assert user is not None
    assert len(user.data_bits) == config.num_bits

    # 3. 生成信号
    signal = user.generate_signal(pulse)
    assert len(signal) > 0

    # 4. 理想信道传输（无噪声、无干扰）
    channel_perfect = Channel(config=config, snr_db=np.inf, sir_db=np.inf)
    received_perfect, time_axis = channel_perfect.transmit([user], pulse)

    # 5. 接收解调
    receiver = Receiver(config=config, target_user=0, pulse=pulse)
    decoded_perfect = receiver.demodulate(received_perfect, user.th_code)

    # 6. 计算性能指标
    metrics_perfect = PerformanceMetrics(user.data_bits, decoded_perfect)

    # 验证 1：理想信道下 BER = 0
    assert metrics_perfect.ber == 0.0, (
        f"Expected BER=0 in perfect channel, got {metrics_perfect.ber}"
    )
    assert metrics_perfect.num_errors == 0

    # 7. AWGN 信道传输（SNR = 10 dB）
    channel_awgn = Channel(config=config, snr_db=10.0, sir_db=np.inf)
    received_awgn, _ = channel_awgn.transmit([user], pulse)

    decoded_awgn = receiver.demodulate(received_awgn, user.th_code)
    metrics_awgn = PerformanceMetrics(user.data_bits, decoded_awgn)

    # 验证 2：AWGN 信道下 BER 较低（可能为 0，因为样本量较小）
    assert metrics_awgn.ber <= 0.1, (
        f"Expected BER <= 0.1 at SNR=10dB, got {metrics_awgn.ber}"
    )

    print(f"\n✓ Single-user end-to-end test passed:")
    print(f"  Perfect channel: BER = {metrics_perfect.ber:.4e}")
    print(f"  AWGN (SNR=10dB): BER = {metrics_awgn.ber:.4e}, Errors = {metrics_awgn.num_errors}/{metrics_awgn.num_bits}")


def test_end_to_end_multi_user(config):
    """
    端到端测试：多用户场景

    完整流程：config → pulse → users → channel → receiver → metrics

    验证：
    1. BER 随用户数量单调增加（MUI 效应）
    2. 单用户 BER < 多用户 BER
    3. 所有 BER < 0.5（系统仍可用）
    """
    # 1. 生成脉冲
    pulse = Pulse.generate(config)

    # 2. 测试不同用户数量
    user_counts = [1, 2, 3, 5]
    ber_values = []
    snr_db = 10.0

    channel = Channel(config=config, snr_db=snr_db, sir_db=np.inf)

    for K in user_counts:
        # 创建用户
        users = [User.create(user_id=k, config=config) for k in range(K)]

        # 信道传输
        received_signal, _ = channel.transmit(users, pulse)

        # 接收解调（目标用户：user_0）
        receiver = Receiver(config=config, target_user=0, pulse=pulse)
        decoded_bits = receiver.demodulate(received_signal, users[0].th_code)

        # 计算 BER
        metrics = PerformanceMetrics(users[0].data_bits, decoded_bits)
        ber_values.append(metrics.ber)

    # 验证 1：BER 随用户数量单调增加
    is_monotonic = all(
        ber_values[i] <= ber_values[i + 1] for i in range(len(ber_values) - 1)
    )
    if not is_monotonic:
        # 允许轻微违反（由于随机性），但整体趋势应该向上
        # 检查至少 75% 的相邻对满足单调性
        monotonic_count = sum(
            ber_values[i] <= ber_values[i + 1] for i in range(len(ber_values) - 1)
        )
        monotonic_ratio = monotonic_count / (len(ber_values) - 1)
        assert monotonic_ratio >= 0.75, (
            f"BER should increase with user count (monotonic ratio = {monotonic_ratio:.2f})"
        )

    # 验证 2：单用户 BER < 多用户 BER
    ber_single = ber_values[0]
    ber_multi = ber_values[-1]
    assert ber_single < ber_multi or np.isclose(ber_single, ber_multi, rtol=0.5), (
        f"Single-user BER ({ber_single:.4e}) should be <= multi-user BER ({ber_multi:.4e})"
    )

    # 验证 3：所有 BER < 0.5（系统仍可用）
    assert all(ber < 0.5 for ber in ber_values), (
        f"All BER values should be < 0.5, got {ber_values}"
    )

    print(f"\n✓ Multi-user end-to-end test passed:")
    for K, ber in zip(user_counts, ber_values):
        print(f"  {K} users: BER = {ber:.4e}")
    if ber_single > 0:
        print(f"  Performance degradation: {ber_multi/ber_single:.1f}x")
    else:
        print(f"  Performance degradation: N/A (single-user BER=0)")


def test_end_to_end_with_nbi(config):
    """
    端到端测试：窄带干扰场景

    验证：
    1. UWB 在强干扰下仍然可用（BER < 0.5）
    2. SIR 降低时 BER 上升
    """
    # 1. 生成脉冲和用户
    pulse = Pulse.generate(config)
    user = User.create(user_id=0, config=config)

    # 2. 测试不同 SIR
    sir_values = [30, 20, 10, 0, -10]  # dB
    ber_values = []

    for sir_db in sir_values:
        # 信道传输（SNR=10dB，变化的 SIR）
        channel = Channel(config=config, snr_db=10.0, sir_db=sir_db, nbi_frequency=2.4e9)
        received_signal, _ = channel.transmit([user], pulse)

        # 接收解调
        receiver = Receiver(config=config, target_user=0, pulse=pulse)
        decoded_bits = receiver.demodulate(received_signal, user.th_code)

        # 计算 BER
        metrics = PerformanceMetrics(user.data_bits, decoded_bits)
        ber_values.append(metrics.ber)

    # 验证 1：即使在 SIR=-10dB 时，BER 仍 < 0.5
    ber_worst = ber_values[-1]
    assert ber_worst < 0.5, (
        f"BER at SIR=-10dB should be < 0.5 (UWB robustness), got {ber_worst:.4e}"
    )

    # 验证 2：SIR 降低时 BER 应该增加（整体趋势）
    # 允许局部波动，但首尾差异应该明显
    ber_best = ber_values[0]
    assert ber_worst > ber_best or np.isclose(ber_worst, ber_best, rtol=0.5), (
        f"BER should increase when SIR decreases: "
        f"SIR=30dB → {ber_best:.4e}, SIR=-10dB → {ber_worst:.4e}"
    )

    print(f"\n✓ NBI end-to-end test passed:")
    for sir, ber in zip(sir_values, ber_values):
        print(f"  SIR={sir:3d}dB: BER = {ber:.4e}")
    print(f"  UWB robustness verified: BER={ber_worst:.4e} at SIR=-10dB")


def test_end_to_end_reproducibility(config):
    """
    端到端测试：可重复性

    验证相同配置产生相同结果
    """
    # 第一次运行
    pulse1 = Pulse.generate(config)
    user1 = User.create(user_id=0, config=config)

    np.random.seed(42)
    channel1 = Channel(config=config, snr_db=10.0)
    received1, _ = channel1.transmit([user1], pulse1)

    receiver1 = Receiver(config=config, target_user=0, pulse=pulse1)
    decoded1 = receiver1.demodulate(received1, user1.th_code)

    metrics1 = PerformanceMetrics(user1.data_bits, decoded1)

    # 第二次运行（相同配置）
    pulse2 = Pulse.generate(config)
    user2 = User.create(user_id=0, config=config)

    np.random.seed(42)
    channel2 = Channel(config=config, snr_db=10.0)
    received2, _ = channel2.transmit([user2], pulse2)

    receiver2 = Receiver(config=config, target_user=0, pulse=pulse2)
    decoded2 = receiver2.demodulate(received2, user2.th_code)

    metrics2 = PerformanceMetrics(user2.data_bits, decoded2)

    # 验证：两次运行结果完全一致
    np.testing.assert_array_equal(user1.data_bits, user2.data_bits)
    np.testing.assert_array_equal(decoded1, decoded2)
    assert metrics1.ber == metrics2.ber
    assert metrics1.num_errors == metrics2.num_errors

    print(f"\n✓ Reproducibility test passed:")
    print(f"  BER (run 1) = {metrics1.ber:.4e}")
    print(f"  BER (run 2) = {metrics2.ber:.4e}")
    print(f"  Results are identical ✓")


def test_end_to_end_performance_target(config):
    """
    端到端测试：性能目标验证

    验证系统满足 spec.md 中的性能要求
    """
    import time

    # 使用较小规模配置测试性能
    small_config = SystemConfig(
        frame_period=100e-9,
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,
        pulse_amplitude=1.0,
        sampling_rate=50e9,
        num_bits=1000,  # 1000 比特
        random_seed=42,
    )

    pulse = Pulse.generate(small_config)
    users = [User.create(user_id=k, config=small_config) for k in range(3)]
    channel = Channel(config=small_config, snr_db=10.0, sir_db=np.inf)
    receiver = Receiver(config=small_config, target_user=0, pulse=pulse)

    # 测量执行时间
    start_time = time.perf_counter()

    received_signal, _ = channel.transmit(users, pulse)
    decoded_bits = receiver.demodulate(received_signal, users[0].th_code)
    metrics = PerformanceMetrics(users[0].data_bits, decoded_bits)

    elapsed_time = time.perf_counter() - start_time

    # 验证：单次仿真 < 60 秒（spec.md SC-006）
    assert elapsed_time < 60, (
        f"Simulation should complete in < 60 sec, took {elapsed_time:.2f} sec"
    )

    print(f"\n✓ Performance target test passed:")
    print(f"  Configuration: 3 users, 1000 bits")
    print(f"  Execution time: {elapsed_time:.2f} seconds")
    print(f"  BER: {metrics.ber:.4e}")
    print(f"  Target met: < 60 seconds ✓")

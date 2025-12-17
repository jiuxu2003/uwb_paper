"""
Pulse 模块单元测试

测试二阶高斯导数脉冲的生成和属性验证。
"""

import pytest
import numpy as np
from src.config import SystemConfig
from src.models.pulse import generate_gaussian_doublet, Pulse


def get_default_config() -> SystemConfig:
    """获取默认测试配置"""
    return SystemConfig(
        frame_period=100e-9,
        slot_width=10e-9,
        num_slots=8,
        ppm_delay=5e-9,
        pulse_tau=0.5e-9,
        pulse_amplitude=1.0,
        sampling_rate=50e9,
        num_bits=1000,
        random_seed=42,
    )


class TestGaussianDoublet:
    """测试 generate_gaussian_doublet() 函数"""

    def test_gaussian_doublet_zero_mean(self):
        """
        T012: 验证二阶高斯导数脉冲的零均值特性

        二阶高斯导数是奇对称函数，积分（均值）应为零。
        """
        # 生成足够长的时间轴以覆盖整个脉冲
        t = np.linspace(-2.5e-9, 2.5e-9, 1000)
        pulse = generate_gaussian_doublet(t, tau=0.5e-9, amplitude=1.0)

        # 验证零均值（数值误差容差 1e-6）
        mean_val = np.mean(pulse)
        assert np.abs(mean_val) < 1e-6, f"脉冲均值应接近零，实际值: {mean_val}"

    def test_gaussian_doublet_peak_at_zero(self):
        """
        T013: 验证二阶高斯导数脉冲的峰值位置

        根据公式 g''(0) = A，峰值应在 t=0 处达到幅度 A。
        """
        # 生成时间轴，确保 t=0 是中心点
        t = np.linspace(-2.5e-9, 2.5e-9, 1001)  # 奇数个点，中心为 t=0
        amplitude = 1.0
        pulse = generate_gaussian_doublet(t, tau=0.5e-9, amplitude=amplitude)

        # 找到 t=0 对应的索引（中心点）
        center_idx = len(t) // 2

        # 验证中心点的值接近幅度 A
        peak_val = pulse[center_idx]
        assert (
            np.abs(peak_val - amplitude) < 0.01
        ), f"峰值应为 {amplitude}，实际值: {peak_val}"

        # 验证峰值确实在中心位置（最大绝对值）
        max_abs_idx = np.argmax(np.abs(pulse))
        assert (
            abs(max_abs_idx - center_idx) <= 1
        ), f"峰值应在中心，实际索引: {max_abs_idx}，中心索引: {center_idx}"

    def test_gaussian_doublet_input_validation(self):
        """验证输入参数验证"""
        t = np.linspace(-1e-9, 1e-9, 100)

        # 测试非 NumPy 数组输入
        with pytest.raises(TypeError):
            generate_gaussian_doublet([0, 1, 2], tau=0.5e-9)

        # 测试非 1D 数组输入
        with pytest.raises(ValueError):
            generate_gaussian_doublet(np.zeros((10, 10)), tau=0.5e-9)

        # 测试负 tau
        with pytest.raises(ValueError):
            generate_gaussian_doublet(t, tau=-0.5e-9)

        # 测试零 tau
        with pytest.raises(ValueError):
            generate_gaussian_doublet(t, tau=0)

        # 测试负 amplitude
        with pytest.raises(ValueError):
            generate_gaussian_doublet(t, tau=0.5e-9, amplitude=-1.0)


class TestPulse:
    """测试 Pulse 类"""

    def test_pulse_generate(self):
        """
        T014: 验证 Pulse.generate() 工厂方法

        验证生成的脉冲实例包含正确的波形和时间轴。
        """
        config = get_default_config()
        pulse = Pulse.generate(config)

        # 验证返回类型
        assert isinstance(pulse, Pulse), "应返回 Pulse 实例"

        # 验证波形和时间轴长度一致
        assert len(pulse.waveform) == len(
            pulse.time_axis
        ), "波形和时间轴长度必须一致"

        # 验证脉冲持续时间约为 5τ
        expected_duration = 5 * config.pulse_tau
        actual_duration = pulse.duration
        relative_error = abs(actual_duration - expected_duration) / expected_duration
        assert (
            relative_error < 0.01
        ), f"脉冲持续时间应约为 5τ={expected_duration:.2e}s，实际: {actual_duration:.2e}s"

        # 验证零均值
        mean_val = np.mean(pulse.waveform)
        assert np.abs(mean_val) < 1e-6, f"脉冲应为零均值，实际均值: {mean_val}"

        # 验证峰值接近配置的幅度
        peak_val = np.max(np.abs(pulse.waveform))
        assert (
            np.abs(peak_val - config.pulse_amplitude) < 0.1
        ), f"峰值应约为 {config.pulse_amplitude}，实际: {peak_val}"

    def test_pulse_energy_positive(self):
        """
        T015: 验证 Pulse.energy 属性返回正值

        脉冲能量定义为 ∫|g''(t)|² dt > 0（非零脉冲）。
        """
        config = get_default_config()
        pulse = Pulse.generate(config)

        # 验证能量为正
        assert pulse.energy > 0, f"脉冲能量必须为正，实际值: {pulse.energy}"

        # 验证能量的合理性（数量级检查）
        # 对于幅度为 1.0，持续时间为 ns 级别的脉冲，能量应在 1e-12 到 1e-8 范围内
        assert (
            1e-12 < pulse.energy < 1e-8
        ), f"脉冲能量数量级异常: {pulse.energy}"

    def test_pulse_duration_property(self):
        """验证 duration 属性"""
        config = get_default_config()
        pulse = Pulse.generate(config)

        # 验证 duration 计算正确
        expected_duration = pulse.time_axis[-1] - pulse.time_axis[0]
        assert (
            pulse.duration == expected_duration
        ), "duration 应等于时间轴的跨度"

    def test_pulse_with_different_parameters(self):
        """测试不同参数下的脉冲生成"""
        config = get_default_config()

        # 测试不同脉冲宽度
        config_narrow = SystemConfig(
            frame_period=200e-9,
            slot_width=20e-9,
            num_slots=8,
            ppm_delay=10e-9,
            pulse_tau=0.25e-9,  # 更窄的脉冲
            pulse_amplitude=1.0,
            sampling_rate=50e9,
            num_bits=1000,
            random_seed=42,
        )
        pulse_narrow = Pulse.generate(config_narrow)

        pulse_normal = Pulse.generate(config)

        # 验证脉冲持续时间比例
        ratio = pulse_narrow.duration / pulse_normal.duration
        expected_ratio = config_narrow.pulse_tau / config.pulse_tau
        assert (
            np.abs(ratio - expected_ratio) < 0.01
        ), "脉冲持续时间应与 tau 成正比"

    def test_pulse_config_validation(self):
        """测试配置参数验证"""
        # 测试违反 Tf >= Nh·Tc 约束
        with pytest.raises(AssertionError):
            SystemConfig(
                frame_period=50e-9,  # 太短
                slot_width=10e-9,
                num_slots=8,
                ppm_delay=5e-9,
                pulse_tau=0.5e-9,
                pulse_amplitude=1.0,
                sampling_rate=50e9,
                num_bits=1000,
                random_seed=42,
            )

        # 测试违反 δ = Tc/2 约束
        with pytest.raises(AssertionError):
            SystemConfig(
                frame_period=100e-9,
                slot_width=10e-9,
                num_slots=8,
                ppm_delay=6e-9,  # 错误的 PPM 延迟
                pulse_tau=0.5e-9,
                pulse_amplitude=1.0,
                sampling_rate=50e9,
                num_bits=1000,
                random_seed=42,
            )

        # 测试违反采样率约束
        with pytest.raises(AssertionError):
            SystemConfig(
                frame_period=100e-9,
                slot_width=10e-9,
                num_slots=8,
                ppm_delay=5e-9,
                pulse_tau=0.5e-9,
                pulse_amplitude=1.0,
                sampling_rate=10e9,  # 太低
                num_bits=1000,
                random_seed=42,
            )

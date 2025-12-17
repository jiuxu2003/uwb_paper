"""
FCC 合规性模块单元测试

测试内容：
1. FCC 室内掩蔽罩生成（T010）
2. FCC 掩蔽罩数值精度验证（T011）
3. 合规性判断功能（T012）
4. 合规性判断准确率验证（T013）
"""

import pytest
import numpy as np
from src.models import fcc


class TestFCCIndoorMask:
    """测试 FCC 室内掩蔽罩生成"""

    def test_fcc_mask_generation(self):
        """T010: 测试 FCC 掩蔽罩生成基本功能"""
        # 生成测试频率轴（0.5-12 GHz，覆盖所有FCC频段）
        frequencies = np.linspace(0.5e9, 12e9, 1000)

        # 调用 FCC 掩蔽罩生成函数
        mask = fcc.get_fcc_indoor_mask(frequencies)

        # 验证输出类型和形状
        assert isinstance(mask, np.ndarray), "mask 应为 numpy 数组"
        assert mask.shape == frequencies.shape, "mask 形状应与 frequencies 一致"
        assert mask.ndim == 1, "mask 应为 1D 数组"

        # 验证数值范围合理（FCC 限制值在 -100 到 0 dBm/MHz 之间）
        assert np.all(mask >= -100), "mask 值不应低于 -100 dBm/MHz"
        assert np.all(mask <= 0), "mask 值不应高于 0 dBm/MHz"

    def test_fcc_band_values(self):
        """T011: 测试 FCC 频段限制值精度

        验证条件：
        - 频段 1 (0.96-1.61 GHz): -75.3 dBm/MHz (误差 < 0.1 dB)
        - 频段 2 (1.61-3.1 GHz): -53.3 dBm/MHz (误差 < 0.1 dB)
        - 频段 3 (3.1-10.6 GHz): -41.3 dBm/MHz (误差 < 0.1 dB)
        - 频段 4 (>10.6 GHz): -51.3 dBm/MHz (误差 < 0.1 dB)
        """
        tolerance = 0.1  # dB

        # 测试频段 1：0.96-1.61 GHz -> -75.3 dBm/MHz
        freq_band1 = np.array([1.0e9, 1.2e9, 1.5e9])
        mask_band1 = fcc.get_fcc_indoor_mask(freq_band1)
        assert np.allclose(mask_band1, -75.3, atol=tolerance), \
            f"频段 1 限制值误差超过 {tolerance} dB: {mask_band1}"

        # 测试频段 2：1.61-3.1 GHz -> -53.3 dBm/MHz
        freq_band2 = np.array([1.8e9, 2.4e9, 3.0e9])
        mask_band2 = fcc.get_fcc_indoor_mask(freq_band2)
        assert np.allclose(mask_band2, -53.3, atol=tolerance), \
            f"频段 2 限制值误差超过 {tolerance} dB: {mask_band2}"

        # 测试频段 3：3.1-10.6 GHz -> -41.3 dBm/MHz (UWB 合法频段)
        freq_band3 = np.array([4.0e9, 6.85e9, 10.0e9])
        mask_band3 = fcc.get_fcc_indoor_mask(freq_band3)
        assert np.allclose(mask_band3, -41.3, atol=tolerance), \
            f"频段 3 限制值误差超过 {tolerance} dB: {mask_band3}"

        # 测试频段 4：>10.6 GHz -> -51.3 dBm/MHz
        freq_band4 = np.array([11.0e9, 12.0e9, 15.0e9])
        mask_band4 = fcc.get_fcc_indoor_mask(freq_band4)
        assert np.allclose(mask_band4, -51.3, atol=tolerance), \
            f"频段 4 限制值误差超过 {tolerance} dB: {mask_band4}"

    def test_boundary_transitions(self):
        """测试 FCC 掩蔽罩边界频点阶跃特性

        验证条件：
        - 3.1 GHz 边界：左侧 -53.3，右侧 -41.3
        - 10.6 GHz 边界：左侧 -41.3，右侧 -51.3
        """
        tolerance = 0.1  # dB

        # 测试 3.1 GHz 边界（从频段 2 跳变到频段 3）
        freq_3_1_left = np.array([3.09e9])   # 3.1 GHz 左侧
        freq_3_1_right = np.array([3.11e9])  # 3.1 GHz 右侧
        mask_left = fcc.get_fcc_indoor_mask(freq_3_1_left)
        mask_right = fcc.get_fcc_indoor_mask(freq_3_1_right)

        assert np.allclose(mask_left, -53.3, atol=tolerance), \
            f"3.1 GHz 左侧应为 -53.3 dBm/MHz: {mask_left}"
        assert np.allclose(mask_right, -41.3, atol=tolerance), \
            f"3.1 GHz 右侧应为 -41.3 dBm/MHz: {mask_right}"

        # 测试 10.6 GHz 边界（从频段 3 跳变到频段 4）
        freq_10_6_left = np.array([10.59e9])  # 10.6 GHz 左侧
        freq_10_6_right = np.array([10.61e9]) # 10.6 GHz 右侧
        mask_left = fcc.get_fcc_indoor_mask(freq_10_6_left)
        mask_right = fcc.get_fcc_indoor_mask(freq_10_6_right)

        assert np.allclose(mask_left, -41.3, atol=tolerance), \
            f"10.6 GHz 左侧应为 -41.3 dBm/MHz: {mask_left}"
        assert np.allclose(mask_right, -51.3, atol=tolerance), \
            f"10.6 GHz 右侧应为 -51.3 dBm/MHz: {mask_right}"

    def test_output_length_consistency(self):
        """测试输出长度与输入一致"""
        # 测试不同长度的频率数组
        for n_points in [10, 100, 1200, 5000]:
            frequencies = np.linspace(0, 12e9, n_points)
            mask = fcc.get_fcc_indoor_mask(frequencies)
            assert len(mask) == len(frequencies), \
                f"输出长度 {len(mask)} 与输入长度 {len(frequencies)} 不一致"

    def test_invalid_frequencies(self):
        """测试无效频率输入（边界条件）"""
        # 测试非递增数组（应触发 assert）
        with pytest.raises(AssertionError, match="frequencies 必须是递增数组"):
            freq_non_increasing = np.array([5e9, 4e9, 3e9])
            fcc.get_fcc_indoor_mask(freq_non_increasing)

        # 测试负频率（应触发 assert）
        with pytest.raises(AssertionError, match="frequencies 必须 ≥ 0"):
            freq_negative = np.array([-1e9, 0, 1e9])
            fcc.get_fcc_indoor_mask(freq_negative)

        # 测试非 1D 数组（应触发 assert）
        with pytest.raises(AssertionError, match="frequencies 必须是 1D 数组"):
            freq_2d = np.array([[1e9, 2e9], [3e9, 4e9]])
            fcc.get_fcc_indoor_mask(freq_2d)


class TestComplianceCheck:
    """测试合规性判断功能"""

    def test_compliant_case(self):
        """T012/T013: 测试合规情况

        验证条件：5 阶脉冲（fc=6.85 GHz）应判定为 "Compliant"
        """
        from src.models import psd

        # 生成 5 阶脉冲（fc=6.85 GHz，UWB 中心频率）
        tau = 0.287e-9  # 脉冲宽度（秒）
        fc = 6.85e9      # 中心频率 6.85 GHz
        sampling_rate = 50e9  # 50 GHz 采样率
        waveform, _ = psd.generate_gaussian_derivative_pulse(
            n=5, tau=tau, fc=fc, sampling_rate=sampling_rate
        )

        # 计算 PSD
        frequencies, psd_values = psd.compute_psd(
            waveform, sampling_rate, freq_resolution=10e6, freq_range=(0, 12e9)
        )

        # 生成 FCC 掩蔽罩
        fcc_mask = fcc.get_fcc_indoor_mask(frequencies)

        # 判断合规性
        result = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)

        # 验证结果
        assert result["status"] == "Compliant", \
            f"5 阶脉冲（fc=6.85 GHz）应判定为 Compliant，实际: {result['status']}"
        assert result["max_excess"] < -0.1, \
            f"max_excess 应 < -0.1 dB，实际: {result['max_excess']:.2f} dB"
        assert result["num_violations"] == 0, \
            f"违规频点数应为 0，实际: {result['num_violations']}"
        assert result["compliance_percentage"] == 100.0, \
            f"合规百分比应为 100%，实际: {result['compliance_percentage']:.1f}%"

    def test_non_compliant_case(self):
        """T012/T013: 测试不合规情况

        验证条件：2 阶脉冲（fc=0）应判定为 "Non-compliant"
        """
        from src.models import psd

        # 生成 2 阶脉冲（fc=0，无调制）
        tau = 0.287e-9  # 脉冲宽度（秒）
        fc = 0           # 无调制
        sampling_rate = 50e9  # 50 GHz 采样率
        waveform, _ = psd.generate_gaussian_derivative_pulse(
            n=2, tau=tau, fc=fc, sampling_rate=sampling_rate
        )

        # 计算 PSD
        frequencies, psd_values = psd.compute_psd(
            waveform, sampling_rate, freq_resolution=10e6, freq_range=(0, 12e9)
        )

        # 生成 FCC 掩蔽罩
        fcc_mask = fcc.get_fcc_indoor_mask(frequencies)

        # 判断合规性
        result = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)

        # 验证结果
        assert result["status"] == "Non-compliant", \
            f"2 阶脉冲（fc=0）应判定为 Non-compliant，实际: {result['status']}"
        assert result["max_excess"] > 0.1, \
            f"max_excess 应 > 0.1 dB，实际: {result['max_excess']:.2f} dB"
        assert result["num_violations"] > 0, \
            f"违规频点数应 > 0，实际: {result['num_violations']}"
        assert result["compliance_percentage"] < 100.0, \
            f"合规百分比应 < 100%，实际: {result['compliance_percentage']:.1f}%"

    def test_marginal_compliant_case(self):
        """测试临界合规情况

        验证条件：max_excess 在 [-0.1, +0.1] dB 范围内
        """
        # 构造临界合规测试数据（人工制作边缘情况）
        frequencies = np.linspace(3.1e9, 10.6e9, 100)  # UWB 频段
        fcc_mask = fcc.get_fcc_indoor_mask(frequencies)  # -41.3 dBm/MHz

        # PSD 值略低于限制，但在容差范围内（max_excess = +0.05 dB）
        psd_values = fcc_mask + 0.05  # 超出 0.05 dB（在 [-0.1, +0.1] 范围内）

        result = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)

        # 验证临界合规
        assert result["status"] == "Marginal Compliant", \
            f"应判定为 Marginal Compliant，实际: {result['status']}"
        assert -0.1 <= result["max_excess"] <= 0.1, \
            f"max_excess 应在 [-0.1, +0.1] dB 范围内，实际: {result['max_excess']:.3f} dB"

    def test_compliance_accuracy(self):
        """T013: 测试合规性判断准确率

        验证条件：准确率 100%（SC-003）
        """
        # 测试 3 个典型场景，验证判断逻辑准确性
        from src.models import psd

        sampling_rate = 50e9
        tau = 0.287e-9

        # 场景 1：5 阶脉冲 fc=6.85 GHz（合规）
        waveform_5th, _ = psd.generate_gaussian_derivative_pulse(
            n=5, tau=tau, fc=6.85e9, sampling_rate=sampling_rate
        )
        freq_5th, psd_5th = psd.compute_psd(
            waveform_5th, sampling_rate, freq_range=(0, 12e9)
        )
        mask_5th = fcc.get_fcc_indoor_mask(freq_5th)
        result_5th = fcc.check_compliance(psd_5th, mask_5th, freq_5th)

        # 场景 2：2 阶脉冲 fc=0（不合规）
        waveform_2nd, _ = psd.generate_gaussian_derivative_pulse(
            n=2, tau=tau, fc=0, sampling_rate=sampling_rate
        )
        freq_2nd, psd_2nd = psd.compute_psd(
            waveform_2nd, sampling_rate, freq_range=(0, 12e9)
        )
        mask_2nd = fcc.get_fcc_indoor_mask(freq_2nd)
        result_2nd = fcc.check_compliance(psd_2nd, mask_2nd, freq_2nd)

        # 场景 3：人工构造临界合规
        freq_marginal = np.linspace(3.1e9, 10.6e9, 100)
        mask_marginal = fcc.get_fcc_indoor_mask(freq_marginal)
        psd_marginal = mask_marginal + 0.05  # 超出 0.05 dB（临界）
        result_marginal = fcc.check_compliance(psd_marginal, mask_marginal, freq_marginal)

        # 验证准确率 100%（3/3 正确判断）
        assert result_5th["status"] == "Compliant", "场景 1 判断错误"
        assert result_2nd["status"] == "Non-compliant", "场景 2 判断错误"
        assert result_marginal["status"] == "Marginal Compliant", "场景 3 判断错误"

        # SC-003 达成：准确率 100%
        accuracy = 3 / 3 * 100
        assert accuracy == 100.0, f"准确率应为 100%，实际: {accuracy:.1f}%"

    def test_tolerance_effect(self):
        """测试容差参数影响"""
        frequencies = np.linspace(3.1e9, 10.6e9, 100)
        fcc_mask = fcc.get_fcc_indoor_mask(frequencies)
        psd_values = fcc_mask + 0.15  # 超出 0.15 dB

        # 测试 tolerance=0.1（应判定为不合规）
        result_0_1 = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)
        assert result_0_1["status"] == "Non-compliant", \
            "tolerance=0.1 时应判定为 Non-compliant"

        # 测试 tolerance=0.2（应判定为临界合规）
        result_0_2 = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.2)
        assert result_0_2["status"] == "Marginal Compliant", \
            "tolerance=0.2 时应判定为 Marginal Compliant"

    def test_violation_list(self):
        """测试违规频点列表生成"""
        # 构造人工测试数据：100 个频点，前 20 个超限
        frequencies = np.linspace(3.1e9, 10.6e9, 100)
        fcc_mask = fcc.get_fcc_indoor_mask(frequencies)
        psd_values = fcc_mask.copy()
        psd_values[:20] = fcc_mask[:20] + 0.5  # 前 20 个超限 0.5 dB

        result = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)

        # 验证违规列表
        assert result["num_violations"] == 20, \
            f"违规频点数应为 20，实际: {result['num_violations']}"
        assert len(result["violations"]) == 20, \
            f"违规列表长度应为 20，实际: {len(result['violations'])}"

        # 验证违规列表格式：[(freq, excess), ...]
        for freq, excess in result["violations"]:
            assert isinstance(freq, float), "违规频率应为 float 类型"
            assert isinstance(excess, float), "超限量应为 float 类型"
            assert excess > 0.1, f"超限量应 > 0.1 dB，实际: {excess:.3f} dB"

    def test_max_excess_calculation(self):
        """测试最大超限量计算"""
        frequencies = np.linspace(3.1e9, 10.6e9, 100)
        fcc_mask = fcc.get_fcc_indoor_mask(frequencies)
        psd_values = fcc_mask.copy()
        psd_values[50] = fcc_mask[50] + 2.5  # 中间点超限 2.5 dB

        result = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)

        # 验证最大超限量
        assert abs(result["max_excess"] - 2.5) < 0.01, \
            f"max_excess 应约为 2.5 dB，实际: {result['max_excess']:.3f} dB"

    def test_compliance_percentage(self):
        """测试合规频点百分比计算"""
        # 构造人工测试数据：100 个频点，80 个合规，20 个超限
        frequencies = np.linspace(3.1e9, 10.6e9, 100)
        fcc_mask = fcc.get_fcc_indoor_mask(frequencies)
        psd_values = fcc_mask.copy()
        psd_values[:20] = fcc_mask[:20] + 0.5  # 前 20 个超限

        result = fcc.check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)

        # 验证合规百分比（80%）
        expected_percentage = 80.0
        assert abs(result["compliance_percentage"] - expected_percentage) < 0.1, \
            f"合规百分比应约为 {expected_percentage}%，实际: {result['compliance_percentage']:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

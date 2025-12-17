"""
FCC Part 15.209 合规性模块

本模块提供 FCC 室内辐射掩蔽罩生成和合规性判断功能。

功能：
1. 生成 FCC Part 15.209 室内辐射掩蔽罩（Indoor Mask）
2. 判断脉冲 PSD 是否符合 FCC 规范
3. 返回合规性状态和违规频点列表
"""

import numpy as np
from typing import Tuple, List, Dict


# FCC Part 15.209 室内辐射掩蔽罩频段定义
FCC_BANDS = [
    (0.96e9, 1.61e9, -75.3),   # 频段 1
    (1.61e9, 3.1e9, -53.3),    # 频段 2
    (3.1e9, 10.6e9, -41.3),    # 频段 3（UWB 合法频段）
    (10.6e9, np.inf, -51.3)    # 频段 4
]


def get_fcc_indoor_mask(frequencies: np.ndarray) -> np.ndarray:
    """
    生成 FCC Part 15.209 室内辐射掩蔽罩

    参数:
        frequencies: 频率轴（Hz），1D 数组，递增

    返回:
        mask_dbm_per_mhz: 功率限制（dBm/MHz），1D 数组

    前置条件:
        - frequencies 为 1D 数组
        - frequencies 递增
        - frequencies ≥ 0

    后置条件:
        - len(mask) == len(frequencies)
        - 频段限制值误差 < 0.1 dB

    FCC Part 15.209 频段定义（参见 research.md Section 2.1）:
        | 频率范围         | 功率限制（EIRP） |
        |-----------------|-----------------|
        | 0.96-1.61 GHz   | -75.3 dBm/MHz   |
        | 1.61-3.1 GHz    | -53.3 dBm/MHz   |
        | 3.1-10.6 GHz    | -41.3 dBm/MHz   | (UWB 合法频段)
        | >10.6 GHz       | -51.3 dBm/MHz   |
    """
    # 参数验证
    assert frequencies.ndim == 1, "frequencies 必须是 1D 数组"
    assert np.all(np.diff(frequencies) > 0), "frequencies 必须是递增数组"
    assert np.all(frequencies >= 0), "frequencies 必须 ≥ 0"

    # 定义 FCC 频段边界条件（使用 np.piecewise 实现阶跃跳变）
    conditions = [
        (frequencies >= 0.96e9) & (frequencies < 1.61e9),   # 频段 1
        (frequencies >= 1.61e9) & (frequencies < 3.1e9),    # 频段 2
        (frequencies >= 3.1e9) & (frequencies < 10.6e9),    # 频段 3 (UWB)
        frequencies >= 10.6e9                                # 频段 4
    ]

    # 对应的功率限制值（dBm/MHz）
    power_limits = [-75.3, -53.3, -41.3, -51.3]

    # 低于 0.96 GHz 的默认值（极低功率限制）
    default_limit = -100.0

    # 使用 np.piecewise 生成掩蔽罩
    mask_dbm_per_mhz = np.piecewise(
        frequencies,
        conditions,
        power_limits + [default_limit]  # 添加默认值作为最后一个参数
    )

    return mask_dbm_per_mhz


def check_compliance(
    psd_values: np.ndarray,
    fcc_mask: np.ndarray,
    frequencies: np.ndarray,
    tolerance: float = 0.1
) -> Dict:
    """
    判断脉冲 PSD 是否符合 FCC 规范

    参数:
        psd_values: 脉冲 PSD（dBm/MHz），1D 数组
        fcc_mask: FCC 掩蔽罩（dBm/MHz），1D 数组
        frequencies: 频率轴（Hz），1D 数组
        tolerance: 边界容差（dB），默认 0.1

    返回:
        result: 合规性结果字典，包含：
            - status: "Compliant" | "Non-compliant" | "Marginal Compliant"
            - max_excess: 最大超限量（dB）
            - violations: 违规频点列表 [(freq, excess), ...]
            - num_violations: 违规频点数量
            - compliance_percentage: 合规频点百分比

    前置条件:
        - len(psd_values) == len(fcc_mask) == len(frequencies)
        - tolerance > 0

    后置条件:
        - 准确率 100%（SC-003）
    """
    # 参数验证
    assert len(psd_values) == len(fcc_mask) == len(frequencies), \
        "psd_values, fcc_mask, frequencies 长度必须一致"
    assert tolerance > 0, f"tolerance 必须 > 0，当前值: {tolerance}"

    # 计算超限量（正值表示超过限制，负值表示低于限制）
    excess = psd_values - fcc_mask

    # 找出最大超限量
    max_excess = float(np.max(excess))

    # 判断合规性状态
    if max_excess < -tolerance:
        # 所有频点都明显低于限制（安全余量 > tolerance）
        status = "Compliant"
    elif max_excess > tolerance:
        # 至少有一个频点明显超过限制
        status = "Non-compliant"
    else:
        # 最大超限量在 [-tolerance, +tolerance] 范围内（临界合规）
        status = "Marginal Compliant"

    # 生成违规频点列表（超限量 > tolerance 的点）
    violation_mask = excess > tolerance
    violation_indices = np.where(violation_mask)[0]
    violations = [
        (float(frequencies[i]), float(excess[i]))
        for i in violation_indices
    ]

    # 计算违规频点数量
    num_violations = len(violations)

    # 计算合规频点百分比（psd <= fcc_mask + tolerance 的点）
    compliant_mask = excess <= tolerance
    num_compliant = np.sum(compliant_mask)
    compliance_percentage = float(num_compliant / len(psd_values) * 100)

    # 返回合规性结果
    result = {
        "status": status,
        "max_excess": max_excess,
        "violations": violations,
        "num_violations": num_violations,
        "compliance_percentage": compliance_percentage,
    }

    return result

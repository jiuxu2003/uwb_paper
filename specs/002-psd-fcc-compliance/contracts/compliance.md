# Contract: FCC Compliance Check

**Feature**: 002-psd-fcc-compliance
**Module**: src/models/fcc.py
**Date**: 2025-12-17

本契约定义了脉冲 PSD 与 FCC 掩蔽罩合规性判断的接口规范。

---

## Function Signature

```python
def check_compliance(
    psd_values: np.ndarray,
    fcc_mask: np.ndarray,
    frequencies: np.ndarray,
    tolerance: float = 0.1
) -> dict:
    """
    判断脉冲 PSD 是否符合 FCC 规范

    参数:
        psd_values: np.ndarray, shape (N,), 脉冲 PSD（dBm/MHz）
        fcc_mask: np.ndarray, shape (N,), FCC 掩蔽罩（dBm/MHz）
        frequencies: np.ndarray, shape (N,), 频率轴（Hz）
        tolerance: float, 边界容差（dB），默认 0.1

    返回:
        result: dict, 包含以下键：
            - "status": str, "Compliant" | "Non-compliant" | "Marginal Compliant"
            - "max_excess": float, 最大超限量（dB）
            - "violations": list[tuple], 违规频点：[(freq, excess), ...]
            - "num_violations": int, 违规频点数量
            - "compliance_percentage": float, 合规频点百分比

    异常:
        ValueError: 如果输入数组长度不一致

    示例:
        >>> result = check_compliance(psd_5th, fcc_mask, freq, tolerance=0.1)
        >>> result["status"]
        "Compliant"
        >>> result["max_excess"]
        -5.2  # 所有频点低于 FCC 限制线至少 5.2 dB
    """
    pass
```

---

## Preconditions

| 参数 | 约束 | 验证 |
|------|------|------|
| 数组长度 | `len(psd_values) == len(fcc_mask) == len(frequencies)` | `assert ...` |
| `tolerance` | > 0 | `assert tolerance > 0` |
| `psd_values` | 非空 | `assert len(psd_values) > 0` |

---

## Postconditions

| 验证项 | 期望值 |
|--------|--------|
| `status` | 三个状态之一 |
| `max_excess` | `max(psd_values - fcc_mask)` |
| `violations` | 如果 `status == "Non-compliant"` 则 `len(violations) > 0` |
| 准确率 | 100%（SC-003） |

---

## Algorithm

### 合规性判断逻辑

```python
def check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1):
    # 计算超限量（dB）
    diff = psd_values - fcc_mask
    max_excess = np.max(diff)

    # 判断状态
    if max_excess < -tolerance:
        status = "Compliant"
        violations = []
    elif max_excess > tolerance:
        status = "Non-compliant"
        # 找出所有违规频点
        violation_indices = np.where(diff > tolerance)[0]
        violations = [
            (frequencies[i], diff[i])
            for i in violation_indices
        ]
    else:  # -tolerance ≤ max_excess ≤ +tolerance
        status = "Marginal Compliant"
        violations = []

    # 计算合规百分比
    compliant_points = np.sum(diff <= tolerance)
    compliance_percentage = 100.0 * compliant_points / len(diff)

    return {
        "status": status,
        "max_excess": float(max_excess),
        "violations": violations,
        "num_violations": len(violations),
        "compliance_percentage": compliance_percentage
    }
```

---

## Status Definitions

| 状态 | 条件 | 说明 |
|------|------|------|
| **Compliant** | `max_excess < -tolerance` | 所有频点低于 FCC 限制线至少 tolerance dB |
| **Non-compliant** | `max_excess > +tolerance` | 存在频点超出 FCC 限制线至少 tolerance dB |
| **Marginal Compliant** | `-tolerance ≤ max_excess ≤ +tolerance` | 在容差范围内接触 FCC 限制线 |

---

## Edge Cases

### 临界情况处理

- **max_excess = 0.099 dB**:
  - `tolerance = 0.1` → "Compliant"
  - 建议：在可视化时标注"接近临界"

- **max_excess = 0.101 dB**:
  - `tolerance = 0.1` → "Marginal Compliant"

- **max_excess = -0.001 dB**:
  - `tolerance = 0.1` → "Marginal Compliant"

### 数值精度

- 使用 `float64` 避免精度损失
- 在比较时使用明确的不等式（`>`, `<`）而非 `>=`, `<=`

---

## Testing Strategy

### Unit Tests

1. **test_compliant_case**:
   - 输入：5 阶脉冲（fc=6.85 GHz）
   - 期望：`status == "Compliant"`

2. **test_non_compliant_case**:
   - 输入：2 阶脉冲（无调制）
   - 期望：`status == "Non-compliant"` and `len(violations) > 0`

3. **test_marginal_case**:
   - 构造 PSD 恰好在 FCC 限制线 ±0.05 dB 内
   - 期望：`status == "Marginal Compliant"`

4. **test_tolerance_effect**:
   - 使用不同 tolerance 值（0.05, 0.1, 0.2）
   - 验证状态判定的变化

5. **test_accuracy**:
   - SC-003 验证：准确率 100%
   - 使用已知合规/不合规的测试用例

---

## Reference

- [research.md](../research.md) Section 2.3: 合规性判断逻辑
- [data-model.md](../data-model.md) Entity 4: Compliance Result
- FCC Part 15.35: Measurement procedures

---

**Contract Status**: ✅ 完成

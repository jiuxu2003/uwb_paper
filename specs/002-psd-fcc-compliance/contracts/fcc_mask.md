# Contract: FCC Indoor Mask Generation

**Feature**: 002-psd-fcc-compliance
**Module**: src/models/fcc.py (新增模块)
**Date**: 2025-12-17

本契约定义了 FCC Part 15.209 室内辐射掩蔽罩的生成规范。

---

## Function Signature

```python
def get_fcc_indoor_mask(
    frequencies: np.ndarray
) -> np.ndarray:
    """
    生成 FCC Part 15.209 室内辐射掩蔽罩

    参数:
        frequencies: np.ndarray, shape (N,), 频率轴（Hz）

    返回:
        mask_dbm_per_mhz: np.ndarray, shape (N,), 功率限制（dBm/MHz）

    异常:
        ValueError: 如果 frequencies 不是递增数组

    示例:
        >>> freq = np.linspace(0, 12e9, 1200)
        >>> mask = get_fcc_indoor_mask(freq)
        >>> mask[(freq >= 3.1e9) & (freq < 10.6e9)][0]
        -41.3  # UWB 合法频段限制
    """
    pass
```

---

## Preconditions

| 参数 | 约束 | 验证 |
|------|------|------|
| `frequencies` | 1D 数组，递增 | `assert np.all(np.diff(frequencies) > 0)` |
| 频率范围 | [0, ∞) | `assert np.all(frequencies >= 0)` |

---

## Postconditions

| 验证项 | 期望值 | 容忍误差 |
|--------|--------|----------|
| 输出长度 | `len(mask) == len(frequencies)` | 严格相等 |
| 频段 1 限制 | `-75.3 dBm/MHz` (0.96-1.61 GHz) | < 0.1 dB |
| 频段 2 限制 | `-53.3 dBm/MHz` (1.61-3.1 GHz) | < 0.1 dB |
| 频段 3 限制 | `-41.3 dBm/MHz` (3.1-10.6 GHz) | < 0.1 dB |
| 频段 4 限制 | `-51.3 dBm/MHz` (>10.6 GHz) | < 0.1 dB |

---

## Algorithm

### FCC 频段定义

```python
FCC_BANDS = [
    (0.96e9, 1.61e9, -75.3),   # 频段 1
    (1.61e9, 3.1e9, -53.3),    # 频段 2
    (3.1e9, 10.6e9, -41.3),    # 频段 3（UWB 合法频段）
    (10.6e9, np.inf, -51.3)    # 频段 4
]
```

### 实现方式（np.piecewise）

```python
def get_fcc_indoor_mask(frequencies):
    # 定义边界条件
    conditions = [
        (frequencies >= 0.96e9) & (frequencies < 1.61e9),
        (frequencies >= 1.61e9) & (frequencies < 3.1e9),
        (frequencies >= 3.1e9) & (frequencies < 10.6e9),
        frequencies >= 10.6e9
    ]

    # 定义对应的功率限制值
    funclist = [-75.3, -53.3, -41.3, -51.3]

    # 默认值（低于 0.96 GHz）
    default = -100.0

    mask = np.piecewise(
        frequencies,
        conditions,
        funclist + [default]
    )

    return mask
```

---

## Edge Cases

### 边界频点处理

- **3.1 GHz 边界**：
  - `f = 3.0999... GHz → -53.3 dBm/MHz`
  - `f = 3.1000... GHz → -41.3 dBm/MHz`
  - 阶跃跳变，非平滑过渡

- **低于 0.96 GHz**：
  - 返回极低功率限制（-100 dBm/MHz）

- **高于 12 GHz**：
  - 继续沿用 `-51.3 dBm/MHz`（频段 4 规则）

---

## Testing Strategy

### Unit Tests

1. **test_fcc_band_values**:
   - 验证每个频段的限制值精度
   - `|mask_value - standard_value| < 0.1` dB

2. **test_boundary_transitions**:
   - 验证边界频点的阶跃特性
   - 例：`mask(3.09999e9) == -53.3` and `mask(3.10001e9) == -41.3`

3. **test_output_length**:
   - 验证输出长度与输入一致
   - `len(mask) == len(frequencies)`

---

## Reference

- FCC Part 15.209: CFR Title 47, Part 15, Subpart C, §15.209
- [research.md](../research.md) Section 2.1: 室内辐射掩蔽罩
- [data-model.md](../data-model.md) Entity 3: FCC Indoor Mask

---

**Contract Status**: ✅ 完成

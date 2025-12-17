# API Contracts: TH-UWB Communication System Simulation

**Feature**: [spec.md](../spec.md) | **Plan**: [plan.md](../plan.md) | **Data Model**: [data-model.md](../data-model.md)
**Phase**: Phase 1 - Design (Contracts)
**Date**: 2025-12-17

## Overview

本目录包含 TH-UWB 仿真系统各模块间的 API 契约定义。契约规定了模块的输入/输出接口、前置条件（Preconditions）、后置条件（Postconditions）、异常处理和性能保证。

## 契约原则

1. **明确性**（Clarity）：所有参数和返回值的类型、单位、值域必须明确
2. **可测试性**（Testability）：所有契约必须可通过单元测试验证
3. **防御性**（Defensive）：输入验证在契约层完成，内部实现可假设输入合法
4. **一致性**（Consistency）：所有模块使用统一的错误处理和日志记录方式

## 模块列表

| 模块 | 契约文档 | 职责 |
|------|---------|------|
| 脉冲生成 | [pulse.md](./pulse.md) | 生成二阶高斯导数脉冲波形 |
| 调制 | [modulation.md](./modulation.md) | TH-PPM 信号调制与跳时序列生成 |
| 信道 | [channel.md](./channel.md) | AWGN/MUI/NBI 信道模型 |
| 接收解调 | [receiver.md](./receiver.md) | 相关接收机与比特判决 |
| 性能分析 | [metrics.md](./metrics.md) | BER 计算与统计分析 |
| 可视化 | [visualization.md](./visualization.md) | 学术论文级图表生成 |

## 数据类型约定

### 标准类型

```python
# 标量类型
TimeSeconds = float  # 时间（秒），例如 100e-9 (100ns)
FrequencyHz = float  # 频率（Hz），例如 2.4e9 (2.4 GHz)
Amplitude = float    # 幅度（无量纲），例如 1.0
PowerDB = float      # 功率（dB），例如 10.0
BER = float          # 误码率（0~1），例如 1e-3

# 数组类型
Signal = np.ndarray[np.float64]  # 时域信号，1D 数组
BitSequence = np.ndarray[np.int32]  # 比特序列，1D 数组，值域 {0, 1}
TimeAxis = np.ndarray[np.float64]  # 时间轴，1D 数组，单位秒
```

### 配置对象

所有模块接受 `SystemConfig` 作为配置源（参考 [data-model.md](../data-model.md#11-systemconfig系统配置)）。

### 随机数生成器

所有需要随机性的函数接受 `rng: np.random.Generator` 参数，确保可重复性。

## 错误处理约定

### 异常类型

```python
class UWBSimulationError(Exception):
    """UWB 仿真系统基础异常"""
    pass

class ConfigurationError(UWBSimulationError):
    """配置参数错误"""
    pass

class SignalProcessingError(UWBSimulationError):
    """信号处理错误"""
    pass

class VisualizationError(UWBSimulationError):
    """可视化错误"""
    pass
```

### 验证规则

- 所有公共函数必须验证输入参数的类型和值域
- 验证失败抛出 `ConfigurationError` 或 `ValueError`
- 内部逻辑错误抛出 `SignalProcessingError` 或 `RuntimeError`

## 性能契约

所有性能目标基于 Success Criteria SC-006：

| 操作 | 性能目标 | 测量方式 |
|------|---------|---------|
| 单点仿真（10^4 比特） | < 1 分钟 | time.perf_counter() |
| 完整曲线（10 个点） | < 15 分钟 | 累积时间 |
| 脉冲生成 | < 10 ms | 一次性开销 |
| 图表生成 | < 5 秒/张 | 包括保存到磁盘 |

## 测试要求

每个契约必须有对应的测试文件：

```
tests/unit/
├── test_pulse.py      # 测试 pulse.md 中的所有契约
├── test_modulation.py # 测试 modulation.md 中的所有契约
├── test_channel.py    # 测试 channel.md 中的所有契约
├── test_receiver.py   # 测试 receiver.md 中的所有契约
├── test_metrics.py    # 测试 metrics.md 中的所有契约
└── test_visualization.py  # 测试 visualization.md 中的所有契约
```

---

**契约版本**: 1.0.0
**最后更新**: 2025-12-17
**维护者**: Claude Sonnet 4.5

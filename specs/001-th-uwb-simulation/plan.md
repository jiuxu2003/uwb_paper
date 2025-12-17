# Implementation Plan: TH-UWB Communication System Simulation

**Branch**: `001-th-uwb-simulation` | **Date**: 2025-12-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-th-uwb-simulation/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

模拟一个完整的 TH-UWB（跳时-脉冲位置调制）通信系统，用于分析多用户干扰（MUI）和窄带干扰（NBI）对系统性能的影响。系统将生成符合 TH-PPM 调制格式的 UWB 信号，使用二阶高斯导数脉冲作为基础波形，为每个用户分配唯一的跳时序列。通过相关接收机解调，计算误码率（BER），并生成三种学术论文级别的性能曲线图：信号时域波形图、BER vs 用户数量曲线、BER vs 信干比曲线。技术实现采用 Python 科学计算生态（NumPy/SciPy/Matplotlib），确保仿真结果具有统计稳定性和可重复性。

## Technical Context

**Language/Version**: Python 3.11+（需要现代 type hints 支持和性能优化特性）
**Primary Dependencies**: NumPy 1.24+（信号处理/矩阵运算）、SciPy 1.10+（高斯脉冲生成/信号分析）、Matplotlib 3.7+（学术级可视化）
**Storage**: 文件系统（保存生成的图表为 PNG/PDF，分辨率 ≥300 DPI）
**Testing**: pytest（单元测试核心算法模块）、NumPy.testing（数值精度验证）
**Target Platform**: 跨平台（Linux/Windows/macOS，纯 Python 科学计算栈）
**Project Type**: Single（科学计算/仿真项目，无前后端分离需求）
**Performance Goals**: 单点仿真（10^4 比特）< 1 分钟，完整性能曲线（10 个点）< 15 分钟（SC-006）
**Constraints**: 统计稳定性（每个性能点 ≥10^4 发送比特，误码率相对误差 <20%）、图表质量（≥300 DPI，适合论文打印）
**Scale/Scope**: 支持 1-20 个用户并发、SNR 范围 -5dB 至 20dB、SIR 范围 -10dB 至 30dB、3 种核心可视化输出

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

验证以下原则是否得到遵守（参考 `.specify/memory/constitution.md`）：

- [x] **生态复用优先**: 是 - 使用 NumPy/SciPy/Matplotlib 成熟生态，所有依赖锁定明确版本号（NumPy 1.24+、SciPy 1.10+、Matplotlib 3.7+）
- [x] **质量第一**: 是 - 技术方案基于 Python 科学计算标准实践，所有性能目标可通过 benchmark 验证（SC-006）
- [x] **工具优先**: 是 - 使用 pytest 作为测试框架，NumPy.testing 验证数值精度，linter/formatter 待 Phase 1 确定（black + pylint）
- [x] **透明记录**: 是 - 关键决策记录在 specs/001-th-uwb-simulation/，代码需包含中文文档（待实施验证）
- [x] **结果导向**: 是 - 定义了可测量成功标准（8 个 SC 指标，包括 BER 范围、统计稳定性、性能时间约束）
- [x] **简单优先**: 是 - 避免过度设计，采用 Single Project 结构，相关接收机假设完美同步（简化复杂度）
- [x] **向后兼容**: N/A - 新项目无向后兼容性问题
- [x] **安全第一**: 是 - 纯本地科学计算，无用户输入验证需求，无密钥管理需求
- [x] **版本控制与远程同步**: 是 - `.gitignore` 已配置，commit message 将遵循约定式提交格式

如有违反，必须在"Complexity Tracking"部分说明理由。

## Project Structure

### Documentation (this feature)

```text
specs/001-th-uwb-simulation/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Single project structure (科学计算/仿真项目)
src/
├── models/              # 核心数学模型
│   ├── pulse.py         # 二阶高斯导数脉冲生成器
│   ├── modulation.py    # TH-PPM 调制器
│   └── channel.py       # 信道模型（AWGN + MUI + NBI）
├── simulation/          # 仿真引擎
│   ├── transmitter.py   # 多用户发射机
│   ├── receiver.py      # 相关接收机
│   └── metrics.py       # BER 计算与统计分析
├── visualization/       # 学术级可视化
│   ├── waveform.py      # 时域波形图生成
│   └── performance.py   # 性能曲线图生成（BER vs 用户数/SIR）
└── config.py            # 系统参数配置（Tf, Tc, δ, 脉冲参数等）

tests/
├── unit/                # 单元测试
│   ├── test_pulse.py    # 脉冲波形验证
│   ├── test_modulation.py  # 调制解调验证
│   └── test_channel.py  # 信道模型验证
└── integration/         # 集成测试
    └── test_end_to_end.py  # 完整仿真流程验证

scripts/                 # 运行脚本
├── run_mui_analysis.py  # 多用户干扰分析
├── run_nbi_analysis.py  # 窄带干扰分析
└── generate_figures.py  # 批量生成论文图表

outputs/                 # 生成的图表输出（Git ignored）
├── waveforms/
├── ber_vs_users/
└── ber_vs_sir/

requirements.txt         # Python 依赖清单
pyproject.toml           # 项目元数据与工具配置
README.md                # 项目说明与使用指南
```

**Structure Decision**: 选择 Single Project 结构（Option 1）。理由：
1. 纯科学计算项目，无前后端分离需求
2. 所有模块高度耦合（信号生成 → 信道传输 → 接收解调 → 性能分析）
3. 主要交付物是可重复的仿真脚本和学术图表，不是生产服务
4. 符合"简单优先"原则，避免过度设计的多项目架构

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

无违反项 - 所有宪法原则均已遵守。

---

## Phase 0: Research (✅ Completed)

**目标**: 解决技术未知项，确保实现方案的可行性。

**输出文档**: [research.md](./research.md)

**研究主题**：
1. ✅ 二阶高斯导数脉冲的 NumPy/SciPy 实现方法
2. ✅ TH-PPM 调制原理与跳时序列生成算法
3. ✅ AWGN/MUI/NBI 信道模型的实现
4. ✅ 相关接收机算法细节
5. ✅ Matplotlib 学术论文级图表配置标准
6. ✅ NumPy 随机数种子管理（确保可重复性）

**关键发现**：
- 二阶高斯导数脉冲公式：`g''(t) = A · (1 - 4π(t/τ)²) · exp(-2π(t/τ)²)`
- 采样频率要求：≥50 GHz（Nyquist 定理）
- TH-PPM 信号表达式：`s(t) = Σ[j=0 to Nf-1] g''(t - jTf - c_j·Tc - d_j·δ)`
- Matplotlib DPI 配置：`savefig.dpi = 300`（满足论文打印要求）
- NumPy SeedSequence 确保多用户独立随机性：`rng = np.random.default_rng([user_id, seed])`

**技术可行性确认**：✅ 所有核心功能均有成熟的 Python 科学计算解决方案

---

## Phase 1: Design (✅ Completed)

**目标**: 设计数据模型、API 契约和系统架构。

**输出文档**：
- [data-model.md](./data-model.md) - 核心实体设计
- [contracts/](./contracts/) - API 契约定义
- [quickstart.md](./quickstart.md) - 快速开始指南

### 核心实体（8 个）

1. **SystemConfig**：系统配置单一来源，包含所有参数（Tf, Tc, δ, τ, fs, 等）
2. **Pulse**：二阶高斯导数脉冲模板，预计算并复用
3. **TimeHoppingCode**：用户专属跳时序列，伪随机生成
4. **User**：通信用户，包含跳时码、数据比特、信号生成能力
5. **Channel**：信道模型，支持 AWGN/MUI/NBI 三种干扰
6. **Receiver**：相关接收机，实现模板匹配和比特判决
7. **PerformanceMetrics**：性能指标，计算 BER 和置信区间
8. **SimulationResult**：仿真结果封装，用于持久化和可视化

### 模块契约（6 个）

- `pulse.py` - 脉冲生成模块（3 个函数/方法）
- `modulation.py` - TH-PPM 调制模块（2 个函数/方法）
- `channel.py` - 信道模块（3 个方法）
- `receiver.py` - 接收解调模块（2 个方法）
- `metrics.py` - 性能分析模块（4 个方法/属性）
- `visualization.py` - 可视化模块（3 个图表生成函数）

### 设计原则验证

- ✅ 简单优先：使用 Python dataclass，避免复杂抽象
- ✅ 单一职责：每个实体职责清晰，模块边界明确
- ✅ 依赖注入：所有模块接受 SystemConfig，确保配置一致性
- ✅ 可测试性：所有契约包含前置/后置条件和测试用例

---

## Phase 2: Implementation Readiness Assessment (✅ Passed)

### Constitution Check (二次验证)

经过 Phase 0 研究和 Phase 1 设计后，重新验证宪法原则：

- ✅ **生态复用优先**：确认 NumPy/SciPy/Matplotlib 可满足所有需求，无需自研组件
- ✅ **质量第一**：所有技术方案基于官方文档和学术文献（research.md 包含引用）
- ✅ **工具优先**：确定使用 pytest + black + pylint 工具链
- ✅ **透明记录**：已生成完整的 research/data-model/contracts 文档
- ✅ **结果导向**：所有 8 个 SC 指标可量化验证
- ✅ **简单优先**：data-model.md 验证无过度设计（Section 6.2）
- ✅ **安全第一**：纯本地科学计算，无安全风险
- ✅ **版本控制与远程同步**：outputs/ 已添加到 .gitignore

### 依赖版本锁定（最终确认）

```txt
# requirements.txt
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
pytest>=7.0.0,<8.0.0

# 开发工具（可选）
black>=23.0.0
pylint>=3.0.0
```

### 性能预估（基于 research.md 分析）

| 操作 | 预估时间 | 性能目标 | 状态 |
|------|---------|---------|------|
| 脉冲生成（一次） | <10 ms | N/A | ✅ 可达成 |
| 单用户信号生成（10^4 比特） | <5 秒 | N/A | ✅ 可达成 |
| 单点仿真（10^4 比特，1 用户） | <10 秒 | <60 秒 | ✅ 可达成 |
| 单点仿真（10^4 比特，10 用户） | <50 秒 | <60 秒 | ✅ 可达成 |
| 完整曲线（10 点） | <10 分钟 | <15 分钟 | ✅ 可达成 |

**瓶颈分析**：
- 主要耗时：用户信号生成（逐帧脉冲叠加）+ 相关接收（逐帧模板匹配）
- 优化策略：向量化运算（NumPy 广播）、脉冲模板复用、批量处理

### 风险评估与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 内存溢出（50 GHz × 10^4 比特） | 高 | 中 | 分块处理（1000 帧/批次） |
| 性能不达标（SC-006） | 中 | 低 | 降低采样率至 25 GHz（需权衡精度） |
| 跳时码碰撞率过高 | 中 | 低 | 使用 Gold 序列替代伪随机（超出当前范围） |
| 数值精度误差 | 低 | 中 | NumPy.testing 验证，设置容差阈值 |

---

## Implementation Plan Summary

### 准备就绪确认

- ✅ 所有技术未知项已解决（Phase 0）
- ✅ 数据模型设计完成（Phase 1）
- ✅ API 契约定义完成（Phase 1）
- ✅ 快速开始指南完成（Phase 1）
- ✅ 宪法原则验证通过（Phase 2）
- ✅ 性能可行性评估通过（Phase 2）
- ✅ 依赖版本锁定完成（Phase 2）

### 下一步：任务分解（Phase 3）

运行 `/speckit.tasks` 命令生成 `tasks.md`，将实现计划分解为可执行的任务列表：

```bash
/speckit.tasks
```

**预期输出**：
- 任务优先级排序（按 User Story 优先级）
- 任务依赖关系图（明确哪些任务必须串行）
- 估算工作量（story points）
- 验收标准（来自 spec.md 的 acceptance scenarios）

### 成功标准验收计划

基于 spec.md 的 Success Criteria (SC-001 to SC-008)：

| SC | 验收方式 | 责任模块 |
|----|---------|---------|
| SC-001 | 目视检查时域波形图（≥3 帧清晰） | visualization.py |
| SC-002 | 验证 BER-用户数曲线趋势（10^-3 → 10^-1） | 集成测试 |
| SC-003 | 验证 BER-SIR 曲线（SIR=-10dB 时 BER<0.4） | 集成测试 |
| SC-004 | 统计稳定性检查（10^4 比特，相对误差<20%） | metrics.py 单元测试 |
| SC-005 | 图表元素完整性检查（标题/轴/网格/图例） | visualization.py 单元测试 |
| SC-006 | 性能基准测试（time.perf_counter()） | 集成测试 + CI/CD |
| SC-007 | 代码审查检查表（注释/模块化/可读性） | 代码审查流程 |
| SC-008 | 可重复性测试（相同种子 → 相同输出） | 集成测试 |

---

## 附录：关键决策记录

### 决策 1：采样频率选择 50 GHz

**背景**：需平衡精度和计算成本。

**选项**：
- A: 25 GHz（更快，精度降低）
- B: 50 GHz（标准，平衡）
- C: 100 GHz（最高精度，最慢）

**决策**：选择 B (50 GHz)

**理由**：
1. 满足 Nyquist 定理（脉冲最高频率约 10 GHz）
2. 主流 UWB 仿真文献标准
3. 性能可达标（SC-006）

**参考**：research.md Section 1.2

---

### 决策 2：跳时序列生成算法

**背景**：需确保多用户码序列独立性。

**选项**：
- A: 简单伪随机（`np.random.integers`）
- B: Gold 序列（准正交）
- C: Kasami 序列（最优正交性）

**决策**：选择 A（简单伪随机）

**理由**：
1. 符合"简单优先"原则
2. Gold/Kasami 序列生成复杂，超出当前范围
3. 使用 SeedSequence 确保用户间独立性
4. 若后续发现碰撞率过高，可升级为 B

**参考**：data-model.md Section 1.3, research.md Section 2.1

---

### 决策 3：相关接收机假设完美同步

**背景**：时序同步是 UWB 系统的难点。

**选项**：
- A: 完美同步（已知跳时码和时序）
- B: 半盲同步（已知跳时码，估计时序）
- C: 全盲同步（估计跳时码和时序）

**决策**：选择 A（完美同步）

**理由**：
1. spec.md FR-008 明确要求完美同步假设
2. 降低实现复杂度，聚焦 MUI/NBI 分析
3. 若需扩展为半盲/全盲，可在 Phase 1 后增加

**参考**：spec.md FR-008, data-model.md Section 1.6

---

**规划完成日期**: 2025-12-17
**规划者**: Claude Sonnet 4.5
**审查状态**: ✅ Approved - Ready for Implementation
**下一步**: 运行 `/speckit.tasks` 生成任务清单

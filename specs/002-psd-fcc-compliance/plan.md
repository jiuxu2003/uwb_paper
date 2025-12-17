# Implementation Plan: UWB 脉冲功率谱密度与 FCC 合规性分析可视化

**Branch**: `002-psd-fcc-compliance` | **Date**: 2025-12-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-psd-fcc-compliance/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

本特性为 UWB 论文添加 FCC Part 15.209 合规性验证可视化功能。核心需求：
1. 计算 2 阶和 5 阶高斯导数脉冲的功率谱密度（PSD）
2. 绘制 FCC 室内辐射掩蔽罩（Indoor Mask）曲线
3. 在同一图表上对比脉冲 PSD 与 FCC 限制线
4. 自动判断脉冲设计的合规性状态
5. 生成符合 IEEE 学术出版标准的图表（≥300 DPI）

技术方法：利用 FFT 将时域脉冲波形转换为频域 PSD，使用 matplotlib 绘制对比图表，通过逐频点比较实现合规性验证。

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: numpy>=1.24.0, scipy>=1.10.0, matplotlib>=3.7.0
**Storage**: N/A（仅输出 PNG 图表文件到 `outputs/` 目录）
**Testing**: pytest>=7.0.0（单元测试 + 集成测试）
**Target Platform**: Linux/macOS（本地开发环境，无头服务器支持）
**Project Type**: Single project（学术仿真项目）
**Performance Goals**:
- PSD 计算：单个脉冲 < 1 秒（0-12 GHz, 10 MHz 分辨率）
- 图表生成：< 2 秒（包含渲染和保存）
- 整体流程：< 10 秒（从参数输入到图表保存，符合 SC-005）

**Constraints**:
- 频率分辨率 ≥10 MHz（SC-002，即 0-12 GHz 范围内 ≥1200 采样点）
- 图表分辨率 ≥300 DPI（SC-004，适合学术论文打印）
- FCC 掩蔽罩精度误差 <0.1 dB（SC-001）
- 合规性判断准确率 100%（SC-003）
- 字体配置：中文使用宋体（SimSun），英文/数字使用 Times New Roman（Constitution X）

**Scale/Scope**:
- 2 种脉冲阶数（2 阶和 5 阶高斯导数脉冲）
- 4 个 FCC 频段（0.96-1.61 GHz, 1.61-3.1 GHz, 3.1-10.6 GHz, >10.6 GHz）
- 1 张核心图表（PSD vs FCC Mask）
- 预计新增代码：~300-500 行（包括脉冲生成、PSD 计算、FCC 掩蔽罩、合规性判断、可视化）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

验证以下原则是否得到遵守（参考 `.specify/memory/constitution.md`）：

- [x] **生态复用优先**: ✅ 使用 numpy.fft 进行 FFT 计算，matplotlib 进行可视化，scipy 用于信号处理，所有依赖锁定版本号（numpy>=1.24.0,<2.0.0）
- [x] **质量第一**: ✅ 技术方案基于 FCC Part 15.209 官方文档，FFT 参数选择有采样定理支撑，频率分辨率有明确计算公式
- [x] **工具优先**: ✅ 使用 pytest 进行测试，black 进行格式化（line-length=100），pylint 进行代码检查
- [x] **透明记录**: ✅ 关键决策记录在 specs/002-psd-fcc-compliance/（spec.md, plan.md, research.md），代码包含中文文档
- [x] **结果导向**: ✅ 成功标准明确定义在 spec.md SC-001 到 SC-008（精度、分辨率、准确率、执行时间等量化指标）
- [x] **简单优先**: ✅ 避免过度设计，直接使用 numpy.fft.rfft 进行 PSD 计算，无需引入复杂信号处理库
- [x] **向后兼容**: ✅ 新功能为独立模块，不修改现有 src/models/pulse.py，仅扩展功能，无破坏性变更
- [x] **安全第一**: ✅ 无用户输入验证需求（脉冲参数由代码直接设置），无敏感信息处理
- [x] **版本控制与远程同步**: ✅ .gitignore 已配置排除 .specify/, __pycache__/, outputs/*.png（仅提交代码，不提交生成图表）
- [x] **学术规范与可视化标准**: ✅ 字体配置遵循 Constitution X（中文宋体 + 英文 Times New Roman），图表分辨率 ≥300 DPI，符合 IEEE 出版标准

**Constitution Check 结果**: ✅ 全部通过，无违规项

## Project Structure

### Documentation (this feature)

```text
specs/002-psd-fcc-compliance/
├── spec.md              # Feature specification ✅ 已完成
├── plan.md              # This file (Implementation plan)
├── research.md          # Phase 0 output (Research findings)
├── data-model.md        # Phase 1 output (Entity definitions)
├── quickstart.md        # Phase 1 output (Usage guide)
├── contracts/           # Phase 1 output (API contracts)
│   ├── pulse.md         # 脉冲生成契约
│   ├── psd.md           # PSD 计算契约
│   ├── fcc_mask.md      # FCC 掩蔽罩契约
│   └── compliance.md    # 合规性判断契约
└── checklists/
    └── requirements.md  # Quality checklist ✅ 已完成
```

### Source Code (repository root)

本项目为 **Single project** 结构（学术仿真项目）：

```text
src/
├── models/              # 数据模型（现有）
│   ├── pulse.py         # 脉冲模型（现有，需扩展高斯导数脉冲）
│   ├── modulation.py    # 调制模型（现有）
│   ├── channel.py       # 信道模型（现有）
│   ├── psd.py           # 新增：PSD 计算模块
│   └── fcc.py           # 新增：FCC 掩蔽罩模块
├── simulation/          # 仿真核心（现有）
│   ├── receiver.py      # 接收器（现有）
│   └── metrics.py       # 性能指标（现有）
├── visualization/       # 可视化模块（现有）
│   ├── waveform.py      # 波形可视化（现有）
│   ├── performance.py   # 性能可视化（现有）
│   └── compliance.py    # 新增：合规性可视化（PSD vs FCC Mask）
├── config.py            # 系统配置（现有）
└── __init__.py          # 包初始化（现有）

scripts/
├── demo_nbi_analysis.py         # 现有脚本
├── run_nbi_analysis.py          # 现有脚本
├── run_mui_analysis.py          # 现有脚本
├── generate_figures.py          # 现有脚本
└── plot_psd_fcc_compliance.py   # 新增：生成 PSD vs FCC Mask 图表

tests/
├── unit/                # 单元测试
│   ├── test_pulse.py    # 脉冲模型测试（现有，需扩展）
│   ├── test_psd.py      # 新增：PSD 计算测试
│   ├── test_fcc.py      # 新增：FCC 掩蔽罩测试
│   └── test_compliance_viz.py  # 新增：合规性可视化测试
└── integration/         # 集成测试
    └── test_psd_fcc_integration.py  # 新增：端到端集成测试

outputs/                 # 输出目录
└── psd_fcc_compliance_2nd_5th.png  # 新增：PSD vs FCC Mask 图表（.gitignore）
```

**Structure Decision**:
- 选择 **Option 1: Single project** 结构，符合当前学术仿真项目的组织方式
- 新增模块（psd.py, fcc.py, compliance.py）遵循现有代码组织规范
- 脚本命名遵循现有约定（plot_*.py 用于可视化脚本）
- 测试目录按功能组织（unit/ 和 integration/）

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**无违规项** - Constitution Check 全部通过，无需记录复杂性例外。

---

## Phase 0: Research & Technology Decisions

**Status**: ✅ Complete

### Unknowns to Research

基于 Technical Context 中的明确需求，以下方面需要深入研究：

1. **高斯导数脉冲的 PSD 计算方法**
   - 2 阶和 5 阶高斯导数脉冲的时域公式
   - 如何选择脉冲中心频率（fc）以满足 FCC 合规性
   - 脉冲宽度（τ）对频谱展宽的影响
   - FFT 参数选择（采样率、采样点数）以达到 10 MHz 频率分辨率

2. **FCC Part 15.209 标准的精确定义**
   - 室内辐射掩蔽罩（Indoor Mask）的 4 个频段及对应功率限制
   - 功率单位转换（dBm/MHz 的物理意义）
   - 边界频点处的处理方式（阶跃 vs 平滑过渡）

3. **合规性判断算法**
   - 逐频点比较的实现方式
   - 边界容差设置（SC-001 要求误差 <0.1 dB）
   - "临界合规"的判定标准

4. **matplotlib 学术标准配置**
   - 字体配置（中文宋体 + 英文 Times New Roman）
   - DPI 设置以满足 ≥300 DPI 要求
   - IEEE 论文排版标准（字体大小、线宽、标记样式）

### Research Tasks

1. **研究高斯导数脉冲的频域特性**
   - 查阅 UWB 领域学术文献（IEEE Trans. Communications, IEEE Trans. Wireless Communications）
   - 推导 2 阶和 5 阶高斯导数脉冲的 PSD 解析表达式
   - 确定中心频率 fc 和脉冲宽度 τ 的关系

2. **验证 FCC Part 15.209 标准数值**
   - 查阅 FCC 官方文档（CFR Title 47, Part 15.209）
   - 确认 4 个频段的准确功率限制值
   - 了解测量方法（RBW, measurement bandwidth）

3. **调研 matplotlib 学术标准配置最佳实践**
   - 查阅 matplotlib 官方文档关于字体配置的部分
   - 调研 IEEE、Springer 等出版社对图表格式的要求
   - 验证 SimSun 和 Times New Roman 字体在 Linux 环境的可用性

**Output**: research.md（包含所有研究发现、技术决策和证据链接）

---

## Phase 1: Design & Contracts

**Status**: ✅ Complete
**Prerequisites**: research.md 完成 ✅

### Data Model Design

从 spec.md 提取 4 个关键实体，详细设计见 data-model.md：

1. **Gaussian Derivative Pulse（高斯导数脉冲）**
   - 属性：阶数（n=2, 5），脉冲宽度（τ），中心频率（fc），幅度归一化参数
   - 方法：generate_waveform()，get_time_axis()

2. **Power Spectral Density (PSD)（功率谱密度）**
   - 属性：频率轴（0-12 GHz），功率值（dBm/MHz）
   - 方法：compute_from_pulse()，to_db()

3. **FCC Indoor Mask（FCC 室内掩蔽罩）**
   - 属性：频段列表（4 个），功率限制值（dBm/MHz）
   - 方法：get_limit_at_frequency()，plot_mask()

4. **Compliance Result（合规性结果）**
   - 属性：合规状态（"合规"/"不合规"/"临界合规"），违规频点列表，最大超限量（dB）
   - 方法：check_compliance()，generate_report()

### API Contracts

基于功能需求（FR-001 到 FR-008），生成以下契约：

1. **contracts/pulse.md** - 脉冲生成接口
   - `generate_gaussian_derivative_pulse(n, tau, fc, sampling_rate) -> waveform, time_axis`
   - 前置条件：n ∈ {2, 5}，τ ∈ [0.1ns, 2ns]，fc > 0
   - 后置条件：len(waveform) 基于采样率和脉冲宽度计算，能量归一化

2. **contracts/psd.md** - PSD 计算接口
   - `compute_psd(waveform, sampling_rate, freq_resolution=10e6) -> frequencies, psd_dbm_per_mhz`
   - 前置条件：waveform 为 1D 数组，freq_resolution ≥10 MHz
   - 后置条件：len(frequencies) ≥1200（0-12 GHz），psd 单位为 dBm/MHz

3. **contracts/fcc_mask.md** - FCC 掩蔽罩接口
   - `get_fcc_indoor_mask(frequencies) -> mask_dbm_per_mhz`
   - 前置条件：frequencies 为递增数组
   - 后置条件：mask 包含 4 个频段限制值，阶跃处理

4. **contracts/compliance.md** - 合规性判断接口
   - `check_compliance(psd, fcc_mask, tolerance=0.1) -> ComplianceResult`
   - 前置条件：len(psd) == len(fcc_mask)
   - 后置条件：准确率 100%（SC-003）

### Quickstart Guide

生成 quickstart.md，内容包括：
- 环境配置（Python 3.11+, 依赖安装）
- 运行示例脚本：`python scripts/plot_psd_fcc_compliance.py`
- 输出文件位置：`outputs/psd_fcc_compliance_2nd_5th.png`
- 参数调整指南（如何修改中心频率 fc 和脉冲宽度 τ）

### Agent Context Update

运行 `.specify/scripts/bash/update-agent-context.sh claude` 更新 AI 助手上下文：
- 添加新技术：高斯导数脉冲、PSD 计算、FCC Part 15.209 标准
- 添加新模块：src/models/psd.py, src/models/fcc.py, src/visualization/compliance.py
- 保留 CLAUDE.md 中的手动添加内容（marker 之间的部分）

**Output**: data-model.md, contracts/, quickstart.md, CLAUDE.md (updated)

---

## Phase 2: Task Breakdown

**Status**: Not Started（由 `/speckit.tasks` 命令执行，不在本计划范围内）

任务分解将在执行 `/speckit.tasks` 命令时生成，输出到 `specs/002-psd-fcc-compliance/tasks.md`。

---

## Validation Checkpoints

### After Phase 0 (Research Complete)
- [ ] research.md 包含所有 4 个研究主题的结论
- [ ] 所有技术决策有证据支撑（文献引用、官方文档链接）
- [ ] 高斯导数脉冲的 PSD 公式经过数学推导验证
- [ ] FCC 掩蔽罩数值与官方标准一致（误差 <0.1 dB）

### After Phase 1 (Design Complete)
- [ ] data-model.md 定义了所有 4 个实体的属性和方法
- [ ] contracts/ 包含 4 个契约文件，前置/后置条件明确
- [ ] quickstart.md 提供可运行的示例代码
- [ ] Constitution Check 重新验证（特别是"生态复用优先"和"简单优先"）

### Before Phase 2 (Tasks Generation)
- [ ] 所有 Phase 0 和 Phase 1 的输出文件已生成并通过评审
- [ ] 无遗留的 "NEEDS CLARIFICATION" 标记
- [ ] 技术方案已与 spec.md 中的成功标准对齐

---

**Plan Status**: ✅ Phase 0 和 Phase 1 已完成
**Next Step**: 执行 `/speckit.tasks` 生成任务分解（tasks.md）

# Task Breakdown: UWB 脉冲功率谱密度与 FCC 合规性分析可视化

**Feature**: 002-psd-fcc-compliance
**Branch**: `002-psd-fcc-compliance`
**Date**: 2025-12-17

本文档将设计分解为具体的开发任务，按用户故事优先级组织。

---

## Task Summary

- **Total Tasks**: 25
- **Parallelizable Tasks**: 12（标记为 [P]）
- **User Stories**: 3（P1, P2, P3）
- **Phases**: 6（Setup, Foundational, US1, US2, US3, Polish）

**MVP Scope**: User Story 1（P1 - FCC 合规性验证）- 12 tasks

---

## Implementation Strategy

### Incremental Delivery

本项目采用**用户故事优先级增量交付**策略：

1. **MVP (Minimum Viable Product)**: 仅实现 P1 用户故事
   - 范围：FCC 合规性验证可视化
   - 价值：核心功能，立即可用
   - 交付物：能生成 2 阶和 5 阶脉冲 PSD vs FCC Mask 对比图

2. **Second Increment**: 添加 P2 用户故事
   - 范围：脉冲对比分析
   - 价值：增强可读性
   - 依赖：P1 完成

3. **Full Feature**: 添加 P3 用户故事
   - 范围：参数优化
   - 价值：高级功能
   - 依赖：P1, P2 完成

### Independent Testing

每个用户故事都有独立的测试标准：

- **P1**: 生成包含 FCC 掩蔽罩和 2 种脉冲 PSD 的图表，合规性判断准确
- **P2**: 图表中 2 种脉冲使用不同颜色/线型区分
- **P3**: 支持参数调整并实时更新图表

---

## Phase 1: Setup（项目初始化）

**Goal**: 配置开发环境和项目结构

**Duration**: ~30 minutes

### Tasks

- [ ] T001 创建新增模块的目录结构（src/models/, src/visualization/）
- [ ] T002 安装必要的字体（SimSun, Times New Roman）并验证可用性
- [ ] T003 [P] 配置 matplotlib 学术标准（字体、DPI、样式）in src/visualization/config.py
- [ ] T004 [P] 创建单元测试目录（tests/unit/test_psd.py, tests/unit/test_fcc.py）

**Validation**:
- [ ] 目录结构与 plan.md 一致
- [ ] 字体验证脚本通过（matplotlib.font_manager 检测）
- [ ] matplotlib 配置符合 Constitution X 标准

---

## Phase 2: Foundational（基础组件 - 阻塞性前置任务）

**Goal**: 实现所有用户故事共享的基础模块

**Duration**: ~2 hours

**Blocking**: 必须在任何用户故事开始前完成

### Tasks

#### 2.1 高斯导数脉冲生成（所有故事的前提）

- [ ] T005 实现 2 阶高斯导数脉冲生成函数 in src/models/psd.py
  - 函数签名：`generate_gaussian_derivative_pulse(n, tau, fc, sampling_rate)`
  - 归一化系数计算：A_2 = sqrt(2 / (sqrt(π) * tau))
  - 时域公式：A_2 * (1 - t²/τ²) * exp(-t²/(2τ²))
  - 载波调制（如果 fc > 0）

- [ ] T006 实现 5 阶高斯导数脉冲生成函数 in src/models/psd.py
  - 归一化系数计算：A_5 = sqrt(32 / (sqrt(π) * tau))
  - 时域公式：A_5 * [15t/τ² - 10t³/τ⁴ + t⁵/τ⁶] * exp(-t²/(2τ²))
  - 能量归一化验证：∫|waveform|² dt ≈ 1（误差 < 1%）

- [ ] T007 单元测试：脉冲能量归一化验证 in tests/unit/test_pulse.py
  - 测试 2 阶和 5 阶脉冲能量守恒
  - 断言：0.99 <= energy <= 1.01

#### 2.2 PSD 计算模块（所有故事的前提）

- [ ] T008 实现 PSD 计算函数 in src/models/psd.py
  - 函数签名：`compute_psd(waveform, sampling_rate, freq_resolution)`
  - FFT 点数计算：n_fft = 2^ceil(log2(sampling_rate / freq_resolution))
  - 实数 FFT：np.fft.rfft(waveform, n=n_fft)
  - 单边功率谱归一化
  - 转换为 dBm/MHz

- [ ] T009 单元测试：PSD 能量守恒验证（Parseval 定理）in tests/unit/test_psd.py
  - 验证：∫PSD df ≈ ∫|waveform|² dt
  - 容忍误差：< 1%

#### 2.3 FCC 掩蔽罩模块（所有故事的前提）

- [ ] T010 实现 FCC 室内掩蔽罩生成函数 in src/models/fcc.py
  - 函数签名：`get_fcc_indoor_mask(frequencies)`
  - 4 个频段定义：(0.96-1.61 GHz: -75.3), (1.61-3.1 GHz: -53.3), (3.1-10.6 GHz: -41.3), (>10.6 GHz: -51.3)
  - 使用 np.piecewise() 实现阶跃边界

- [ ] T011 单元测试：FCC 掩蔽罩数值精度验证 in tests/unit/test_fcc.py
  - 验证每个频段限制值与标准误差 < 0.1 dB
  - 验证边界频点阶跃特性

**Validation**:
- [ ] 所有单元测试通过（pytest tests/unit/）
- [ ] 脉冲能量归一化（误差 < 1%）
- [ ] PSD 频率分辨率 ≤ 10 MHz
- [ ] FCC 掩蔽罩数值精度 < 0.1 dB

---

## Phase 3: User Story 1（P1 - 验证脉冲设计的 FCC 合规性）

**Priority**: P1（核心价值）

**Goal**: 生成包含 2 阶和 5 阶脉冲 PSD 与 FCC 掩蔽罩的对比图表，自动判断合规性状态。

**Independent Test**: 运行脚本 `python scripts/plot_psd_fcc_compliance.py`，生成图表 `outputs/psd_fcc_compliance_2nd_5th.png`，图表包含：
- FCC 掩蔽罩（红色点线）
- 2 阶脉冲 PSD（蓝色实线，标注"不合规"）
- 5 阶脉冲 PSD（绿色虚线，标注"合规"）
- 合规性判断准确率 100%

**Duration**: ~3 hours

### Tasks

#### 3.1 合规性判断模块

- [ ] T012 [P] [US1] 实现合规性判断函数 in src/models/fcc.py
  - 函数签名：`check_compliance(psd_values, fcc_mask, frequencies, tolerance=0.1)`
  - 计算超限量：diff = psd_values - fcc_mask
  - 状态判定：Compliant / Non-compliant / Marginal Compliant
  - 返回违规频点列表

- [ ] T013 [US1] 单元测试：合规性判断准确率验证 in tests/unit/test_fcc.py
  - 测试 5 阶脉冲（fc=6.85 GHz）→ "Compliant"
  - 测试 2 阶脉冲（fc=0）→ "Non-compliant"
  - 准确率：100%（SC-003）

#### 3.2 可视化模块（核心功能）

- [ ] T014 [P] [US1] 实现合规性可视化函数 in src/visualization/compliance.py
  - 函数签名：`plot_psd_fcc_compliance(psd_2nd, psd_5th, fcc_mask, frequencies, save_path)`
  - 绘制 3 条曲线：2 阶 PSD（蓝色实线），5 阶 PSD（绿色虚线），FCC Mask（红色点线）
  - 标注关键频率点：3.1 GHz, 10.6 GHz（灰色竖线）
  - 轴标签：横轴"频率 (GHz)"，纵轴"功率谱密度 (dBm/MHz)"
  - 图例：位于 "best" 位置
  - 保存：dpi=300, bbox_inches="tight"

#### 3.3 端到端脚本

- [ ] T015 [US1] 创建主脚本 scripts/plot_psd_fcc_compliance.py
  - Step 1: 生成 2 阶脉冲（tau=0.5ns, fc=0）
  - Step 2: 生成 5 阶脉冲（tau=0.5ns, fc=6.85e9）
  - Step 3: 计算 PSD（sampling_rate=50e9, freq_resolution=10e6）
  - Step 4: 生成 FCC 掩蔽罩
  - Step 5: 合规性判断
  - Step 6: 可视化并保存到 outputs/psd_fcc_compliance_2nd_5th.png
  - 输出执行时间和合规性结果

#### 3.4 集成测试

- [ ] T016 [US1] 集成测试：端到端验证 in tests/integration/test_psd_fcc_integration.py
  - 运行完整流程（脉冲生成 → PSD 计算 → 合规性判断 → 可视化）
  - 验证输出文件存在：outputs/psd_fcc_compliance_2nd_5th.png
  - 验证图表分辨率 ≥ 300 DPI
  - 验证执行时间 < 10 秒（SC-005）

**Validation Checklist** (User Story 1):
- [ ] 图表包含 FCC 掩蔽罩和 2 种脉冲 PSD
- [ ] 2 阶脉冲在低频段（<3.1 GHz）超出 FCC 限制，标注为"不合规"
- [ ] 5 阶脉冲主要能量集中在 3.1-10.6 GHz，标注为"合规"
- [ ] 所有 4 个 FCC 频段限制值清晰标注
- [ ] 图表分辨率 ≥ 300 DPI
- [ ] 执行时间 < 10 秒

---

## Phase 4: User Story 2（P2 - 比较不同阶数脉冲的频谱特性）

**Priority**: P2（技术对比分析）

**Goal**: 在同一图表上用不同颜色/线型对比 2 阶和 5 阶脉冲的频谱分布。

**Independent Test**: 运行脚本后，图表中 2 阶脉冲（蓝色实线）和 5 阶脉冲（绿色虚线）清晰可辨，图例正确标注，5 阶脉冲峰值频率约为 2 阶脉冲的 2.5 倍。

**Dependencies**: User Story 1 完成

**Duration**: ~1 hour

### Tasks

#### 4.1 增强可视化（对比分析）

- [ ] T017 [P] [US2] 优化图表样式以突出对比 in src/visualization/compliance.py
  - 2 阶脉冲：蓝色实线（linewidth=1.5）+ 圆形标记（每 100 点）
  - 5 阶脉冲：绿色虚线（linestyle="--", linewidth=1.5）+ 方形标记（每 100 点）
  - 图例：添加峰值频率标注（如 "2 阶脉冲 (f_peak≈710 MHz)"）

- [ ] T018 [US2] 添加频谱对比分析功能 in src/models/psd.py
  - 函数签名：`analyze_spectrum_shift(psd_2nd, psd_5th, frequencies)`
  - 计算峰值频率：f_peak_2nd, f_peak_5th
  - 计算频率比：f_peak_5th / f_peak_2nd
  - 返回分析报告（文本格式）

#### 4.2 单元测试

- [ ] T019 [US2] 单元测试：频谱对比分析 in tests/unit/test_psd.py
  - 验证 5 阶脉冲峰值频率 > 2 阶脉冲峰值频率
  - 验证频率比约为 2.5（理论值：sqrt(5)/sqrt(2) ≈ 1.58，但调制后约 2.5）

**Validation Checklist** (User Story 2):
- [ ] 图表用不同颜色和线型区分 2 种脉冲
- [ ] 图例清晰标注"2 阶高斯导数脉冲"和"5 阶高斯导数脉冲"
- [ ] 5 阶脉冲能量分布向高频偏移（峰值频率约为 2 阶的 2.5 倍）

---

## Phase 5: User Story 3（P3 - 优化脉冲参数以满足 FCC 规范）

**Priority**: P3（参数优化功能）

**Goal**: 支持调整脉冲参数（fc, τ），实时重新生成图表，观察 PSD 曲线变化。

**Independent Test**: 修改脚本中的 fc 参数（如从 6.85 GHz 改为 8 GHz），重新运行，图表显示 PSD 曲线整体向高频偏移。

**Dependencies**: User Story 1, 2 完成

**Duration**: ~2 hours

### Tasks

#### 5.1 参数调优接口

- [ ] T020 [P] [US3] 添加参数扫描功能 in scripts/optimize_pulse_parameters.py
  - 扫描中心频率：fc_values = np.linspace(5e9, 8e9, 7)
  - 对每个 fc：生成脉冲 → 计算 PSD → 判断合规性
  - 输出最优参数（使 5 阶脉冲合规且最大化 UWB 频段能量占比）

- [ ] T021 [US3] 添加参数标注功能 in src/visualization/compliance.py
  - 在图表标题或图例中标注当前参数：fc, τ
  - 格式："5 阶脉冲（fc=6.85 GHz, τ=0.5 ns）"

#### 5.2 参数敏感性分析

- [ ] T022 [P] [US3] 实现参数敏感性分析 in src/models/psd.py
  - 函数签名：`analyze_parameter_sensitivity(n, tau_range, fc_range, sampling_rate)`
  - 输出热力图：横轴 fc，纵轴 τ，颜色表示合规性状态
  - 保存到 outputs/parameter_sensitivity_heatmap.png

#### 5.3 集成测试

- [ ] T023 [US3] 集成测试：参数优化验证 in tests/integration/test_parameter_optimization.py
  - 运行参数扫描脚本
  - 验证最优参数在合理范围（fc ∈ [6, 7.5] GHz）
  - 验证热力图文件生成

**Validation Checklist** (User Story 3):
- [ ] 参数扫描脚本正常运行
- [ ] 图表中包含参数值标注
- [ ] 热力图清晰展示参数对合规性的影响

---

## Phase 6: Polish & Cross-Cutting Concerns（完善与横切关注点）

**Goal**: 代码质量、文档完善、性能优化

**Duration**: ~1 hour

### Tasks

#### 6.1 代码质量

- [ ] T024 运行代码格式化和 linter
  - black src/ tests/ scripts/ --line-length 100
  - pylint src/ tests/ scripts/
  - 修复所有 linter 警告

#### 6.2 文档完善

- [ ] T025 更新 README.md
  - 添加 PSD/FCC 合规性可视化功能说明
  - 添加快速开始示例（引用 quickstart.md）
  - 添加输出图表示例

**Validation Checklist** (Polish):
- [ ] 所有代码通过 black 和 pylint 检查
- [ ] README.md 包含新功能说明

---

## Dependency Graph（依赖关系图）

### User Story Completion Order

```
Phase 1 (Setup)
    ↓
Phase 2 (Foundational) ← BLOCKING for all user stories
    ↓
    ├─→ Phase 3 (User Story 1 - P1) ✓ MVP
    │       ↓
    ├─→ Phase 4 (User Story 2 - P2) depends on US1
    │       ↓
    └─→ Phase 5 (User Story 3 - P3) depends on US1, US2
            ↓
        Phase 6 (Polish)
```

### Task Dependencies

**Blocking Chains**:
1. T001-T004 (Setup) → T005-T011 (Foundational) → **ALL user story tasks**
2. T012-T016 (US1) → T017-T019 (US2)
3. T012-T016 (US1) + T017-T019 (US2) → T020-T023 (US3)

**No Dependencies** (can run in parallel after prerequisites):
- Within Foundational: T005 [P] T006, T008 [P] T010
- Within US1: T012 [P] T014
- Within US2: T017 [P] (after US1)
- Within US3: T020 [P] T022 (after US1, US2)

---

## Parallel Execution Examples

### After Setup Phase (T001-T004 complete)

**Parallel Group 1** (Foundational):
```bash
# Terminal 1
实现 T005: 2 阶脉冲生成

# Terminal 2
实现 T006: 5 阶脉冲生成

# Terminal 3
实现 T010: FCC 掩蔽罩生成
```

### After Foundational Phase (T005-T011 complete)

**Parallel Group 2** (User Story 1):
```bash
# Terminal 1
实现 T012: 合规性判断函数

# Terminal 2
实现 T014: 可视化函数
```

### After User Story 1 Complete

**Parallel Group 3** (User Story 2):
```bash
# Terminal 1
实现 T017: 优化图表样式

# Terminal 2
实现 T018: 频谱对比分析
```

---

## Validation Criteria

### Per-Phase Validation

| Phase | Validation Command | Expected Result |
|-------|-------------------|-----------------|
| Setup | `python -c "import matplotlib.font_manager as fm; print('Times New Roman' in [f.name for f in fm.fontManager.ttflist])"` | `True` |
| Foundational | `pytest tests/unit/ -v` | All tests pass |
| US1 | `python scripts/plot_psd_fcc_compliance.py` | File `outputs/psd_fcc_compliance_2nd_5th.png` created, time < 10s |
| US2 | Visual inspection of plot | 2 curves clearly distinguishable |
| US3 | `python scripts/optimize_pulse_parameters.py` | Optimal fc ∈ [6, 7.5] GHz |
| Polish | `black --check src/ && pylint src/` | Exit code 0 |

### Overall Success Criteria (from spec.md)

- [ ] SC-001: FCC 掩蔽罩数值精度 < 0.1 dB
- [ ] SC-002: 频率分辨率 ≥ 10 MHz（1200+ 采样点）
- [ ] SC-003: 合规性判断准确率 100%
- [ ] SC-004: 图表分辨率 ≥ 300 DPI，符合 IEEE 标准
- [ ] SC-005: 执行时间 < 10 秒
- [ ] SC-006: 图表自动保存到 outputs/，命名规范清晰
- [ ] SC-007: 5 阶脉冲展示能量集中在 3.1-10.6 GHz
- [ ] SC-008: 2 阶脉冲展示低频段违规

---

## Execution Notes

### MVP First Approach

建议首先完成 **Phase 1 + Phase 2 + Phase 3**（Setup + Foundational + User Story 1），这样可以：
- 交付核心价值（FCC 合规性验证）
- 验证技术可行性
- 获得早期反馈
- 预计总时间：~6 hours

### Incremental Delivery

- **第一次交付**（MVP）：Phase 1-3 完成后
  - 功能：生成 PSD vs FCC Mask 图表，合规性判断
  - 验收：运行脚本，查看输出图表

- **第二次交付**：Phase 4 完成后
  - 功能：增强对比分析可视化
  - 验收：图表样式改进，图例更清晰

- **最终交付**：Phase 5-6 完成后
  - 功能：参数优化，代码质量完善
  - 验收：所有成功标准满足

### Troubleshooting

**常见问题**：
1. **字体缺失**（T002 失败）：
   - Linux: `sudo apt-get install ttf-mscorefonts-installer`
   - 验证：`fc-list | grep -i times`

2. **FFT 精度问题**（T009 失败）：
   - 使用 `np.float64` 避免精度损失
   - 验证 Parseval 定理（误差 < 1%）

3. **图表分辨率不足**（SC-004 失败）：
   - 检查 `matplotlib.rcParams["savefig.dpi"]` 是否 ≥ 300
   - 使用 `file outputs/*.png` 验证实际分辨率

---

**Task Breakdown Status**: ✅ 完成
**Total Tasks**: 25
**Parallelizable Tasks**: 12
**Estimated Total Time**: ~10 hours（MVP: ~6 hours）
**Next Step**: 执行 `/speckit.implement` 开始实施

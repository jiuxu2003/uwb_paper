# Tasks: TH-UWB Communication System Simulation

**Input**: Design documents from `/specs/001-th-uwb-simulation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Tasks include unit tests and integration tests as specified in the contracts. Tests should be written alongside implementation to validate functionality.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Project structure**: Single project (`src/`, `tests/` at repository root)
- All file paths are relative to repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per plan.md (src/, tests/, scripts/, outputs/ directories)
- [ ] T002 Initialize Python project with requirements.txt (numpy>=1.24.0, scipy>=1.10.0, matplotlib>=3.7.0, pytest>=7.0.0)
- [ ] T003 [P] Create pyproject.toml with project metadata and tool configuration
- [ ] T004 [P] Configure black formatter (line-length=100, target-version=py311)
- [ ] T005 [P] Configure pylint linter in .pylintrc or pyproject.toml
- [ ] T006 [P] Update .gitignore to exclude outputs/ directory and Python cache files
- [ ] T007 Create __init__.py files for all Python packages (src/, src/models/, src/simulation/, src/visualization/, tests/)

**Checkpoint**: Project structure ready, dependencies can be installed with `pip install -r requirements.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T008 Implement SystemConfig dataclass in src/config.py with all simulation parameters (Tf, Tc, δ, τ, fs, num_bits, random_seed) and validation constraints
- [ ] T009 Implement generate_gaussian_doublet() function in src/models/pulse.py per contract (pulse.md)
- [ ] T010 Implement Pulse.generate() class method in src/models/pulse.py per contract
- [ ] T011 Implement Pulse.energy property in src/models/pulse.py per contract
- [ ] T012 [P] Write unit test test_gaussian_doublet_zero_mean() in tests/unit/test_pulse.py
- [ ] T013 [P] Write unit test test_gaussian_doublet_peak_at_zero() in tests/unit/test_pulse.py
- [ ] T014 [P] Write unit test test_pulse_generate() in tests/unit/test_pulse.py
- [ ] T015 [P] Write unit test test_pulse_energy_positive() in tests/unit/test_pulse.py

**Checkpoint**: Foundation ready - SystemConfig and Pulse are functional and tested. User story implementation can now begin in parallel.

---

## Phase 3: User Story 1 - 基础 TH-UWB 信号生成与可视化 (Priority: P1) 🎯 MVP

**Goal**: 生成标准的 TH-UWB 信号并可视化时域波形，展示 TH-PPM 工作原理

**Independent Test**: 运行 scripts/demo_waveform.py 应生成清晰的时域波形图（≥3 帧），保存到 outputs/waveform_demo.png，图表包含时间轴（ns）、幅度轴、网格、图例、标题

### Implementation for User Story 1

- [ ] T016 [P] [US1] Implement generate_th_code() function in src/models/modulation.py per contract (modulation.md)
- [ ] T017 [P] [US1] Implement TimeHoppingCode dataclass in src/models/modulation.py with generate() class method
- [ ] T018 [US1] Implement User dataclass in src/models/modulation.py with create() factory method (depends on T016, T017)
- [ ] T019 [US1] Implement User.generate_signal() method in src/models/modulation.py per contract
- [ ] T020 [P] [US1] Implement plot_waveform() function in src/visualization/waveform.py per research.md Section 6.2 (Matplotlib配置: dpi=300, 网格, 图例, 中文标签)
- [ ] T021 [US1] Create demo_waveform.py script in scripts/ per quickstart.md Step 1 (config with 3 frames, generate pulse, create user, generate signal, plot and save)
- [ ] T022 [P] [US1] Write unit test test_generate_th_code_length() in tests/unit/test_modulation.py
- [ ] T023 [P] [US1] Write unit test test_generate_th_code_uniformity() in tests/unit/test_modulation.py
- [ ] T024 [P] [US1] Write unit test test_user_generate_signal_length() in tests/unit/test_modulation.py
- [ ] T025 [P] [US1] Write unit test test_user_generate_signal_reproducibility() in tests/unit/test_modulation.py

**Checkpoint**: At this point, User Story 1 should be fully functional. Running `python scripts/demo_waveform.py` should generate a publication-quality waveform plot in outputs/ directory. All unit tests for pulse and modulation modules should pass.

---

## Phase 4: User Story 2 - 多用户干扰性能分析 (Priority: P2)

**Goal**: 模拟多用户同时通信场景，分析用户数量对误码率的影响，生成 BER vs 用户数量性能曲线

**Independent Test**: 运行 scripts/demo_mui_analysis.py（快速版，1000比特）应生成 BER vs 用户数量曲线图，保存到 outputs/ber_vs_users_demo.png，曲线显示随用户增加误码率上升趋势（SNR=10dB，用户数1/2/3/5）

### Implementation for User Story 2

- [ ] T026 [P] [US2] Implement Channel dataclass in src/models/channel.py with __init__ accepting config, snr_db, sir_db, nbi_frequency
- [ ] T027 [P] [US2] Implement Channel.add_awgn() method in src/models/channel.py per contract (channel.md)
- [ ] T028 [P] [US2] Implement Channel.add_nbi() method in src/models/channel.py per contract (预留窄带干扰接口，US3使用)
- [ ] T029 [US2] Implement Channel.transmit() method in src/models/channel.py per contract (handles multi-user signal aggregation, depends on T026, T027, T028)
- [ ] T030 [P] [US2] Implement Receiver dataclass in src/simulation/receiver.py with __init__ accepting config, target_user, pulse
- [ ] T031 [P] [US2] Implement Receiver.generate_templates() method in src/simulation/receiver.py per research.md Section 4.1
- [ ] T032 [US2] Implement Receiver.demodulate() method in src/simulation/receiver.py per contract (depends on T031)
- [ ] T033 [P] [US2] Implement PerformanceMetrics dataclass in src/simulation/metrics.py with ber, num_errors, num_bits properties
- [ ] T034 [P] [US2] Implement PerformanceMetrics.ber_confidence_interval() method in src/simulation/metrics.py (Wilson confidence interval)
- [ ] T035 [P] [US2] Implement SimulationResult dataclass in src/simulation/metrics.py with to_dict() method
- [ ] T036 [P] [US2] Implement plot_ber_vs_users() function in src/visualization/performance.py per research.md Section 6.2 (semilogy, 网格, 图例, 轴标签)
- [ ] T037 [US2] Create demo_mui_analysis.py script in scripts/ per quickstart.md Step 2 (1000 bits for fast demo, user_counts=[1,2,3,5], SNR=10dB)
- [ ] T038 [US2] Create run_mui_analysis.py script in scripts/ for full simulation (10000 bits, user_counts=[1,2,3,5,7,10], SNR=10dB)
- [ ] T039 [P] [US2] Write unit test test_add_awgn_snr() in tests/unit/test_channel.py
- [ ] T040 [P] [US2] Write unit test test_channel_transmit_multi_user() in tests/unit/test_channel.py
- [ ] T041 [P] [US2] Write unit test test_receiver_demodulate() in tests/unit/test_receiver.py (requires mock signal)
- [ ] T042 [P] [US2] Write unit test test_metrics_ber_calculation() in tests/unit/test_metrics.py
- [ ] T043 [US2] Write integration test test_end_to_end_single_user() in tests/integration/test_end_to_end.py (full pipeline: config → pulse → user → channel → receiver → metrics)
- [ ] T044 [US2] Write integration test test_end_to_end_multi_user() in tests/integration/test_end_to_end.py (validates MUI effect: BER increases with user count)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Running `python scripts/demo_mui_analysis.py` should complete in ~1 minute and generate a BER vs users plot. Running `python scripts/run_mui_analysis.py` generates the full publication-quality figure (10000 bits, may take 5-10 minutes). All unit tests and integration tests should pass.

---

## Phase 5: User Story 3 - 窄带干扰抑制性能分析 (Priority: P3)

**Goal**: 在信道中加入强窄带干扰（2.4 GHz 单频正弦波），分析信干比（SIR）对误码率的影响，验证 UWB 抗干扰能力

**Independent Test**: 运行 scripts/demo_nbi_analysis.py（快速版，1000比特）应生成 BER vs SIR 曲线图，保存到 outputs/ber_vs_sir_demo.png，曲线显示随 SIR 降低误码率上升，但即使 SIR=-10dB 时 BER<0.5（单用户场景）

### Implementation for User Story 3

- [ ] T045 [P] [US3] Implement plot_ber_vs_sir() function in src/visualization/performance.py per research.md Section 6.2 (semilogy, 红色方块标记, 网格, 图例)
- [ ] T046 [US3] Create demo_nbi_analysis.py script in scripts/ per quickstart.md Step 3 (1000 bits, single user, SIR=[30,20,10,0,-10]dB, freq=2.4GHz)
- [ ] T047 [US3] Create run_nbi_analysis.py script in scripts/ for full simulation (10000 bits, single user, SIR from 30dB to -10dB with 10 points)
- [ ] T048 [P] [US3] Write unit test test_add_nbi_sir() in tests/unit/test_channel.py (validates NBI power matches SIR definition)
- [ ] T049 [P] [US3] Write unit test test_add_nbi_frequency() in tests/unit/test_channel.py (FFT validates interference at 2.4 GHz)
- [ ] T050 [US3] Write integration test test_end_to_end_with_nbi() in tests/integration/test_end_to_end.py (validates UWB robustness: BER<0.5 even at SIR=-10dB)

**Checkpoint**: All three user stories are now independently functional. Running NBI analysis scripts validates UWB's anti-interference capability. All tests pass.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and finalize the project for publication

- [ ] T051 Create generate_figures.py script in scripts/ to batch-generate all three publication figures (waveform, BER-users, BER-SIR with 10000 bits)
- [ ] T052 [P] Create README.md at repository root with project overview, installation instructions, quick start guide reference, and citation of quickstart.md
- [ ] T053 [P] Add docstrings to all public functions and classes (中文注释，说明参数、返回值、物理意义)
- [ ] T054 [P] Run black formatter on all Python files in src/ and tests/
- [ ] T055 [P] Run pylint on all Python files and fix warnings (target score ≥8.0/10)
- [ ] T056 Validate performance targets per SC-006: run full simulations and measure execution time (single point <1 min, full curve <15 min)
- [ ] T057 Validate all 8 Success Criteria from spec.md (SC-001 to SC-008): check figure quality, BER trends, statistical stability, reproducibility
- [ ] T058 [P] Create .github/workflows/ci.yml for CI/CD (optional, if requested): run pytest on push, validate code formatting
- [ ] T059 Update quickstart.md examples if any script interfaces changed during implementation
- [ ] T060 Final validation: run all three demo scripts consecutively and verify outputs/ contains three high-quality figures

**Checkpoint**: Project is publication-ready. All figures meet academic standards (≥300 DPI, 清晰的中文标签). All tests pass. Code is clean and documented.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all three user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
  - Delivers: Signal generation + waveform visualization
  - Independently testable: Run demo_waveform.py

- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 components (User, Pulse)
  - Builds on: User.generate_signal() from US1
  - Adds: Channel, Receiver, PerformanceMetrics, BER analysis
  - Independently testable: Run demo_mui_analysis.py with multiple users

- **User Story 3 (P3)**: Can start after US2 completion - Reuses Channel.add_nbi() from US2
  - Builds on: All US2 components (Channel, Receiver, Metrics)
  - Adds: NBI-specific analysis script and visualization
  - Independently testable: Run demo_nbi_analysis.py with varying SIR

**Note**: While US2 builds on US1 and US3 builds on US2, each story delivers independent value and can be tested/demoed separately.

### Within Each User Story

- Tests can run in parallel with implementation (not strict TDD)
- Models before services/algorithms (TimeHoppingCode + User before signal generation)
- Core algorithms before visualization (Channel + Receiver before performance plots)
- Demo scripts after all components are ready
- Integration tests after unit tests pass

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005, T006 can run in parallel (different config files)

**Phase 2 (Foundational)**:
- T012, T013, T014, T015 (all pulse unit tests) can run in parallel

**Phase 3 (US1)**:
- T016, T017 (th_code generation) can run in parallel
- T020 (visualization) can develop in parallel with T016-T019 (modulation)
- T022, T023, T024, T025 (all modulation unit tests) can run in parallel

**Phase 4 (US2)**:
- T026, T027, T028 (Channel methods) can run in parallel
- T030, T031 (Receiver setup) can run in parallel with Channel work
- T033, T034, T035 (Metrics module) can run in parallel with Channel/Receiver
- T036 (visualization) can develop in parallel with simulation modules
- T039, T040, T041, T042 (unit tests) can run in parallel

**Phase 5 (US3)**:
- T048, T049 (NBI unit tests) can run in parallel
- T046, T047 (scripts) can develop in parallel if NBI is already implemented in US2

**Phase 6 (Polish)**:
- T052, T053, T054, T055, T058 can run in parallel (different aspects)

---

## Parallel Example: User Story 2

```bash
# After Foundational phase completes, launch Channel + Receiver + Metrics in parallel:

# Developer A:
Task: "Implement Channel dataclass in src/models/channel.py"
Task: "Implement Channel.add_awgn() method"
Task: "Implement Channel.add_nbi() method"
Task: "Implement Channel.transmit() method"

# Developer B (parallel):
Task: "Implement Receiver dataclass in src/simulation/receiver.py"
Task: "Implement Receiver.generate_templates() method"
Task: "Implement Receiver.demodulate() method"

# Developer C (parallel):
Task: "Implement PerformanceMetrics dataclass in src/simulation/metrics.py"
Task: "Implement SimulationResult dataclass"
Task: "Implement plot_ber_vs_users() in src/visualization/performance.py"

# After all complete, integrate:
Task: "Create demo_mui_analysis.py script (uses Channel + Receiver + Metrics)"
Task: "Write integration test test_end_to_end_multi_user()"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T007) - ~30 min
2. Complete Phase 2: Foundational (T008-T015) - ~2 hours
3. Complete Phase 3: User Story 1 (T016-T025) - ~3 hours
4. **STOP and VALIDATE**: Run `pytest tests/unit/` (all pulse + modulation tests pass)
5. **DEMO**: Run `python scripts/demo_waveform.py` → outputs/waveform_demo.png generated
6. **CHECKPOINT**: MVP ready - can generate and visualize TH-UWB signals

**Total MVP time estimate**: ~6 hours

### Incremental Delivery

1. **Sprint 1**: Setup + Foundational → Foundation ready (~2.5 hours)
2. **Sprint 2**: User Story 1 → Test independently → Demo waveform (MVP!) (~3 hours)
3. **Sprint 3**: User Story 2 → Test independently → Demo multi-user analysis (~5 hours)
4. **Sprint 4**: User Story 3 → Test independently → Demo NBI analysis (~3 hours)
5. **Sprint 5**: Polish → Generate all publication figures, finalize documentation (~2 hours)

**Total project time estimate**: ~15-16 hours

### Parallel Team Strategy

With 3 developers after Foundational phase:

1. Team completes Setup + Foundational together (~2.5 hours)
2. Once Foundational is done (after T015):
   - **Developer A**: User Story 1 (T016-T025) → 3 hours
   - **Developer B**: User Story 2 (T026-T044) → 5 hours (waits for US1 User class)
   - **Developer C**: User Story 3 (T045-T050) → 3 hours (waits for US2 Channel/Receiver)
3. Team reconvenes for Polish phase (T051-T060) → 2 hours

**Parallel completion time**: ~10 hours (with 3 developers, accounting for some waiting)

---

## Performance Validation Checkpoints

Per Success Criteria SC-006, validate performance at these milestones:

- **After T021 (US1 demo)**: Measure script execution time with 3 frames (should be instant, <1 sec)
- **After T037 (US2 demo)**: Measure demo_mui_analysis.py with 1000 bits (target: <2 min per user count)
- **After T038 (US2 full)**: Measure run_mui_analysis.py with 10000 bits, 6 user points (target: <10 min total)
- **After T046 (US3 demo)**: Measure demo_nbi_analysis.py with 1000 bits, 5 SIR points (target: <2 min total)
- **After T047 (US3 full)**: Measure run_nbi_analysis.py with 10000 bits, 10 SIR points (target: <5 min total)
- **T056 (Final validation)**: Confirm all performance targets met (single point <1 min, full curve <15 min)

If performance targets not met, apply optimizations from research.md Section 8 (vectorization, batching).

---

## Notes

- [P] tasks = different files, no dependencies - can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Tests validate functionality alongside implementation (not strict TDD, but comprehensive coverage)
- Commit after each task or logical group (e.g., after completing a module + its tests)
- Stop at any checkpoint to validate story independently
- Priority order: P1 (US1) delivers MVP, P2 (US2) adds multi-user analysis, P3 (US3) adds NBI analysis
- All file paths use forward slashes, compatible with Linux/macOS/Windows
- 中文注释和文档字符串确保代码可读性和可维护性（Constitution: 透明记录原则）

---

**任务总计**: 60 个任务
**预估总时长**: 15-16 小时（单人顺序执行）/ ~10 小时（3人并行）
**MVP 范围**: Phase 1-3（T001-T025，约 6 小时）
**并行任务数**: 32 个任务标记 [P]，可并行执行

**就绪状态**: ✅ All tasks defined, dependencies mapped, ready for implementation

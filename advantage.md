# 项目优势、同类产品缺陷与后续改进方向

这份文档用于总结当前 ML 项目的产品定位、与同类工具相比的优势、当前仍存在的短板，以及后续最值得继续完善的方向。它更适合用于项目答辩、比赛展示、README 扩展说明和后续开发规划。

## 1. 当前项目定位

当前项目不应该被包装成“万能蛋白质预测软件”，也不应该被简单理解成“3D 可视化软件”。

更准确的定位是：

> 一个面向 VHH / Nanobody 候选筛选的本地交互式分析工具，重点解决“候选是否可能阻断目标蛋白 pocket / 功能区域”的批量评估、可解释排序、质量控制和结果导出问题。

当前核心能力包括：

- 从 `input_pose_table.csv` 或 `pose_features.csv` 启动完整流程。
- 对 docking pose / conformer / nanobody 做分层评分与排序。
- 同时保留 Rule 路线和 ML 路线，不完全依赖黑盒模型。
- 输出 Rule + ML + QC 风险融合后的共识排名。
- 生成候选报告卡、下一轮实验建议、参数敏感性分析和运行摘要。
- 生成候选横向对比解释，说明为什么 A 排在 B 前面，并提示 close decision。
- 自动检查 top 候选是否已有真实验证证据，避免把“模型排序”误讲成“实验已验证”。
- 支持本地交互式页面、历史运行、多运行对比、导出 HTML / PDF / zip。
- 支持一键 synthetic demo 与可运行 toy PDB mini 示例，能分别验证“报告链路”和“真实 PDB 输入链路”。
- 针对 CD38 pocket benchmark 已有 ligand-contact、P2Rank、方法共识、参数敏感性、fpocket 接入入口、公开结构 starter 一键刷新入口和外部工具 next-run runbook。
- 已补 [ML.md](ML.md)，可以直接向评审解释 Rule、MLP、pose/conformer/nanobody 聚合和共识排序的架构。

## 2. 同类型产品常见缺陷

这里的“同类型产品”不是单一软件，而是几类常见工具的组合：分子可视化工具、pocket finder、docking scoring 工具、商业建模平台、AutoML / notebook 工作流和实验数据管理工具。

### 2.1 通用分子可视化工具的缺陷

典型问题：

- 更擅长“看结构”，不擅长批量筛选候选。
- 用户需要手动加载结构、手动选残基、手动截图，结果难以批量复现。
- 对纳米抗体是否阻断 pocket 的判断通常依赖人工经验。
- 不会自动生成候选排序、低可信原因、下一轮实验建议。
- 很多结论停留在视觉层面，缺少结构化 CSV / JSON / report 输出。

对本项目的启发：

- 我们不需要优先做复杂 3D viewer。
- 更有价值的是把结构分析结果变成可批量比较、可导出、可解释的表格和报告。

### 2.2 Pocket finder 工具的缺陷

典型问题：

- 能找 pocket，但通常不直接回答“哪个纳米抗体更可能阻断这个 pocket”。
- 不同工具或参数输出的 pocket 边界可能差异很大。
- coverage 高不代表 pocket 定义足够紧，过宽 pocket 会导致误判。
- 很多工具只输出候选 pocket，不会自动与已知功能残基、催化残基或实验关注位点做闭环验证。
- 对初学者来说，输出文件格式多、整合成本高。

对本项目的启发：

- 我们把 pocket 结果纳入下游 Rule / ML / QC / benchmark，而不是只停留在 pocket 预测本身。
- 当前已经补了 pocket 方法共识、CD38 benchmark、参数敏感性和 `pocket_shape_overwide_proxy`，能把“pocket 是否偏宽”显式暴露出来。

### 2.3 Docking scoring / ranking 工具的缺陷

典型问题：

- 常见 docking score 更偏结合强度或构象打分，不一定等价于功能阻断能力。
- 单个 score 难解释，容易出现“分数高但没有挡住关键区域”的情况。
- 多 pose、多 conformer、多 nanobody 的层级聚合通常需要用户自己写脚本。
- 很多流程没有把 pocket 命中、口部遮挡、底物路径阻断、催化位点覆盖这些信号整合到一个可解释排序中。

对本项目的启发：

- 我们的核心差异不是单纯替代 docking score，而是把 docking pose 转成“是否阻断功能口袋”的多维证据。
- 当前的 pose -> conformer -> nanobody 分层聚合更贴近真实候选筛选决策。

### 2.4 商业建模平台的缺陷

典型问题：

- 功能强，但成本高、部署重、学习门槛高。
- 很多平台更适合专家交互，不一定适合比赛展示或快速交付一个小型专项工具。
- 自动化批处理和自定义解释报告通常需要额外脚本或商业模块。
- 对特定课题如“纳米抗体阻断 CD38 pocket”的专项逻辑不一定开箱即用。

对本项目的启发：

- 本项目的优势不在“覆盖所有建模功能”，而在“围绕一个明确科研问题做轻量、可解释、可本地运行的闭环工具”。

### 2.5 Notebook / 脚本型工作流的缺陷

典型问题：

- 灵活，但对非开发者不友好。
- 文件路径、输入格式、依赖环境和运行顺序容易出错。
- 结果散落在多个 CSV / JSON / notebook cell 中，不利于复现和展示。
- 缺少本地软件级入口、历史运行、导出、失败诊断和参数模板。

对本项目的启发：

- 当前本地交互壳、桌面启动器、便携版、standalone 版、运行历史和报告导出，是把研究脚本变成可用软件的重要优势。

## 3. 我们当前的核心优势

### 3.1 任务定位更垂直

很多工具是通用结构分析或通用 docking 分析，本项目聚焦在一个更具体的问题：

> 纳米抗体 / VHH 候选是否可能阻断目标蛋白 pocket 或功能区域。

这个定位带来的优势是：

- 特征设计更贴近 pocket blocking，而不是只看结合分数。
- 输出排序更接近实验候选选择，而不是只输出 pose 分数。
- 报告解释可以围绕“为什么可能阻断”展开。
- 很适合比赛展示，因为问题边界清晰、结果容易讲明白。

### 3.2 Rule + ML 双路线更稳

当前项目不是只给一个黑盒 ML 分数，而是同时保留：

- Rule ranking：可解释、可 sanity check。
- ML ranking：能学习多特征组合。
- Consensus ranking：融合 Rule、ML、QC 风险和来源覆盖度。

优势：

- 当数据量不足时，Rule 路线仍能工作。
- 当 ML 结果和 Rule 结果冲突时，可以显式标记低可信。
- 更适合科研早期阶段，因为不会把伪标签模型包装成绝对真值。
- 方便调试：如果 ML 排名异常，可以和 Rule 排名做对照。

### 3.3 从 pose 到 nanobody 的层级聚合更符合实际决策

候选筛选不是只看一个 pose，而是要考虑：

- 一个 nanobody 有多个 conformer。
- 一个 conformer 有多个 docking pose。
- top pose、平均 top-k、构象一致性和波动都影响最终判断。

当前项目已经做了：

- pose-level 特征构建。
- pose-level ML / Rule score。
- conformer-level 聚合。
- nanobody-level 最终排序。
- pocket consistency 和 std penalty。

这比“只输出每个 pose 一个 score”的流程更接近真实实验候选选择。

### 3.4 几何特征更贴近“阻断”问题

当前项目已经不仅仅计算距离，还引入了多类阻断相关 proxy：

- pocket hit fraction
- catalytic hit fraction
- mouth occlusion score
- mouth axis / aperture block fraction
- substrate overlap score
- ligand path block score
- ligand path bottleneck score
- ligand path exit block fraction
- pocket shape overwide proxy

### 3.5 从演示到真实输入的迁移成本更低

很多同类 notebook 或脚本项目只给一份空模板，用户不知道 PDB、chain、pocket residue、ligand template 应该如何组织。当前项目已经把这一步拆成两层：

- synthetic demo：用于确认安装、模型流程、报告和导出是否完整。
- `REAL_DATA_STARTER/MINI_PDB_EXAMPLE`：用于确认真实 `input_csv -> PDB 解析 -> residue mapping -> feature table -> pipeline` 路径是否完整。

这个设计的优势是：用户可以先用 toy PDB 检查输入格式和运行环境，再替换成自己的真实 PDB，减少“第一次接入真实数据就被路径和格式卡住”的概率。

优势：

- 能区分“靠近 pocket”和“真正可能挡住 pocket”。
- 能提示 pocket 定义是否过宽。
- 能把结构判断转换成可训练、可排序、可解释的特征。

需要注意：

- 这些仍然是 proxy，不是严格物理模拟。
- 当前优势在“可解释筛选”和“风险提示”，不应宣传成高精度动力学预测。

### 3.5 质量控制和失败诊断比较完整

同类脚本常见问题是失败了也不知道哪里坏了。本项目已经具备：

- `feature_qc.json`
- 失败行 / warning 行统计
- 运行前输入检查
- 缺依赖预检
- 本地软件错误摘要
- pocket shape QC
- 几何 proxy 一致性审计
- 真实验证证据审计
- 参数敏感性分析
- smoke test
- grouped CV benchmark

优势：

- 更容易发现输入问题、路径问题、数据缺失问题。
- 更适合交给别人使用，而不是只能作者自己跑。
- 更适合展示工程完整度。

### 3.6 结果可解释、可导出、可展示

当前项目输出不只是 CSV，还包括：

- `recommended_pipeline_report.md`
- `consensus_ranking.csv`
- `candidate_report_cards/index.html`
- `next_experiment_suggestions_report.md`
- `parameter_sensitivity_report.md`
- 多运行对比 HTML / PDF
- 当前运行汇总 zip

优势：

- 适合答辩、比赛和团队沟通。
- 可以直接把结果交给实验人员看。
- 降低“模型跑完但没人看得懂”的风险。

### 3.7 本地运行和数据传输更方便

当前项目支持：

- Streamlit 本地交互界面。
- Windows 一键启动。
- 桌面 launcher。
- 便携目录版。
- standalone 单文件自解压版。
- zip 数据包导入。
- 本地目录扫描。
- 历史运行和模板保存。

优势：

- 不依赖远程服务器，适合本地文件和结构数据处理。
- 对初学者比命令行脚本更友好。
- 便于在比赛现场或本机演示。

## 4. 我们当前仍然存在的短板

### 4.1 真实标签和外部验证仍不足

当前 ML 的主要问题不是代码能不能跑，而是真实监督信号仍然不足：

当前已经有实验 ledger、真实验证回灌报告、验证证据审计和再训练前后对照报告，但真实标签本身仍需要实验数据补充。也就是说，工具现在能明确告诉用户“哪些候选还没被验证”，但不能替代真实实验。

- 伪标签不等于实验真值。
- grouped CV 仍然不是外部独立 test set。
- 当前 CD38 benchmark 主要验证 pocket 覆盖，不等于完整纳米抗体阻断实验验证。

改进方向：

- 引入真实实验标签，例如是否阻断功能、IC50 变化、酶活抑制、结合/不结合结果。
- 建立独立 test set，不参与任何权重校准。
- 记录失败案例，专门分析模型误判原因。

### 4.2 几何 proxy 仍然是近似

当前 mouth / path / occupancy / overwide 等特征已经比简单距离强，但仍不是：

- 分子动力学模拟。
- 显式溶剂模型。
- 严格能量计算。
- 真实底物进入路径模拟。

改进方向：

- 用更多结构 benchmark 校准 mouth/path 权重。
- 把 ligand-bound 与 apo 结构分开评估。
- 对关键候选引入更高精度的二级验证流程。

### 4.3 pocket 定义对结果仍有较大影响

当前已经支持手动 pocket、P2Rank、fpocket、ligand-contact 和方法共识，但本质上：

- pocket 边界不同会影响特征。
- pocket 过宽可能让模型误判“覆盖很好”。
- pocket 过窄可能漏掉真实功能区域。

改进方向：

- 默认输出 pocket 定义风险等级。
- 对每个候选显示“结果对 pocket 定义是否敏感”。
- 积累更多 CD38 / 非 CD38 benchmark 来决定 overwide penalty 是否默认启用。

### 4.4 当前软件仍偏研究原型

虽然已有本地软件壳，但还不是成熟商业软件：

- 依赖环境仍可能在不同机器上出问题。
- UI 虽然功能多，但可能需要继续简化入口。
- 大规模数据运行时的性能和内存压力还需要验证。
- 缺少正式安装器、自动更新和用户权限管理。

改进方向：

- 做真实跨机器测试。
- demo 数据集和一键演示模式已补基础版：`run_demo_pipeline.py` / `run_demo_pipeline.bat` 可生成 synthetic 特征、synthetic validation override，并跑完整推荐 pipeline；本地软件侧边栏也能生成、载入或立即运行 demo，并在输出目录写入 `DEMO_OVERVIEW.html`、`DEMO_README.md`、`DEMO_INTERPRETATION.md` 与 `REAL_DATA_STARTER/`；摘要页可一键打开 HTML 导览和 starter 文件夹。
- 进一步收敛主界面的默认流程，让用户少做选择。
- 增加更清晰的启动失败诊断。

### 4.5 与外部工具的集成还可以更顺

当前已经有 P2Rank / fpocket / ligand-contact 的接入入口，但仍需要用户先在外部跑工具。

改进方向：

- 外部工具输出目录自动识别向导已补 fpocket 基础版：`prepare_cd38_fpocket_panel.py` 会生成 manifest、summary JSON 和 readiness report。
- 外部工具输出目录自动识别向导已补 P2Rank 基础版：`prepare_cd38_p2rank_panel.py` 会发现 `*_predictions.csv`、校验必需列、生成 manifest 和 readiness report。
- CD38 扩展计划已补基础版：`build_cd38_benchmark_expansion_plan.py` 会按结构/方法列出 `complete`、`needs_fpocket_output`、`needs_ligand_metadata`、`needs_p2rank_output` 等缺口。
- ligand-contact 适用性检查已补基础版：`inspect_cd38_ligand_candidates.py` 能判断结构是否真的含有接触 CD38 truth residues 的 ligand-like HETATM，避免把 apo 结构误拿来做 ligand-contact baseline。
- readiness 一键刷新已补基础版：`refresh_cd38_benchmark_readiness.py` 能把 ligand scan、扩展计划、P2Rank/fpocket readiness 聚合成一份总览，减少接入外部输出时漏步骤。
- 外部工具输入包已补基础版：`prepare_cd38_external_tool_inputs.py` 能生成 PDB 输入副本、P2Rank/fpocket PowerShell 和 Bash 模板、输出目录约定、expected return manifest/checklist 和后续 readiness 刷新脚本，让外部工具接入从“手工找文件”变成“按包执行、按清单返回”。
- 外部工具转移包已补基础版：`package_cd38_external_tool_inputs.py` 能生成干净的 transfer zip，方便传到 Linux/WSL 或另一台机器运行 P2Rank/fpocket，再把输出复制回来导入。
- 外部输出导入已补基础版：`import_cd38_external_tool_outputs.py` 能从返回目录或 zip 中只接收 P2Rank/fpocket 输出，默认不覆盖已有文件；支持返回包外面多包一层目录，并输出 scan manifest、coverage manifest、repair plan 和 `source_diagnosis`，能直接识别“原始输入包被误当成返回输出包”的情况，也能按 `PDB × method` 判断返回包缺哪些结构方法组合，并把缺口转成应运行模板、应返回路径和 dry-run 验证命令，减少手工复制结果时路径错位。
- 外部工具环境/输出预检已补基础版：`check_cd38_external_tool_environment.py` 能明确区分 `prank` / `fpocket` 未安装、PDB 输入缺失、P2Rank CSV 缺失和 fpocket pocket 文件缺失；并采用 `package_portable_first`，输入包跨机器移动后优先看当前 package 内路径，避免真实 benchmark 卡在不透明的路径问题上。
- 外部 benchmark finalize 已补基础版：`finalize_cd38_external_benchmark.py` 把返回包导入、preflight、readiness、可选 benchmark 导入和可选参数敏感性刷新合成一个安全收尾入口；支持 `--import_source <returned_zip_or_dir>`，默认只检查不导入；如果 readiness 发现 0 条可运行 rows，会跳过旧 panel 汇总，并在报告中链接 repair plan，减少误判。
- 几何 proxy 校准报告已补基础版：`build_cd38_proxy_calibration_report.py` 会把运行时 `pocket_shape_overwide_proxy` 与 CD38 truth-based benchmark 指标对照，输出阈值候选和默认惩罚权重建议；当前明确给出“proxy 有方向性，但证据仍不足以改变默认权重”的保守结论。
- 公开结构 starter 已补基础版：`run_cd38_public_starter.py` 可一键刷新本地公开 CD38 benchmark 面板、ligand scan、参数敏感性、proxy calibration、preflight、readiness、action plan 和 next-run runbook；本地软件“诊断”页也已接入同一入口，可直接刷新、查看缺口、打开报告并下载 action plan / expected returns / next-run 清单，还能生成带 next-run 脚本的 transfer zip、检查返回包、执行 finalize、运行返回包导入自测和返回包安全门控；本地软件和 CLI finalize 都会在正式导入前自动 gate 拦截非 `PASS_*` 返回包，让新用户先看一份总览报告，再决定是否补外部 fpocket/P2Rank 输出。
- 对 fpocket / P2Rank 输出已支持批量导入和统一命名。
- 在结果面板中展示不同 pocket 方法的差异。
- 后续如条件允许，再做一键调用外部工具。

## 5. 建议后续重点改进方向

### 5.1 第一优先级：把“可信度解释”做得更细

建议新增或增强：

- 低可信原因拆解。
- Rule/ML 分歧解释。
- top-k 分数接近程度。
- conformer 间波动解释。
- pocket overwide 对当前候选的影响说明。

价值：

- 比单纯提高模型复杂度更实际。
- 更适合科研使用，因为用户需要知道“为什么要复核”。

### 5.2 第二优先级：补真实 benchmark 和 demo 数据

建议新增或增强：

- 小型 demo 数据集已补基础版：可用 `python run_demo_pipeline.py` 直接生成 `demo_data/` 和 `demo_outputs/`，也可在本地软件侧边栏生成并立即运行 demo；demo 输出自带 `DEMO_OVERVIEW.html`、`DEMO_README.md`、`DEMO_INTERPRETATION.md` 和真实数据 starter 模板包，适合安装检查、流程演示和答辩展示。
- CD38 更多结构；当前已经把 `3F6Y` 放入待补目标表，并确认它不适合 ligand-contact baseline，下一步需要补 P2Rank/fpocket 外部工具输出。
- 真实 fpocket 输出 baseline；当前已经有外部工具输入包、readiness report 和 manifest 入口，下一步关键是在外部运行 fpocket、放入真实 `pocket*_atm.pdb` 并复跑 panel。
- 真实 P2Rank 输出扩展 baseline；当前已经有外部工具输入包、readiness report 和 manifest 入口，下一步关键是补 `3F6Y` 的真实 `*_predictions.csv`。
- 非 CD38 的第二个蛋白 benchmark。
- 带真实标签的独立 test set。

价值：

- 提升可信度。
- 让比赛展示更有说服力。
- 避免只在 synthetic / 单个蛋白上说明效果。

### 5.3 第三优先级：增强候选对比页

当前状态：已完成基础增强。

同时已完成“低可信原因拆解”基础增强：

- `consensus_ranking.csv` 已输出 `review_reason_flags` 和 `low_confidence_reasons`。
- 细分原因包括 Rule/ML 排名分歧、Rule/ML 分数分歧、pocket overwide、构象间波动、相邻候选分差过近、失败行和 warning 行。
- 候选报告卡和下一轮实验建议已经读取这些字段，便于直接给出“为什么需要复核”。
- 下一轮实验建议已加入 diversity-aware ordering，避免实验资源过度集中在同一类候选，并保留原始 priority score 便于审计。
- 现在还能输出 `experiment_plan.csv/md` 和 `experiment_plan_state_ledger.csv`，把当前轮要做的候选、备用候选和暂缓候选整理成可执行计划单，并支持人工覆盖 include/exclude/standby/defer、实验状态、负责人、成本和备注；本地软件可直接编辑后作为下一轮输入，也可跨多个历史运行汇总成全局 ledger，并保守生成真实验证回灌报告。
- 推荐 pipeline 还会自动输出 `validation_evidence_audit/validation_evidence_report.md`、top-k 验证覆盖表和行动清单，用来回答“当前高排名候选是否已有真实实验支撑，还缺哪些结果”。
- 验证回灌已经形成安全再训练入口：带实验标签的特征表默认写入 `experiment_label`，本地软件可一键切换到 `feature_csv + label_col=experiment_label`，避免覆盖原始 `label`。
- 全局 ledger 不只是下载 CSV，还能在本地软件里按状态、结果、override 和关键词筛选，并直接看到可回灌标签数和状态分布，这比很多只给静态表格的工具更适合持续实验管理。
- 验证回灌后还能生成“再训练前后对照报告”，把标签数量、Rule/ML 一致性、训练 loss、top-k 重叠和候选 rank delta 放在同一份报告里，避免只看一次重新运行的结果。
- 当运行次数变多后，项目还能自动生成结果归档索引和长期趋势表，把每次运行的关键指标、关键产物路径和验证回灌对照趋势汇总到固定目录，减少“结果散落在多个文件夹里”的问题。

已经新增：

- `build_candidate_comparisons.py`
- `candidate_tradeoff_table.csv`
- `candidate_pairwise_comparisons.csv`
- `candidate_group_comparison_summary.csv`
- `candidate_comparison_summary.json`
- `candidate_comparison_report.md`
- 本地软件“排名结果”页中的候选对比解释预览和下载
- 本地软件“排名结果”页中的自定义候选对比，可手工选择 2 到 5 个 nanobody
- 本地软件“排名结果”页中的候选分组对比小结，可按 diversity/family/status/risk 分组查看共同优势和共同风险
- 候选报告卡内嵌相邻候选对比摘要、close decision 和解释文本

价值：

- 很符合真实决策场景。
- 比单张大排名表更容易用于答辩。
- 用户打开单个候选报告卡时就能看到“它和相邻候选相比强在哪里、弱在哪里”。

后续仍可增强：

- 对 close decision 候选自动生成“建议一起验证”的实验建议。
- 把候选分组小结进一步嵌入单个候选报告卡。

### 5.4 第四优先级：完善 provenance 运行卡片

当前状态：已完成基础版。

已经新增：

- 输入文件 hash。
- 输入 CSV / feature CSV 引用的 `pdb_path`、`pocket_file`、`catalytic_file`、`ligand_file` 行级 manifest。
- 参数 hash 和 manifest hash。
- 脚本文件 hash。
- 依赖版本。
- Git commit、dirty 状态和本地变更数量。
- 输出文件清单。
- `run_provenance_card.json/md`、`run_artifact_manifest.csv`、`run_input_file_manifest.csv` 和 `run_provenance_integrity.json`。
- `verify_run_provenance.py` 可重新计算 SHA256，检查 provenance 卡片、artifact manifest 和 input file manifest 是否被后续误改。

价值：

- 提升科研可复现性。
- 更适合团队协作和长期迭代。

后续仍可增强：

- 如需更强审计，再增加带私钥的正式数字签名；当前基础版是无私钥 SHA256 完整性封存。
- 当前已补 `result_archive_lineage.csv` 和 `result_archive_lineage_graph.json/html/md` 基础版；可在本地归档页直接预览共享输入、共享特征表和共享参数的复跑时间线。

### 5.5 第五优先级：AI 解释层，但不替代评分

当前状态：已完成基础版。

已经新增：

- `ai_assistant.py`
- 推荐 pipeline 的 `--enable_ai_assistant`
- 本地软件“AI 解释（可选）”开关和“AI 解释”结果页
- 离线摘要默认可用；OpenAI provider 可选启用

价值：

- 把分散的 summary、排名、候选对比、下一轮实验建议整理成更容易阅读的解释报告。
- 比同类只输出 score 的工具更适合比赛展示和团队沟通。
- 默认不上传原始 PDB 或完整 CSV，降低数据泄露风险。

后续仍可增强：

- 让用户在 UI 中对某个候选直接提问。
- 生成“答辩讲稿版”摘要。
- 失败时读取 `app_run_stderr.txt` 做更具体的一键排查。
- 接入本地 LLM provider，满足不能联网的数据环境。

### 5.6 第六优先级：进一步产品化本地软件

建议新增或优化：

- 一键 demo 已有基础增强版，命令行、bat 和本地软件侧边栏均可进入；本地软件已经支持“生成并立即运行 demo”、demo HTML 导览、demo 输出说明文件、demo 结果解读页、真实数据 starter 模板包、可运行 mini PDB 示例包和摘要页一键打开导览；CD38 public starter、transfer zip、next-run runbook、返回包 dry-run/finalize、返回包导入自测、安全门控和导入前 gate 保护也已进入诊断页。后续重点是补真实外部 P2Rank/fpocket 输出，而不是继续增加入口。
- 新手模式 / 高级模式。
- 更清晰的数据导入向导。
- 输出目录自动命名和归档。
- 跨机器 standalone 验证。
- 启动失败一键诊断包。

价值：

- 降低使用门槛。
- 更像一个可交付的软件，而不是脚本集合。

## 6. 可以对外强调的优势表述

推荐表述：

> 本项目不是单纯的 docking 打分器，也不是普通的结构查看器，而是一个围绕“纳米抗体是否可能阻断蛋白 pocket / 功能区域”构建的本地交互式筛选系统。它把结构几何特征、规则评分、伪标签 ML、QC 风险、方法共识、参数敏感性和候选报告卡整合到同一条流程中，既能批量处理候选，又能输出可解释、可复核、可展示的结果。

更简短的答辩版本：

> 同类工具往往只能看结构、找 pocket 或输出 docking score，但不能直接回答“哪个纳米抗体更值得做下一步实验”。我们的优势是把 pocket blocking 这个具体问题做成了从输入、批量特征、Rule/ML 共识排序、可信度解释、benchmark 到报告导出的闭环流程。

需要避免的过度表述：

- 不要说“已经能准确预测真实药效”。
- 不要说“完全替代 P2Rank / fpocket / docking 工具”。
- 不要说“模型已经具备严格泛化能力”。
- 不要说“几何 proxy 等价于真实动力学模拟”。

## 7. 当前最清晰的竞争优势总结

| 维度 | 同类工具常见情况 | 当前项目优势 |
|---|---|---|
| 任务聚焦 | 通用结构查看、通用 pocket 查找或通用 docking score | 聚焦纳米抗体 pocket blocking 候选筛选 |
| 批量处理 | 往往需要用户自己写脚本，路径错位后只能手工排查 | 已支持 input 表、zip、目录扫描、队列、历史、缺失路径自动定位、修复建议 CSV、修复版 input_csv 和 PASS/WARN/FAIL 质量门控 |
| 评分方式 | 单一 score 或人工经验 | Rule + ML + QC 风险共识排序 |
| 解释能力 | 输出分数后解释不足 | explanation、分数解释卡片、本批次结论摘要、候选报告卡、候选横向对比、低可信提示、diversity-aware 下一轮实验建议、可编辑且可跨批次继承的实验计划单、可选 AI 摘要 |
| pocket 风险 | pocket 边界不确定性经常被隐藏 | 方法共识、overwide proxy、geometry proxy audit、CD38 benchmark、参数敏感性、外部工具输入包、transfer zip、返回结果导入、preflight、readiness、finalize 检查、外部 benchmark action plan 和 next-run runbook |
| 展示交付 | 结果散落在文件或 notebook | 本批次一页结论、HTML / PDF / zip / 本地软件界面、一键 demo 数据、演示输出、HTML 导览、示例结果解读页和真实数据迁移模板 |
| 复现与调试 | 依赖用户手工记录 | smoke test、run summary、QC、质量门控、历史运行、版本元数据、provenance 运行卡片 |
| 真实验证闭环 | 实验结果和模型输入容易脱节 | 全局实验 ledger、状态筛选图表、验证标签审计报告、验证证据审计、一页式 batch decision 证据提示、可选带 experiment_label 的特征表、一键验证标签再训练入口、再训练前后对照报告、长期趋势归档 |

新增的“本批次结论摘要”进一步强化了这个优势：同类工具通常只给结果文件或单一分数，用户还要自己判断本批数据能不能用、哪个候选最稳、哪个风险最大、下一轮该先做谁。本项目现在会把这些判断收敛成 `batch_decision_summary.md/json/csv`，并在本地软件摘要页直接展示，适合比赛答辩和非开发用户快速理解。

验证证据审计已经进一步合入本批次结论摘要：当 Quality Gate 通过但 top 候选还没有真实 `experiment_result` 或 `validation_label` 时，摘要会明确提示“可解读排序，但实验前需要补验证证据”。这能避免把模型推荐误讲成实验确认，是科研展示里很重要的边界控制。

新增的 CD38 公开结构 starter、外部 benchmark action plan 和 next-run runbook 也强化了“可执行验证闭环”这一点：同类工具经常只说“还需要更多 benchmark”，但不会告诉用户当前已有几行公开结构结果、下一步具体缺哪个结构、哪个工具输出、应该复制回哪个路径、外部机器上到底先跑哪条脚本。本项目现在可以在本地软件“诊断”页或命令行用 `run_cd38_public_starter.py` 生成总览，再把 `3ROP/4OGW/3F6Y × P2Rank/fpocket` 的缺口拆成 priority 清单，区分真正影响 benchmark 完整性的 blocker 和仅用于外部包复现的补充项；同时可以在界面里生成包含 `run_cd38_external_next_benchmark.*` 的 transfer zip，并对返回包做 dry-run 诊断，避免把原始输入包误当成真实结果导入。新增的返回包导入自测和安全门控进一步保证路径约定自身可回归验证，并且能把 synthetic fixture 显式拦截，避免真实结果回来后才发现 importer 规则不匹配或误把自测包当证据。

新增的 geometry proxy audit 则补强了另一个差异点：它不会盲目相信单一阻断分数，而是检查 mouth、path、pocket contact、catalytic contact 和 pocket shape 这些静态 proxy 是否自洽。这样可以在不改变模型输出的前提下，把“看起来分高但可能只堵住一部分路径”的候选提前标出来。

## 8. 一句话结论

当前项目最大的优势不是“模型最复杂”，而是把一个明确的生物信息学筛选问题做成了较完整的本地软件闭环：能批量输入、能解释排序、能提示风险、能导出报告、能持续 benchmark。后续最重要的改进不是盲目加模型，而是补真实验证数据、增强可信度解释、继续校准几何 proxy，并把本地软件体验进一步简化。

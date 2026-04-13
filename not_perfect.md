# 当前仍不完善的版块

这份文档记录的是：目前已经能跑、也已经能支撑验证的部分，但仍然偏启发式、近似或缺少系统性校准的地方。后续迭代优先从这些点补强。

## 已完成的后续完善（2026-04-08）

已落地如下增强：

- 新增 [calibrate_rule_ranker.py](calibrate_rule_ranker.py)：支持基于 `label` 的规则权重自动校准（特征权重搜索 + 聚合权重网格搜索）。
- 新增 [compare_rule_ml_rankings.py](compare_rule_ml_rankings.py)：输出规则版 vs ML 版相关性、Top-K 重叠和差异样本报表。
- 新增 [pipeline_smoke_test.py](pipeline_smoke_test.py)：自动生成可复现 synthetic `pose_features.csv` 并串联端到端烟测。
- 增加 `requirements.txt`，补齐基础依赖声明。
- 在当前 Python 3.13 环境中补齐可运行路径（含 `pandas` 安装与验证）。

## 已完成的后续完善（2026-04-10）

- 在 [geometry_features.py](geometry_features.py) 增加路径连续阻断与瓶颈代理特征：
  - `ligand_path_block_fraction`
  - `ligand_path_block_fraction_weighted`
  - `ligand_path_min_clearance`
  - `ligand_path_bottleneck_score`
- 将上述路径特征接入 [rule_ranker.py](rule_ranker.py)、[train_pose_model.py](train_pose_model.py)、[rank_nanobodies.py](rank_nanobodies.py) 的可选评分链路。
- 已在 smoke test synthetic 数据中加入新列，确保新增特征有端到端回归覆盖。
- 在 [calibrate_rule_ranker.py](calibrate_rule_ranker.py) 增加联合目标能力：可同时优化 pose AUC、nanobody AUC 和与 ML 排名一致性（Spearman）。
- 在校准流程中增加可选约束与选择策略（`selection_metric`、`min_nanobody_auc`、`min_rank_consistency`），并在 smoke test 中支持 baseline 守护，降低“校准后退化”概率。
- 新增 [optimize_calibration_strategy.py](optimize_calibration_strategy.py)，可基于已生成 trial 快速扫描策略并输出推荐配置，减少手工反复试参。

## 已完成的后续完善（2026-04-14）

- 在 [pipeline_smoke_test.py](pipeline_smoke_test.py) 增加“推荐策略回灌”能力：若检测到上一轮 `strategy_optimization/recommended_strategy.json`，会自动将推荐的 `rank_consistency_weight`、`selection_metric` 与约束阈值注入下一轮校准。
- 支持通过 `--strategy_seed_json` 指定任意推荐策略文件，并可用 `--disable_auto_seed_from_previous_strategy` 禁用自动回灌。
- 在 [pipeline_smoke_test.py](pipeline_smoke_test.py) 的 summary 中新增 `calibration_strategy_seed` 字段，记录本轮是否启用回灌、来源文件及最终生效参数，便于追踪可复现性。
- 在 [calibrate_rule_ranker.py](calibrate_rule_ranker.py) 中将 `rank_consistency_weight` 默认值固化为 `0.40`，并在未提供 `ml_ranking_csv` 时自动降级为有效值 `0.0`，避免默认配置误用。

## 已完成的本地交互壳基础版（2026-04-14）

- 新增 [local_ml_app.py](local_ml_app.py)，用最小侵入方式把现有 CLI 包成了一个本地交互式运行界面。
- 新增 [start_local_app.bat](start_local_app.bat)，在 Windows 下可直接启动本地交互界面。
- 当前基础版已支持：
  - 从 `input_csv` 或 `pose_features.csv` 启动
  - 上传文件或直接填写本机路径
  - 设置默认 `pocket/catalytic/ligand` 文件
  - 设置常用训练和排序参数
  - 运行 [run_recommended_pipeline.py](run_recommended_pipeline.py)
  - 在页面中预览 `nanobody_ranking.csv`、`pose_predictions.csv`、`recommended_pipeline_report.md`
  - 下载关键 JSON / CSV / Markdown 产物
- 新增 `.gitignore`，避免把 `.venv`、本地运行产物和测试输出目录直接提交到 GitHub。

## 已推进的展示层准备（2026-04-14）

- 新增 [export_structure_annotations.py](export_structure_annotations.py)，在不改现有 ranking / training 主链路的前提下，补出一层面向 viewer 的结构化导出。
- 当前已支持对单个复合物导出：
  - `residue_annotations.csv`
  - `residue_annotations.json`
  - `interface_residues.csv`
  - `interface_pairs.csv`
  - `key_residues.csv`
  - `structure_annotation_summary.json`
  - `pocket_payload.json`
  - `blocking_summary.json`
  - `analysis_bundle.json`
- 导出内容已覆盖第一步最关键的 residue/interface 基础载荷：
  - 残基 UID、链 ID、残基名、编号
  - 是否属于 `pocket` / `catalytic` / `interface`
  - 是否属于 `user key residue`
  - 到对链和 pocket 的最短距离
  - 最近对链 partner residue 的明细
  - pocket-contact coverage 等展示型 summary
- `analysis_bundle.json` 现已包含更适合 viewer 直接消费的两层：
  - `pockets`
  - `blocking_summary`
- 已支持通过命令行直接输入或文件输入用户关键残基：
  - `--key_residues`
  - `--key_residue_file`
  - `--key_residue_default_chain`
- 这一步的目标不是替代现有打分脚本，而是先把“点击残基面板、界面高亮、pocket 联动”所需的数据契约固定下来。

## 目标修正（2026-04-14）

这里补一条方向修正，避免后续继续误判：

- 当前更优先的目标，不是做 `3D viewer`。
- 当前更优先的目标，是把现有 ML 分析流程包成一个“本地可交互软件”。
- 重点不是三维结构交互，而是：
  - 更方便地上传 `CSV / PDB / pocket / catalytic / ligand` 等输入文件
  - 更直观地填写运行参数
  - 一键执行现有 pipeline
  - 更清楚地查看 ranking / report / summary / error 信息
  - 更方便地导出结果文件并做本地复核

因此，后续“可视化”应优先理解为：

- 表格化结果展示
- 指标摘要卡片
- 日志/报错可读化
- 输出文件下载与路径管理

而不是优先去做三维结构 viewer。

## 面向“本地交互式软件”的现状判断

先按最小改动原则看当前仓库：

- 当前内核已经比较完整：
  - [build_feature_table.py](build_feature_table.py)
  - [rule_ranker.py](rule_ranker.py)
  - [train_pose_model.py](train_pose_model.py)
  - [rank_nanobodies.py](rank_nanobodies.py)
  - [run_recommended_pipeline.py](run_recommended_pipeline.py)
- 当前最适合直接复用的统一入口，其实不是某个单独模型文件，而是 [run_recommended_pipeline.py](run_recommended_pipeline.py)。
- [pipeline_runner_common.py](pipeline_runner_common.py) 已经具备很适合做本地软件壳的基础能力：
  - 子命令执行
  - 输出文件校验
  - Markdown 报告拼装
  - 执行摘要序列化

这意味着：

- 没必要重写算法主链路。
- 没必要先做复杂后端服务。
- 更合理的方向是：在现有脚本外面补一层“本地交互式运行壳”。

## 面向“本地交互式软件”的最小改动方案

当前最建议的技术路线不是重做成大型前后端分离系统，而是：

1. 保留当前 Python 分析核心和 CLI。
2. 新增一个本地 UI 壳，只负责收集输入、触发现有脚本、展示结果。
3. 让 UI 首先消费这些现成产物：
   - `recommended_pipeline_summary.json`
   - `recommended_pipeline_report.md`
   - `nanobody_ranking.csv`
   - `pose_predictions.csv`
   - `feature_qc.json`
4. 后续如果需要，再把这个本地 UI 壳打包成桌面应用。

这个方向的核心优点：

- 改动小
- 风险低
- 不破坏现有脚本可用性
- 很适合比赛展示和本地演示

## 面向“本地交互式软件”的建议功能优先级

### 第一优先级：先把“能用”做出来

- 本地页面/窗口里选择 `input_csv` 或 `pose_features.csv`
- 选择默认 `pocket_file` / `catalytic_file` / `ligand_file`
- 填写常用参数：
  - `top_k`
  - `epochs`
  - `batch_size`
  - `seed`
- 点击按钮后执行 [run_recommended_pipeline.py](run_recommended_pipeline.py)
- 在界面里显示：
  - 当前运行状态
  - 命令执行成功/失败
  - 输出目录
  - 关键产物路径

### 第二优先级：把“结果更直观”做出来

- 直接在界面里预览：
  - `nanobody_ranking.csv`
  - `pose_predictions.csv`
  - `training_summary.json`
  - `recommended_pipeline_report.md`
- 增加“只看 Top-N 排名”的筛选
- 增加基础统计卡片：
  - 样本数
  - nanobody 数
  - 是否启用 calibration
  - rule/ML 主要指标
- 增加失败行或 warning 的可读展示

### 第三优先级：把“本地软件体验”补齐

- 最近一次运行历史
- 一键打开输出目录
- 一键导出当前结果汇总
- 输入参数保存/载入
- 常见错误提示模板化

## 面向“本地交互式软件”的建议后续顺序

1. [未完成] 把 [run_recommended_pipeline.py](run_recommended_pipeline.py) 抽成“CLI + 可调用函数”双入口，减少 UI 壳和命令行逻辑重复。
2. [已完成基础版] 新增本地交互入口文件，先做最小运行面板，不碰算法逻辑。
3. [已完成基础版] 把 `recommended_pipeline_summary.json`、`recommended_pipeline_report.md`、`nanobody_ranking.csv` 接成结果面板。
4. [进行中] 再补上传体验、历史记录、导出按钮。
5. [未完成] 最后再考虑是否打包成桌面可执行程序。

## 1. 几何特征仍以 proxy 为主（已部分收紧）

文件: [geometry_features.py](geometry_features.py)

当前已实现的几何特征已经可以区分很多“堵口袋”和“普通表面结合”的情形，但其中一部分仍是静态代理量，不是严格物理模拟：

- `mouth_occlusion_score` 依赖口袋口部候选点和局部接触的近似定义。
- `delta_pocket_occupancy_proxy` 和 `pocket_block_volume_proxy` 是占据/阻断的代理，不是显式体积模拟。
- `ligand_path_block_score` 已升级为融合连续阻断比例/瓶颈分数的静态近似，但仍不是完整动力学路径评估。
- `substrate_overlap_score` 主要衡量几何冲突近似，仍可能受 ligand template 质量影响。

这意味着：几何特征已经可用，但仍需要用真实数据或更强的结构验证继续校准。

## 2. pocket/catalytic/ligand 输入质量仍决定上限

文件: [build_feature_table.py](build_feature_table.py)、[pocket_io.py](pocket_io.py)、[pdb_parser.py](pdb_parser.py)

当前流程对输入质量已经做了不少容错，但仍有上限：

- pocket、catalytic、ligand 文件如果定义不完整，相关特征会退化为 `NaN` 或默认值。
- PDB 的链 ID、TER、altloc、插入码等问题已经加固，但极端不规范结构仍可能影响拆链和映射。
- `build_feature_table.py` 的 `skip_failed_rows` 能保证批量不中断，但也可能掩盖真实失败率，必须结合 `feature_qc.json` 一起看。

## 3. 规则版排名仍是启发式加权，不是拟合出来的分数

文件: [rule_ranker.py](rule_ranker.py)

规则版排名可以很好地用来做 pipeline sanity check，但它本质上仍是人工定义的加权公式：

- `rule_blocking_score` 依赖固定方向图和固定权重。
- `conformer_rule_score` 和 `final_rule_score` 的权重目前是可配置的默认值，但不是从 benchmark 自动学习得到的。
- `pocket_consistency_score` 是稳定性奖励的近似实现，适合做排序辅助，但不是严格的统计一致性度量。

结论：规则版适合“先验验证”和“特征回归检查”，不适合当最终可泛化评分器直接上线。

## 4. ML 伪标签仍然是伪标签

文件: [train_pose_model.py](train_pose_model.py)

当前 ML 路线已经更稳，但核心训练目标仍然是伪标签驱动：

- `build_pseudo_labels(...)` 使用固定方向图、稳健归一化和阈值模式，适合做初始化，但不等于真实标签。
- `pseudo_rank`、`pseudo_components` 能提升可解释性，但不能替代真实的监督信号。
- 当没有真实 `label` 时，模型学到的仍然是“当前 feature pipeline 的偏好”，而不是严格意义上的 ground truth。

如果后续要把模型结果做成正式评分，最好引入一批人工标注或实验验证样本做校准。

## 5. 训练与评估还缺少更完整的外部验证

文件: [train_pose_model.py](train_pose_model.py)、[rank_nanobodies.py](rank_nanobodies.py)

当前训练和排序流程已经有基本闭环，但还不算完整评估体系：

- 目前主要依赖单次 train/val 切分，没有系统交叉验证。
- 没有独立 test set、外部 benchmark、calibration curve 或 reliability analysis。
- 模型选择主要基于 `val_loss`，还没有和实际结构验证结果做闭环对齐。
- `rank_nanobodies.py` 的最终分数已经加入一致性项，但仍需要真实成功案例/失败案例做权重回归。

## 6. 解释字段仍是规则拼接，不是因果解释

文件: [rank_nanobodies.py](rank_nanobodies.py)、[rule_ranker.py](rule_ranker.py)

`explanation` 字段已经比第一轮强很多，但本质上仍是阈值 + 规则拼接：

- 它能解释“为什么分高”，但不能证明“为什么一定阻断 pocket”。
- 对 ligand template 冲突、mouth occlusion、高一致性等描述，仍属于经验型总结。
- 目前还没有 SHAP、特征归因图或结构可视化输出来辅助人工审阅。

## 7. 对照报表已补齐，但真实真值闭环仍不足

当前已经会输出：

- `pose_features.csv`
- `feature_qc.json`
- `pose_predictions.csv`
- `conformer_scores.csv`
- `nanobody_ranking.csv`
- `pose_rule_scores.csv`
- `conformer_rule_scores.csv`
- `nanobody_rule_ranking.csv`

本轮已补齐：

- 规则版 vs ML 版并排对照表（`ranking_comparison_table.csv`）
- 对照汇总 JSON 与 Markdown 报告
- 权重/阈值调参历史（feature/aggregation calibration trials）

仍缺少：

- 基于真实实验结果的长期跟踪报表（跨批次、跨时间）
- 真值驱动的外部 benchmark（而不只是 synthetic 或局部样本）

## 8. 运行环境风险下降，但仍需固定化

当前 `Python 3.13` 已可导入并运行 `numpy/pandas/biopython/torch`，并可执行端到端 smoke test。

已完成：

- 依赖版本已固定到 `requirements.txt`。
- 新增 `.github/workflows/smoke-test.yml`，在 push/PR 自动执行端到端 smoke test。

仍建议继续做：

- 如果后续需要 GPU/CPU 双环境一致性，补一份针对不同平台的 lock 文件或约束文件（例如 constraints）。
- 在 CI 中增加多种种子和更小批次数据规模的快速回归矩阵，以更早捕获不稳定性。

## 9. 建议的后续完善顺序

1. [已完成] 用真实/标签可用数据时执行规则权重校准入口（脚本已提供）。
2. [已完成] 规则版与 ML 版对照评估报表。
3. [进行中] 把几何 proxy 再往物理意义上收紧，尤其口袋口部和路径阻断。
4. [已完成] 增加固定回归数据能力（synthetic 可复现数据生成 + smoke test 固定种子）。
5. [已完成] 在 Python 3.13 环境补一轮端到端烟测流程。
6. [已完成基础版] 导出 residue annotation / interface residue 的结构化载荷，给后续 viewer 或展示壳复用。

## 10. 如果后续扩展为“可视化展示型软件”，当前仍缺的版块

先说明现状：

- 当前仓库本质上还是“结构解析 + 特征工程 + 排名/报告”的分析内核。
- 目前没有真正的前端 viewer、后端 API、残基点击面板或图形化导出界面。
- 但现有代码已经具备一批很适合做可视化底层支撑的模块：
  - [pdb_parser.py](pdb_parser.py)：链拆分、残基 UID、残基/原子提取
  - [pocket_io.py](pocket_io.py)：pocket/catalytic 残基定义读取与归一化匹配
  - [geometry_features.py](geometry_features.py)：接触、mouth、路径阻断、interface centroid proxy 等几何基础
  - [build_feature_table.py](build_feature_table.py)：逐 pose 串联结构解析与特征汇总
  - [rule_ranker.py](rule_ranker.py)、[rank_nanobodies.py](rank_nanobodies.py)：可解释分数与 explanation 文本骨架

因此，后续如果真要做“展示型软件”，最合理的方式不是重写这些分析模块，而是先补一层面向 UI 的结构化导出。

### 10.1 residue-level 标注已有基础版，但仍缺更细的交互扩展字段

现在已经能导出 residue annotation，但要直接支撑完整点击面板，还建议继续补：

- `is_user_key_residue`
- `annotation_tags`
- 更细的距离/接触解释字段
- 与 pocket/interface 之外的自定义功能位点标签

### 10.2 interface residue 现在已支持 residue/pair 基础版阈值法

当前基础版已经能导出 interface residue，但后续仍建议继续做：

- 分链导出 interface partner 信息
- 输出每个界面残基的最近对链残基
- 输出 interface residue 对的明细而不只是 residue-level 标记
- 把阈值、链选择和 pocket-overlap 逻辑显式写入 summary

### 10.3 用户手动关键残基输入已支持基础版，但仍缺更完整的交互契约

当前基础版已经支持“用户输入一串残基编号然后高亮”所需的最小解析能力，但后续仍建议继续补强：

- `A:45`
- `A:45,A:46,A:47`
- `A:45 46 47`
- `B:102A`

后续仍可继续增加：

- pocket / antigen / nanobody 的分组视图
- 用户输入顺序保留
- 输入错误定位与提示信息
- 与 UI 侧“选中 residue chips”完全对齐的 payload

### 10.4 pocket 候选区域的“展示载荷”已有单 pocket 基础版，但仍缺多 pocket 能力

当前已经能表达一个 pocket residue 集合，并已输出基础版 `pocket_payload.json` / `analysis_bundle.json["pockets"]`，但还没有形成更完整的多 pocket 联动结构：

- pocket ID / pocket label
- pocket residue list
- pocket centroid
- pocket 来源（manual/json/csv/rule-based）
- 多 pocket 并列展示结构

当前已支持的基础版：

- 手工 residue 列表
- JSON/CSV 读入 pocket residue 列表
- 基于现有 pocket 定义的单 pocket 输出

之后再预留 fpocket / P2Rank 的适配入口。

### 10.5 简化阻断评分已有基础版，但仍缺与几何主链路更深的融合

当前已有 ranking 分数与 explanation，也已经新增基础版 `blocking_summary.json`，但后续仍可继续加强：

- `pocket_contact_coverage`
- `pocket_contact_count`
- `min_antibody_to_pocket_distance`
- `nanobody_pocket_contact_ratio`
- `mouth_occlusion_score`
- `blocking_score_simple`

当前基础版更偏“直观展示”，后续仍建议继续接入更多已有几何信号，例如 mouth/path proxy，而不是只停留在 residue contact 层。

### 10.6 还缺面向 viewer 的轻量统一 bundle / facade

虽然现在已经有了 `analysis_bundle.json` 的基础版，但后续仍建议继续稳定成真正的 viewer 契约：

- 结构文件路径
- pocket residues
- catalytic residues
- interface residues
- residue annotations
- blocking summary
- ranking summary

等这层 bundle 稳定后，再接前端 viewer 或比赛展示壳，返工成本会最低。

## 11. 面向“可视化展示型软件”的最小改动推进顺序

1. [已完成基础版] 先补 residue annotation JSON/CSV 导出。
2. [已完成基础版] 再补 interface residue 检测与 residue-level 导出。
3. [已完成基础版] 再补“用户手工关键残基输入”的统一解析函数。
4. [已完成基础版] 再补 pocket payload 标准化。
5. [已完成基础版] 再补简化阻断评分的拆解版导出。
6. 最后再补统一 `analysis_bundle.json` 或轻量 API facade。

## 12. 当前最建议的下一步

如果下一轮继续贴近“本地交互式软件”的目标，最建议先做的是：

1. 把 [run_recommended_pipeline.py](run_recommended_pipeline.py) 收敛成可直接被 UI 调用的函数入口，减少 subprocess 包装层。
2. 在 [local_ml_app.py](local_ml_app.py) 中补运行历史、错误定位、参数模板保存/载入。
3. 最后再考虑把当前本地页面进一步打包成桌面可执行程序。

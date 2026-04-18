# Minimal ML Test: Nanobody Pocket-Blocking Pipeline

本仓库用于做 VHH/Nanobody 的口袋阻断筛选，包含两条完整路线：

1. 规则路线（不依赖 ML）
2. ML 路线（伪标签 + MLP）

目标是先验证特征工程是否合理，再做模型训练与排序。

## 快速阅读指南

| 你想做什么 | 优先看 | 最短操作 |
|---|---|---|
| 直接打开本地软件 | [run.md](run.md) | 双击 `portable_dist\standalone_onefile\ML_Local_App_Standalone.exe`，没有单文件版就双击 `start_local_app.bat` |
| 没有数据，先看完整效果 | 本 README / [MODEL_QUICKSTART.md](MODEL_QUICKSTART.md) | `python run_demo_pipeline.py`，或双击 `run_demo_pipeline.bat` |
| 用最少命令跑模型 | [MODEL_QUICKSTART.md](MODEL_QUICKSTART.md) | `python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs` |
| 理解 ML 具体架构 | [ML.md](ML.md) | 先看总体数据流，再看 Rule、MLP、聚合和共识层 |
| 理解整个项目功能 | 本 README | 先看“当前推荐使用路径”，再按需看 CLI 和输入输出契约 |
| 继续开发下一步 | [not_perfect.md](not_perfect.md) | 默认优先推进真实 fpocket/P2Rank 输出导入和 benchmark 完整化 |
| 看 CD38 pocket benchmark | [CD38.md](CD38.md) / [benchmarks/cd38/BASELINE_RESULTS.md](benchmarks/cd38/BASELINE_RESULTS.md) | 先在本地软件“诊断”页刷新 CD38 public starter，或运行 `python run_cd38_public_starter.py`，再看 action plan 和 next-run runbook |

## 当前推荐使用路径

如果你还没有自己的数据，先运行一键 demo：

```bash
python run_demo_pipeline.py
```

它会生成可复现的 synthetic 示例输入、带 synthetic validation label 的实验 override，以及完整推荐 pipeline 输出。注意：demo 标签只用于演示工作流，不代表真实湿实验或真实生物学结论。

如果你正在使用本地软件，也可以在左侧“没有数据时”点击“生成并立即运行 demo”；如果想先检查参数，则点击“生成并载入 demo 输入”，再手动点击“立即运行”。

1. 准备 `input_pose_table.csv`，至少包含 `nanobody_id`、`conformer_id`、`pose_id`、`pdb_path`。
2. 运行 `python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs`。
3. 先打开 `my_outputs/recommended_pipeline_report.md`，确认流程是否完整跑完。
4. 再看 `my_outputs/batch_decision_summary/batch_decision_summary.md`，快速判断本批次能不能解读、优先看谁、先修什么风险。
5. 如果分数解释涉及 mouth/path/pocket 阻断，查看 `my_outputs/geometry_proxy_audit/geometry_proxy_audit_report.md`，确认几何 proxy 没有明显自相矛盾。
6. 如果要安排实验验证，查看 `my_outputs/validation_evidence_audit/validation_evidence_report.md`，确认 top 候选是否已有真实验证证据、还缺哪些结果。
7. 最后看 `my_outputs/score_explanation_cards/score_explanation_cards.html`、`my_outputs/consensus_outputs/consensus_ranking.csv` 和 `my_outputs/candidate_report_cards/index.html`，决定优先候选。

## 核心输出怎么理解

| 文件 | 作用 | 优先级 |
|---|---|---|
| `recommended_pipeline_report.md` | 本次流程执行摘要，先判断有没有失败或跳过 | 最高 |
| `quality_gate/quality_gate_report.md` | 统一 PASS/WARN/FAIL 质量判定，先看能不能解读本次结果 | 最高 |
| `batch_decision_summary/batch_decision_summary.md` | 把质量门控、最佳候选、稳定候选、最高风险候选、真实验证证据、失败/Warning Top-N 和下一步建议合成一页结论 | 最高 |
| `consensus_outputs/consensus_ranking.csv` | 综合 Rule、ML、QC 风险后的推荐排序 | 最高 |
| `geometry_proxy_audit/geometry_proxy_audit_report.md` | 检查 mouth/path/pocket/contact 等几何 proxy 是否互相矛盾，不改变分数 | 高 |
| `score_explanation_cards/score_explanation_cards.html` | 把分数、可信度、主要正向因素、主要风险和建议动作翻译成可读卡片 | 高 |
| `candidate_report_cards/index.html` | 每个候选的可读报告卡，适合展示和人工复核 | 高 |
| `candidate_comparisons/candidate_comparison_report.md` | 解释为什么候选 A 排在候选 B 前面 | 高 |
| `ai_outputs/ai_run_summary.md` | 可选 AI/离线解释摘要，适合快速读懂本次结果 | 高 |
| `provenance/run_provenance_card.md` | 记录输入、参数、依赖、代码、输出 hash 和输入文件引用 manifest，便于复现审计 | 高 |
| `experiment_suggestions/next_experiment_suggestions.csv` | 下一轮实验/复核优先级建议 | 高 |
| `experiment_suggestions/experiment_plan.md` | 带预算、quota、standby 和人工覆盖状态的本轮实验计划单 | 高 |
| `experiment_suggestions/experiment_plan_state_ledger.csv` | 可编辑/可继承的实验状态 ledger，可作为下一轮 override 来源 | 高 |
| `validation_evidence_audit/validation_evidence_report.md` | 检查 top 候选是否已有真实验证标签、正负标签是否平衡、下一步该补哪些实验结果 | 高 |
| `local_app_runs/experiment_state_ledger_global.csv` | 本地软件跨批次汇总出的最新实验状态，可直接作为下一轮 override | 高 |
| `local_app_runs/experiment_validation_feedback/experiment_validation_report.md` | 从实验 ledger 生成的真实验证回灌报告 | 高 |
| `ml_ranking_outputs/nanobody_ranking.csv` | 纯 ML 聚合排名 | 中 |
| `model_outputs/pose_predictions.csv` | pose 级别预测和特征追溯 | 中 |

---

## 1. 当前已实现的全部功能

### 1.1 结构与配置基础能力

文件: core_utils.py

已实现能力:
- 配置对象
- GeometryConfig: 几何阈值与裁剪
- TrainConfig: 训练超参数与校验
- ProjectConfig: 项目级配置封装
- 运行与复现
- log_message: 统一日志输出
- set_seed: Python/numpy/torch 统一随机种子
- ensure_dir: 目录安全创建
- 数值安全工具
- safe_to_float
- sanitize_numeric_array
- robust_minmax_scale
- ZScoreScaler
- CSV 读入与列检查辅助

### 1.2 复合物解析与抗原/纳米抗体拆分

文件: pdb_parser.py

已实现能力:
- 稳健解析 PDB，支持 first MODEL 提取
- 处理链 ID 缺失场景（含 TER/synthetic chain fallback）
- altloc 原子选择与重排
- 结构质量摘要与校验
- 抗原/纳米抗体拆分
- 支持显式链指定与自动推断
- 输出 split_mode/source_detail 便于追踪
- 原子/残基抽取与几何基础函数

### 1.3 pocket/catalytic/ligand 输入解析与映射

文件: pocket_io.py

已实现能力:
- pocket/catalytic 文本定义解析
- 支持 `A:45`、`A 45`、`A:45,67`、`A:37-40`、`B:37-40`、`C:37-40`、插入码等
- 结构残基索引与稳健匹配
- 精确匹配 + 受控宽松匹配
- 输出标准化匹配结果与摘要
- ligand 模板 PDB 解析（含 altloc 选择）
- 外部口袋工具输出适配入口
- parse_pykvfinder_output
- parse_fpocket_output
- parse_p2rank_output

### 1.4 几何特征工程

文件: geometry_features.py

已实现能力:
- 口袋命中与距离
- pocket_hit_count
- pocket_hit_fraction
- min_distance_to_pocket
- mean_min_distance_to_pocket_residues
- 催化位点命中
- catalytic_hit_count
- catalytic_hit_fraction
- min_distance_to_catalytic_residues
- catalytic_block_flag
- 中心/界面几何
- distance_to_pocket_center
- nanobody_centroid_to_pocket_center
- distance_interface_centroid_to_pocket_center
- 口袋口部阻挡
- mouth_hit_count
- mouth_hit_fraction
- mouth_occlusion_score
- mouth_axis_block_fraction
- mouth_aperture_block_fraction
- mouth_min_clearance
- 当前 `mouth_occlusion_score` 更强调轴向与孔径阻断的一致性，不再主要依赖单一子信号
- 局部占据代理
- delta_pocket_access_proxy
- delta_pocket_occupancy_proxy
- pocket_block_volume_proxy
- pocket 定义边界 QC
- pocket_shape_residue_count
- pocket_shape_centroid_radius_mean / p90 / max
- pocket_shape_atom_bbox_volume
- pocket_shape_overwide_proxy
- pocket_shape_tightness_proxy
- 底物冲突与路径阻断
- substrate_clash_count
- substrate_clash_fraction
- substrate_overlap_score
- ligand_path_block_score
- ligand_path_block_fraction
- ligand_path_block_fraction_weighted
- ligand_path_min_clearance
- ligand_path_bottleneck_score
- ligand_path_exit_block_fraction
- ligand_path_candidate_count
- 当前 `ligand_path_block_score` 会在多出口阻断共识之外，额外考虑是否仍存在较开放的逃逸路径
- 统一总入口
- compute_all_geometry_features
- 缺失输入容错，返回默认值与 debug 摘要

### 1.5 pose_features 批量构建

文件: build_feature_table.py

已实现能力:
- 从输入 pose 表逐行构建特征
- 与 pdb_parser/pocket_io/geometry_features 串联
- 自动合并可选数值列
- hdock_score/mmgbsa/interface_dg/rsite_accuracy/label 等
- 行级故障隔离
- 单行失败不影响全表
- 输出 status/error_message/warning_message
- 支持 PDB 结构缓存
- 输出质量统计
- summarize_processing_results
- collect_feature_qc_summary
- 保存 pose_features.csv 与 feature_qc.json
- CLI 支持默认 pocket/catalytic/ligand 文件、默认链、跳过失败行

### 1.6 ML 训练与预测

文件: train_pose_model.py

已实现能力:
- 自动选择可用特征列
- 过滤高缺失/近常量特征
- 伪标签构建
- 方向统一 + 稳健归一化 + 加权融合
- 阈值模式
- top_fraction
- quantile
- fixed
- 额外伪标签输出
- pseudo_score
- pseudo_label
- pseudo_rank
- pseudo_components
- 组级切分
- 按 nanobody_id 做 train/val 切分防泄漏
- MLP 训练增强
- BCEWithLogits + 可选 soft target 辅助损失
- early stopping
- gradient clipping
- ReduceLROnPlateau 调度
- 评估指标
- loss/auc/accuracy/precision/recall/f1
- 推理输出
- pred_prob
- pred_logit
- top_contributing_features
- 训练产物
- best_model.pt
- train_log.csv
- feature_columns.json
- training_summary.json
- training_summary.csv
- pose_predictions.csv

### 1.7 ML 结果聚合排名

文件: rank_nanobodies.py

已实现能力:
- pose -> conformer 聚合
- 按 pred_prob 选 top-k
- 计算 mean_topk_pred_prob/best_pose_prob
- 聚合 top-k 几何均值
- mean_topk_pocket_hit_fraction
- mean_topk_catalytic_hit_fraction
- mean_topk_mouth_occlusion_score
- mean_topk_mouth_aperture_block_fraction
- mean_topk_substrate_overlap_score
- mean_topk_ligand_path_exit_block_fraction
- conformer_score 采用 pred 主导 + 几何辅助修正
- conformer -> nanobody 聚合
- mean_conformer_score
- best_conformer_score
- std_conformer_score
- pocket_consistency_score
- final_score 采用可配置加权公式
- 自动 explanation 文本
- 输出 conformer_scores.csv 与 nanobody_ranking.csv

### 1.8 规则版排名（非 ML）

文件: rule_ranker.py

已实现能力:
- 从 pose_features.csv 直接构建 rule_blocking_score
- 特征方向统一 + 稳健归一化
- 支持几何 + 可选能量特征
- 已接入更强的 mouth/path 几何信号（轴向/孔径覆盖、多候选出口阻断）
- pose -> conformer 聚合
- rule 分数主导 + 几何辅助修正
- conformer -> nanobody 聚合
- 含 pocket_consistency_score 稳定性修正
- 可配置 final_rule_score 权重
- explanation 文本生成
- 输出
- pose_rule_scores.csv
- conformer_rule_scores.csv
- nanobody_rule_ranking.csv

### 1.9 后续完善工具链（校准 + 对照 + 烟测 + benchmark）

文件: calibrate_rule_ranker.py / compare_rule_ml_rankings.py / pipeline_smoke_test.py / benchmark_pose_pipeline.py

已实现能力:
- 规则权重自动校准（需 label）
- 特征权重随机/扰动搜索 + 聚合权重网格搜索
- 同时评估 pose-level AUC 与 nanobody-level AUC
- 输出可复用的 calibrated_rule_config.json
- 规则版 vs ML 版并排评估
- score/rank 相关性（Spearman/Kendall）
- Top-K overlap 与差异样本表
- 一键端到端 smoke test（自动造可复现 synthetic pose_features）
- 自动串联 rule -> ML -> consensus -> calibration -> comparison
- 输出 smoke_test_summary.json
- 输出 smoke_test_report.md 便于快速审阅本轮回归

- 新增 grouped CV benchmark 入口 `benchmark_pose_pipeline.py`
- 支持按 `nanobody_id` 做分组交叉验证
- 支持输出 pose / nanobody 两层 reliability curve、ECE、Brier
- 支持直接生成 ML / rule 并排的 nanobody benchmark 表
- 支持输出 `geometry_proxy_benchmark.csv`，用于快速比较各个几何 proxy 的 held-out 表现
- 新增推荐执行入口 `run_recommended_pipeline.py`
- 支持从 `input_csv` 或已有 `pose_features.csv` 启动
- 已支持“CLI + 可调用函数”双入口
- 自动识别是否有足够 `label` 来继续 compare/calibrate/strategy optimize
- 输出 `recommended_pipeline_report.md`，汇总本次执行了哪些步骤、关键指标和关键产物
- 新增本地交互入口 [local_ml_app.py](local_ml_app.py)
- 支持通过本地页面上传或直接指定本机文件路径
- 支持导入单个 `zip` 数据包并自动识别 `input_csv` / `feature_csv` / `pocket` / `catalytic` / `ligand` 文件
- 支持直接扫描本地数据目录并自动识别同类输入
- 支持直接下载 `input_csv` 与 `pose_features.csv` 模板
- 支持在页面中填写常用参数、执行推荐流程、预览排名和下载关键产物
- 支持在正式运行前检查主输入必需列、label 状态和默认文件来源
- 支持把当前表单配置加入批量任务队列，并按顺序启动
- 支持停止当前后台运行任务，保留剩余队列
- 支持把当前运行导出为单文件展示摘要 HTML，适合直接发给别人或浏览器打印成 PDF
- 支持把当前运行直接自动导出为 PDF 摘要，不再依赖手工浏览器打印
- 自动化 PDF 当前已具备更适合展示的基础版式：标题区、指标卡、分区卡片、表格块和页脚页码
- 支持单独查看 `feature_qc.json`、失败行和 warning 行，便于快速定位输入问题
- 支持保存/载入参数模板，减少重复填写
- 支持加载最近运行历史并恢复结果与表单参数
- 支持在页面中选择多次历史运行做并排对比，并导出对比表 CSV
- 支持把当前多运行对比直接导出成 HTML 汇总页，并在本机打开
- 支持把当前多运行对比直接自动导出为 PDF 汇总页
- 多运行对比 PDF 当前也会带统一版式，而不再只是纯文本堆叠
- 支持基于指定基准运行查看多运行趋势和 run-to-run 差异解释
- 支持对每个运行输出更细的差异归因，包括正向驱动、负向拖累和归因摘要
- 支持按 `started_at` 日期对多运行结果做跨批次趋势聚合，并导出批次汇总 CSV/HTML/PDF
- 支持按关键词 / 状态 / `start_mode` 筛选历史运行记录和多运行对比候选集
- 支持对任务队列中的单条任务做上移、下移和移除，不再只能整队清空
- 支持直接基于历史运行复制配置并重新入队，减少重复填写
- 支持把当前排名页 / Pose 页的筛选结果直接导出为 CSV
- 支持在“历史”页查看应用产物目录占用，并安全清理历史运行、对比导出和导入缓存
- 支持在 ranking / pose 结果页自定义显示列、排序列，并分别导出“当前可见视图”与“当前全筛选结果”
- 支持在 ranking / pose 结果页按数值阈值做细筛选，并与现有文本筛选、列选择、导出联动
- 支持在运行对比页对批次表、归因表、差异表和完整对比表做列裁剪、排序与当前视图导出
- 支持在运行对比页对批次表、归因表、差异表和完整对比表按数值阈值做细筛选
- 支持让运行对比页的 HTML / PDF 导出同步当前页面四张表的显示列、排序和阈值筛选结果，并附带 view/full 两套 CSV
- 支持在页面中输出失败诊断摘要，帮助快速定位输入路径/依赖/阶段性失败
- 支持在运行前检查中自动定位缺失的 `pdb_path` / `pocket_file` / `catalytic_file` / `ligand_file`，生成路径修复建议，并可下载或保存修复版 `input_csv`
- 支持在页面中直接查看 `training_summary.json` 摘要和训练曲线
- 支持对 ranking / pose 结果做 Top-N 预览和关键 ID 过滤
- 支持查看运行产物清单，并一键打开最近运行目录或输出目录
- 支持把当前运行的关键结果、日志和 metadata 一键打包成 zip 汇总包
- 支持生成每个 nanobody 的候选报告卡，并在排名结果页打开索引或下载 zip
- 支持生成候选横向对比解释，在排名结果页查看“为什么 A 排在 B 前面”
- 支持生成下一轮实验建议，按高分低可信、Rule/ML 分歧和 QC 风险给出验证/复核优先级
- 页面当前默认通过 CLI 子进程方式执行推荐流程，便于支持后台运行、任务队列和停止任务
- `run_recommended_pipeline(...)` 仍然保留为可复用的 Python 入口，便于后续继续扩展
- 新增桌面启动器 [ml_desktop_launcher.py](ml_desktop_launcher.py) 与构建脚本 [build_desktop_app.bat](build_desktop_app.bat)
- 已可构建 `dist/ML_Local_App.exe`，双击后会拉起本地交互界面
- 新增便携目录打包脚本 [build_portable_bundle.py](build_portable_bundle.py) 与 [build_portable_bundle.bat](build_portable_bundle.bat)
- 已可构建 `portable_dist/ML_Portable/`，整个目录可直接拷走使用
- 新增单文件打包脚本 [build_standalone_onefile.py](build_standalone_onefile.py) 与 [build_standalone_onefile.bat](build_standalone_onefile.bat)
- 已可构建 `portable_dist/standalone_onefile/ML_Local_App_Standalone.exe`，以 onefile 自解压方式内嵌 `app/`
- 新增脱离源码目录的 standalone 校验脚本 [validate_standalone_onefile.py](validate_standalone_onefile.py)
- 新增发布打包脚本 [build_portable_release.py](build_portable_release.py) 与 [build_portable_release.bat](build_portable_release.bat)
- 已可构建 `portable_dist/ML_Portable_release.zip`，方便直接分发
- 新增统一版本元数据 [app_metadata.py](app_metadata.py)
- 当前页面标题、桌面启动器、自检输出、便携包和发布 manifest 已统一携带版本信息
- 新增品牌资产生成脚本 [generate_brand_assets.py](generate_brand_assets.py)
- 已生成 `assets/app_icon.png` 与 `assets/app_icon.ico`
- 当前 Streamlit 页面 favicon、桌面启动器窗口和桌面 exe 已统一接入应用图标
- 新增 GitHub Actions 工作流 [desktop-release.yml](.github/workflows/desktop-release.yml)
- 现在可通过手动触发或推送 `v*` tag 自动构建桌面版和便携发布包
- 新增 [export_structure_annotations.py](export_structure_annotations.py)
- 支持对单个复合物导出 residue/interface 的 viewer-friendly JSON/CSV
- 输出 `analysis_bundle.json` 作为后续展示层的基础数据契约
- 支持通过 `--key_residues` / `--key_residue_file` 标记用户关键残基
- 输出 `interface_pairs.csv` 与 `key_residues.csv` 便于做 viewer 联动
- 输出 `pocket_payload.json` 与 `blocking_summary.json`，便于直接做 pocket 面板和阻断解释

### 1.10 Rule + ML 共识排名

文件: build_consensus_ranking.py

已实现能力:
- 合并 `nanobody_rule_ranking.csv` 与 `nanobody_ranking.csv`
- 计算 Rule/ML 分数均值、排名一致性、分数一致性和来源覆盖度
- 可选读取 `pose_features.csv` 聚合失败行、warning 行和 pocket overwide 风险
- 输出 `consensus_score`、`confidence_score`、`confidence_level`、`decision_tier` 和中文解释
- 输出 `review_reason_flags` / `low_confidence_reasons`，细分 Rule/ML 分歧、pocket 偏宽、构象波动、close decision、失败行和 warning 行等原因
- 不替代原有 Rule 或 ML 排名，只作为更适合人工决策和展示的第三张汇总表

### 1.10.0 质量门控 PASS/WARN/FAIL

文件: build_quality_gate.py

已实现能力:
- 读取 `pose_features.csv` 和可选 `feature_qc.json`
- 输出统一 `overall_status = PASS / WARN / FAIL`
- `FAIL`：例如没有可用行或存在 failed 行，不建议直接解读排名
- `WARN`：例如 warning 行、全空列、pocket overwide 风险或 label 不足，需要先复核
- `PASS`：通过基础输入、QC、pocket shape 和 label 覆盖检查
- 输出 `quality_gate_summary.json`、`quality_gate_checks.csv` 和 `quality_gate_report.md`
- 推荐 pipeline、smoke test、本地软件 QC 页和结果归档均已接入
- 该步骤只做质量判定，不改变模型、规则或共识分数

单独运行示例：

```bash
python build_quality_gate.py --feature_csv pose_features.csv --feature_qc_json feature_qc.json --out_dir quality_gate --label_col label
```

### 1.10.1 分数解释卡片

文件: build_score_explanation_cards.py

已实现能力:
- 基于 `consensus_ranking.csv` 生成 `score_explanation_cards.csv/md/html/json`
- 把 `consensus_score`、`confidence_level`、Rule/ML 一致性、QC 风险、pocket overwide 风险和 label 覆盖情况翻译成可读说明
- 输出 `main_positive_factors`、`main_risk_factors`、`rule_ml_interpretation`、`label_context`、`recommended_action` 和 `plain_language_summary`
- 推荐 pipeline 和 smoke test 会自动生成该目录
- 本地软件“排名结果”页会预览、筛选和下载分数解释卡片，并可直接预览 HTML
- 不重新训练、不改变任何 Rule/ML/consensus 分数，只做面向用户的解释层

### 1.10.2 本批次结论摘要

文件: build_batch_decision_summary.py

已实现能力:
- 读取 `recommended_pipeline_summary.json`、`quality_gate`、`score_explanation_cards`、`consensus_ranking`、`experiment_suggestions`、`experiment_plan` 和 `validation_evidence_audit`
- 输出 `batch_decision_summary.json`、`batch_decision_summary.md` 和 `batch_decision_summary_cards.csv`
- 汇总本批次是否可解读、最高综合排名候选、当前证据最稳定候选、最需要复核候选和下一轮实验优先候选
- 汇总 Quality Gate WARN/FAIL、`warning_message` Top-N、`error_message` Top-N 和最高风险输入行
- 汇总真实验证证据状态、top-k 验证覆盖率、正负标签数量和待补行动项
- 推荐 pipeline、smoke test、本地软件摘要页和结果归档均已接入
- 不重新训练、不改变任何 Rule/ML/consensus 分数，只做批次级解释和阅读顺序收敛

单独运行示例：

```bash
python build_batch_decision_summary.py --summary_json my_outputs/recommended_pipeline_summary.json --out_dir my_outputs/batch_decision_summary
```

### 1.11 候选报告卡

文件: build_candidate_report_cards.py

已实现能力:
- 基于 `consensus_ranking.csv` 为每个 nanobody 生成单页 HTML 报告卡
- 报告卡汇总 consensus / ML / Rule 分数、可信度、风险标记、解释文本、pose top 行和 feature/QC 摘要
- 如果提供 `candidate_pairwise_comparisons.csv`，报告卡会嵌入该候选与相邻候选的胜出/落后关系、close decision 和解释文本
- 自动生成 `index.html`、`candidate_report_cards.csv`、`candidate_report_cards_summary.json`
- 自动打包 `candidate_report_cards.zip`，方便直接下载、发送或归档
- 不改变任何模型或规则分数，只做展示与人工复核层

### 1.12 候选横向对比解释

文件: build_candidate_comparisons.py

已实现能力:
- 基于 `consensus_ranking.csv` 生成 top 候选横向对比表
- 输出相邻候选之间“为什么 A 排在 B 前面”的解释
- 同时列出较低排名候选的反向优势，避免只给单向结论
- 标记 `is_close_decision`，提示分差很小、应一起人工复核的候选
- 自动按 `diversity_group`、`sequence_cluster`、`nanobody_family`、`epitope_cluster`、`experiment_status` 或风险分层生成候选分组小结
- 输出 `candidate_tradeoff_table.csv`、`candidate_pairwise_comparisons.csv`、`candidate_group_comparison_summary.csv`、`candidate_comparison_summary.json` 和 `candidate_comparison_report.md`
- 不重新训练、不改变排名，只把已有共识排名转成更适合答辩和人工决策的解释层

### 1.13 下一轮实验建议

文件: suggest_next_experiments.py

已实现能力:
- 基于 `consensus_ranking.csv` 生成主动学习式候选复核队列
- 综合高分低可信、Rule/ML 分歧、QC 风险和来源覆盖度，输出 `experiment_priority_score`
- 默认启用轻量 diversity-aware ordering，按显式 cluster/family 列或候选风险画像分组，避免建议队列过度集中在同一类候选
- 保留原始 `experiment_priority_score`，额外输出 `diversity_adjusted_priority_score`、`diversity_group`、`diversity_adjustment` 和 `diversity_note`
- 默认生成 `experiment_plan.csv/md`，把候选分成 `include_now`、`standby` 和 `defer`
- 支持实验预算、validate/review 分层 budget、standby 数量和每个 diversity group 的 quota
- 可选读取 `experiment_plan_override.csv`，手工锁定/排除/延后候选，并把实验状态、负责人、成本和备注写入计划单
- 默认输出 `experiment_plan_state_ledger.csv`，把本轮计划和状态整理成可继续编辑、可跨轮继承的 ledger
- 新增 `build_experiment_state_ledger.py`，可扫描 `local_app_runs` 汇总多个历史运行的实验状态，并优先采用最新编辑后的 override
- 给出 `suggestion_tier`、`suggestion_reason` 和 `recommended_next_action`
- 如果共识表中存在 `low_confidence_reasons`，实验建议会把细分复核原因带入 `suggestion_reason`
- 输出 `next_experiment_suggestions.csv`、`next_experiment_suggestions_summary.json` 和 `next_experiment_suggestions_report.md`
- 不替代实验判断，只用于安排下一轮验证、复核和补数据优先级

### 1.14 Pocket 方法共识分析

文件: compare_pocket_method_consensus.py

已实现能力:
- 比较多个 pocket residue list，例如 manual、ligand-contact、P2Rank、fpocket
- 输出共识 pocket、并集 pocket、方法特异 residue、逐残基方法支持和两两 overlap
- 可选接入 truth residue 文件，输出 truth coverage、precision、missing truth risk 和 overwide risk
- 无 truth 时也会给出基于 union-vs-consensus 扩张和方法大小离散度的 overwide 风险 proxy
- 该脚本不运行外部 pocket 工具，只比较已经转换成 residue list 的结果

### 1.15 排名参数敏感性分析

文件: analyze_ranking_parameter_sensitivity.py

已实现能力:
- 基于已有 `consensus_ranking.csv` 做轻量 post-processing，不重跑结构特征、不重新训练模型
- 扫描 Rule/ML 权重、rank-agreement 权重和 QC risk penalty 权重
- 输出每个场景下的 ranking、Top-N overlap、rank Spearman 和候选 rank span
- 标记 `top_n_unstable`、`is_sensitive` 和 `sensitivity_reason`
- 推荐 pipeline 和 smoke test 会自动生成 `parameter_sensitivity/` 目录
- 本地软件“排名结果”页已支持预览和下载参数敏感性表

### 1.16 CD38 pocket 参数敏感性分析

文件: analyze_cd38_pocket_parameter_sensitivity.py

已实现能力:
- 基于现有 CD38 benchmark 本地文件做结构侧敏感性分析，不运行外部 pocket finder
- 扫描 ligand-contact distance cutoff，例如 `3.5 / 4.0 / 4.5 / 5.0 / 5.5 A`
- 扫描 P2Rank `source_predictions.csv` 中不同 rank pocket 的 truth coverage 和 precision
- 读取已接入 manifest 的 fpocket pocket 结果，输出 fpocket pocket choice 的 coverage / precision 表
- 扫描 method consensus 的 `min_method_count`，比较 union-style 与 strict consensus-style pocket
- 对 overwide penalty weight 做轻量 adjusted utility 对比
- 输出 `benchmarks/cd38/parameter_sensitivity/`，便于解释 pocket 定义是否对参数敏感

### 1.17 CD38 fpocket 批量接入入口

文件: prepare_cd38_fpocket_panel.py

已实现能力:
- 扫描一批外部 `fpocket` 输出目录中的 `pocket*_atm.pdb`
- 自动生成只包含 fpocket rows 的 CD38 benchmark manifest
- 自动生成 `*.report.md` readiness report，先检查目录、跳过原因和下一步命令
- 支持 `--run` 直接调用 `run_cd38_benchmark_manifest.py` 批量加入 panel
- 支持 `--dry_run --run` 先检查将要执行的命令，不实际跑 benchmark

### 1.18 CD38 P2Rank 批量接入入口

文件: prepare_cd38_p2rank_panel.py

已实现能力:
- 扫描一批外部 `P2Rank` 输出目录中的 `*_predictions.csv`
- 校验是否包含 `rank` 和 `residue_ids` 必需列
- 自动生成只包含 P2Rank rows 的 CD38 benchmark manifest
- 自动生成 `*.report.md` readiness report，先检查目录、跳过原因和下一步命令
- 支持 `--rank_by_pdb 3ROP=2,4OGW=1` 处理同源链或 rank 选择问题
- 支持 `--run` 直接调用 `run_cd38_benchmark_manifest.py` 批量加入 panel

### 1.19 CD38 外部工具输入包生成器

文件: prepare_cd38_external_tool_inputs.py

已实现能力:
- 读取 `benchmarks/cd38/cd38_structure_targets.csv`
- 自动下载或复用 `3ROP`、`4OGW`、`3F6Y` 等目标 PDB
- 生成给外部 P2Rank / fpocket 使用的 PDB 输入副本
- 生成可直接改用的 PowerShell 和 Bash 命令模板
- 生成环境/输出预检 PowerShell 入口
- 生成外部工具跑完后的 readiness 刷新脚本
- 生成 expected return manifest 和 return checklist，明确外部机器跑完后应该带回哪些 `PDB × method` 输出
- 生成输入 manifest、summary JSON 和 Markdown 操作说明

### 1.20 CD38 外部工具环境和输出预检

文件: check_cd38_external_tool_environment.py

已实现能力:
- 检查本机是否能在 PATH 中找到 P2Rank 的 `prank` 命令和 `fpocket` 命令
- 检查 `external_tool_inputs/` 中每个结构的 PDB 输入是否存在
- 检查每个结构预期的 P2Rank `*_predictions.csv` 是否存在
- 检查每个结构预期的 fpocket `pocket*_atm.pdb` 是否存在
- preflight 会优先解析当前 package 内的 portable 路径，避免 transfer zip 移动到另一台机器后仍被旧绝对路径误导
- 输出逐结构 CSV、summary JSON 和 Markdown 报告
- 可从 `external_tool_inputs/check_external_tool_environment.ps1` 直接调用

### 1.21 CD38 外部 benchmark finalize 入口

文件: finalize_cd38_external_benchmark.py

已实现能力:
- 一条命令串联外部工具 preflight 和 CD38 benchmark readiness
- 支持 `--import_source <returned_zip_or_dir>` 先安全导入返回的 P2Rank/fpocket 输出，再继续 preflight/readiness
- 默认不导入 benchmark，只生成最终检查报告，避免外部输出未准备好时误运行
- 支持 `--run_discovered` 在 readiness manifest 正确后导入真实 P2Rank/fpocket rows
- 如果 readiness 没有发现可运行 rows，会跳过 benchmark panel 汇总和参数敏感性刷新，避免把旧结果误当成新导入结果
- finalize 报告会链接 import repair plan，直接列出仍缺的 `PDB × method` 输出和应运行的模板
- finalize 现在会自动刷新外部 benchmark action plan，把 `3ROP/4OGW/3F6Y × P2Rank/fpocket` 的缺口、优先级、返回路径和验证命令合并到一份清单
- 支持 `--run_sensitivity` 在导入后刷新 CD38 pocket 参数敏感性表，并同步生成 CD38 proxy 校准报告
- 可从 `external_tool_inputs/finalize_external_benchmark.ps1` 直接调用

### 1.22 CD38 外部工具 transfer zip 打包

文件: package_cd38_external_tool_inputs.py

已实现能力:
- 把 `external_tool_inputs/` 打包成可传到 Linux/WSL 或另一台机器运行的 zip
- 打包前自动刷新 `cd38_external_tool_next_run.*`，并把 next-run 说明、CSV、PowerShell 和 Bash 脚本一起放入 zip
- 默认只包含 PDB 输入、PowerShell/Bash 模板、next-run runbook 和说明文件
- 默认排除 `preflight/`、`finalize/`、旧 `p2rank_outputs/` 和 fpocket `*_out/` 结果目录
- 额外生成 transfer manifest、summary JSON 和 Markdown 报告
- 支持 `--include_existing_outputs` 在需要时把已有外部输出一起打包

### 1.23 CD38 外部工具返回结果导入

文件: import_cd38_external_tool_outputs.py

已实现能力:
- 从返回的目录或 zip 中安全提取外部 P2Rank/fpocket 输出
- 只导入 `p2rank_outputs/` 和 `fpocket_runs/*/*_out/`
- 默认不覆盖已有文件，支持 `--overwrite`
- 支持 `--dry_run` 先检查会导入哪些文件
- 支持返回 zip/目录外面多包一层目录，只要内部能找到 `p2rank_outputs/` 或 `fpocket_runs/`
- 额外输出 scan manifest，列出扫描了哪些文件、哪些被忽略以及忽略原因
- 在 summary/report 中输出 `source_diagnosis` 和 `diagnostic_message`，可直接判断是不是把原始输入包当成返回输出包导入
- 额外输出 coverage manifest，按 package manifest 检查 `PDB × method` 是否已在返回包中覆盖
- 额外输出 repair plan CSV，把缺失的 `PDB × method` 转成应运行的模板、应返回的相对路径和下一条 dry-run 验证命令
- 输出导入 manifest、summary JSON 和 Markdown 报告

### 1.24 CD38 benchmark readiness 一键刷新

文件: refresh_cd38_benchmark_readiness.py

已实现能力:
- 一条命令刷新 ligand candidate 检查和 CD38 扩展计划
- 可选扫描外部 P2Rank / fpocket 输出目录
- 汇总生成 readiness summary JSON、commands JSON 和 Markdown 总览
- 如果发现可运行的 P2Rank/fpocket manifest row，会提示审核 readiness report 后再 `--run_discovered`
- 如果还没有外部 P2Rank/fpocket 输出，会提示先运行 `prepare_cd38_external_tool_inputs.py`，再运行环境/输出预检
- 默认不运行外部工具、不重跑 benchmark，只做准备状态刷新

### 1.25 CD38 benchmark 扩展计划

文件: build_cd38_benchmark_expansion_plan.py

已实现能力:
- 读取 `benchmarks/cd38/cd38_structure_targets.csv`
- 对照当前 `cd38_benchmark_manifest.csv` 和 `results/cd38_benchmark_panel.csv`
- 按结构和方法输出 `complete`、`manifest_ready`、`needs_ligand_metadata`、`needs_p2rank_output`、`needs_fpocket_output` 等状态
- 自动生成下一步命令，尤其是把真实 fpocket / P2Rank 输出接入对应准备脚本
- 当前目标表包含 `3ROP`、`4OGW` 和 apo-like 测试结构 `3F6Y`

### 1.26 CD38 ligand candidate 检查

文件: inspect_cd38_ligand_candidates.py

已实现能力:
- 自动下载或复用目标结构 PDB
- 扫描 HETATM residues，区分 ligand-like、water/buffer、ion、glycan
- 根据 CD38 truth residues 距离判断哪些 HETATM 适合作为 ligand-contact baseline
- 输出推荐 ligand candidates、建议版 target CSV 和 Markdown 报告
- 当前确认 `3ROP` 的 `50A/NCA`、`4OGW` 的 `NMN` 可作为 ligand-contact baseline；`3F6Y` 没有活性口袋 ligand candidate，因此不再做 ligand-contact baseline

### 1.27 CD38 几何 proxy 校准报告

文件: build_cd38_proxy_calibration_report.py

已实现能力:
- 读取 `benchmarks/cd38/results/cd38_benchmark_panel.csv` 和 `benchmarks/cd38/parameter_sensitivity/`
- 重新计算每个 benchmark pocket 的运行时 `pocket_shape_overwide_proxy`
- 对照 truth-based `overwide_pocket_score`、coverage 和 precision，输出阈值候选表
- 输出是否建议改变默认 `pocket_overwide_penalty_weight` 的证据等级和阻塞项
- 当前 4 组 CD38 baseline 的结论是：proxy 能识别 `4OGW + P2Rank` 偏宽问题，但结构数、方法数和真实 fpocket 行仍不足，默认惩罚权重继续保持 `0.0`

单独运行示例：

```bash
python build_cd38_proxy_calibration_report.py
```

### 1.28 CD38 外部 benchmark action plan

文件: build_cd38_external_benchmark_action_plan.py

已实现能力:
- 读取 `expansion_plan`、`preflight`、`readiness`、`expected returns` 和导入 repair plan
- 输出一张按优先级排序的 `PDB × method` 执行清单
- 区分直接影响 benchmark 完整性的 `benchmark_completion` 行，以及只影响外部包复现性的 `package_reproducibility` 行
- 当前会把 `3ROP/4OGW` 的真实 fpocket 输出列为 priority `1`，把 `3F6Y` 的 P2Rank/fpocket 输出列为 priority `2`
- 同时保留 `3ROP/4OGW` 的 P2Rank 外部复现实验项，但不会把它们误判成当前 benchmark 必须补齐的 blocker
- `finalize_cd38_external_benchmark.py` 会自动刷新这份 action plan

单独运行示例：

```bash
python build_cd38_external_benchmark_action_plan.py
```

默认输出：

- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md`
- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.csv`
- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.json`

### 1.28.1 CD38 外部工具 next-run runbook

文件: build_cd38_external_tool_runbook.py

已实现能力:
- 读取 `cd38_external_benchmark_action_plan.csv`
- 默认只选择仍缺失的 `benchmark_completion` 行，不重复跑只用于外部包复现的 priority `3` 行
- 输出外部机器可直接执行的 `run_cd38_external_next_benchmark.ps1` 和 `run_cd38_external_next_benchmark.sh`
- 输出 `cd38_external_tool_next_run.md`、`cd38_external_tool_next_run_plan.csv` 和 summary JSON，方便先人工检查将要运行的结构/方法
- 支持 `--include_package_reproducibility`，需要时把 `3ROP/4OGW P2Rank` 外部包复现动作也纳入脚本
- 已接入 `run_cd38_public_starter.py` 和 `package_cd38_external_tool_inputs.py`

单独运行示例：

```bash
python build_cd38_external_tool_runbook.py
```

当前默认 next-run 动作：

- `3ROP fpocket`
- `4OGW fpocket`
- `3F6Y fpocket`
- `3F6Y P2Rank`

### 1.29 CD38 公开结构 starter 一键入口

文件: run_cd38_public_starter.py

已实现能力:
- 不运行外部 pocket finder，仅刷新本地已有公开 CD38 结构 benchmark 产物
- 串联 panel 汇总、ligand candidate scan、参数敏感性、proxy calibration、外部工具输入包/preflight、readiness、action plan 和 next-run runbook
- 自动扫描当前仓库中已有的 P2Rank `source_predictions.csv`，默认使用 `3ROP=2,4OGW=1` 的 active-site rank 映射
- 输出 `benchmarks/cd38/public_starter/cd38_public_starter_report.md` 和 summary JSON，适合作为公开结构 benchmark 的第一入口
- 本地软件“诊断”页已接入同一入口，可一键刷新、打开报告、下载 action plan CSV、expected returns CSV、返回检查清单和 next-run 文件
- 本地软件“诊断”页还可以生成 CD38 外部工具 transfer zip；返回 zip/目录带回后，可先 dry-run 检查，再导入并 finalize

最短命令:

```bash
python run_cd38_public_starter.py
```

当前本地验证结果：公开 CD38 panel 汇总为 4 行，其中 `ligand_contact=2`、`p2rank=2`；starter 9/9 子步骤通过；action plan 仍显示真实外部 fpocket/P2Rank 输出不足，因此默认不改变 pocket overwide penalty。

如果要把 CD38 外部工具输入包传到 WSL/Linux 或另一台机器，在本地软件“诊断”页点击“生成 transfer zip”，或运行：

```bash
python package_cd38_external_tool_inputs.py
```

当前 transfer zip 会输出到：

```text
benchmarks/cd38/external_tool_transfer/cd38_external_tool_inputs_transfer.zip
```

外部机器跑完 P2Rank/fpocket 后，把返回 zip 或目录路径填回“诊断”页，先点“Dry-run 检查返回包”。如果 dry-run 显示 `input_package_without_external_outputs`，说明带回来的还是原始输入包，不是跑完后的返回包。

如果想先确认“返回包导入路径规则”本身没有问题，可在“诊断”页展开“返回包导入流程自测”，或运行：

```bash
python selftest_cd38_return_import_workflow.py
```

这个自测会生成一个合成返回包并 dry-run 导入，目标是确认 importer 能识别 `p2rank_outputs/` 和 `fpocket_runs/*/*_out/`，并让 expected coverage 达到 `6/6`。它只用于路径/coverage 回归测试，不是真实 P2Rank/fpocket benchmark 结果。

返回包 dry-run 或导入后，也可以生成安全门控报告：

```bash
python build_cd38_return_package_gate.py
```

门控会把当前返回包判断为 `PASS_READY_FOR_IMPORT`、`WARN_PARTIAL_RETURN`、`FAIL_INPUT_PACKAGE` 或 `FAIL_SYNTHETIC_FIXTURE` 等状态。只有 `PASS_*` 状态才适合继续导入或 finalize；如果是 synthetic fixture，会被明确拦截，避免把自测数据误当成 benchmark 证据。

本地软件里的“导入返回包并 finalize”按钮已经内置导入前 gate：它会先在隔离目录做一次 dry-run，再生成 gate；如果 gate 不是 `PASS_*`，软件会直接阻止正式导入。命令行 `finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir>` 现在也默认启用同样的导入前 gate；只有你已经人工复核且明确要绕过时，才使用 `--skip_import_gate`。如果用于自动化/CI，希望 gate 不通过时进程直接失败，可加 `--strict_import_gate`。

如果要一次性检查 CD38 外部工具链路是否仍然完整，可运行：

```bash
python selftest_cd38_external_workflow.py
```

它会验证 transfer zip 生成、原始 transfer zip 的 strict gate 拦截、synthetic returned fixture 的 importer/gate 行为，以及 public starter 刷新。本地软件“诊断”页也提供“CD38 外部工具链路一键自检”按钮。该自检不运行 P2Rank/fpocket，也不产生真实 benchmark 证据。

### 1.30 几何 proxy 一致性审计

文件: build_geometry_proxy_audit.py

已实现能力:
- 读取已有 `pose_features.csv`，不重建结构特征、不训练模型、不改变 Rule/ML/Consensus 分数
- 检查 `mouth_occlusion_score`、`ligand_path_block_score`、`pocket_hit_fraction`、`catalytic_hit_fraction`、`pocket_shape_overwide_proxy` 等静态 proxy 是否互相矛盾
- 标记常见风险，例如“mouth 阻断高但 ligand path 仍开放”“path 阻断高但 pocket/catalytic contact 低”“pocket 定义偏宽”
- 输出 pose 级 flagged rows、nanobody 级汇总、特征覆盖统计和 Markdown 审计报告
- 推荐 pipeline 和 smoke test 已自动运行该审计；本地软件 QC 面板会展示审计状态和下载入口

单独运行示例：

```bash
python build_geometry_proxy_audit.py --feature_csv pose_features.csv --out_dir geometry_proxy_audit
```

默认输出：

- `geometry_proxy_audit/geometry_proxy_audit_summary.json`
- `geometry_proxy_audit/geometry_proxy_audit_report.md`
- `geometry_proxy_audit/geometry_proxy_feature_summary.csv`
- `geometry_proxy_audit/geometry_proxy_flagged_poses.csv`
- `geometry_proxy_audit/geometry_proxy_candidate_audit.csv`

### 1.30 真实验证证据审计

文件: build_validation_evidence_audit.py

已实现能力:
- 读取已有 `consensus_ranking.csv`、`experiment_plan.csv` 和 `experiment_plan_state_ledger.csv`，不训练模型、不改变 Rule/ML/Consensus 分数
- 检查当前 top-k 候选是否已经有明确 `validation_label` 或可解释的 `experiment_result`
- 明确区分 `planned`、`completed_needs_result`、`blocked_or_cancelled`、`in_progress`、`validated` 等证据状态
- 只把明确 `validation_label=1/0` 或 `experiment_result=positive/negative` 视为真实验证证据；`completed` 本身不会被误当成阳性或阴性
- 输出 top-k 验证覆盖率、正负标签数量、是否具备 compare/calibration 条件、待补行动清单和 Markdown 报告
- 推荐 pipeline 和 smoke test 已自动运行该审计；本地软件“排名结果”页会展示审计状态、top-k 表和行动清单

单独运行示例：

```bash
python build_validation_evidence_audit.py --consensus_csv consensus_outputs/consensus_ranking.csv --experiment_plan_csv experiment_suggestions/experiment_plan.csv --experiment_plan_state_ledger_csv experiment_suggestions/experiment_plan_state_ledger.csv --out_dir validation_evidence_audit
```

默认输出：

- `validation_evidence_audit/validation_evidence_summary.json`
- `validation_evidence_audit/validation_evidence_report.md`
- `validation_evidence_audit/validation_evidence_by_candidate.csv`
- `validation_evidence_audit/validation_evidence_topk.csv`
- `validation_evidence_audit/validation_evidence_action_items.csv`

---

## 2. 端到端数据流

推荐流程:

1. 输入 pose 基表
- 至少包含: nanobody_id, conformer_id, pose_id, pdb_path

2. 生成结构几何特征
- 运行 build_feature_table.py
- 输出 pose_features.csv + feature_qc.json

3. 分两条路线验证
- 规则路线: 运行 rule_ranker.py
- ML 路线:
- 运行 train_pose_model.py 得到 pose_predictions.csv
- 再运行 rank_nanobodies.py 得到最终排序
- 共识路线: 运行 build_consensus_ranking.py 合并 Rule / ML / QC 风险，得到 consensus_ranking.csv

---

## 3. 核心评分公式

### 3.1 规则版 pose 分数

对每个可用特征 f:
- 正向特征: aligned_f = robust_norm(f)
- 负向特征: aligned_f = 1 - robust_norm(f)

rule_blocking_score:

score = sum(w_f * aligned_f) / sum(w_f over finite aligned_f)

### 3.2 conformer 聚合（规则版与 ML 排名器风格一致）

conformer_score = (1 - w_geo) * (0.7 * topk_mean + 0.3 * top1_best) + w_geo * geo_aux

说明:
- topk_mean 来自 top-k pose
- top1_best 为最佳 pose 分数
- geo_aux 为 pocket/catalytic/mouth/substrate 等几何辅助信号

### 3.3 nanobody 最终分数

final_score = w1 * mean_conformer_score + w2 * best_conformer_score + w3 * pocket_consistency_score - w4 * std_conformer_score

说明:
- 奖励整体均值与最佳构象
- 奖励跨构象口袋命中一致性
- 惩罚跨构象波动

### 3.4 Rule + ML 共识分数

consensus_score = 0.70 * mean(ML_score, Rule_score) + 0.20 * rank_agreement + 0.10 * score_alignment - 0.15 * qc_risk

confidence_score = 0.45 * rank_agreement + 0.25 * score_alignment + 0.20 * source_coverage + 0.10 * (1 - qc_risk)

说明:
- `rank_agreement` 衡量 Rule 与 ML 排名是否接近。
- `score_alignment` 衡量 Rule 与 ML 分数是否接近。
- `source_coverage` 区分两套排名是否都覆盖该候选。
- `qc_risk` 汇总失败行、warning 行和 pocket overwide 风险。
- `low_confidence_reasons` 不是新的分数，而是把低可信来源拆成可读原因，方便人工复核。

---

## 4. CLI 用法

### 4.1 构建特征表

python build_feature_table.py --input_csv input_pose_table.csv --out_csv pose_features.csv

常用参数:
- --atom_contact_threshold
- --catalytic_contact_threshold
- --substrate_clash_threshold
- --mouth_residue_fraction
- --default_pocket_file
- --default_catalytic_file
- --default_ligand_file
- --default_antigen_chain
- --default_nanobody_chain
- --skip_failed_rows
- --qc_json

### 4.2 规则版排名

python rule_ranker.py --feature_csv pose_features.csv --out_dir rule_outputs

常用参数:
- --top_k
- --lower_q / --upper_q
- --conformer_geo_weight
- --pocket_overwide_penalty_weight / --pocket_overwide_threshold
- --w_mean / --w_best / --w_consistency / --w_std_penalty
- --consistency_hit_threshold

### 4.3 训练与预测

python train_pose_model.py --feature_csv pose_features.csv --out_dir model_outputs

常用参数:
- --epochs --batch_size --lr --weight_decay
- --soft_target_weight
- --pseudo_threshold_mode (top_fraction/quantile/fixed)
- --pseudo_threshold_value
- --feature_min_non_nan_ratio
- --feature_max_missing_ratio
- --feature_near_constant_ratio
- --early_stopping_patience --min_delta
- --grad_clip_norm
- --lr_scheduler_patience --lr_scheduler_factor

### 4.4 ML 结果聚合排名

python rank_nanobodies.py --pred_csv model_outputs/pose_predictions.csv --out_dir ranking_outputs

常用参数:
- --top_k
- --optional_weight
- --disable_optional_blend
- --pocket_overwide_penalty_weight / --pocket_overwide_threshold
- --alpha
- --w_mean --w_best --w_consistency --w_std_penalty
- --consistency_hit_threshold

### 4.5 规则权重校准（建议在有 label 时执行）

python calibrate_rule_ranker.py --feature_csv pose_features.csv --label_col label --out_dir calibration_outputs

常用参数:
- --n_feature_trials
- --top_feature_trials_for_agg
- --feature_jitter_sigma
- --w_mean_grid --w_best_grid --w_consistency_grid --w_std_penalty_grid
- --pose_objective_weight --nanobody_objective_weight
- --ml_ranking_csv --ml_score_col
- --rank_consistency_metric (score_spearman/rank_spearman)
- --rank_consistency_weight (默认 0.40；无 --ml_ranking_csv 时自动按 0.0 生效)
- --selection_metric (objective/nanobody_auc/rank_consistency)
- --pocket_overwide_penalty_weight / --pocket_overwide_threshold
- --min_nanobody_auc --min_rank_consistency (可选约束)

### 4.6 规则版 vs ML 版对照评估

python compare_rule_ml_rankings.py --rule_csv rule_outputs/nanobody_rule_ranking.csv --ml_csv ranking_outputs/nanobody_ranking.csv --out_dir comparison_outputs --pose_feature_csv pose_features.csv --label_col label

常用参数:
- --topk_list
- --pose_feature_csv
- --label_col

### 4.7 Rule + ML 共识排名

python build_consensus_ranking.py --rule_csv rule_outputs/nanobody_rule_ranking.csv --ml_csv ranking_outputs/nanobody_ranking.csv --feature_csv pose_features.csv --out_dir consensus_outputs

输出:
- consensus_ranking.csv
- consensus_summary.json
- consensus_report.md

说明:
- `consensus_score` 侧重候选综合优先级。
- `confidence_score` / `confidence_level` 用于提示 Rule 与 ML 是否一致、QC 风险是否较高。
- `decision_tier` 会给出 `priority` / `review` / `standard`，便于人工复核。
- `review_reason_flags` / `low_confidence_reasons` 会拆出低可信原因，例如 Rule/ML 排名差、分数差、pocket overwide、构象波动、相邻候选分差过近、失败行和 warning 行。

### 4.8 候选报告卡

python build_candidate_report_cards.py --consensus_csv consensus_outputs/consensus_ranking.csv --rule_csv rule_outputs/nanobody_rule_ranking.csv --ml_csv ranking_outputs/nanobody_ranking.csv --feature_csv pose_features.csv --pose_predictions_csv model_outputs/pose_predictions.csv --candidate_pairwise_csv candidate_comparisons/candidate_pairwise_comparisons.csv --out_dir candidate_report_cards

输出:
- candidate_report_cards/index.html
- candidate_report_cards/cards/*.html
- candidate_report_cards/candidate_report_cards.csv
- candidate_report_cards/candidate_report_cards_summary.json
- candidate_report_cards.zip

说明:
- 打开 `index.html` 后可点击每个 nanobody 的单独报告卡。
- 如果先生成了 `candidate_pairwise_comparisons.csv`，每张报告卡会内嵌该候选与相邻候选的对比解释。
- 如需 PDF，可用浏览器打开单个报告卡后打印为 PDF。
- 该步骤只汇总已有输出，不重新训练、不改变 ranking 分数。

### 4.8.1 候选横向对比解释

python build_candidate_comparisons.py --consensus_csv consensus_outputs/consensus_ranking.csv --out_dir candidate_comparisons --top_n 12 --pair_mode adjacent

输出:
- candidate_tradeoff_table.csv
- candidate_pairwise_comparisons.csv
- candidate_group_comparison_summary.csv
- candidate_comparison_summary.json
- candidate_comparison_report.md

说明:
- 该步骤回答“为什么 A 排在 B 前面”，适合候选筛选讨论和答辩展示。
- `winner_key_advantages` 展示高排名候选的主要优势。
- `runner_up_counterpoints` 展示低排名候选的反向优势，避免只给单向解释。
- `is_close_decision=True` 表示两者共识分差很小，建议一起人工复核。
- `candidate_group_comparison_summary.csv` 会按 `--group_col auto` 自动选择 diversity/family/status 类字段，汇总每组共同优势、共同风险和最高排名候选。
- 如需手工挑选候选，可用 `--selected_nanobody_ids NB001,NB003,NB007 --pair_mode all` 生成 2 到 5 个指定候选之间的自定义对比。
- 本地软件“排名结果 -> 候选对比解释”区域也支持手工选择 2 到 5 个候选，预览、下载并保存自定义对比结果。
- 推荐 pipeline 会自动输出到 `my_outputs/candidate_comparisons/`。

### 4.9 下一轮实验建议

python suggest_next_experiments.py --consensus_csv consensus_outputs/consensus_ranking.csv --out_dir experiment_suggestions

输出:
- next_experiment_suggestions.csv
- next_experiment_suggestions_summary.json
- next_experiment_suggestions_report.md
- experiment_plan.csv
- experiment_plan_summary.json
- experiment_plan.md
- experiment_plan_state_ledger.csv

说明:
- `experiment_priority_score` 综合高分潜力、低可信、Rule/ML 分歧、QC 风险和来源缺失。
- `diversity_adjusted_priority_score` 只用于建议队列排序，不改变原始分数；如果有 `sequence_cluster`、`nanobody_family`、`epitope_cluster` 等列，会优先作为 diversity group，否则使用候选风险画像分组。
- `experiment_plan.csv` 会在建议队列上应用预算和 quota，给出 `include_now` / `standby` / `defer`。
- 如果提供 `--experiment_plan_override_csv experiment_plan_override.csv`，可用 `plan_override` 手工指定 `include` / `exclude` / `standby` / `defer`，也可用 `experiment_status` 回灌 `completed` / `blocked` / `in_progress` 等状态。
- 覆盖 CSV 推荐列：`nanobody_id,plan_override,experiment_status,experiment_result,validation_label,experiment_owner,experiment_cost,experiment_note`。
- `experiment_plan_state_ledger.csv` 是自动生成的状态账本；在本地软件中编辑保存后，可直接作为下一轮 `experiment_plan_override_csv`。
- `suggestion_tier` 会给出 `validate_now` / `review_first` / `backup`。
- 该步骤只生成下一轮验证建议，不重新训练、不改变 ranking 分数。

### 4.9.1 汇总跨批次实验状态

如果已经用本地软件跑过多次，并在计划单里编辑过实验状态，可以汇总所有历史状态：

```bash
python build_experiment_state_ledger.py --local_app_runs local_app_runs
```

输出：
- `local_app_runs/experiment_state_ledger_global.csv`
- `local_app_runs/experiment_state_ledger_global_history.csv`
- `local_app_runs/experiment_state_ledger_global_summary.json`

说明：
- `experiment_state_ledger_global.csv` 每个 `nanobody_id` 保留最新状态，可直接作为下一轮 `--experiment_plan_override_csv`。
- 如果同一轮同时存在自动 ledger 和用户编辑后的 override，优先采用用户编辑后的 override。
- 本地软件“历史”页也提供“跨批次实验状态汇总”，可一键生成并设为下一轮 override。
- 本地软件的全局 ledger 面板支持按 `experiment_status`、`experiment_result`、`plan_override` 和关键词筛选，并显示筛选后的状态分布图与可回灌标签数。

### 4.9.2 真实验证结果回灌报告

如果全局 ledger 中已经写入了 `experiment_result` 或 `validation_label`，可以生成验证标签表和审计报告：

```bash
python build_experiment_validation_report.py ^
  --ledger_csv local_app_runs/experiment_state_ledger_global.csv ^
  --out_dir local_app_runs/experiment_validation_feedback
```

可选：如果想把标签合并回特征表，额外传入 `--feature_csv pose_features.csv`：

```bash
python build_experiment_validation_report.py ^
  --ledger_csv local_app_runs/experiment_state_ledger_global.csv ^
  --feature_csv pose_features.csv ^
  --label_col experiment_label ^
  --out_dir local_app_runs/experiment_validation_feedback
```

输出：
- `experiment_validation_status_report.csv`
- `experiment_validation_labels.csv`
- `experiment_validation_summary.json`
- `experiment_validation_report.md`
- `pose_features_with_experiment_labels.csv`，仅在传入 `--feature_csv` 时生成

安全规则：
- `completed` 本身不会被当成阳性或阴性。
- 只有明确的 `validation_label` 或可解释的 `experiment_result=positive/negative` 才会进入训练标签。
- `blocked` / `cancelled` 只表示无法实验，不会被误当成阴性标签。
- 本地软件“历史 -> 跨批次实验状态汇总”中也可以一键生成该验证回灌报告；填写 `pose_features.csv` 路径后，会同时生成 `pose_features_with_experiment_labels.csv`。
- 本地软件可把 `pose_features_with_experiment_labels.csv` 一键设为下一轮输入，并自动把左侧 `label_col` 切换为 `experiment_label`；这不会覆盖原始 `label` 列。

如果要在命令行中直接使用验证回灌后的标签重新运行：

```bash
python run_recommended_pipeline.py ^
  --feature_csv local_app_runs/experiment_validation_feedback/pose_features_with_experiment_labels.csv ^
  --label_col experiment_label ^
  --out_dir recommended_pipeline_validation_retrain
```

### 4.9.3 验证回灌再训练前后对照

如果已经分别跑完“回灌前”和“使用 `experiment_label` 后”的两次推荐 pipeline，可以自动生成对照报告：

```bash
python build_validation_retrain_comparison.py ^
  --before_summary local_app_runs/<before_run>/outputs ^
  --after_summary local_app_runs/<after_run>/outputs ^
  --validation_labels_csv local_app_runs/experiment_validation_feedback/experiment_validation_labels.csv ^
  --top_k 10 ^
  --out_dir local_app_runs/validation_retrain_comparisons/<before_run>__vs__<after_run>
```

输出：
- `validation_retrain_metric_comparison.csv`
- `validation_retrain_candidate_rank_delta.csv`
- `validation_retrain_comparison_summary.json`
- `validation_retrain_comparison_report.md`

本地软件也可以生成：进入“历史 -> 多运行对比”，先选择回灌前和回灌后两次运行，再打开“验证回灌再训练前后对照报告”生成和下载。

### 4.9.4 结果自动归档与长期趋势

如果本地已经积累了多次运行和多次验证回灌对照，可以生成统一归档索引：

```bash
python build_result_archive.py ^
  --local_app_runs local_app_runs ^
  --out_dir local_app_runs/result_archive
```

输出：
- `result_archive_runs.csv`，每次运行的状态、标签列、标签数、Rule/ML 对照指标、训练摘要和 QC 摘要
- `result_archive_artifact_manifest.csv`，每次运行关键产物是否存在、路径、大小和修改时间
- `result_archive_validation_retrain_trends.csv`，所有验证回灌再训练前后对照的长期趋势
- `result_archive_lineage.csv`，按 provenance hash 汇总跨批次 lineage，例如共享输入文件 manifest、共享 feature CSV、共享参数 hash 的运行组
- `result_archive_lineage_graph.json/html/md`，把共享输入、共享特征表和共享参数的复跑关系做成可下载的图形化时间线
- `result_archive_summary.json`
- `result_archive_report.md`

本地软件入口：进入“历史 -> 结果自动归档与长期趋势”，点击“生成/刷新结果归档索引”。页面会显示 lineage 图边数、共享 lineage 图组，并可预览/下载图形化 HTML。

### 4.9.5 input_csv 缺失路径自动定位

如果 `input_csv` 里有 `pdb_path`、`pocket_file`、`catalytic_file` 或 `ligand_file` 指向了旧目录，可以先生成修复建议：

```bash
python input_path_repair.py ^
  --input_csv input_pose_table.csv ^
  --search_root . ^
  --out_dir input_path_repair_outputs ^
  --write_repaired_csv
```

输出：
- `input_path_repair_plan.csv`，逐行列出缺失路径、候选文件、可信度和建议动作
- `input_path_repair_summary.json`
- `input_path_repair_report.md`
- `*_repaired.csv`，只替换高可信同名文件匹配，低可信候选仍保留人工确认

本地软件入口：使用本地路径或 zip/目录导入 `input_csv` 后，点击“检查当前输入”。如果能在 CSV 所在目录树下找到缺失文件，页面会显示“缺失路径自动定位建议”，并提供修复建议 CSV 和自动修复版 `input_csv` 下载；也可以点击“保存并使用自动修复版 input_csv”直接回填左侧路径。

### 4.10 一键端到端 smoke test（无真实数据时推荐）

python pipeline_smoke_test.py --out_dir smoke_test_outputs

说明:
- 若存在 `smoke_test_outputs/strategy_optimization/recommended_strategy.json`，会在校准阶段自动回灌推荐参数（`rank_consistency_weight` / `selection_metric` / 约束阈值）。
- 可通过参数显式指定其他推荐文件，或禁用自动回灌。
- 执行后建议先看 `smoke_test_report.md`，再需要时下钻到 `smoke_test_summary.json`。

常用参数:
- --seed
- --n_nanobodies --n_conformers --n_poses
- --train_epochs
- --n_feature_trials --top_feature_trials_for_agg
- --calibration_rank_consistency_weight (默认 0.40)
- --calibration_rank_consistency_metric
- --calibration_selection_metric
- --disable_calibration_baseline_guard
- --calibration_rank_guard_tolerance
- --calibration_auc_guard_tolerance
- --strategy_seed_json
- --disable_auto_seed_from_previous_strategy

### 4.10.1 一键 demo 演示流程（没有真实数据时最推荐）

如果你只是想确认软件能完整跑通，并快速看到排名、报告卡、质量门控、验证证据审计和批次结论摘要，优先运行：

```bash
python run_demo_pipeline.py
```

Windows 下也可以直接双击：

```text
run_demo_pipeline.bat
```

默认输出：

- `demo_data/demo_pose_features.csv`：可复现 synthetic 特征表
- `demo_data/demo_experiment_plan_override.csv`：synthetic 验证标签和实验 override
- `demo_data/demo_manifest.json`：demo 数据生成说明
- `demo_outputs/DEMO_OVERVIEW.html`：浏览器友好的 demo 欢迎页，适合演示和快速打开结果
- `demo_outputs/DEMO_README.md`：本次 demo 输出阅读顺序
- `demo_outputs/DEMO_INTERPRETATION.md`：示例结果解读页，说明 PASS/候选排序/验证证据在 demo 中应如何理解
- `demo_outputs/REAL_DATA_STARTER/`：把 demo 替换成真实数据时使用的 input_csv、pose_features、override、pocket/catalytic 和检查清单模板
- `demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/`：可运行的 toy PDB 示例包，用于验证真实 `input_csv -> build_feature_table.py -> pipeline` 路径
- `demo_outputs/batch_decision_summary/batch_decision_summary.md`：一页式批次结论
- `demo_outputs/candidate_report_cards/index.html`：候选报告卡入口

注意：demo 数据用于检查安装、交互流程和导出效果；其中的 validation label 来自 synthetic proxy signal，不应当作为真实实验验证证据引用。

本地软件入口：左侧“没有数据时”点击“生成并立即运行 demo”，软件会自动生成 demo 特征表和 experiment override，并启动后台运行。若想先检查参数，则点击“生成并载入 demo 输入”，软件会自动切到 `feature_csv` 模式并填入路径，确认后再点击“立即运行”。

本地软件启动的 demo 也会在本次运行输出目录写入 `DEMO_OVERVIEW.html`、`DEMO_README.md` 和 `DEMO_INTERPRETATION.md`。HTML 页面适合直接浏览和展示；README 列出推荐阅读顺序；Interpretation 解释 demo 里的 PASS、候选排序和 synthetic 验证证据应该如何理解。

本地软件的“摘要”页会在检测到 demo 输出时显示“Demo 快速导览”，可直接打开 `DEMO_OVERVIEW.html`，也可下载 README 和解读文件；如果存在 `REAL_DATA_STARTER`，还可以一键打开真实数据 starter 文件夹和 `MINI_PDB_EXAMPLE` 示例文件夹。

如果你想确认真实 PDB 输入链路是否可用，可以先用 mini 示例从仓库根目录运行：

```bash
python build_feature_table.py --input_csv demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/input_pose_table.csv --out_csv demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/mini_pose_features.csv
```

mini 示例只用于 parser、residue mapping、ligand template 和几何特征流程检查；里面的 toy label 不是实验验证证据。

### 4.11 校准前后改进汇总

python summarize_rule_ml_improvement.py --baseline_summary comparison_rule_vs_ml/ranking_comparison_summary.json --calibrated_summary comparison_calibrated_rule_vs_ml/ranking_comparison_summary.json --calibrated_config calibration_outputs/calibrated_rule_config.json --out_dir improvement_summary

常用参数:
- --baseline_summary
- --calibrated_summary
- --calibrated_config
- --out_dir

### 4.12 校准策略自动搜索（基于已生成 trial）

python optimize_calibration_strategy.py --aggregation_trials_csv calibration_outputs/aggregation_calibration_trials.csv --baseline_summary_json comparison_rule_vs_ml/ranking_comparison_summary.json --use_baseline_guard --out_dir strategy_optimization

常用参数:
- --rank_weight_grid
- --selection_metrics
- --use_baseline_guard
- --rank_guard_tolerance --auc_guard_tolerance
- --min_nanobody_auc --min_rank_consistency
- --out_dir

提示:
- 推荐输出 `recommended_strategy.json` 可在下一次 `pipeline_smoke_test.py` 中直接作为 `--strategy_seed_json` 输入，形成策略优化 -> 校准回灌闭环。

### 4.13 推荐下一步统一入口

如果你的目标是直接执行 `not_perfect.md` 里建议的下一步，而不是手工串多条命令，优先使用：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir recommended_pipeline_outputs
```

如果你已经有 `pose_features.csv`，可直接从特征表启动：

```bash
python run_recommended_pipeline.py --feature_csv pose_features.csv --out_dir recommended_pipeline_outputs
```

如果特征表中的训练标签不是默认 `label`，用 `--label_col` 指定。例如验证回灌特征表默认使用 `experiment_label`：

```bash
python run_recommended_pipeline.py --feature_csv local_app_runs/experiment_validation_feedback/pose_features_with_experiment_labels.csv --label_col experiment_label --out_dir recommended_pipeline_validation_retrain
```

如果想在跑完后自动生成一份更适合阅读的解释摘要，可开启可选 AI 解释层：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir recommended_pipeline_outputs --enable_ai_assistant
```

默认 `--ai_provider none`，完全离线，不调用外部 API，只根据已有 summary/CSV/Markdown 生成本地解释文件。

如需接入 OpenAI：

```bash
set OPENAI_API_KEY=你的_key
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir recommended_pipeline_outputs --enable_ai_assistant --ai_provider openai
```

AI 解释层默认只读取压缩后的 `recommended_pipeline_summary.json` 和结果表格前几行，不上传原始 PDB 或完整输入 CSV；如果 OpenAI 调用失败，会自动回退到本地离线摘要。

特点:
- 自动执行 rule + ML 主链路
- 自动判断是否有足够 `label`
- 有标签时自动继续 compare/calibrate/strategy optimize
- 支持传入 `--pocket_overwide_penalty_weight` 和 `--pocket_overwide_threshold`，默认不改变排名
- 输出 `recommended_pipeline_summary.json` 便于回溯
- 输出 `recommended_pipeline_report.md` 便于先快速确认执行路径、关键指标和关键产物
- 可选输出 `ai_outputs/ai_run_summary.md`、`ai_top_candidates_explanation.md` 和 `ai_failure_diagnosis.md`
- 默认输出 `provenance/run_provenance_card.json`、`run_provenance_card.md`、`run_artifact_manifest.csv`、`run_input_file_manifest.csv` 和 `run_provenance_integrity.json`，用于复现审计和完整性校验
- 下一轮实验建议默认启用 diversity-aware ordering；可用 `--suggestion_diversity_mode none` 关闭
- 默认输出 `experiment_suggestions/experiment_plan.csv/md`；可用 `--experiment_plan_budget`、`--experiment_plan_group_quota` 等参数控制计划单预算
- 可用 `--experiment_plan_override_csv` 导入人工 include/exclude、实验状态、负责人、成本和备注；本地软件侧也提供同名 CSV 上传入口和计划单内置编辑器

验证 provenance 文件是否被改动：

```bash
python verify_run_provenance.py --signature_json recommended_pipeline_outputs\provenance\run_provenance_integrity.json --strict
```

说明：这是无私钥的 SHA256 完整性封存，适合发现文件被误改、复制不完整或报告与 manifest 不一致；它不是带私钥的正式数字签名。

### 4.14 导出结构标注与 interface residue

如果你的目标是给后续 viewer 或比赛展示壳准备结构化数据，而不是先改前端，优先使用：

```bash
python export_structure_annotations.py --pdb_path complex.pdb --out_dir structure_annotation_outputs
```

如果已经知道抗原链和 nanobody 链，建议显式指定：

```bash
python export_structure_annotations.py ^
  --pdb_path complex.pdb ^
  --out_dir structure_annotation_outputs ^
  --antigen_chain A ^
  --nanobody_chain H ^
  --pocket_file pocket.txt ^
  --catalytic_file catalytic.txt ^
  --key_residues "A:45,H:101"
```

特点:
- 不改现有 rule / ML 主链路
- 直接导出 residue-level annotation
- 导出 interface residue 列表
- 导出 interface residue pair 明细
- 支持用户关键残基输入并写回统一 bundle
- 导出 pocket payload 与简化阻断摘要
- 导出 `analysis_bundle.json`，便于后续前端或展示壳直接消费

### 4.15 本地交互式运行界面

如果你的目标不是手工敲多条命令，而是更直观地上传文件、填写参数、查看结果，优先使用本地交互入口：

```bash
python -m streamlit run local_ml_app.py
```

Windows 下也可以直接双击：

```bash
start_local_app.bat
```

本地软件侧边栏的“AI 解释（可选）”可以打开解释摘要。默认 provider 为 `none`，只生成离线摘要；选择 `openai` 时需要先设置 `OPENAI_API_KEY`，且只发送压缩摘要上下文，不发送原始结构文件。

当前这个脚本会优先使用仓库内 `.venv`，并直接调用桌面启动器，不再要求手工输入 `streamlit` 命令。

特点:
- 不改现有 `build_feature_table.py` / `rule_ranker.py` / `train_pose_model.py` / `rank_nanobodies.py` 主链路
- 页面里可选择从 `input_csv` 或 `pose_features.csv` 启动
- 支持上传文件，也支持直接填写本机文件路径
- 支持导入单个 zip 数据包，并自动回填 `input_csv` / `feature_csv` / 默认文件路径
- 支持直接填写一个本地数据目录路径并自动扫描、回填输入
- 当导入源没有现成 `input_pose_table.csv` / `pose_features.csv`、但能识别到 PDB 文件时，会自动生成 `auto_input_pose_table.csv`
- 自动生成的输入表会保守填入 `nanobody_id` / `conformer_id` / `pose_id` / `pdb_path`，并尽量复用识别到的 `pocket` / `catalytic` / `ligand` 默认文件
- 支持直接下载 `input_csv` / `pose_features.csv` 模板，减少手工整理格式
- 支持填写常用参数：
  - `top_k`
  - `train_epochs`
  - `train_batch_size`
  - `train_val_ratio`
  - `seed`
- 支持在运行前做输入预检：
  - 检查必需列是否完整
  - 展示批量数据画像：行数、nanobody 数、conformer 数、pose 数和重复 ID 行
  - 检查 `pdb_path` / `pocket_file` / `catalytic_file` / `ligand_file` 的路径覆盖与缺失情况
  - 预览本次会执行哪些 pipeline 阶段，以及哪些 label-aware 步骤会执行或跳过
  - 检查 label 是否足够支持 compare/calibration
  - 检查默认 `pocket/catalytic/ligand` 文件是否已有来源
  - 检查当前运行环境是否缺少 `torch` / `biopython`
  - 可下载 `input_preflight_report.json` 作为本批次输入检查记录
- 支持基础任务调度：
  - 立即运行当前表单配置
  - 把多组配置加入队列顺序执行
  - 在需要时停止当前后台任务
- 支持把当前结果导出成展示摘要 HTML：
  - 汇总关键指标、训练摘要、排名预览、诊断和执行报告
  - 可直接下载 HTML
  - 可用浏览器打印为 PDF
- 支持把当前结果导出成自动化 PDF：
  - 不依赖额外 PDF 工具链
  - 可直接下载 PDF
  - 可在本机直接打开
- 支持 `QC/Warning` 面板：
  - 展示统一 `PASS / WARN / FAIL` 质量门控结论
  - 展示 `feature_qc.json` 的处理摘要
  - 展示 failed 行列表
  - 展示 warning 行列表
  - 展示全空列和近常量列提示
- 支持 `运行对比` 面板：
  - 选择多条历史运行做并排比较
  - 对比 rule/ML 指标、训练摘要和 QC 摘要
  - 查看主指标趋势图
  - 按 `started_at` 日期聚合查看跨批次趋势
  - 查看每个批次的运行数、成功率、clean rate 和主指标批次均值
  - 直接下载批次聚合 CSV
  - 指定基准运行并生成 run-to-run 差异解释
  - 查看每个运行的正向驱动、负向拖累和归因摘要
  - 查看归因总表
  - 查看相对基准的差异表
  - 直接下载对比表 CSV
  - 生成并下载当前对比 HTML 汇总页
  - 生成并下载当前对比 PDF 汇总页
- 页面当前默认通过 [run_recommended_pipeline.py](run_recommended_pipeline.py) 的 CLI 子进程方式执行
- 这样可以支持后台运行、任务队列和停止当前任务
- 同时保留 `run_recommended_pipeline(...)` 这个 Python 级入口，便于后续继续扩展
- 运行后可在页面中预览：
  - `nanobody_ranking.csv`
  - `consensus_ranking.csv`
  - `pose_predictions.csv`
  - `training_summary.json`
  - `recommended_pipeline_report.md`
  - `recommended_pipeline_summary.json`
- 支持保存和载入参数模板
- 支持查看最近运行历史并恢复某次运行结果
- 支持按关键词 / 状态 / `start_mode` 筛选历史记录，并导出当前筛选结果 CSV
- 支持对队列中的单条任务做上移、下移和移除
- 支持基于历史运行直接复制配置并重新加入队列
- 支持按失败阶段给出诊断摘要和下一步建议
- 支持在页面中查看训练摘要卡片和 `train_log.csv` 曲线
- 支持对 ranking / pose 结果做 Top-N 过滤和关键 ID 检索
- 支持把当前 ranking / pose 筛选结果直接下载成 CSV
- 支持在 ranking / pose 结果页自定义显示列和排序方式
- 支持分别导出“当前可见列视图”与“当前全筛选结果”
- 支持在 ranking / pose 结果页按数值列设置最小值 / 最大值阈值
- 支持在运行对比页自定义批次表 / 归因表 / 差异表 / 完整对比表的显示列和排序方式
- 支持导出运行对比页的当前可见视图 CSV 和全表 CSV
- 支持在运行对比页按数值列设置最小值 / 最大值阈值
- 支持让运行对比页的 HTML / PDF 导出同步当前页面视图，而不是退回未裁剪的原始全表
- 支持让桌面启动器和打包 exe 使用 `--selftest-json` 输出结构化自检结果，便于自动验证路径解析
- 支持在“历史”页预览应用产物目录并做带确认的清理
- 支持展示运行产物清单，并可直接打开最近运行目录或输出目录
- 支持把当前运行的关键结果、日志和 metadata 一键打包成 zip 汇总包
- 支持生成每个 nanobody 的候选报告卡，并在排名结果页打开索引或下载 zip
- 支持生成下一轮实验建议，并在排名结果页筛选和下载
- 关键输出会写到：
  - `local_app_runs/<run_name>/outputs`

说明:
- 这是一层“本地交互式 UI 壳”，不是重写后的新后端。
- 当前更适合本机运行和比赛演示；后续如需要，再进一步打包成桌面程序。

### 4.16 构建桌面可执行程序

如果你希望直接双击启动，而不是先手工打开终端，可以使用桌面启动器：

```bash
build_desktop_app.bat
```

构建完成后会生成：

```bash
dist\ML_Local_App.exe
```

你也可以先做无界面自检：

```bash
python ml_desktop_launcher.py --selftest
python ml_desktop_launcher.py --selftest-json local_app_runs\launcher_selftest.json
dist\ML_Local_App.exe --selftest
```

当前桌面版的定位:
- 这是一个“桌面 launcher”，不是把整套分析内核完全重新打包成全独立便携版
- 它会自动定位当前仓库、复用当前 `.venv` 和 [local_ml_app.py](local_ml_app.py)
- 双击 `ML_Local_App.exe` 后，会弹出一个小型桌面控制窗口，并自动打开浏览器中的本地交互界面
- 关闭这个桌面控制窗口时，会同时停止本地 Streamlit 服务
- 当前启动器窗口和自检输出都带版本号，便于区分不同发布批次
- 当前 exe 已接入 `assets/app_icon.ico`，不再使用默认图标
- 当前会在真正启动前检查目标环境里是否已安装 `streamlit`，避免晚到运行阶段才报错

注意:
- 当前 `dist\ML_Local_App.exe` 最好保留在本仓库目录树内使用
- 如果你把 exe 单独拷到别处，它将无法自动找到当前仓库中的 `local_ml_app.py` 和 `.venv`

### 4.17 构建便携目录版

如果你希望做成“整个目录可拷走”的版本，而不只是当前仓库里的 launcher，可执行：

```bash
build_portable_bundle.bat
```

构建完成后会生成：

```bash
portable_dist\ML_Portable\
```

目录中会包含：
- `ML_Local_App.exe`
- `APP_VERSION.json`
- `app\` 下的运行源码
- `app\assets\app_icon.png`
- `app\assets\app_icon.ico`
- `app\.venv\`
- `app\local_app_runs\`
- `portable_bundle_manifest.json`
- `PORTABLE_README.txt`

当前便携目录版的定位:
- 它不是“单个 exe 完全独立”
- 它是“整个目录可直接拷走”的便携包
- 便携包中的 exe 会优先定位同级 `app\` 目录，并使用 `app\.venv` 启动本地界面
- 只要保持整个目录结构不变，就可以在其他 Windows 机器上直接双击 `ML_Local_App.exe`

### 4.18 构建单文件自解压版

如果你希望进一步压成单个 exe，而不是保留外部 `app\` 目录，可执行：

```bash
build_standalone_onefile.bat
```

构建完成后会生成：

```bash
portable_dist\standalone_onefile\ML_Local_App_Standalone.exe
portable_dist\standalone_onefile\ML_Local_App_Standalone.manifest.json
```

当前单文件版的定位:
- 这是“单文件自解压基础版”，不是已经跨所有机器完全验证完毕的最终发布形态
- exe 内嵌 `app\` 与 `app\.venv\`，运行时通过 PyInstaller onefile 机制解压到临时目录
- 启动器现在支持优先从内嵌 `app\` 根目录定位运行环境，而不是只能依赖外部仓库或便携目录
- 当前已在本机完成构建，并增加了“复制到系统临时目录后再运行”的自动化校验
- `build_standalone_onefile.bat` 现在会额外生成：

```bash
portable_dist\standalone_onefile_validation\standalone_validation_latest.json
```

- 这份校验报告会确认：
  - `repo_root_source = meipass_app`
  - `python_executable` 来自内嵌 `app\.venv`
  - 当前运行没有回退到宿主仓库目录
- 后续仍建议继续做真实外部机器验证，重点关注内嵌 `.venv` 在不同 Windows 环境下的一致性

### 4.19 构建可分发 zip 发布包

如果你希望直接得到一个可发给别人的 zip，而不是手工压缩便携目录，可执行：

```bash
build_portable_release.bat
```

构建完成后会生成：

```bash
portable_dist\ML_Portable_release.zip
portable_dist\ML_Portable_release.manifest.json
```

其中：
- `ML_Portable_release.zip` 是可直接分发的发布包
- `ML_Portable_release.manifest.json` 记录了 zip 的版本号、SHA256、文件大小和打包条目

当前发布包的定位:
- 先自动重建桌面 launcher
- 再自动重建 `portable_dist\ML_Portable\`
- 最后把整个便携目录压缩成单个 zip
- 解压后，仍然是“目录便携版”的运行方式；也就是保持解压后的目录结构不变，再双击 `ML_Local_App.exe`

### 4.20 GitHub 自动发布

仓库当前已经补了 GitHub Actions 发布工作流：

```text
.github/workflows/desktop-release.yml
```

它支持两种方式：
- 在 GitHub Actions 页面手动 `Run workflow`
- 推送形如 `v0.1.0` 的 tag

工作流会自动：
- 在 `windows-latest` 上创建 `.venv`
- 安装 `requirements.txt`
- 构建 `dist/ML_Local_App.exe`
- 构建 `portable_dist/ML_Portable/`
- 构建 `portable_dist/ML_Portable_release.zip`
- 上传这些产物到 Actions artifacts

如果是 `v*` tag 触发：
- 还会自动创建 GitHub Release
- 并附上：
  - `ML_Local_App.exe`
  - `ML_Portable_release.zip`
  - `ML_Portable_release.manifest.json`

### 4.21 分组交叉验证 benchmark（有真实 label 时推荐）

```bash
python benchmark_pose_pipeline.py --feature_csv pose_features.csv --out_dir benchmark_outputs --folds 5
```

常用参数:
- --folds --seed --device
- --epochs --batch_size --lr --weight_decay
- --soft_target_weight
- --pseudo_threshold_mode --pseudo_threshold_value
- --top_k --optional_weight
- --w_mean --w_best --w_consistency --w_std_penalty
- --consistency_hit_threshold
- --reliability_bins

关键输出:
- fold_metrics.csv
- pose_cv_predictions.csv
- nanobody_benchmark_table.csv
- pose_reliability_curve.csv
- nanobody_reliability_curve.csv
- geometry_proxy_benchmark.csv
- benchmark_summary.json
- benchmark_report.md

### 4.22 CD38 口袋准确性 test（已知关键残基 baseline）

如果你想先不跑整条抗体/纳米抗体主链，而是先拿单个 CD38 结构验证 pocket residue 结果是否覆盖已知关键位点，可执行：

```bash
python benchmark_cd38_pocket_accuracy.py --rcsb_pdb_id 3F6Y --predicted_pocket_file benchmarks\cd38\cd38_active_site_truth.txt --out_dir cd38_pocket_benchmark_outputs
```

说明：
- `benchmarks\cd38\cd38_active_site_truth.txt` 是基于文献整理的 CD38 baseline truth
- 上面这条命令属于 self-check；真正评估时，应把 `--predicted_pocket_file` 换成 `fpocket` / `P2Rank` / 手工 pocket 文件
- 脚本会输出 exact overlap、near-hit coverage、precision、Jaccard、F1、truth-based `overwide_pocket_score` 和逐残基距离表
- 如果你已经有 `P2Rank` 的 `predictions.csv`，可直接运行 `python run_cd38_p2rank_benchmark.py --predictions_csv your_p2rank_dir\\3ROP.pdb_predictions.csv --rcsb_pdb_id 3ROP --rank 2 --chain_filter A --out_dir benchmarks\\cd38\\results\\3ROP_p2rank_rank2_chainA`
- 如果你已经有 `fpocket` 的 `pocket*_atm.pdb`，可直接运行 `python run_cd38_fpocket_benchmark.py --fpocket_pocket_pdb your_fpocket_dir\\pockets\\pocket1_atm.pdb --rcsb_pdb_id 3ROP --chain_filter A --out_dir benchmarks\\cd38\\results\\3ROP_fpocket_pocket1_chainA`
- 如果你已经有一批 `fpocket` 输出目录，可运行 `python prepare_cd38_fpocket_panel.py --fpocket_root your_fpocket_outputs --rcsb_pdb_id 3ROP`，先生成 fpocket manifest 和 readiness report；确认后再加 `--run` 批量加入 benchmark panel
- 如果你想直接从 ligand-bound 结构生成 baseline，可运行 `python run_cd38_ligand_contact_benchmark.py --rcsb_pdb_id 3ROP --protein_chain A --ligand_chain A --ligand_resnames 50A,NCA --out_dir benchmarks\\cd38\\results\\3ROP_ligand_contact_chainA_50A_NCA`
- 如果你想把当前 CD38 结构化 benchmark 结果汇总成总表，可运行 `python summarize_cd38_benchmarks.py --results_root benchmarks\\cd38\\results`
- 如果你想按 manifest 批量检查或复跑当前 panel，可运行 `python run_cd38_benchmark_manifest.py`
- 更详细说明见 [benchmarks/cd38/README.md](benchmarks/cd38/README.md)
- 当前仓库内已经补了两组真实 ligand-contact baseline，以及两组真实 `P2Rank` baseline，见 [benchmarks/cd38/BASELINE_RESULTS.md](benchmarks/cd38/BASELINE_RESULTS.md)
- 当前仓库内还补了两组 ligand-contact vs P2Rank 的 pocket 方法共识结果，用于区分“共同支持的核心 pocket”和“单一方法带出的边缘 residue”
- 当前聚合总表见 [benchmarks/cd38/results/cd38_benchmark_panel.csv](benchmarks/cd38/results/cd38_benchmark_panel.csv) 和 [benchmarks/cd38/results/cd38_benchmark_panel.md](benchmarks/cd38/results/cd38_benchmark_panel.md)
- 主链几何特征现在也会输出无 truth 依赖的 `pocket_shape_*` 字段，其中 `pocket_shape_overwide_proxy` 可用于提示 pocket 定义是否偏宽
- `feature_qc.json` 会汇总 `pocket_shape_qc`，本地软件的 QC/Warning 面板和 HTML 摘要会显示高 overwide 行
- `rank_nanobodies.py` 与 `rule_ranker.py` 的 explanation 会在 `pocket_shape_overwide_proxy` 较高时提示“建议复核口袋边界”
- 当前还提供默认关闭的轻量惩罚项：设置 `--pocket_overwide_penalty_weight > 0` 后，会对 `pocket_shape_overwide_proxy` 高于 `--pocket_overwide_threshold` 的结果降低少量分数；默认 `0.0` 不改变原有排名

### 4.23 Pocket 方法共识分析

如果你已经有多种来源的 pocket residue 文件，例如手工定义、ligand-contact、P2Rank 或 fpocket，可以用下面的脚本比较它们是否指向同一个核心口袋：

```bash
python compare_pocket_method_consensus.py ^
  --method ligand_contact=benchmarks\cd38\results\3ROP_ligand_contact_chainA_50A_NCA\predicted_pocket.txt ^
  --method p2rank=benchmarks\cd38\results\3ROP_p2rank_rank2_chainA\predicted_pocket.txt ^
  --truth_file benchmarks\cd38\cd38_active_site_truth.txt ^
  --out_dir benchmarks\cd38\results\3ROP_method_consensus_ligand_p2rank
```

输出:
- `consensus_pocket_residues.txt`
- `union_pocket_residues.txt`
- `residue_method_membership.csv`
- `method_specific_residues.csv`
- `method_overlap_matrix.csv`
- `pocket_method_consensus_summary.json`
- `pocket_method_consensus_report.md`

说明:
- `consensus_pocket_residues.txt` 是至少被指定数量方法共同支持的核心 pocket residue。
- `union_pocket_residues.txt` 是所有方法的 residue 并集，更适合看覆盖范围。
- `method_specific_residues.csv` 用于定位只被单一方法支持、可能导致 pocket 偏宽的 residue。
- 如果提供 `--truth_file`，脚本会输出 truth coverage、precision、missing truth risk 和 overwide risk。
- 如果没有 `--truth_file`，脚本仍会输出基于方法一致性和 union/consensus 扩张程度的 proxy 风险。
- 该脚本不直接运行 P2Rank/fpocket；外部工具输出需要先通过仓库里的提取脚本转换为 residue list。

### 4.24 排名参数敏感性分析

如果你已经跑出了 `consensus_ranking.csv`，可以直接检查候选排名是否对权重和 QC 惩罚敏感：

```bash
python analyze_ranking_parameter_sensitivity.py ^
  --consensus_csv consensus_outputs\consensus_ranking.csv ^
  --out_dir parameter_sensitivity_outputs ^
  --top_n 5
```

输出:
- `scenario_rankings.csv`
- `scenario_summary.csv`
- `candidate_rank_sensitivity.csv`
- `sensitive_candidates.csv`
- `parameter_sensitivity_summary.json`
- `parameter_sensitivity_report.md`

说明:
- 该步骤只读取 `consensus_ranking.csv`，不重新训练模型。
- `candidate_rank_sensitivity.csv` 适合看每个候选的 best/worst rank 和 rank span。
- `sensitive_candidates.csv` 只保留对参数变化敏感的候选。
- `top_n_unstable=True` 表示候选可能跨过 Top-N 决策边界，建议人工复核。
- 推荐 pipeline 会自动把结果写到 `my_outputs/parameter_sensitivity/`。

### 4.25 CD38 pocket 参数敏感性分析

如果你想检查 CD38 pocket benchmark 对 contact cutoff、P2Rank rank、method consensus 阈值和 overwide penalty 是否敏感，可运行：

```bash
python analyze_cd38_pocket_parameter_sensitivity.py ^
  --manifest_csv benchmarks\cd38\cd38_benchmark_manifest.csv ^
  --results_root benchmarks\cd38\results ^
  --truth_file benchmarks\cd38\cd38_active_site_truth.txt ^
  --out_dir benchmarks\cd38\parameter_sensitivity
```

输出:
- `contact_cutoff_sensitivity.csv`
- `p2rank_rank_sensitivity.csv`
- `fpocket_pocket_sensitivity.csv`
- `method_consensus_threshold_sensitivity.csv`
- `overwide_penalty_sensitivity.csv`
- `cd38_pocket_parameter_sensitivity_summary.json`
- `cd38_pocket_parameter_sensitivity_report.md`

说明:
- 该脚本只使用当前仓库已有的 CD38 本地 PDB、ligand-contact 结果和 P2Rank `source_predictions.csv`。
- 不运行外部 P2Rank/fpocket；如果 manifest 里已有 fpocket rows，则读取对应 `predicted_pocket.txt` 或 `pocket*_atm.pdb` 结果做同口径评估。
- 当前结果显示：`4OGW` 在较小 ligand-contact cutoff 下更紧；`3ROP` 的 P2Rank 必须选链 A 对应的 `rank 2`；`4OGW` 的 strict method consensus 能明显收紧 P2Rank 偏宽 pocket。

如果你想进一步判断主链运行时的 `pocket_shape_overwide_proxy` 阈值是否应该改变，运行：

```bash
python build_cd38_proxy_calibration_report.py
```

输出：

- `benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_report.md`
- `benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_summary.json`
- `benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_rows.csv`
- `benchmarks\cd38\proxy_calibration\cd38_proxy_threshold_candidates.csv`
- `benchmarks\cd38\proxy_calibration\cd38_proxy_method_summary.csv`
- `benchmarks\cd38\proxy_calibration\cd38_proxy_penalty_summary.csv`

当前校准报告会明确保留默认策略：`pocket_overwide_penalty_weight=0.0`，因为当前只有 2 个 CD38 结构、2 类方法且没有真实 fpocket 行；`0.15` 只建议作为敏感性复核参数，不建议作为全局默认值。

### 4.26 生成 CD38 P2Rank/fpocket 外部工具输入包

如果你还没有真实 P2Rank / fpocket 输出，先不要手工到处复制 PDB。直接生成一份外部工具输入包：

```bash
python prepare_cd38_external_tool_inputs.py
```

输出：

- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_input_manifest.csv`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_expected_returns.csv`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_return_checklist.md`
- `benchmarks\cd38\external_tool_inputs\run_p2rank_templates.ps1`
- `benchmarks\cd38\external_tool_inputs\run_fpocket_templates.ps1`
- `benchmarks\cd38\external_tool_inputs\run_p2rank_templates.sh`
- `benchmarks\cd38\external_tool_inputs\run_fpocket_templates.sh`
- `benchmarks\cd38\external_tool_inputs\check_external_tool_environment.ps1`
- `benchmarks\cd38\external_tool_inputs\refresh_readiness_after_external_tools.ps1`
- `benchmarks\cd38\external_tool_inputs\finalize_external_benchmark.ps1`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_inputs_summary.json`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_inputs.md`

推荐流程：

1. 先运行 `check_external_tool_environment.ps1`，确认本机是否有 `prank` / `fpocket`，以及输出是否已经存在。
2. 在已安装 P2Rank 的机器上运行或改写 `run_p2rank_templates.ps1`。
3. 在已安装 fpocket 的机器上运行或改写 `run_fpocket_templates.ps1`。
4. 如果是在 Linux/WSL 运行，改用 `run_p2rank_templates.sh` 和 `run_fpocket_templates.sh`。
5. 如果是在别的机器运行，先对照 `cd38_external_tool_return_checklist.md` 或 `cd38_external_tool_expected_returns.csv`，确认应返回的 `PDB × method` 输出是否都存在。
6. 把输出目录复制回 `external_tool_inputs/`。
7. 再运行 `check_external_tool_environment.ps1`，确认 P2Rank CSV 和 fpocket pocket 文件已经落在预期位置。
8. 运行 `finalize_external_benchmark.ps1`，一条命令完成 preflight + readiness 汇总。
9. 确认 report 后，再用 `finalize_cd38_external_benchmark.py --run_discovered` 加入 benchmark panel。
10. 如果仍缺输出，先看 `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md`，按 priority 补齐真实外部工具结果。

说明：
- 该脚本不内置 P2Rank/fpocket，也不替代外部工具本身；它解决的是输入组织、命令模板、目录约定和后续导入问题。
- 当前会为 `3ROP`、`4OGW`、`3F6Y` 生成输入，`3F6Y` 只做 P2Rank/fpocket，不做 ligand-contact baseline。

### 4.27 打包 CD38 外部工具输入包用于转移运行

如果你要把输入包传到另一台机器、Linux 或 WSL 上运行 P2Rank/fpocket，先生成 transfer zip：

```bash
python package_cd38_external_tool_inputs.py
```

输出：

- `benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer.zip`
- `benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer_manifest.csv`
- `benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer_summary.json`
- `benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer_report.md`

默认 zip 只包含运行外部工具必要文件：PDB 输入、PowerShell/Bash 模板、manifest 和说明。它不会把 `preflight/`、`finalize/`、旧 `p2rank_outputs/` 或 fpocket `*_out/` 目录一起打进去。

如果你明确想把已有外部输出也一起打包，可加：

```bash
python package_cd38_external_tool_inputs.py --include_existing_outputs
```

### 4.28 导入外部机器返回的 P2Rank/fpocket 输出

如果你在另一台机器或 WSL/Linux 上跑完了 P2Rank/fpocket，可以把返回的 `external_tool_inputs` 目录或 zip 交给导入脚本：

```bash
python import_cd38_external_tool_outputs.py --source returned_external_tool_inputs.zip
```

也支持先 dry run：

```bash
python import_cd38_external_tool_outputs.py --source returned_external_tool_inputs.zip --dry_run
```

输出：

- `benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_manifest.csv`
- `benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_scan.csv`
- `benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_coverage.csv`
- `benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_repair_plan.csv`
- `benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_summary.json`
- `benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_report.md`

说明：
- 只会导入 `p2rank_outputs/` 和 `fpocket_runs/*/*_out/`。
- 返回 zip/目录可以是 `external_tool_inputs/` 本身，也可以外面多包一层目录；脚本会自动剥离到 package 相对路径。
- 如果候选输出文件为 0，先看 `cd38_external_tool_output_import_scan.csv`，它会列出被忽略文件和原因。
- import report 会给出 `source_diagnosis`；如果是 `input_package_without_external_outputs`，说明你导入的是原始输入包，还需要先在外部机器运行 P2Rank/fpocket。
- coverage CSV 会按 `3ROP:p2rank`、`3ROP:fpocket` 这类粒度列出返回包已经覆盖和仍缺的结构方法组合。
- repair plan CSV 会把缺失项展开成可执行清单，例如应运行 `run_p2rank_templates.ps1` 还是 `run_fpocket_templates.sh`、应返回哪个相对路径、下一条 dry-run 验证命令是什么。
- 默认不覆盖已有文件；需要替换旧输出时显式加 `--overwrite`。
- 导入完成后再运行 `python finalize_cd38_external_benchmark.py --run_discovered`。
- 如果想少一步，也可以直接运行 `python finalize_cd38_external_benchmark.py --import_source returned_external_tool_inputs.zip --run_discovered`。

### 4.29 检查 CD38 外部工具环境和输出

如果你不确定外部工具到底是“没安装”、还是“已经跑了但输出没放对位置”，运行：

```bash
python check_cd38_external_tool_environment.py
```

也可以在输入包目录中直接运行：

```powershell
benchmarks\cd38\external_tool_inputs\check_external_tool_environment.ps1
```

输出：

- `benchmarks\cd38\external_tool_inputs\preflight\cd38_external_tool_preflight_status.csv`
- `benchmarks\cd38\external_tool_inputs\preflight\cd38_external_tool_preflight_summary.json`
- `benchmarks\cd38\external_tool_inputs\preflight\cd38_external_tool_preflight_report.md`

说明：
- preflight 使用 `package_portable_first` 路径解析策略。
- 即使 manifest 里保留生成时的绝对路径，检查时也会优先看当前 `external_tool_inputs/` 里的 `pdbs/`、`p2rank_outputs/` 和 `fpocket_runs/`。
- `*_status.csv` 里会显示 `pdb_input_source`、`p2rank_source` 和 `fpocket_source`，方便判断到底用了包内路径还是 manifest 路径。

当前本机预检结论：
- P2Rank `prank` 命令未在 PATH 中发现。
- `fpocket` 命令未在 PATH 中发现。
- `3ROP`、`4OGW`、`3F6Y` 的 PDB 输入都已准备好。
- 3 个 P2Rank 输出和 3 个 fpocket 输出仍缺，需要在已安装外部工具的环境运行模板后再导入。

### 4.30 一键 finalize CD38 外部 benchmark 接入

外部 P2Rank/fpocket 输出复制回 `external_tool_inputs/` 后，推荐先运行：

```bash
python finalize_cd38_external_benchmark.py
```

如果拿回来的是整个目录或 zip，可以让 finalize 先导入再检查：

```bash
python finalize_cd38_external_benchmark.py --import_source returned_external_tool_inputs.zip
```

或直接运行输入包里的 PowerShell 快捷入口：

```powershell
benchmarks\cd38\external_tool_inputs\finalize_external_benchmark.ps1
```

输出：

- `benchmarks\cd38\external_tool_inputs\finalize\cd38_external_benchmark_finalize_summary.json`
- `benchmarks\cd38\external_tool_inputs\finalize\cd38_external_benchmark_finalize_report.md`
- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md`
- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.csv`
- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.json`
- 同步刷新 `preflight/` 和 `benchmarks\cd38\readiness\`

默认只检查和汇总，不导入 benchmark。确认 readiness report 里 P2Rank/fpocket manifest rows 正确后，再运行：

```bash
python finalize_cd38_external_benchmark.py --run_discovered
```

如果 readiness 没有发现任何可运行 P2Rank/fpocket rows，`--run_discovered` 会被安全跳过；finalize 报告会显示 `Runnable external benchmark rows: 0` 和 `Benchmark follow-up status: skipped_no_runnable_external_rows`。

如果已经确认返回包就是本次真实输出，也可以合并成一条命令：

```bash
python finalize_cd38_external_benchmark.py --import_source returned_external_tool_inputs.zip --run_discovered
```

如果导入后还想刷新参数敏感性分析：

```bash
python finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity
```

`--run_sensitivity` 成功后还会自动刷新：

- `benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_report.md`
- `benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_summary.json`

当前本机因为缺少真实外部输出，finalize 报告会停在 `P2Rank missing outputs = 3`、`fpocket missing outputs = 3`，不会误导入 benchmark。

### 4.30.1 查看 CD38 外部 benchmark action plan

如果你想直接看“下一步到底该补哪个结构、哪个工具输出、复制回哪里”，运行：

```bash
python build_cd38_external_benchmark_action_plan.py
```

默认输出：

- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md`
- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.csv`
- `benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.json`

当前清单会列出 6 个外部输出动作，其中 4 个是 benchmark completion blocker：

- priority `1`: `3ROP fpocket`、`4OGW fpocket`
- priority `2`: `3F6Y P2Rank`、`3F6Y fpocket`
- priority `3`: `3ROP P2Rank`、`4OGW P2Rank`，主要用于外部包复现，不是当前 benchmark panel 的新增 blocker

### 4.30.2 生成 CD38 外部工具 next-run runbook

如果你不想在外部机器上运行所有模板，只想跑当前 action plan 里真正缺的 benchmark blocker，可以生成 next-run runbook：

```bash
python build_cd38_external_tool_runbook.py
```

默认输出在 `benchmarks\cd38\external_tool_inputs\`：

- `cd38_external_tool_next_run.md`
- `cd38_external_tool_next_run_plan.csv`
- `cd38_external_tool_next_run_summary.json`
- `run_cd38_external_next_benchmark.ps1`
- `run_cd38_external_next_benchmark.sh`

当前默认会选择 4 个 benchmark completion 动作：`3ROP fpocket`、`4OGW fpocket`、`3F6Y fpocket`、`3F6Y P2Rank`。如果还想补齐外部包复现用的 `3ROP/4OGW P2Rank`，运行：

```bash
python build_cd38_external_tool_runbook.py --include_package_reproducibility
```

`package_cd38_external_tool_inputs.py` 现在会在打包 transfer zip 前自动刷新这份 runbook，并把 next-run 说明、CSV、PowerShell 和 Bash 脚本一起放入 zip。外部机器上优先运行 `run_cd38_external_next_benchmark.*`；只有在明确需要所有模板时，再运行 `run_p2rank_templates.*` 和 `run_fpocket_templates.*`。

### 4.31 批量准备 fpocket benchmark panel

如果你已经用外部 `fpocket` 跑出了一批输出目录，可以先让仓库自动发现 `pocket*_atm.pdb` 并生成 CD38 benchmark manifest：

```bash
python prepare_cd38_fpocket_panel.py ^
  --fpocket_root your_fpocket_outputs ^
  --rcsb_pdb_id 3ROP ^
  --chain_filter A ^
  --manifest_out benchmarks\cd38\fpocket_discovered_manifest.csv ^
  --report_md benchmarks\cd38\fpocket_discovered_manifest.csv.report.md ^
  --run
```

输出:
- `fpocket_discovered_manifest.csv`
- `fpocket_discovered_manifest.csv.summary.json`
- `fpocket_discovered_manifest.csv.report.md`
- `benchmarks/cd38/results/<PDB>_fpocket_pocket*_chainA/`
- `benchmarks/cd38/results/manifest_run_summary.json`
- `benchmarks/cd38/results/cd38_benchmark_panel.csv`

说明:
- 脚本扫描 `**/pockets/pocket*_atm.pdb` 和 `**/pocket*_atm.pdb`。
- 如果路径名里包含 `3ROP_out`、`4OGW_out` 这类首位为数字的 PDB ID，可自动推断 PDB ID；否则用 `--rcsb_pdb_id` 明确指定。
- `*.report.md` 会列出扫描目录、可运行 row 数、跳过文件原因和建议下一步，适合先检查真实 fpocket 输出是否能接入 benchmark。
- 加 `--dry_run --run` 可以只检查将要执行的命令，不真正跑 benchmark。
- 后续再运行 `analyze_cd38_pocket_parameter_sensitivity.py` 时，`fpocket_pocket_sensitivity.csv` 会纳入这些 fpocket pocket 结果。

### 4.32 批量准备 P2Rank benchmark panel

如果你已经用外部 `P2Rank` 跑出了一批 `*_predictions.csv`，可以先让仓库自动发现并生成 CD38 benchmark manifest：

```bash
python prepare_cd38_p2rank_panel.py ^
  --p2rank_root your_p2rank_outputs ^
  --chain_filter A ^
  --manifest_out benchmarks\cd38\p2rank_discovered_manifest.csv ^
  --rank_by_pdb 3ROP=2,4OGW=1
```

输出:
- `p2rank_discovered_manifest.csv`
- `p2rank_discovered_manifest.csv.summary.json`
- `p2rank_discovered_manifest.csv.report.md`

说明:
- 脚本扫描 `**/*_predictions.csv`、`**/*predictions.csv` 和 `**/predictions.csv`。
- 如果路径或文件名里包含 `3F6Y.pdb_predictions.csv`、`3ROP_predictions.csv` 这类首位为数字的 PDB ID，可自动推断 PDB ID；否则用 `--rcsb_pdb_id` 明确指定。
- `*.report.md` 会列出扫描目录、可运行 row 数、跳过文件原因和建议下一步，适合先检查真实 P2Rank 输出是否能接入 benchmark。
- 如果同源链导致 active-site pocket 不是全局 rank 1，可用 `--rank_by_pdb 3ROP=2` 指定。
- 确认无误后加 `--run` 可直接批量加入 benchmark panel。

### 4.33 一键刷新 CD38 benchmark 准备状态

如果你想一次性刷新 ligand 检查、结构扩展缺口和外部工具输出准备状态，可运行：

```bash
python refresh_cd38_benchmark_readiness.py
```

默认输出：

- `benchmarks\cd38\readiness\cd38_benchmark_readiness_summary.json`
- `benchmarks\cd38\readiness\cd38_benchmark_readiness_commands.json`
- `benchmarks\cd38\readiness\cd38_benchmark_readiness_report.md`

如果你已经有外部 P2Rank / fpocket 输出目录，可以一起扫描：

```bash
python refresh_cd38_benchmark_readiness.py ^
  --p2rank_root your_p2rank_outputs ^
  --fpocket_root your_fpocket_outputs ^
  --rank_by_pdb 3ROP=2,4OGW=1
```

说明：
- 默认不会执行 benchmark，只刷新准备状态和报告。
- 如果 readiness report 显示 manifest row 正确，再加 `--run_discovered` 或手动运行生成的 manifest。
- 这是目前推荐的 CD38 benchmark 下一步入口，比单独手动跑多个检查脚本更不容易漏步骤。

### 4.34 生成 CD38 benchmark 扩展计划

如果你想知道当前 CD38 benchmark 还缺哪些结构和方法结果，可运行：

```bash
python build_cd38_benchmark_expansion_plan.py
```

默认读取：

- `benchmarks\cd38\cd38_structure_targets.csv`
- `benchmarks\cd38\cd38_benchmark_manifest.csv`
- `benchmarks\cd38\results\cd38_benchmark_panel.csv`

输出：

- `benchmarks\cd38\expansion_plan\cd38_benchmark_expansion_plan.csv`
- `benchmarks\cd38\expansion_plan\cd38_benchmark_missing_actions.csv`
- `benchmarks\cd38\expansion_plan\cd38_benchmark_expansion_summary.json`
- `benchmarks\cd38\expansion_plan\cd38_benchmark_expansion_plan.md`

当前计划会明确显示：

- `3ROP` 和 `4OGW` 的 ligand-contact / P2Rank baseline 已完成
- `3ROP`、`4OGW`、`3F6Y` 都还缺真实 fpocket 输出
- `3F6Y` 已确认没有活性口袋 ligand candidate，因此只作为 P2Rank/fpocket pocket-finder 测试结构

### 4.35 检查 CD38 结构里的 ligand candidates

如果你不确定某个 CD38 结构是否适合做 ligand-contact baseline，可运行：

```bash
python inspect_cd38_ligand_candidates.py
```

默认读取：

- `benchmarks\cd38\cd38_structure_targets.csv`
- `benchmarks\cd38\cd38_active_site_truth.txt`

输出：

- `benchmarks\cd38\ligand_candidates\cd38_ligand_candidates.csv`
- `benchmarks\cd38\ligand_candidates\cd38_recommended_ligand_candidates.csv`
- `benchmarks\cd38\ligand_candidates\cd38_structure_targets_suggested.csv`
- `benchmarks\cd38\ligand_candidates\cd38_ligand_candidate_summary.json`
- `benchmarks\cd38\ligand_candidates\cd38_ligand_candidate_report.md`

当前结果：

- `3ROP`: 推荐 `A:50A:301` 和 `A:NCA:302`
- `4OGW`: 推荐 `A:NMN:401`
- `3F6Y`: 未发现活性口袋 ligand-like HETATM，因此不做 ligand-contact baseline

---

## 5. 输入输出契约

### 5.1 build_feature_table.py 输入

必需列:
- nanobody_id
- conformer_id
- pose_id
- pdb_path

可选列:
- antigen_chain
- nanobody_chain
- pocket_file
- catalytic_file
- ligand_file
- 其他可转数值列（将自动透传）

### 5.2 关键输出文件

build_feature_table.py:
- pose_features.csv
- feature_qc.json

train_pose_model.py:
- pose_predictions.csv
- best_model.pt
- train_log.csv
- feature_columns.json
- training_summary.json
- training_summary.csv

rank_nanobodies.py:
- conformer_scores.csv
- nanobody_ranking.csv

rule_ranker.py:
- pose_rule_scores.csv
- conformer_rule_scores.csv
- nanobody_rule_ranking.csv

calibrate_rule_ranker.py:
- feature_calibration_trials.csv
- aggregation_calibration_trials.csv
- calibrated_rule_config.json
- calibrated_rule_outputs/pose_rule_scores.csv
- calibrated_rule_outputs/conformer_rule_scores.csv
- calibrated_rule_outputs/nanobody_rule_ranking.csv

compare_rule_ml_rankings.py:
- ranking_comparison_table.csv
- ranking_comparison_summary.json
- ranking_comparison_report.md

build_consensus_ranking.py:
- consensus_ranking.csv
- consensus_summary.json
- consensus_report.md

build_quality_gate.py:
- quality_gate_summary.json
- quality_gate_checks.csv
- quality_gate_report.md

build_score_explanation_cards.py:
- score_explanation_cards.csv
- score_explanation_cards_summary.json
- score_explanation_cards.md
- score_explanation_cards.html

build_batch_decision_summary.py:
- batch_decision_summary.json
- batch_decision_summary.md
- batch_decision_summary_cards.csv

build_cd38_proxy_calibration_report.py:
- cd38_proxy_calibration_summary.json
- cd38_proxy_calibration_report.md
- cd38_proxy_calibration_rows.csv
- cd38_proxy_threshold_candidates.csv
- cd38_proxy_method_summary.csv
- cd38_proxy_penalty_summary.csv

build_candidate_report_cards.py:
- index.html
- cards/*.html
- candidate_report_cards.csv
- candidate_report_cards_summary.json
- candidate_report_cards.zip

build_candidate_comparisons.py:
- candidate_tradeoff_table.csv
- candidate_pairwise_comparisons.csv
- candidate_comparison_summary.json
- candidate_comparison_report.md

suggest_next_experiments.py:
- next_experiment_suggestions.csv
- next_experiment_suggestions_summary.json
- next_experiment_suggestions_report.md

build_validation_evidence_audit.py:
- validation_evidence_summary.json
- validation_evidence_report.md
- validation_evidence_by_candidate.csv
- validation_evidence_topk.csv
- validation_evidence_action_items.csv

analyze_ranking_parameter_sensitivity.py:
- scenario_rankings.csv
- scenario_summary.csv
- candidate_rank_sensitivity.csv
- sensitive_candidates.csv
- parameter_sensitivity_summary.json
- parameter_sensitivity_report.md

analyze_cd38_pocket_parameter_sensitivity.py:
- contact_cutoff_sensitivity.csv
- p2rank_rank_sensitivity.csv
- fpocket_pocket_sensitivity.csv
- method_consensus_threshold_sensitivity.csv
- overwide_penalty_sensitivity.csv
- cd38_pocket_parameter_sensitivity_summary.json
- cd38_pocket_parameter_sensitivity_report.md

prepare_cd38_external_tool_inputs.py:
- cd38_external_tool_input_manifest.csv
- run_p2rank_templates.ps1
- run_fpocket_templates.ps1
- run_p2rank_templates.sh
- run_fpocket_templates.sh
- check_external_tool_environment.ps1
- refresh_readiness_after_external_tools.ps1
- finalize_external_benchmark.ps1
- cd38_external_tool_expected_returns.csv
- cd38_external_tool_return_checklist.md
- cd38_external_tool_inputs_summary.json
- cd38_external_tool_inputs.md

package_cd38_external_tool_inputs.py:
- cd38_external_tool_inputs_transfer.zip
- cd38_external_tool_inputs_transfer_manifest.csv
- cd38_external_tool_inputs_transfer_summary.json
- cd38_external_tool_inputs_transfer_report.md

import_cd38_external_tool_outputs.py:
- cd38_external_tool_output_import_manifest.csv
- cd38_external_tool_output_import_scan.csv
- cd38_external_tool_output_import_coverage.csv
- cd38_external_tool_output_import_repair_plan.csv
- cd38_external_tool_output_import_summary.json
- cd38_external_tool_output_import_report.md

check_cd38_external_tool_environment.py:
- cd38_external_tool_preflight_status.csv
- cd38_external_tool_preflight_summary.json
- cd38_external_tool_preflight_report.md

finalize_cd38_external_benchmark.py:
- cd38_external_benchmark_finalize_summary.json
- cd38_external_benchmark_finalize_report.md

build_cd38_external_benchmark_action_plan.py:
- cd38_external_benchmark_action_plan.md
- cd38_external_benchmark_action_plan.csv
- cd38_external_benchmark_action_plan.json

build_cd38_external_tool_runbook.py:
- cd38_external_tool_next_run.md
- cd38_external_tool_next_run_plan.csv
- cd38_external_tool_next_run_summary.json
- run_cd38_external_next_benchmark.ps1
- run_cd38_external_next_benchmark.sh

build_geometry_proxy_audit.py:
- geometry_proxy_audit_summary.json
- geometry_proxy_audit_report.md
- geometry_proxy_feature_summary.csv
- geometry_proxy_flagged_poses.csv
- geometry_proxy_candidate_audit.csv

prepare_cd38_fpocket_panel.py:
- fpocket_discovered_manifest.csv
- fpocket_discovered_manifest.csv.summary.json
- fpocket_discovered_manifest.csv.report.md
- results/manifest_run_summary.json（使用 --run 时）

prepare_cd38_p2rank_panel.py:
- p2rank_discovered_manifest.csv
- p2rank_discovered_manifest.csv.summary.json
- p2rank_discovered_manifest.csv.report.md
- results/manifest_run_summary.json（使用 --run 时）

refresh_cd38_benchmark_readiness.py:
- cd38_benchmark_readiness_summary.json
- cd38_benchmark_readiness_commands.json
- cd38_benchmark_readiness_report.md

build_cd38_benchmark_expansion_plan.py:
- cd38_benchmark_expansion_plan.csv
- cd38_benchmark_missing_actions.csv
- cd38_benchmark_expansion_summary.json
- cd38_benchmark_expansion_plan.md

inspect_cd38_ligand_candidates.py:
- cd38_ligand_candidates.csv
- cd38_recommended_ligand_candidates.csv
- cd38_structure_targets_suggested.csv
- cd38_ligand_candidate_summary.json
- cd38_ligand_candidate_report.md

summarize_rule_ml_improvement.py:
- calibration_improvement_metrics.csv
- calibration_improvement_summary.json
- calibration_improvement_report.md

optimize_calibration_strategy.py:
- strategy_sweep_results.csv
- recommended_strategy.json
- recommended_strategy_report.md

benchmark_pose_pipeline.py:
- fold_metrics.csv
- pose_cv_predictions.csv
- nanobody_benchmark_table.csv
- pose_reliability_curve.csv
- nanobody_reliability_curve.csv
- geometry_proxy_benchmark.csv
- benchmark_summary.json
- benchmark_report.md

CD38 pocket benchmark:
- benchmarks/cd38/results/*/cd38_pocket_accuracy_summary.json
- benchmarks/cd38/results/*/cd38_pocket_accuracy_report.md
- benchmarks/cd38/results/*/predicted_pocket.txt
- benchmarks/cd38/results/cd38_benchmark_panel.csv
- benchmarks/cd38/results/cd38_benchmark_panel.md
- benchmarks/cd38/results/manifest_run_summary.json

compare_pocket_method_consensus.py:
- consensus_pocket_residues.txt
- union_pocket_residues.txt
- residue_method_membership.csv
- method_specific_residues.csv
- method_overlap_matrix.csv
- pocket_method_consensus_summary.json
- pocket_method_consensus_report.md

export_structure_annotations.py:
- residue_annotations.csv
- residue_annotations.json
- interface_residues.csv
- interface_pairs.csv
- key_residues.csv
- structure_annotation_summary.json
- pocket_payload.json
- blocking_summary.json
- analysis_bundle.json

run_recommended_pipeline.py:
- recommended_pipeline_summary.json
- recommended_pipeline_report.md
- rule_outputs/*
- model_outputs/*
- ml_ranking_outputs/*
- consensus_outputs/*
- parameter_sensitivity/*
- candidate_report_cards/*
- experiment_suggestions/*
- comparison_rule_vs_ml/*
- calibration_outputs/*
- comparison_calibrated_rule_vs_ml/*
- improvement_summary/*
- strategy_optimization/*

pipeline_smoke_test.py:
- input/pose_features.csv
- smoke_test_summary.json
- smoke_test_report.md
- rule_outputs/*
- model_outputs/*
- ml_ranking_outputs/*
- consensus_outputs/*
- parameter_sensitivity/*
- candidate_report_cards/*
- experiment_suggestions/*
- calibration_outputs/*
- comparison_rule_vs_ml/*
- comparison_calibrated_rule_vs_ml/*
- improvement_summary/*
- strategy_optimization/*

---

## 6. explanation 字段生成逻辑

在 rank_nanobodies.py 与 rule_ranker.py 中，解释文本由真实聚合指标自动拼接，典型包含:
- 跨构象 pocket 命中是否稳定
- catalytic 覆盖是否高
- mouth occlusion 是否高
- 与 ligand template 的冲突迹象
- 跨构象波动是否低
- 是否存在高置信最佳构象
- pocket_shape_overwide_proxy 是否偏高，若偏高会提示复核 pocket 边界
- 如果启用 `--pocket_overwide_penalty_weight`，输出中会保留 penalty 前分数、penalty 强度和实际权重，方便追踪排名变化原因

注意：`pocket_shape_overwide_proxy` 默认只进入 QC / explanation，不直接改变 `final_score` 或 `final_rule_score`。只有显式设置 `--pocket_overwide_penalty_weight` 大于 0 时，才会作为轻量惩罚项进入聚合分数。

---

## 7. 依赖与环境

推荐:
- Python 3.13
- numpy
- pandas
- biopython
- torch
- streamlit

安装示例:

pip install -r requirements.txt

当前仓库已锁定版本（见 `requirements.txt`）:
- numpy==2.4.2
- pandas==3.0.2
- biopython==1.86
- torch==2.8.0（Python < 3.14）
- torch==2.9.0（Python >= 3.14）
- streamlit==1.56.0

说明:
- 规则路线不依赖 torch
- ML 路线需要 torch
- 当前入口已补运行时依赖预检：缺少 `torch` / `biopython` / `streamlit` 时，会在启动前或运行前直接提示
- 若环境缺少 pandas，训练/表格流程会在运行时报错

CI 烟测:
- 已提供 `.github/workflows/smoke-test.yml`
- 在 push / pull request 时自动执行 `pipeline_smoke_test.py`

---

## 8. 设计原则（当前实现）

- 保留接口稳定性，采用增量增强而非重写
- 单行失败不拖垮全流程
- 优先可解释与可追溯（状态列、调试摘要、解释字段）
- 对缺失输入做稳健退化
- 聚合逻辑围绕“口袋阻断”目标设计，而非仅做打分平均

---

## 9. 按建议顺序执行后续完善

1) 跑完整 smoke test（自动生成可复现数据）
python pipeline_smoke_test.py --out_dir smoke_test_outputs

2) 如果你有真实 label，做规则权重校准
python calibrate_rule_ranker.py --feature_csv pose_features.csv --label_col label --out_dir calibration_outputs

3) 生成规则版 vs ML 版对照报表
python compare_rule_ml_rankings.py --rule_csv rule_outputs/nanobody_rule_ranking.csv --ml_csv ranking_outputs/nanobody_ranking.csv --out_dir comparison_outputs --pose_feature_csv pose_features.csv --label_col label

4) 生成 Rule + ML 共识排名
python build_consensus_ranking.py --rule_csv rule_outputs/nanobody_rule_ranking.csv --ml_csv ranking_outputs/nanobody_ranking.csv --feature_csv pose_features.csv --out_dir consensus_outputs

5) 生成候选报告卡
python build_candidate_report_cards.py --consensus_csv consensus_outputs/consensus_ranking.csv --rule_csv rule_outputs/nanobody_rule_ranking.csv --ml_csv ranking_outputs/nanobody_ranking.csv --feature_csv pose_features.csv --pose_predictions_csv model_outputs/pose_predictions.csv --out_dir candidate_report_cards

6) 生成下一轮实验建议
python suggest_next_experiments.py --consensus_csv consensus_outputs/consensus_ranking.csv --out_dir experiment_suggestions

---

## 10. 最小可执行示例（原流程）

1) 生成特征
python build_feature_table.py --input_csv input_pose_table.csv --out_csv pose_features.csv

2) 跑规则版
python rule_ranker.py --feature_csv pose_features.csv --out_dir rule_outputs

3) 跑 ML
python train_pose_model.py --feature_csv pose_features.csv --out_dir model_outputs
python rank_nanobodies.py --pred_csv model_outputs/pose_predictions.csv --out_dir ranking_outputs

---

如需扩展:
- 可继续加入更多几何 proxy 或能量项
- 可在聚合阶段增加任务特定一致性指标
- 可把规则版和 ML 版做并排评估，用于特征质量回归检查

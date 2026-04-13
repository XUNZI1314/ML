# Minimal ML Test: Nanobody Pocket-Blocking Pipeline

本仓库用于做 VHH/Nanobody 的口袋阻断筛选，包含两条完整路线：

1. 规则路线（不依赖 ML）
2. ML 路线（伪标签 + MLP）

目标是先验证特征工程是否合理，再做模型训练与排序。

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
- 支持 A:45、A 45、A:45,67、插入码等
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
- 局部占据代理
- delta_pocket_access_proxy
- delta_pocket_occupancy_proxy
- pocket_block_volume_proxy
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

### 1.9 后续完善工具链（校准 + 对照 + 烟测）

文件: calibrate_rule_ranker.py / compare_rule_ml_rankings.py / pipeline_smoke_test.py

已实现能力:
- 规则权重自动校准（需 label）
- 特征权重随机/扰动搜索 + 聚合权重网格搜索
- 同时评估 pose-level AUC 与 nanobody-level AUC
- 输出可复用的 calibrated_rule_config.json
- 规则版 vs ML 版并排评估
- score/rank 相关性（Spearman/Kendall）
- Top-K overlap 与差异样本表
- 一键端到端 smoke test（自动造可复现 synthetic pose_features）
- 自动串联 rule -> ML -> calibration -> comparison
- 输出 smoke_test_summary.json
- 输出 smoke_test_report.md 便于快速审阅本轮回归
- 新增推荐执行入口 `run_recommended_pipeline.py`
- 支持从 `input_csv` 或已有 `pose_features.csv` 启动
- 自动识别是否有足够 `label` 来继续 compare/calibrate/strategy optimize
- 输出 `recommended_pipeline_report.md`，汇总本次执行了哪些步骤、关键指标和关键产物
- 新增本地交互入口 [local_ml_app.py](local_ml_app.py)
- 支持通过本地页面上传或直接指定本机文件路径
- 支持在页面中填写常用参数、执行推荐流程、预览排名和下载关键产物
- 新增 [export_structure_annotations.py](export_structure_annotations.py)
- 支持对单个复合物导出 residue/interface 的 viewer-friendly JSON/CSV
- 输出 `analysis_bundle.json` 作为后续展示层的基础数据契约
- 支持通过 `--key_residues` / `--key_residue_file` 标记用户关键残基
- 输出 `interface_pairs.csv` 与 `key_residues.csv` 便于做 viewer 联动
- 输出 `pocket_payload.json` 与 `blocking_summary.json`，便于直接做 pocket 面板和阻断解释

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
- --min_nanobody_auc --min_rank_consistency (可选约束)

### 4.6 规则版 vs ML 版对照评估

python compare_rule_ml_rankings.py --rule_csv rule_outputs/nanobody_rule_ranking.csv --ml_csv ranking_outputs/nanobody_ranking.csv --out_dir comparison_outputs --pose_feature_csv pose_features.csv --label_col label

常用参数:
- --topk_list
- --pose_feature_csv
- --label_col

### 4.7 一键端到端 smoke test（无真实数据时推荐）

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

### 4.8 校准前后改进汇总

python summarize_rule_ml_improvement.py --baseline_summary comparison_rule_vs_ml/ranking_comparison_summary.json --calibrated_summary comparison_calibrated_rule_vs_ml/ranking_comparison_summary.json --calibrated_config calibration_outputs/calibrated_rule_config.json --out_dir improvement_summary

常用参数:
- --baseline_summary
- --calibrated_summary
- --calibrated_config
- --out_dir

### 4.9 校准策略自动搜索（基于已生成 trial）

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

### 4.10 推荐下一步统一入口

如果你的目标是直接执行 `not_perfect.md` 里建议的下一步，而不是手工串多条命令，优先使用：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir recommended_pipeline_outputs
```

如果你已经有 `pose_features.csv`，可直接从特征表启动：

```bash
python run_recommended_pipeline.py --feature_csv pose_features.csv --out_dir recommended_pipeline_outputs
```

特点:
- 自动执行 rule + ML 主链路
- 自动判断是否有足够 `label`
- 有标签时自动继续 compare/calibrate/strategy optimize
- 输出 `recommended_pipeline_summary.json` 便于回溯
- 输出 `recommended_pipeline_report.md` 便于先快速确认执行路径、关键指标和关键产物

### 4.11 导出结构标注与 interface residue

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

### 4.12 本地交互式运行界面

如果你的目标不是手工敲多条命令，而是更直观地上传文件、填写参数、查看结果，优先使用本地交互入口：

```bash
python -m streamlit run local_ml_app.py
```

Windows 下也可以直接双击：

```bash
start_local_app.bat
```

特点:
- 不改现有 `build_feature_table.py` / `rule_ranker.py` / `train_pose_model.py` / `rank_nanobodies.py` 主链路
- 页面里可选择从 `input_csv` 或 `pose_features.csv` 启动
- 支持上传文件，也支持直接填写本机文件路径
- 支持填写常用参数：
  - `top_k`
  - `train_epochs`
  - `train_batch_size`
  - `train_val_ratio`
  - `seed`
- 直接调用 [run_recommended_pipeline.py](run_recommended_pipeline.py)
- 运行后可在页面中预览：
  - `nanobody_ranking.csv`
  - `pose_predictions.csv`
  - `recommended_pipeline_report.md`
  - `recommended_pipeline_summary.json`
- 关键输出会写到：
  - `local_app_runs/<run_name>/outputs`

说明:
- 这是一层“本地交互式 UI 壳”，不是重写后的新后端。
- 当前更适合本机运行和比赛演示；后续如需要，再进一步打包成桌面程序。

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

summarize_rule_ml_improvement.py:
- calibration_improvement_metrics.csv
- calibration_improvement_summary.json
- calibration_improvement_report.md

optimize_calibration_strategy.py:
- strategy_sweep_results.csv
- recommended_strategy.json
- recommended_strategy_report.md

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
- torch==2.8.0
- streamlit==1.56.0

说明:
- 规则路线不依赖 torch
- ML 路线需要 torch
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

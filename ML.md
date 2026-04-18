# ML 架构导览

这份文档解释当前项目里的“ML”到底是什么、数据怎么流动、模型怎么训练、排序结果怎么来，以及应该从哪些文件继续看源码。

相关文档：

| 文档 | 作用 |
|---|---|
| [README.md](README.md) | 项目总览和完整功能清单 |
| [MODEL_QUICKSTART.md](MODEL_QUICKSTART.md) | 最短命令行使用方式 |
| [RESULT_TREE_STANDARD.md](RESULT_TREE_STANDARD.md) | 标准 `A/result/vhh/CD38_x/pose/pose.pdb` 批量目录格式 |
| [run.md](run.md) | 本地软件打开方式 |
| [advantage.md](advantage.md) | 项目定位、优势和短板 |
| [not_perfect.md](not_perfect.md) | 后续推进项和未收尾问题 |

当前软件的 ML 不是 3D viewer，也不是端到端深度学习结构模型。它更准确的定位是：

> 一个面向“纳米抗体是否可能阻断蛋白 pocket / 功能区域”的本地 tabular ML 筛选系统。

它把 PDB 结构、口袋/功能位点定义、可选 docking/打分列转成一张 `pose_features.csv`，再用规则评分和轻量 MLP 共同生成候选纳米抗体排序、解释、风险提示和下一步实验建议。

## 1. 总体架构

```text
输入数据
  |
  |-- A/result/<nanobody_id>/<CD38_variant>/<pose_id>/<pose_id>.pdb
  |     |  optional: local app auto import, build_input_from_result_tree.py,
  |     |            build_pose_features_from_result_tree.py
  |     v
  |-- input_pose_table.csv
  |-- PDB complex files
  |-- pocket / catalytic / ligand template files
  |-- optional pocket evidence layer:
  |     build_pocket_evidence.py / build_project_pocket_evidence.py
  |     -> candidate_curated_pocket.txt, review_residues.txt, audit CSVs
  |-- optional numeric scores: hdock_score, mmgbsa, interface_dg, iptm, plddt...
  v
build_feature_table.py
  |
  |-- pdb_parser.py
  |-- pocket_io.py
  |-- geometry_features.py
  v
pose_features.csv
  |
  +--> build_quality_gate.py
  +--> build_geometry_proxy_audit.py
  |
  +--> rule_ranker.py
  |       v
  |     nanobody_rule_ranking.csv
  |
  +--> train_pose_model.py
  |       v
  |     pose_predictions.csv
  |       |
  |       v
  |     rank_nanobodies.py
  |       v
  |     nanobody_ranking.csv
  |
  v
build_consensus_ranking.py
  |
  v
consensus_ranking.csv
  |
  +--> build_score_explanation_cards.py
  +--> build_candidate_report_cards.py
  +--> build_candidate_comparisons.py
  +--> suggest_next_experiments.py
  +--> build_run_provenance.py
```

主入口是 `run_recommended_pipeline.py`。本地软件 `local_ml_app.py` 本质上是把这个入口包装成更容易上传文件、点按钮、查看结果的交互界面。

## 2. 两种启动方式

当前推荐 pipeline 支持两种起点。

| 起点 | 输入 | 会不会重新算几何特征 | 适合场景 |
|---|---|---:|---|
| `--input_csv` | 原始 pose 表 + PDB 路径 | 会 | 第一次跑真实数据 |
| `--feature_csv` | 已有 `pose_features.csv` | 不会 | 已经算好特征，只想重跑 ML/排序 |

最完整的命令形态：

```bash
python run_recommended_pipeline.py ^
  --input_csv input_pose_table.csv ^
  --out_dir recommended_pipeline_outputs ^
  --default_pocket_file pocket.txt ^
  --default_catalytic_file catalytic.txt ^
  --default_ligand_file ligand_template.pdb ^
  --default_antigen_chain B ^
  --default_nanobody_chain A
```

在当前项目约定里，所有运行默认链角色都是 `antigen_chain=B`、`nanobody_chain=A`；命令里显式写出来只是为了可读性，省略时也是这个默认值。

如果已有特征表：

```bash
python run_recommended_pipeline.py ^
  --feature_csv pose_features.csv ^
  --out_dir recommended_pipeline_outputs
```

## 3. 输入层

如果真实数据已经按 `A/result/vhh1/CD38_1/1/1.pdb` 这种目录保存，可以在本地软件导入父目录 `A/`。软件会先找 `pose_features.csv`，没有时找 `input_pose_table.csv`，两者都没有时自动定位 `A/result/` 并生成输入表。

命令行也可以在 `A/` 目录使用标准目录转换器：

```bash
python build_input_from_result_tree.py --result_root . --out_csv input_pose_table.csv
```

详细目录约定见 [RESULT_TREE_STANDARD.md](RESULT_TREE_STANDARD.md)。转换后仍然走同一条 `input_pose_table.csv -> pose_features.csv -> Rule/ML/Consensus` 主链路。

在这个标准目录下，`top_k` 的准确含义是：每个 `vhh/CD38_i/` 文件夹内按 `MMPBSA_energy` 选最低的 K 个 pose。`MMPBSA_energy` 可以由 `FINAL_RESULTS_MMPBSA.dat` 自动解析，也可以提前整理到 `pose_features.csv`。如果没有能量列，才回退到 Rule/ML 分数最高的 K 个 pose。

### 3.1 最小输入表

`build_feature_table.py` 要求最少有这些列：

| 列名 | 含义 |
|---|---|
| `nanobody_id` | 候选纳米抗体 ID |
| `conformer_id` | 同一纳米抗体的构象 ID |
| `pose_id` | docking pose 或复合物 pose ID |
| `pdb_path` | 该 pose 对应的 PDB 复合物路径 |

可选列：

| 列名 | 作用 |
|---|---|
| `antigen_chain` | 抗原链 ID |
| `nanobody_chain` | 纳米抗体链 ID |
| `pocket_file` | pocket residue 定义 |
| `catalytic_file` | catalytic / function residue 定义 |
| `ligand_file` | ligand template PDB |
| `hdock_score` | 可选 docking 分数，越低通常越好 |
| `mmgbsa` | 可选能量项，越低通常越好 |
| `interface_dg` | 可选界面能量项 |
| `buried_sasa` | 可选界面埋藏面积 |
| `iptm`, `pae`, `plddt` | 可选 AlphaFold / 结构可信度特征 |
| `label` | 可选真实标签，存在时用于监督训练、校准和 benchmark |
| `MMPBSA_energy` | 可选；越低越好。存在时 Top-K pose 选择优先按它升序选择 |

### 3.2 pocket 文本格式

`pocket_io.py` 负责读取 residue 定义。当前支持链名和残基编号，例如：

```text
A:45
A 45
A:45,67
A:37-40
B:125
C:189-193
```

其中 `A:37-40` 等价于 `A:37,38,39,40`。链名不局限于 A，也可以是 B、C 等。

`catalytic_file` 现在还有一个额外作用：如果文献、M-CSA、UniProt 或 PDBj 只给了酶的关键催化/功能残基，而没有直接给 pocket 位置，主程序会用这些残基作为 3D anchor，在 PDB 结构中生成 4A、6A、8A 的 catalytic-anchor shell pocket 诊断。这个逻辑不会替代 `pocket_file`，而是输出一组可复核的 `catalytic_anchor_*` 特征列。

### 3.3 pocket 证据构建层

如果目标是提高 pocket 定义精度，可以先运行 `build_pocket_evidence.py`，再把输出的 `candidate_curated_pocket.txt` 作为下一轮 `pocket_file`。它不训练模型，也不改变排名权重，只做 residue-level evidence fusion。

支持的证据来源：

| 来源 | 输入参数 | 作用 |
|---|---|---|
| 人工 / rsite pocket | `--manual_pocket_file` | 最高优先级的人工口袋定义 |
| 文献功能位点 | `--literature_file` | 文献、数据库整理出的功能/口袋 residue |
| 催化锚点 | `--catalytic_file` | 生成 catalytic core 和 4A/6A/8A 结构邻域 shell |
| ligand/template | `--ligand_file` | 用同坐标系 ligand 附近 residue 作为结构证据 |
| P2Rank | `--p2rank_file` | 读取 P2Rank `predictions.csv` 或 residue list |
| fpocket | `--fpocket_file` | 读取 fpocket `pocket*_atm.pdb`、目录或 residue list |
| AI prior | `--ai_pocket_file` | 只作为辅助 residue prior，不视为 ground truth |

可选审计输入：

| 审计输入 | 参数 | 作用 |
|---|---|---|
| literature 来源表 | `--literature_source_table` | 保留 paper、PMID、DOI、UniProt、M-CSA、来源句子、人工确认状态 |
| catalytic 来源表 | `--catalytic_source_table` | 追溯催化/功能锚点来源，避免把宽 pocket 文件误当 catalytic core |
| AI prior 来源表 | `--ai_source_table` / `--ai_prior_source_table` | 离线记录 AI 抽取来源句子、证据等级、模型/提示版本和人工复核状态 |

主要输出：

| 文件 | 含义 |
|---|---|
| `pocket_evidence.csv` | 每条 residue 证据明细 |
| `pocket_residue_support.csv` | residue 聚合支持分、来源数和复核原因 |
| `candidate_curated_pocket.txt` | 可作为下一轮 `pocket_file` 的候选口袋 |
| `review_residues.txt` | 只有低/单一证据支持、需要人工复核的 residue |
| `evidence_source_audit.csv` | literature/catalytic/AI 来源追溯审计 |
| `evidence_source_template.csv` | 下一轮手工补来源字段的模板 |
| `ai_prior_audit.csv` | AI prior 专用审计表 |
| `ai_prior_template.csv` | AI prior 专用补填模板 |
| `POCKET_EVIDENCE_REPORT.md` | 人可读证据报告 |

当前这个层有两个安全边界：

- P2Rank/fpocket 过宽时会启用 external precision guard。缺少 manual/literature/catalytic core/ligand-contact 高置信支持的边缘 residue 会标记 `external_overwide_guard`，进入 review，而不是直接进入 curated pocket。
- AI prior 只作为待复核线索。它不参与 curated 判定的支持分/方法数，也不能和外部工具一起直接成为 ground truth。人工确认后应转写到 `manual_pocket_file` 或 `literature_file`。

这个层的定位是“提高 pocket 输入质量”，不是替代 `build_feature_table.py`。正式排序仍然从 `pose_features.csv` 开始。

## 4. 特征工程层

核心文件：

| 文件 | 职责 |
|---|---|
| `build_feature_table.py` | 逐行读取 pose，调用结构解析和几何特征计算，输出 `pose_features.csv` |
| `pdb_parser.py` | 读取 PDB、拆分 antigen / nanobody |
| `pocket_io.py` | 读取 pocket、catalytic、ligand template |
| `geometry_features.py` | 计算几何 proxy 特征 |

`build_feature_table.py` 的一行输入对应一个 pose。每个 pose 会被转换成一行特征。

主要几何特征包括：

| 特征 | 直观含义 |
|---|---|
| `pocket_hit_fraction` | 纳米抗体是否接触 pocket residues |
| `catalytic_hit_fraction` | 是否接触功能/催化 residues |
| `catalytic_anchor_primary_shell_hit_fraction` | 是否接触催化残基 3D 邻域推断出的主 shell pocket |
| `catalytic_anchor_min_distance_to_primary_shell` | 到 catalytic-anchor 主 shell 的最近距离 |
| `catalytic_anchor_manual_overlap_fraction_of_shell` | catalytic-anchor shell 与人工 pocket 的重叠比例 |
| `catalytic_anchor_shell_overwide_proxy` | catalytic-anchor shell 是否可能过宽 |
| `min_distance_to_pocket` | 到 pocket 最近距离 |
| `mouth_occlusion_score` | 是否遮挡口袋口部 |
| `mouth_axis_block_fraction` | 是否沿口袋入口轴向阻断 |
| `mouth_aperture_block_fraction` | 是否覆盖口袋开口区域 |
| `mouth_min_clearance` | 口袋开口处最小剩余空间 |
| `substrate_overlap_score` | 是否与 ligand template 空间冲突 |
| `ligand_path_block_score` | 是否阻断 ligand 进入路径 |
| `ligand_path_block_fraction` | ligand 路径被阻断比例 |
| `ligand_path_bottleneck_score` | 是否形成路径瓶颈 |
| `ligand_path_exit_block_fraction` | 是否阻断路径出口 |
| `ligand_path_min_clearance` | ligand 路径最小净空 |
| `delta_pocket_occupancy_proxy` | pocket 占据变化 proxy |
| `pocket_shape_overwide_proxy` | pocket 定义是否可能过宽 |
| `pocket_shape_tightness_proxy` | pocket 定义是否较集中 |

这些都是静态结构 proxy，不是分子动力学，也不是严格自由能计算。它们的目标是把“可能阻断口袋”的几何证据量化成机器可读列。

catalytic-anchor 模块的边界需要单独记住：催化残基本身不等于完整口袋，它只是高可信空间锚点。真正用于判断时，应同时看人工 `pocket_file`、文献位点、3D shell 大小、overwide 风险和与 VHH 接触的方向是否一致。

## 5. Rule baseline

核心文件：`rule_ranker.py`

Rule baseline 是一个非 ML 的可解释评分器，用来做两件事：

1. 给 ML 一个强基线，避免模型结果完全不可解释。
2. 在标签不足时，仍然能生成可用排序。

### 5.1 Pose 级规则分数

`rule_ranker.py` 先对每个 pose 计算 `rule_blocking_score`。

公式近似为：

```text
aligned_feature = robust_minmax(feature)          如果该特征越大越好
aligned_feature = 1 - robust_minmax(feature)      如果该特征越小越好

rule_blocking_score =
  sum(weight_i * aligned_feature_i) / sum(weight_i for valid features)
```

默认参与规则评分的特征包括：

```text
pocket_hit_fraction
catalytic_hit_fraction
mouth_occlusion_score
mouth_axis_block_fraction
mouth_aperture_block_fraction
mouth_min_clearance
delta_pocket_occupancy_proxy
substrate_overlap_score
ligand_path_block_score
ligand_path_block_fraction
ligand_path_bottleneck_score
ligand_path_exit_block_fraction
ligand_path_min_clearance
min_distance_to_pocket
rsite_accuracy
MMPBSA_energy
mmgbsa
interface_dg
```

### 5.2 Pose -> conformer -> nanobody

Rule baseline 不直接只看单个 pose，而是分层聚合：

```text
pose_rule_scores.csv
  -> conformer_rule_scores.csv
  -> nanobody_rule_ranking.csv
```

conformer 级：

```text
conformer_rule_score =
  (1 - w_geo) * (0.7 * mean_topk_rule + 0.3 * best_pose_rule)
  + w_geo * geo_aux_score
```

nanobody 级：

```text
final_rule_score =
  w_mean        * mean_conformer_rule_score
  + w_best      * best_conformer_rule_score
  + w_consistency * pocket_consistency_score
  - w_std       * std_conformer_rule_score
```

默认权重：

| 权重 | 默认值 |
|---|---:|
| `w_mean` | 0.50 |
| `w_best` | 0.25 |
| `w_consistency` | 0.20 |
| `w_std_penalty` | 0.15 |

解释：既看平均表现，也看最佳构象，同时奖励跨构象稳定，惩罚构象间波动。

## 6. ML 模型核心

核心文件：`train_pose_model.py`

当前 ML 模型是一个轻量 tabular MLP，不是序列模型，也不是图神经网络。

### 6.1 输入

模型输入是 `pose_features.csv` 中自动筛选出的数值列。

会排除：

```text
nanobody_id
conformer_id
pose_id
pdb_path
pocket_file
catalytic_file
ligand_file
status
error_message
warning_message
label / pseudo_label / pseudo_score 等目标列
```

特征筛选逻辑：

| 规则 | 默认 |
|---|---:|
| 最小非空比例 | `0.05` |
| 最大缺失比例 | `0.80` |
| 近似常数过滤阈值 | `0.995` |

数值预处理：

```text
1. 缺失值用训练集 median 填充
2. 用训练集 mean/std 做标准化
3. 标准化参数写入 feature_columns.json
```

### 6.2 标签模式

模型支持两种训练模式。

| 模式 | 触发条件 | 含义 |
|---|---|---|
| `real_label` | `pose_features.csv` 有可用 `label` 列 | 用真实标签训练 |
| `pseudo_label` | 没有真实标签 | 用几何/能量 proxy 生成伪标签训练 |

如果没有真实标签，会用这些特征生成 `pseudo_score`：

```text
pocket_hit_fraction
catalytic_hit_fraction
mouth_occlusion_score
mouth_axis_block_fraction
mouth_aperture_block_fraction
mouth_min_clearance
delta_pocket_occupancy_proxy
substrate_overlap_score
ligand_path_block_score
ligand_path_block_fraction
ligand_path_bottleneck_score
ligand_path_exit_block_fraction
ligand_path_min_clearance
min_distance_to_pocket
rsite_accuracy
hdock_score
MMPBSA_energy
mmgbsa
interface_dg
```

伪标签流程：

```text
1. 每个特征按方向对齐，越符合阻断越高
2. robust min-max 缩放到 0~1
3. 加权平均得到 pseudo_score
4. 默认取 top_fraction=0.25 的 pose 作为 pseudo_label=1
```

这意味着：没有真实实验标签时，ML 不是凭空学出生物真相，而是在学习当前几何 proxy 的非线性组合。这个模式适合排序和展示，不应被描述成已完成真实实验验证。

### 6.3 网络结构

类名：`PoseMLP`

默认结构：

```text
input_dim = 自动筛选出的数值特征数

Linear(input_dim -> 128)
ReLU
Dropout(0.20)
Linear(128 -> 64)
ReLU
Dropout(0.20)
Linear(64 -> 1)
sigmoid -> pred_prob
```

训练时实际输出是 logit，推理时用 sigmoid 转成 `pred_prob`。

### 6.4 训练目标

训练主损失：

```text
BCEWithLogitsLoss(pos_weight)
```

如果存在 `pseudo_score`，还会加一个软目标辅助损失：

```text
loss =
  BCEWithLogitsLoss(logit, hard_label)
  + soft_target_weight * MSE(sigmoid(logit), pseudo_score)
```

默认 `soft_target_weight = 0.25`。

优化器和训练策略：

| 项 | 默认 |
|---|---:|
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| batch size | `64` |
| direct train epochs | `80` |
| recommended pipeline epochs | `20` |
| early stopping patience | direct train `12`，pipeline `8` |
| LR scheduler | ReduceLROnPlateau |
| gradient clipping | `1.0` |
| train/val split | 按 `nanobody_id` 分组切分，减少同一候选泄漏 |

输出：

| 文件 | 含义 |
|---|---|
| `best_model.pt` | PyTorch checkpoint |
| `pose_predictions.csv` | 每个 pose 的 `pred_prob` 和贡献特征摘要 |
| `train_log.csv` | 每轮 loss/AUC/accuracy/precision/recall/F1 |
| `feature_columns.json` | 特征列、填充值、mean/std |
| `training_summary.json` | 训练摘要 |

## 7. ML 排序聚合

核心文件：`rank_nanobodies.py`

`train_pose_model.py` 只预测 pose 级概率：

```text
pose_id -> pred_prob
```

但最终用户关心的是：

```text
nanobody_id -> 是否值得优先做实验
```

所以需要两级聚合。

### 7.1 Pose -> conformer

对每个 `(nanobody_id, conformer_id)`：

```text
1. 如果存在 MMPBSA_energy / mmgbsa，按能量从低到高排序
2. 没有能量列时，才按 pred_prob 从高到低排序
3. 取 top_k 个 pose，默认 top_k=3
4. 计算 mean_topk_pred_prob、mean_topk_MMPBSA_energy 和 best_pose_prob
5. 可选融合 geo_aux_score
```

conformer 分数：

```text
w_geo = 0.15，如果有足够几何辅助信号
pred_mass = 1 - w_geo

conformer_score_raw =
  0.70 * pred_mass * mean_topk_pred_prob
  + 0.30 * pred_mass * best_pose_prob
  + w_geo * geo_aux_score
```

如果显式启用 pocket overwide penalty：

```text
conformer_score =
  conformer_score_raw - pocket_overwide_penalty_weight * overwide_penalty
```

当前默认 `pocket_overwide_penalty_weight = 0.0`，也就是只提示风险，不默认扣分。

### 7.2 Conformer -> nanobody

最终纳米抗体分数：

```text
final_score =
  w_mean        * mean_conformer_score
  + w_best      * best_conformer_score
  + w_consistency * pocket_consistency_score
  - w_std       * std_conformer_score
```

默认权重：

| 权重 | 默认值 |
|---|---:|
| `w_mean` | 0.50 |
| `w_best` | 0.25 |
| `w_consistency` | 0.20 |
| `w_std_penalty` | 0.15 |

输出：

| 文件 | 含义 |
|---|---|
| `conformer_scores.csv` | 构象级分数 |
| `nanobody_ranking.csv` | ML 路线的纳米抗体最终排序 |

## 8. Rule + ML 共识层

核心文件：`build_consensus_ranking.py`

这个层不是新训练模型，而是决策支持层。它把两条路线合并：

```text
Rule route: nanobody_rule_ranking.csv
ML route:   nanobody_ranking.csv
```

共识层会输出：

| 字段 | 含义 |
|---|---|
| `consensus_score` | Rule 和 ML 的合并排序分数 |
| `confidence_score` | 两条路线一致性、QC、风险综合可信度 |
| `decision_tier` | `priority` / `review` / `standard` |
| `risk_flags` | 低可信原因，如 `rule_ml_disagreement`、`pocket_overwide` |
| `review_reason` | 人可读复核理由 |

它的意义是：不盲信 ML，也不只看规则，而是让两条不同偏差的路线互相制衡。

## 9. 质量控制和风险解释

核心文件：

| 文件 | 职责 |
|---|---|
| `build_quality_gate.py` | 对输入、特征、标签覆盖给出 PASS/WARN/FAIL |
| `build_geometry_proxy_audit.py` | 检查 mouth/path/pocket/contact proxy 是否自洽 |
| `build_score_explanation_cards.py` | 给每个候选生成分数解释卡 |
| `build_candidate_report_cards.py` | 生成 HTML 候选报告卡 |
| `build_candidate_comparisons.py` | 做候选横向对比 |
| `analyze_ranking_parameter_sensitivity.py` | 看排序对参数是否敏感 |
| `build_validation_evidence_audit.py` | 检查 top 候选真实验证证据覆盖 |

这些模块不会改变主分数，主要负责回答：

```text
为什么这个候选排前？
这个结果可信度怎样？
是不是 pocket 定义过宽？
Rule 和 ML 是否一致？
是否需要人工复核？
下一步实验应该先做谁？
```

## 10. 真实标签、校准和 benchmark

如果 `pose_features.csv` 有足够真实 `label`，pipeline 会自动启用更多监督步骤。

### 10.1 Rule vs ML 对照

文件：`compare_rule_ml_rankings.py`

作用：

```text
比较 Rule 排名和 ML 排名：
- Spearman 相关
- Top-K overlap
- 与真实 label 的关系
- 差异候选列表
```

### 10.2 Rule 权重校准

文件：`calibrate_rule_ranker.py`

它搜索两类参数：

```text
1. pose 级 rule feature weights
2. conformer/nanobody 聚合权重
```

目标可以同时考虑：

```text
pose-level AUC
nanobody-level AUC
与 ML ranking 的一致性
```

校准结果会生成：

```text
calibrated_rule_config.json
calibrated_rule_outputs/
aggregation_calibration_trials.csv
```

### 10.3 分组交叉验证 benchmark

文件：`benchmark_pose_pipeline.py`

这是更正式的评估入口。它按 `nanobody_id` 做 grouped CV，减少同一纳米抗体不同 pose 同时进入训练/验证造成的数据泄漏。

输出包括：

```text
pose_oof_predictions.csv
nanobody_benchmark_table.csv
benchmark_summary.json
benchmark_report.md
geometry_proxy_benchmark.csv
reliability_curve.csv
```

## 11. 推荐 pipeline 的输出目录结构

典型输出在 `recommended_pipeline_outputs/` 下。

| 子目录 | 主要文件 |
|---|---|
| 根目录 | `pose_features.csv`, `feature_qc.json`, `recommended_pipeline_summary.json`, `recommended_pipeline_report.md` |
| `quality_gate/` | `quality_gate_report.md`, `quality_gate_summary.json` |
| `geometry_proxy_audit/` | proxy 自洽审计 |
| `rule_outputs/` | `pose_rule_scores.csv`, `conformer_rule_scores.csv`, `nanobody_rule_ranking.csv` |
| `model_outputs/` | `best_model.pt`, `pose_predictions.csv`, `train_log.csv`, `training_summary.json` |
| `ml_ranking_outputs/` | `conformer_scores.csv`, `nanobody_ranking.csv` |
| `consensus_outputs/` | `consensus_ranking.csv`, `consensus_report.md` |
| `score_explanation_cards/` | 分数解释卡 |
| `candidate_report_cards/` | HTML 候选报告卡 |
| `candidate_comparisons/` | 候选对比 |
| `parameter_sensitivity/` | 参数敏感性 |
| `experiment_suggestions/` | 下一轮实验建议和计划单 |
| `validation_evidence_audit/` | 验证证据审计 |
| `provenance/` | 可复现记录和 artifact manifest |

最应该先看的三个文件：

```text
recommended_pipeline_summary.json
consensus_outputs/consensus_ranking.csv
batch_decision_summary/batch_decision_summary.md
```

如果想看模型训练是否正常：

```text
model_outputs/training_summary.json
model_outputs/train_log.csv
model_outputs/pose_predictions.csv
```

如果想看最终排序为什么这么排：

```text
consensus_outputs/consensus_report.md
score_explanation_cards/score_explanation_cards.html
candidate_report_cards/index.html
candidate_comparisons/candidate_comparison_report.md
```

## 12. 本地软件和 ML 的关系

`local_ml_app.py` 不是另一个 ML 模型。它是交互壳，负责：

```text
1. 上传/选择 CSV、PDB、pocket、ligand 等文件
2. 生成本地运行目录
3. 调用 run_recommended_pipeline.py
4. 展示 summary、ranking、report、cards
5. 管理历史运行、模板、导出包、demo、CD38 benchmark 诊断
```

也就是说：

```text
ML 核心逻辑在 CLI 脚本里
本地软件负责让这些 CLI 更容易运行和查看
```

这样设计的优点是：即使 UI 出问题，命令行 pipeline 仍然可以独立运行和复现。

## 13. 当前架构的优势

1. 不是纯黑盒：Rule、ML、Consensus 三条视角都能看。
2. 标签不足时也能跑：没有真实 label 时，会退回 pseudo-label + rule baseline。
3. 能处理批量 pose：同一纳米抗体可以有多个 conformer 和多个 pose。
4. 有防泄漏意识：训练/验证按 `nanobody_id` 分组。
5. 有 QC 和解释层：不是只给一个分数，还会给风险、原因和下一步实验建议。
6. 可复现：输出 summary、train log、feature config、provenance manifest。
7. 可逐步升级：以后可以替换 MLP、增加真实标签、接入更强 pocket finder，但不需要推倒当前主链路。

## 14. 当前架构的边界

这些点需要明确：

1. 当前 MLP 是 tabular MLP，不是 3D equivariant model、GNN 或分子动力学模型。
2. 如果没有真实 label，模型学习的是几何 proxy 的组合，不等于真实实验结论。
3. `pocket_shape_overwide_proxy` 默认只提示风险，不默认扣分。
4. 几何特征是静态结构近似，不能替代真实结合实验、MD、自由能计算。
5. 当前 CD38 benchmark 主要验证 pocket 覆盖和 proxy 合理性，不等价于完整纳米抗体阻断实验。

## 15. 源码阅读顺序

如果你想真正读懂代码，建议按下面顺序：

1. `run_recommended_pipeline.py`
2. `build_feature_table.py`
3. `geometry_features.py`
4. `rule_ranker.py`
5. `train_pose_model.py`
6. `rank_nanobodies.py`
7. `build_consensus_ranking.py`
8. `build_score_explanation_cards.py`
9. `suggest_next_experiments.py`
10. `benchmark_pose_pipeline.py`

如果只关心模型训练，直接看：

```text
train_pose_model.py
rank_nanobodies.py
```

如果只关心为什么最终推荐某个候选，直接看：

```text
build_consensus_ranking.py
build_score_explanation_cards.py
build_candidate_report_cards.py
```

## 16. 一句话总结

当前 ML 架构可以概括为：

> 用结构几何 proxy 构建 pose 级特征，用 Rule baseline 和轻量 MLP 分别评估 pocket blocking 可能性，再把 pose 聚合到 conformer 和 nanobody 层，最后通过 Rule+ML 共识、QC 风险和解释卡输出适合实验决策的候选排序。

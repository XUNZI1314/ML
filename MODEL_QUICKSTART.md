# 模型最简使用说明

这份文档只保留最少信息，目标是让你用最短路径跑通当前仓库里的模型流程。

## 1. 先说结论

当前仓库里，最简单的使用方式不是手工分别运行多个脚本，也不是单独加载 `best_model.pt` 做推理。

最简单可执行方式是直接运行统一入口：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs
```

如果你已经有 `pose_features.csv`，则直接运行：

```bash
python run_recommended_pipeline.py --feature_csv pose_features.csv --out_dir my_outputs
```

## 2. 最少安装

```bash
pip install -r requirements.txt
```

## 3. 最简单办法 A：从原始输入表开始

这是最推荐的方式，因为你不需要手工整理特征列。

### 3.1 最小命令

如果每一行都已经各自带了 `pocket_file`、`catalytic_file`、`ligand_file`：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs
```

如果这些文件对整批数据都相同，最简单命令可以写成：

```bash
python run_recommended_pipeline.py ^
  --input_csv input_pose_table.csv ^
  --out_dir my_outputs ^
  --default_pocket_file pocket.txt ^
  --default_catalytic_file catalytic.txt ^
  --default_ligand_file ligand.pdb
```

### 3.2 `input_pose_table.csv` 最小格式

必需列只有 4 个：

- `nanobody_id`
- `conformer_id`
- `pose_id`
- `pdb_path`

最小示例：

```csv
nanobody_id,conformer_id,pose_id,pdb_path
NB_001,CF_01,P_001,data/NB_001_CF_01_P_001.pdb
NB_001,CF_01,P_002,data/NB_001_CF_01_P_002.pdb
NB_002,CF_01,P_001,data/NB_002_CF_01_P_001.pdb
```

### 3.3 常见可选列

如果你希望每一行自己指定链或 pocket 定义，可以加这些列：

- `antigen_chain`
- `nanobody_chain`
- `pocket_file`
- `catalytic_file`
- `ligand_file`
- `label`

带可选列的示例：

```csv
nanobody_id,conformer_id,pose_id,pdb_path,antigen_chain,nanobody_chain,pocket_file,catalytic_file,ligand_file,label
NB_001,CF_01,P_001,data/NB_001_CF_01_P_001.pdb,A,H,data/pocket.txt,data/catalytic.txt,data/ligand.pdb,1
NB_001,CF_01,P_002,data/NB_001_CF_01_P_002.pdb,A,H,data/pocket.txt,data/catalytic.txt,data/ligand.pdb,0
```

说明：

- `label` 不是必需列。
- 如果没有 `label`，流程仍然会跑完，但会自动跳过 compare/calibrate 这类依赖标签的步骤。
- 如果有 `label`，并且标签里同时有正负两类，统一入口会自动多跑一套对照和校准步骤。

## 4. 最简单办法 B：你已经有 `pose_features.csv`

如果你已经用别的方法生成好了 `pose_features.csv`，可以直接从特征表启动：

```bash
python run_recommended_pipeline.py --feature_csv pose_features.csv --out_dir my_outputs
```

### 4.1 推荐做法

推荐由脚本自动生成 `pose_features.csv`，不建议手工写。

### 4.2 `pose_features.csv` 至少应包含

- `nanobody_id`
- `conformer_id`
- `pose_id`

并建议至少包含当前主流程常用的几何/打分列：

- `pocket_hit_fraction`
- `catalytic_hit_fraction`
- `mouth_occlusion_score`
- `mouth_axis_block_fraction`
- `mouth_aperture_block_fraction`
- `mouth_min_clearance`
- `delta_pocket_occupancy_proxy`
- `substrate_overlap_score`
- `ligand_path_block_score`
- `ligand_path_block_fraction`
- `ligand_path_bottleneck_score`
- `ligand_path_exit_block_fraction`
- `ligand_path_min_clearance`
- `min_distance_to_pocket`
- `rsite_accuracy`
- `mmgbsa`
- `interface_dg`

可选：

- `label`
- `status`
- `hdock_score`

### 4.3 一个真实表头示例

下面是当前仓库 smoke test 里实际生成的 `pose_features.csv` 表头：

```csv
nanobody_id,conformer_id,pose_id,status,pocket_hit_fraction,catalytic_hit_fraction,mouth_occlusion_score,mouth_axis_block_fraction,mouth_aperture_block_fraction,mouth_min_clearance,delta_pocket_occupancy_proxy,substrate_overlap_score,ligand_path_block_score,ligand_path_block_fraction,ligand_path_bottleneck_score,ligand_path_exit_block_fraction,ligand_path_min_clearance,min_distance_to_pocket,rsite_accuracy,mmgbsa,interface_dg,hdock_score,label
```

## 5. 跑完后先看什么

最先看这 3 个文件：

- `my_outputs/recommended_pipeline_report.md`
- `my_outputs/ml_ranking_outputs/nanobody_ranking.csv`
- `my_outputs/model_outputs/pose_predictions.csv`

如果有 `label`，再看：

- `my_outputs/comparison_rule_vs_ml/ranking_comparison_summary.json`
- `my_outputs/calibration_outputs/calibrated_rule_config.json`
- `my_outputs/strategy_optimization/recommended_strategy.json`

## 6. 最终输出格式长什么样

### 6.1 `pose_predictions.csv`

这是 pose 级别预测结果，常用列包括：

- `nanobody_id`
- `conformer_id`
- `pose_id`
- `pred_prob`
- `pred_logit`
- `top_contributing_features`

示例表头：

```csv
nanobody_id,conformer_id,pose_id,pred_prob,pred_logit,top_contributing_features,pocket_hit_fraction,catalytic_hit_fraction,mouth_occlusion_score,mouth_axis_block_fraction,mouth_aperture_block_fraction,mouth_min_clearance,substrate_overlap_score,ligand_path_block_score,ligand_path_block_fraction,ligand_path_bottleneck_score,ligand_path_exit_block_fraction,ligand_path_min_clearance,delta_pocket_occupancy_proxy,min_distance_to_pocket,rsite_accuracy,label,pseudo_label,pseudo_score,pseudo_rank,pseudo_components
```

### 6.2 `nanobody_ranking.csv`

这是最终排序结果，最重要的列是：

- `rank`
- `nanobody_id`
- `final_score`
- `best_conformer`
- `best_pose_id`
- `explanation`

示例表头：

```csv
rank,nanobody_id,num_conformers,best_conformer,best_pose_id,best_pose_prob,mean_conformer_score,best_conformer_score,std_conformer_score,pocket_consistency_score,final_score,explanation,w_mean,w_best,w_consistency,w_std_penalty,mean_topk_pocket_hit_fraction,mean_topk_catalytic_hit_fraction,mean_topk_mouth_occlusion_score,mean_topk_mouth_axis_block_fraction,mean_topk_mouth_aperture_block_fraction,mean_topk_mouth_min_clearance,mean_topk_substrate_overlap_score,mean_topk_ligand_path_block_score,mean_topk_ligand_path_block_fraction,mean_topk_ligand_path_bottleneck_score,mean_topk_ligand_path_exit_block_fraction,mean_topk_ligand_path_min_clearance,mean_topk_delta_pocket_occupancy_proxy,mean_topk_pocket_block_volume_proxy,mean_pocket_hit_fraction,mean_catalytic_hit_fraction,mean_mouth_occlusion_score,mean_substrate_overlap_score
```

## 7. 你只需要记住的最短版本

只有一句话：

1. 准备一个 `input_pose_table.csv`
2. 运行 `python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs`
3. 打开 `my_outputs/recommended_pipeline_report.md`
4. 看 `my_outputs/ml_ranking_outputs/nanobody_ranking.csv`

如果你已经有 `pose_features.csv`，就把上面的 `--input_csv ...` 换成 `--feature_csv pose_features.csv`。

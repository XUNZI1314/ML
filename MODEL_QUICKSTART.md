# 模型最简使用说明

这份文档只保留最少信息，目标是让你用最短路径跑通当前仓库里的模型流程。

## 0. 先判断你手里有什么

| 你现在有的文件 | 应该怎么跑 | 说明 |
|---|---|---|
| `input_pose_table.csv` 和 PDB 文件 | 用 `--input_csv` | 最推荐，会自动构建特征 |
| 标准 `A/result/vhh/CD38_x/pose/pose.pdb` 文件夹 | 在本地软件导入 `A/`，或用 `--result_root .` | 软件会自动定位 `A/result/` 并生成输入表 |
| 已经生成好的 `pose_features.csv` | 用 `--feature_csv` | 跳过特征构建，直接排名和训练 |
| 只有一堆 PDB 文件 | 先用本地软件导入目录或 zip | 软件会尝试生成 `auto_input_pose_table.csv` |
| 没有自己的数据，只想试用完整流程 | 运行 `python run_demo_pipeline.py` | 自动生成 synthetic demo 数据并跑完整推荐流程 |
| 只想打开软件点按钮 | 看 [run.md](run.md) | 这份文档只讲命令行最短路径 |
| 想理解 ML 内部架构 | 看 [ML.md](ML.md) | 解释 Rule、MLP、聚合、共识和输出文件的关系 |

## 0.0 没有数据时的一键 demo

最短命令：

```bash
python run_demo_pipeline.py
```

Windows 下也可以双击：

```text
run_demo_pipeline.bat
```

默认会生成：

- `demo_data/demo_pose_features.csv`
- `demo_data/demo_experiment_plan_override.csv`
- `demo_data/demo_manifest.json`
- `demo_outputs/DEMO_OVERVIEW.html`
- `demo_outputs/DEMO_README.md`
- `demo_outputs/DEMO_INTERPRETATION.md`
- `demo_outputs/REAL_DATA_STARTER/`
- `demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/`
- `demo_outputs/batch_decision_summary/batch_decision_summary.md`
- `demo_outputs/candidate_report_cards/index.html`

说明：demo 的验证标签来自 synthetic proxy signal，只用于演示安装、运行、报告和导出流程；不要把它当成真实实验结论。

如果你已经打开本地软件，左侧“没有数据时”有两个按钮：

- “生成并立即运行 demo”：最省事，生成 demo 输入后直接启动后台运行。
- “生成并载入 demo 输入”：先填好表单路径，适合你检查参数后再点“立即运行”。

本地软件生成的 demo 运行目录里会额外写入 `DEMO_OVERVIEW.html`、`DEMO_README.md` 和 `DEMO_INTERPRETATION.md`。优先用浏览器打开 `DEMO_OVERVIEW.html` 看展示版摘要，再用两个 Markdown 文件定位批次结论、候选报告卡、synthetic 验证证据审计，并确认 demo 结果不能被当成真实实验结论。

如果是在本地软件里运行 demo，结果完成后切到“摘要”页，看“Demo 快速导览”，可以直接打开 HTML 导览页。

如果要换成自己的数据，先打开 `demo_outputs/REAL_DATA_STARTER/README_REAL_DATA_STARTER.md`，按里面的 `input_pose_table_template.csv` 和 `real_data_checklist.csv` 改路径。

如果你想先确认“真实 PDB 输入表”这条路径能跑，不要直接从空模板开始。先运行 starter 里的 mini PDB 示例：

```bash
python build_feature_table.py --input_csv demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/input_pose_table.csv --out_csv demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/mini_pose_features.csv
```

也可以跑完整轻量流程：

```bash
python run_recommended_pipeline.py --input_csv demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/input_pose_table.csv --out_dir demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/mini_outputs --train_epochs 1 --disable_label_aware_steps
```

边界：`MINI_PDB_EXAMPLE` 是 toy 结构包，只用于检查 PDB 解析、链拆分、口袋位点映射、ligand template 和 pipeline 产物生成，不代表真实生物学结论。

## 0.1 最小输入检查清单

如果你的真实数据已经按下面结构保存，优先使用这个固定格式。推荐在 `result/` 的父目录 `A/` 里运行或导入：

```text
A/
  result/
    vhh1/
      CD38_1/
        1/
          1.pdb
          FINAL_RESULTS_MMPBSA.dat
        2/
          2.pdb
      CD38_2/
        1/
          1.pdb
```

含义固定如下：

| 目录层级 | 映射到输入表 |
|---|---|
| `vhh1` | `nanobody_id` |
| `CD38_1` | `conformer_id`、`target_variant_id` |
| `1` | `pose_id`、`pose_index` |
| `1/1.pdb` | `pdb_path` |
| `1/FINAL_RESULTS_MMPBSA.dat` | `MMPBSA_energy` |

本地软件左侧“导入目录/zip”选择 `A/` 即可。导入逻辑是：先找 `pose_features.csv`；没有时找 `input_pose_table.csv`；两者都没有时自动识别 `A/result/` 或 `A/rsite/result/` 并生成 `auto_input_pose_table.csv`。因此标准目录下不需要手动填写 `input_csv`。

如果只想一键从标准结果目录提取正式特征表，在 `A/` 目录执行：

```bash
python build_pose_features_from_result_tree.py
```

这会在当前目录生成：

```text
pose_features.csv
input_pose_table.csv
feature_qc.json
input_pose_table.report.md
```

如果存在 `rsite/rsite.txt`，会自动按默认 `antigen_chain=B` 过滤口袋残基，并把派生 pocket 写到 `.ml_auto/auto_pocket_antigen_B.txt`。同时会把 `FINAL_RESULTS_MMPBSA.dat`、`MMPBSA_normalized.txt`、`score.txt`、`*_accuracy.txt`、`*_interface.sc`、`FINAL_DECOMP_MMPBSA.dat` 解析进 `pose_features.csv`。

如果提供 `catalytic_file`，主程序现在会额外启用 catalytic-anchor pocket 诊断：以催化/功能残基为 3D anchor，在 PDB 结构里自动生成 4A、6A、8A 邻域 shell，并输出 `catalytic_anchor_*` 特征列。这个模块用于解释和复核口袋定义，不会替代人工 `pocket_file`。

如果只想先生成输入表，在 `A/` 目录执行：

```bash
python build_input_from_result_tree.py --result_root . --out_csv input_pose_table.csv
```

如果有统一 CD38 pocket / catalytic 文件，可以一起写入：

```bash
python build_input_from_result_tree.py --result_root . --out_csv input_pose_table.csv --default_pocket_file cd38_pocket.txt --default_catalytic_file cd38_catalytic.txt
```

默认链角色已经在所有项目中固定为 `antigen_chain=B`、`nanobody_chain=A`。在当前 `A/` 数据里，这对应 CD38 在 B 链、VHH 在 A 链。只有遇到特殊数据时才需要手动覆盖。

然后再跑主流程：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs
```

这里的 `top_k` 对应你的真实目录时，含义是：在每个 `vhh/CD38_i/` 文件夹下，按 `MMPBSA_energy` 从低到高选择 K 个 pose。也就是说 `top_k=3` 会从每个 `CD38_1`、`CD38_2`、`CD38_3` 的 10 个 pose 中各选 MMPBSA 最低的 3 个。若没有 `MMPBSA_energy` / `mmgbsa` 能量列，系统才会回退到模型或规则分数最高的 K 个 pose。

| 检查项 | 要求 |
|---|---|
| `nanobody_id` | 同一个纳米抗体用同一个 ID |
| `conformer_id` | 同一个纳米抗体不同构象要能区分 |
| `pose_id` | 每个 docking pose 要能唯一定位 |
| `pdb_path` | 路径必须能在当前机器上找到 PDB 文件 |
| `pocket_file` | 可选；没有时可用默认 pocket 文件 |
| `catalytic_file` | 可选；没有时可用默认 catalytic 文件 |
| `ligand_file` | 可选；有 ligand template 时几何解释更完整 |
| `label` | 可选；有正负标签时会自动启用对照、校准和策略搜索 |

## 0.2 实验计划覆盖 CSV（可选）

如果你已经知道某些候选必须做、不能做、已经完成或暂时缺样，可以额外准备一个 `experiment_plan_override.csv`：

```csv
nanobody_id,plan_override,experiment_status,experiment_result,validation_label,experiment_owner,experiment_cost,experiment_note
NB001,include,pending,,,Alice,2,must validate
NB002,exclude,blocked,,,Bob,1,no material
NB003,,completed,positive,,Carol,3,validated blocker
NB004,,completed,,0,Dan,1,manual negative label
```

运行时加上：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs --experiment_plan_override_csv experiment_plan_override.csv
```

说明：`plan_override` 支持 `include`、`exclude`、`standby`、`defer`；`experiment_status` 支持用 `completed`、`blocked`、`in_progress` 等状态回灌计划单。要进入训练标签，建议显式填写 `experiment_result=positive/negative` 或 `validation_label=1/0`。

跑完后也会自动生成 `my_outputs/experiment_suggestions/experiment_plan_state_ledger.csv`。如果你在本地软件的“本轮实验计划单”里编辑并保存，它会生成 `experiment_plan_override_edited.csv`，可直接用于下一轮。

推荐 pipeline 还会自动生成 `my_outputs/validation_evidence_audit/validation_evidence_report.md`。它只检查当前 top 候选有没有真实验证证据，不会改变模型分数；`completed` 状态不会自动变成阳性或阴性，只有 `experiment_result=positive/negative` 或 `validation_label=1/0` 才算可用验证标签。

如果你已经有多次本地软件历史运行，可以汇总最新状态：

```bash
python build_experiment_state_ledger.py --local_app_runs local_app_runs
```

生成的 `local_app_runs/experiment_state_ledger_global.csv` 每个 nanobody 保留最新状态，可直接作为下一轮 `--experiment_plan_override_csv`。

本地软件“历史 -> 跨批次实验状态汇总”里可以直接看全局 ledger，支持按状态、结果、override 和关键词筛选，并显示可回灌标签数与状态分布图。

如果想从全局 ledger 生成真实验证标签和审计报告：

```bash
python build_experiment_validation_report.py --ledger_csv local_app_runs/experiment_state_ledger_global.csv --out_dir local_app_runs/experiment_validation_feedback
```

如果要把标签合并回特征表：

```bash
python build_experiment_validation_report.py --ledger_csv local_app_runs/experiment_state_ledger_global.csv --feature_csv pose_features.csv --label_col experiment_label --out_dir local_app_runs/experiment_validation_feedback
```

注意：`completed` 本身不会变成阳性或阴性；只有明确的 `experiment_result` 或 `validation_label` 才会进入训练标签。

如果要直接用回灌后的真实验证标签重新运行：

```bash
python run_recommended_pipeline.py --feature_csv local_app_runs/experiment_validation_feedback/pose_features_with_experiment_labels.csv --label_col experiment_label --out_dir recommended_pipeline_validation_retrain
```

本地软件里也可以完成同样操作：进入“历史 -> 跨批次实验状态汇总”，生成验证回灌报告时填写 `pose_features.csv` 路径，然后点击“使用带实验标签特征表作为下一轮输入”。左侧会自动切到 `feature_csv`，并把 `label_col` 改为 `experiment_label`。

回灌后建议再生成一次前后对照报告：

```bash
python build_validation_retrain_comparison.py --before_summary local_app_runs/<before_run>/outputs --after_summary local_app_runs/<after_run>/outputs --validation_labels_csv local_app_runs/experiment_validation_feedback/experiment_validation_labels.csv --top_k 10 --out_dir local_app_runs/validation_retrain_comparisons/<before_run>__vs__<after_run>
```

本地软件入口：进入“历史 -> 多运行对比”，选择回灌前/回灌后两次运行，打开“验证回灌再训练前后对照报告”。

如果运行次数变多，建议生成统一归档索引：

```bash
python build_result_archive.py --local_app_runs local_app_runs --out_dir local_app_runs/result_archive
```

本地软件入口：进入“历史 -> 结果自动归档与长期趋势”，点击“生成/刷新结果归档索引”。它会生成运行索引、关键产物 manifest 和验证回灌长期趋势表。

归档现在也会生成 `local_app_runs/result_archive/result_archive_lineage.csv` 和 `result_archive_lineage_graph.json/html/md`，用于判断哪些历史运行共享同一输入文件 manifest、同一 feature CSV hash 或同一参数 hash，并用图形化时间线查看多轮复跑来源关系。

如果运行前检查提示 `pdb_path` / `pocket_file` / `catalytic_file` / `ligand_file` 路径不存在，可以先生成修复建议：

```bash
python input_path_repair.py --input_csv input_pose_table.csv --search_root . --out_dir input_path_repair_outputs --write_repaired_csv
```

本地软件中也可以直接完成：点击“检查当前输入”，查看“缺失路径自动定位建议”，下载修复建议 CSV 或保存并使用自动修复版 `input_csv`。自动修复只替换高可信同名文件匹配；低可信候选需要人工确认。

## 0.3 pocket / catalytic 文件格式

| 写法 | 含义 |
|---|---|
| `A:45` | 链 A 的 45 号残基 |
| `A 45` | 和 `A:45` 等价 |
| `A:45,67,89` | 链 A 的 45、67、89 号残基 |
| `A:37-40` | 链 A 的 37、38、39、40 号残基 |
| `B:37-40` | 链 B 的 37、38、39、40 号残基 |
| `C:37-40` | 链 C 的 37、38、39、40 号残基 |
| `B:102A` | 链 B 的 102A 插入码残基 |

`catalytic_file` 的格式和 `pocket_file` 完全相同。区别是：`pocket_file` 表示你已经确认的口袋残基；`catalytic_file` 表示文献、数据库或人工确认的催化/功能锚点。主程序会同时计算直接催化残基接触和 catalytic-anchor shell 口袋诊断。

当没有直接 pocket 文献时，推荐做法是：

1. 把文献/M-CSA/UniProt/PDBj 查到的关键催化残基写入 `catalytic_file`。
2. 继续保留人工或已有工具给出的 `pocket_file`。
3. 运行主流程后查看 `catalytic_anchor_primary_shell_*` 和 `catalytic_anchor_manual_overlap_*` 列，判断催化锚点推断出的 3D 口袋是否和人工 pocket 一致。

### 0.3.1 高精度 pocket 证据整合（可选）

如果你愿意花时间追求 pocket 精度，不要只依赖单一工具。当前主程序新增了一个证据整合入口：

```bash
python build_pocket_evidence.py ^
  --pdb_path A\rsite\result\vhh1\CD38_1\1\1.pdb ^
  --manual_pocket_file A\cd38_pocket_from_rsite_chain_B.txt ^
  --catalytic_file A\cd38_pocket_from_rsite_chain_B.txt ^
  --catalytic_source_table catalytic_source_audit.csv ^
  --ai_source_table ai_prior_source_audit.csv ^
  --antigen_chain B ^
  --out_dir pocket_evidence_outputs
```

它会把人工/rsite、文献功能残基、catalytic-anchor 3D shell、ligand-contact、P2Rank、fpocket 和 AI residue prior 统一成证据表。输出包括：

- `pocket_evidence.csv`
- `pocket_residue_support.csv`
- `candidate_curated_pocket.txt`
- `review_residues.txt`
- `evidence_source_audit.csv`
- `evidence_source_template.csv`
- `ai_prior_audit.csv`
- `ai_prior_template.csv`
- `pocket_evidence_summary.json`
- `POCKET_EVIDENCE_REPORT.md`

默认还会启用 P2Rank/fpocket precision guard：如果外部工具给出的 pocket 明显过宽，且某些 residue 没有 manual、文献、catalytic core 或 ligand-contact 这类高置信支持，它们会被标记为 `external_overwide_guard`，进入 `review_residues.txt`，不会直接进入 `candidate_curated_pocket.txt`。需要调阈值时使用 `--external_overwide_max_residue_count` 和 `--external_overwide_max_fraction`。

AI prior 只能作为待复核线索。`--ai_pocket_file` 中的 residue 不参与 curated 判定的支持分/方法数，也不能和外部工具一起直接变成 ground truth。如果使用 AI 从文献中抽取 residue，请同时提供 `--ai_source_table`，至少保留 `source_sentence`、`evidence_level` 和 `review_status`；人工确认后再转写到 `manual_pocket_file` 或 `literature_file`。

边界：这个脚本只生成 pocket 证据和候选 pocket 文件，不改变正式 Rule / ML 权重。确认 `candidate_curated_pocket.txt` 合理后，可以把它作为下一轮 `pocket_file` 使用。

如果 literature/catalytic residue 来自论文、UniProt 或 M-CSA，建议补一个来源审计表再运行：

```csv
residue_key,evidence_role,source_kind,paper_title,pmid,doi,uniprot_id,uniprot_feature,mcsa_id,pdb_id,source_sentence,evidence_level,curator,review_status,manual_note
B:82,catalytic,paper,Example title,12345678,10.xxxx/example,P28907,active site,,1ABC,Source sentence,strong,your_name,confirmed,manual note
```

可以用 `--literature_source_table` / `--catalytic_source_table` 传入该表。输出的 `evidence_source_audit.csv` 会标出缺少来源、缺少可追溯字段或还没有人工确认的 residue；`evidence_source_template.csv` 可以直接作为下一轮补填模板。`residue_key` 支持 `B:83-84` 这类范围写法。

AI source audit 表建议写成：

```csv
residue_key,evidence_role,source_kind,paper_title,pmid,doi,source_sentence,evidence_level,ai_model,ai_prompt_id,ai_extraction_confidence,curator,review_status,manual_note
B:82,ai_prior,ai_extraction,Example title,12345678,10.xxxx/example,Sentence copied from source,medium,gpt-x,prompt_v1,0.72,your_name,unreviewed,needs manual check
```

本地软件也有同一入口：打开“诊断”页里的“Pocket 证据整合”，填写代表 PDB 和各类 pocket 证据文件后点击“构建 pocket 证据”。生成后可以按 curated、needs review、single-method、anchor-shell-only、AI-prior-only 和 not-found-in-structure 分组预览 support 表，并点击“设为下一轮 default_pocket_file”。

如果你已经有标准 `result/` 父目录，不想手动选代表 PDB，可以运行项目级批量入口：

```bash
python build_project_pocket_evidence.py --project_root A --target_prefix CD38
```

它会自动发现 `A/result/` 或 `A/rsite/result/`，按 `MMPBSA_energy` 最低值选择一个代表 PDB，生成项目级 pocket evidence，并在项目目录写出 `input_pose_table_with_pocket_evidence.csv`。本地软件“Pocket 证据整合”面板里也有“从 result 父目录批量构建 pocket evidence”按钮，完成后会把生成的 input CSV 回填为下一轮输入。

## 0.4 CD38 公开结构 benchmark 最短入口（可选）

如果只是想刷新当前仓库自带的公开 CD38 benchmark，不需要先安装 P2Rank/fpocket，直接运行：

```bash
python run_cd38_public_starter.py
```

如果你使用本地软件，也可以打开“诊断”页，点击“刷新 CD38 public starter”。这个按钮调用同一个脚本，并在页面里显示 panel 行数、缺失外部输出数量、action plan 状态，同时提供 action plan CSV、expected returns CSV、返回检查清单和 next-run 文件下载。

同一个“诊断”页还提供外部工具闭环入口：

1. 点击“生成 transfer zip”，把 `benchmarks/cd38/external_tool_transfer/cd38_external_tool_inputs_transfer.zip` 拿到 WSL/Linux 或另一台机器运行 P2Rank/fpocket；zip 内优先看 `cd38_external_tool_next_run.md`，再运行 `run_cd38_external_next_benchmark.*`。
2. 跑完后把返回 zip/目录路径填入“返回 zip 或返回目录路径”。
3. 先点“Dry-run 检查返回包”，确认不是原始输入包，也确认候选输出数量大于 0。
4. 再点“导入返回包并 finalize”，把真实输出接入 CD38 benchmark。

如果 dry-run 显示 `input_package_without_external_outputs`，说明当前文件只是原始 transfer zip，还没有包含 `p2rank_outputs/` 或 `fpocket_runs/*/*_out/`。

如果你想先测试导入器是否能识别“正确返回包形状”，但手头还没有真实 P2Rank/fpocket 输出，可运行：

```bash
python selftest_cd38_return_import_workflow.py
```

或在本地软件“诊断”页展开“返回包导入流程自测”。这个自测只生成 synthetic fixture，用来验证路径和 coverage，不是真实 benchmark 证据。

返回包 dry-run 或导入后，如果你想得到一个更直接的“能不能继续导入”结论，运行：

```bash
python build_cd38_return_package_gate.py
```

或在“诊断”页查看“返回包安全门控”。`PASS_READY_FOR_IMPORT` 才表示 dry-run 返回包已准备好导入；`FAIL_INPUT_PACKAGE` 表示拿回来的是原始输入包；`FAIL_SYNTHETIC_FIXTURE` 表示这是自测 fixture，不能用于真实 benchmark。

本地软件的“导入返回包并 finalize”会自动执行导入前 gate。如果 gate 不是 `PASS_*`，它会阻止正式导入，避免污染本地 external_tool_inputs。命令行 `python finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir>` 也默认启用同样的 gate；只有已经人工复核并确认要保留旧行为时，才加 `--skip_import_gate`。如果你在自动脚本或 CI 里希望 gate 失败时命令也失败，加 `--strict_import_gate`。

如果要一次性回归检查外部工具链路，运行：

```bash
python selftest_cd38_external_workflow.py
```

或在本地软件“诊断”页点击“运行外部链路一键自检”。该自检只验证 packaging/gate/importer/starter 链路，不运行真实 P2Rank/fpocket。

它会刷新：

- `benchmarks/cd38/results/cd38_benchmark_panel.csv`
- `benchmarks/cd38/ligand_candidates/cd38_ligand_candidate_report.md`
- `benchmarks/cd38/parameter_sensitivity/cd38_pocket_parameter_sensitivity_report.md`
- `benchmarks/cd38/proxy_calibration/cd38_proxy_calibration_report.md`
- `benchmarks/cd38/readiness/cd38_benchmark_readiness_report.md`
- `benchmarks/cd38/action_plan/cd38_external_benchmark_action_plan.md`
- `benchmarks/cd38/public_starter/cd38_public_starter_report.md`

边界：这个入口是公开结构 benchmark helper，不是 nanobody 筛选运行，也不运行外部 pocket finder；如果报告显示缺 fpocket/P2Rank 输出，再按 action plan 去外部机器或 WSL 补。

## 0.5 如果要接入外部 P2Rank/fpocket 输出做 CD38 benchmark（可选）

如果你还没有真实 P2Rank / fpocket 输出，先生成外部工具输入包：

```bash
python prepare_cd38_external_tool_inputs.py
```

重点看：

- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_inputs.md`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_expected_returns.csv`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_return_checklist.md`
- `benchmarks\cd38\external_tool_inputs\run_p2rank_templates.ps1`
- `benchmarks\cd38\external_tool_inputs\run_fpocket_templates.ps1`
- `benchmarks\cd38\external_tool_inputs\run_p2rank_templates.sh`
- `benchmarks\cd38\external_tool_inputs\run_fpocket_templates.sh`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_next_run.md`
- `benchmarks\cd38\external_tool_inputs\run_cd38_external_next_benchmark.ps1`
- `benchmarks\cd38\external_tool_inputs\run_cd38_external_next_benchmark.sh`
- `benchmarks\cd38\external_tool_inputs\check_external_tool_environment.ps1`
- `benchmarks\cd38\external_tool_inputs\refresh_readiness_after_external_tools.ps1`
- `benchmarks\cd38\external_tool_inputs\finalize_external_benchmark.ps1`

推荐顺序是：先运行 `check_external_tool_environment.ps1` 看本机是否已经有 `prank` / `fpocket`，再优先运行 `run_cd38_external_next_benchmark.*`，它只跑当前 action plan 里真正缺的 benchmark blocker；如果你明确想跑全部模板，再运行 `run_p2rank_templates.*` 和 `run_fpocket_templates.*`。外部工具跑完后，先用 `cd38_external_tool_return_checklist.md` 或 `cd38_external_tool_expected_returns.csv` 对照应该返回哪些 `PDB × method` 输出；然后把输出目录放回 `external_tool_inputs/`；再次运行预检脚本确认输出到位；最后运行 `finalize_external_benchmark.ps1` 生成最终接入报告。

也可以直接从命令行预检：

```bash
python check_cd38_external_tool_environment.py
```

预检结果在 `benchmarks\cd38\external_tool_inputs\preflight\cd38_external_tool_preflight_report.md`。

preflight 会优先按当前 package 内的 portable 路径检查，即使 manifest 里还有生成时的绝对路径，移动到另一台机器或 WSL 后也会优先看当前目录下的 `pdbs/`、`p2rank_outputs/` 和 `fpocket_runs/`。

如果外部输出已经复制回来了，推荐用 finalize 入口统一收尾：

```bash
python finalize_cd38_external_benchmark.py
```

确认 readiness report 正确后再导入：

```bash
python finalize_cd38_external_benchmark.py --run_discovered
```

如果导入后还要刷新参数敏感性：

```bash
python finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity
```

如果要把 CD38 外部工具输入包传到另一台机器或 WSL/Linux 上运行，先打包：

```bash
python package_cd38_external_tool_inputs.py
```

把 `benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer.zip` 发到目标机器。zip 内只包含 PDB 输入、PowerShell/Bash 模板、next-run runbook 和说明文件，不包含旧 preflight/finalize 报告。

打包脚本会自动刷新并包含 `cd38_external_tool_next_run.md`、`cd38_external_tool_next_run_plan.csv`、`run_cd38_external_next_benchmark.ps1` 和 `run_cd38_external_next_benchmark.sh`。在外部机器上优先运行 next-run 脚本；它当前只包含 4 个 benchmark completion 动作：`3ROP fpocket`、`4OGW fpocket`、`3F6Y fpocket` 和 `3F6Y P2Rank`。

外部机器跑完后，如果你拿回来的是整个目录或 zip，不要手工一个个复制结果，先导入：

```bash
python import_cd38_external_tool_outputs.py --source returned_external_tool_inputs.zip --dry_run
python import_cd38_external_tool_outputs.py --source returned_external_tool_inputs.zip
```

导入脚本只会接收 `p2rank_outputs/` 和 `fpocket_runs/*/*_out/`，默认不覆盖已有文件；如果返回 zip 外面多包了一层目录，也会自动剥离到正确相对路径。确认导入报告后再运行 `python finalize_cd38_external_benchmark.py --run_discovered`。如果候选文件数为 0，先看 import report 的 `source_diagnosis`；若是 `input_package_without_external_outputs`，说明拿到的是原始输入包而不是跑完工具后的返回包。需要细查时看 `cd38_external_tool_output_import_scan.csv`；要看缺哪些结构/方法，看 `cd38_external_tool_output_import_coverage.csv`；要直接知道下一步应该补哪些输出，看 `cd38_external_tool_output_import_repair_plan.csv`。

也可以直接用 finalize 入口完成“导入返回包 + readiness + 可选导入 benchmark”：

```bash
python finalize_cd38_external_benchmark.py --import_source returned_external_tool_inputs.zip --run_discovered
```

如果返回包里没有真实 `p2rank_outputs/` 或 `fpocket_runs/*/*_out/` 输出，finalize 会显示 `Runnable external benchmark rows: 0`，并跳过 benchmark 汇总和参数敏感性刷新，不会把旧结果误当成新导入。finalize 报告也会链接 repair plan CSV，直接列出每个缺失 `PDB × method` 应运行的模板和应返回的相对路径。

如果你只是想先知道 CD38 benchmark 目前缺什么，优先运行一键刷新：

```bash
python refresh_cd38_benchmark_readiness.py
```

看 `benchmarks\cd38\readiness\cd38_benchmark_readiness_report.md`。如果你已经有外部 P2Rank / fpocket 输出目录：

```bash
python refresh_cd38_benchmark_readiness.py ^
  --p2rank_root your_p2rank_outputs ^
  --fpocket_root your_fpocket_outputs ^
  --rank_by_pdb 3ROP=2,4OGW=1
```

默认只刷新报告，不会重跑 benchmark；确认 readiness report 后再加 `--run_discovered`。

如果你已经在外部跑好了 `fpocket`，先不要手工整理每个 pocket。把输出目录交给准备脚本：

```bash
python prepare_cd38_fpocket_panel.py ^
  --fpocket_root your_fpocket_outputs ^
  --rcsb_pdb_id 3ROP ^
  --chain_filter A ^
  --manifest_out benchmarks\cd38\fpocket_discovered_manifest.csv
```

先看生成的 `benchmarks\cd38\fpocket_discovered_manifest.csv.report.md`。它会告诉你发现了多少 `pocket*_atm.pdb`、哪些文件能进入 benchmark、哪些因为无法识别 PDB ID 被跳过。

确认无误后再加 `--run` 批量跑：

```bash
python prepare_cd38_fpocket_panel.py ^
  --fpocket_root your_fpocket_outputs ^
  --rcsb_pdb_id 3ROP ^
  --chain_filter A ^
  --manifest_out benchmarks\cd38\fpocket_discovered_manifest.csv ^
  --run
```

如果目录名已经包含 `3ROP_out`、`4OGW_fpocket` 这类首位为数字的 PDB ID，可以不写 `--rcsb_pdb_id`；否则建议明确指定。

如果你已经在外部跑好了 `P2Rank`，也可以批量导入 `*_predictions.csv`：

```bash
python prepare_cd38_p2rank_panel.py ^
  --p2rank_root your_p2rank_outputs ^
  --chain_filter A ^
  --manifest_out benchmarks\cd38\p2rank_discovered_manifest.csv
```

先看生成的 `benchmarks\cd38\p2rank_discovered_manifest.csv.report.md`。如果某个结构的 active-site pocket 不是 rank 1，可加 `--rank_by_pdb 3ROP=2`。

如果你不确定当前 CD38 benchmark 还缺什么，先生成扩展计划：

```bash
python build_cd38_benchmark_expansion_plan.py
```

重点看 `benchmarks\cd38\expansion_plan\cd38_benchmark_missing_actions.csv`。当前它会把 3ROP / 4OGW 缺真实 fpocket 输出、3F6Y 缺外部 P2Rank/fpocket 输出这些事项列出来。

如果你不确定某个 CD38 结构有没有适合 ligand-contact baseline 的配体，先跑：

```bash
python inspect_cd38_ligand_candidates.py
```

当前扫描结论是：`3ROP` 推荐 `50A/NCA`，`4OGW` 推荐 `NMN`，`3F6Y` 没有活性口袋 ligand candidate，所以 3F6Y 不再做 ligand-contact baseline。

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
NB_001,CF_01,P_001,data/NB_001_CF_01_P_001.pdb,B,A,data/pocket.txt,data/catalytic.txt,data/ligand.pdb,1
NB_001,CF_01,P_002,data/NB_001_CF_01_P_002.pdb,B,A,data/pocket.txt,data/catalytic.txt,data/ligand.pdb,0
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

### 4.0 `input.csv` 和 `pose_features.csv` 的区别

- `input.csv` 是原始输入表，给 `build_feature_table.py` 用，通常至少包含 `nanobody_id`、`conformer_id`、`pose_id`、`pdb_path`。
- `pose_features.csv` 是特征表，是 `build_feature_table.py` 处理 `input.csv` 后生成的结果，后续的 `rule_ranker.py`、`train_pose_model.py`、`calibrate_rule_ranker.py` 主要都用它。
- 如果你手里只有 `input.csv`，先跑特征构建；如果你已经有 `pose_features.csv`，就可以直接从特征表开始后续步骤。

### 4.1 推荐做法

推荐由脚本自动生成 `pose_features.csv`，不建议手工写。

### 4.2 `pose_features.csv` 至少应包含

- `nanobody_id`
- `conformer_id`
- `pose_id`

并建议至少包含当前主流程常用的几何/打分列：

- `pocket_hit_fraction`
- `catalytic_hit_fraction`
- `catalytic_anchor_primary_shell_hit_fraction`
- `catalytic_anchor_min_distance_to_primary_shell`
- `catalytic_anchor_manual_overlap_fraction_of_shell`
- `catalytic_anchor_shell_overwide_proxy`
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
- `MMPBSA_energy`
- `mmgbsa`
- `interface_dg`

可选：

- `label`
- `status`
- `hdock_score`

### 4.3 一个真实表头示例

下面是当前仓库 smoke test 里实际生成的 `pose_features.csv` 表头：

```csv
nanobody_id,conformer_id,pose_id,status,pocket_hit_fraction,catalytic_hit_fraction,mouth_occlusion_score,mouth_axis_block_fraction,mouth_aperture_block_fraction,mouth_min_clearance,delta_pocket_occupancy_proxy,substrate_overlap_score,ligand_path_block_score,ligand_path_block_fraction,ligand_path_bottleneck_score,ligand_path_exit_block_fraction,ligand_path_min_clearance,min_distance_to_pocket,rsite_accuracy,MMPBSA_energy,mmgbsa,interface_dg,hdock_score,label
```

## 5. 跑完后先看什么

最先看这几个文件：

| 文件 | 你要看什么 |
|---|---|
| `my_outputs/recommended_pipeline_report.md` | 流程是否跑完，哪些步骤执行或跳过 |
| `my_outputs/quality_gate/quality_gate_report.md` | 本次运行是否 PASS/WARN/FAIL，是否适合直接解读排名 |
| `my_outputs/batch_decision_summary/batch_decision_summary.md` | 本批次一页结论：能不能解读、优先看谁、先修什么风险、验证证据是否足够、是否建议进入下一轮 |
| `my_outputs/geometry_proxy_audit/geometry_proxy_audit_report.md` | mouth/path/pocket/contact 等几何 proxy 是否互相矛盾，不改变分数 |
| `my_outputs/validation_evidence_audit/validation_evidence_report.md` | 当前 top 候选有没有真实验证标签、正负标签是否平衡、还缺哪些实验结果 |
| `my_outputs/consensus_outputs/consensus_ranking.csv` | 最终优先候选、可信度、风险提示、低可信原因拆解 |
| `my_outputs/score_explanation_cards/score_explanation_cards.html` | 分数解释卡片，直接说明高分原因、主要风险、label 状态和建议动作 |
| `my_outputs/candidate_report_cards/index.html` | 每个候选的可读报告卡，已内嵌候选对比上下文 |
| `my_outputs/candidate_comparisons/candidate_comparison_report.md` | 为什么 A 排在 B 前面，哪些候选分差很小；包含候选分组小结，本地软件里可手工选择 2 到 5 个候选做自定义对比 |
| `my_outputs/candidate_comparisons/candidate_group_comparison_summary.csv` | 按 diversity/family/status 等字段汇总候选组的共同优势、共同风险和最高排名候选 |
| `my_outputs/ai_outputs/ai_run_summary.md` | 可选 AI/离线解释摘要，快速读懂本次结果 |
| `my_outputs/provenance/run_provenance_card.md` | 复现审计卡片，记录输入/输出/代码/依赖 hash 和输入文件引用 manifest |
| `my_outputs/experiment_suggestions/next_experiment_suggestions.csv` | 下一轮优先验证或复核谁，含 diversity-aware 队列排序 |
| `my_outputs/experiment_suggestions/experiment_plan.md` | 本轮实验计划单，含 include_now / standby / defer；可带人工覆盖、负责人、成本和状态 |
| `my_outputs/experiment_suggestions/experiment_plan_state_ledger.csv` | 实验状态账本，可编辑后作为下一轮 override |
| `my_outputs/parameter_sensitivity/candidate_rank_sensitivity.csv` | 候选排名对权重/QC 惩罚是否敏感 |
| `my_outputs/ml_ranking_outputs/nanobody_ranking.csv` | 纯 ML 排名，用于和共识排名对照 |
| `my_outputs/model_outputs/pose_predictions.csv` | pose 级别预测，用于追溯原因 |

阅读顺序建议：先看 `recommended_pipeline_report.md` 判断流程是否跑完整，再看 `quality_gate_report.md` 判断本次结果是 PASS、WARN 还是 FAIL；如果不是 FAIL，优先打开 `batch_decision_summary.md` 看一页结论。涉及 mouth/path/pocket 阻断解释时，再看 `geometry_proxy_audit_report.md` 确认几何 proxy 是否自洽；要安排实验时，再看 `validation_evidence_report.md` 确认 top 候选真实验证覆盖是否足够。然后看 `score_explanation_cards.html` 理解分数含义和风险，最后看 `consensus_ranking.csv` 里的 `low_confidence_reasons` 和候选报告卡决定优先候选；报告卡里已经嵌入相邻候选对比。如果某个候选 `top_n_unstable=True`、`rank_span` 很大，或在候选对比中 `is_close_decision=True`，先人工复核再做实验决策。

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
nanobody_id,conformer_id,pose_id,pred_prob,pred_logit,top_contributing_features,pocket_hit_fraction,catalytic_hit_fraction,mouth_occlusion_score,mouth_axis_block_fraction,mouth_aperture_block_fraction,mouth_min_clearance,substrate_overlap_score,ligand_path_block_score,ligand_path_block_fraction,ligand_path_bottleneck_score,ligand_path_exit_block_fraction,ligand_path_min_clearance,delta_pocket_occupancy_proxy,min_distance_to_pocket,rsite_accuracy,MMPBSA_energy,mmgbsa,label,pseudo_label,pseudo_score,pseudo_rank,pseudo_components
```

如果输入表或标准 `result/` 目录提供了 `MMPBSA_energy` / `mmgbsa`，后续 conformer 聚合会优先按这些能量列从低到高选 top-k pose。

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

如果要确认 top-k 到底选了哪些 pose，看 `my_outputs/ml_ranking_outputs/conformer_scores.csv` 或 `my_outputs/rule_outputs/conformer_rule_scores.csv` 里的 `topk_selection_mode`、`topk_selection_column`、`topk_pose_ids` 和 `mean_topk_MMPBSA_energy`。

## 7. 你只需要记住的最短版本

只有一句话：

1. 准备一个 `input_pose_table.csv`
2. 运行 `python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs`
3. 打开 `my_outputs/recommended_pipeline_report.md`
4. 看 `my_outputs/consensus_outputs/consensus_ranking.csv`
5. 打开 `my_outputs/candidate_report_cards/index.html`
6. 看 `my_outputs/candidate_comparisons/candidate_comparison_report.md` 判断相邻候选差异
7. 如果要安排实验，看 `my_outputs/validation_evidence_audit/validation_evidence_report.md`
8. 如果要检查排名是否稳，看 `my_outputs/parameter_sensitivity/candidate_rank_sensitivity.csv`

如果你已经有 `pose_features.csv`，就把上面的 `--input_csv ...` 换成 `--feature_csv pose_features.csv`。

## 8. 可选 AI 解释摘要

不想只看 CSV 时，可以让 pipeline 额外生成一份解释性 Markdown：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs --enable_ai_assistant
```

默认是完全离线的本地摘要，不需要 API Key，也不会上传文件。生成位置：

- `my_outputs/ai_outputs/ai_run_summary.md`
- `my_outputs/ai_outputs/ai_top_candidates_explanation.md`
- `my_outputs/ai_outputs/ai_failure_diagnosis.md`
- `my_outputs/ai_outputs/ai_assistant_summary.json`

如果你想接 OpenAI，把 Key 放到环境变量，再指定 provider：

```bat
set OPENAI_API_KEY=你的_key
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs --enable_ai_assistant --ai_provider openai
```

安全边界：AI 解释层只读取已经生成的 summary、排名表前几行、候选对比和实验建议，不上传原始 PDB 或完整输入 CSV；OpenAI 不可用时会自动回退到离线摘要。

## 9. 复现审计卡片

推荐 pipeline 默认会生成 provenance 文件，不需要额外参数：

- `my_outputs/provenance/run_provenance_card.json`
- `my_outputs/provenance/run_provenance_card.md`
- `my_outputs/provenance/run_artifact_manifest.csv`
- `my_outputs/provenance/run_input_file_manifest.csv`
- `my_outputs/provenance/run_provenance_integrity.json`

里面会记录：

- 本次输入 CSV / feature CSV 的 SHA256
- input CSV 或 feature CSV 中引用的 `pdb_path` / `pocket_file` / `catalytic_file` / `ligand_file` 是否存在、大小和 SHA256
- 关键输出文件的 SHA256
- 参与运行的脚本文件 hash
- Python、依赖版本、Git commit 和 dirty 状态
- 运行命令、参数 hash、artifact manifest hash、input file manifest hash 和 integrity seal hash

如果输出文件很大，默认超过 100 MB 的文件不会计算 SHA256，只会记录大小和 `hash_status`。可以用下面参数调整：

```bash
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs --provenance_hash_max_mb 300
```

如果想确认 provenance 文件后续没有被误改：

```bash
python verify_run_provenance.py --signature_json my_outputs\provenance\run_provenance_integrity.json --strict
```

注意：这是 SHA256 完整性封存，不是带私钥的正式数字签名。

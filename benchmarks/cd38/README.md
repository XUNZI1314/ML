# CD38 Pocket Benchmark

这组文件用于做一个很轻量的 CD38 口袋/活性位点 benchmark。

当前真值文件：

- `cd38_active_site_truth.txt`

它不是“完整 pocket 全景图”，而是一个偏保守的 baseline truth，用于检测 pocket 结果是否覆盖 CD38 已知关键位点。

## 先选你要做的事

| 目标 | 使用脚本 | 结果看哪里 |
|---|---|---|
| 刷新公开结构 starter 总览 | `run_cd38_public_starter.py` | `public_starter/cd38_public_starter_report.md` |
| 用已知 truth 做自检 | `benchmark_cd38_pocket_accuracy.py` | `cd38_pocket_benchmark_outputs/` |
| 从 ligand-bound 结构生成 baseline | `run_cd38_ligand_contact_benchmark.py` | `benchmarks\cd38\results\<name>/` |
| 评估 P2Rank 输出 | `run_cd38_p2rank_benchmark.py` | `benchmarks\cd38\results\<name>/` |
| 评估 fpocket 输出 | `run_cd38_fpocket_benchmark.py` | `benchmarks\cd38\results\<name>/` |
| 生成外部工具输入包和命令模板 | `prepare_cd38_external_tool_inputs.py` | `external_tool_inputs/` |
| 打包外部工具输入包用于转移运行 | `package_cd38_external_tool_inputs.py` | `external_tool_transfer/` |
| 导入外部机器返回的输出 | `import_cd38_external_tool_outputs.py` | `external_tool_inputs/imported_outputs/` |
| 检查外部工具环境和输出缺口 | `check_cd38_external_tool_environment.py` | `external_tool_inputs/preflight/` |
| 一键 finalize 外部 benchmark 接入 | `finalize_cd38_external_benchmark.py` | `external_tool_inputs/finalize/` |
| 生成外部 benchmark 执行清单 | `build_cd38_external_benchmark_action_plan.py` | `action_plan/` |
| 生成外部机器 next-run 脚本 | `build_cd38_external_tool_runbook.py` | `external_tool_inputs/cd38_external_tool_next_run.md` |
| 批量发现 fpocket 输出并生成 panel | `prepare_cd38_fpocket_panel.py` | `fpocket_discovered_manifest.csv` |
| 批量发现 P2Rank 输出并生成 panel | `prepare_cd38_p2rank_panel.py` | `p2rank_discovered_manifest.csv` |
| 一键刷新 benchmark 准备状态 | `refresh_cd38_benchmark_readiness.py` | `readiness/` |
| 查看还缺哪些结构/方法结果 | `build_cd38_benchmark_expansion_plan.py` | `expansion_plan/` |
| 检查结构是否适合 ligand-contact baseline | `inspect_cd38_ligand_candidates.py` | `ligand_candidates/` |
| 批量复跑当前 panel | `run_cd38_benchmark_manifest.py` | `results/cd38_benchmark_panel.csv` |
| 比较多个 pocket 方法是否一致 | `compare_pocket_method_consensus.py` | `results/*_method_consensus_*/` |
| 检查 pocket 参数是否敏感 | `analyze_cd38_pocket_parameter_sensitivity.py` | `parameter_sensitivity/` |
| 校准 pocket overwide proxy | `build_cd38_proxy_calibration_report.py` | `proxy_calibration/` |

当前基线残基：

- `A:125`
- `A:127`
- `A:146`
- `A:155`
- `A:189`
- `A:193`
- `A:226`

用途：

- 验证 `fpocket` / `P2Rank` / 手工 pocket 文件是否覆盖已知 CD38 关键位点
- 先做 baseline sanity check，再决定是否继续扩成更完整的 CD38 pocket benchmark

推荐结构：

- `3F6Y`
- `3ROP`

最小用法：

```bash
python run_cd38_public_starter.py
```

这条命令不运行外部 pocket finder。它会刷新当前仓库已有的公开 CD38 PDB benchmark 面板、ligand candidate scan、参数敏感性、proxy calibration、外部工具 preflight、readiness、action plan 和 next-run runbook，并输出：

- `benchmarks\cd38\public_starter\cd38_public_starter_report.md`
- `benchmarks\cd38\public_starter\cd38_public_starter_summary.json`
- `benchmarks\cd38\external_tool_inputs\cd38_external_tool_next_run.md`
- `benchmarks\cd38\external_tool_inputs\run_cd38_external_next_benchmark.ps1`
- `benchmarks\cd38\external_tool_inputs\run_cd38_external_next_benchmark.sh`

如果只是想确认当前 CD38 benchmark 处于什么状态，优先看这个 starter 报告。

如果使用本地软件，也可以在“诊断”页点击“刷新 CD38 public starter”。界面会显示 panel rows、缺失/待补输出数量、action plan 状态，并提供 starter 报告、action plan CSV、expected returns CSV、返回检查清单和 next-run 文件下载。

同一诊断页还支持生成 `external_tool_transfer/cd38_external_tool_inputs_transfer.zip`。transfer zip 会带上 `cd38_external_tool_next_run.md` 和 `run_cd38_external_next_benchmark.*`，外部机器上优先运行 next-run 脚本。跑完 P2Rank/fpocket 后，把返回 zip/目录填回页面，先 dry-run 检查候选输出和 coverage，再导入并 finalize。如果 dry-run 诊断为 `input_package_without_external_outputs`，说明带回来的仍是原始输入包，还没有真实外部工具输出。

如果要回归测试导入路径规则，可运行 `python selftest_cd38_return_import_workflow.py`，或在本地软件“诊断”页运行“返回包导入流程自测”。该自测会生成 synthetic fixture 并验证 importer coverage 能达到 `6/6`，但它不是 pocket accuracy benchmark。

返回包 dry-run 或导入后，可运行 `python build_cd38_return_package_gate.py`，或在本地软件“诊断”页查看“返回包安全门控”。门控会给出 `PASS_READY_FOR_IMPORT`、`WARN_PARTIAL_RETURN`、`FAIL_INPUT_PACKAGE`、`FAIL_SYNTHETIC_FIXTURE` 等状态；只有 `PASS_*` 状态才适合继续导入或 finalize。本地软件和 `finalize_cd38_external_benchmark.py --import_source` 正式导入前都会自动先跑隔离 dry-run 和 gate，非 `PASS_*` 会被拦截；如已人工复核且必须绕过，可在 CLI 使用 `--skip_import_gate`。如需 CI/自动化在 gate 失败时返回非零，使用 `--strict_import_gate`。

如果要一次性回归检查整个外部工具链路，可运行 `python selftest_cd38_external_workflow.py`，或在本地软件“诊断”页运行“CD38 外部工具链路一键自检”。该自检会检查 transfer 打包、原始包 strict gate 拦截、synthetic fixture 拦截和 public starter 刷新。

自检用法：

```bash
python benchmark_cd38_pocket_accuracy.py ^
  --rcsb_pdb_id 3F6Y ^
  --predicted_pocket_file benchmarks\cd38\cd38_active_site_truth.txt ^
  --out_dir cd38_pocket_benchmark_outputs
```

说明：

- 上面这条命令属于 self-check，等价于“truth 对 truth”，结果应接近 1.0
- 真正评估时，把 `--predicted_pocket_file` 换成 `fpocket` / `P2Rank` / 手工预测出的 pocket residue 文件
- 输出会包含 exact overlap、near-hit coverage、precision、Jaccard、F1、truth-based `overwide_pocket_score` 和逐残基距离表
- `overwide_pocket_score` 不是判断“是否找到口袋”的指标，而是用于量化“找到了但 pocket 边界偏宽”的风险

如果你已经有 `P2Rank` 的 `<protein>_predictions.csv`，可以直接串起来跑：

```bash
python run_cd38_p2rank_benchmark.py ^
  --predictions_csv your_p2rank_dir\3ROP.pdb_predictions.csv ^
  --rcsb_pdb_id 3ROP ^
  --rank 2 ^
  --chain_filter A ^
  --out_dir benchmarks\cd38\results\3ROP_p2rank_rank2_chainA
```

这条包装命令会自动完成两步：

- 从 `predictions.csv` 里提取指定 pocket 的 residue 文件
- 直接对 CD38 baseline truth 跑 benchmark，并落盘 summary / report / residue tables

如果你已经有 `fpocket` 输出的 `pocket*_atm.pdb`，可以直接串起来跑：

```bash
python run_cd38_fpocket_benchmark.py ^
  --fpocket_pocket_pdb your_fpocket_dir\pockets\pocket1_atm.pdb ^
  --rcsb_pdb_id 3ROP ^
  --chain_filter A ^
  --out_dir benchmarks\cd38\results\3ROP_fpocket_pocket1_chainA
```

这条包装命令会自动完成两步：

- 从 `pocket*_atm.pdb` 里提取 residue 文件
- 直接对 CD38 baseline truth 跑 benchmark，并落盘 summary / report / residue tables

如果你还没有真实 P2Rank / fpocket 输出，先生成外部工具输入包：

```bash
python prepare_cd38_external_tool_inputs.py
```

这条命令会生成：

- `external_tool_inputs/cd38_external_tool_input_manifest.csv`
- `external_tool_inputs/cd38_external_tool_expected_returns.csv`
- `external_tool_inputs/cd38_external_tool_return_checklist.md`
- `external_tool_inputs/run_p2rank_templates.ps1`
- `external_tool_inputs/run_fpocket_templates.ps1`
- `external_tool_inputs/run_p2rank_templates.sh`
- `external_tool_inputs/run_fpocket_templates.sh`
- `external_tool_inputs/check_external_tool_environment.ps1`
- `external_tool_inputs/refresh_readiness_after_external_tools.ps1`
- `external_tool_inputs/finalize_external_benchmark.ps1`
- `external_tool_inputs/cd38_external_tool_inputs_summary.json`
- `external_tool_inputs/cd38_external_tool_inputs.md`

如果只想跑当前 action plan 里真正缺的 benchmark blocker，先生成 next-run runbook：

```bash
python build_cd38_external_tool_runbook.py
```

它会在 `external_tool_inputs/` 下生成 `cd38_external_tool_next_run.md`、`cd38_external_tool_next_run_plan.csv`、`run_cd38_external_next_benchmark.ps1` 和 `run_cd38_external_next_benchmark.sh`。当前默认选择 4 个 benchmark completion 动作：`3ROP fpocket`、`4OGW fpocket`、`3F6Y fpocket`、`3F6Y P2Rank`。

推荐流程：

1. 先运行 `check_external_tool_environment.ps1`，确认本机工具和已有输出状态。
2. 优先在已安装外部工具的环境运行 `run_cd38_external_next_benchmark.ps1`。
3. 如果在 Linux/WSL 环境运行，改用 `run_cd38_external_next_benchmark.sh`。
4. 只有在明确要跑所有模板时，才运行 `run_p2rank_templates.*` 和 `run_fpocket_templates.*`。
5. 如果在别的机器运行，先对照 `cd38_external_tool_return_checklist.md` 或 `cd38_external_tool_expected_returns.csv`，确认应返回的 `PDB × method` 输出是否都存在。
6. 把输出目录复制回 `external_tool_inputs/`；如果拿回来的是整个目录或 zip，优先用 `python finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir>` 导入后再检查。
7. 再运行 `check_external_tool_environment.ps1`，确认预期 P2Rank/fpocket 输出已经存在。
8. 运行 `finalize_external_benchmark.ps1`，生成 preflight + readiness 的最终接入报告。
9. readiness report 无明显问题后，再运行 `python finalize_cd38_external_benchmark.py --run_discovered` 导入 benchmark；如果返回包已经确认无误，也可直接运行 `python finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir> --run_discovered`。
10. 如果导入后需要同时刷新参数敏感性和 proxy 校准报告，运行 `python finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity`。
11. 如果 finalize 仍显示缺输出，打开 `action_plan/cd38_external_benchmark_action_plan.md`，按 priority 补齐真实外部工具结果。

如果只想检查环境和输出缺口，可执行：

```bash
python check_cd38_external_tool_environment.py
```

它会生成：

- `external_tool_inputs/preflight/cd38_external_tool_preflight_status.csv`
- `external_tool_inputs/preflight/cd38_external_tool_preflight_summary.json`
- `external_tool_inputs/preflight/cd38_external_tool_preflight_report.md`

preflight 使用 `package_portable_first` 路径解析策略：如果输入包移动到了 WSL、Linux 或另一台机器，会优先检查当前 package 内的 `pdbs/`、`p2rank_outputs/` 和 `fpocket_runs/`，不会只盯着 manifest 里旧的绝对路径。

如果要把输入包传到 Linux/WSL 或另一台机器上运行，可先打包：

```bash
python package_cd38_external_tool_inputs.py
```

默认输出：

- `external_tool_transfer/cd38_external_tool_inputs_transfer.zip`
- `external_tool_transfer/cd38_external_tool_inputs_transfer_manifest.csv`
- `external_tool_transfer/cd38_external_tool_inputs_transfer_summary.json`
- `external_tool_transfer/cd38_external_tool_inputs_transfer_report.md`

transfer zip 会自动包含 next-run runbook 和脚本。外部机器上优先看 `cd38_external_tool_next_run.md`，再运行 `run_cd38_external_next_benchmark.*`；这比直接运行全部模板更不容易重复跑已经有结果的项目。

外部机器跑完后，如果拿回来的是整个目录或 zip，可导入：

```bash
python import_cd38_external_tool_outputs.py --source returned_external_tool_inputs.zip --dry_run
python import_cd38_external_tool_outputs.py --source returned_external_tool_inputs.zip
```

导入脚本只复制 `p2rank_outputs/` 和 `fpocket_runs/*/*_out/`，默认不覆盖已有文件；如果返回 zip/目录外面多包了一层目录，也会自动剥离到 package 相对路径。输出：

- `external_tool_inputs/imported_outputs/cd38_external_tool_output_import_manifest.csv`
- `external_tool_inputs/imported_outputs/cd38_external_tool_output_import_scan.csv`
- `external_tool_inputs/imported_outputs/cd38_external_tool_output_import_coverage.csv`
- `external_tool_inputs/imported_outputs/cd38_external_tool_output_import_repair_plan.csv`
- `external_tool_inputs/imported_outputs/cd38_external_tool_output_import_summary.json`
- `external_tool_inputs/imported_outputs/cd38_external_tool_output_import_report.md`

如果候选输出文件为 0，优先看 scan CSV，它会列出扫描文件、被忽略原因和是否缺少 `p2rank_outputs/` / `fpocket_runs/*/*_out/`。
import report 还会给出 `source_diagnosis`；如果显示 `input_package_without_external_outputs`，说明当前 zip 更像原始输入包，不是外部工具运行后的返回包。
coverage CSV 会按 `PDB × method` 列出返回包覆盖情况，例如 `3ROP:p2rank`、`3ROP:fpocket` 是否已经出现。
repair plan CSV 会把仍缺的 `PDB × method` 展开成下一步动作，包括应运行的模板、应返回的相对路径和 dry-run 验证命令。

如果不想分两步执行，也可以直接用 finalize 导入返回包并接着刷新 readiness：

```bash
python finalize_cd38_external_benchmark.py --import_source returned_external_tool_inputs.zip
python finalize_cd38_external_benchmark.py --import_source returned_external_tool_inputs.zip --run_discovered
```

如果 readiness 没有发现可运行 rows，`--run_discovered` 不会继续汇总旧 benchmark panel；报告中会显示 `Benchmark follow-up status: skipped_no_runnable_external_rows`。
如果返回包缺输出，finalize report 会链接 `cd38_external_tool_output_import_repair_plan.csv`，优先按该清单补齐后再重新导入。

如果外部输出已经复制回 `external_tool_inputs/`，推荐统一收尾：

```bash
python finalize_cd38_external_benchmark.py
```

这条命令会串联 preflight 和 readiness，并生成：

- `external_tool_inputs/finalize/cd38_external_benchmark_finalize_summary.json`
- `external_tool_inputs/finalize/cd38_external_benchmark_finalize_report.md`
- `action_plan/cd38_external_benchmark_action_plan.md`
- `action_plan/cd38_external_benchmark_action_plan.csv`
- `action_plan/cd38_external_benchmark_action_plan.json`

默认只检查不导入。确认 manifest row 正确后，再执行：

```bash
python finalize_cd38_external_benchmark.py --run_discovered
```

如果只想单独刷新外部 benchmark 执行清单：

```bash
python build_cd38_external_benchmark_action_plan.py
```

当前 action plan 会把 `3ROP/4OGW` 的 fpocket 输出列为 priority `1`，把 `3F6Y` 的 P2Rank/fpocket 输出列为 priority `2`，并把 `3ROP/4OGW` 的 P2Rank 外部包复现实验列为 priority `3`。

如果要把这份 action plan 转成外部机器可直接执行的最短脚本：

```bash
python build_cd38_external_tool_runbook.py
```

如果也要补齐 priority `3` 的外部包复现动作，加 `--include_package_reproducibility`。

如果你已经有一批 `fpocket` 输出目录，可以先自动发现所有 `pocket*_atm.pdb` 并生成一份只包含 fpocket rows 的 manifest：

```bash
python prepare_cd38_fpocket_panel.py ^
  --fpocket_root your_fpocket_outputs ^
  --rcsb_pdb_id 3ROP ^
  --chain_filter A ^
  --manifest_out benchmarks\cd38\fpocket_discovered_manifest.csv ^
  --report_md benchmarks\cd38\fpocket_discovered_manifest.csv.report.md ^
  --run
```

这条命令会：

- 扫描 `**/pockets/pocket*_atm.pdb` 和 `**/pocket*_atm.pdb`
- 生成 `fpocket_discovered_manifest.csv`
- 生成 `fpocket_discovered_manifest.csv.summary.json`
- 生成 `fpocket_discovered_manifest.csv.report.md`，列出扫描目录、可运行 rows、跳过原因和下一步命令
- 在 `--run` 时调用 `run_cd38_benchmark_manifest.py` 批量跑 fpocket benchmark
- 后续可用 `analyze_cd38_pocket_parameter_sensitivity.py` 生成 `parameter_sensitivity/fpocket_pocket_sensitivity.csv`

说明：

- 自动 PDB ID 推断只接受首位为数字的 4 字符 PDB ID，例如 `3ROP_out`、`4OGW_fpocket`。
- 如果你的 fpocket 输出目录名不含结构 ID，单结构批次直接加 `--rcsb_pdb_id 3ROP` 更稳。

如果你已经有一批 `P2Rank` 输出目录，可以先自动发现所有 `*_predictions.csv` 并生成一份只包含 P2Rank rows 的 manifest：

```bash
python prepare_cd38_p2rank_panel.py ^
  --p2rank_root your_p2rank_outputs ^
  --chain_filter A ^
  --manifest_out benchmarks\cd38\p2rank_discovered_manifest.csv ^
  --rank_by_pdb 3ROP=2,4OGW=1
```

这条命令会：

- 扫描 `**/*_predictions.csv`、`**/*predictions.csv` 和 `**/predictions.csv`
- 校验 `rank` 和 `residue_ids` 必需列
- 生成 `p2rank_discovered_manifest.csv`
- 生成 `p2rank_discovered_manifest.csv.summary.json`
- 生成 `p2rank_discovered_manifest.csv.report.md`，列出扫描目录、可运行 rows、跳过原因和下一步命令
- 在 `--run` 时调用 `run_cd38_benchmark_manifest.py` 批量跑 P2Rank benchmark

说明：

- 自动 PDB ID 推断支持 `3F6Y.pdb_predictions.csv`、`3ROP_predictions.csv` 这类文件名。
- 如果路径或文件名不含结构 ID，单结构批次直接加 `--rcsb_pdb_id 3F6Y` 更稳。
- 如果 active-site pocket 不是 P2Rank 全局 rank 1，可以用 `--rank_by_pdb` 明确指定。

如果你想从 ligand-bound 结构直接生成一版 ligand-contact baseline，可执行：

```bash
python run_cd38_ligand_contact_benchmark.py ^
  --rcsb_pdb_id 3ROP ^
  --protein_chain A ^
  --ligand_chain A ^
  --ligand_resnames 50A,NCA ^
  --out_dir benchmarks\cd38\results\3ROP_ligand_contact_chainA_50A_NCA
```

如果你想把当前 `results/` 下的结构化结果聚合成一个总表，可执行：

```bash
python summarize_cd38_benchmarks.py --results_root benchmarks\cd38\results
```

这条命令会生成：

- `results/cd38_benchmark_panel.csv`
- `results/cd38_benchmark_panel.md`

如果你想一次性刷新 ligand 适用性、扩展缺口和外部工具输出准备状态，可执行：

```bash
python refresh_cd38_benchmark_readiness.py
```

如果已经有外部 P2Rank / fpocket 输出目录，可一起扫描：

```bash
python refresh_cd38_benchmark_readiness.py ^
  --p2rank_root your_p2rank_outputs ^
  --fpocket_root your_fpocket_outputs ^
  --rank_by_pdb 3ROP=2,4OGW=1
```

这条命令会生成：

- `readiness/cd38_benchmark_readiness_summary.json`
- `readiness/cd38_benchmark_readiness_commands.json`
- `readiness/cd38_benchmark_readiness_report.md`

默认只刷新准备状态，不运行外部工具、不重跑 benchmark；如果生成的 P2Rank/fpocket readiness report 没问题，再加 `--run_discovered` 或手动运行生成的 manifest。

如果你想按 manifest 批量复跑当前 CD38 benchmark panel，可执行：

```bash
python run_cd38_benchmark_manifest.py
```

这条命令会读取：

- `cd38_benchmark_manifest.csv`

并生成或更新：

- `results/manifest_run_summary.json`
- `results/cd38_benchmark_panel.csv`
- `results/cd38_benchmark_panel.md`

如果你想检查当前 CD38 结果对参数是否敏感，可执行：

```bash
python analyze_cd38_pocket_parameter_sensitivity.py ^
  --manifest_csv benchmarks\cd38\cd38_benchmark_manifest.csv ^
  --results_root benchmarks\cd38\results ^
  --truth_file benchmarks\cd38\cd38_active_site_truth.txt ^
  --out_dir benchmarks\cd38\parameter_sensitivity
```

这条命令会生成：

- `parameter_sensitivity/contact_cutoff_sensitivity.csv`
- `parameter_sensitivity/p2rank_rank_sensitivity.csv`
- `parameter_sensitivity/fpocket_pocket_sensitivity.csv`
- `parameter_sensitivity/method_consensus_threshold_sensitivity.csv`
- `parameter_sensitivity/overwide_penalty_sensitivity.csv`
- `parameter_sensitivity/cd38_pocket_parameter_sensitivity_summary.json`
- `parameter_sensitivity/cd38_pocket_parameter_sensitivity_report.md`

如果你想先规划“下一步该补哪个结构、哪个方法”，可执行：

```bash
python build_cd38_benchmark_expansion_plan.py
```

这条命令会读取：

- `cd38_structure_targets.csv`
- `cd38_benchmark_manifest.csv`
- `results/cd38_benchmark_panel.csv`

并生成：

- `expansion_plan/cd38_benchmark_expansion_plan.csv`
- `expansion_plan/cd38_benchmark_missing_actions.csv`
- `expansion_plan/cd38_benchmark_expansion_summary.json`
- `expansion_plan/cd38_benchmark_expansion_plan.md`

当前计划显示：`3ROP` / `4OGW` 的 ligand-contact 和 P2Rank baseline 已完成，真实 fpocket 输出仍缺；`3F6Y` 已列入 apo-like 测试结构，不做 ligand-contact baseline，后续需要补外部 P2Rank / fpocket 输出。

如果你想先判断目标结构是否真的含有适合 ligand-contact baseline 的活性口袋配体，可执行：

```bash
python inspect_cd38_ligand_candidates.py
```

这条命令会扫描目标结构 PDB 的 HETATM residues，并用 CD38 truth residues 做距离检查，生成：

- `ligand_candidates/cd38_ligand_candidates.csv`
- `ligand_candidates/cd38_recommended_ligand_candidates.csv`
- `ligand_candidates/cd38_structure_targets_suggested.csv`
- `ligand_candidates/cd38_ligand_candidate_summary.json`
- `ligand_candidates/cd38_ligand_candidate_report.md`

当前扫描结论：

- `3ROP`: `A:50A:301` 和 `A:NCA:302` 接触 CD38 truth residues，适合 ligand-contact baseline。
- `4OGW`: `A:NMN:401` 接触 CD38 truth residues，适合 ligand-contact baseline。
- `3F6Y`: 只检测到常见离子/水等 HETATM，未发现活性口袋 ligand-like candidate，因此不做 ligand-contact baseline，只保留 P2Rank/fpocket 测试。

如果你想比较同一结构上不同 pocket 方法是否指向同一个核心区域，可执行方法共识分析：

```bash
python compare_pocket_method_consensus.py ^
  --method ligand_contact=benchmarks\cd38\results\3ROP_ligand_contact_chainA_50A_NCA\predicted_pocket.txt ^
  --method p2rank=benchmarks\cd38\results\3ROP_p2rank_rank2_chainA\predicted_pocket.txt ^
  --truth_file benchmarks\cd38\cd38_active_site_truth.txt ^
  --out_dir benchmarks\cd38\results\3ROP_method_consensus_ligand_p2rank
```

```bash
python compare_pocket_method_consensus.py ^
  --method ligand_contact=benchmarks\cd38\results\4OGW_ligand_contact_chainA_NMN\predicted_pocket.txt ^
  --method p2rank=benchmarks\cd38\results\4OGW_p2rank_rank1_chainA\predicted_pocket.txt ^
  --truth_file benchmarks\cd38\cd38_active_site_truth.txt ^
  --out_dir benchmarks\cd38\results\4OGW_method_consensus_ligand_p2rank
```

方法共识分析会生成：

- `consensus_pocket_residues.txt`
- `union_pocket_residues.txt`
- `residue_method_membership.csv`
- `method_specific_residues.csv`
- `method_overlap_matrix.csv`
- `pocket_method_consensus_summary.json`
- `pocket_method_consensus_report.md`

说明：

- `overwide_pocket_score` 需要已知 truth，所以只适合 benchmark。
- 主链 `build_feature_table.py` 间接调用的几何特征会额外输出 `pocket_shape_overwide_proxy`，这个不依赖 truth，可用于普通运行中的 pocket 输入 QC。
- `rank_nanobodies.py`、`rule_ranker.py` 和 `run_recommended_pipeline.py` 已支持默认关闭的 `--pocket_overwide_penalty_weight`；如果真实 benchmark 显示 pocket 边界经常偏宽，可显式设为大于 0 的小权重做保守惩罚。
- `compare_pocket_method_consensus.py` 不替代单方法 benchmark，而是补充回答“哪些 residue 被多个方法共同支持，哪些 residue 只来自单一方法”。

当前已跑通的 baseline 结果见：

- `BASELINE_RESULTS.md`

当前仓库内已经保存了一组真实 P2Rank 结果：

- `results/3ROP_p2rank_rank2_chainA/`
- `results/4OGW_p2rank_rank1_chainA/`

当前仓库内也已经保存了两组结构化 ligand-contact baseline：

- `results/3ROP_ligand_contact_chainA_50A_NCA/`
- `results/4OGW_ligand_contact_chainA_NMN/`

当前仓库内已经保存了两组方法共识结果：

- `results/3ROP_method_consensus_ligand_p2rank/`
- `results/4OGW_method_consensus_ligand_p2rank/`

当前仓库内也已经保存了结构侧参数敏感性结果：

- `parameter_sensitivity/`

当前仓库内还保存了几何 proxy 校准报告：

- `proxy_calibration/cd38_proxy_calibration_report.md`
- `proxy_calibration/cd38_proxy_calibration_summary.json`
- `proxy_calibration/cd38_proxy_calibration_rows.csv`
- `proxy_calibration/cd38_proxy_threshold_candidates.csv`

运行方式：

```bash
python build_cd38_proxy_calibration_report.py
```

当前结论：`pocket_shape_overwide_proxy` 能把 `4OGW + P2Rank` 的偏宽 pocket 标出来，但当前只有 2 个结构、2 类方法且没有真实 fpocket 行，所以还不能据此改变全局默认排序策略；`pocket_overwide_penalty_weight` 继续默认 `0.0`，`0.15` 只作为敏感性复核参数。

# CD38 Baseline Results

这份说明记录的是当前仓库内已经实际跑通过的两组 CD38 ligand-contact baseline，以及两组真实 P2Rank baseline。

目的不是声称“已经完成 pocket finding”，而是先验证：

- CD38 的文献关键位点 baseline truth 可以在真实结构上正确匹配
- 从 ligand-bound 结构导出的 pocket residue 集合，能否覆盖这些已知位点

## 一页结论

| 对比项 | coverage | precision | 主要结论 |
|---|---:|---:|---|
| `3ROP ligand-contact` | `0.8571` | `0.4615` | 覆盖大多数 truth，`A:155` 只在 near-hit 下被覆盖 |
| `4OGW ligand-contact` | `1.0000` | `0.5000` | 覆盖全部 truth，但 `4OGW` 是 mutant，更适合做位置验证 |
| `3ROP P2Rank` | `1.0000` | `0.5000` | 能稳定抓到 CD38 活性口袋核心区域 |
| `4OGW P2Rank` | `1.0000` | `0.2692` | 召回稳定，但 pocket 明显偏宽 |
| `3ROP ligand-contact vs P2Rank consensus` | `0.8571` | `0.5455` | 共识更保守，但会丢掉 `A:155` |
| `4OGW ligand-contact vs P2Rank consensus` | `1.0000` | `0.5385` | 共识能收紧 P2Rank 偏宽输出，同时保留全部 truth |

当前最重要结论：CD38 适合继续作为 pocket benchmark；P2Rank 对核心位点召回稳定，但不同结构状态会影响 pocket 边界宽度，因此后续需要参数敏感性分析和更多结构验证。

## 当前仍缺的真实外部输出

当前仓库已经有 `3ROP/4OGW` 的 ligand-contact baseline 和真实 P2Rank baseline，但还没有真实 fpocket 输出行，也还缺 `3F6Y` 的真实 P2Rank/fpocket 输出。

最短下一步不是手工翻多个报告，而是先生成并查看 next-run runbook：

```bash
python build_cd38_external_tool_runbook.py
```

当前默认 next-run 动作是：

- `3ROP fpocket`
- `4OGW fpocket`
- `3F6Y fpocket`
- `3F6Y P2Rank`

如果要把输入包拿到 WSL/Linux 或另一台机器运行，执行：

```bash
python package_cd38_external_tool_inputs.py
```

生成的 transfer zip 会包含 `cd38_external_tool_next_run.md` 和 `run_cd38_external_next_benchmark.*`。真实外部工具输出带回后，再用 `finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir> --run_discovered --run_sensitivity` 导入并刷新 benchmark。

## 参数敏感性一页结论

当前已经运行：

- `analyze_cd38_pocket_parameter_sensitivity.py`

结果目录：

- `parameter_sensitivity/`

核心观察：

- ligand-contact cutoff 从 `3.5 A` 增大到 `5.5 A` 时，coverage 基本稳定，但 predicted residue 数量增加，precision 下降。
- `3ROP` 的 ligand-contact 在 `3.5 A` 到 `4.5 A` 都缺 `A:155`，说明它不是单纯靠放宽 cutoff 就能稳定 exact 命中的位点。
- `4OGW` 的 ligand-contact 在 `3.5 A` 已覆盖全部 truth，且 precision 比 `4.5 A` 更高。
- `3ROP + P2Rank` 对 rank 选择非常敏感：链 A truth 对应 `rank 2`，不是全局 `rank 1`。
- `4OGW + P2Rank rank 1` coverage 稳定为 `1.0`，但 precision 较低，说明它更偏召回优先。
- method consensus 在 `4OGW` 上有明显价值：`min_method_count=2` 能把 P2Rank 偏宽 pocket 收紧，同时保持 coverage `1.0`。
- method consensus 在 `3ROP` 上更保守：`min_method_count=2` precision 上升，但会丢掉 `A:155`；`min_method_count=1` coverage 更高但更宽。

## 使用的 truth

文件：

- `cd38_active_site_truth.txt`

当前 baseline truth 残基：

- `A:125`
- `A:127`
- `A:146`
- `A:155`
- `A:189`
- `A:193`
- `A:226`

## 结构 1: 3ROP

说明：

- RCSB: `3ROP`
- 标题：human CD38 in complex with compound CZ-50B
- pocket 预测方式：从链 `A` 上，与链 `A` 配体 `50A/NCA` 在 `4.5 A` 内接触的蛋白残基导出

导出的 predicted pocket：

- `A:124`
- `A:125`
- `A:126`
- `A:127`
- `A:145`
- `A:146`
- `A:189`
- `A:193`
- `A:196`
- `A:220`
- `A:221`
- `A:222`
- `A:226`

benchmark 结果：

- exact_truth_coverage: `0.8571`
- exact_predicted_precision: `0.4615`
- exact_jaccard: `0.4286`
- exact_f1: `0.6000`
- truth_near_coverage @ 4.5A: `1.0000`
- predicted_near_precision @ 4.5A: `1.0000`

解释：

- 7 个 truth 位点中，exact overlap 命中了 6 个
- `A:155` 没有被 exact 命中，但在 near-hit 指标下仍被覆盖
- 说明 ligand-contact baseline 对 CD38 核心 catalytic pocket 的覆盖已经比较接近，但仍会带出一部分周边 pocket residue

## 结构 2: 4OGW

说明：

- RCSB: `4OGW`
- 标题：human CD38 mutant complexed with NMN
- pocket 预测方式：从链 `A` 上，与链 `A` 配体 `NMN` 在 `4.5 A` 内接触的蛋白残基导出

导出的 predicted pocket：

- `A:124`
- `A:125`
- `A:126`
- `A:127`
- `A:129`
- `A:145`
- `A:146`
- `A:155`
- `A:189`
- `A:193`
- `A:220`
- `A:221`
- `A:222`
- `A:226`

benchmark 结果：

- exact_truth_coverage: `1.0000`
- exact_predicted_precision: `0.5000`
- exact_jaccard: `0.5000`
- exact_f1: `0.6667`
- truth_near_coverage @ 4.5A: `1.0000`
- predicted_near_precision @ 4.5A: `0.9286`

解释：

- 这组 baseline 覆盖了全部 7 个 truth 位点
- 但 `4OGW` 是 CD38 mutant 结构，链 `A:226` 在结构里对应的是 `GLN`，不是文献 baseline 里的 `GLU`
- 因此这组结果更适合用来做“口袋位置覆盖”验证，不适合直接当作野生型功能结论

## 结构 3: 3ROP + P2Rank

说明：

- RCSB: `3ROP`
- pocket 预测方式：真实 `P2Rank` 输出 `3ROP.pdb_predictions.csv`
- 选取 pocket：`rank 2`
- 链过滤：`A`
- 结果目录：`results/3ROP_p2rank_rank2_chainA/`

为什么选 `rank 2`：

- `3ROP` 是双链结构，`pocket1` 对应链 `B` 的活性口袋，`pocket2` 对应链 `A` 的同源活性口袋
- 当前 baseline truth 用的是链 `A`，因此这里应选 `rank 2 + chain A`

导出的 predicted pocket：

- `A:125`
- `A:126`
- `A:127`
- `A:129`
- `A:145`
- `A:146`
- `A:155`
- `A:156`
- `A:189`
- `A:193`
- `A:220`
- `A:221`
- `A:222`
- `A:226`

benchmark 结果：

- exact_truth_coverage: `1.0000`
- exact_predicted_precision: `0.5000`
- exact_jaccard: `0.5000`
- exact_f1: `0.6667`
- truth_near_coverage @ 4.5A: `1.0000`
- predicted_near_precision @ 4.5A: `1.0000`
- overwide_pocket_score: `0.2917`

解释：

- 7 个 truth 位点全部被 exact overlap 命中
- 额外预测出的残基主要是口袋边缘邻近位点，而不是完全偏离 catalytic cleft
- 从这组真实工具输出看，`P2Rank` 在 `3ROP` 上已经能稳定抓到 CD38 活性口袋核心区域
- 这也说明当前 CD38 baseline truth 可以拿来做真实 pocket tool 的第一轮筛查，而不只是做手工 sanity check

## 结构 4: 4OGW + P2Rank

说明：

- RCSB: `4OGW`
- pocket 预测方式：真实 `P2Rank` 输出 `4OGW.pdb_predictions.csv`
- 选取 pocket：`rank 1`
- 链过滤：`A`
- 结果目录：`results/4OGW_p2rank_rank1_chainA/`

导出的 predicted pocket：

- `A:52`
- `A:53`
- `A:123`
- `A:124`
- `A:125`
- `A:126`
- `A:127`
- `A:129`
- `A:145`
- `A:146`
- `A:155`
- `A:156`
- `A:157`
- `A:158`
- `A:173`
- `A:174`
- `A:175`
- `A:176`
- `A:183`
- `A:185`
- `A:186`
- `A:189`
- `A:193`
- `A:221`
- `A:222`
- `A:226`

benchmark 结果：

- exact_truth_coverage: `1.0000`
- exact_predicted_precision: `0.2692`
- exact_jaccard: `0.2692`
- exact_f1: `0.4242`
- truth_near_coverage @ 4.5A: `1.0000`
- predicted_near_precision @ 4.5A: `0.6923`
- overwide_pocket_score: `0.6175`

解释：

- 这组结果仍然完整覆盖了全部 7 个 truth 位点
- 但 `4OGW` 上的 `P2Rank pocket1` 比 `3ROP` 上对应 pocket 更宽，带进来了更多边缘残基
- 因此它适合说明 `P2Rank` 对 CD38 catalytic pocket 的召回是稳定的，但结构状态不同会明显影响 pocket 的“宽度”
- 这也说明后续如果要把 pocket 结果直接喂给下游 proxy，最好增加一层 pocket 大小/边界收紧，而不是只看是否命中 truth

## 方法共识分析: ligand-contact vs P2Rank

这一步不是新增一种 pocket finder，而是比较已有方法的 residue list 是否互相支持。它回答两个问题：

- 哪些 residue 同时被 ligand-contact 和 P2Rank 支持，可以当作更稳的核心 pocket。
- 哪些 residue 只被单一方法支持，可能代表 pocket 边界偏宽或结构状态差异。

### 3ROP 方法共识

输入：

- `results/3ROP_ligand_contact_chainA_50A_NCA/predicted_pocket.txt`
- `results/3ROP_p2rank_rank2_chainA/predicted_pocket.txt`

输出目录：

- `results/3ROP_method_consensus_ligand_p2rank/`

结果：

- union_residue_count: `16`
- consensus_residue_count: `11`
- method_specific_residue_count: `5`
- pairwise_jaccard: `0.6875`
- consensus_truth_coverage: `0.8571`
- consensus_truth_precision: `0.5455`
- missing_truth_risk: `0.1429`
- overwide_risk: `0.4545`
- missing_truth_residues: `A:155`

解释：

- 两种方法在 `3ROP` 上高度重合，11 个 residue 被两者共同支持。
- 共识 pocket 比单独 P2Rank 更保守，但会丢掉 `A:155` 这个 exact truth 位点。
- 因此 `3ROP` 的共识结果适合当“高置信核心 pocket”，但不适合直接替代召回优先的完整 pocket。

### 4OGW 方法共识

输入：

- `results/4OGW_ligand_contact_chainA_NMN/predicted_pocket.txt`
- `results/4OGW_p2rank_rank1_chainA/predicted_pocket.txt`

输出目录：

- `results/4OGW_method_consensus_ligand_p2rank/`

结果：

- union_residue_count: `27`
- consensus_residue_count: `13`
- method_specific_residue_count: `14`
- pairwise_jaccard: `0.4815`
- consensus_truth_coverage: `1.0000`
- consensus_truth_precision: `0.5385`
- missing_truth_risk: `0.0000`
- overwide_risk: `0.4615`

解释：

- `4OGW` 的共识 pocket 覆盖了全部 7 个 baseline truth 位点。
- P2Rank 单方法输出明显更宽，26 个 residue 中有 13 个是 P2Rank-specific。
- 方法共识能把过宽 pocket 收紧到 13 个 residue，同时保留全部 truth，说明共识分析对 4OGW 这类偏宽输出有实际价值。

## 当前结论

- CD38 很适合做单蛋白口袋准确性 benchmark
- 用 ligand-contact 导出的 pocket baseline，已经能稳定覆盖大部分甚至全部 baseline truth 位点
- 真实 `P2Rank` 现在已经在 `3ROP` 和 `4OGW` 上完成了首轮对比，结果显示对核心 truth 位点的覆盖是足够强的
- 但 `4OGW + P2Rank` 也提示了另一个问题：coverage 很稳，不代表 pocket 边界就足够紧，结构状态变化会显著影响 precision
- 方法共识分析进一步确认：`4OGW` 上的 P2Rank-specific residue 较多，共识 pocket 能明显收紧边界；`3ROP` 上共识更保守但会丢掉 `A:155`
- 为了量化这个问题，benchmark 现在新增了 truth-based `overwide_pocket_score`；主链几何特征也新增了不依赖 truth 的 `pocket_shape_overwide_proxy`
- 最小验证中，`3ROP + P2Rank` 的 `pocket_shape_overwide_proxy` 约为 `0.219`，`4OGW + P2Rank` 约为 `0.594`，与 benchmark 过宽评分方向一致
- 新增 `build_cd38_proxy_calibration_report.py` 后，当前 4 组 benchmark 已能自动重算 proxy 并输出阈值候选表；报告确认 `4OGW + P2Rank` 是唯一 truth-risk 行，默认 `0.55` 阈值能标出该风险
- 但校准报告同时指出证据等级仍为 `low`：只有 2 个结构、2 类方法、0 条真实 fpocket 行、truth-risk/non-risk split 为 `1/3`，因此 `pocket_overwide_penalty_weight` 仍应默认保持 `0.0`
- 当前 benchmark 仍更像“口袋位置覆盖验证”，不是 `fpocket/P2Rank` 的最终替代
- 当前仓库内已经把这 4 组基线结果都固化成了结构化结果目录，并额外生成了 `results/cd38_benchmark_panel.csv` 与 `results/cd38_benchmark_panel.md`
- 当前已经新增 `cd38_benchmark_manifest.csv` 和 `run_cd38_benchmark_manifest.py`，后续扩更多结构时优先新增 manifest 行，再批量复跑/汇总。
- 当前已经新增 `extract_fpocket_pocket_residues.py` 与 `run_cd38_fpocket_benchmark.py`，后续只要拿到真实 `fpocket` 的 `pocket*_atm.pdb`，即可按同口径加入 panel。
- 当前已经新增 `prepare_cd38_fpocket_panel.py`，可批量发现真实 `fpocket` 输出目录中的 `pocket*_atm.pdb`，自动生成 fpocket manifest，并可直接调用 manifest runner 加入 panel。
- 当前已经新增 `analyze_cd38_pocket_parameter_sensitivity.py`，并保存了 `parameter_sensitivity/` 结果，说明当前 CD38 结论中“coverage 是否稳定”和“pocket 是否偏宽”确实会受参数影响；其中 `fpocket_pocket_sensitivity.csv` 已作为真实 fpocket 输出接入后的固定落盘表。
- 当前已经新增 `build_cd38_external_benchmark_action_plan.py`，会把外部输出缺口收敛成 `action_plan/cd38_external_benchmark_action_plan.md/csv/json`；当前 4 个 benchmark blocker 是 `3ROP/4OGW fpocket` 和 `3F6Y P2Rank/fpocket`。
- 下一步最有价值的是：
  - 在 manifest 中继续加入更多 CD38 结构，而不是只看 `3ROP` / `4OGW`
  - 按 action plan 优先补真实 `fpocket` 输出目录，再交给 `prepare_cd38_fpocket_panel.py`，与 `P2Rank` / ligand-contact baseline 做同口径比较
  - 把当前参数敏感性分析扩展到更多结构和真实 `fpocket` 输出

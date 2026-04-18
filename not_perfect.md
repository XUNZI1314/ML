# 当前仍不完善的版块

这份文档记录的是：目前已经能跑、也已经能支撑验证的部分，但仍然偏启发式、近似或缺少系统性校准的地方。后续迭代优先从这些点补强。

## 当前推进看板

| 模块 | 当前状态 | 下一步 |
|---|---|---|
| 本地软件打开和运行 | 已有源码启动、桌面版、便携版和 standalone 单文件版 | 继续做跨机器兼容性验证和启动失败提示收敛 |
| 批量输入和运行体验 | 已支持 zip、目录扫描、标准 `A/result/vhh/CD38_x/pose/pose.pdb` 父目录自动识别、所有项目默认链角色 `antigen_chain=B / nanobody_chain=A`、`FINAL_RESULTS_MMPBSA.dat` 能量抽取、自动生成输入表、`pose_features.csv` 优先启动、队列、历史、导出、缺失文件自动定位、修复版 input_csv 下载、PASS/WARN/FAIL 质量门控和本批次结论摘要 | 后续补更多 sidecar 计算输出文件的自动指标抽取和人工确认编辑 |
| Demo 数据集与一键演示 | 已有 deterministic synthetic demo 数据生成、synthetic validation override、一键推荐 pipeline、Windows bat 入口、本地软件侧边栏载入/立即运行按钮、demo HTML 欢迎页、运行说明文件、示例结果解读页、摘要页一键打开导览、真实数据 starter 模板包和可运行 `MINI_PDB_EXAMPLE` toy PDB 示例包 | 后续可补更多真实 PDB 小样本示例 |
| Rule + ML 共识排序 | 已完成共识排名、分数解释卡片、本批次结论摘要、候选报告卡、候选横向对比解释、自定义候选对比、候选分组小结和低可信原因拆解 | 后续补真实验证结果对照和更完整 fpocket benchmark |
| 候选报告卡 | 已完成 HTML 报告卡、zip 导出和报告卡内嵌候选对比 | 后续可补批量 PDF、候选报告卡批注和更细的可视化分组 |
| AI 解释层 | 已完成基础版，默认离线，可选 OpenAI provider | 后续补候选级追问、本地 LLM provider 和答辩讲稿版摘要 |
| ML 架构说明 | 已新增 `ML.md`，解释输入、特征、Rule、MLP、聚合、共识、QC 和源码阅读顺序 | 后续随模型结构变动同步更新 |
| 下一轮实验建议 | 已完成 diversity-aware、预算 quota、实验计划单、人工覆盖、状态 ledger、本地编辑器、跨批次全局 ledger、真实验证回灌报告、验证证据审计、验证标签再训练入口、ledger 筛选图表、再训练前后对照报告、结果自动归档和长期趋势基础版 | 后续补更完整的真实 fpocket benchmark |
| CD38 pocket benchmark | 已有 ligand-contact、P2Rank、方法共识、结构侧参数敏感性、fpocket/P2Rank 批量接入入口、外部工具输入包、transfer zip、返回结果导入、导入前返回包安全门控、环境/输出预检、finalize 一键收尾、外部 benchmark action plan、next-run runbook、readiness 一键刷新、结构扩展计划、ligand candidate 扫描、结果汇总、公开结构 starter 一键入口、本地软件诊断页刷新/打包/返回包检查入口、返回包导入自测和外部链路一键自检 | 后续按 next-run runbook 实际运行外部 P2Rank/fpocket 并导入真实输出 |
| 几何 proxy | 已收紧 mouth/path，新增 pocket shape QC、几何 proxy 一致性审计、CD38 proxy 校准报告，并把 catalytic-anchor 3D shell pocket 诊断接入主程序 | 后续用更多真实 benchmark 校准阈值和权重，暂不把 catalytic-anchor 直接写入 rule 正式权重 |
| 参数敏感性分析 | ranking 权重/QC 惩罚基础版、CD38 结构侧基础版、fpocket pocket 表输出、fpocket/P2Rank 输入准备诊断、proxy 校准报告、外部工具输入包、transfer zip、返回结果导入、环境/输出预检、finalize 一键收尾、外部 benchmark action plan、结构扩展缺口表、readiness 总览和 ligand-contact 适用性检查均已完成 | 后续用更多结构和真实 fpocket/P2Rank 输出填充验证集 |
| provenance 可复现记录 | 已完成运行卡片、artifact manifest、输入文件引用 manifest、跨批次 lineage、lineage 图形化和 SHA256 完整性封存基础版 | 后续如需更强审计，再补带私钥数字签名 |

## 如果只回复“继续”，当前默认下一步

优先做“真实 fpocket benchmark / 更多结构扩展验证”：

1. [已完成基础增强] 支持 `experiment_plan_override.csv` 手工锁定或排除候选，例如强制 include/exclude/standby/defer。
2. [已完成基础增强] 给计划单增加实验成本、样本状态、负责人、完成状态和备注字段。
3. [已完成基础增强] 输出 `experiment_plan_state_ledger.csv`，作为可编辑、可继承的实验状态账本。
4. [已完成基础增强] 本地软件“本轮实验计划单”支持表格编辑、下载、保存并设置为下一轮 override。
5. [已完成基础增强] 新增 `build_experiment_state_ledger.py` 和本地软件历史页入口，可把多个历史运行的实验状态汇总成全局 ledger，并自动选择最近状态。
6. [已完成基础增强] 候选对比页支持用户手工选择 2 到 5 个 nanobody 做自定义对比，可预览、下载并保存。
7. [已完成基础增强] 新增 `build_experiment_validation_report.py`，可把全局 ledger 中明确的 `experiment_result` / `validation_label` 转成验证标签表、审计报告和可选的带实验标签特征表。
8. [已完成基础增强] 本地软件支持可配置 `label_col`；验证回灌报告可在传入 `pose_features.csv` 后生成 `pose_features_with_experiment_labels.csv`，并一键设为下一轮 `feature_csv + experiment_label` 输入。
9. [已完成基础增强] 本地软件全局 ledger 增加状态/结果/override/关键词筛选、可回灌标签计数和状态分布图。
10. [已完成基础增强] 新增 `build_validation_retrain_comparison.py`，可对照回灌前/回灌后两次运行的标签数、Rule/ML 对照指标、训练 loss、top-k 重叠和候选 rank delta。
11. [已完成基础增强] 本地软件“历史 -> 多运行对比”中新增验证回灌再训练前后对照报告入口，可生成并下载 CSV/JSON/Markdown。
12. [已完成基础增强] 新增 `build_result_archive.py`，可扫描 `local_app_runs` 生成运行归档索引、关键产物 manifest、验证回灌长期趋势表、跨批次 lineage 表、summary JSON 和 Markdown 报告。
13. [已完成基础增强] 本地软件“历史 -> 结果自动归档与长期趋势”支持一键刷新归档索引，并预览/下载运行索引、产物 manifest、长期趋势表和归档报告。
14. [已完成基础增强] 候选对比解释新增 `candidate_group_comparison_summary.csv`，可按 diversity/family/status/risk 分组汇总共同优势、共同风险和最高排名候选，并接入本地软件预览/下载。
15. [已完成基础增强] `prepare_cd38_fpocket_panel.py` 新增 `*.report.md` readiness report，可列出扫描目录、可运行 rows、跳过原因、推荐命令，并收紧 PDB ID 自动推断规则，避免把普通 `test` 文件夹误判成结构 ID。
16. [已完成基础增强] 新增 `build_cd38_benchmark_expansion_plan.py` 和 `benchmarks/cd38/cd38_structure_targets.csv`，可按结构/方法输出 CD38 benchmark 缺口表、missing actions、summary JSON 和 Markdown 计划。
17. [已完成基础增强] 新增 `inspect_cd38_ligand_candidates.py`，可扫描目标结构 HETATM 并判断是否适合 ligand-contact baseline；当前确认 `3ROP=50A/NCA`、`4OGW=NMN`，`3F6Y` 没有活性口袋 ligand candidate，因此从 ligand-contact 目标中移除。
18. [已完成基础增强] 新增 `prepare_cd38_p2rank_panel.py`，可批量发现 `P2Rank *_predictions.csv`，生成 manifest、summary JSON 和 readiness report，并支持 `--rank_by_pdb` 处理 active-site pocket rank 不是 1 的情况。
19. [已完成基础增强] 新增 `refresh_cd38_benchmark_readiness.py`，可一键刷新 ligand scan、扩展计划、P2Rank readiness 和 fpocket readiness，并生成总览 summary / commands / Markdown 报告；如果没有外部输出，会优先提示先生成外部工具输入包并运行环境/输出预检。
20. [已完成基础增强] 新增 `prepare_cd38_external_tool_inputs.py`，可为 `3ROP/4OGW/3F6Y` 生成外部 P2Rank/fpocket 输入 PDB、PowerShell/Bash 命令模板、输出目录约定、后续 readiness 刷新脚本、manifest、summary JSON 和 Markdown 操作说明；现在还会生成 expected return manifest 和 return checklist，明确外部机器跑完后应该带回哪些 `PDB × method` 输出；transfer 包内 Markdown 已改用包内相对路径，避免跨机器运行时被本机绝对路径误导。
21. [已完成基础增强] 新增 `check_cd38_external_tool_environment.py` 和 `external_tool_inputs/check_external_tool_environment.ps1`，可检查 `prank` / `fpocket` 是否在 PATH 中、PDB 输入是否存在、预期 P2Rank CSV 和 fpocket `pocket*_atm.pdb` 是否到位，并输出 preflight CSV/JSON/Markdown；路径解析已改成 `package_portable_first`，输入包移动后会优先看当前 package 内路径，避免旧绝对路径误导。
22. [已完成基础增强] 新增 `finalize_cd38_external_benchmark.py` 和 `external_tool_inputs/finalize_external_benchmark.ps1`，可一条命令串联返回包导入、preflight、readiness、可选 `--run_discovered` 导入和可选 `--run_sensitivity` 参数敏感性刷新；支持 `--import_source <returned_zip_or_dir>`，默认只检查不导入；当 readiness 发现 0 条可运行外部 rows 时会跳过 benchmark 汇总和参数敏感性刷新，避免把旧结果误当成新导入；finalize 报告会链接 import repair plan 和 action plan，直接提示缺失输出、优先级和对应模板。
23. [已完成基础增强] 新增 `package_cd38_external_tool_inputs.py`，可把外部工具输入包压成 transfer zip，默认只包含 PDB、PowerShell/Bash 模板和说明文件，排除 preflight/finalize 报告及旧输出，便于传到 Linux/WSL 或另一台机器运行。
24. [已完成基础增强] 新增 `import_cd38_external_tool_outputs.py`，可从返回目录或 zip 中安全导入 `p2rank_outputs/` 和 `fpocket_runs/*/*_out/`，默认不覆盖已有文件；现在支持返回包外面多包一层目录，并会生成 scan manifest，列出扫描文件、候选文件、忽略原因、导入 manifest、summary 和 Markdown 报告；同时输出 `source_diagnosis`，可直接识别“误把原始输入包当返回输出包导入”的情况；新增 coverage manifest，按 `PDB × method` 检查返回包覆盖和缺口；新增 repair plan CSV，把缺失项转成应运行模板、应返回路径和 dry-run 验证命令。
25. [已完成基础增强] 新增 `build_cd38_external_benchmark_action_plan.py`，把 expansion missing actions、preflight、readiness、expected returns 和 import repair plan 合并成 `benchmarks/cd38/action_plan/cd38_external_benchmark_action_plan.md/csv/json`；当前明确 4 个 benchmark completion blocker：priority `1` 的 `3ROP/4OGW fpocket`，priority `2` 的 `3F6Y P2Rank/fpocket`，另有 2 个 `3ROP/4OGW P2Rank` package reproducibility 行。
26. [已完成基础增强] 新增 `build_geometry_proxy_audit.py`，并接入推荐 pipeline、smoke test 和本地软件 QC 面板；用于检查 mouth/path/pocket/contact proxy 是否自洽，不改变分数和排序。
27. [已完成基础增强] 新增 `build_validation_evidence_audit.py`，并接入推荐 pipeline、smoke test 和本地软件排名结果页；用于检查 top 候选真实验证证据覆盖、正负标签平衡和待补行动项，不改变分数和排序。
28. [已完成基础增强] 新增 `demo_data_utils.py`、`run_demo_pipeline.py`、`run_demo_pipeline.bat`、`demo_report_utils.py` 和 `real_data_starter_utils.py`，可一键生成 synthetic demo 特征表、synthetic validation override，并跑完整推荐 pipeline，输出 `demo_outputs/DEMO_OVERVIEW.html`、`DEMO_README.md`、`DEMO_INTERPRETATION.md` 与 `REAL_DATA_STARTER/`；本地软件侧边栏也已支持“生成并载入 demo 输入”和“生成并立即运行 demo”，并会在 demo 运行输出目录写入同样的 demo 导览、说明、解读文件和真实数据迁移模板；摘要页“Demo 快速导览”可一键打开 HTML 导览和 starter 文件夹。
29. [已完成基础增强] `REAL_DATA_STARTER/` 现在会包含 `MINI_PDB_EXAMPLE/`，内置 12 个可解析 toy 复合物 PDB、`input_pose_table.csv`、`A:37-40` pocket 定义、catalytic 文件、ligand template 和说明文档；已验证 `build_feature_table.py` 12/12 行成功，轻量 `run_recommended_pipeline.py` 可完整生成 ranking/report/provenance。注意该示例只用于真实输入链路检查，不是生物学 benchmark。
30. [已完成基础增强] 新增 `run_cd38_public_starter.py`，可一键刷新公开 CD38 结构 starter：串联 panel 汇总、ligand candidate scan、参数敏感性、proxy calibration、外部工具输入包/preflight、readiness、action plan 和 next-run runbook；已验证 9/9 子步骤成功，当前 panel 为 4 行（`ligand_contact=2`、`p2rank=2`），action plan 仍显示 4 个 benchmark gap。
31. [已完成基础增强] 本地软件“诊断”页新增 CD38 public starter 面板，可一键刷新公开结构 starter，显示 panel rows、missing/pending、action plan 状态和 proxy 校准策略，并提供 starter 报告、action plan CSV、expected returns CSV、返回检查清单、next-run runbook、preflight/readiness 报告的一键打开或下载入口。
32. [已完成基础增强] 本地软件“诊断”页新增 CD38 外部工具 transfer/return 面板，可一键生成 `cd38_external_tool_inputs_transfer.zip`，下载或打开 transfer 目录；返回 zip/目录带回后可先 dry-run 检查，再导入并 finalize；已验证原始 transfer zip 会被识别为 `input_package_without_external_outputs`，不会误当成真实返回结果。
33. [已完成基础增强] 新增 `selftest_cd38_return_import_workflow.py` 和本地软件“返回包导入流程自测”入口，可生成 synthetic returned package fixture，并 dry-run 验证导入器能识别 6 个候选输出、expected coverage 达到 6/6；该自测只验证路径和 coverage，不作为真实 CD38 benchmark 证据。
34. [已完成基础增强] 新增 `build_cd38_return_package_gate.py` 和本地软件“返回包安全门控”，可把最新返回包 dry-run/import 结果判断为 `PASS_READY_FOR_IMPORT`、`WARN_PARTIAL_RETURN`、`FAIL_INPUT_PACKAGE`、`FAIL_SYNTHETIC_FIXTURE` 等状态；已验证原始 transfer zip 会被判为 `FAIL_INPUT_PACKAGE`，synthetic fixture 会被判为 `FAIL_SYNTHETIC_FIXTURE`。
35. [已完成基础增强] 本地软件“导入返回包并 finalize”新增导入前 gate 保护：正式导入前会先在隔离目录 dry-run 返回包并生成 gate，只有 `PASS_*` 状态才继续调用 finalize，避免原始输入包、自测 fixture 或不完整返回包污染本地 `external_tool_inputs`。
36. [已完成基础增强] `finalize_cd38_external_benchmark.py --import_source` 现在默认启用同样的导入前 gate；CLI 会先 dry-run 返回包并生成 gate，非 `PASS_*` 时跳过真实导入；保留 `--skip_import_gate` 作为人工复核后的专家绕过开关；新增 `--strict_import_gate`，适合 CI/自动化在 gate 非 PASS 时返回非零。已验证原始 transfer zip 在 CLI 下被拦截为 `FAIL_INPUT_PACKAGE`，strict 模式会返回非零。
37. [已完成基础增强] 新增 `selftest_cd38_external_workflow.py` 和本地软件“CD38 外部工具链路一键自检”，可一次验证 transfer zip 生成、原始 transfer zip strict gate 拦截、synthetic returned fixture 的 importer/gate 行为和 public starter 刷新；已验证整体状态 `pass`。
38. [已完成基础增强] 新增 `build_cd38_external_tool_runbook.py`，可把 action plan 转成外部机器可直接执行的 next-run runbook、CSV、PowerShell 和 Bash 脚本；`package_cd38_external_tool_inputs.py` 会在打包前自动刷新并把这些文件放入 transfer zip；本地软件 transfer 面板也提供 next-run 说明和脚本下载。当前默认选择 4 个 benchmark completion 动作：`3ROP/4OGW/3F6Y fpocket` 和 `3F6Y P2Rank`。
39. [已完成文档增强] 新增 `ML.md`，把当前 ML 本体解释为“结构几何特征 -> Rule baseline -> tabular MLP -> pose/conformer/nanobody 聚合 -> Rule+ML 共识 -> QC/解释/实验建议”的架构，并列出源码阅读顺序和边界。
40. [已完成基础增强] 新增 `result_tree_io.py` 和 `build_input_from_result_tree.py`，将真实数据目录固定为 `A/result/<nanobody_id>/<CD38_variant>/<pose_id>/<pose_id>.pdb`，可自动生成 `input_pose_table.csv`，并已接入本地软件目录/zip 自动导入逻辑。
41. [已完成基础增强] 标准目录扫描会读取每个 pose 目录下的 `FINAL_RESULTS_MMPBSA.dat`，解析最后一个 `DELTA TOTAL` 为 `MMPBSA_energy`；Rule/ML 聚合的 `top_k` 现在默认优先按 `MMPBSA_energy` / `mmgbsa` 低能量升序选择每个 `CD38_i` 下最低的 K 个 pose，没有能量列时才回退到分数降序。
42. [已完成基础增强] 本地软件导入目录时现在按 `pose_features.csv -> input_pose_table.csv -> A/result/ 自动生成 input_csv -> 普通 PDB 扫描` 的顺序处理；因此在标准 `A/result/` 数据结构下，不需要手工填写 `input_csv`。
43. [已完成基础增强] 结合真实 `A/rsite/result` 测试，所有项目默认链角色已统一固定为 `antigen_chain=B`、`nanobody_chain=A`；标准目录转换、推荐 pipeline、本地软件表单和自动输入表都会使用该默认值，特殊数据仍可通过参数覆盖。
44. [已完成基础增强] 新增 catalytic-anchor pocket 诊断：当提供 `catalytic_file` 时，主程序会以催化/功能残基为 3D anchor，在抗原结构中生成 4A/6A/8A shell pocket 特征，输出 `catalytic_anchor_*` 列，并在 Rule/ML 聚合结果中保留诊断均值；该模块用于解释和人工复核，暂不加入 rule baseline 正式权重。
45. 后续继续做更完整的真实 fpocket benchmark，并优先按 next-run runbook 实际运行外部模板补 `3ROP/4OGW/3F6Y` 的真实 fpocket 输出；`3F6Y` 还需要补真实 P2Rank 输出。

AI 解释层基础版已经完成，暂不作为默认下一步继续扩大；后续如果继续做 AI，优先做候选级追问和本地 LLM provider，而不是让 AI 参与模型打分。

需要真实外部数据时再继续“真实 fpocket / 更多结构扩展验证”：

1. 如果还没有真实 P2Rank/fpocket 输出，先运行 `python prepare_cd38_external_tool_inputs.py`，生成 PDB 输入包和 PowerShell 模板。
2. 先运行 `python check_cd38_external_tool_environment.py` 或 `external_tool_inputs/check_external_tool_environment.ps1`，明确当前缺的是外部工具还是输出文件；如果输入包被移动过，优先看 preflight CSV 中的 `pdb_input_source`、`p2rank_source`、`fpocket_source` 是否为 `package_portable`。
3. 先运行 `python build_cd38_external_tool_runbook.py`，把 action plan 转成 `external_tool_inputs/cd38_external_tool_next_run.md` 和 `run_cd38_external_next_benchmark.*`；默认只包含当前真正缺的 benchmark blocker。
4. 在已安装外部工具的环境优先运行 next-run 脚本；如果明确需要所有模板，再运行或改写 `external_tool_inputs/run_p2rank_templates.*` 与 `external_tool_inputs/run_fpocket_templates.*`；跑完后先对照 `cd38_external_tool_return_checklist.md` 或 `cd38_external_tool_expected_returns.csv`。
5. 如果要转移到另一台机器运行，先运行 `python package_cd38_external_tool_inputs.py` 生成 transfer zip；zip 会自动包含 next-run runbook 和脚本。
6. 如果拿回来的是整个目录或 zip，可直接运行 `python finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir>`；该命令现在会先做导入前 gate，只有 `PASS_*` 才正式导入输出并生成收尾报告。需要先看候选文件时再用 `python import_cd38_external_tool_outputs.py --source <returned_zip_or_dir> --dry_run`，若候选数为 0 则先看 `source_diagnosis`，再查看 `cd38_external_tool_output_import_scan.csv` 的忽略原因；若要看缺哪些结构/方法，看 `cd38_external_tool_output_import_coverage.csv`；若要直接按清单补缺，看 `cd38_external_tool_output_import_repair_plan.csv`。
7. 外部工具跑完后再次运行 preflight，确认输出到位。
8. 运行 `python finalize_cd38_external_benchmark.py` 或 `external_tool_inputs/finalize_external_benchmark.ps1`，统一生成接入收尾报告。
9. 如果仍缺输出，先看 `benchmarks/cd38/action_plan/cd38_external_benchmark_action_plan.md`，按 priority 补齐真实外部工具结果。
10. readiness report 无明显问题后，运行 `python finalize_cd38_external_benchmark.py --run_discovered` 批量加入同口径 panel；如果返回包已确认无误，也可一条命令运行 `python finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir> --run_discovered`。
11. 如需同时刷新参数敏感性，运行 `python finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity`，或在返回包导入时合并为 `python finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir> --run_discovered --run_sensitivity`。
   如果报告显示 `Runnable external benchmark rows: 0`，说明没有真实外部 rows 被导入，需先检查返回包是否包含 `p2rank_outputs/` 或 `fpocket_runs/*/*_out/`。
12. 在 CD38 manifest 中加入更多可用结构；对无 ligand candidate 的结构，只做 P2Rank/fpocket 这类 pocket finder 测试。
13. 根据更多结构结果再决定是否默认启用 pocket overwide penalty。

## 默认自动推进规则（2026-04-15）

后续如果用户只回复“继续”，默认按下面的执行约定推进，不需要每一轮重复指定：

1. 默认按 [not_perfect.md](not_perfect.md) 中“未完成”或“仍建议继续做”的事项顺序继续推进。
2. 优先做最小侵入的增量修改，不重写主链路，不随意换技术栈，不推倒现有界面和脚本。
3. 每一轮默认直接落代码、做必要验证、同步更新本文件；如 README 或运行说明已受影响，也一并更新。
4. 每一轮完成后，如果没有明显 blocker，下一次收到“继续”时直接进入下一个未完成项，不再重复做大段方案讨论。
5. 默认优先级顺序如下：
   - 先做会影响当前可用性和运行稳定性的项
   - 再做本地软件体验和导出一致性的项
   - 再做 ML 评估完整性、几何 proxy 收紧、解释能力增强
   - 最后再继续收紧更重的分发形态，例如“单文件自解压版的跨机器兼容性”
6. 只有在以下情况才暂停并显式询问用户：
   - 需要用户做方向选择，且不同选项会明显影响后续实现
   - 需要外部账号、人工登录、联网认证或仓库权限
   - 需要执行高风险或破坏性操作
   - 发现现有需求之间存在直接冲突
   - 发现代码现状与文档要求严重不一致，继续改会放大返工风险
7. 如果只是普通的小 blocker，优先自行解决；能通过补依赖、补文档、补兼容逻辑或补验证解决的，不单独停下来询问。
8. 每一轮结束时，默认只汇报四件事：
   - 做了什么
   - 验证结果
   - 当前还没完成的关键点
   - 下一步最自然要做什么

这条规则的目的是：后续尽量把“继续”收敛成真正的连续推进，而不是每轮都重新协商执行方式。

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

## 已完成的一键启动入口增强版（2026-04-15）

- 在 [start_local_app.bat](start_local_app.bat) 中把源码启动入口收敛成了一键脚本。
- 当前已经支持：
  - 自动优先使用仓库内 `.venv`
  - 自动调用 [ml_desktop_launcher.py](ml_desktop_launcher.py)
  - 缺少 `pythonw` 时自动回退到 `python`
- 这一步主要解决“源码目录虽然能跑，但启动方式仍然像开发脚本”的问题。

## 已完成的运行时依赖预检基础增强版（2026-04-15）

- 在 [runtime_dependency_utils.py](runtime_dependency_utils.py) 中新增了轻量依赖预检工具。
- 当前已经支持：
  - 在 [local_ml_app.py](local_ml_app.py) 的“运行前检查”里直接提示缺失的 `torch` / `biopython`
  - 在点击“立即运行”或“加入队列”前自动拦截缺失依赖，而不是等到后台任务跑失败
  - 在 [run_recommended_pipeline.py](run_recommended_pipeline.py) 中提前做运行时依赖检查，并输出明确的缺包错误
  - 在 [ml_desktop_launcher.py](ml_desktop_launcher.py) 的 `--selftest` 和实际启动前检查 `streamlit`
  - 在 [requirements.txt](requirements.txt) 中按 Python 版本区分 `torch` 版本，兼容当前 Python 3.14 环境
- 这一步主要解决“软件能打开，但真正进入 ML 训练或桌面启动时才晚报依赖错误”的问题。

## 已完成的本地交互壳增强版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中补齐了“更像本地软件”的三块基础体验：
  - 运行历史
  - 参数模板保存/载入
  - 失败诊断摘要
- 当前已经支持：
  - 扫描 `local_app_runs/` 下的历史运行记录
  - 一键载入某次历史结果，并恢复对应表单参数
  - 将当前表单参数保存成模板 JSON
  - 从已保存模板恢复常用参数组合
  - 在失败时展示失败阶段、输入定位、stderr 摘要和下一步建议
- 这一轮仍然没有改动现有 rule / ML 主链路，增强点仍然全部落在本地交互壳。

## 已完成的本地交互壳结果面板增强版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中继续补了“结果更直观”的几个低侵入增强点：
  - `training_summary.json` 摘要卡片
  - `train_log.csv` 训练曲线预览
  - ranking / pose 结果 Top-N 过滤
  - 运行产物清单
  - 一键打开最近运行目录 / 输出目录
- 当前已经支持：
  - 在“摘要”页直接查看训练模式、训练/验证样本数、最佳 epoch、最佳验证损失等关键指标
  - 在页面中预览训练曲线，而不需要手工打开 `train_log.csv`
  - 对 `nanobody_ranking.csv`、`pose_predictions.csv` 做前 N 行预览和关键 ID 过滤
  - 将 `pose_predictions.csv`、`training_summary.json`、`train_log.csv` 也纳入关键下载产物
  - 在本机直接打开最近一次运行目录和输出目录，便于人工复核和拷贝结果
- 这一轮依然没有改动算法主链路，增强点仍然全部落在本地交互壳。

## 已完成的本地交互壳汇总导出版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中补了“当前运行一键汇总打包”能力。
- 当前已经支持：
  - 对最近一次运行生成 `summary_bundle.zip`
  - 将关键结果、训练摘要、日志、metadata 和 manifest 一起打包
  - 在页面里直接下载该 zip，或打开汇总包目录
- 默认汇总包当前会优先包含：
  - `app_run_metadata.json`
  - `app_stdout.log`
  - `app_stderr.log`
  - `nanobody_rule_ranking.csv`
  - `nanobody_ranking.csv`
  - `pose_predictions.csv`
  - `training_summary.json`
  - `train_log.csv`
  - `bundle_manifest.json`
- 这一步依然只增强本地交互壳，不改动现有算法和推荐流程。

## 已完成的 orchestration 双入口版（2026-04-14）

- 在 [run_recommended_pipeline.py](run_recommended_pipeline.py) 中补了“CLI + 可调用函数”双入口。
- 当前已经支持：
  - 保留原有命令行入口不变
  - 通过 `run_recommended_pipeline(...)` 直接从 Python 内部调用推荐流程
  - 返回结构化 `summary`，便于本地交互壳直接消费
- 当前这层双入口仍然保留：
  - 函数入口适合后续 Python 内部复用
  - CLI 入口适合本地软件做后台运行、任务队列和停止任务
- [local_ml_app.py](local_ml_app.py) 当前默认已经切回 CLI 子进程执行模式：
  - 这样才能最小代价支持基础调度能力
  - 同时仍保留 `pipeline_kwargs` 和 CLI 命令写入 metadata，便于复现和排错
- 这一步仍然没有改动底层 rule / ML 子脚本，只是把 orchestration 层整理成更适合后续两种调用方式共存的结构。

## 已完成的桌面启动器基础版（2026-04-14）

- 新增 [ml_desktop_launcher.py](ml_desktop_launcher.py)，提供一个很薄的桌面 launcher。
- 新增 [build_desktop_app.bat](build_desktop_app.bat)，可直接构建 `dist/ML_Local_App.exe`。
- 当前已经支持：
  - 双击 `ML_Local_App.exe` 启动本地交互界面
  - 自动定位当前仓库根目录
  - 自动复用当前 `.venv`
  - 自动打开浏览器中的本地页面
  - 通过一个小型桌面控制窗口统一停止本地 Streamlit 服务
  - 使用 `--selftest` 做无界面自检
- 这一步的定位是“桌面启动器基础版”，不是把整个仓库彻底做成可脱离源码目录的全独立便携包。

## 已完成的便携目录版（2026-04-14）

- 新增 [build_portable_bundle.py](build_portable_bundle.py) 和 [build_portable_bundle.bat](build_portable_bundle.bat)。
- 当前已经支持：
  - 构建 `portable_dist/ML_Portable/`
  - 将桌面 launcher、运行源码和 `app/.venv` 一起整理成一个便携目录
  - 在便携目录里直接双击 `ML_Local_App.exe`
  - 让 exe 优先命中同级 `app/` 目录，而不是回退到原仓库
  - 通过 `--selftest` 验证便携目录内部的路径解析
- 这一步的定位是“可拷走的便携目录版”，不是“单个 exe 完全独立版”。

## 已完成的 zip 发布版（2026-04-14）

- 新增 [build_portable_release.py](build_portable_release.py) 和 [build_portable_release.bat](build_portable_release.bat)。
- 当前已经支持：
  - 单按钮重建桌面 launcher
  - 单按钮重建便携目录版
  - 自动生成 `portable_dist/ML_Portable_release.zip`
  - 自动生成 `portable_dist/ML_Portable_release.manifest.json`
  - 在 manifest 中记录 zip 的 SHA256、大小和条目清单
- 这一步的定位是“可直接分发的 zip 发布版”，但运行形态仍然是“解压后按便携目录版运行”。

## 已完成的单文件自解压基础版（2026-04-16）

- 新增 [build_standalone_onefile.py](build_standalone_onefile.py) 和 [build_standalone_onefile.bat](build_standalone_onefile.bat)。
- 当前已经支持：
  - 构建 `portable_dist/standalone_onefile/ML_Local_App_Standalone.exe`
  - 将 `app/` 运行源码、`app/.venv` 和桌面启动器一起内嵌进单个 exe
  - 通过 PyInstaller onefile 机制在运行时自动解压内嵌 `app/`
  - 让启动器优先识别内嵌 `app/` 根目录，而不是只能依赖外部仓库或便携目录
  - 生成 `ML_Local_App_Standalone.manifest.json`，记录单文件产物的版本号、SHA256 和内嵌条目
- 这一步的定位是“单文件自解压基础版”，已经不再要求外部再保留一个 `app/` 文件夹。
- 当前仍然不应过度宣称为“跨机器完全验证完毕”的最终形态，后续更适合继续推进的是：
  - 单文件自解压版的跨机器兼容性验证与收紧
  - 更接近真实物理含义的 mouth / path blocking proxy
  - 更完整的 ML 外部验证与 benchmark

## 已完成的单文件自解压校验增强版（2026-04-16）

- 在 [ml_desktop_launcher.py](ml_desktop_launcher.py) 中补了结构化自检输出。
- 当前已经支持：
  - 通过 `--selftest-json <path>` 将路径解析结果、依赖状态、`repo_root_source`、`python_executable` 等信息落成 JSON
  - 对源码版、桌面 launcher exe、便携版 exe 和单文件版 exe 统一使用同一套自检 payload
- 新增 [validate_standalone_onefile.py](validate_standalone_onefile.py)。
- 当前已经支持：
  - 把 `ML_Local_App_Standalone.exe` 复制到系统临时目录
  - 在脱离宿主仓库目录的情况下执行 `--selftest-json`
  - 自动校验 `repo_root_source = meipass_app`
  - 自动校验 `python_executable` 来自内嵌 `app/.venv`
  - 自动校验当前没有回退到宿主仓库目录
  - 将验证结果写入 `portable_dist/standalone_onefile_validation/standalone_validation_latest.json`
- 这一步主要解决“单文件版虽然已经能构建，但还缺少一条可重复执行的自动化验证链，难以确认它是否真的在脱离源码目录后仍使用内嵌 app 运行”的问题。
- 这一轮做完后，更适合继续推进的点主要是：
  - 真实外部 Windows 机器上的兼容性验证
  - 更接近真实物理含义的 mouth / path blocking proxy
  - 更完整的 ML 外部验证与 benchmark

## 已完成的分组交叉验证 benchmark 基础增强版（2026-04-16）

- 新增 [benchmark_pose_pipeline.py](benchmark_pose_pipeline.py)。
- 当前已经支持：
  - 基于 `nanobody_id` 的分组交叉验证，而不再只看单次 train/val 切分
  - 每折独立训练 [train_pose_model.py](train_pose_model.py) 并在 held-out fold 上输出 `pose_cv_predictions.csv`
  - 直接复用 [rank_nanobodies.py](rank_nanobodies.py) 和 [rule_ranker.py](rule_ranker.py)，生成 ML / rule 两条 nanobody benchmark 对照
  - 自动输出 `fold_metrics.csv`、`nanobody_benchmark_table.csv`、`benchmark_summary.json`、`benchmark_report.md`
  - 自动输出 pose 和 nanobody 两层的 reliability curve / ECE / Brier
  - 自动输出 `geometry_proxy_benchmark.csv`，用于直接看各个几何 proxy 在 held-out fold 上的单项表现
- 这一步主要解决“当前已有训练和排序主链，但还缺一条低侵入、可重复、能直接看 ML 与 rule benchmark 表现的系统评估链”的问题。
- 这一轮做完后，更适合继续推进的点主要是：
  - 使用真实实验数据做独立 test / 外部 benchmark，而不只是交叉验证
  - 把 benchmark 结果继续反哺到几何 proxy 收紧和权重回归
  - 更接近真实物理含义的 mouth / path blocking proxy

## 已完成的几何 proxy 收紧基础增强版（2026-04-16）

- 在 [geometry_features.py](geometry_features.py) 中继续收紧了两条最关键的静态 proxy 合成逻辑。
- 当前已经支持：
  - 让 `mouth_occlusion_score` 更强调 `mouth_axis_block_fraction` 与 `mouth_aperture_block_fraction` 的一致阻断，而不是让单一子信号过强地主导总分
  - 让 `ligand_path_block_score` 在保留多候选出口阻断共识的同时，显式考虑“是否仍存在一条较开放的逃逸路径”
  - 保持现有输出字段和下游脚本接口不变，不增加主链路迁移成本
- 这一步主要解决“已有 mouth/path proxy 虽然能区分很多情形，但对单侧阻断或单一路径仍开放的情况还可能偏乐观”的问题。
- 这一轮做完后，更适合继续推进的点主要是：
  - 用真实 benchmark 检查这些 proxy 的权重是否还需要继续回归
  - 继续把 mouth/path proxy 往更接近真实物理意义的方向收紧

## 已完成的 CD38 口袋准确性 test 脚手架（2026-04-16）

- 新增 [benchmark_cd38_pocket_accuracy.py](benchmark_cd38_pocket_accuracy.py)。
- 新增 [run_cd38_p2rank_benchmark.py](run_cd38_p2rank_benchmark.py)，把 `P2Rank predictions.csv -> residue 提取 -> CD38 benchmark` 串成单条命令。
- 新增 [run_cd38_ligand_contact_benchmark.py](run_cd38_ligand_contact_benchmark.py)，把 `ligand-bound 结构 -> ligand-contact residue -> CD38 benchmark` 串成单条命令。
- 新增 [extract_fpocket_pocket_residues.py](extract_fpocket_pocket_residues.py) 和 [run_cd38_fpocket_benchmark.py](run_cd38_fpocket_benchmark.py)，把 `fpocket pocket*_atm.pdb -> residue 提取 -> CD38 benchmark` 串成单条命令。
- 新增 [run_cd38_benchmark_manifest.py](run_cd38_benchmark_manifest.py) 和 [benchmarks/cd38/cd38_benchmark_manifest.csv](benchmarks/cd38/cd38_benchmark_manifest.csv)，支持按 manifest 批量复跑 ligand-contact / P2Rank / fpocket / residue_file benchmark。
- 新增 [summarize_cd38_benchmarks.py](summarize_cd38_benchmarks.py)，把 `benchmarks/cd38/results/` 下的结构化结果目录聚合成总表。
- 新增 [benchmarks/cd38/cd38_active_site_truth.txt](benchmarks/cd38/cd38_active_site_truth.txt) 和 [benchmarks/cd38/README.md](benchmarks/cd38/README.md)。
- 当前已经支持用单个 CD38 结构对 pocket residue 结果做 exact overlap / near-hit coverage / precision / Jaccard / F1 检测。
- 当前已经支持直接下载 RCSB PDB（如 `3F6Y`）做本地 benchmark。
- 当前已经支持输出逐残基命中表、missed truth 表、extra predicted 表和 near-hit 距离表。
- 已经在 `3ROP` 和 `4OGW` 上跑通 ligand-contact baseline，对 CD38 truth pocket 做了第一轮真实结构 sanity check。
- 已经在 `3ROP` 和 `4OGW` 上跑通真实 `P2Rank` 输出，当前仓库结果目录分别是 `benchmarks/cd38/results/3ROP_p2rank_rank2_chainA/` 与 `benchmarks/cd38/results/4OGW_p2rank_rank1_chainA/`。
- 这两组真实 `P2Rank` baseline 的 `exact_truth_coverage` 都是 `1.0`，说明 CD38 baseline truth 已经能用于真实 pocket tool 的首轮筛查。
- 当前仓库内已经把这 4 组结果都固化成了结构化结果目录，并生成了 `benchmarks/cd38/results/cd38_benchmark_panel.csv` 与 `cd38_benchmark_panel.md`。
- 当前新增的 `4OGW + P2Rank` 也暴露了一个更具体的问题：coverage 虽稳，但 pocket 边界会因为结构状态变化而明显变宽，precision 会下降到 `0.2692`。
- 新增 truth-based `overwide_pocket_score`，专门量化“找到了口袋但边界过宽”的风险；当前 `4OGW + P2Rank` 的该分数为 `0.6175`，明显高于其他 3 组。
- 在 [geometry_features.py](geometry_features.py) 中新增无 truth 依赖的 `pocket_shape_*` 特征，主链运行时会输出 `pocket_shape_overwide_proxy` / `pocket_shape_tightness_proxy`。
- 最小验证显示 `3ROP + P2Rank` 的 `pocket_shape_overwide_proxy` 约为 `0.219`，`4OGW + P2Rank` 约为 `0.594`，能捕捉当前 panel 暴露的过宽问题。
- 在 [build_feature_table.py](build_feature_table.py) 的 `feature_qc.json` 中新增 `pocket_shape_qc`，会统计高 overwide 行数、占比、p95/max 和前 10 个最值得复核的 pose。
- 在 [local_ml_app.py](local_ml_app.py) 的 QC/Warning 面板和展示摘要 HTML 中新增 Pocket Shape QC 展示。
- 在 [rank_nanobodies.py](rank_nanobodies.py) 与 [rule_ranker.py](rule_ranker.py) 中保留 top-k 的 `pocket_shape_*` 聚合字段，并在 `explanation` 中前置提示“pocket 定义偏宽，建议复核口袋边界”。
- 在 [ranking_common.py](ranking_common.py)、[rank_nanobodies.py](rank_nanobodies.py)、[rule_ranker.py](rule_ranker.py)、[calibrate_rule_ranker.py](calibrate_rule_ranker.py)、[run_recommended_pipeline.py](run_recommended_pipeline.py) 和 [local_ml_app.py](local_ml_app.py) 中新增默认关闭的 `pocket_overwide_penalty_weight` / `pocket_overwide_threshold`。
- 当前默认 `pocket_overwide_penalty_weight=0.0`，所以不改变已有排名；如果后续 benchmark 证明确实需要，可在本地软件或 CLI 中显式启用小权重。
- 已做函数级和 CLI 级最小验证：默认权重下分数保持不变；设置 `pocket_overwide_penalty_weight=0.2` 后，高 overwide 的 NB1 被扣分，低 overwide 的 NB2 不受影响；`rule_ranker.py` 与 `rank_nanobodies.py` 都能落盘 `pocket_overwide_penalty` 字段。
- 已做 fpocket 解析器最小验证：构造 `pocket1_atm.pdb` 后能提取 `A:125/A:127`，并能通过 [run_cd38_fpocket_benchmark.py](run_cd38_fpocket_benchmark.py) 生成完整 CD38 benchmark 输出。
- 新增 [prepare_cd38_fpocket_panel.py](prepare_cd38_fpocket_panel.py)，可扫描一批真实 `fpocket` 输出目录，自动发现 `pocket*_atm.pdb`，生成 `fpocket_discovered_manifest.csv`、summary JSON 和 readiness report，并可通过 `--run` 直接批量加入 CD38 benchmark panel。
- 已做 fpocket panel 准备层最小验证：临时构造两个 `pocket*_atm.pdb` 文件后，脚本能生成两条 fpocket manifest row，并能以 `--dry_run --run` 串到 [run_cd38_benchmark_manifest.py](run_cd38_benchmark_manifest.py)。
- 已做 manifest runner 验证：`python run_cd38_benchmark_manifest.py` 能识别现有 4 组结果、跳过已存在输出、重新生成 `cd38_benchmark_panel.csv/md` 和 `manifest_run_summary.json`。
- [analyze_cd38_pocket_parameter_sensitivity.py](analyze_cd38_pocket_parameter_sensitivity.py) 已新增 `fpocket_pocket_sensitivity.csv` 输出；当前没有真实 fpocket baseline 行时该表为空但表头稳定，后续接入真实 fpocket manifest 后会自动填充。
- 已做 fpocket 敏感性非空最小验证：临时构造 fpocket manifest 和 `predicted_pocket.txt` 后，能输出 fpocket pocket choice 的 coverage、precision、F1、missing truth 和 utility。
- 这一步主要解决“单个 CD38 不适合直接跑完整抗体主链，但很适合先拿来验证 pocket finding 是否覆盖已知关键位点”的问题。
- 当前仍未做完的主要不是“有没有批量扩展入口”，而是在 manifest 中加入更多 CD38 结构并实际跑出结果，避免只看 `3ROP/4OGW`。
- 当前仍未做完的主要不是“有没有 fpocket 接入口或批量接入口”，而是拿真实 `fpocket` 输出文件跑一组同口径 baseline。
- 当前仍未做完的主要不是“有没有轻量 ranker 惩罚项”，而是用更多 CD38/非 CD38 benchmark 决定推荐权重是否仍应保持默认 `0.0`。

## 已完成的版本化发布元数据版（2026-04-14）

- 新增 [app_metadata.py](app_metadata.py)，把应用名称、版本号、通道和发布日期统一收口。
- 当前已经支持：
  - 本地页面标题显示版本号
  - 桌面启动器窗口显示版本号和发布通道
  - `--selftest` 输出版本号
  - `portable_dist/ML_Portable/APP_VERSION.json`
  - `portable_dist/ML_Portable_release.manifest.json` 中记录版本号和发布通道
- 这一步的意义主要是让桌面版、便携版和 zip 发布包之间有统一版本追踪，不再是“只有文件名不同”。

## 已完成的品牌图标基础版（2026-04-14）

- 新增 [generate_brand_assets.py](generate_brand_assets.py)。
- 当前已经生成：
  - `assets/app_icon.png`
  - `assets/app_icon.ico`
- 当前已经接入：
  - Streamlit 页面 favicon
  - 桌面启动器窗口图标
  - `PyInstaller` 生成的桌面 exe 图标
  - 便携目录版中的 `app/assets/`
  - zip 发布包中的 `app/assets/`
- 这一步的目标不是做完整品牌系统，而是先让桌面版和发布包不再使用默认图标，提升软件完成度。

## 已完成的 GitHub 自动发布工作流版（2026-04-14）

- 新增 [desktop-release.yml](.github/workflows/desktop-release.yml)。
- 当前已经支持：
  - 在 GitHub Actions 页面手动触发桌面版构建
  - 在推送 `v*` tag 时自动构建桌面版和便携发布包
  - 自动上传 `ML_Local_App.exe`
  - 自动上传 `ML_Portable/` 目录产物
- 自动上传 `ML_Portable_release.zip`
- 在 tag 场景下自动创建 GitHub Release 并附带关键产物
- 这一步的意义是把“本地能构建”推进到“GitHub 仓库也能稳定产出可下载发布物”。

## 已完成的上传体验增强版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中继续补了“更方便上传和导入”的一轮低侵入增强。
- 当前已经支持：
  - 上传单个 `zip` 数据包并自动识别 `input_csv` / `pose_features.csv` / `pocket` / `catalytic` / `ligand` 文件
  - 直接扫描本地数据目录并自动识别同类输入，不需要先手工压缩 zip
  - 自动把识别到的路径回填到现有表单，不需要重写页面结构
  - 在页面里直接下载 `input_pose_table.csv` 与 `pose_features.csv` 模板
  - 在运行前检查主输入必需列、`label` 状态和默认文件来源
  - 在主页面直接看到“当前输入状态”，减少误传文件后才发现问题
- 这一步仍然没有改动现有算法主链路，增强点继续只落在本地交互壳。
- 这一轮做完后，当前剩余更适合继续推进的点主要是：
  - 支持把当前结果页导出成更适合演示的 HTML/PDF 摘要

## 已完成的基础任务调度版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中补了一个最小可用的任务调度层。
- 当前已经支持：
  - 把当前表单配置直接启动为后台运行任务
  - 把多组配置加入队列后顺序执行
  - 在页面中查看当前活动任务、队列长度和待执行项
  - 手动刷新运行状态
  - 停止当前后台运行任务，同时保留剩余队列
- 这一轮的实现方式仍然是复用现有 CLI 命令，不改分析脚本本身，只把本地交互壳从“同步单次执行”推进到“基础调度执行”。
- 这一步做完后，当前更适合继续推进的点主要是：
  - 进一步补失败行/warning 的可读展示

## 已完成的展示摘要 HTML 导出版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中补了“当前运行单文件展示摘要 HTML”导出。
- 当前已经支持：
  - 从最近一次运行直接生成 `presentation_summary.html`
  - 把关键指标、训练摘要、规则/ML 排名预览、诊断摘要和执行报告整理到一个 HTML 页面
  - 在页面里直接下载 HTML
  - 在本机直接打开该 HTML
  - 通过浏览器“打印为 PDF”做轻量 PDF 导出，而不额外引入新的 PDF 依赖
- 这一步仍然没有改动分析主链路，只是在本地交互壳的导出层补了一种更适合展示和汇报的结果形态。
- 当前还没做的只剩更进一步的版式增强，例如：
  - 更强的多运行汇总导出

## 已完成的 QC / Warning 展示版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中补了面向 `feature_qc.json` 和 `pose_features.csv` 的可读展示。
- 当前已经支持：
  - 在独立 `QC/Warning` 页签中查看处理摘要
  - 展示 `failed_rows`、`rows_with_warning_message` 和 `status_counts`
  - 直接预览 failed 行列表
  - 直接预览 warning 行列表
  - 展示全空列和近常量列提示
  - 在摘要页给出 failed/warning 的快速提示
- 这一步继续遵守最小侵入原则，没有改分析脚本，只把现有输出结果真正接到了本地软件页面上。
- 这一轮做完后，本地软件层更适合继续推进的点主要是：
  - 更强的 HTML 版式增强

## 已完成的多运行对比基础版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中新增了一个低侵入的“运行对比”页签。
- 当前已经支持：
  - 从历史运行记录里一次选择多条运行结果做并排对比
  - 在同一张表里对比执行状态、label 情况、rule/ML 对照指标、训练摘要和 QC 摘要
  - 在页面里切换主要对比指标，例如 `calibrated_rank_spearman`、`best_val_loss`、`failed_rows`、`warning_rows`
  - 用基础柱状图快速比较关键数值列
  - 直接下载当前对比表 CSV
- 这一步的定位仍然是“基础版多运行对比”，不是完整的实验追踪系统。
- 后续如果继续补强，更适合做的是：
  - 更细的差异归因
  - 跨批次趋势聚合

## 已完成的多运行对比 HTML 导出版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中继续补了多运行对比的展示导出层。
- 当前已经支持：
  - 基于当前选中的历史运行生成 `history_compare_summary.html`
  - 同时落出 `history_compare_table.csv`
  - 在页面中直接下载对比 HTML
  - 在本机直接打开当前对比 HTML
  - 继续通过浏览器“打印为 PDF”做轻量导出
- 这一步仍然保持最小侵入，没有引入新的 PDF 或浏览器自动化依赖。
- 这一轮做完后，更适合继续推进的点主要是：
  - PDF 版式进一步美化

## 已完成的自动化 PDF 导出基础增强版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 中补了真正自动化的 PDF 导出。
- 当前已经支持：
  - 从最近一次运行直接生成 `presentation_summary.pdf`
  - 从当前多运行对比直接生成 `history_compare_summary.pdf`
  - 在页面里直接下载 PDF
  - 在本机直接打开 PDF
  - 将单次运行 PDF 一并纳入当前运行汇总包
- 这一轮实现仍然遵守最小侵入原则：
  - 复用现有 `Pillow` 能力生成 PDF
  - 没有额外引入新的 PDF 依赖
  - 没有改动算法主链路
- 这一步做完后，更适合继续推进的点主要是：
  - 更细的差异归因
  - 单个 exe 完全独立运行

## 已完成的 PDF 版式增强基础版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 中继续补了自动化 PDF 的展示版式。
- 当前已经支持：
  - 首页标题区与版本信息
  - 指标卡片式摘要布局
  - 分区卡片和轻量配色分层
  - 表格块与等宽字体预览
  - 页脚页码与统一导出头部
  - 单次运行摘要 PDF 与多运行对比 PDF 共用同一套版式体系
- 这一步仍然是“基础版式增强”，不是最终的高保真排版系统。
- 这一轮做完后，更适合继续推进的点主要是：
  - 跨批次趋势聚合
  - 单个 exe 完全独立运行

## 已完成的多运行趋势与差异解释基础增强版（2026-04-14）

- 在 [local_ml_app.py](local_ml_app.py) 的“运行对比”页签中继续补了趋势和差异解释层。
- 当前已经支持：
  - 按 `started_at` 从早到晚查看主指标趋势
  - 选择一个基准运行做 run-to-run 对比
  - 自动生成差异解释文字
  - 展示相对基准的差异表，包括主指标、AUC、`best_val_loss`、`failed_rows`、`warning_rows` 的变化
  - 将这些差异解释和趋势快照一并带入对比 HTML 导出
- 这一步的定位仍然是“基础增强版”，重点是让实验对比更直观，而不是替代完整实验追踪平台。
- 这一轮做完后，更适合继续推进的点主要是：
  - 跨批次趋势聚合

## 已完成的更细差异归因基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的“运行对比”页签中继续补了更细的差异归因层。
- 当前已经支持：
  - 针对每个运行生成 `attribution_tag`
  - 拆分主要正向驱动
  - 拆分主要负向拖累
  - 输出一条归因摘要
  - 在页面里按运行查看归因明细
  - 在归因总表中并排查看各运行的主要驱动与摘要
  - 将归因总表和运行级归因明细一起带入 HTML / PDF 导出
- 这一步依然是基于现有 summary/QC/训练摘要做启发式解释，不是严格因果归因系统。
- 这一轮做完后，更适合继续推进的点主要是：
  - 跨批次趋势聚合
  - 单个 exe 完全独立运行

## 已完成的跨批次趋势聚合基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的“运行对比”页签中继续补了跨批次趋势聚合层。
- 当前已经支持：
  - 按 `started_at` 日期把选中的运行聚合成批次
  - 查看每个批次的运行数、成功运行数、clean run 数
  - 查看主指标批次均值、主指标批次最优值、`Best Val Loss` 均值
  - 查看每个批次的 `Failed Rows` / `Warning Rows` 合计
  - 输出批次级观察文字，帮助快速判断哪个批次更稳、哪个批次噪声更大
  - 直接下载批次聚合 CSV
  - 将批次聚合总表和批次级观察一起带入 HTML / PDF 导出
- 这一步的定位仍然是“基础增强版”，当前 batch 的定义是 `started_at` 的日期维度，不是完整实验追踪平台中的正式批次实体。
- 这一轮做完后，更适合继续推进的点主要是：
  - 单个 exe 完全独立运行

## 已识别的本地软件功能优化路线（2026-04-15）

1. [已完成基础增强] 历史记录 / 运行对比筛选增强。
2. [已完成基础增强] 队列项单条移除与顺序调整。
3. [已完成基础增强] 从历史运行一键复制为新任务或重新入队。
4. [已完成基础增强] 排名结果 / Pose 结果导出当前筛选视图。
5. [已完成基础增强] 导出目录与历史运行清理工具。

这些点都属于“继续把本地软件做顺手”的功能型优化，不改算法主链路，不换技术栈，优先做低侵入增强。

## 已完成的历史记录与运行对比筛选增强基础版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 中补了历史记录与运行对比候选集的筛选层。
- 当前已经支持：
  - 在左侧“运行历史”选择器里按关键词快速筛选历史记录
  - 在“历史”页按关键词 / 状态 / `start_mode` 过滤记录
  - 在“历史”页导出当前筛选结果 CSV
  - 在“运行对比”页先筛选候选运行，再进入多运行对比
  - 在历史筛选结果里直接看到可见运行数、成功运行数和失败/取消运行数
- 这一步的目标是解决历史运行变多后选择器、历史页和对比页越来越拥挤的问题，不改已有运行链路和导出链路。
- 这一轮做完后，更适合继续推进的点主要是：
  - 导出目录与历史运行清理工具

## 已完成的队列项单条移除与顺序调整基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的“运行调度”区域补了队列项级别的管理动作。
- 当前已经支持：
  - 对队列中的单条任务做上移
  - 对队列中的单条任务做下移
  - 直接移除选中的队列项
  - 调整后自动刷新当前队列显示和选中位置
- 这一步主要解决“当前队列只能整队清空，不能做细粒度调整”的问题，仍然复用现有后台 CLI 调度模式，不改现有执行主链路。
- 这一轮做完后，更适合继续推进的点主要是：
  - 导出目录与历史运行清理工具

## 已完成的历史运行复制入队基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的左侧“运行历史”区域补了“复制并入队”动作。
- 当前已经支持：
  - 从所选历史运行中提取已保存的表单配置
  - 自动回填该次历史运行解析过的输入路径
  - 直接生成一个新的 rerun 任务名并加入当前队列
  - 让用户在不重新上传、不重新填写参数的前提下快速重跑历史配置
- 这一步主要解决“想基于旧任务快速重跑，但当前只能先载入再手动加入队列”的问题，仍然复用现有表单和后台 CLI 调度链路。
- 这一轮做完后，更适合继续推进的点主要是：
  - 导出目录与历史运行清理工具

## 已完成的筛选视图导出基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的“排名结果”和“Pose 结果”页补了当前筛选视图导出。
- 当前已经支持：
  - 将当前筛选后的 ML 排名结果直接下载为 CSV
  - 将当前筛选后的 Rule 排名结果直接下载为 CSV
  - 将当前筛选后的 Pose 结果直接下载为 CSV
  - 导出的内容与当前筛选条件保持一致，而不是固定导出原始全量表
- 这一步主要解决“页面里虽然已经能筛选和预览，但用户还需要手工再处理一次才能拿到当前视图数据”的问题。

## 已完成的导出目录与历史运行清理基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的“历史”页补了应用产物清理工具。
- 当前已经支持：
  - 统计历史运行目录、对比导出目录和导入缓存目录的数量与占用空间
  - 预览每个可清理目录的类别、名称、创建时间、大小和路径
  - 先打开所选目录，再决定是否清理
- 仅允许清理由应用自身生成的 `local_app_runs` 子目录
- 删除前必须手工输入 `DELETE` 确认
- 自动阻止删除当前正在运行的任务目录
- 这一步主要解决“本地软件越用越久后，运行目录、compare 导出和导入缓存不断堆积”的问题，同时尽量把误删风险收紧在应用自管目录内部。

## 已识别的结果页深化优化路线（2026-04-15）

1. [已完成基础增强] ranking / pose 结果页列选择、排序与当前视图导出。
2. [已完成基础增强] 运行对比页列裁剪与当前视图导出。
3. [已完成基础增强] ranking / pose 结果页按数值阈值做细筛选。

这些点仍然属于“结果已经能看，但还可以更顺手”的增强，不改算法输出，只增强结果浏览和导出层。

## 已完成的结果页列选择与当前视图导出基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的 `ML 排名`、`Rule 排名` 和 `Pose 结果` 页补了统一的数据视图控制。
- 当前已经支持：
  - 在结果页自定义显示列
  - 在结果页切换排序列
  - 选择升序或降序显示
  - 分别导出“当前可见列视图 CSV”和“当前全筛选结果 CSV”
  - 在不改原始结果文件的情况下做更适合汇报或复核的局部导出
- 这一步主要解决“结果表列很多、页面里想聚焦重点列，但当前只能看全表或导出全表”的问题。
- 这一轮做完后，更适合继续推进的点主要是：
  - 运行对比页列裁剪与当前视图导出

## 已完成的结果页数值阈值细筛选基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的 `ML 排名`、`Rule 排名` 和 `Pose 结果` 页补了数值阈值筛选层。
- 当前已经支持：
  - 在结果页选择一个或多个数值列启用阈值筛选
  - 分别设置每个数值列的最小值和最大值
  - 与现有文本筛选、列选择、排序和导出联动
- 导出时保留当前阈值筛选后的可见视图或全筛选结果
- 这一步主要解决“想快速只看高分段、低损失段或特定几何特征范围，但当前只能靠人工导出后再筛”的问题。

## 已识别的运行对比页深化优化路线（2026-04-15）

1. [已完成基础增强] 运行对比页列裁剪与当前视图导出。
2. [已完成基础增强] 运行对比页按数值阈值做细筛选。
3. [已完成基础增强] 将当前对比可见视图同步进 HTML / PDF 导出。

这些点仍然属于“对比已经能看，但还可以更聚焦、更便于分享”的增强，不改 compare 的底层统计逻辑，只增强 compare 结果浏览和导出层。

## 已完成的运行对比页列裁剪与当前视图导出基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的“运行对比”页补了表级数据视图控制。
- 当前已经支持：
  - 对批次聚合表自定义显示列和排序方式
  - 对归因总表自定义显示列和排序方式
  - 对差异表自定义显示列和排序方式
  - 对完整对比表自定义显示列和排序方式
- 分别导出这些表的当前可见视图 CSV 和全表 CSV
- 这一步主要解决“运行对比页已经有很多表，但当前只能原样看全表、导出全表，难以快速聚焦重点列”的问题。
- 这一轮做完后，更适合继续推进的点主要是：
  - 运行对比页按数值阈值做细筛选
  - 单个 exe 完全独立运行

## 已完成的运行对比页数值阈值细筛选基础增强版（2026-04-15）

- 在 [local_ml_app.py](local_ml_app.py) 的“运行对比”页补了表级数值阈值筛选层。
- 当前已经支持：
  - 对批次聚合表按数值列设置最小值 / 最大值
  - 对归因总表按数值列设置最小值 / 最大值
  - 对差异表按数值列设置最小值 / 最大值
  - 对完整对比表按数值列设置最小值 / 最大值
  - 与现有列裁剪、排序和 CSV 导出联动
- 这一步主要解决“运行对比页虽然已经能看很多表，但想快速只保留改善更大、失败更少或批次更稳定的记录时，仍然要手工导出再筛”的问题。
- 这一轮做完后，更适合继续推进的点主要是：
  - 单个 exe 完全独立运行
  - 更接近真实物理含义的 mouth / path blocking proxy

## 已完成的运行对比页导出视图同步基础增强版（2026-04-16）

- 在 [local_ml_app.py](local_ml_app.py) 的“运行对比”页补了导出层的当前视图同步。
- 当前已经支持：
  - 生成运行对比 HTML 时，使用当前页面四张表的显示列、排序结果和数值阈值筛选结果
  - 生成运行对比 PDF 时，使用当前页面四张表的显示列、排序结果和数值阈值筛选结果
  - 在导出目录中同时写出每张表的 `*_view.csv` 与全筛选结果 CSV
  - 在导出的 HTML / PDF 中增加“当前页面视图同步说明”，明确当前展示行数、列数和筛选后总行数
- 这一步主要解决“页面里已经裁好了重点视图，但导出 HTML / PDF 仍然退回原始全表，导致展示内容和页面所见不一致”的问题。
- 这一轮做完后，更适合继续推进的点主要是：
  - 单个 exe 完全独立运行
  - 更接近真实物理含义的 mouth / path blocking proxy
  - 更完整的 ML 外部验证与 benchmark

## 完整度核查（2026-04-14）

这轮重新核查后，需要明确区分“当前阶段可用”与“完全收尾”：

- 当前可以视为“当前阶段完整可用”的部分：
  - 推荐流程的 `CLI + Python` 双入口
  - 本地软件的输入导入、运行前检查、后台运行、基础队列、停止当前任务
  - 历史加载、参数模板、失败诊断、QC/Warning 展示
  - 多运行对比的页面展示、趋势查看、差异解释、归因拆解和 HTML/CSV/PDF 导出
  - 桌面 launcher、便携目录版、zip 发布版、GitHub 自动发布链路
- 当前不应称为“完全版”，而应继续按“基础版 / 基础增强版”理解的部分：
  - 多运行对比
  - 展示摘要 HTML 版式
  - 单个 exe 完全独立运行
  - 面向 viewer 的结构化 bundle 深化
- 后续文档中的“已完成”如果带有“基础版 / 基础增强版”字样，应理解为：
  - 功能已经能用、能演示、能继续复用
  - 但仍保留进一步补强空间
  - 不等于这个方向已经彻底结束

## 已识别的功能说明与创新增强路线（2026-04-16）

这部分不是要推翻现有代码，而是基于当前已经能跑的本地软件、批量 pipeline、benchmark 和导出能力，继续补“更容易理解、更容易批量用、更有展示创新点”的功能。

### A. 功能说明仍可完善的地方

1. [已完成基础增强] 批量输入说明还可以更直观。
   - 当前已经在 [local_ml_app.py](local_ml_app.py) 的运行前检查中补了“输入数据检查报告”基础版。
   - 已支持列出必需列、可选列、行数、nanobody 数、conformer 数、pose 数、重复 ID 行、label 覆盖情况。
   - 已支持检查 `pdb_path` / `pocket_file` / `catalytic_file` / `ligand_file` 的路径覆盖、缺失路径和上传 CSV 相对路径未确认风险。
   - [已完成基础增强] 新增 [input_path_repair.py](input_path_repair.py)，可递归扫描 CSV 所在目录树或指定 `--search_root`，为缺失的 `pdb_path` / `pocket_file` / `catalytic_file` / `ligand_file` 输出修复建议、summary、Markdown 报告和可选修复版 CSV。
   - [已完成基础增强] 本地软件运行前检查已接入缺失路径自动定位，可下载路径修复建议 CSV，也可保存并使用自动修复版 `input_csv`。
   - 已支持预览本次会执行哪些 pipeline 阶段，以及 label-aware 步骤会执行还是跳过。
   - 已支持下载 `input_preflight_report.json`，方便保留本批次输入检查记录。
   - 后续仍可继续加强为更强命名识别、按子目录聚合 nanobody_id、行级 pocket/ligand 匹配和人工确认编辑。

2. [已完成基础增强] 结果分数说明还可以更产品化。
   - 当前有 `final_score` / `final_rule_score` / `explanation`，但用户仍可能不知道高分代表什么。
   - [已完成基础增强] 新增 [build_score_explanation_cards.py](build_score_explanation_cards.py)，基于 `consensus_ranking.csv` 输出 `score_explanation_cards.csv/md/html/json`。
   - 分数解释卡片会说明高分原因、主要正向因素、主要风险因素、是否 pocket 过宽、是否缺少 label、是否存在 QC warning，以及推荐动作。
   - 推荐 pipeline、smoke test 和本地软件“排名结果”页已接入该产物；本地页面可预览 HTML、筛选 CSV 和下载结果。
   - 后续可继续把 ML 排名和 Rule 排名的区别做成更显眼的页面内说明。

3. [已完成基础增强] QC / Warning 还可以变成明确的运行判定。
   - 当前已经有 `feature_qc.json`、failed/warning 行和 pocket shape QC。
   - [已完成基础增强] 新增 [build_quality_gate.py](build_quality_gate.py)，读取 `pose_features.csv` 和可选 `feature_qc.json`，输出统一 `PASS / WARN / FAIL`。
   - 当前规则：没有可用行或存在 failed 行为 `FAIL`；warning 行、全空列、pocket overwide 风险或 label 不足为 `WARN`；基础检查都通过为 `PASS`。
   - 推荐 pipeline、smoke test、本地软件 QC/Warning 页和结果归档已接入 `quality_gate_summary.json`、`quality_gate_checks.csv` 和 `quality_gate_report.md`。
   - 后续可继续把 PASS/WARN/FAIL 接到运行历史和多运行对比表中，便于批次级筛选。

4. [已完成基础增强] 批量运行后的汇总说明还可以更清楚。
   - 当前已有运行历史、多运行对比和趋势聚合。
   - [已完成基础增强] 新增 [build_batch_decision_summary.py](build_batch_decision_summary.py)，可读取推荐流水线 summary、Quality Gate、共识排名、分数解释卡片、实验建议和实验计划，输出 `batch_decision_summary.json/md` 与 `batch_decision_summary_cards.csv`。
   - 当前已汇总本批次最高综合排名候选、当前证据最稳定候选、最需要复核候选、下一轮实验优先候选、Quality Gate WARN/FAIL、warning/error Top-N、最高风险输入行、真实验证证据状态和建议先打开的文件。
   - 推荐 pipeline、smoke test、本地软件摘要页和结果归档均已接入该产物；这比单纯给一堆 CSV 更适合比赛展示和非开发用户理解。

5. README / run.md / MODEL_QUICKSTART.md 之间还可以进一步分工。
   - README 适合放完整说明。
   - run.md 适合只放“怎么打开软件”。
   - MODEL_QUICKSTART.md 适合只放“如何准备输入、如何运行、如何看输出”。
   - 后续建议避免三份文档重复长段内容，而是互相引用。

### B. 可以作为创新点继续做的功能

1. 批量数据导入向导。
   - [已完成基础增强] 当前已经支持 zip 导入、本地目录扫描、CSV 批量处理，并在缺少现成 `input_pose_table.csv` / `pose_features.csv` 时自动生成 `auto_input_pose_table.csv`。
   - 当前自动生成逻辑会扫描 PDB，保守推断 `nanobody_id` / `conformer_id` / `pose_id`，并尽量复用识别到的 `pocket` / `catalytic` / `ligand` 默认文件。
   - 自动生成表会在本地软件中展示预览，并支持下载；运行前仍建议点击“检查当前输入”确认 ID 和路径是否符合预期。
   - 这属于最实用的创新，因为它直接降低使用门槛。

2. 批量任务 manifest。
   - 当前 CD38 benchmark 已经有 `cd38_benchmark_manifest.csv` 和批量 runner。
   - 主 ML pipeline 也可以补类似 manifest：一张表描述多批输入、参数模板、输出目录和是否启用 label-aware steps。
   - 这样可以一次性排队跑多批数据，而不是只在 UI 队列里手工点。

3. Rule + ML 共识排名。
   - [已完成基础增强] 当前已经有 Rule 排名、ML 排名、Rule vs ML 对照，以及新增的 `build_consensus_ranking.py`。
   - 推荐 pipeline 会自动输出 `consensus_outputs/consensus_ranking.csv`、`consensus_summary.json` 和 `consensus_report.md`。
   - 共识表综合 ML 分数、Rule 分数、两者一致性、来源覆盖度、QC 风险和 pocket overwide 风险，并给出 `confidence_level`、`decision_tier` 与中文解释。
   - [已完成基础增强] 新增分数解释卡片，把共识分、可信度、正向因素、风险因素、label 状态和 recommended action 翻译成适合非开发用户阅读的 CSV/HTML/Markdown。
   - 本地软件“排名结果”页已新增共识排名预览、筛选和下载；HTML/PDF 展示摘要也会包含共识排名预览。
   - 创新点是“不是盲信黑盒 ML，而是保留规则可解释性和几何 QC 的共识决策”。

4. 每个 nanobody 的“候选报告卡”。
   - [已完成基础增强] 新增 `build_candidate_report_cards.py`，可基于 `consensus_ranking.csv` 为每个候选生成单页 HTML 报告卡。
   - 当前报告卡会汇总最终排名、ML 分数、Rule 分数、可信度、风险标记、主要解释、top pose 行、feature/QC warning、pocket 覆盖和几何风险。
   - 推荐 pipeline 会自动输出 `candidate_report_cards/index.html`、`candidate_report_cards/cards/*.html`、`candidate_report_cards.csv`、`candidate_report_cards_summary.json` 和 `candidate_report_cards.zip`。
   - 本地软件“排名结果”页已新增候选报告卡索引打开和 zip 下载入口。
   - 这适合导出 HTML；如需 PDF，可用浏览器打开单个报告卡后打印为 PDF。

5. 候选横向对比解释。
   - [已完成基础增强] 新增 `build_candidate_comparisons.py`，可基于 `consensus_ranking.csv` 生成候选 trade-off 表和 pairwise 对比解释。
   - [已完成基础增强] 支持 `--selected_nanobody_ids` 做指定候选自定义对比；本地软件可手工选择 2 到 5 个候选并保存结果。
   - 当前输出 `candidate_tradeoff_table.csv`、`candidate_pairwise_comparisons.csv`、`candidate_comparison_summary.json` 和 `candidate_comparison_report.md`。
   - 推荐 pipeline 和 smoke test 会自动输出 `candidate_comparisons/`；本地软件“排名结果”页已支持预览、筛选、下载和查看 Markdown 报告。
   - 这一步回答“为什么 A 排在 B 前面”，同时列出较低排名候选的反向优势和 close decision，属于同类工具通常缺少的决策解释层。

6. 不确定性 / 低可信样本提示。
   - [已完成基础增强] `consensus_ranking.csv` 已输出 `confidence_score` 与 `confidence_level = high / medium / low`。
   - [已完成基础增强] `consensus_ranking.csv` 已输出 `review_reason_flags` 与 `low_confidence_reasons`，把 Rule/ML 排名差、分数差、pocket overwide、conformer instability、close-score competition、失败行和 warning 行拆成独立原因。
   - 当前可信度基于 Rule/ML 排名一致性、分数一致性、来源覆盖度和 QC 风险；候选报告卡和下一轮实验建议已经接入这些细分原因。
   - 后续还可以把训练 label 覆盖、候选序列/结构多样性进一步显式加入低可信原因拆解。

7. 主动学习式下一轮实验建议。
   - [已完成基础增强] 新增 `suggest_next_experiments.py`，可从 `consensus_ranking.csv` 生成下一轮实验建议表。
   - 当前会根据高分但低可信、Rule/ML 分歧大、QC / pocket overwide 风险高和来源缺失等信号，输出 `experiment_priority_score`、`suggestion_tier`、`suggestion_reason` 和 `recommended_next_action`。
   - [已完成基础增强] 新增 diversity-aware ordering，输出 `diversity_adjusted_priority_score`、`diversity_group`、`diversity_adjustment` 和 `diversity_note`，避免下一轮建议过度集中在同一类候选。
   - [已完成基础增强] 新增 `experiment_plan.csv/md`，支持总预算、validate/review 分层 budget、standby 数量和 diversity group quota。
   - [已完成基础增强] 新增 `experiment_plan_state_ledger.csv`，并在本地软件里提供计划单编辑器，可保存为下一轮 `experiment_plan_override_csv`。
   - [已完成基础增强] 新增跨批次全局 ledger，扫描 `local_app_runs` 汇总历史实验状态，输出 `experiment_state_ledger_global.csv`。
   - [已完成基础增强] 新增真实验证回灌报告，只有明确的 `validation_label` 或 `experiment_result=positive/negative` 才生成训练标签，避免把 `completed` 或 `blocked` 误当成生物学标签。
   - [已完成基础增强] 新增真实验证证据审计，推荐 pipeline 会输出 `validation_evidence_audit/validation_evidence_report.md`、top-k 表和行动清单，检查当前高排名候选是否已经有足够实验支撑。
   - 推荐 pipeline 会自动输出 `experiment_suggestions/next_experiment_suggestions.csv`、`next_experiment_suggestions_summary.json` 和 `next_experiment_suggestions_report.md`。
   - 本地软件“排名结果”页已新增下一轮实验建议预览、筛选和下载。
   - 这可以作为科研产品创新点：软件不只是排序，还能建议下一步实验投入，并提示“当前排序还缺哪些真实验证证据”。

8. pocket 方法共识分析。
   - [已完成基础增强] 新增 `compare_pocket_method_consensus.py`，可比较 manual、ligand-contact、P2Rank、fpocket 等多来源 pocket residue list。
   - 当前输出 `consensus_pocket_residues.txt`、`union_pocket_residues.txt`、`residue_method_membership.csv`、`method_specific_residues.csv`、`method_overlap_matrix.csv`、`pocket_method_consensus_summary.json` 和 `pocket_method_consensus_report.md`。
   - 有 truth 时会输出 truth coverage、precision、missing truth risk 和 overwide risk；无 truth 时会输出基于方法一致性和 union/consensus 扩张程度的 proxy 风险。
   - 已在 CD38 的 `3ROP ligand-contact vs P2Rank` 和 `4OGW ligand-contact vs P2Rank` 上跑通。
   - `3ROP` 当前共识 pocket 覆盖 truth 约 `0.8571`，precision 约 `0.5455`，仍缺 `A:155` 这个 exact truth 位点。
   - `4OGW` 当前共识 pocket 覆盖 truth 为 `1.0000`，precision 约 `0.5385`，说明核心位点覆盖稳定，但仍有约一半共识 residue 属于 truth 之外的邻近口袋区域。
   - 这已经把“口袋预测不确定性”从隐藏风险变成可解释结果；后续增强重点应转向更多方法、更多结构和参数敏感性。

9. 参数敏感性分析。
   - [已完成基础增强] 新增 `analyze_ranking_parameter_sensitivity.py`，可基于已有 `consensus_ranking.csv` 做 post-processing，不重跑结构特征、不重新训练模型。
   - 当前会扫描 Rule/ML 权重、rank-agreement 权重和 QC risk penalty 权重，输出 `scenario_rankings.csv`、`scenario_summary.csv`、`candidate_rank_sensitivity.csv`、`sensitive_candidates.csv`、`parameter_sensitivity_summary.json` 和 `parameter_sensitivity_report.md`。
   - 推荐 pipeline 和 smoke test 会自动生成 `parameter_sensitivity/`；本地软件“排名结果”页已支持预览和下载。
   - [已完成基础增强] 新增 `analyze_cd38_pocket_parameter_sensitivity.py`，可基于当前 CD38 本地 benchmark 文件扫描 contact cutoff、P2Rank rank、fpocket pocket choice、method consensus 阈值和 overwide penalty。
   - 当前结构侧输出 `contact_cutoff_sensitivity.csv`、`p2rank_rank_sensitivity.csv`、`fpocket_pocket_sensitivity.csv`、`method_consensus_threshold_sensitivity.csv`、`overwide_penalty_sensitivity.csv`、`cd38_pocket_parameter_sensitivity_summary.json` 和 `cd38_pocket_parameter_sensitivity_report.md`。
   - [已完成基础增强] 新增 `prepare_cd38_fpocket_panel.py`，用于把真实 `fpocket` 输出目录批量转换成 CD38 benchmark manifest，并可直接串到 manifest runner。
   - [已完成基础增强] `prepare_cd38_fpocket_panel.py` 现在会同步生成 readiness report，检查扫描目录、可运行 rows、跳过原因和推荐命令；PDB ID 自动推断已收紧为首位数字的真实 PDB ID 格式，避免把普通文件夹名误判成结构。
   - [已完成基础增强] 新增 `prepare_cd38_p2rank_panel.py`，可批量发现真实 `P2Rank` 的 `*_predictions.csv`，生成 manifest、summary JSON 和 readiness report，并支持 `--rank_by_pdb`。
   - [已完成基础增强] 新增 `build_cd38_benchmark_expansion_plan.py` 和 `benchmarks/cd38/cd38_structure_targets.csv`，把“更多 CD38 结构 / 真实 fpocket baseline”拆成结构-方法级状态矩阵，并输出 `complete`、`needs_fpocket_output`、`needs_ligand_metadata`、`needs_p2rank_output` 等缺口状态。
   - [已完成基础增强] 新增 `inspect_cd38_ligand_candidates.py`，用 HETATM + CD38 truth residue 距离判断结构是否适合 ligand-contact baseline；当前 `3F6Y` 被识别为无活性口袋 ligand candidate，因此扩展计划只保留 P2Rank/fpocket。
   - [已完成基础增强] 新增 `refresh_cd38_benchmark_readiness.py`，一条命令聚合 ligand scan、扩展计划、P2Rank readiness 和 fpocket readiness，默认不运行 benchmark，适合每次接入外部输出前先刷新总览。
   - [已完成基础增强] 新增 `prepare_cd38_external_tool_inputs.py`，把真实外部 P2Rank/fpocket benchmark 的下一步拆成输入 PDB、PowerShell/Bash 命令模板、输出目录约定、expected return checklist 和 readiness 刷新脚本，减少手工整理目录时出错。
   - [已完成基础增强] 新增 `check_cd38_external_tool_environment.py`，把“工具没装”“输入缺失”“P2Rank CSV 缺失”“fpocket pocket 文件缺失”拆成可审计 preflight 报告；preflight 现在优先解析当前 package 内 portable 路径，移动输入包后不会被旧绝对路径误导；当前本机 `prank` 和 `fpocket` 都未在 PATH 中发现。
   - [已完成基础增强] 新增 `finalize_cd38_external_benchmark.py`，把外部输出复制回来后的收尾动作压成一条命令：可选返回包导入、preflight、readiness、可选导入 benchmark、可选刷新参数敏感性；如果没有可运行外部 rows，会明确跳过旧 panel 汇总，并在有缺口时链接 repair plan。
   - [已完成基础增强] 新增 `package_cd38_external_tool_inputs.py`，把外部工具输入包打成可转移 zip，默认排除中间报告和旧输出，降低跨机器运行时复制错文件的风险。
   - [已完成基础增强] 新增 `import_cd38_external_tool_outputs.py`，把返回目录/zip 中的真实输出安全导入本地 package，降低手工复制输出目录时错位或覆盖的风险；导入脚本现在能处理多包一层目录的返回 zip，并输出 scan manifest、coverage manifest、repair plan 和 `source_diagnosis` 解释为什么候选数为 0、缺哪些结构/方法以及下一步应补什么。
   - [已完成基础增强] 新增 `build_cd38_proxy_calibration_report.py`，可基于当前 CD38 panel 重算运行时 `pocket_shape_overwide_proxy`，对照 truth-based `overwide_pocket_score`、coverage 和 precision 输出阈值候选、方法汇总、penalty simulation 和默认策略建议；当前结论是 proxy 能识别 `4OGW + P2Rank` 偏宽问题，但证据等级仍为 `low`，默认 `pocket_overwide_penalty_weight` 继续保持 `0.0`。
   - [已完成基础增强] 新增 `build_cd38_external_benchmark_action_plan.py`，把外部 benchmark 的缺口从多份 report 合并成一张可执行清单；当前 action plan 显示 6 个外部输出动作，其中 4 个是 benchmark completion blocker，优先补 `3ROP/4OGW fpocket` 和 `3F6Y P2Rank/fpocket`。
   - [已完成基础增强] 新增 `build_geometry_proxy_audit.py`，可从 `pose_features.csv` 审计 mouth/path/pocket/contact 等 proxy 是否互相矛盾，输出 pose 级 flagged rows、candidate 级汇总、特征覆盖表和 Markdown 报告；推荐 pipeline、smoke test 和本地软件 QC 面板已自动接入，且不改变任何分数。
   - 当前基础版可以回答“候选排名是否对权重/QC 惩罚敏感”“CD38 pocket 结论是否对结构侧参数敏感”“当前 proxy 证据是否足以改变默认惩罚权重”“外部 benchmark 下一步具体缺哪些输出”和“本批几何 proxy 是否自洽”；后续重点应从真实 `fpocket` 输出和更多结构中补样本。

10. 数据与结果 provenance。
   - 为每次运行记录输入文件 hash、脚本版本、参数模板、依赖环境、随机种子、输出文件清单。
   - [已完成基础增强] 推荐 pipeline 现在会生成 `provenance/run_input_file_manifest.csv`，按行记录 `input_csv` / `feature_csv` 中引用的 `pdb_path`、`pocket_file`、`catalytic_file`、`ligand_file`，包括是否存在、大小、SHA256、缺失状态和默认文件来源。
   - [已完成基础增强] 结果归档现在会生成 `result_archive_lineage.csv`，用 provenance hash 追踪共享输入文件 manifest、共享 feature CSV 和共享参数 hash 的历史运行，并标出上一个同源运行。
   - [已完成基础增强] 推荐 pipeline 现在会生成 `provenance/run_provenance_integrity.json`，并提供 `verify_run_provenance.py` 对 provenance 卡片、artifact manifest 和 input file manifest 重新计算 SHA256，识别文件误改或复制不完整。
   - [已完成基础增强] 结果归档现在会生成 `result_archive_lineage_graph.json/html/md`，把共享输入 manifest、共享 feature CSV 和共享参数 hash 的复跑关系做成图形化时间线，本地软件归档页可预览和下载。
   - 当前已有 provenance 运行卡片、artifact manifest、输入文件引用 manifest、跨批次 lineage、lineage 图形化、SHA256 完整性封存、metadata 和版本号；后续如需更强审计，再转向带私钥数字签名。
   - 这对科研场景很重要，也适合展示工程规范。

11. Demo 数据集与一键演示模式。
   - [已完成基础增强] 新增 `demo_data_utils.py`，可生成可复现 synthetic `pose_features.csv`。
   - [已完成基础增强] 新增 synthetic `experiment_plan_override.csv`，让 demo 同时覆盖验证证据审计、批次结论和报告导出。
   - [已完成基础增强] 新增 `run_demo_pipeline.py` 和 `run_demo_pipeline.bat`，一条命令或双击即可生成 demo 数据并跑完整推荐 pipeline。
   - [已完成基础增强] 本地软件侧边栏新增“生成并载入 demo 输入”，会自动切到 `feature_csv` 模式并填好 demo 特征表和 experiment override。
   - [已完成基础增强] 本地软件侧边栏新增“生成并立即运行 demo”，可自动生成 demo 输入并启动后台运行。
   - [已完成基础增强] 命令行 demo 输出 `demo_outputs/DEMO_README.md`；本地软件 demo 运行也会在当前输出目录写入 `DEMO_README.md`，按阅读顺序链接批次结论、候选报告卡、验证证据审计和关键 CSV。
   - [已完成基础增强] 新增 `DEMO_INTERPRETATION.md`，解释 demo 中的 Quality Gate、候选排序、top-k 验证覆盖和 synthetic 验证边界。
   - [已完成基础增强] 新增 `DEMO_OVERVIEW.html`，用浏览器展示 demo 摘要、候选亮点、top consensus ranking 和关键结果链接。
   - [已完成基础增强] 本地软件“摘要”页新增“Demo 快速导览”，可一键打开 `DEMO_OVERVIEW.html` 并下载 demo README / 解读文件。
   - [已完成基础增强] 新增 `REAL_DATA_STARTER/`，包含 `input_pose_table_template.csv`、`pose_features_template.csv`、`experiment_plan_override_template.csv`、pocket/catalytic 模板和真实数据检查清单。
   - 后续可补可运行的真实小型 PDB 示例包；demo 标签仍必须标注为 synthetic，不可冒充真实湿实验结果。

### C. 建议下一阶段优先级

1. 先做“输入数据检查报告”和“上传后预览将如何批量处理”。
   - 状态：已完成基础增强。
   - 后续剩余：缺失文件自动定位和修复建议已有基础版；可以继续补更强命名识别、批量人工确认编辑和按目录规则自动分组。

2. 再做“批量数据导入向导 -> 自动生成 input_pose_table.csv”。
   - 状态：已完成基础增强。
   - 后续剩余：可以继续补更强的命名规则识别、按子目录聚合 nanobody_id、行级 pocket/ligand 匹配和人工确认编辑。

3. 再做“候选报告卡”“候选对比解释”和“共识排名”。
   - 状态：共识排名、候选报告卡、候选横向对比解释、自定义候选对比和候选分组小结均已完成基础增强。
   - 后续剩余：可以继续把报告卡升级为批量 PDF、增加报告卡批注和真实验证结果对照。

4. 再做“可信度分级”和“主动学习建议”。
   - 状态：可信度分级、主动学习建议、diversity-aware 队列、实验计划单、状态 ledger、本地编辑器、跨批次全局 ledger、真实验证回灌报告、验证标签再训练入口、ledger 筛选图表、再训练前后对照报告、结果自动归档和长期趋势汇总均已完成基础增强。
   - 后续剩余：可以继续做更完整的真实 fpocket benchmark。

5. 最后做“参数敏感性分析”和“pocket 方法共识分析”。
   - 原因：更偏验证和科研增强，需要更多真实数据或外部 tool 输出支撑。
   - 改动范围：适合用 manifest runner 和 benchmark 工具继续扩展。
   - 状态：pocket 方法共识分析、ranking 参数敏感性分析、CD38 结构侧参数敏感性分析、fpocket/P2Rank 批量接入入口、外部工具输入包、环境/输出预检、finalize 一键收尾、外部 benchmark action plan、readiness 一键刷新、CD38 扩展计划和 ligand-contact 适用性检查均已完成基础增强；后续继续补更多结构和真实 `fpocket` / `P2Rank` 输出。

### D. 当前不建议优先做的方向

1. 不建议现在重写整个前端。
   - 当前 Streamlit 本地软件已经能支撑上传、运行、历史、导出和对比，继续局部增强更划算。

2. 不建议现在默认启用 pocket overwide 惩罚。
   - 当前样本量还不足，默认保持 `0.0` 更稳。

3. 不建议现在把模型包装成“已严格验证的科研结论生成器”。
   - 当前更准确的定位仍是“可解释排序 + 几何 proxy + benchmark 辅助决策工具”。

4. 不建议优先做 3D viewer。
   - 你当前真正需要的是更方便上传、批量处理、理解结果和导出报告；3D viewer 可以等结构化 bundle 更稳定后再接。

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
  - `top_k`（标准 CD38 目录下表示每个 `vhh/CD38_i/` 中 MMPBSA 最低的 K 个 pose）
  - `top_k_selection_col`（默认 `auto`，优先用 `MMPBSA_energy` / `mmgbsa`，没有能量列才回退到分数）
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

- [已完成基础增强] 直接在界面里预览：
  - `nanobody_ranking.csv`
  - `pose_predictions.csv`
  - `training_summary.json`
  - `recommended_pipeline_report.md`
- [已完成基础增强] 增加“只看 Top-N 排名”的筛选
- [已完成基础增强] 增加基础统计卡片：
  - 样本数
  - nanobody 数
  - 是否启用 calibration
  - rule/ML 主要指标
- [已完成基础增强] 增加失败行或 warning 的可读展示

### 第三优先级：把“本地软件体验”补齐

- [已完成基础增强] 最近一次运行历史
- [已完成基础增强] 多运行对比页
- [已完成基础增强] 多运行对比 HTML 导出
- [已完成基础增强] 自动化 PDF 导出
- [已完成基础增强] PDF 版式进一步美化
- [已完成基础增强] 多运行趋势与 run-to-run 差异解释
- [已完成基础增强] 更细的差异归因
- [已完成基础增强] 跨批次趋势聚合
- [已完成基础增强] 一键打开输出目录
- [已完成基础增强] 一键导出当前结果汇总
- [已完成基础增强] 输入参数保存/载入
- [已完成基础增强] 常见错误提示模板化
- [已完成基础增强] zip 数据包一键导入与自动识别
- [已完成基础增强] 本地数据目录一键扫描与自动识别
- [已完成基础增强] 输入模板下载与运行前检查
- [已完成基础增强] 基础任务队列与运行中止
- [已完成基础增强] 展示摘要 HTML 导出
- [已完成基础增强] 失败行 / warning 的可读展示

## 面向“本地交互式软件”的建议后续顺序

1. [已完成基础增强] 把 [run_recommended_pipeline.py](run_recommended_pipeline.py) 抽成“CLI + 可调用函数”双入口，减少 UI 壳和命令行逻辑重复。
2. [已完成基础版] 新增本地交互入口文件，先做最小运行面板，不碰算法逻辑。
3. [已完成基础版] 把 `recommended_pipeline_summary.json`、`recommended_pipeline_report.md`、`nanobody_ranking.csv` 接成结果面板。
4. [已完成基础增强] 再补上传体验、历史记录、导出按钮、训练摘要和结果筛选。
5. [已完成基础增强] 把“导出当前结果汇总”补成单文件打包或单按钮汇总导出。
6. [已完成基础版] 打包成桌面可执行程序的 launcher 版本。
7. [已完成基础版] 做成“整个目录可拷走”的便携版。
8. [已完成基础版] 补出可直接分发的 zip 发布版。
9. [已完成基础增强] 补齐统一版本元数据与发布追踪信息。
10. [已完成基础增强] 补齐品牌图标基础版。
11. [已完成基础增强] 补 GitHub 自动发布工作流。
12. [已完成基础增强] 补 zip 数据包导入、输入模板下载和运行前检查。
13. [已完成基础增强] 补本地数据目录扫描导入。
14. [已完成基础增强] 补基础任务队列与运行中止。
15. [已完成基础增强] 补展示摘要 HTML 导出。
16. [已完成基础增强] 补失败行 / warning 的可读展示。
17. [已完成基础增强] 补多运行对比页。
18. [已完成基础增强] 补多运行对比 HTML 导出。
19. [已完成基础增强] 补自动化 PDF 导出。
20. [已完成基础增强] 补 PDF 版式进一步美化。
21. [已完成基础增强] 补多运行趋势与 run-to-run 差异解释。
22. [已完成基础增强] 补更细的差异归因。
23. [已完成基础增强] 补跨批次趋势聚合。
24. [已完成基础版] 已补单文件自解压的 standalone onefile 版；后续如需继续推进，重点是跨机器兼容性验证与收紧。

## 1. 几何特征仍以 proxy 为主（已部分收紧）

文件: [geometry_features.py](geometry_features.py)

当前已实现的几何特征已经可以区分很多“堵口袋”和“普通表面结合”的情形，而且 `mouth_occlusion_score` 与 `ligand_path_block_score` 已经做过一轮收紧；但其中一部分仍是静态代理量，不是严格物理模拟：

- `mouth_occlusion_score` 已经更强调口部轴向和孔径覆盖的一致性，但口部候选点本身仍来自静态几何推断。
- `delta_pocket_occupancy_proxy` 和 `pocket_block_volume_proxy` 是占据/阻断的代理，不是显式体积模拟。
- `ligand_path_block_score` 已升级为融合连续阻断比例/瓶颈分数并显式惩罚开放逃逸路径的静态近似，但仍不是完整动力学路径评估。
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

## 5. 训练与评估仍缺真实外部验证闭环（已补 grouped CV benchmark 基础版）

文件: [train_pose_model.py](train_pose_model.py)、[rank_nanobodies.py](rank_nanobodies.py)、[benchmark_pose_pipeline.py](benchmark_pose_pipeline.py)

当前训练和排序流程已经不再只是“单次 train/val 切分”，因为已经补了独立的 grouped CV benchmark：

- 已有基于 `nanobody_id` 的系统交叉验证。
- 已有 pose / nanobody 两层的 reliability curve、ECE 和 Brier。
- 已可并排比较 ML ranking 与 rule ranking 的 AUC / Spearman / score delta。

但它仍然不是最终完整评估体系：

- 还没有真实外部独立 test set。
- 还没有跨实验批次、跨时间的长期 benchmark。
- 模型选择仍主要基于 `val_loss` + 交叉验证指标，还没有和真实结构验证结果做闭环对齐。
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
- 当前已经补了运行时依赖预检，可在桌面启动器、本地软件和推荐流程入口提前识别缺失的 `streamlit` / `torch` / `biopython`。
- `requirements.txt` 中的 `torch` 已按 Python 版本分流，降低 Python 3.14 环境下“锁定版本不可安装”的风险。
- 新增 `.github/workflows/smoke-test.yml`，在 push/PR 自动执行端到端 smoke test。

仍建议继续做：

- 如果后续需要 GPU/CPU 双环境一致性，补一份针对不同平台的 lock 文件或约束文件（例如 constraints）。
- 在 CI 中增加多种种子和更小批次数据规模的快速回归矩阵，以更早捕获不稳定性。

## 9. 建议的后续完善顺序

1. [已完成] 用真实/标签可用数据时执行规则权重校准入口（脚本已提供）。
2. [已完成] 规则版与 ML 版对照评估报表。
3. [进行中] 把几何 proxy 再往物理意义上收紧，尤其口袋口部和路径阻断；当前已补 mouth/path 收紧、pocket shape QC、CD38 proxy 校准和不改分数的 geometry proxy audit。
4. [已完成] 增加固定回归数据能力（synthetic 可复现数据生成 + smoke test 固定种子 + 一键 demo pipeline）。
5. [已完成] 在 Python 3.13 环境补一轮端到端烟测流程。
6. [已完成基础版] 导出 residue annotation / interface residue 的结构化载荷，给后续 viewer 或展示壳复用。
7. [已完成基础版] 增加 AI/离线解释层，只解释已有结果，不改变 Rule/ML 分数和排序。
8. [已完成基础增强] 增加 provenance 运行卡片，记录输入、输出、代码、依赖、Git 状态、参数 hash、输入文件引用 manifest 和 SHA256 完整性封存。
9. [已完成基础增强] 候选报告卡内嵌 candidate pairwise comparison context，减少报告切换成本。
10. [已完成基础增强] 下一轮实验建议加入 diversity-aware ordering，不改变原始 priority score，只调整建议队列展示顺序。
11. [已完成基础增强] 下一轮实验建议输出 `experiment_plan.csv/md`，支持预算、分层 quota、standby 和 defer。
12. [已完成基础增强] 新增一键 demo 数据和演示流程，覆盖 synthetic 特征、synthetic 验证 override、推荐 pipeline、批次结论和候选报告卡。
13. [已完成基础增强] 新增真实输入链路 mini PDB 示例包，覆盖 PDB 解析、显式链拆分、`A:37-40` residue range、ligand template、feature table 和完整 pipeline 轻量运行。

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
- `A:37-40`，等价于 `A:37,A:38,A:39,A:40`
- `B:37-40` / `C:37-40` 等其他链名范围
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

1. [已完成基础增强] 把 [run_recommended_pipeline.py](run_recommended_pipeline.py) 收敛成可直接被 UI 调用的函数入口，减少 subprocess 包装层。
2. [已完成基础增强] 在 [local_ml_app.py](local_ml_app.py) 中补运行历史、错误定位、参数模板保存/载入。
3. [已完成基础增强] 在 [local_ml_app.py](local_ml_app.py) 中补训练摘要、训练曲线、结果筛选、产物清单和本地目录打开能力。
4. [已完成基础增强] 把“关键产物下载”进一步收敛成单次运行的一键打包导出。
5. [已完成基础版] 把当前本地页面进一步打包成桌面可执行程序 launcher。
6. [已完成基础版] 把桌面程序从“依赖当前仓库目录”推进到“整个目录可拷走”的便携版。
7. [已完成基础版] 在便携目录版基础上补出可直接分发的 zip 发布包。
8. [已完成基础增强] 在桌面版、便携版和 zip 发布包中补齐统一版本元数据。
9. [已完成基础增强] 补品牌图标和桌面版基础品牌化资源。
10. [已完成基础增强] 把桌面版和便携版接入 GitHub 自动发布工作流。
11. [已完成基础增强] 在本地交互壳中补 zip 数据包导入、输入模板下载和运行前检查。
12. [已完成基础增强] 在本地交互壳中补本地数据目录扫描导入。
13. [已完成基础增强] 在本地交互壳中补基础任务队列与运行中止。
14. [已完成基础增强] 在本地交互壳中补展示摘要 HTML 导出。
15. [已完成基础增强] 在本地交互壳中补 failed 行 / warning 行的可读展示。
16. [已完成基础增强] 在本地交互壳中补多运行对比页。
17. [已完成基础增强] 在本地交互壳中补多运行对比 HTML 导出。
18. [已完成基础增强] 在本地交互壳中补自动化 PDF 导出。
19. [已完成基础增强] 在本地交互壳中补 PDF 版式进一步美化。
20. [已完成基础增强] 在本地交互壳中补多运行趋势与 run-to-run 差异解释。
21. [已完成基础增强] 在本地交互壳中补更细的差异归因。
22. [已完成基础版] 已补单文件自解压的 standalone onefile 版；后续如需继续推进，重点是跨机器兼容性验证与收紧。
23. [已完成基础版] 在推荐 pipeline 和本地交互壳中补 AI/离线解释摘要；默认离线，可选 OpenAI provider，失败自动回退，不上传原始结构文件。
24. [已完成基础增强] 在推荐 pipeline 和本地交互壳中补 provenance 运行卡片；默认生成 `provenance/run_provenance_card.json/md`、`run_artifact_manifest.csv`、`run_input_file_manifest.csv` 和 `run_provenance_integrity.json`，并加入下载和汇总包；结果归档页已补 `result_archive_lineage.csv` 下载与预览，并新增 `result_archive_lineage_graph.json/html/md` 图形化时间线。
25. [已完成基础增强] 在候选报告卡内嵌相邻候选对比摘要；推荐 pipeline 和 smoke test 会先生成 `candidate_pairwise_comparisons.csv`，再构建报告卡。
26. [已完成基础增强] 在下一轮实验建议中补 diversity-aware 队列排序；本地软件默认按 `suggestion_rank` 展示，并显示 diversity 分组和惩罚原因。
27. [已完成基础增强] 在下一轮实验建议中补实验计划单导出；推荐 pipeline、本地软件下载和 smoke test 均接入 `experiment_plan.csv/md/json`。
28. [已完成基础增强] 实验计划单支持 `experiment_plan_override.csv`，可手工 include/exclude/standby/defer，并回灌 owner、cost、status 和 note。
29. [已完成基础增强] 实验计划单新增 `experiment_plan_state_ledger.csv` 和本地软件内置编辑器；可保存编辑结果并设置为下一轮 override。
30. [已完成基础增强] 新增 `build_experiment_state_ledger.py` 和本地软件历史页全局 ledger 面板；可汇总多个历史运行并设置为下一轮 override。
31. [已完成基础增强] 候选对比支持自定义选择；CLI 可用 `--selected_nanobody_ids`，本地软件可选择 2 到 5 个候选生成、下载和保存对比解释。
32. [已完成基础增强] 新增 `build_experiment_validation_report.py`，从全局 ledger 生成 `experiment_validation_labels.csv`、`experiment_validation_status_report.csv`、`experiment_validation_report.md` 和可选 `pose_features_with_experiment_labels.csv`。
33. [已完成基础增强] 本地软件支持可配置 `label_col`，并可把验证回灌后的 `pose_features_with_experiment_labels.csv` 一键设为下一轮 `feature_csv + experiment_label` 输入。
34. [已完成基础增强] 本地软件全局 ledger 面板新增状态/结果/override/关键词筛选、筛选后关键计数和状态分布图。
35. [已完成基础增强] 新增验证回灌再训练前后对照报告，输出指标对照、候选排名变化、summary JSON 和 Markdown，并接入本地软件历史对比页。
36. [已完成基础增强] 新增结果自动归档和长期趋势汇总，输出运行索引、产物 manifest、验证回灌长期趋势表和 Markdown 报告，并接入本地软件历史页。
37. [已完成基础增强] 候选对比解释新增自动分组小结，输出 `candidate_group_comparison_summary.csv`，并在本地软件排名结果页支持预览和下载。
38. [已完成基础增强] 新增验证证据审计，输出 `validation_evidence_summary.json`、`validation_evidence_report.md`、`validation_evidence_by_candidate.csv`、`validation_evidence_topk.csv` 和 `validation_evidence_action_items.csv`，并在本地软件排名结果页支持预览和下载。
39. [已完成基础增强] `batch_decision_summary.md/json` 已整合验证证据审计，首页摘要可直接显示 Validation Evidence 状态和 top-k 覆盖率。
40. [已完成基础增强] 新增一键 demo 数据集和演示入口，支持 `python run_demo_pipeline.py`、双击 `run_demo_pipeline.bat`，或在本地软件侧边栏生成/载入/立即运行 demo 输入；demo 输出包含 `DEMO_OVERVIEW.html`、`DEMO_README.md`、`DEMO_INTERPRETATION.md` 和 `REAL_DATA_STARTER/`，本地软件摘要页可一键打开 HTML 导览和真实数据 starter 文件夹。
41. [已完成基础版] 新增 `build_pocket_evidence.py`，把人工/rsite、文献 residue、catalytic-anchor shell、ligand-contact、P2Rank、fpocket 和 AI prior 合并成 `pocket_evidence.csv`、`pocket_residue_support.csv`、`candidate_curated_pocket.txt` 和 `POCKET_EVIDENCE_REPORT.md`。该脚本只提升 pocket 输入质量，不改变正式 Rule/ML 权重。

## 13. pocket 精修证据链后续还需要收尾的地方

当前已经有 pocket evidence builder，但还没有完全变成“本地软件里一键点完”的闭环。后续建议按下面顺序继续：

1. [已完成基础增强] 在本地软件中增加“构建 pocket 证据”按钮，允许选择 PDB、manual/rsite、literature/catalytic、P2Rank、fpocket、ligand 和 AI prior 文件。
2. [已完成基础增强] 在本地软件中预览 `pocket_residue_support.csv`，突出 high-confidence、single-method、anchor-shell-only、AI-prior-only 和 not-found-in-structure residue，并提供分组视图、风险等级和操作建议。
3. [已完成基础版] 增加“把 `candidate_curated_pocket.txt` 设置为下一轮默认 `pocket_file`”按钮，避免手工复制路径。
4. [已完成基础版] 增加批量模式：从标准 `result/` 父目录自动选一个代表 PDB，先生成全项目 pocket evidence，再批量写入输入表默认 pocket。
5. [已完成基础增强] 对 literature/catalytic 输入增加来源字段模板和审计输出：支持 `--literature_source_table` / `--catalytic_source_table`，输出 `evidence_source_audit.csv` 和 `evidence_source_template.csv`，可记录 paper、PMID、DOI、UniProt、M-CSA、PDB、来源句子、证据等级、人工确认状态和备注。
6. [已完成基础增强] 增加 multi-method precision guard：当 P2Rank/fpocket 给出很宽 pocket 时，缺少 manual/literature/catalytic core/ligand-contact 高置信支持的边缘 residue 会标记 `external_overwide_guard`，进入 `review_residues.txt`，不直接进入 `candidate_curated_pocket.txt`。
7. [已完成基础增强] 增加 AI 辅助文献抽取的离线审计格式：支持 `--ai_source_table` / `--ai_prior_source_table`，输出 `ai_prior_audit.csv` 和 `ai_prior_template.csv`；AI prior 必须保留来源句子、证据等级和人工确认状态，只能作为待复核线索，且不再参与 curated 判定的支持分/方法数，不能直接当 ground truth。

本轮完成情况：

- `build_pocket_evidence.py` 已接入本地软件“诊断”页，保持对主运行表单低侵入。
- 已支持单个代表 PDB 的 pocket evidence 构建、结果预览、风险分组预览和关键输出下载。
- `pocket_residue_support.csv` 预览已新增 UI 派生列 `ui_risk_level`、`ui_review_group` 和 `ui_action_hint`，但不会写回原始 CSV。
- 已支持把 `candidate_curated_pocket.txt` 一键回填为下一轮 `default_pocket_file`。
- 已新增 `build_project_pocket_evidence.py` 和本地软件批量入口：可从标准 `result/` 父目录自动发现 `result/` 或 `rsite/result/`，按 `MMPBSA_energy` 最低值选择代表 PDB，生成项目级 pocket evidence，并输出带 `candidate_curated_pocket.txt` 的 `input_pose_table_with_pocket_evidence.csv`。
- 已验证不改变 Rule / ML 权重，不把 A-only 调教参数写进正式模型。
- literature/catalytic 来源审计模板、外部工具 overwide precision guard 和 AI 辅助文献抽取离线审计格式均已完成基础版；下一步建议回到整体 pocket evidence 使用体验，补一键示例模板和更严格的 UI 提示。

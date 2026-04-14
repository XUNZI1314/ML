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
  - 真正自动化的 PDF 导出

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
  - 真正自动化的 PDF 导出

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
  - 多运行趋势视图
  - 对比结果的 HTML 汇总导出
  - 更细的 run-to-run 差异解释

## 完整度核查（2026-04-14）

这轮重新核查后，需要明确区分“当前阶段可用”与“完全收尾”：

- 当前可以视为“当前阶段完整可用”的部分：
  - 推荐流程的 `CLI + Python` 双入口
  - 本地软件的输入导入、运行前检查、后台运行、基础队列、停止当前任务
  - 历史加载、参数模板、失败诊断、QC/Warning 展示
  - 桌面 launcher、便携目录版、zip 发布版、GitHub 自动发布链路
- 当前不应称为“完全版”，而应继续按“基础版 / 基础增强版”理解的部分：
  - 多运行对比
  - 展示摘要 HTML 版式
  - 自动化 PDF 导出
  - 单个 exe 完全独立运行
  - 面向 viewer 的结构化 bundle 深化
- 后续文档中的“已完成”如果带有“基础版 / 基础增强版”字样，应理解为：
  - 功能已经能用、能演示、能继续复用
  - 但仍保留进一步补强空间
  - 不等于这个方向已经彻底结束

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
18. [未完成] 如果后续需要，再继续推进成“单个 exe 也能完全独立”的完整便携版。

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
17. [未完成] 如果后续需要，再把桌面程序从“整个目录便携”推进到“单个 exe 完全独立”。

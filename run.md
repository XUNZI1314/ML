# 软件打开方法

这份说明对应当前仓库里的本地软件 `ML Local App`。

## 先看这一张表

| 你看到的文件/目录 | 应该打开什么 | 适合场景 |
|---|---|---|
| `portable_dist\standalone_onefile\ML_Local_App_Standalone.exe` | 双击这个 exe | 最省事，优先用 |
| `portable_dist\ML_Portable\ML_Local_App.exe` | 进入目录后双击这个 exe | 便携目录版，适合拷走 |
| `dist\ML_Local_App.exe` | 在当前仓库里双击这个 exe | 开发机本地版 |
| `start_local_app.bat` | 双击这个 bat | 没有打包版时用源码启动 |
| `run_demo_pipeline.bat` | 双击这个 bat | 没有真实数据时，一键生成 demo 数据并跑完整流程 |

如果你想先理解软件里的 ML 架构，而不是直接运行，先看 [ML.md](ML.md)。如果你想快速用命令行跑模型，先看 [MODEL_QUICKSTART.md](MODEL_QUICKSTART.md)。

优先推荐的打开顺序：

1. 已有单文件版时：双击 `portable_dist\standalone_onefile\ML_Local_App_Standalone.exe`
2. 已有便携版目录时：双击 `portable_dist\ML_Portable\ML_Local_App.exe`
3. 已有仓库内桌面版时：双击 `dist\ML_Local_App.exe`
4. 只用源码运行时：双击 `start_local_app.bat`

如果你不是要打开交互界面，而是想先看完整模型流程效果，可以直接双击 `run_demo_pipeline.bat`。

---

## 方式 1：打开单文件版

这是当前最省事的打开方式。

如果你已经有这个文件：

```text
portable_dist\standalone_onefile\ML_Local_App_Standalone.exe
```

直接双击它即可。

注意：

- 这是“单文件自解压基础版”
- 启动时会先在临时目录解压内嵌的 `app\`
- 当前已经不要求你额外保留同级 `app\` 文件夹

---

## 方式 2：打开便携版

这是最适合直接使用和拷走分发的方式。

如果你已经有这个目录：

```text
portable_dist\ML_Portable\
```

直接进入该目录，然后双击：

```text
ML_Local_App.exe
```

如果你拿到的是压缩包，先解压：

```text
portable_dist\ML_Portable_release.zip
```

解压后进入 `ML_Portable\`，再双击 `ML_Local_App.exe`。

注意：

- 不要只单独复制 `ML_Local_App.exe`
- 要保留整个 `ML_Portable\` 目录结构
- `app\` 和 `app\.venv\` 都是运行所需内容

---

## 方式 3：打开仓库里的桌面版

如果你已经构建过桌面版，并且仓库里有：

```text
dist\ML_Local_App.exe
```

那么直接双击它即可。

这个方式适合你就在当前仓库目录里使用。

注意：

- `dist\ML_Local_App.exe` 最好放在当前仓库目录树内使用
- 如果把这个 exe 单独拿到别处，它可能找不到 `local_ml_app.py` 和 `.venv`

---

## 方式 4：直接从源码启动

如果你还没打包，或者只是想本机直接运行，优先用下面两种方式之一。

### 方式 4A：双击批处理

直接双击：

```text
start_local_app.bat
```

这个脚本现在会自动：

- 优先使用仓库里的 `.venv`
- 自动调用 `ml_desktop_launcher.py`
- 自动走桌面启动器逻辑，不需要你自己再手工敲 `streamlit`

### 方式 4B：命令行启动

在仓库根目录执行：

```bash
python ml_desktop_launcher.py
```

如果你只是想绕过桌面启动器，才用最原始方式：

```bash
python -m streamlit run local_ml_app.py
```

---

## 打开后会发生什么

正常情况下你会看到：

- 一个桌面启动器窗口，或者一个本地 Streamlit 服务进程
- 浏览器自动打开本地页面
- 运行结果写到 `local_app_runs\` 或便携版中的 `app\local_app_runs\`
- 运行完成后，可以在页面里预览共识排名、候选报告卡和下一轮实验建议，并下载对应结果文件
- 如果要看 CD38 公开结构 benchmark 状态，打开页面里的“诊断”页，点击“刷新 CD38 public starter”，然后查看 action plan 和 expected returns 清单
- 如果你的项目有酶的关键催化/功能残基文件，在左侧“可选默认文件”上传或填写 `default_catalytic_file`；主程序会自动生成 catalytic-anchor 4A/6A/8A 3D shell pocket 诊断列，用于复核口袋位置
- 如果要精修 pocket 输入，打开“诊断”页的“Pocket 证据整合”。这里可以运行单 PDB evidence builder，也可以选择标准 `result/` 父目录批量构建项目级 pocket evidence；输出的 `candidate_curated_pocket.txt` 可以一键回填为下一轮 `default_pocket_file`
- Pocket 证据整合默认带安全检查：P2Rank/fpocket 过宽 residue 会进入 `external-overwide-guard` 分组；AI prior 只作为待复核线索，必须通过 `ai_prior_audit.csv` 保留来源句子、证据等级和人工复核状态
- 如果要把 CD38 外部工具输入包拿到另一台机器或 WSL 跑，仍在“诊断”页点击“生成 transfer zip”；zip 里会带 `cd38_external_tool_next_run.md` 和 `run_cd38_external_next_benchmark.*`，外部机器上优先运行这个 next-run 脚本，只跑当前缺的 benchmark blocker；跑完拿回 zip/目录后，先点“Dry-run 检查返回包”，再点“导入返回包并 finalize”
- 如果只是想确认返回包导入器本身能识别正确目录形状，在“诊断”页展开“返回包导入流程自测”；它只做路径/coverage 自测，不代表真实 P2Rank/fpocket 结果
- 返回包 dry-run 后先看“返回包安全门控”：只有 `PASS_*` 状态才适合继续导入；`FAIL_INPUT_PACKAGE` 表示拿回来的是原始输入包，`FAIL_SYNTHETIC_FIXTURE` 表示自测包不能当 benchmark 证据；本地软件和命令行 finalize 正式导入前都会自动 gate 拦截非 `PASS_*` 返回包。自动化脚本可加 `--strict_import_gate` 让 gate 失败时命令返回非零
- 如果想一次确认整条 CD38 外部链路没有断，在“诊断”页展开“CD38 外部工具链路一键自检”，或运行 `python selftest_cd38_external_workflow.py`

---

## 无真实数据时：运行一键 demo

如果你现在还没有自己的 `input_pose_table.csv` 或 `pose_features.csv`，可以先用 demo 检查完整流程。

如果你已经打开本地软件，在左侧“没有数据时”点击：

```text
生成并立即运行 demo
```

软件会自动生成 demo 特征表和 experiment override，并启动后台运行。

运行完成后，本次运行输出目录里会有：

```text
DEMO_OVERVIEW.html
DEMO_README.md
DEMO_INTERPRETATION.md
REAL_DATA_STARTER\
REAL_DATA_STARTER\MINI_PDB_EXAMPLE\
```

先用浏览器打开 `DEMO_OVERVIEW.html`，它适合直接展示 demo 结果。再打开 `DEMO_README.md`，可以快速找到批次结论、候选报告卡和 synthetic 验证证据审计。最后看 `DEMO_INTERPRETATION.md`，它会解释 demo 里的 PASS、候选排序和验证证据边界。

如果 demo 是从本地软件启动的，运行完成后也可以直接在“摘要”页的“Demo 快速导览”里点击“打开 Demo HTML 导览”。如果要换成自己的真实数据，打开 `REAL_DATA_STARTER\README_REAL_DATA_STARTER.md`，从里面的模板开始改；如果想先确认真实 PDB 输入链路，打开 `REAL_DATA_STARTER\MINI_PDB_EXAMPLE\README_MINI_PDB_EXAMPLE.md`。

mini PDB 示例可以从仓库根目录这样跑：

```bash
python build_feature_table.py --input_csv demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/input_pose_table.csv --out_csv demo_outputs/REAL_DATA_STARTER/MINI_PDB_EXAMPLE/mini_pose_features.csv
```

它只用于确认 PDB 解析、口袋位点映射和 pipeline 输入格式，不是生物学 benchmark。

如果你想先检查参数，则点击：

```text
生成并载入 demo 输入
```

软件会自动切到 `feature_csv` 模式，并填入 demo 特征表和 experiment override。确认参数后再点击“立即运行”。

直接双击：

```text
run_demo_pipeline.bat
```

或在仓库根目录运行：

```bash
python run_demo_pipeline.py
```

跑完后优先打开：

```text
demo_outputs\DEMO_README.md
```

常用输出：

- `demo_data\demo_pose_features.csv`
- `demo_data\demo_experiment_plan_override.csv`
- `demo_outputs\batch_decision_summary\batch_decision_summary.md`
- `demo_outputs\candidate_report_cards\index.html`

注意：demo 数据是 synthetic 示例，只用于确认软件流程、报告和导出能力，不代表真实实验验证结论。

---

## 有真实 result 目录时

如果你的数据已经整理成：

```text
A\result\vhh1\CD38_1\1\1.pdb
A\result\vhh1\CD38_1\1\FINAL_RESULTS_MMPBSA.dat
```

可以直接在本地软件左侧“导入目录/zip”选择父目录 `A\`。软件会先找 `pose_features.csv`，没有时找 `input_pose_table.csv`，两者都没有时自动识别 `A\result\` 或 `A\rsite\result\`，生成 `auto_input_pose_table.csv`，并把 `FINAL_RESULTS_MMPBSA.dat` 解析为 `MMPBSA_energy`。

如果你只想一键提取正式 `pose_features.csv`，在 `result` 的父目录执行：

```powershell
python build_pose_features_from_result_tree.py
```

该命令默认输出到当前目录：

```text
pose_features.csv
input_pose_table.csv
feature_qc.json
input_pose_table.report.md
```

如果当前目录下有 `rsite\rsite.txt`，程序会自动按 `antigen_chain=B` 派生 `.ml_auto\auto_pocket_antigen_B.txt`，并把 `*_interface.sc`、`FINAL_DECOMP_MMPBSA.dat`、`MMPBSA_normalized.txt`、`score.txt`、`*_accuracy.txt` 等 sidecar 文件解析进 `pose_features.csv`。

也可以只生成输入表：

```powershell
python build_input_from_result_tree.py --result_root . --out_csv input_pose_table.csv
```

后续 `top_k` 的含义固定为：每个 `vhh/CD38_i/` 下面按 `MMPBSA_energy` 选最低的 K 个 pose；没有能量列时才回退到 Rule/ML 分数。

如果你想在正式跑 ML 前先生成项目级精修 pocket，在同一个父目录下执行：

```powershell
python build_project_pocket_evidence.py --project_root . --target_prefix CD38
```

它会输出 `input_pose_table_with_pocket_evidence.csv`，并把 `candidate_curated_pocket.txt` 写入每一行的 `pocket_file`。如果使用本地软件，则在“诊断 -> Pocket 证据整合 -> 批量 result 父目录模式”点击“从 result 父目录批量构建 pocket evidence”即可。

---

## 启动前快速检查

如果你担心打不开，可以先做自检。

源码目录下执行：

```bash
python ml_desktop_launcher.py --selftest
```

如果你已经有桌面版 exe，也可以执行：

```text
dist\ML_Local_App.exe --selftest
```

如果你已经有单文件版，也可以直接执行：

```text
portable_dist\standalone_onefile\ML_Local_App_Standalone.exe --selftest
```

---

## 打不开时怎么查

先判断是哪一种问题：

| 现象 | 最可能原因 | 先做什么 |
|---|---|---|
| 双击提示拒绝访问 | Windows 权限、杀软拦截或 exe 在受限目录 | 右键属性解除阻止，或移动到普通用户目录再运行 |
| 双击没反应 | 目录结构被拆开或后台进程启动失败 | 看 `desktop_launcher_runtime.log` |
| 打开窗口但网页没出来 | 浏览器没有自动打开 | 点启动器里的“打开浏览器”，或手动访问本地地址 |
| 页面能开但运行失败 | 输入路径、依赖或数据格式问题 | 在页面里先点运行前检查 |

### 情况 1：双击没反应

- 确认是在 Windows 上运行
- 确认目录结构没有被拆开
- 确认 `ML_Local_App.exe`、`app\`、`app\.venv\` 都还在

### 情况 2：窗口开了，但浏览器没自动打开

- 点击启动器窗口里的“打开浏览器”
- 或手动访问窗口显示的本地地址

### 情况 3：程序启动失败

优先查看日志：

便携版：

```text
portable_dist\ML_Portable\app\local_app_runs\desktop_launcher_runtime.log
```

源码 / 仓库版：

```text
local_app_runs\desktop_launcher_runtime.log
```

运行过程产生的结果和日志，也都优先在这些目录里找：

```text
local_app_runs\
```

或便携版中的：

```text
portable_dist\ML_Portable\app\local_app_runs\
```

---

## 最短版本

最简单的打开方式：

- 有单文件版就双击 `portable_dist\standalone_onefile\ML_Local_App_Standalone.exe`
- 没有单文件版但有便携版就双击 `portable_dist\ML_Portable\ML_Local_App.exe`
- 没有便携版就双击 `start_local_app.bat`
- 没有真实数据但想看完整效果，就双击 `run_demo_pipeline.bat`

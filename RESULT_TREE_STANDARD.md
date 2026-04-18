# 标准真实数据目录格式

这个项目后续统一按下面的目录格式接收你的 CD38 复合物批量结果。推荐你在 `result/` 的父目录里使用本地软件或命令行，例如 `A/result/...` 时进入或导入 `A/`。

## 固定目录结构

```text
A/
  result/
    vhh1/
      CD38_1/
        1/
          1.pdb
          FINAL_RESULTS_MMPBSA.dat
          其他该构象对应的计算输出文件
        2/
          2.pdb
        ...
        10/
          10.pdb
      CD38_2/
        1/
          1.pdb
        ...
      CD38_3/
        1/
          1.pdb
        ...
    vhh2/
      CD38_1/
      CD38_2/
      CD38_3/
```

固定含义：

| 层级 | 示例 | 含义 | 写入输入表 |
|---|---|---|---|
| 第 1 层 | `vhh1` | 纳米抗体 ID | `nanobody_id=vhh1` |
| 第 2 层 | `CD38_1` | CD38 结构/构象/批次编号 | `conformer_id=CD38_1`，`target_variant_id=CD38_1` |
| 第 3 层 | `1` | 该 VHH-CD38 组合下的 pose 编号 | `pose_id=1`，`pose_index=1` |
| PDB 文件 | `1/1.pdb` | 当前 pose 的复合物结构 | `pdb_path=<...>/result/vhh1/CD38_1/1/1.pdb` |
| 能量文件 | `1/FINAL_RESULTS_MMPBSA.dat` | 当前 pose 的 MMPBSA 结果 | `MMPBSA_energy` |

建议保持每个 `CD38_x` 下面 10 个 pose 文件夹。系统不强制只能 10 个，但如果数量不一致，后续报告会更容易看出缺失。

## 自动识别优先级

本地软件或转换脚本现在按这个顺序处理：

1. 如果在你导入的目录或 zip 中发现 `pose_features.csv`，优先直接使用它，不再重建 `input_csv`。
2. 如果没有 `pose_features.csv`，但发现 `input_pose_table.csv`，使用现有输入表。
3. 如果两者都没有，会自动浅层查找标准 `result/` 目录，例如你选择 `A/` 时自动识别 `A/result/`。
4. 只有找不到标准 `result/` 时，才回退到普通 PDB 扫描并保守生成 `auto_input_pose_table.csv`。

因此标准批量数据不需要你手工填写 `input_csv`。

## 生成 input_pose_table.csv

如果当前目录是 `A/`，且数据在 `A/result/`，最短命令是：

```powershell
python build_input_from_result_tree.py --result_root . --out_csv input_pose_table.csv
```

如果当前目录外已经明确知道 `result/` 路径，也可以写成：

```powershell
python build_input_from_result_tree.py --result_root result --out_csv input_pose_table.csv
```

如果有统一 pocket / catalytic 定义：

```powershell
python build_input_from_result_tree.py ^
  --result_root . ^
  --out_csv input_pose_table.csv ^
  --default_pocket_file cd38_pocket.txt ^
  --default_catalytic_file cd38_catalytic.txt ^
  --default_antigen_chain A
```

生成后直接跑主流程：

```powershell
python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir my_outputs
```

## 生成的输入表列

核心列：

| 列 | 来源 |
|---|---|
| `nanobody_id` | `result/vhh1` |
| `conformer_id` | `result/vhh1/CD38_1` |
| `pose_id` | `result/vhh1/CD38_1/1` |
| `pdb_path` | `result/vhh1/CD38_1/1/1.pdb` |
| `MMPBSA_energy` | 从 `result/vhh1/CD38_1/1/FINAL_RESULTS_MMPBSA.dat` 的最后一个 `DELTA TOTAL` 解析 |

额外追踪列：

| 列 | 作用 |
|---|---|
| `target_id` | 例如 `CD38` |
| `target_variant_id` | 例如 `CD38_1` |
| `target_variant_index` | 例如 `1` |
| `pose_index` | 例如 `1` |
| `complex_id` | `vhh1__CD38_1__pose_1` |
| `pose_dir` | 当前 pose 文件夹路径 |
| `source_relative_dir` | 相对 `result/` 的目录 |
| `sidecar_file_count` | 除 PDB 外的计算输出文件数量 |
| `sidecar_files` | 除 PDB 外的计算输出文件名列表 |
| `mmpbsa_result_file` | `FINAL_RESULTS_MMPBSA.dat` 路径 |
| `mmpbsa_parse_status` | `ok` / `missing` / `warning` 等解析状态 |
| `layout_status` | `ok` / `warning` |
| `layout_warning` | 不规范命名时的提示 |

这些额外列不会破坏现有模型。`build_feature_table.py` 会继续使用核心列构建结构几何特征，并忽略不能转成数值的文本追踪列。

## 命名规则

强烈建议：

| 对象 | 推荐命名 |
|---|---|
| VHH 文件夹 | `vhh1`、`vhh2`、`vhh3` |
| CD38 结构文件夹 | `CD38_1`、`CD38_2`、`CD38_3` |
| pose 文件夹 | `1` 到 `10` |
| pose PDB 文件 | 和 pose 文件夹同名，例如 `1/1.pdb` |

默认扫描器只接受 `pose_id/pose_id.pdb`。如果某些目录里只有一个 PDB，但文件名不是 `1.pdb`，可以临时放宽：

```powershell
python build_input_from_result_tree.py --result_root . --out_csv input_pose_table.csv --allow_single_pdb_fallback
```

这个参数只适合过渡期。长期建议把 PDB 文件改成标准名，避免 9000 个文件批处理时出现歧义。

## Top-K 在这个目录里的准确含义

在这个项目里，Top-K 对你的目录结构不是“Top-K 个 VHH”，也不是“Top-K 个 CD38_i”。它的准确含义是：

> 在每个 `result/<vhh>/<CD38_i>/` 下面，从 10 个 pose 里按 `MMPBSA_energy` 选最低的 K 个 pose。

例如 `top_k=3` 时：

```text
result/vhh1/CD38_1/ 选 MMPBSA 最低的 3 个 pose
result/vhh1/CD38_2/ 选 MMPBSA 最低的 3 个 pose
result/vhh1/CD38_3/ 选 MMPBSA 最低的 3 个 pose
```

如果 `FINAL_RESULTS_MMPBSA.dat` 还没有解析出来，但你已经把结果整理到 `pose_features.csv`，列名用 `MMPBSA_energy` 即可。系统会优先识别这些能量列：

```text
MMPBSA_energy
mmpbsa_energy
MMGBSA_energy
mmgbsa_energy
mmgbsa
MMGBSA
```

默认排序规则：

| 情况 | Top-K 选择方式 |
|---|---|
| 有 `MMPBSA_energy` / `mmgbsa` 等能量列 | 选能量最低的 K 个 pose |
| 没有能量列 | 回退到模型分数或规则分数最高的 K 个 pose |

推荐默认仍是 `top_k=3`。如果每个 `CD38_i` 下面固定 10 个 pose，`top_k=3` 表示每个 CD38 构象取最低能量的 30% pose。

## 不要提交 result/ 到 GitHub

`result/` 已加入 `.gitignore`。真实批量结构数据应该保留在本地机器或数据盘里，不直接提交到 GitHub。仓库只保存脚本、文档、benchmark 小样本和可复现配置。

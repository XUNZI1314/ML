# Source Bundle Fallback

这个文件说明的是一个临时兜底方案：

- 当前仓库已经有完整说明文档：`README.md`、`MODEL_QUICKSTART.md`、`run.md`、`ML.md`、`CD38.md`、`advantage.md`、`not_perfect.md` 和 `benchmarks/cd38/README.md`。
- 如果当前机器无法直接完成 `git push` 全量同步，可以先把完整源码打包成 `ML_source_bundle.zip.base64` 放到仓库。
- 这样至少可以保证 GitHub 仓库里已经保存了一份完整源码快照。

## 什么时候需要看这个文件

| 情况 | 是否需要 |
|---|---|
| GitHub 正常 push 可用 | 不需要，直接用正常 Git 提交和推送 |
| GitHub 登录、凭据或网络暂时不可用 | 可以用这个 bundle 做临时备份 |
| 需要给别人一个完整源码快照 | 可以用，但优先还是发 release zip 或正常仓库 |
| 新增脚本、benchmark、文档或本地软件功能后 | 需要重新生成 bundle，否则里面还是旧版本 |

## 当前文档入口

| 文档 | 作用 |
|---|---|
| `README.md` | 项目总览、功能清单和长版命令说明 |
| `MODEL_QUICKSTART.md` | 最短模型运行路径 |
| `run.md` | 本地软件打开方法 |
| `ML.md` | ML 架构、模型训练和排序逻辑 |
| `CD38.md` | CD38 当前能力、benchmark 结果边界和下一步补齐路径 |
| `advantage.md` | 项目优势、同类产品缺陷和后续改进方向 |
| `not_perfect.md` | 当前未收尾项和默认继续推进规则 |
| `benchmarks/cd38/README.md` | CD38 pocket benchmark 专项说明 |

## 恢复方式

在 Windows PowerShell 中执行：

```powershell
.\restore_source_bundle.ps1
```

执行后会：

1. 读取 `ML_source_bundle.zip.base64`
2. 还原出 `ML_source_bundle.zip`
3. 解压到 `restored_source_bundle/`

## 说明

- 这是在本机 `git push` 凭据不可用时的兜底措施。
- 后续一旦本机 GitHub 认证恢复正常，仍建议再把源码按正常 Git 历史完整推送到仓库。
- 如果新增了分析脚本、benchmark 结果、next-run runbook 或文档更新，应重新生成源码 bundle；否则 `ML_source_bundle.zip.base64` 只代表旧快照。

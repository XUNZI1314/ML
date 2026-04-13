# Source Bundle Fallback

这个文件说明的是一个临时兜底方案：

- 当前仓库已经有在线 README。
- 如果当前机器无法直接完成 `git push` 全量同步，可以先把完整源码打包成 `ML_source_bundle.zip.base64` 放到仓库。
- 这样至少可以保证 GitHub 仓库里已经保存了一份完整源码快照。

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

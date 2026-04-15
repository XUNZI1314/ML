# 软件打开方法

这份说明对应当前仓库里的本地软件 `ML Local App`。

优先推荐的打开方式有 3 种：

1. 已有便携版目录时：双击 `portable_dist\ML_Portable\ML_Local_App.exe`
2. 已有仓库内桌面版时：双击 `dist\ML_Local_App.exe`
3. 只用源码运行时：双击 `start_local_app.bat`

---

## 方式 1：打开便携版

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

## 方式 2：打开仓库里的桌面版

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

## 方式 3：直接从源码启动

如果你还没打包，或者只是想本机直接运行，优先用下面两种方式之一。

### 方式 3A：双击批处理

直接双击：

```text
start_local_app.bat
```

这个脚本现在会自动：

- 优先使用仓库里的 `.venv`
- 自动调用 `ml_desktop_launcher.py`
- 自动走桌面启动器逻辑，不需要你自己再手工敲 `streamlit`

### 方式 3B：命令行启动

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

---

## 打不开时怎么查

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

- 有便携版就双击 `portable_dist\ML_Portable\ML_Local_App.exe`
- 没有便携版就双击 `start_local_app.bat`

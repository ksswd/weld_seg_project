# GitHub 基本操作文档（本项目）

本文档用于把当前项目上传到 GitHub，并支持后续日常更新。

---

## 1. 前置准备

1. 安装 Git（确认可用）
```bash
git --version
```

2. 配置用户名和邮箱（首次使用需要）
```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

3. 在 GitHub 上新建一个空仓库（建议不要勾选 README/.gitignore/license，避免首次推送冲突）

---

## 2. 在项目目录初始化并首次上传

在项目根目录执行：

```bash
# 进入项目目录
cd /你的项目路径/weld_seg_project

# 初始化仓库（若已初始化可跳过）
git init

# 查看当前状态
git status

# 添加全部文件到暂存区
git add .

# 首次提交
git commit -m "init: first commit"

# 绑定远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/你的仓库名.git

# 推送到 main 分支
# 若本地默认分支不是 main，可先执行: git branch -M main
git branch -M main
git push -u origin main
```

---

## 3. 后续日常更新流程

每次改完代码后执行：

```bash
git status
git add .
git commit -m "feat/fix/docs: 本次修改说明"
git push
```

建议提交信息采用规范前缀：
- `feat:` 新功能
- `fix:` 修复
- `docs:` 文档改动
- `refactor:` 重构
- `test:` 测试相关

---

## 4. 常见问题

### 4.1 远程已存在提交，首次推送失败
如果 GitHub 仓库不是空仓库，可能报错 `rejected`。可先拉取再推送：

```bash
git pull --rebase origin main
git push -u origin main
```

### 4.2 推送需要认证
GitHub 已不支持账号密码直接推送，建议：
- 使用 Personal Access Token（PAT）
- 或配置 SSH Key（推荐长期使用）

### 4.3 不想上传大文件/中间结果
建议配置 `.gitignore`，忽略如：
- 训练权重 `weights/`, `weights2/`
- 日志与输出 `logs/`, `data/new/full_eval_all/`
- 临时文件 `*.out`, `__pycache__/`

---

## 5. 建议的首个 `.gitignore` 条目（可按需增减）

```gitignore
# Python
__pycache__/
*.pyc

# logs / outputs
logs/
*.out

# model weights
weights/
weights2/

# evaluation outputs
data/new/full_eval_all/
data/new/full_eval/

# cache
.cursor/
```

---

## 6. 推荐操作习惯

1. 每次改动后先 `git status` 检查将被提交的文件。  
2. 大功能用分支开发：
```bash
git checkout -b feat/soft-label-v2
```
完成后再合并到 `main`。  
3. 每次推送前写清楚提交信息，方便以后回溯实验版本。

---

## 7. 一条命令快速查看提交历史

```bash
git log --oneline --graph --decorate --all
```

可用于快速定位某次实验对应的代码版本。

---

## 8. 定制场景：保留远程 `main` 不动，直接开新分支上传

你当前场景：
- GitHub 上已有 `main` 分支；
- 本地和远程 `main` 冲突较多，不想继续在 `main` 上对齐；
- 希望把当前本地版本直接作为一个新分支上传。

### 方案A（推荐）：从当前本地状态直接新建并推送分支

```bash
# 1) 查看当前状态
git status

# 2) 若有未提交改动，先提交（避免切分支丢内容）
git add .
git commit -m "chore: snapshot before creating new branch"

# 3) 基于当前代码新建分支（分支名可自定义）
git checkout -b exp/new-pipeline

# 4) 推送这个新分支到远程（不会改动远程main）
git push -u origin exp/new-pipeline
```

之后你所有更新都在这个新分支上进行：

```bash
git add .
git commit -m "feat: your update"
git push
```

### 方案B：如果你本地当前就在 `main`，也可以直接“切出并推送”

```bash
# 在本地main上切新分支
git switch -c exp/new-pipeline

# 推送新分支
git push -u origin exp/new-pipeline
```

### 方案C：彻底避免误推 main（可选但推荐）

将本地默认工作分支也改成新分支：

```bash
# 确保你已经在新分支上
git branch --show-current

# 可选：把本地main切走，日常只在新分支工作
git switch exp/new-pipeline
```

并在推送时显式指定分支，避免误操作：

```bash
git push origin exp/new-pipeline
```

### 补充：如果你未来想把新分支内容合并回 main

在 GitHub 页面发起 Pull Request：
- base: `main`
- compare: `exp/new-pipeline`

这样可以可视化审查冲突，再决定是否合并。

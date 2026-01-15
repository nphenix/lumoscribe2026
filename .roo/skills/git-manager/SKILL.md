---
name: git-manager
description: "git, commit, branch, merge, pull, push, version control, repository"
allowed-tools:
  - read_file
  - write_file
  - execute_command
---

# Git 管理器

## 用途

本技能提供AI驱动的Git版本控制管理能力，包括：
- 自动生成符合Conventional Commits规范的提交信息
- 分支创建、切换、合并和删除管理
- 代码变更分析和差异对比
- 合并冲突辅助识别和解决
- Git工作流最佳实践建议

## 使用时机

在以下情况下调用此技能：
- 需要为代码变更生成规范的提交信息
- 需要创建、切换或合并Git分支
- 需要分析变更历史和差异
- 需要解决合并冲突
- 需要优化Git工作流程
- 需要同步 skills 到其他工具（如 Cursor）
- **自动触发**：修改了 `speckit/skills/` 下的任何文件后，必须自动执行同步

## 不使用时机

- 只需要基础的git status查询（直接使用对话）
- 仓库不包含代码或不需要版本控制
- 已熟悉Git操作不需要AI辅助

## 工具使用说明

### 命令执行工具

使用命令执行工具执行Git命令：

```bash
# 查看仓库状态
git status

# 查看暂存的变更
git diff --cached

# 查看所有变更（包括未暂存）
git diff

# 查看分支列表
git branch -a

# 查看提交历史
git log --oneline -10

# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout develop

# 合并分支
git merge feature/xxx

# 暂存所有变更
git add .

# 提交变更
git commit -m "message"

# 推送到远程
git push origin branch-name

# 拉取最新变更
git pull origin branch-name

# 同步 skills 到工具（执行 PowerShell 脚本）
pwsh scripts/sync-skills.ps1

# 预览同步（不执行）
pwsh scripts/sync-skills.ps1 -WhatIf
```

### 文件读取工具

使用文件读取工具查看文件内容：

```bash
# 读取提交信息模板
读取 speckit/skills/git-manager/references/template-commit.md

# 读取Git策略文档
读取 git-strategy.md

# 读取变更记录模板
读取 speckit/templates/change-template.md
```

### 文件写入工具

使用文件写入工具创建或修改文件：

```bash
# 创建或更新提交信息模板
写入文件 speckit/skills/git-manager/references/template-commit.md "内容"

# 记录变更
写入文件 speckit/specs/changes/xxx-change.md "内容"
```

## 工作流程

### 生成提交信息

1. 使用命令执行工具执行 `git diff --cached` 查看暂存的代码变更
2. 分析变更内容，确定变更类型（feat/fix/docs/style/refactor/test/chore）
3. 识别变更涉及的模块（作用域）
4. 生成符合Conventional Commits规范的提交信息
5. 返回建议的提交信息供用户确认

### 管理分支

1. 使用命令执行工具执行 `git branch` 查看当前分支状态
2. 使用命令执行工具执行 `git checkout -b` 创建新分支
3. 使用命令执行工具执行 `git merge` 合并分支
4. 处理可能的合并冲突
5. 验证操作结果并提供反馈

### 分析变更

1. 使用命令执行工具执行 `git diff` 比较差异
2. 使用命令执行工具执行 `git log` 查看提交历史
3. 识别变更的代码区域和影响范围
4. 提供变更摘要和优化建议

### 解决冲突

1. 使用文件读取工具查看冲突文件
2. 分析冲突原因
3. 与用户沟通解决方案
4. 编辑文件解决冲突
5. 使用命令执行工具执行 `git add` 标记冲突已解决
6. 使用命令执行工具执行 `git commit` 完成合并

### 同步 Skills 到 Cursor 和 Roo Code

**自动触发条件**：当检测到修改了 `speckit/skills/` 目录下的文件时，必须自动执行同步到两个工具。

**自动同步工作流程**：
1. 检测到修改路径包含 `speckit/skills/`
2. **自动执行**：`pwsh scripts/sync-skills.ps1`（同时同步到 `.cursor/skills/` 和 `.roo/skills/`）
3. 检查输出，确认两个目标的同步结果
4. 提示用户：`执行了 skills 同步，建议提交变更：git add .cursor/skills .roo/skills && git commit -m "chore: sync skills"`

**手动触发**（备用）：
1. 用户执行 `/sync-skills` 或说"同步 skills"
2. 执行脚本：`pwsh scripts/sync-skills.ps1`
3. 可选参数：`-Target cursor`（只同步 Cursor）或 `-Target roo`（只同步 Roo Code）
4. 检查输出，确认同步结果

## 提交信息规范

### 格式

```
<type>(<scope>): <subject>
```

### 类型（Type）

| 类型 | 描述 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(git-manager): add branch management` |
| fix | 修复bug | `fix(debug-diagnostic): resolve parsing error` |
| docs | 文档更新 | `docs: update git strategy` |
| style | 代码格式 | `style: format code` |
| refactor | 重构 | `refactor: improve performance` |
| test | 测试相关 | `test: add unit tests` |
| chore | 构建或辅助工具 | `chore: update dependencies` |

### 作用域（Scope）

| 作用域 | 描述 |
|--------|------|
| git-manager | Git管理技能 |
| skill-generator | 技能生成器 |
| plan-creator | 计划创建器 |
| task-breakdown | 任务分解器 |
| change-manager | 变更管理器 |
| debug-diagnostic | 调试诊断器 |
| templates | 模板相关 |
| config | 配置相关 |

### 示例

```
feat(git-manager): 添加MiniMax API集成支持

- 支持使用MiniMax API生成提交信息
- 自动过滤敏感数据
- 集成Git Hooks自动生成提交信息

Closes #123
```

## 最佳实践

### Skills 同步

- **源目录**：`speckit/skills/`
- **目标目录**：
  - `.cursor/skills/` (Cursor)
  - `.roo/skills/` (Roo Code)
- 修改源文件后**自动执行同步**
- 同步后提交变更：`git add .cursor/skills .roo/skills && git commit -m "chore: sync skills"`
- 使用 `/sync-skills` 命令可手动触发

### 提交规范

- 使用Conventional Commits格式
- 描述清晰简洁，不超过72字符
- 说明变更原因和影响
- 关联相关Issue或PR

### 分支管理

- 遵循Git Flow分支策略
- main/master分支保持稳定
- develop分支作为开发主干
- 功能开发使用feature分支
- 发布准备使用release分支
- 紧急修复使用hotfix分支

### 提交频率

- 小而频繁的提交
- 每个提交一个逻辑单元
- 保持原子性提交
- 便于回滚和审查

### 代码审查

- 通过Pull Request进行审查
- 至少一人代码审查
- CI/CD检查必须通过
- 遵循项目编码规范

## 注意事项

### 敏感信息

- 过滤API密钥、密码等敏感数据
- 不在提交信息中暴露敏感路径
- 使用环境变量存储凭证

### 操作确认

- 危险操作需要用户确认（force push、删除分支等）
- 执行前说明操作影响
- 提供撤销方案

### 错误处理

- 提供清晰的错误信息
- 说明错误原因和解决方案
- 保留操作历史便于恢复

## 相关资源

- `speckit/skills/git-manager/references/template-commit.md` - 提交信息模板
- `speckit/templates/change-template.md` - 变更记录模板
- `git-strategy.md` - Git分支策略文档

## 版本

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0.0 | 2026-01-12 | 初始版本，支持提交信息生成和基础Git操作 |
| 1.1.0 | 2026-01-12 | 重构为纯Claude Agent Skill格式，使用通用工具描述 |
| 1.2.0 | 2026-01-12 | 添加自动同步 skills 到 Cursor 能力 |

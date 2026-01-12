---
name: "git-manager"
description: "AI驱动的Git管理技能。自动生成规范的提交信息、管理分支、解决冲突和提供Git工作流建议。在需要执行Git操作、分析变更历史或优化版本控制流程时使用。"
allowed-tools: Bash, Read, Write
---

# Git管理器技能

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

## 不使用时机

- 只需要基础的git status查询（直接使用对话）
- 仓库不包含代码或不需要版本控制
- 已熟悉Git操作不需要AI辅助

## 工具使用说明

### Bash工具

使用Bash工具执行Git命令：

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
```

### Read工具

使用Read工具查看文件内容：

```bash
# 读取提交信息模板
Read speckit/skills/git-manager/references/template-commit.md

# 读取Git策略文档
Read git-strategy.md

# 读取变更记录模板
Read speckit/templates/change-template.md
```

### write_to_file工具

使用write_to_file工具创建或修改文件：

```bash
# 创建或更新提交信息模板
write_to_file speckit/skills/git-manager/references/template-commit.md "内容"

# 记录变更
write_to_file speckit/specs/changes/xxx-change.md "内容"
```

## 工作流程

### 生成提交信息

1. 使用Bash执行 `git diff --cached` 查看暂存的代码变更
2. 分析变更内容，确定变更类型（feat/fix/docs/style/refactor/test/chore）
3. 识别变更涉及的模块（作用域）
4. 生成符合Conventional Commits规范的提交信息
5. 返回建议的提交信息供用户确认

### 管理分支

1. 使用Bash执行 `git branch` 查看当前分支状态
2. 使用Bash执行 `git checkout -b` 创建新分支
3. 使用Bash执行 `git merge` 合并分支
4. 处理可能的合并冲突
5. 验证操作结果并提供反馈

### 分析变更

1. 使用Bash执行 `git diff` 比较差异
2. 使用Bash执行 `git log` 查看提交历史
3. 识别变更的代码区域和影响范围
4. 提供变更摘要和优化建议

### 解决冲突

1. 使用Read工具查看冲突文件
2. 分析冲突原因
3. 与用户沟通解决方案
4. 编辑文件解决冲突
5. 使用Bash执行 `git add` 标记冲突已解决
6. 使用Bash执行 `git commit` 完成合并

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

1.0.0 | 2026-01-12 | 初始版本，支持提交信息生成和基础Git操作
1.1.0 | 2026-01-12 | 重构为纯Claude Agent Skill格式，使用内置Bash/Read/write_to_file工具
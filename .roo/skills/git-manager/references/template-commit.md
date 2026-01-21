# 提交信息模板

## Conventional Commits 格式

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

## 类型定义 (Type)

| 类型 | 描述 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(git-manager): add commit message generation` |
| `fix` | 修复bug | `fix(debug-diagnostic): resolve template parsing error` |
| `docs` | 文档更新 | `docs: update git integration guide` |
| `style` | 代码格式 | `style: format code according to eslint rules` |
| `refactor` | 重构 | `refactor: improve git command execution` |
| `test` | 测试相关 | `test: add unit tests for git hooks` |
| `chore` | 构建或辅助工具 | `chore: update dependencies` |

## 作用域定义 (Scope)

| Scope | 描述 | 适用场景 |
|-------|------|---------|
| `skill-generator` | 技能生成器模块 | 新增或修改AI skills |
| `plan-creator` | 计划创建器模块 | 计划相关功能 |
| `task-breakdown` | 任务分解器模块 | 任务分解功能 |
| `change-manager` | 变更管理器模块 | 变更管理功能 |
| `debug-diagnostic` | 调试诊断器模块 | 调试功能 |
| `git-manager` | Git管理器模块 | Git集成功能 |
| `templates` | 模板相关 | 模板文件修改 |
| `config` | 配置相关 | 配置文件修改 |

## 提交信息示例

### 功能开发

```
feat(git-manager): add MiniMax API integration for commit messages

This commit adds support for using MiniMax API (OpenAI compatible mode)
to generate conventional commit messages automatically.

- Configurable API endpoint
- Support for multiple models
- Safety features for sensitive data

Closes #123
```

### Bug修复

```
fix(debug-diagnostic: resolve template parsing error when handling special characters

The template parser was failing when encountering special characters like
backticks and dollar signs. This fix adds proper escaping.

Before: Template parsing failed with "Unexpected token"
After: Templates with special characters are handled correctly

Fixes #456
```

### 重构

```
refactor(git-manager): extract Git command execution to separate module

This refactoring moves Git command execution logic into a dedicated module
for better separation of concerns and easier testing.

- Created GitExecutor class
- Improved error handling
- Added unit tests

Related to #789
```

## AI生成提示词

当使用AI生成提交信息时，提供以下上下文：

```
分析以下代码变更并生成提交信息：

变更的文件：
{script}

Git diff：
{diff}

要求：
1. 使用 Conventional Commits 格式
2. 类型只能是：feat, fix, docs, style, refactor, test, chore
3. 作用域必须是：git-manager, skill-generator, plan-creator, task-breakdown, change-manager, debug-diagnostic, templates, config
4. 描述简洁明了，不超过72字符
5. 如果有相关Issue，请包含在footer中

生成格式：
type(scope): description
```

## 提交信息检查清单

在提交前检查：

- [ ] 类型是否正确
- [ ] 作用域是否准确
- [ ] 描述是否简洁（不超过72字符）
- [ ] 是否符合项目规范
- [ ] 是否包含相关Issue引用
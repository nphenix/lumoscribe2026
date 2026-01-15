# Agent Skills 集成配置

本项目使用 Claude Agent Skills 标准（https://github.com/agentskills/agentskills）定义 AI 代理能力。

## Source of Truth (单一事实来源)

本项目采用 **`speckit/`** 目录作为所有 AI Agent Skills 的单一事实来源：

### 主源目录（不可手改）

- `speckit/skills/` - 技能定义库（SKILL.md 模板）
- `speckit/templates/` - 项目模板库
- `speckit/specs/` - 技术规范和设计文档
- `speckit/constitution.md` - 项目核心宪章
- `speckit/directory-structure.md` - 目录结构规范

### 同步产物目录（自动生成，禁止手改）

- `.cursor/skills/` - Cursor Agent Skills 同步目录
- `.roo/skills/` - Roo Code Skills 同步目录

### 同步机制

所有同步产物通过 `scripts/sync-skills.ps1` 脚本自动生成。

**同步命令：**

```powershell
# 同步所有技能到 Cursor 和 Roo
pwsh scripts/sync-skills.ps1
```

**重要：** 同步产物目录中的文件 **不要手动修改**，所有修改应在主源目录进行，然后重新运行同步脚本。

## 最短上手路径

### 新增一个技能

1. 在 `speckit/skills/` 下创建新技能目录
2. 编写 `SKILL.md`（遵循 Agent Skills 规范）
3. 在 `speckit/skills/README.md` 中注册新技能
4. 运行同步脚本：`pwsh scripts/sync-skills.ps1`
5. 在 Cursor/Roo 中验证新技能是否可用

### 修改现有技能

1. 在 `speckit/skills/<skill-name>/SKILL.md` 中进行修改
2. 运行同步脚本：`pwsh scripts/sync-skills.ps1`
3. 在 Cursor/Roo 中验证修改效果

## 可用技能

| # | 名称 | 位置 |
|---|------|------|
| 1 | git-manager | `speckit/skills/git-manager/SKILL.md` |
| 2 | code-review | `speckit/skills/code-review/SKILL.md` |
| 3 | plan-creator | `speckit/skills/plan-creator/SKILL.md` |
| 4 | task-breakdown | `speckit/skills/task-breakdown/SKILL.md` |
| 5 | spec-generator | `speckit/skills/spec-generator/SKILL.md` |
| 6 | change-manager | `speckit/skills/change-manager/SKILL.md` |
| 7 | debug-diagnostic | `speckit/skills/debug-diagnostic/SKILL.md` |

## 技能使用指南

### 自动激活

当对话内容匹配技能的 `description` 时，AI 代理会自动激活相应技能。

### 手动激活

也可以直接引用技能名称：

```
使用 git-manager 技能生成提交信息
使用 plan-creator 技能创建项目计划
```

### 渐进式披露

技能采用渐进式加载机制：

1. **启动时**: 仅加载技能名称和描述（轻量级）
2. **激活时**: 加载完整技能指令
3. **执行时**: 加载引用文件和脚本

## 项目配置

### Roo Code 配置

- **规则目录**: `.roo/rules/`
- **命令目录**: `.roo/commands/`
- **AGENTS.md**: 本文件（跨工具兼容）

### Cursor 配置

- **规则目录**: `.cursor/rules/`
- **命令目录**: `.cursor/commands/`

## 交互方式

本项目 **不使用** 斜杠命令（如 `/plan`）。请直接使用自然语言描述你的意图：

- "请帮我规划一下这个功能的架构" -> 激活 `plan-creator`
- "我要提交代码" -> 激活 `git-manager`
- "帮我 review 一下刚才的修改" -> 激活 `code-review`
| `/code-review` | 使用 code-review 进行代码审查 |
| `/plan` | 使用 plan-creator 创建项目计划 |

## 技能开发

### 创建新技能

1. 在 `speckit/skills/` 创建新目录
2. 创建 `SKILL.md` 文件（必需）
3. 添加 `scripts/`、`references/`、`assets/` 子目录（可选）
4. 遵循 Agent Skills 规范格式

### SKILL.md 格式

```yaml
---
name: skill-name
description: 技能描述，说明用途和触发条件
allowed-tools: 允许使用的工具列表
---
# 技能详细说明
Markdown 格式的指令和指南
```

## 最佳实践

1. **技能数量**: 建议每个项目不超过 10-15 个活跃技能
2. **职责单一**: 每个技能专注于一个特定领域
3. **描述清晰**: description 应简洁准确，便于智能匹配
4. **版本控制**: 使用语义版本，添加版本更新日志
5. **跨工具兼容**: 确保技能可在不同 AI 代理工具间共享

## 版本信息

- **技能规范版本**: 1.0+
- **最后更新**: 2026-01-12
- **维护者**: AI Development Team

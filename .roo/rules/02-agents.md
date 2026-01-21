# Agent Skills 引用
引用自 [`AGENTS.md`](../AGENTS.md)。

## Source of Truth (单一事实来源)

本项目采用 **`speckit/`** 目录作为所有 AI Agent Skills 的单一事实来源：

- `speckit/skills/` - 技能定义库（SKILL.md 模板）
- `speckit/templates/` - 项目模板库
- `speckit/specs/` - 技术规范和设计文档

**同步产物**（自动生成，禁止手改）：
- `.cursor/skills/` - Cursor Agent Skills 同步目录
- `.roo/skills/` - Roo Code Skills 同步目录

## 技能
| # | 名称 | 位置 |
|---|------|------|
| 1 | git-manager | `speckit/skills/git-manager/SKILL.md` |
| 2 | code-review | `speckit/skills/code-review/SKILL.md` |
| 3 | plan-creator | `speckit/skills/plan-creator/SKILL.md` |
| 4 | task-breakdown | `speckit/skills/task-breakdown/SKILL.md` |
| 5 | spec-generator | `speckit/skills/spec-generator/SKILL.md` |
| 6 | change-manager | `speckit/skills/change-manager/SKILL.md` |
| 7 | debug-diagnostic | `speckit/skills/debug-diagnostic/SKILL.md` |

## 命令
`/list-skills` `/generate-commit` `/code-review` `/plan`
> 完整说明见 [`AGENTS.md`](../AGENTS.md)

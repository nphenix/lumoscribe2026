# Speckit 代理技能系统

## 概述

本目录包含一套专为软件开发工作流设计的 AI 代理技能。这些技能基于 Speckit 模板系统构建，并遵循 Anthropic 的 [Agent Skills](https://github.com/agentskills/agentskills) 标准。

## 集成状态

### ✅ 已集成工具

| 工具 | 状态 | 配置目录 | 配置方式 | 命令前缀 |
|------|------|----------|----------|----------|
| **Roo Code** | ✅ 已集成 | `.roo/` | 引用 `AGENTS.md` | `/` |
| **Cursor** | ✅ 已集成 | `.cursor/` | 内联集成 | `/` |
| **Trae IDE** | ✅ 已集成 | `.trae/` | 复制 `SKILL.md` 到 `rules/` | 自然语言 |
| **Claude Code** | ⚪ 待集成 | `.claude/` | 待配置 | 待配置 |

## 技能架构

```
speckit/
├── skills/                           # 技能库
│   ├── README.md                     # 本文件
│   ├── git-manager/                  # Git管理技能
│   │   ├── SKILL.md                  # 技能定义
│   │   └── references/               # 参考文档
│   │       └── template-commit.md    # 提交模板参考
│   ├── code-review/                  # 代码审查技能
│   │   ├── SKILL.md                  # 技能定义
│   │   └── references/               # 参考文档
│   │       └── template-review.md    # 审查模板参考
│   ├── spec-generator/
│   │   ├── SKILL.md                  # 技能定义
│   │   └── references/               # 参考文档
│   │       └── template-spec.md      # 规格模板参考
│   ├── plan-creator/
│   │   ├── SKILL.md                  # 技能定义
│   │   └── references/               # 参考文档
│   │       └── template-plan.md      # 计划模板参考
│   ├── task-breakdown/
│   │   ├── SKILL.md                  # 技能定义
│   │   └── references/               # 参考文档
│   │       └── template-tasks.md     # 任务模板参考
│   ├── change-manager/
│   │   ├── SKILL.md                  # 技能定义
│   │   └── references/               # 参考文档
│   │       └── template-change.md    # 变更模板参考
│   └── debug-diagnostic/
│       ├── SKILL.md                  # 技能定义
│       └── references/               # 参考文档
│           └── template-debug.md     # 调试模板参考
├── templates/                        # 模板库
└── specs/                            # 规格文档
    ├── features/[name]/              # 功能文档
    ├── changes/[id]/                 # 变更记录
    └── debug/[id]/                   # 调试记录
```

## 技能清单

### git-manager（Git管理器）
**用途**: AI驱动的Git版本控制管理，包括提交信息生成、分支管理、冲突解决
**触发**: git, commit, branch, merge, version control, push, pull
**输出**: 规范的提交信息、分支操作、冲突解决方案
**位置**: `speckit/skills/git-manager/SKILL.md`

### code-review（代码审查器）
**用途**: 自动化代码质量审查，包括识别代码异味、安全漏洞、性能问题
**触发**: code review, quality analysis, best practices, static analysis
**输出**: 结构化审查报告、改进建议
**位置**: `speckit/skills/code-review/SKILL.md`

### spec-generator（规格生成器）
**用途**: 根据用户需求生成功能规格  
**触发**: 用户需要记录新功能或能力  
**输出**: `speckit/specs/features/[name]/spec.md`

### plan-creator（计划创建器）
**用途**: 根据规格创建实施方案  
**触发**: 功能规格已完成，需要技术方案  
**输出**: `speckit/specs/features/[name]/plan.md`

### task-breakdown（任务分解器）
**用途**: 将计划分解为可执行任务  
**触发**: 实施方案已完成，需要执行  
**输出**: `speckit/specs/features/[name]/tasks.md`

### change-manager（变更管理器）
**用途**: 跟踪和管理功能变更  
**触发**: 范围变更、API 更新或需求修改  
**输出**: `speckit/specs/changes/[id]/change.md`

### debug-diagnostic（调试诊断器）
**用途**: 记录调试过程和解决方案  
**触发**: 开发过程中发现 bug 或问题  
**输出**: `speckit/specs/debug/[id]/debug.md`

## 使用方法

### 技能调用方式

1. **隐式触发**: Claude 根据文件路径和内容上下文自动检测使用哪个技能。

2. **显式触发**: 用户明确请求技能：
   - "为用户认证功能创建规格文档"
   - "将登录功能分解为任务"
   - "记录这个 API 变更"
   - "生成符合规范的提交信息"

3. **工作流触发**: 技能可以链式调用：
   ```
   spec-generator → plan-creator → task-breakdown
   git-manager → code-review → change-manager
   ```

### 技能选择指南

| 用户意图 | 技能 | 模板 |
|----------|------|------|
| Git操作和提交信息 | git-manager | template-commit.md |
| 代码质量审查 | code-review | - |
| 定义要构建什么 | spec-generator | spec-template.md |
| 规划如何构建 | plan-creator | plan-template.md |
| 分解为任务 | task-breakdown | tasks-template.md |
| 跟踪范围变更 | change-manager | change-template.md |
| 记录 bug 修复 | debug-diagnostic | debug-template.md |

## 与模板的集成

每个技能与 `speckit/templates/` 中的特定模板集成：

| 技能 | 主要模板 | 相关模板 |
|------|----------|----------|
| git-manager | template-commit.md | change-template.md |
| code-review | template-review.md | constitution.md |
| spec-generator | spec-template.md | constitution.md |
| plan-creator | plan-template.md | spec-template.md, constitution.md |
| task-breakdown | tasks-template.md | plan-template.md, spec-template.md |
| change-manager | change-template.md | spec-template.md, constitution.md |
| debug-diagnostic | debug-template.md | change-template.md, constitution.md |

### 路径约定

技能遵循一致的输出路径：

```
speckit/specs/
├── features/[name]/
│   ├── spec.md          # spec-generator 输出
│   ├── plan.md          # plan-creator 输出
│   └── tasks.md         # task-breakdown 输出
├── changes/[id]/
│   └── change.md        # change-manager 输出
└── debug/[id]/
    └── debug.md         # debug-diagnostic 输出
```

## 质量门禁

每个技能强制执行 `speckit/constitution.md` 中的质量标准：

1. **代码质量**: DRY、SOLID、设计模式
2. **文档**: 中文语言、时间戳、一致术语
3. **错误处理**: 结构化错误、上下文日志
4. **禁止模式**: Magic numbers、TODO 注释、硬编码值

## 最佳实践

### 1. 从规格开始
在实施规划之前，始终先创建功能规格。

### 2. 增量开发
将工作分解为可独立完成和测试的小任务。

### 3. 记录变更
使用变更提案跟踪任何修改，避免范围蔓延。

### 4. 从调试中学习
每次调试会话应产生经验教训，防止类似问题再次发生。

### 5. Git 工作流
- 使用 Conventional Commits 格式
- 小而频繁的提交
- 通过 Pull Request 进行审查

## 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 1.3.0 | 2026-01-12 | 优化配置架构：Roo Code 改为引用方式，添加 02-agents.md |
| 1.2.0 | 2026-01-12 | 添加 code-review 技能，完善 Roo Code AGENTS.md 配置 |
| 1.1.0 | 2026-01-12 | 添加 git-manager 技能，集成 Roo Code 和 Cursor |
| 1.0.0 | 2026-01-12 | 基于 Speckit 模板的初始技能系统 |

## 配置架构说明

### 引用 vs 复制

本项目采用**引用方式**而非复制方式来配置 Agent Skills：

- **Roo Code**: 使用 `.roo/rules/02-agents.md` 引用项目根目录的 `AGENTS.md`
  - 好处：当 `AGENTS.md` 更新时，所有引用它的地方自动生效
  - 配置：`.roo/rules/02-agents.md` → `AGENTS.md`

- **Cursor**: 内联集成在 `.cursor/rules/constitution.mdc` 中
  - 好处：无需额外文件，技能列表直接可见
  - 配置：`.cursor/rules/constitution.mdc`（内联 Skills 列表）

### 统一配置

所有技能的**源定义**集中在 `speckit/skills/` 目录：
- 每个技能有独立的 `SKILL.md` 文件
- 包含完整的工具使用说明和工作流程
- Roo Code 和 Cursor 都引用这些源定义

## 相关文件

- `speckit/constitution.md` - 核心原则和规则
- `speckit/templates/INDEX.md` - 模板选择指南
- `AGENTS.md` - **跨工具 Agent Skills 配置**（源配置）
- `.roo/rules/02-agents.md` - Roo Code 引用配置
- `.cursor/rules/constitution.mdc` - Cursor 内联配置
- `.trae/rules/` - Trae IDE 规则目录
- `CLAUDE.md` - 项目的 Claude Code 说明

## 许可证

本技能系统是 Speckit 项目的一部分，遵循相同的许可证。

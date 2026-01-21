# 关键路径规范

## 概述

项目遵循完整生命周期管理路径：`spec` → `plan` → `task` → `change` → `review` → `commit` → `debug`

本规范定义了每个阶段的输入/输出工件位置、触发条件以及状态标记，确保开发流程可追溯、可管理。

## 目录结构

```
speckit/
├── specs/
│   ├── features/           # 功能规格文档
│   │   └── <feature-name>/ # 每个功能的独立目录
│   │       ├── spec.md     # 功能规格（唯一入口，包含 ## API 章节）
│   │       ├── plan.md     # 项目计划（生成，架构/策略/决策）
│   │       ├── tasks.md    # 任务分解（生成，任务清单+验收标准）
│   │       └── *.md        # 其他相关文档（可选）
│   ├── changes/            # 变更记录
│   │   └── YYYY-MM/        # 按月份归档
│   │       ├── YYYY-MM-DD-<description>.md
│   │       └── YYYY-MM-DD-<description>-review.md
│   ├── debug/              # 调试报告
│   │   └── YYYY-MM/        # 按月份归档
│   │       └── YYYY-MM-DD-<description>-debug.md
│   └── PATHWAY.md          # 本文档
├── templates/              # 模板文件
│   ├── spec-template.md    # 规格模板（含 ## API 章节要求）
│   ├── plan-template.md    # 计划模板
│   ├── tasks-template.md   # 任务模板
│   ├── change-template.md  # 变更模板
│   └── debug-template.md   # 调试模板
└── skills/                 # Agent Skills
```

## 关键路径详细规范

### 1. spec-generator → 功能规格

**描述**：根据用户需求生成功能规格文档。

**输入**：
- 用户需求（自然语言描述）
- 需求来源（Issue、PRD、会议纪要等）

**输出工件**：
- **落盘位置**：`speckit/specs/features/<feature-name>/`
- **文件**：`spec.md` - 功能规格（唯一入口，**必须包含 ## API 章节**）

**触发条件**：
- 用户主动请求生成规格：`"请为 [功能名称] 生成规格"`
- 检测到新需求（Issue 创建、PRD 更新）

**前置依赖**：
- 无（流程起点）

**下一步**：`plan-creator`

**状态标记**：
- `DRAFT` - 草稿，待评审
- `REVIEWED` - 已评审，待计划

---

### 2. plan-creator → 项目规划

**描述**：根据功能规格生成项目实施计划。

**输入**：
- 功能规格文档路径：`speckit/specs/features/<feature-name>/spec.md`

**输出工件**：
- **落盘位置**：`speckit/specs/features/<feature-name>/`
- **文件**：`plan.md` - 项目计划

**触发条件**：
- 规格文档通过评审（用户确认）
- 检测到 `speckit/specs/features/<feature-name>/spec.md` 状态为 `REVIEWED`
- 用户主动请求生成计划：`"为 [功能名称] 创建计划"`

**前置依赖**：
- spec-generator 阶段完成
- 规格文档状态为 `REVIEWED`

**下一步**：`task-breakdown`

**状态标记**：
- `DRAFT` - 草稿，待评审
- `APPROVED` - 已批准，实施中

---

### 3. task-breakdown → 任务分解

**描述**：将项目计划分解为可执行的具体任务。

**输入**：
- 项目计划文档路径：`speckit/specs/features/<feature-name>/plan.md`

**输出工件**：
- **落盘位置**：`speckit/specs/features/<feature-name>/`
- **文件**：`tasks.md` - 任务分解

**触发条件**：
- 计划文档通过评审
- 检测到 `speckit/specs/features/<feature-name>/plan.md` 状态为 `APPROVED`
- 用户主动请求任务分解：`"为 [功能名称] 分解任务"`

**前置依赖**：
- plan-creator 阶段完成
- 计划文档状态为 `APPROVED`

**下一步**：`change-manager`

**状态标记**：
- `PENDING` - 待实施
- `IN_PROGRESS` - 实施中
- `COMPLETED` - 已完成

---

### 4. change-manager → 变更记录

**描述**：记录和管理代码变更，确保变更可追溯。

**输入**：
- 任务分解文档路径：`speckit/specs/features/<feature-name>/tasks.md`
- Git 变更（可选）：`git diff` 输出
- 变更描述：自然语言描述

**输出工件**：
- **落盘位置**：`speckit/specs/changes/YYYY-MM/`
- **文件**：`YYYY-MM-DD-<description>.md`
- **模板参考**：`speckit/templates/change-template.md`

**触发条件**：
- 任务开始实施前
- Git commit 钩子触发（可选）
- 用户主动记录变更：`"记录变更：[描述]"`
- 检测到未记录的代码变更

**前置依赖**：
- task-breakdown 阶段完成
- 任务状态为 `IN_PROGRESS`

**下一步**：`code-review`

**状态标记**：
- `DRAFT` - 待审查
- `REVIEWED` - 已审查
- `MERGED` - 已合并

---

### 5. code-review → 代码审查

**描述**：自动化代码质量检查和审查。

**输入**：
- 变更记录路径：`speckit/specs/changes/YYYY-MM/YYYY-MM-DD-<description>.md`
- 代码 diff：`git diff` 或 PR/MR diff
- 审查标准：项目编码规范、设计模式

**输出工件**：
- **落盘位置**：`speckit/specs/changes/YYYY-MM/`
- **文件**：`YYYY-MM-DD-<description>-review.md` - 审查报告
- **审查结果**：通过/需修改/阻塞

**触发条件**：
- 变更记录已创建
- PR/MR 创建时
- 用户主动请求审查：`"审查 [变更ID]"`

**前置依赖**：
- change-manager 阶段完成
- 变更记录状态为 `DRAFT`

**下一步**：`git-manager`

**状态标记**：
- `PENDING` - 待审查
- `APPROVED` - 已通过
- `CHANGES_REQUESTED` - 需修改
- `BLOCKED` - 阻塞

---

### 6. git-manager → 提交管理

**描述**：生成规范的 Git 提交信息，管理版本控制。

**输入**：
- 代码审查结果：`speckit/specs/changes/YYYY-MM/YYYY-MM-DD-<description>-review.md`
- 代码 diff：`git diff --cached`
- 变更记录：`speckit/specs/changes/YYYY-MM/YYYY-MM-DD-<description>.md`

**输出工件**：
- **Git commit**：规范化提交信息
- **模板文件**：`speckit/templates/template-commit.md`
- **落盘位置**：`speckit/specs/changes/YYYY-MM/`（提交记录归档）

**触发条件**：
- 代码审查通过（状态为 `APPROVED`）
- 用户主动请求提交：`"生成提交信息"` 或 `/generate-commit`

**前置依赖**：
- code-review 阶段完成
- 审查结果状态为 `APPROVED`

**下一步**：`debug-diagnostic` 或下一轮开发

**状态标记**：
- `PENDING` - 待提交
- `COMMITTED` - 已提交
- `PUSHED` - 已推送

---

### 7. debug-diagnostic → 调试诊断

**描述**：分析错误日志、定位问题根因、提供修复方案。

**输入**：
- 错误日志：控制台输出、文件日志
- 堆栈跟踪：Stack trace
- 复现步骤：问题描述
- 相关代码：出错的源文件

**输出工件**：
- **落盘位置**：`speckit/specs/debug/YYYY-MM/`
- **文件**：`YYYY-MM-DD-<description>-debug.md`
- **模板参考**：`speckit/templates/debug-template.md`
- **修复建议**：具体的修复方案

**触发条件**：
- 检测到错误：异常抛出、测试失败
- 用户主动请求调试：`"调试 [错误描述]"` 或 `/debug`
- 自动化监控触发告警

**前置依赖**：
- 无（可从任意阶段触发）

**下一步**：
- 问题在规格阶段：回到 `spec-generator`
- 问题在实现阶段：回到 `change-manager`
- 问题已修复：回到 `git-manager`

**状态标记**：
- `IDENTIFIED` - 问题已识别
- `IN_PROGRESS` - 修复中
- `RESOLVED` - 已解决
- `WONT_FIX` - 不修复

---

## 状态标记总览

| 阶段 | 状态 | 说明 |
|------|------|------|
| spec-generator | `DRAFT` | 草稿，待评审 |
| spec-generator | `REVIEWED` | 已评审，待计划 |
| plan-creator | `DRAFT` | 草稿，待评审 |
| plan-creator | `APPROVED` | 已批准，实施中 |
| task-breakdown | `PENDING` | 待实施 |
| task-breakdown | `IN_PROGRESS` | 实施中 |
| task-breakdown | `COMPLETED` | 已完成 |
| change-manager | `DRAFT` | 待审查 |
| change-manager | `REVIEWED` | 已审查 |
| change-manager | `MERGED` | 已合并 |
| code-review | `PENDING` | 待审查 |
| code-review | `APPROVED` | 已通过 |
| code-review | `CHANGES_REQUESTED` | 需修改 |
| code-review | `BLOCKED` | 阻塞 |
| git-manager | `PENDING` | 待提交 |
| git-manager | `COMMITTED` | 已提交 |
| git-manager | `PUSHED` | 已推送 |
| debug-diagnostic | `IDENTIFIED` | 问题已识别 |
| debug-diagnostic | `IN_PROGRESS` | 修复中 |
| debug-diagnostic | `RESOLVED` | 已解决 |
| debug-diagnostic | `WONT_FIX` | 不修复 |

---

## 触发条件总览

| 当前阶段 | 触发条件 | 下一步 |
|----------|----------|--------|
| spec-generator | 用户请求或检测到新需求 | plan-creator |
| plan-creator | 规格文档状态为 `REVIEWED` | task-breakdown |
| task-breakdown | 计划文档状态为 `APPROVED` | change-manager |
| change-manager | 任务开始实施或用户请求 | code-review |
| code-review | 变更记录已创建或 PR/MR | git-manager |
| git-manager | 审查结果状态为 `APPROVED` | debug-diagnostic 或下一轮 |
| debug-diagnostic | 检测到错误或用户请求 | spec-generator 或 change-manager |

---

## 示例工作流

### 示例 1：完整功能开发流程

```markdown
1. 用户: "请为用户认证功能生成规格"
   → spec-generator
   →落盘: speckit/specs/features/auth/spec.md（包含 ## API 章节）
   →状态: DRAFT

2. 用户评审后标记: "REVIEWED"
   → plan-creator
   →落盘: speckit/specs/features/auth/plan.md
   →状态: DRAFT

3. 用户评审后标记: "APPROVED"
   → task-breakdown
   →落盘: speckit/specs/features/auth/tasks.md
   →状态: PENDING

4. 开始实施第一个任务
   → change-manager
   →落盘: speckit/specs/changes/2026-01/2026-01-12-auth-login.md
   →状态: DRAFT

5. 提交代码审查
   → code-review
   →落盘: speckit/specs/changes/2026-01/2026-01-12-auth-login-review.md
   →状态: APPROVED

6. 生成提交信息并提交
   → git-manager
   → Git commit
   →状态: COMMITTED
```

### 示例 2：调试流程

```markdown
1. 检测到错误: "Authentication failed"
   → debug-diagnostic
   →分析日志和堆栈跟踪
   →落盘: speckit/specs/debug/2026-01/2026-01-12-auth-failed-debug.md
   →状态: IDENTIFIED

2. 确认为实现问题
   → change-manager
   →落盘: speckit/specs/changes/2026-01/2026-01-12-auth-fix.md
   →继续后续流程
```

---

## 最佳实践

1. **状态同步**：每次状态变更后更新相关文档的元数据
2. **路径规范**：严格遵循落盘路径规范，确保工件可追溯
3. **触发检查**：进入下一阶段前，验证触发条件是否满足
4. **文档归档**：每月整理并归档完成的工件
5. **版本控制**：所有工件纳入 Git 版本控制

---

## YAML Frontmatter 状态规范

为实现机器可读性，所有工件文档应使用 YAML frontmatter 存储状态信息：

```yaml
---
id: [唯一标识符]
status: [状态值]
created: [YYYY-MM-DD]
updated: [YYYY-MM-DD]
links: [相关文档路径列表]
---
```

### 各阶段状态值

| 文档类型 | 状态值 |
|----------|--------|
| `spec.md` | `DRAFT` | `REVIEWED` | `APPROVED` |
| `plan.md` | `DRAFT` | `APPROVED` |
| `tasks.md` | `PENDING` | `IN_PROGRESS` | `COMPLETED` |
| `change/*.md` | `DRAFT` | `REVIEWED` | `MERGED` |
| `*.md-review.md` | `PENDING` | `APPROVED` | `CHANGES_REQUESTED` | `BLOCKED` |
| `*-debug.md` | `IDENTIFIED` | `IN_PROGRESS` | `RESOLVED` | `WONT_FIX` |

---

## 修订历史

| 版本 | 日期 | 描述 |
|------|------|------|
| 1.0 | 2026-01-12 | 初始版本 |
| 1.1 | 2026-01-12 | 统一规范：README.md + API.md → spec.md（含 ## API 章节）；添加 YAML frontmatter 状态规范 |

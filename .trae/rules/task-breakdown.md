---
description: task, breakdown, subtask, estimation, sprint, work breakdown
---
---
name: task-breakdown
description: "task, breakdown, subtask, estimation, sprint, work breakdown"
allowed-tools:
  - read_file
  - write_file
  - new_task
---

# 任务分解器

## 用途

本技能将实施方案分解为粒度化、可操作的任务，可以由开发人员执行。它定义任务优先级、识别并行化机会，并将任务映射到用户故事。

## 使用时机

在以下情况下调用此技能：
- 实施方案已完成并批准
- 需要将工作分解为可管理单元
- 需要准备任务分配
- 正在进行冲刺或迭代规划

## 不使用时机

- 实施方案不存在时（先使用 plan-creator）
- 任务已经定义良好时
- 只需要高层里程碑时

## 输入要求

### 最小输入

- 实施方案文件路径（`speckit/specs/features/[name]/plan.md`）
- 功能规格文件路径（`speckit/specs/features/[name]/spec.md`）
- 功能名称

### 理想输入

- 包含技术上下文的完整实施方案
- 带有优先级的用户故事
- 项目结构决策
- 
- 时间线团队容量和专业能力约束

## 输出规范

### 文件路径

```
speckit/specs/features/[kebab-case-功能名称]/tasks.md
```

## 工作流程

### 第一步：分析输入

1. 阅读并理解实施方案
2. 审查功能规格和用户故事
3. 识别技术依赖
4. 将任务映射到用户故事

### 第二步：定义任务结构

1. 创建阶段结构（设置 → 基础 → 故事 → 完善）
2. 识别阻塞依赖
3. 定义任务边界

### 第三步：分配优先级

1. 将阻塞任务标记为 P0
2. 将核心功能标记为 P1
3. 将重要功能标记为 P2
4. 将可选功能标记为 P3

### 第四步：识别并行化

1. 将无依赖任务标记为 [P]
2. 分组独立任务
3. 识别跨故事依赖

### 第五步：生成文档

1. 应用 tasks-template.md 结构
2. 全程使用中文
3. 添加时间戳和元数据
4. 保存到正确路径

## 质量检查清单

在最终确定任务分解之前：

- [ ] 所有用户故事都有对应的任务
- [ ] 阻塞任务标记为 P0
- [ ] 独立任务标记为 [P]
- [ ] 任务依赖关系清晰
- [ ] 测试任务先于实施任务
- [ ] 文档任务已包含
- [ ] 文档全程使用中文
- [ ] 时间戳格式为 YYYY-MM-DD
- [ ] 文件路径符合约定

## 目录说明管理

**创建文件后必须更新00-目录说明.md**：

1. **新建目录时**
   - 在目录根创建 `00-目录说明.md` 文件
   - 记录目录用途和文件列表

2. **新建文件时**
   - 更新所在目录的 `00-目录说明.md`
   - 添加文件记录：文件名、用途、创建日期

3. **更新文件时**
   - 如文件用途变更，更新 `00-目录说明.md` 中的说明
   - 更新时间戳

4. **删除文件时**
   - 从 `00-目录说明.md` 中移除记录
   - 添加删除日志

## 任务收尾管理

### 完成任务后必须更新 tasks.md

**适用场景**：完成任何代码开发、文档编写、配置修改等任务后。

**操作步骤**：
1. 在对应任务的 tasks.md 文件中，找到该任务的条目
2. 在"相关文件"部分标注所有新增/修改的文件：
   - Python 文件（.py）
   - Markdown 文件（.md）
   - 配置文件（.toml, .json, .yaml, .yml 等）
   - 其他重要文件

3. 文件路径格式：
   - 使用相对路径，如 `src/domain/entities/user.py`
   - 使用 Markdown 链接格式：`[文件名](路径)`

**示例**：
```markdown
| 文件 | 用途 |
|------|------|
| [user.py](src/domain/entities/user.py) | 用户实体模型 |
| [user_service.py](src/application/services/user_service.py) | 用户业务逻辑 |
```

**验收标准**：
- ✅ tasks.md 中的文件路径正确
- ✅ 文件路径与实际位置一致
- ✅ 涵盖所有新增和修改的重要文件

## 相关模板

- `speckit/templates/tasks-template.md` - 此技能生成的模板
- `speckit/templates/plan-template.md` - 输入计划
- `speckit/templates/spec-template.md` - 输入规格
- `speckit/constitution.md` - 需要遵循的质量标准

## 版本

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0.0 | 2026-01-12 | 初始版本 |
| 1.1.0 | 2026-01-12 | 添加统一的 allowed-tools 格式 |


# Speckit 模板索引与选择指南

## 模板清单

| 模板 | 文件 | 用途 | 输出位置 |
|------|------|------|----------|
| 章程 | `constitution.md` | 核心原则与规则 | speckit/ |
| 规格 | `spec-template.md` | 功能需求说明 | `speckit/specs/features/[name]/spec.md` |
| 计划 | `plan-template.md` | 技术实施方案 | `speckit/specs/features/[name]/plan.md` |
| 任务 | `tasks-template.md` | 任务分解与排期 | `speckit/specs/features/[name]/tasks.md` |
| 变更 | `change-template.md` | 变更提案管理 | `speckit/specs/changes/[id]/change.md` |
| 调试 | `debug-template.md` | 问题诊断与修复 | `speckit/specs/debug/[id]/debug.md` |

## 场景匹配

### 原型探索

**特征**: 快速验证概念、范围不确定、可能废弃

**推荐模板**:
- `spec-template.md`（简化版）
- `tasks-template.md`（最小任务列表）

**建议**:
- 跳过 `plan-template.md` 详细技术上下文
- 不创建 `data-model.md`、`quickstart.md`
- 任务数量控制在 5 个以内

### 功能开发

**特征**: 需求明确、长期维护、多迭代

**推荐模板**:
- `spec-template.md` → `plan-template.md` → `tasks-template.md`
- 完整流程

**建议**:
- 完整记录用户场景与验收标准
- 任务优先级明确（P0/P1/P2/P3）
- 包含横切关注点（文档、安全、性能）

### 缺陷修复

**特征**: 问题定位、影响范围有限、快速交付

**推荐模板**:
- `debug-template.md`
- `change-template.md`（如行为变更）

**建议**:
- 跳过 `spec-template.md`、`plan-template.md`
- 详细记录复现步骤与根因分析
- 验证修复不影响其他功能

### 重大重构

**特征**: 架构调整、API 变更、涉及多个模块

**推荐模板**:
- `plan-template.md`（迁移计划）
- `change-template.md`（API 变更）
- `debug-template.md`（兼容性测试）

**建议**:
- 跳过 `spec-template.md` 详细用户故事
- 包含迁移方案与兼容性策略
- 分阶段实施，避免大爆炸式发布

## 模板裁剪规则

### 可省略章节

| 场景 | 可省略内容 |
|------|-----------|
| 原型 | 边界与异常情况、成功标准 |
| 小型项目 | 阶段 N 完善、并行任务标记 |
| 快速修复 | 技术上下文、项目结构规划 |

### 必填内容

| 模板 | 必填章节 |
|------|----------|
| spec-template.md | 用户场景与测试、需求 |
| plan-template.md | 摘要、技术上下文、项目结构 |
| tasks-template.md | 阶段 1/2、至少一个用户故事 |
| change-template.md | 变更原因、Delta 操作、影响范围 |

## 路径约定

所有规格文档统一存储在 `speckit/specs/` 目录下：

```
speckit/specs/
├── features/[feature-name]/
│   ├── spec.md          # 规格说明
│   ├── plan.md          # 实施计划
│   ├── data-model.md    # 数据模型（可选）
│   ├── quickstart.md    # 快速开始（可选）
│   ├── contracts/       # 契约定义（可选）
│   └── tasks.md         # 任务分解
├── changes/[change-id]/
│   ├── change.md        # 变更提案
│   └── migration/       # 迁移脚本（可选）
└── debug/[issue-id]/
    └── debug.md         # 调试指南
```

## 版本兼容

| 模板版本 | 最低依赖 |
|---------|----------|
| constitution.md 1.3.0 | - |
| spec-template.md 1.0.0 | constitution.md 1.3.0 |
| Project-template.md 1.0.0 | constitution.md 1.3.0 |
| tasks-template.md 1.1.0 | constitution.md 1.3.0 |
| change-template.md 1.0.0 | constitution.md 1.3.0 |
| debug-template.md 1.0.0 | constitution.md 1.3.0 |

## 快速参考

### 新功能开发流程

1. 使用 `spec-template.md` 创建功能规格
2. 使用 `Project-template.md` 制定实施方案
3. 使用 `tasks-template.md` 分解任务
4. 按任务实施，标记完成
5. 更新相关文档

### 变更提交流程

1. 使用 `change-template.md` 创建变更提案
2. 获得人工批准
3. 更新相关规格与代码
4. 运行验证
5. 归档变更

### 问题修复流程

1. 使用 `debug-template.md` 记录问题
2. 分析根因
3. 提出解决方案
4. 实施修复
5. 总结经验教训

## 注意事项

- 保持模板结构一致，便于 AI 理解
- 遵循路径约定，避免文档散落
- 根据项目规模选择模板深度
- 避免过度工程化，保持实用主义

---

**最后更新**: 2026-01-12

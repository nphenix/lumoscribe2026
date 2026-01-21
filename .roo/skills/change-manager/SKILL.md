---
name: change-manager
description: "change, modify, update, version, history, change log"
allowed-tools:
  - read_file
  - write_file
  - list_files
---

# 变更管理器

## 用途

本技能管理功能规格、实施方案和代码库的变更。它提供范围修改、结构化跟踪、影响分析和实施规划。

## 使用时机

在以下情况下调用此技能：
- 功能范围需要在初始规格后更改
- 需要 API 或接口修改
- 需要解决技术债务
- 需要添加、移除或修改需求
- 需要实施后改进

## 不使用时机

- 创建初始功能规格时（使用 spec-generator）
- 创建初始实施方案时（使用 plan-creator）
- 分解任务时（使用 task-breakdown）
- 记录 bug 时（使用 debug-diagnostic）

## 输入要求

### 最小输入

- 变更类型（ADDED、MODIFIED、REMOVED、RENAMED）
- 受影响的功能或模块
- 变更描述

### 理想输入

- 关联的功能规格或计划
- 变更原因
- 影响范围（破坏性或非破坏性）
- 实施方案
- 审批状态

## 输出规范

### 文件路径

```
speckit/specs/changes/[kebab-case-变更-id]/change.md
```

## 工作流程

### 第一步：分类变更

1. 确定变更类型（ADDED、MODIFIED、REMOVED、RENAMED）
2. 评估影响（破坏性或非破坏性）
3. 识别受影响的组件

### 第二步：记录 Delta

1. 列出所有新增内容
2. 记录所有修改内容
3. 记录所有移除内容
4. 注意所有重命名内容

### 第三步：分析影响

1. 识别受影响的规格
2. 映射受影响的代码区域
3. 识别依赖组件
4. 评估向后兼容性

### 第四步：规划实施

1. 定义实施阶段
2. 识别先决条件
3. 规划验证步骤

### 第五步：生成文档

1. 应用 change-template.md 结构
2. 全程使用中文
3. 添加时间戳和元数据
4. 保存到正确路径

## 质量检查清单

在最终确定变更提案之前：

- [ ] 变更原因已清晰说明
- [ ] 所有 delta 已记录（ADDED/MODIFIED/REMOVED/RENAMED）
- [ ] 破坏性变更已标记
- [ ] 已识别受影响的规格
- [ ] 已映射受影响的代码区域
- [ ] **Prompt 管理**: 无硬编码，已注册到 Registry
- [ ] 实施方案可操作
- [ ] 审批工作流已定义
- [ ] 文档全程使用中文
- [ ] 时间戳格式为 YYYY-MM-DD
- [ ] 文件路径符合约定

## 相关模板

- `speckit/templates/change-template.md` - 此技能生成的模板
- `speckit/templates/spec-template.md` - 源规格
- `speckit/constitution.md` - 需要遵循的质量标准
- `debug-diagnostic` - 变更相关 bug 的相关技能

## 版本

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0.0 | 2026-01-12 | 初始版本 |
| 1.1.0 | 2026-01-12 | 添加统一的 allowed-tools 格式 |

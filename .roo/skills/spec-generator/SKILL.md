---
name: spec-generator
description: "spec, API, documentation, interface, design document, specification"
allowed-tools:
  - read_file
  - write_file
---

# 规格生成器

## 用途

本技能将用户需求转换为结构化的功能规格，可用于实施规划和任务分解。

## 使用时机

在以下情况下调用此技能：
- 用户希望记录新功能或能力
- 需求需要在实施前正式化
- 用户故事或功能请求需要详细规格
- 创建范围讨论和批准的文档

## 不使用时机

- 需求已经以 spec-template 格式文档化时
- 只需要高层概述时（使用对话即可）
- 功能已处于实施阶段时

## 输入要求

### 最小输入

- 功能名称或描述
- 高层目标或用户需求
- 目标用户或利益相关者

### 理想输入

- 产品负责人的用户故事
- 用户研究或反馈
- 现有系统上下文
- 约束或依赖
- 成功指标或 KPI

## 输出规范

### 文件路径

```
speckit/specs/features/[kebab-case-功能名称]/spec.md
```

## 工作流程

### 第一步：理解需求

1. 如果输入不足，询问澄清问题
2. 识别要解决的核心问题
3. 确定目标用户及其目标
4. 记录现有约束

### 第二步：识别用户故事

1. 提取用户角色和目标
2. 为故事排序（P1 为 MVP，P2 为重要，P3 为可选）
3. 定义清晰的验收场景
4. 确保每个故事可独立测试

### 第三步：定义边界

1. 识别边缘情况和错误条件
2. 记录数据验证规则
3. 映射业务规则和约束

### 第四步：正式化需求

1. 将故事转换为功能需求
2. 识别关键实体和关系
3. 定义成功标准

### 第五步：生成文档

1. 应用 spec-template.md 结构
2. 全程使用中文
3. 添加时间戳和元数据
4. 保存到正确路径

## LLM 集成检查

### 检查条件

生成规格后，检查功能是否需要调用 LLM（满足任一条件即触发）：
- 用户故事提到"AI"、"LLM"、"生成"、"分析"
- 验收场景涉及"文本处理"、"内容创作"
- 需求描述包含"智能"、"自动化"
- 技术方案涉及 LLM Provider 或 LLM Model

### 触发提示

如果检测到需要 LLM 集成，在输出末尾添加提示：

```
⚠️ 此功能需要 LLM 集成，请遵循提示词管理规范：

1. 创建模块 prompts.py
   路径: src/application/services/[module]/prompts.py
   模板: 参考 src/application/services/document_cleaning/prompts.py

2. 定义提示词
   - SCOPE_[MODULE]_[ACTION] = "module:action"
   - PROMPTS 字典包含 content、description、format

3. 注册到注册表
   文件: src/shared/constants/prompts.py
   格式: from src.application.services.[module].prompts import PROMPTS

4. 注入使用
   - Service __init__ 注入 PromptService
   - 调用时使用 prompt_service.get_active_prompt(SCOPE_xxx)

参考: speckit/specs/prompt-management-standards.md
```

## 质量检查清单

在最终确定规格之前：

- [ ] 功能有清晰、唯一的名称
- [ ] 用户故事按优先级排序（P1/P2/P3）
- [ ] 每个故事有 2-5 个验收场景
- [ ] 场景使用"假设-当-则"格式
- [ ] 已记录边界条件
- [ ] 功能需求可追溯到故事
- [ ] 已定义关键实体及其属性
- [ ] 文档全程使用中文
- [ ] 时间戳格式为 YYYY-MM-DD
- [ ] 文件路径符合约定

## 相关模板

- `speckit/templates/spec-template.md` - 此技能生成的模板
- `speckit/constitution.md` - 需要遵循的质量标准
- `plan-creator` - 工作流中的下一个技能
- `speckit/templates/tasks-template.md` - 任务分解技能的输入

## 版本

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0.0 | 2026-01-12 | 初始版本 |
| 1.1.0 | 2026-01-12 | 添加统一的 allowed-tools 格式 |
| 1.2.0 | 2026-01-17 | 添加 LLM 集成检查主动提示 |

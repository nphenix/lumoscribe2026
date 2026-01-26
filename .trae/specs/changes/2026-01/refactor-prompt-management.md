# 变更记录：提示词管理重构 (Code-First Seed 模式)

> 本文档记录将硬编码提示词重构为 Code-First Seed 模式的变更细节。

## 背景

在 `ai-doc-platform-phase1` 开发过程中，发现 T031 (文档清洗) 和 T032 (图表提取) 的提示词硬编码在 Service 代码中。导致无法热更新、无法 A/B 测试，且前端提示词管理功能无法管控这些核心业务逻辑。

## 变更内容

### 1. 架构调整

采用 **Code-First Seed & DB-Managed** 模式：
- **Source of Truth**: 运行时以数据库为准。
- **Code Seed**: 代码库中保留默认提示词作为“出厂设置” (Seed)。
- **Sync**: 系统启动/初始化时，将 Seed 写入数据库（如果不存在）。

### 2. 代码变更

#### Backend
- **新增**: `src/shared/constants/prompts.py` - 定义 `SCOPE_*` 常量和 `DEFAULT_PROMPTS` 字典。
- **修改**: `scripts/init-db.py` - 增加 `seed_prompts` 步骤，在初始化数据库时播种提示词。
- **重构**: `src/application/services/document_cleaning_service.py`
    - 移除私有方法中的硬编码字符串。
    - 注入 `PromptService`。
    - 优先从 DB 获取激活的 Prompt，获取失败回退到 Seed。
    - 使用 `.replace()` 进行简单模板渲染，避免 JSON 大括号转义问题。

#### Frontend
- **修复**: `src/interfaces/admin-web/hooks/use-prompts.ts` - 修复 Prompt 接口定义，适配后端 `scope`/`active` 字段。
- **优化**: `src/interfaces/admin-web/app/prompts/page.tsx`
    - 适配新字段显示。
    - 修复编辑功能（状态管理、数据预填充）。
    - 增加 Chat 格式的简单支持。

### 3. 数据迁移

本次变更是开发阶段的重构，直接通过 `init-db.py` 播种数据。

## 验证

- [x] **集成测试**: 创建临时测试 `tests/test_prompt_integration.py`，验证 API 创建新版本 Prompt 后，Service 能立即使用新版本。
- [x] **功能验证**: 前端页面能正常显示、新建、编辑提示词。

## 影响范围

- `DocumentCleaningService`
- `ChartExtractionService`
- `PromptService`
- Admin Web 提示词管理模块

## 后续规划

- 制定《提示词管理规范》，约束后续开发。
- 完善前端对 Chat 格式 (JSON Messages) 的可视化编辑支持。

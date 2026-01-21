# 实施计划：提示词管理重构

> 基于 spec.md 的实施计划。

## 1. 任务分解

### 阶段 1：架构调整
- [x] 创建 `src/application/services/document_cleaning/prompts.py`
- [x] 改造 `src/shared/constants/prompts.py` 为注册表模式
- [x] 更新 `docs/guides/prompt-management-standards.md` 规范文档

### 阶段 2：代码迁移
- [x] 将 T031/T032 提示词迁移至新文件
- [x] 验证 `document_cleaning_service.py` 引用正常（通过 Registry 间接引用或直接引用 Scope 常量）

### 阶段 3：验证
- [x] 运行集成测试确保数据库播种正常
- [x] 验证前端管理功能

## 2. 风险控制
- 确保 `init-db.py` 幂等性，避免重复插入或覆盖用户已修改的活跃版本。

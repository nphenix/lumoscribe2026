# 功能规格：提示词管理重构

> 本文档定义了将提示词管理从硬编码模式迁移到 Code-First Seed 模式的架构设计。

## 1. 背景与目标

### 问题
- 提示词硬编码在 Service 中，无法热更新。
- 前端管理界面无法管控核心业务提示词。
- `shared/constants/prompts.py` 单一大文件存在膨胀和冲突风险。

### 目标
- 实现提示词的“代码定义种子，数据库托管运行”模式。
- 建立分散式提示词定义与集中式注册机制，避免单点膨胀。
- 完善开发规范，确保可维护性。

## 2. 架构设计

### 2.1 数据流
```mermaid
graph TD
    A[业务模块 (src/application/services/xxx/prompts.py)] -->|定义| B[注册表 (src/shared/constants/prompts.py)]
    B -->|Sync Script| C[数据库 (Prompts Table)]
    C -->|运行时加载| D[PromptService]
    D -->|提供| E[业务 Service]
```

### 2.2 模块化设计
- **分布式定义**: 每个业务模块维护自己的 `prompts.py`。
- **集中式注册**: `shared/constants/prompts.py` 仅作为 Registry，不存放具体内容。

## 3. 接口变更

无外部 API 变更，仅内部实现重构。

## 4. 迁移策略

1. 提取现有硬编码提示词到各模块 `prompts.py`。
2. 更新 Registry。
3. 运行 `init-db.py` 播种。
4. 验证 Service 调用。

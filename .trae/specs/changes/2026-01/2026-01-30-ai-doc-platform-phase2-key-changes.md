---
template: Change
version: 1.0.0
lastUpdated: 2026-01-30
charter: ../constitution.md
---

# 变更提案：AI 文档平台第二阶段关键变更汇总

**变更ID**: 2026-01-30-ai-doc-platform-phase2-key-changes  
**创建日期**: 2026-01-30  
**状态**: 已实施  
**关联功能**: [ai-doc-platform-phase2/spec.md](../../features/ai-doc-platform-phase2/spec.md)

---

## 变更原因

第二阶段引入了“内容生成链路重构 + 可视化链路收敛 + 中台能力增强”，需要形成可审计的变更记录，明确为什么改、改了什么、影响范围与验证入口。

## 变更内容

- **Html 生成链（内容生成）**：从单体编排演进为“按章生成 → 后处理 → 可选润色”的可观测管线，并通过统一 API 触发生成与落盘。
- **可视化链路（图表渲染）**：统一图表渲染入口与占位符注入规则，保证离线可渲染与锚点稳定插入。
- **中台能力（LLM 配置/Prompt/并发）**：Provider/CallSite/Prompt 管理与并发门控贯通前后端，避免 LLM 调用点绕过统一配置路径。

**重大变更**: 无（以兼容性改造与能力增强为主）

## Delta 操作

### 新增需求 (ADDED)

- **统一白皮书生成入口与落盘机制**：新增/完善 `targets/whitepaper/generate(_stream)` 生成端点与 `target_files` 落盘写库。
- **章节生成管线与后处理器**：新增管线编排、语义去重、空条目清理、润色 validator 等能力。
- **中台 Prompt 管理与差异/回滚**：新增 Prompt 版本列表、差异查看、启用/停用与回滚能力。
- **LLM 并发配置与门控**：新增 Provider/CallSite 并发字段、DB 轻量迁移与运行时门控。

### 修改需求 (MODIFIED)

- **LLM 调用路径**：阶段二新增或改造的 LLM 调用点统一经由 Provider/CallSite/Prompt 配置路径访问，避免业务侧硬编码。
- **生成链路结构**：由“单链路耦合（RAG/正文/图表）”调整为更明确的职责拆分（内容生成、后处理、渲染/注入）。

### 移除需求 (REMOVED)

- 无

### 重命名需求 (RENAMED)

- 无

## 影响范围

**受影响的规格**:

- `speckit/specs/features/ai-doc-platform-phase2/spec.md`
- `speckit/specs/features/ai-doc-platform-phase2/plan.md`
- `speckit/specs/features/ai-doc-platform-phase2/tasks.md`

**受影响的代码**:

- **API 端点**:
  - `POST /v1/targets/whitepaper/generate`
  - `POST /v1/targets/whitepaper/generate_stream`
  - LLM 配置：`/v1/llm/providers`、`/v1/llm/call-sites`、`/v1/llm/capabilities`
  - Prompt 管理：`/v1/prompts`（含 scopes/diff/patch 等能力）
- **关键文件（按第二阶段 tasks.md 事实列出）**:
  - `src/application/services/llm_runtime_service.py`
  - `src/application/repositories/prompt_repository.py`
  - `src/interfaces/api/routes/prompts.py`
  - `src/interfaces/admin-web/app/prompts/page.tsx`
  - `src/domain/entities/llm_provider.py`
  - `src/domain/entities/llm_call_site.py`
  - `src/shared/db.py`
  - `src/shared/config.py`
  - `src/interfaces/api/routes/llm.py`
  - `src/interfaces/admin-web/app/llm/providers/page.tsx`
  - `src/interfaces/admin-web/app/llm/call-sites/page.tsx`
  - `src/application/services/content_generation/pipeline.py`
  - `src/interfaces/api/routes/targets.py`

---

## 实施计划

### 阶段 1：准备

- [x] C001 审查第二阶段规格/计划/任务分解（phase2）
- [x] C002 汇总阶段二关键落地点（Html 生成链、可视化链路、中台能力）
- [x] C003 记录变更原因与影响范围

### 阶段 2：实施

- [x] C004 以“按章生成 → 后处理 → 可选润色”落地生成链路与可观测数据面
- [x] C005 贯通 Prompt 管理（版本列表/差异/回滚）与前端页面
- [x] C006 贯通 LLM 并发配置（Provider/CallSite）与运行时门控

### 阶段 3：验证

- [x] C007 运行阶段二相关测试（详见 `tests/` 与 tasks.md 的阶段四测试项）
- [x] C008 确认关键调用点均走统一配置路径
- [x] C009 归档变更记录到 `speckit/specs/changes/`

---

**版本**: 1.0.0 | **创建**: 2026-01-30 | **最后更新**: 2026-01-30

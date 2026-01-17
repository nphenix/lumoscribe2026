---
id: ai-doc-platform-phase1
status: IN_PROGRESS
created: 2026-01-16
updated: 2026-01-16
links:
  - ./spec.md
  - ./plan.md
---

# 任务：第一阶段 AI 文档生成平台

**输入**: 设计文档  
**先决条件**: plan.md（必需）、spec.md（必需）

---

## 阶段1：设置（共享基础设施）

**目的**: 建立可运行骨架与最小闭环（API 创建任务 → Redis 队列 → Worker 执行 → 状态回写 → API 查询）。

- [x] T001 [P0] 创建符合分层架构的项目结构（`src/`、`tests/`、各目录 `00-目录说明.md`）

  **相关文件**:
  -源码目录: [`src/domain/`](src/domain)、[`src/application/`](src/application)、[`src/interfaces/`](src/interfaces)、[`src/shared/`](src/shared)
  - 测试目录: [`tests/`]
  - 脚本目录: [`scripts/`]
  - 文档目录: [`docs/`](docs/00-目录说明.md)
- [x] T002 [P0] 初始化 Python 项目依赖（`pyproject.toml`），并提供 `env.example`

  **相关文件**:
  - 项目配置: [`pyproject.toml`](pyproject.toml)
  - 环境示例: [`env.example`](.env.example)
  - 依赖锁定: [`uv.lock`](uv.lock)
- [x] T003 [P0] 建立 FastAPI 最小服务（`/v1/health`）与统一错误响应格式

  **相关文件**:
  - 主入口: [`main.py`](src/interfaces/api/main.py)
  - 应用实例: [`app.py`](src/interfaces/api/app.py)
  - 健康检查: [`health.py`](src/interfaces/api/routes/health.py)
  - 错误处理: [`errors.py`](src/shared/errors.py)
  - 依赖注入: [`deps.py`](src/interfaces/api/deps.py)
- [x] T004 [P0] 建立 SQLite 元数据与 `jobs` 表（最小字段）

  **相关文件**:
  - 数据库连接: [`db.py`](src/shared/db.py)
  - 配置管理: [`config.py`](src/shared/config.py)
  - 路由: [`jobs.py`](src/interfaces/api/routes/jobs.py)
  - 数据库脚本: [`init-db.py`](../scripts/init-db.py)、[`verify-db.py`](../scripts/verify-db.py)
- [x] T005 [P0] 建立 Celery + Redis Worker（占位任务）并接入 `/v1/jobs` 创建即入队

  **相关文件**:
  - Celery 应用: [`celery_app.py`](src/interfaces/worker/celery_app.py)
  - Worker 任务: [`tasks.py`](src/interfaces/worker/tasks.py)
  - 路由: [`jobs.py`](src/interfaces/api/routes/jobs.py)
  - 数据库连接: [`db.py`](src/shared/db.py)
- [x] T006 [P1] 规范化安装与运行命令（README：安装依赖、启动 API/Worker/Redis）

  **相关文件**:
  - 主文档: [`README.md`](README.md)
  - 快速开始: [`QUICKSTART.md`](QUICKSTART.md)
  - Agent 配置: [`AGENTS.md`](AGENTS.md)

---

## 阶段2：文档/模板/目标文件管理（US2）

**目标**: 完成 CRUD + 归档 + 中间态观测/删除，形成平台治理能力基础。

- [x] T010 [P1] [US2] 设计并实现 `source_files/templates/target_files/intermediate_artifacts` 数据表

  **相关文件**:
  - Entities: [`source_file.py`](src/domain/entities/source_file.py)、[`template.py`](src/domain/entities/template.py)、[`intermediate_artifact.py`](src/domain/entities/intermediate_artifact.py)、[`target_file.py`](src/domain/entities/target_file.py)
  - 存储: [`storage.py`](src/shared/storage.py)
  - 数据目录: [`data/sources/`](../data/sources)、[`data/templates/`](../data/templates)、[`data/intermediates/`](../data/intermediates)、[`data/targets/`](../data/targets)
  - 数据库脚本: [`init-db.py`](../scripts/init-db.py)、[`verify-db.py`](../scripts/verify-db.py)
- [x] T011 [P1] [US2] 实现源文件上传/查询/更新/删除/归档 API（含文件落盘）

  **相关文件**:
  - Entity: [`source_file.py`](src/domain/entities/source_file.py)
  - Repository: [`source_file_repository.py`](src/application/repositories/source_file_repository.py)
  - Service: [`source_file_service.py`](src/application/services/source_file_service.py)
  - Route: [`sources.py`](src/interfaces/api/routes/sources.py)
  - Schema: [`source_file.py`](src/application/schemas/source_file.py)
  - 目录说明: [`entities/`](src/domain/entities/00-目录说明.md)、[`schemas/`](src/application/schemas/00-目录说明.md)
- [x] T012 [P1] [US2] 实现模板上传/CRUD/锁定/预处理校验 API（预处理不使用 LLM）

  **相关文件**:
  - Entity: [`template.py`](src/domain/entities/template.py)
  - Repository: [`template_repository.py`](src/application/repositories/template_repository.py)
  - Service: [`template_service.py`](src/application/services/template_service.py)
  - Route: [`templates.py`](src/interfaces/api/routes/templates.py)
  - Schema: [`template.py`](src/application/schemas/template.py)
- [x] T013 [P2] [US2] 实现中间态列表/详情/删除 API

  **相关文件**:
  - Entity: [`intermediate_artifact.py`](src/domain/entities/intermediate_artifact.py)
  - Repository: [`intermediate_artifact_repository.py`](src/application/repositories/intermediate_artifact_repository.py)
  - Service: [`intermediate_artifact_service.py`](src/application/services/intermediate_artifact_service.py)
  - Route: [`intermediates.py`](src/interfaces/api/routes/intermediates.py)
  - Schema: [`intermediate_artifact.py`](src/application/schemas/intermediate_artifact.py)
  - 目录说明: [`entities/`](src/domain/entities/00-目录说明.md)、[`schemas/`](src/application/schemas/00-目录说明.md)
- [x] T014 [P2] [US2] 实现目标文件查询/下载 API

  **相关文件**:
  - Entity: [`target_file.py`](src/domain/entities/target_file.py)
  - Repository: [`target_file_repository.py`](src/application/repositories/target_file_repository.py)
  - Service: [`target_file_service.py`](src/application/services/target_file_service.py)
  - Route: [`targets.py`](src/interfaces/api/routes/targets.py)
  - Schema: [`target_file.py`](src/application/schemas/target_file.py)

---

## 阶段3：LLM 配置与提示词管理（US2/US3）

**目标**: LangChain 1.0 统一封装 + 能力映射可配置；提示词可观测与可编辑。
**技术栈**: LangChain 1.0 (Python)

- [x] T020 [P1] 设计并实现 `llm_providers/llm_models/llm_capabilities/prompts` 数据表

  **相关文件**:
  - Entities: [`llm_provider.py`](src/domain/entities/llm_provider.py)、[`llm_model.py`](src/domain/entities/llm_model.py)、[`llm_capability.py`](src/domain/entities/llm_capability.py)、[`prompt.py`](src/domain/entities/prompt.py)
  - 数据库初始化: [`init-db.py`](../scripts/init-db.py)、[`verify-db.py`](../scripts/verify-db.py)
- [x] T021 [P1] 实现 LLM 基础数据 CRUD API

  **相关文件**:
  - Route: [`llm.py`](src/interfaces/api/routes/llm.py)
  - Schemas: [`llm.py`](src/application/schemas/llm.py)
  - Repositories: [`llm_provider_repository.py`](src/application/repositories/llm_provider_repository.py)、[`llm_model_repository.py`](src/application/repositories/llm_model_repository.py)、[`llm_capability_repository.py`](src/application/repositories/llm_capability_repository.py)
  - Services: [`llm_provider_service.py`](src/application/services/llm_provider_service.py)、[`llm_model_service.py`](src/application/services/llm_model_service.py)、[`llm_capability_service.py`](src/application/services/llm_capability_service.py)
  - Tests: [`test_llm_config_api.py`](tests/test_llm_config_api.py)
  - Provider 支持: OpenAI 兼容, ChatGPT, Gemini, Ollama, vLLM, GPUStack, FlagEmbedding，llama.cpp，mineru
  - Model 类型: 推理 (Chat/Completion), Embedding, Rerank, Multimodal, OCR
- [x] T022 [P1] 实现 prompts CRUD + version/active 切换 API

  **相关文件**:
  - Route: [`prompts.py`](src/interfaces/api/routes/prompts.py)
  - Schemas: [`prompt.py`](src/application/schemas/prompt.py)
  - Repository: [`prompt_repository.py`](src/application/repositories/prompt_repository.py)
  - Service: [`prompt_service.py`](src/application/services/prompt_service.py)
  - Tests: [`test_llm_config_api.py`](tests/test_llm_config_api.py)
- [x] T023 [P2] 实现 LLM 统一调用封装（LangChain 1.0）

  **相关文件**:
  - Runtime: [`llm_runtime_service.py`](src/application/services/llm_runtime_service.py)
  - Repositories: [`llm_provider_repository.py`](src/application/repositories/llm_provider_repository.py)、[`llm_model_repository.py`](src/application/repositories/llm_model_repository.py)、[`llm_capability_repository.py`](src/application/repositories/llm_capability_repository.py)、[`prompt_repository.py`](src/application/repositories/prompt_repository.py)
  - 维度1: 技术适配层（标准化不同 Provider 的调用）
  - 维度2: 业务能力层（按 Capability 路由模型：MinerU, 清洗, 润色, 图转JSON, 长文生成, 向量生成）
- [x] T024 [P1] 重构提示词管理为 Code-First Seed 模式
  
  **相关文件**:
  - 常量: [`src/shared/constants/prompts.py`](src/shared/constants/prompts.py)
  - 脚本: [`scripts/init-db.py`](scripts/init-db.py)
  - 业务: [`document_cleaning_service.py`](src/application/services/document_cleaning_service.py)
  - 变更记录: [`refactor-prompt-management.md`](../changes/2026-01/refactor-prompt-management.md)
  - 架构文档: [`spec.md`](../features/prompt-management-refactor/spec.md)、[`plan.md`](../features/prompt-management-refactor/plan.md)

---

## 阶段4：Ingest 流水线（US1）

**目标**: 图片型 PDF → OCR → 清洗 → 图表 JSON → 切块 → 知识库（SQLite+BM25+Chroma）可用。

- [x] T030 [P0] [US1] 接入 MinerU 在线服务：请求/超时/重试/落盘 `mineru_raw`

  **相关文件**:
  - 服务: [`mineru_service.py`](src/application/services/mineru_service.py)
  - Schema: [`mineru.py`](src/application/schemas/mineru.py)
  - 能力: 异步客户端、重试机制、预签名上传、任务轮询、结果落盘
- [x] T031 [P1] [US1] 文档清洗：规则过滤 + 推理 LLM 清洗（去除广告/无意义信息，产出 `cleaned_doc`）

  **相关文件**:
  - 服务: [`document_cleaning_service.py`](src/application/services/document_cleaning_service.py)
  - Schema: [`document_cleaning.py`](src/application/schemas/document_cleaning.py)
  - 能力: 广告过滤、噪声移除、空白标准化、重复去除、LLM 智能清洗
- [x] T032 [P1] [US1] 图表提取与图转 JSON（调用多模态模型，产出 `chart_json`）

  **相关文件**:
  - 服务: [`document_cleaning_service.py`](src/application/services/document_cleaning_service.py)
  - Schema: [`document_cleaning.py`](src/application/schemas/document_cleaning.py)
  - 能力: 图表类型检测、多模态模型调用、JSON 结构化输出
- [x] T033 [P1] [US1] 切块：基于文档结构-语义-句子-长度的顺序切分；写入 `kb_chunks`（基于llamaIndex）

  **相关文件**:
  - 服务: [`chunking_service.py`](src/application/services/chunking_service.py)
  - Schema: [`ingest.py`](src/application/schemas/ingest.py)
  - 策略: 结构感知切分、语义切分、句子切分、长度约束
- [x] T034 [P1] [US1] 向量写入 ChromaDB（调用 FlagEmbedding 模型）

  **相关文件**:
  - 服务: [`vector_storage_service.py`](src/application/services/vector_storage_service.py)
  - Schema: [`ingest.py`](src/application/schemas/ingest.py)
  - 嵌入模型: BAAI/bge-large-zh-v1.5, HuggingFaceEmbedding
- [x] T035 [P2] [US1] 构建混合检索索引（SQLite 元数据 + BM25 + Vector，基于 LlamaIndex）

  **相关文件**:
  - 服务: [`hybrid_search_service.py`](src/application/services/hybrid_search_service.py)
  - 检索策略: RRF 融合、Cross-Encoder 重排序、BM25 + Vector 双路召回

---

## 阶段5：按模板生成单 HTML（US3）

**目标**: 模板驱动长文生成；模板骨架不变；输出单 HTML。

- [x] T040 [P0] [US3] 模板预处理/校验：占位符与结构校验；锁定后禁止修改（不使用 LLM）

   **相关文件**:
   - 服务: [`template_service.py`](src/application/services/template_service.py)
   - Schema: [`template.py`](src/application/schemas/template.py)
   - 能力: 占位符校验 (\{\{\w+\}\})、Markdown 结构校验、Office 格式校验、模板锁定
- [x] T041 [P1] [US3] RAG 检索与上下文组装（基于 LlamaIndex 混合检索 +  Rerank）

   **相关文件**:
   - 服务: [`hybrid_search_service.py`](src/application/services/hybrid_search_service.py)
   - Schema: [`ingest.py`](src/application/schemas/ingest.py)
   - 能力: 向量检索、BM25 检索、RRF 融合、Cross-Encoder 重排序、上下文组装
- [x] T042 [P1] [US3] 按模板 section 生成内容（推理 LLM 润色），召回的图表要使用T043能力重新渲染，输出单 HTML 并写入 `target_files`

   **相关文件**:
   - 服务: [`content_generation_service.py`](src/application/services/content_generation_service.py)
   - Schema: [`chart_spec.py`](src/application/schemas/chart_spec.py)
   - 能力: 模板 section 解析、RAG 上下文注入、LLM 内容生成、Markdown 转 HTML、最终 HTML 组装
- [x] T043 [P2] [US3] 图表 JSON → 图表渲染原子能力（基于 JSON 动态绘制 SVG/PNG/HTML snippet）

   **相关文件**:
   - 服务: [`chart_renderer_service.py`](src/application/services/chart_renderer_service.py)
   - Schema: [`chart_spec.py`](src/application/schemas/chart_spec.py)
   - 能力: ECharts 渲染、Chart.js 渲染、动态图表生成、SVG/PNG/HTML 输出

---

## 阶段6：中台管理前端（Next.js + Shadcn UI）

**目标**: 提供可视化管理界面，涵盖文档管理、LLM 配置、提示词管理与任务观测。
**技术栈**: Next.js 14+, Shadcn UI, Tailwind CSS

- [x] T050 [P0] 初始化前端项目（Next.js, Shadcn UI, Tailwind, Axios/TanStack Query）

  **相关文件**:
  - 源码目录: [`src/interfaces/admin-web/`](src/interfaces/admin-web/00-目录说明.md)
  - 布局: [`layout.tsx`](src/interfaces/admin-web/app/layout.tsx)、[`sidebar.tsx`](src/interfaces/admin-web/components/layout/sidebar.tsx)
  - 基础设施: [`api.ts`](src/interfaces/admin-web/lib/api.ts)、[`providers.tsx`](src/interfaces/admin-web/components/providers.tsx)
- [x] T051 [P1] 实现文档管理页面（源文件上传/列表/归档，模板管理/锁定，目标文件下载/预览，中间态观测）

  **相关文件**:
  - 页面: [`sources/`](src/interfaces/admin-web/app/documents/sources/page.tsx)、[`templates/`](src/interfaces/admin-web/app/documents/templates/page.tsx)、[`targets/`](src/interfaces/admin-web/app/documents/targets/page.tsx)、[`intermediates/`](src/interfaces/admin-web/app/documents/intermediates/page.tsx)
  - Hooks: [`use-documents.ts`](src/interfaces/admin-web/hooks/use-documents.ts)
- [x] T052 [P1] 实现 LLM 配置页面（Provider/Model/Capability 增删改查与测试）

  **相关文件**:
  - 页面: [`providers/`](src/interfaces/admin-web/app/llm/providers/page.tsx)、[`models/`](src/interfaces/admin-web/app/llm/models/page.tsx)、[`capabilities/`](src/interfaces/admin-web/app/llm/capabilities/page.tsx)
  - Hooks: [`use-llm.ts`](src/interfaces/admin-web/hooks/use-llm.ts)
- [x] T053 [P1] 实现提示词管理页面（Prompt 列表/版本管理/编辑）

  **相关文件**:
  - 页面: [`prompts/`](src/interfaces/admin-web/app/prompts/page.tsx)
  - Hooks: [`use-prompts.ts`](src/interfaces/admin-web/hooks/use-prompts.ts)
- [x] T054 [P2] 实现任务与知识库观测页面（Jobs 状态流转，KB 状态查看）

  **相关文件**:
  - 页面: [`dashboard/`](src/interfaces/admin-web/app/page.tsx)、[`jobs/`](src/interfaces/admin-web/app/observation/jobs/page.tsx)
  - Hooks: [`use-observation.ts`](src/interfaces/admin-web/hooks/use-observation.ts)

---

## 测试与验收（横切）

- [x] T090 [P1] 建立 API 基础测试（health、错误格式）
- [ ] T091 [P1] 增加 job 创建/查询的契约测试（含 request_id、celery_task_id）
- [ ] T092 [P2] 增加 ingest→kb→generate 的端到端 happy path（mock MinerU/LLM）

---

**版本**: 1.4.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-16

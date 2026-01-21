---
id: ai-doc-platform-phase1
status: IN_PROGRESS
created: 2026-01-16
updated: 2026-01-19
links:
  - ./spec.md
  - ./plan.md
---

# 任务：第一阶段 AI 文档生成平台

|**输入**: 设计文档  
|**先决条件**: plan.md（必需）、spec.md（必需）

---

## 阶段1：设置（共享基础设施）

|**目的**: 建立可运行骨架与最小闭环（API 创建任务 → Redis 队列 → Worker 执行 → 状态回写 → API 查询）。

|- [x] T001 [P0] 创建符合分层架构的项目结构（`src/`、`tests/`、各目录 `00-目录说明.md`）

  **相关文件**:
  -源码目录: [`src/domain/`](src/domain)、[`src/application/`](src/application)、[`src/interfaces/`](src/interfaces)、[`src/shared/`](src/shared)
  - 测试目录: [`tests/`]
  - 脚本目录: [`scripts/`]
  - 文档目录: [`docs/`](docs/00-目录说明.md)
|- [x] T002 [P0] 初始化 Python 项目依赖（`pyproject.toml`），并提供 `env.example`

  **相关文件**:
  - 项目配置: [`pyproject.toml`](pyproject.toml)
  - 环境示例: [`env.example`](.env.example)
  - 依赖锁定: [`uv.lock`](uv.lock)
|- [x] T003 [P0] 建立 FastAPI 最小服务（`/v1/health`）与统一错误响应格式

  **相关文件**:
  - 主入口: [`main.py`](src/interfaces/api/main.py)
  - 应用实例: [`app.py`](src/interfaces/api/app.py)
  - 健康检查: [`health.py`](src/interfaces/api/routes/health.py)
  - 错误处理: [`errors.py`](src/shared/errors.py)
  - 依赖注入: [`deps.py`](src/interfaces/api/deps.py)
|- [x] T004 [P0] 建立 SQLite 元数据与 `jobs` 表（最小字段）

  **相关文件**:
  - 数据库连接: [`db.py`](src/shared/db.py)
  - 配置管理: [`config.py`](src/shared/config.py)
  - 路由: [`jobs.py`](src/interfaces/api/routes/jobs.py)
  - 数据库脚本: [`init-db.py`](../scripts/init-db.py)、[`verify-db.py`](../scripts/verify-db.py)
|- [x] T005 [P0] 建立 Celery + Redis Worker（占位任务）并接入 `/v1/jobs` 创建即入队

  **相关文件**:
  - Celery 应用: [`celery_app.py`](src/interfaces/worker/celery_app.py)
  - Worker 任务: [`tasks.py`](src/interfaces/worker/tasks.py)
  - 路由: [`jobs.py`](src/interfaces/api/routes/jobs.py)
  - 数据库连接: [`db.py`](src/shared/db.py)
|- [x] T006 [P1] 规范化安装与运行命令（README：安装依赖、启动 API/Worker/Redis）

  **相关文件**:
  - 主文档: [`README.md`](README.md)
  - 快速开始: [`QUICKSTART.md`](QUICKSTART.md)
  - Agent 配置: [`AGENTS.md`](AGENTS.md)

---

## 阶段2：文档/模板/目标文件管理（US2）

|**目标**: 完成 CRUD + 归档 + 中间态观测/删除，形成平台治理能力基础。

|- [x] T010 [P1] [US2] 设计并实现 `source_files/templates/target_files/intermediate_artifacts` 数据表

  **相关文件**:
  - Entities: [`source_file.py`](src/domain/entities/source_file.py)、[`template.py`](src/domain/entities/template.py)、[`intermediate_artifact.py`](src/domain/entities/intermediate_artifact.py)、[`target_file.py`](src/domain/entities/target_file.py)
  - 存储: [`storage.py`](src/shared/storage.py)
  - 数据目录: [`data/sources/`](../data/sources)、[`data/templates/`](../data/templates)、[`data/intermediates/`](../data/intermediates)、[`data/targets/`](../data/targets)
  - 数据库脚本: [`init-db.py`](../scripts/init-db.py)、[`verify-db.py`](../scripts/verify-db.py)
|- [x] T011 [P1] [US2] 实现源文件上传/查询/更新/删除/归档 API（含文件落盘）

  **相关文件**:
  - Entity: [`source_file.py`](src/domain/entities/source_file.py)
  - Repository: [`source_file_repository.py`](src/application/repositories/source_file_repository.py)
  - Service: [`source_file_service.py`](src/application/services/source_file_service.py)
  - Route: [`sources.py`](src/interfaces/api/routes/sources.py)
  - Schema: [`source_file.py`](src/application/schemas/source_file.py)
  - 目录说明: [`entities/`](src/domain/entities/00-目录说明.md)、[`schemas/`](src/application/schemas/00-目录说明.md)
|- [x] T012 [P1] [US2] 实现模板上传/CRUD/锁定/预处理校验 API（预处理不使用 LLM）

  **相关文件**:
  - Entity: [`template.py`](src/domain/entities/template.py)
  - Repository: [`template_repository.py`](src/application/repositories/template_repository.py)
  - Service: [`template_service.py`](src/application/services/template_service.py)
  - Route: [`templates.py`](src/interfaces/api/routes/templates.py)
  - Schema: [`template.py`](src/application/schemas/template.py)
|- [x] T013 [P2] [US2] 实现中间态列表/详情/删除 API

  **相关文件**:
  - Entity: [`intermediate_artifact.py`](src/domain/entities/intermediate_artifact.py)
  - Repository: [`intermediate_artifact_repository.py`](src/application/repositories/intermediate_artifact_repository.py)
  - Service: [`intermediate_artifact_service.py`](src/application/services/intermediate_artifact_service.py)
  - Route: [`intermediates.py`](src/interfaces/api/routes/intermediates.py)
  - Schema: [`intermediate_artifact.py`](src/application/schemas/intermediate_artifact.py)
  - 目录说明: [`entities/`](src/domain/entities/00-目录说明.md)、[`schemas/`](src/application/schemas/00-目录说明.md)
|- [x] T014 [P2] [US2] 实现目标文件查询/下载 API

  **相关文件**:
  - Entity: [`target_file.py`](src/domain/entities/target_file.py)
  - Repository: [`target_file_repository.py`](src/application/repositories/target_file_repository.py)
  - Service: [`target_file_service.py`](src/application/services/target_file_service.py)
  - Route: [`targets.py`](src/interfaces/api/routes/targets.py)
  - Schema: [`target_file.py`](src/application/schemas/target_file.py)

---

## 阶段3：LLM 配置与提示词管理（US2/US3）

|**目标**: LangChain 1.0 统一封装 + 能力映射可配置；提示词可观测与可编辑。
|**技术栈**: LangChain 1.0 (Python)

|- [x] T020 [P1] 设计并实现 `llm_providers/llm_capabilities/llm_call_sites/prompts` 数据表

  **相关文件**:
  - Entities: [`llm_provider.py`](src/domain/entities/llm_provider.py)、[`llm_capability.py`](src/domain/entities/llm_capability.py)、[`llm_call_site.py`](src/domain/entities/llm_call_site.py)、[`prompt.py`](src/domain/entities/prompt.py)
  - 数据库初始化: [`init-db.py`](../scripts/init-db.py)、[`verify-db.py`](../scripts/verify-db.py)
  - **架构说明**: CallSite 和 Capability 直接绑定 Provider，不再使用 Model 中间层
|- [x] T021 [P1] 实现 LLM 基础数据 CRUD API

  **相关文件**:
  - Route: [`llm.py`](src/interfaces/api/routes/llm.py)
  - Schemas: [`llm.py`](src/application/schemas/llm.py)
  - Repositories: [`llm_provider_repository.py`](src/application/repositories/llm_provider_repository.py)、[`llm_capability_repository.py`](src/application/repositories/llm_capability_repository.py)、[`llm_call_site_repository.py`](src/application/repositories/llm_call_site_repository.py)
  - Services: [`llm_provider_service.py`](src/application/services/llm_provider_service.py)、[`llm_capability_service.py`](src/application/services/llm_capability_service.py)、[`llm_call_site_service.py`](src/application/services/llm_call_site_service.py)
  - Tests: [`test_llm_config_api.py`](tests/test_llm_config_api.py)
  - Provider 支持: OpenAI 兼容, ChatGPT, Gemini, Ollama, vLLM, GPUStack, FlagEmbedding，llama.cpp，mineru
  - 模型类型: 通过 CallSite 的 `expected_model_kind` 字段确定（Chat/Completion, Embedding, Rerank, Multimodal, OCR）
  - **配置方式**: 通过中台管理界面配置 LLM Provider（HTTP endpoint + API Key + model 字段），无需硬编码
  - **架构变更** (2026-01-18): 移除 Model 层，CallSite 和 Capability 直接绑定 Provider
|- [x] T022 [P1] 实现 prompts CRUD + version/active 切换 API

  **相关文件**:
  - Route: [`prompts.py`](src/interfaces/api/routes/prompts.py)
  - Schemas: [`prompt.py`](src/application/schemas/prompt.py)
  - Repository: [`prompt_repository.py`](src/application/repositories/prompt_repository.py)
  - Service: [`prompt_service.py`](src/application/services/prompt_service.py)
  - Tests: [`test_llm_config_api.py`](tests/test_llm_config_api.py)
|- [x] T023 [P2] 实现 LLM 统一调用封装（LangChain 1.0）

  **相关文件**:
  - Runtime: [`llm_runtime_service.py`](src/application/services/llm_runtime_service.py)
  - Repositories: [`llm_provider_repository.py`](src/application/repositories/llm_provider_repository.py)、[`llm_capability_repository.py`](src/application/repositories/llm_capability_repository.py)、[`llm_call_site_repository.py`](src/application/repositories/llm_call_site_repository.py)、[`prompt_repository.py`](src/application/repositories/prompt_repository.py)
  - 维度1: 技术适配层（标准化不同 Provider 的调用）
  - 维度2: 业务能力层（按 CallSite 或 Capability 路由 Provider：MinerU, 清洗, 润色, 图转JSON, 长文生成, 向量生成）
  - **配置方式**: 从数据库加载 Provider 配置（config_json 包含 base_url、api_key、model 等），动态构建模型
  - **架构变更** (2026-01-18): 运行时直接从 Provider 构建模型，使用 CallSite 的 `expected_model_kind` 确定模型类型，模型名称从 Provider 的 `config_json` 中获取
|- [x] T024 [P1] 重构提示词管理为 Code-First Seed 模式
  
  **相关文件**:
  - 常量: [`src/shared/constants/prompts.py`](src/shared/constants/prompts.py)
  - 脚本: [`scripts/init-db.py`](scripts/init-db.py)
  - 业务: [`document_cleaning_service.py`](src/application/services/document_cleaning_service.py)
  - 变更记录: [`refactor-prompt-management.md`](../changes/2026-01/refactor-prompt-management.md)
  - 架构文档: [`spec.md`](../features/prompt-management-refactor/spec.md)、[`plan.md`](../features/prompt-management-refactor/plan.md)
|- [x] T025 [P1] 引入"LLM 调用点（CallSite）"配置（按调用位置细粒度绑定模型与参数覆盖）

  **相关文件**:
  - Entity: [`llm_call_site.py`](src/domain/entities/llm_call_site.py)
  - Repository: [`llm_call_site_repository.py`](src/application/repositories/llm_call_site_repository.py)
  - Service: [`llm_call_site_service.py`](src/application/services/llm_call_site_service.py)
  - Runtime: [`llm_runtime_service.py`](src/application/services/llm_runtime_service.py)
  - Route: [`llm.py`](src/interfaces/api/routes/llm.py)
  - Code-First 注册: [`llm_callsites.py`](src/shared/constants/llm_callsites.py)、各模块 `callsites.py`
  - 数据库初始化/迁移: [`init-db.py`](../scripts/init-db.py)、[`db.py`](src/shared/db.py)
  - **架构变更** (2026-01-18): CallSite 直接绑定 Provider（`provider_id`），不再通过 Model 中间层。模型类型由 `expected_model_kind` 字段确定，模型名称从 Provider 的 `config_json` 中获取

---

## 阶段4：Ingest 流水线（US1）

|**目标**: 图片型 PDF → OCR → 清洗 → 图表 JSON → 切块 → 知识库（SQLite+BM25+Chroma）可用。

|- [x] T030 [P0] [US1] 接入 MinerU 在线服务：请求/超时/重试/落盘 `mineru_raw`

  **相关文件**:
  - 服务: [`mineru_service.py`](src/application/services/mineru_service.py)
  - Schema: [`mineru.py`](src/application/schemas/mineru.py)
  - 能力: 异步客户端、重试机制、预签名上传、任务轮询、结果落盘
|- [x] T031 [P1] [US1] 文档清洗：规则过滤 + 推理 LLM 清洗（去除广告/无意义信息，产出 `cleaned_doc`）

  **相关文件**:
  - 服务: [`document_cleaning_service.py`](src/application/services/document_cleaning_service.py)
  - Schema: [`document_cleaning.py`](src/application/schemas/document_cleaning.py)
  - 能力: 广告过滤、噪声移除、空白标准化、重复去除、LLM 智能清洗
|- [x] T032 [P1] [US1] 图表提取与图转 JSON（调用多模态模型，产出 `chart_json`）

  **相关文件**:
  - 服务: [`document_cleaning_service.py`](src/application/services/document_cleaning_service.py)
  - Schema: [`document_cleaning.py`](src/application/schemas/document_cleaning.py)
  - 能力: 图表类型检测、多模态模型调用、JSON 结构化输出
|- [x] T033 [P1] [US1] 切块：基于文档结构-语义-句子-长度的顺序切分；写入 `kb_chunks`（基于llamaIndex）

  **相关文件**:
  - 服务: [`chunking_service.py`](src/application/services/chunking_service.py)
  - Schema: [`ingest.py`](src/application/schemas/ingest.py)
  - 策略: 结构感知切分、语义切分、句子切分、长度约束
|- [x] T034 [P1] [US1] 向量写入 ChromaDB（调用 FlagEmbedding 模型）

  **相关文件**:
  - 服务: [`vector_storage_service.py`](src/application/services/vector_storage_service.py)
  - Schema: [`ingest.py`](src/application/schemas/ingest.py)
  - 嵌入模型: BAAI/bge-large-zh-v1.5, HuggingFaceEmbedding
|- [x] T035 [P2] [US1] 构建混合检索索引（SQLite 元数据 + BM25 + Vector，基于 LlamaIndex）

  **相关文件**:
  - 服务: [`hybrid_search_service.py`](src/application/services/hybrid_search_service.py)
  - 检索策略: RRF 融合、Cross-Encoder 重排序、BM25 + Vector 双路召回
|- [ ] T036 [P1] [US1] 实现 Ingest API 路由（`POST /v1/ingest`），支持单文档/批量摄入

  **相关文件**:
  - 路由: [`ingest.py`](src/interfaces/api/routes/ingest.py)
  - Schema: [`ingest.py`](src/application/schemas/ingest.py)
  - 能力: 任务创建、参数配置、异步任务触发
|- [ ] T037 [P1] [US1] 实现 Ingest Pipeline Celery 任务编排

  **相关文件**:
  - 任务: [`ingest_tasks.py`](src/interfaces/worker/ingest_tasks.py)
  - 编排: MinerU → 清洗 → 图转 JSON → 切块 → 向量写入
  - 中间产物落盘与状态回写

---

## 阶段5：按模板生成单 HTML（US3）

|**目标**: 模板驱动长文生成；模板骨架不变；输出单 HTML。

|- [x] T040 [P0] [US3] 模板预处理/校验：占位符与结构校验；锁定后禁止修改（不使用 LLM）

   **相关文件**:
   - 服务: [`template_service.py`](src/application/services/template_service.py)
   - Schema: [`template.py`](src/application/schemas/template.py)
   - 能力: 占位符校验 (\{\{\w+\}\})、Markdown 结构校验、Office 格式校验、模板锁定
|- [x] T041 [P1] [US3] RAG 检索与上下文组装（基于 LlamaIndex 混合检索 +  Rerank）

   **相关文件**:
   - 服务: [`hybrid_search_service.py`](src/application/services/hybrid_search_service.py)
   - Schema: [`ingest.py`](src/application/schemas/ingest.py)
   - 能力: 向量检索、BM25 检索、RRF 融合、Cross-Encoder 重排序、上下文组装
|- [x] T042 [P1] [US3] 按模板 section 生成内容（推理 LLM 润色），召回的图表要使用T043能力重新渲染，输出单 HTML 并写入 `target_files`

   **相关文件**:
   - 服务: [`content_generation_service.py`](src/application/services/content_generation_service.py)
   - Schema: [`chart_spec.py`](src/application/schemas/chart_spec.py)
   - 能力: 模板 section 解析、RAG 上下文注入、LLM 内容生成、Markdown 转 HTML、最终 HTML 组装
|- [x] T043 [P2] [US3] 图表 JSON → 图表渲染原子能力（基于 JSON 动态绘制 SVG/PNG/HTML snippet）

   **相关文件**:
   - 服务: [`chart_renderer_service.py`](src/application/services/chart_renderer_service.py)
   - Schema: [`chart_spec.py`](src/application/schemas/chart_spec.py)
   - 能力: ECharts 渲染、Chart.js 渲染、动态图表生成、SVG/PNG/HTML 输出
|- [ ] T044 [P1] [US3] 实现 Generate API 路由（`POST /v1/generate`），支持模板+知识库生成

   **相关文件**:
   - 路由: [`generate.py`](src/interfaces/api/routes/generate.py)
   - Schema: [`content_generation.py`](src/application/schemas/content_generation.py)
   - 能力: 任务创建、大纲润色选项、异步任务触发
|- [ ] T045 [P1] [US3] 实现 Generate Pipeline Celery 任务编排

   **相关文件**:
   - 任务: [`generate_tasks.py`](src/interfaces/worker/generate_tasks.py)
   - 编排: 模板解析 → RAG 检索 → 内容生成 → 图表渲染 → HTML 组装
   - 目标文件落盘与状态回写

---

## 阶段6：中台管理前端（Next.js + Shadcn UI）

|**目标**: 提供可视化管理界面，涵盖文档管理、LLM 配置、提示词管理与任务观测。
|**技术栈**: Next.js 14+, Shadcn UI, Tailwind CSS

|- [x] T050 [P0] 初始化前端项目（Next.js, Shadcn UI, Tailwind, Axios/TanStack Query）

  **相关文件**:
  - 源码目录: [`src/interfaces/admin-web/`](src/interfaces/admin-web/00-目录说明.md)
  - 布局: [`layout.tsx`](src/interfaces/admin-web/app/layout.tsx)、[`sidebar.tsx`](src/interfaces/admin-web/components/layout/sidebar.tsx)
  - 基础设施: [`api.ts`](src/interfaces/admin-web/lib/api.ts)、[`providers.tsx`](src/interfaces/admin-web/components/providers.tsx)
|- [x] T051 [P1] 实现文档管理页面（源文件上传/列表/归档，模板管理/锁定，目标文件下载/预览，中间态观测）

  **相关文件**:
  - 页面: [`sources/`](src/interfaces/admin-web/app/documents/sources/page.tsx)、[`templates/`](src/interfaces/admin-web/app/documents/templates/page.tsx)、[`targets/`](src/interfaces/admin-web/app/documents/targets/page.tsx)、[`intermediates/`](src/interfaces/admin-web/app/documents/intermediates/page.tsx)
  - Hooks: [`use-documents.ts`](src/interfaces/admin-web/hooks/use-documents.ts)
|- [x] T052 [P1] 实现 LLM 配置页面（Provider/Capability/CallSite 增删改查与测试）

  **相关文件**:
  - 页面: [`providers/`](src/interfaces/admin-web/app/llm/providers/page.tsx)、[`capabilities/`](src/interfaces/admin-web/app/llm/capabilities/page.tsx)、[`call-sites/`](src/interfaces/admin-web/app/llm/call-sites/page.tsx)
  - Hooks: [`use-llm.ts`](src/interfaces/admin-web/hooks/use-llm.ts)
  - **架构变更** (2026-01-18): 移除 Model 管理页面，CallSite 和 Capability 页面改为直接绑定 Provider
|- [x] T055 [P1] 实现"LLM 调用点"管理页面（绑定模型/参数覆盖/prompt_scope/启用）

  **相关文件**:
  - 页面: [`call-sites/`](src/interfaces/admin-web/app/llm/call-sites/page.tsx)
  - Hooks: [`use-llm.ts`](src/interfaces/admin-web/hooks/use-llm.ts)
  - 导航: [`sidebar.tsx`](src/interfaces/admin-web/components/layout/sidebar.tsx)
|- [x] T053 [P1] 实现提示词管理页面（Prompt 列表/版本管理/编辑）

  **相关文件**:
  - 页面: [`prompts/`](src/interfaces/admin-web/app/prompts/page.tsx)
  - Hooks: [`use-prompts.ts`](src/interfaces/admin-web/hooks/use-prompts.ts)
|- [x] T054 [P2] 实现任务与知识库观测页面（Jobs 状态流转，KB 状态查看）

  **相关文件**:
  - 页面: [`dashboard/`](src/interfaces/admin-web/app/page.tsx)、[`jobs/`](src/interfaces/admin-web/app/observation/jobs/page.tsx)
  - Hooks: [`use-observation.ts`](src/interfaces/admin-web/hooks/use-observation.ts)

---

## 测试与验收（横切）

|- [x] T090 [P1] 建立 API 基础测试（health、错误格式）
|- [ ] T091 [P1] 增加 job 创建/查询的契约测试（含 request_id、celery_task_id）
|- [ ] T092 [P2] 增加 ingest→kb→generate 的端到端 happy path（mock MinerU/LLM）
|- [x] T093 [P1] 实现 MinerU 清洗功能测试脚本（`test_mineru_cleaning.py`）

  **测试范围**:
  - MinerU 在线服务连接与认证
  - 预签名 URL 申请逻辑
  - 任务提交与状态轮询
  - 结果落盘验证（输出至 `data/intermediates/{source_file_id}/mineru_raw/`）
  - 测试数据: `data/sources/default/*.pdf`
|- [x] T097 [P1] 实现文档噪音清洗功能测试脚本（`test_document_cleaning.py`）

  **测试范围**:
  - 规则过滤（广告、噪声、重复内容）
  - LLM 智能清洗
  - 输入来源: `data/intermediates/{source_file_id}/mineru_raw/` 的 md/json 文件
  - 输出验证: `data/intermediates/{source_file_id}/cleaned_doc/` 清洗后的完整 md/json
  - **保留完整内容**: md 和 json 文件均完整保留，供后续图转 JSON 使用
  - **测试方法**: 不使用 mock，直接调用服务处理真实 MinerU 输出

  **数据流转**:
  ```
  data/sources/default/*.pdf → MinerU OCR → data/intermediates/{id}/mineru_raw/
  ↓
  文档噪音清洗（T097）
  ↓
  data/intermediates/{id}/cleaned_doc/ → 图转 JSON（T094）
  ```
|- [x] T094 [P1] 完成图转 JSON 批处理测试脚本（`scripts/t094-pic-to-json.py`）

  **测试命令（PowerShell）**:
  ```powershell
  uv run python "scripts/t094-pic-to-json.py" --strict --concurrency 1 --progress-every 1 --resume
  ```

  **测试范围**:
  - 图表检测算法验证
  - 多模态模型调用（需配置 GPT-4V 或类似）
  - JSON 输出格式验证
  - 中间产物落盘验证

  **关键约束（避免遗漏）**:
  - **依赖 T097 输出完整性**：T094 的输入来自 `data/intermediates/{id}/cleaned_doc/`，需要 **完整保留** `md/json/images`，用于图表定位与后续图转 JSON。
  - **图片链接必须可追溯**：T097 清洗后 Markdown 中的图片引用（`![](images/xxx.jpg)`）必须保留（至少保留链接信息），否则 T094 无法可靠关联图片与正文位置。
  - **不要在 T094 前做不可逆裁剪**：封面/作者照/版权页等“装饰性图片”的**文件删除或 JSON 剔除**应在 T094 之后、且基于图表识别结果（“有意义图表白名单/装饰图黑名单”）再执行，避免误删图表导致 T094 失败。
  - **建议产出清单工件**：T094 测试中建议落盘“识别为有意义图表的图片列表”（如 `chart_image_paths` / `chart_ids`），供后续阶段做装饰图剔除与 UI 关联。
|- [ ] T095 [P1] 实现知识库构建功能测试脚本（`test_knowledge_base.py`）

  **测试范围**:
  - 文档切分质量验证（边界检测、去重）
  - ChromaDB 向量写入与检索
  - 混合检索 RRF 融合效果
  - 重排序功能验证
|- [ ] T096 [P1] 实现生成白皮书功能测试脚本（`test_content_generation.py`）

  **测试范围**:
  - 模板 section 解析完整性
  - RAG 检索上下文质量
  - LLM 生成内容与模板格式对齐
  - 图表渲染正确性
  - **大纲润色功能验证**（polish_outline）

---

|**版本**: 1.5.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-19

## 架构变更记录

### 2026-01-18: FlagEmbedding 配置项完善

|**变更内容**:
- 为 `flagembedding` Provider 类型添加完整的配置项支持
  - `EMBEDDING_MODEL_PATH`: Embedding 模型本地路径（如 `./models/bge-m3`）
  - `EMBEDDING_DIMENSION`: 向量维度（如 `1024`）
  - `RERANK_MODEL_PATH`: Rerank 模型本地路径（如 `./models/bge-reranker-v2-m3`）
  - `RERANK_TOP_K`: 返回 Top-K 结果数量（如 `10`）
- 更新后端运行时服务，支持从配置中读取模型路径（优先级：`embedding_model_path`/`rerank_model_path` > `model` > `model_name` > `provider.name`）

|**相关文件**:
- 常量: [`llm_providers.py`](src/shared/constants/llm_providers.py)
- 前端: [`providers/page.tsx`](src/interfaces/admin-web/app/llm/providers/page.tsx)
- 运行时: [`llm_runtime_service.py`](src/application/services/llm_runtime_service.py)

|**影响任务**:
- T021: LLM 基础数据 CRUD API（Provider 配置项扩展）
- T023: LLM 统一调用封装（支持模型路径配置）
- T052: LLM 配置页面（Provider 配置界面完善）

### 2026-01-18: Provider 配置增强与大纲润色功能

|**变更内容**:
- 为 `flagembedding` 和 `ollama` Provider 类型添加 `max_tokens` 配置项
- 新增用户自定义大纲的 LLM 润色功能
  - 新增调用点：`content_generation:polish_outline`
  - 新增提示词：`SCOPE_OUTLINE_POLISH`
  - 在 `ContentGenerationService` 中实现 `polish_outline` 方法

|**相关文件**:
- 常量: [`llm_providers.py`](src/shared/constants/llm_providers.py)
- 前端: [`providers/page.tsx`](src/interfaces/admin-web/app/llm/providers/page.tsx)
- 调用点: [`content_generation/callsites.py`](src/application/services/content_generation/callsites.py)
- 提示词: [`content_generation/prompts.py`](src/application/services/content_generation/prompts.py)
- 服务: [`content_generation_service.py`](src/application/services/content_generation_service.py)
- 常量聚合: [`prompts.py`](src/shared/constants/prompts.py)

|**影响任务**:
- T021: LLM 基础数据 CRUD API（Provider 配置项扩展）
- T023: LLM 统一调用封装（支持大纲润色调用点）
- T025: LLM 调用点配置（新增大纲润色调用点）
- T052: LLM 配置页面（Provider 配置界面增强）

### 2026-01-18: 移除 Model 层，CallSite 和 Capability 直接绑定 Provider

|**变更内容**:
- 删除 `llm_models` 数据表和 `LLMModel` 实体
- CallSite 和 Capability 的 `model_id` 字段改为 `provider_id`
- 运行时服务直接从 Provider 构建模型，使用 CallSite 的 `expected_model_kind` 确定模型类型
- 模型名称从 Provider 的 `config_json` 中获取（`model` 或 `model_name` 字段）
- 删除前端 Model 管理页面

|**影响任务**:
- T020: 数据表设计（移除 `llm_models`）
- T021: CRUD API（移除 Model 相关路由）
- T023: 运行时封装（改为直接从 Provider 构建）
- T025: CallSite 配置（改为绑定 Provider）
- T052: 前端配置页面（移除 Model 页面）

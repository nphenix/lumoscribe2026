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
  - 根目录: [`00-目录说明.md`](00-目录说明.md)
  - 源码目录: [`src/00-目录说明.md`](src/00-目录说明.md)、[`src/domain/`](src/domain)、[`src/application/`](src/application)、[`src/interfaces/`](src/interfaces)、[`src/shared/`](src/shared)
  - 测试目录: [`tests/00-目录说明.md`](tests/00-目录说明.md)
  - 脚本目录: [`scripts/00-目录说明.md`](scripts/00-目录说明.md)
  - 文档目录: [`docs/00-目录说明.md`](docs/00-目录说明.md)
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
  - 目录说明: [`entities/00-目录说明.md`](src/domain/entities/00-目录说明.md)、[`schemas/00-目录说明.md`](src/application/schemas/00-目录说明.md)
- [ ] T012 [P1] [US2] 实现模板上传/CRUD/锁定/预处理校验 API（预处理不使用 LLM）

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
  - 目录说明: [`entities/00-目录说明.md`](src/domain/entities/00-目录说明.md)、[`schemas/00-目录说明.md`](src/application/schemas/00-目录说明.md)
- [ ] T014 [P2] [US2] 实现目标文件查询/下载 API

  **相关文件**:
  - Entity: [`target_file.py`](src/domain/entities/target_file.py)
  - Repository: [`target_file_repository.py`](src/application/repositories/target_file_repository.py)
  - Service: [`target_file_service.py`](src/application/services/target_file_service.py)
  - Route: [`targets.py`](src/interfaces/api/routes/targets.py)
  - Schema: [`target_file.py`](src/application/schemas/target_file.py)

---

## 阶段3：LLM 配置与提示词管理（US2/US3）

**目标**: LangChain 1.0 统一封装 + 能力映射可配置；提示词可观测与可编辑。

- [ ] T020 [P1] 设计并实现 `llm_providers/llm_models/llm_capabilities/prompts` 数据表
- [ ] T021 [P1] 实现 provider/model/capability CRUD API
- [ ] T022 [P1] 实现 prompts CRUD + version/active 切换 API
- [ ] T023 [P2] 实现运行时按 capability 选择模型并构建 LangChain runnable 的封装

---

## 阶段4：Ingest 流水线（US1）

**目标**: 图片型 PDF → OCR → 清洗 → 图表 JSON → 切块 → 知识库（SQLite+BM25+Chroma）可用。

- [ ] T030 [P0] [US1] 接入 MinerU 在线服务：请求/超时/重试/落盘 `mineru_raw`
- [ ] T031 [P1] [US1] 文档清洗：规则过滤 + 推理 LLM 清洗（产出 `cleaned_doc`）
- [ ] T032 [P1] [US1] 图表提取与图转 JSON（多模态模型，产出 `chart_json`）
- [ ] T033 [P1] [US1] 切块：结构→语义→句子→长度；写入 `kb_chunks`
- [ ] T034 [P1] [US1] 向量写入 ChromaDB（FlagEmbedding）
- [ ] T035 [P2] [US1] BM25 索引与混合检索骨架（LlamaIndex）

---

## 阶段5：按模板生成单 HTML（US3）

**目标**: 模板驱动长文生成；模板骨架不变；输出单 HTML。

- [ ] T040 [P0] [US3] 模板预处理/校验：占位符与结构校验；锁定后禁止修改
- [ ] T041 [P1] [US3] RAG 检索与上下文组装（混合检索 + 可选 rerank）
- [ ] T042 [P1] [US3] 按模板 section 生成内容（推理 LLM），输出单 HTML 并写入 `target_files`
- [ ] T043 [P2] [US3] 图表 JSON → 图表渲染原子能力（SVG/PNG/HTML snippet）

---

## 测试与验收（横切）

- [x] T090 [P1] 建立 API 基础测试（health、错误格式）
- [ ] T091 [P1] 增加 job 创建/查询的契约测试（含 request_id、celery_task_id）
- [ ] T092 [P2] 增加 ingest→kb→generate 的端到端 happy path（mock MinerU/LLM）

---

**版本**: 1.1.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-16

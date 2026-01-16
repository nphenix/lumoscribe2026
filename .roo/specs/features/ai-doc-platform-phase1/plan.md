---
id: ai-doc-platform-phase1
status: DRAFT
created: 2026-01-16
updated: 2026-01-16
links:
  - ./spec.md
  - ./tasks.md
---

# 实施计划：第一阶段 AI 文档生成平台

**分支**: `feature/phase1-ai-doc-platform` | **日期**: 2026-01-16 | **规格**: [spec.md](./spec.md)

**输入**: 功能规格说明

---

## 摘要

第一阶段交付一个可集成的 API + 中台管理平台，实现端到端闭环：**图片型 PDF → MinerU OCR/结构化 → 文档清洗 → 图表转 JSON → 切块 → 知识库（BM25+向量） → 按模板 RAG+LLM 生成单 HTML**；并提供文档/模板/中间态/目标文件、LLM 配置、提示词的统一管理与观测。

## 技术上下文

- **语言/版本**: Python 3.12
- **主要依赖**: FastAPI、Celery、Redis、SQLAlchemy、httpx
- **Agent/LLM**: LangChain 1.0（统一模型封装与动态选择）
- **RAG**: LlamaIndex（混合检索与组装）
- **存储**: SQLite（元数据/配置/任务状态） + ChromaDB（向量库） + 本地文件系统（工件文件）
- **测试**: pytest（默认依赖）
- **目标平台**: Windows 11（开发）、后续可容器化部署
- **项目类型**: 后端（API + Worker）+ 中台（Next.js，后续里程碑）
- **约束条件**:
  - 输出必须严格遵守模板骨架（模板不可被 LLM 改写）
  - 错误响应与日志必须统一、可追踪（request_id/job_id）
  - 不硬编码外部 endpoint/key/model；一律走配置与 DB

## 架构与关键决策

### 进程拆分

- **API 服务**：负责 CRUD、任务创建、状态查询、文件下载。
- **Worker**：Celery 执行耗时任务（ingest/generate/render_chart），通过 Redis 队列调度。

### 任务状态权威来源

- **SQLite `jobs` 表**作为任务状态权威来源；Celery task id 仅作为关联字段（便于排障/追踪）。

### LLM 封装策略（两维度）

- **Provider 维度**：openai-compatible（覆盖 vLLM/GPUStack 等）、Gemini、Ollama 等。
- **Capability 维度**：pdf_ocr/doc_clean/chart_to_json/long_doc_generate/embed/rerank 等能力映射到模型。

### 知识库检索策略

- **混合检索**：BM25 + 向量召回融合（RRF/加权），可选 rerank 能力二次排序。

## 项目结构（本功能）

```
speckit/specs/features/ai-doc-platform-phase1/
├── spec.md
├── plan.md
└── tasks.md
```

## 源代码结构（仓库根目录）

```
src/
├── domain/
├── application/
├── interfaces/
│   ├── api/
│   ├── worker/
│   └── admin-web/
└── shared/

tests/
```

**结构决策**：采用 `domain/application/interfaces/shared` 分层，避免业务逻辑与接口/基础设施耦合，便于后续扩展多模型、多流水线与多前端集成。

---

**版本**: 1.0.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-16

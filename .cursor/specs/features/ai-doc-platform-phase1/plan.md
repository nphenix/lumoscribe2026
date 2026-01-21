---
id: ai-doc-platform-phase1
status: IN_PROGRESS
created: 2026-01-16
updated: 2026-01-18
links:
  - ./spec.md
  - ./tasks.md
---

# 实施计划：第一阶段 AI 文档生成平台

|**分支**: `feature/phase1-ai-doc-platform` | **日期**: 2026-01-16 | **规格**: [spec.md](./spec.md) | **任务**: [tasks.md](./tasks.md)

|**输入**: 功能规格说明

---

## 摘要

第一阶段交付一个可集成的 API + 中台管理平台，实现端到端闭环：**图片型 PDF → MinerU OCR/结构化 → 文档清洗 → 图表转 JSON → 切块 → 知识库（BM25+向量） → 按模板 RAG+LLM 生成单 HTML**；并提供文档/模板/中间态/目标文件、LLM 配置、提示词的统一管理与观测。

## 技术上下文

|- **语言/版本**: Python 3.12, TypeScript (Next.js 14+)
|- **主要依赖**: FastAPI、Celery、Redis、SQLAlchemy、httpx, React, Shadcn UI, Tailwind CSS
|- **Agent/LLM**: LangChain 1.0（统一模型封装与动态选择）
|- **RAG**: LlamaIndex（混合检索与组装）
|- **存储**: SQLite（元数据/配置/任务状态） + ChromaDB（向量库） + 本地文件系统（工件文件）
|- **测试**: pytest（后端）, jest/react-testing-library（前端）
|- **目标平台**: Windows 11（开发）、后续可容器化部署
|- **项目类型**: 后端（API + Worker）+ 中台（Next.js）
|- **约束条件**:
  - 输出必须严格遵守模板骨架（模板不可被 LLM 改写）
  - 错误响应与日志必须统一、可追踪（request_id/job_id）
  - 不硬编码外部 endpoint/key/model；一律走配置与 DB（通过中台管理界面配置 LLM Provider）

## 架构与关键决策

### 进程拆分

|- **API 服务**：负责 CRUD、任务创建、状态查询、文件下载。
|- **Worker**：Celery 执行耗时任务（ingest/generate/render_chart），通过 Redis 队列调度。

### 任务状态权威来源

|- **SQLite `jobs` 表**作为任务状态权威来源；Celery task id 仅作为关联字段（便于排障/追踪）。

### LLM 封装策略（两维度）

|- **Provider 维度**：OpenAI 兼容（覆盖 vLLM/GPUStack 等）、ChatGPT、Gemini、Ollama、FlagEmbedding 等。
|- **Capability 维度**：pdf_ocr（MinerU）、doc_clean、chart_to_json、long_doc_generate、embed、rerank 等能力映射到模型。
|- **配置方式**：通过中台管理界面动态配置 Provider（HTTP endpoint + API Key + model 字段），运行时从数据库加载配置。

### 知识库检索策略

|- **混合检索**：BM25 + 向量召回融合（RRF/加权），可选 rerank 能力二次排序。基于 LlamaIndex 实现。

### 测试策略

|- **独立测试**：每个核心功能（mineru清洗、图转json、知识库构建、生成白皮书）通过独立测试脚本验证
|- **流水线测试**：待 API 路由和 Celery 任务补齐后，再进行端到端测试

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
│   │   └── routes/
│   │       ├── ingest.py      # T036: Ingest API 路由（待实现）
│   │       └── generate.py    # T044: Generate API 路由（待实现）
│   ├── worker/
│   │   ├── ingest_tasks.py    # T037: Ingest Pipeline Celery 任务（待实现）
│   │   └── generate_tasks.py  # T045: Generate Pipeline Celery 任务（待实现）
│   └── admin-web/
└── shared/
    └── constants/
        ├── llm_providers.py   # Provider 配置常量
        └── llm_callsites.py   # CallSite 配置常量

tests/
├── test_mineru_cleaning.py    # T093: MinerU 清洗测试（待实现）
├── test_knowledge_base.py     # T095: 知识库构建测试（待实现）
└── test_content_generation.py # T096: 生成白皮书测试（待实现）
```

|**结构决策**：
1. 后端采用 `domain/application/interfaces/shared` 分层，避免业务逻辑与接口/基础设施耦合。
2. 前端采用 Next.js App Router 结构，位于 `src/interfaces/admin-web/` 目录，与后端代码库共存（Monorepo 风格），便于全栈开发与管理。
3. LLM 配置通过中台管理界面存储到数据库，运行时动态加载，不硬编码任何 Provider 信息。

## 实施路线图

### 阶段 1-3：已完成

|- 基础设施搭建（API、Worker、DB）
|- 文档/模板 CRUD
|- LLM 配置与提示词管理

### 阶段 4-5：核心服务层（已完成）

|- MinerU 接入
|- 文档清洗
|- 图表提取
|- 文档切分
|- 向量存储
|- 混合检索
|- 内容生成（含大纲润色）
|- 图表渲染

### 阶段 4-5：API 层与 Worker 层（待补齐）

|- T036: Ingest API 路由
|- T037: Ingest Pipeline Celery 任务
|- T044: Generate API 路由
|- T045: Generate Pipeline Celery 任务

### 测试阶段（当前进行）

|- T093: MinerU 清洗功能测试
|- T094: 图转 JSON 功能测试
|- T095: 知识库构建功能测试
|- T096: 生成白皮书功能测试

## 验收标准

### 功能验收

|- [ ] MinerU 清洗功能测试通过
|- [ ] 图转 JSON 功能测试通过
|- [ ] 知识库构建功能测试通过
|- [ ] 生成白皮书功能测试通过（含大纲润色）

### 非功能验收

|- [ ] 所有测试脚本可独立运行
|- [ ] 测试结果可复现
|- [ ] 错误信息清晰可追踪

---

|**版本**: 1.2.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-18

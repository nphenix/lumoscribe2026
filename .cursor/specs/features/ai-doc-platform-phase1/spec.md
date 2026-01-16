---
id: ai-doc-platform-phase1
status: DRAFT
created: 2026-01-16
updated: 2026-01-16
---

# 功能规格说明：第一阶段 AI 文档生成平台

**功能分支**: `feature/phase1-ai-doc-platform`  
**创建日期**: 2026-01-16  
**状态**: 草稿  
**输入**: “图片型 PDF 清洗后构建知识库，通过指定模板使用 RAG + LLM 生成规定格式长文件（单 HTML），并提供 API 与中台管理。”

---

## 用户场景与测试

### 用户故事 1 - 上传 PDF 并构建知识库（P1）

用户上传图片型 PDF；系统调用 MinerU 在线服务完成 OCR/结构化转换；再进行文档清洗、图表 JSON 抽取、切块与索引，最终得到可用于 RAG 的知识库。

**为何是该优先级**: 第一阶段业务闭环的入口，直接决定后续 RAG/生成质量。  
**独立测试**: 以小样 PDF 触发 ingest 任务，能看到中间态产物与知识库 chunk/向量入库。  
**验收场景**:
1. 假设用户上传 PDF，当创建 ingest 任务，则任务最终成功并产出可检索知识库
2. 假设 MinerU 服务超时，当 ingest 执行，则任务失败并返回统一错误码与 request_id

---

### 用户故事 2 - 管理文档/模板/中间态/目标文件（P1）

用户（中台管理员）可以对源文件、模板文件、目标文件进行增删改查；中间态文件可观测与删除；源文件支持归档。

**为何是该优先级**: 治理能力是平台可用性的基础。  
**独立测试**: 通过 API 完成 CRUD 与归档，验证列表/详情/删除行为。  
**验收场景**:
1. 假设存在源文件，当归档后，则列表中状态为 archived 且默认查询可过滤
2. 假设存在中间态工件，当删除工件，则工件不可再下载且元数据被标记删除

---

### 用户故事 3 - 使用模板生成单 HTML 目标文件（P1）

用户选择模板与知识库，系统使用混合检索（BM25+向量，可扩展 rerank）召回上下文，推理 LLM 生成长文内容，并严格遵守模板格式，输出单 HTML。

**为何是该优先级**: 第一阶段直接交付的核心价值。  
**独立测试**: 以固定模板生成目标 HTML，验证模板骨架未被改写、内容完整。  
**验收场景**:
1. 假设模板被锁定，当生成目标文件，则输出严格遵守模板结构
2. 假设知识库为空，当生成目标文件，则任务失败并明确提示缺少可用知识库内容

---

## 需求

### 功能性需求

- **FR-001**: 系统必须提供源文件管理（PDF）CRUD，并支持归档/取消归档
- **FR-002**: 系统必须提供模板文件管理 CRUD；模板分为 custom/fixed；支持锁定（锁定后不得变更模板内容）
- **FR-003**: 系统必须提供中间态文件的观测与删除能力
- **FR-004**: 系统必须提供目标文件（单 HTML）的查询与下载能力
- **FR-005**: 系统必须支持 LLM 统一配置、按能力动态选择（LangChain 1.0），支持 openai-compatible / Gemini / Ollama / vLLM / GPUStack / FlagEmbedding 等
- **FR-006**: 系统必须提供提示词管理能力（可观测、编辑、版本化、启用/停用）
- **FR-007**: 系统必须调用 MinerU 在线服务完成图片型 PDF OCR/结构化转换
- **FR-008**: 系统必须基于 OCR 产物完成文档清洗（去广告/无意义信息/目录等）以满足 RAG 构建
- **FR-009**: 系统必须对保留图表调用多模态模型完成图转 JSON
- **FR-010**: 系统必须实现切块：结构 → 语义 → 句子 → 长度顺序切块
- **FR-011**: 系统必须建设知识库：SQLite 元数据 + BM25 + ChromaDB 向量库，支持混合检索
- **FR-012**: 系统必须提供“图表 JSON → 绘制美观图表文件”的原子能力
- **FR-013**: 系统必须通过任务系统异步执行 ingest/generate 等长任务，并可查询状态

### 关键实体

- **SourceFile**: 源 PDF 文件元数据（状态、存储路径、hash、归档标记）
- **Template**: 模板文件（custom/fixed、版本、锁定标记、存储路径）
- **IntermediateArtifact**: 中间态产物（类型枚举、来源、可删除标记、存储路径）
- **TargetFile**: 目标 HTML 文件（模板/知识库/任务关联、存储路径）
- **LLMProvider/LLMModel/Capability**: Provider/模型配置与业务能力映射
- **Prompt**: 提示词（scope、content、version、active）
- **KnowledgeBase/Chunk**: 知识库与切块元数据（向量/文本/结构路径）
- **Job**: 异步任务（kind/status/progress/error、request_id 贯穿）

---

## API

> 本章节是 API 的唯一归档（REST + 任务 + 配置 + schema）。统一前缀：`/v1`。

### REST API（第一阶段）

```yaml
# Health
- GET    /v1/health

# Jobs（异步任务）
- POST   /v1/jobs                 # 创建任务（ingest/generate/render_chart）
- GET    /v1/jobs/{job_id}        # 查询任务状态

# Source files（源文件）
- POST   /v1/sources
- GET    /v1/sources
- GET    /v1/sources/{source_id}
- PATCH  /v1/sources/{source_id}
- DELETE /v1/sources/{source_id}
- POST   /v1/sources/{source_id}/archive
- POST   /v1/sources/{source_id}/unarchive

# Templates（模板）
- POST   /v1/templates
- GET    /v1/templates
- GET    /v1/templates/{template_id}
- PATCH  /v1/templates/{template_id}
- DELETE /v1/templates/{template_id}
- POST   /v1/templates/{template_id}/preprocess
- POST   /v1/templates/{template_id}/lock

# Intermediates（中间态）
- GET    /v1/intermediates
- GET    /v1/intermediates/{artifact_id}
- DELETE /v1/intermediates/{artifact_id}

# Targets（目标文件）
- GET    /v1/targets
- GET    /v1/targets/{target_id}
- GET    /v1/targets/{target_id}/download

# Generate（最终生成）
- POST   /v1/generate             # 创建生成任务（返回 job_id）

# LLM config
- GET    /v1/llm/providers
- POST   /v1/llm/providers
- PATCH  /v1/llm/providers/{provider_id}
- DELETE /v1/llm/providers/{provider_id}
- GET    /v1/llm/models
- POST   /v1/llm/models
- PATCH  /v1/llm/models/{model_id}
- DELETE /v1/llm/models/{model_id}
- GET    /v1/llm/capabilities
- PATCH  /v1/llm/capabilities

# Prompts
- GET    /v1/prompts
- POST   /v1/prompts
- PATCH  /v1/prompts/{prompt_id}
- DELETE /v1/prompts/{prompt_id}
```

### 错误响应（统一格式）

```json
{
  "error": {
    "code": "validation_error",
    "message": "request validation failed",
    "request_id": "string",
    "details": {}
  }
}
```

**版本**: 1.0.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-16

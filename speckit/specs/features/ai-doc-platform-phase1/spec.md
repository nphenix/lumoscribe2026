---
id: ai-doc-platform-phase1
status: IN_PROGRESS
created: 2026-01-16
updated: 2026-01-26
links:
  - ./plan.md
  - ./tasks.md
---

# 功能规格说明：第一阶段 AI 文档生成平台

|**功能分支**: `feature/phase1-ai-doc-platform`  
|**创建日期**: 2026-01-16  
|**状态**: 进行中  
|**输入**: "图片型 PDF 清洗后构建知识库，通过指定模板使用 RAG + LLM 生成规定格式长文件（单 HTML），并提供 API 与中台管理。"
|**配置方式**: LLM Provider 通过中台管理界面配置，运行时从数据库加载

---

## 用户场景与测试

### 用户故事 1 - 上传 PDF 并构建知识库（P1）

用户上传图片型 PDF；系统调用 MinerU 在线服务完成 OCR/结构化转换；再进行文档清洗、图表 JSON 抽取、切块与索引，最终得到可用于 RAG 的知识库。

|**为何是该优先级**: 第一阶段业务闭环的入口，直接决定后续 RAG/生成质量。  
|**独立测试**: 以小样 PDF 触发 ingest 任务，能看到中间态产物与知识库 chunk/向量入库。  
|**验收场景**:
1. 假设用户上传 PDF，当创建 ingest 任务，则任务最终成功并产出可检索知识库
2. 假设 MinerU 服务超时，当 ingest 执行，则任务失败并返回统一错误码与 request_id
3. 假设用户执行 MinerU 清洗测试脚本，则能验证 MinerU 服务连通性和结果正确性

---

### 用户故事 2 - 管理文档/模板/中间态/目标文件（P1）

用户（中台管理员）可以对源文件、模板文件、目标文件进行增删改查；中间态文件可观测与删除；源文件支持归档。

|**为何是该优先级**: 治理能力是平台可用性的基础。  
|**独立测试**: 通过 API 完成 CRUD 与归档，验证列表/详情/删除行为。  
|**验收场景**:
1. 假设存在源文件，当归档后，则列表中状态为 archived 且默认查询可过滤
2. 假设存在中间态工件，当删除工件，则工件不可再下载且元数据被标记删除

---

### 用户故事 3 - 使用模板生成单 HTML 目标文件（P1）

用户选择模板与知识库，系统使用混合检索（BM25+向量，可扩展 rerank）召回上下文，推理 LLM 生成长文内容，并严格遵守模板格式，输出单 HTML。**BM25 支持多索引预建加载**：查询阶段加载同 collection 的全部 BM25 索引并合并去重，确保覆盖与向量库一致。支持用户自定义大纲的 LLM 润色。

|**为何是该优先级**: 第一阶段直接交付的核心价值。  
|**独立测试**: 以固定模板生成目标 HTML，验证模板骨架未被改写、内容完整。  
|**验收场景**:
1. 假设模板被锁定，当生成目标文件，则输出严格遵守模板结构
2. 假设知识库为空，当生成目标文件，则任务失败并明确提示缺少可用知识库内容
3. 假设用户传入自定义大纲，当执行润色功能，则输出结构更清晰的大纲
4. 假设执行生成白皮书测试脚本，则能验证模板解析、RAG 检索、LLM 生成、图表渲染的完整性

---

### 用户故事 4 - 中台管理与观测（P1）

中台管理员通过可视化界面配置 LLM Provider/Model，管理提示词版本，并观测文档处理与生成任务的实时状态。

|**为何是该优先级**: 降低运维成本，提升系统可观测性。  
|**独立测试**: 在 Web 界面添加一个 Ollama Provider 并成功保存；查看一个正在运行的 Ingest 任务进度。  
|**验收场景**:
1. 假设新增一个 vLLM 模型配置，当保存后，则 Capability 映射中可选该模型
2. 假设 Ingest 任务失败，当在 Web 端查看，则能看到具体的错误日志与 Request ID
3. 假设配置 MinerU Provider，当测试连通性时，则能成功连接 MinerU 在线服务

---

## 需求

### 功能性需求

|- **FR-001**: 系统必须提供源文件管理（PDF）CRUD，并支持归档/取消归档
|- **FR-002**: 系统必须提供模板文件管理 CRUD；模板分为 custom/fixed；支持锁定（锁定后不得变更模板内容）
|- **FR-003**: 系统必须提供中间态文件的观测与删除能力
|- **FR-004**: 系统必须提供目标文件（单 HTML）的查询与下载能力
|- **FR-005**: 系统必须支持 LLM 统一配置、按能力动态选择（LangChain 1.0），支持 openai-compatible / Gemini / Ollama / vLLM / GPUStack / FlagEmbedding 等，涵盖推理、Embedding、Rerank、多模态、OCR 模型。**配置方式：通过中台管理界面动态配置**
|- **FR-006**: 系统必须提供提示词管理能力（可观测、编辑、版本化、启用/停用），通过中台进行统一管理
|- **FR-007**: 系统必须调用 MinerU 在线服务完成图片型 PDF OCR/结构化转换
|- **FR-008**: 系统必须基于 OCR 产物完成文档清洗（去广告/无意义信息/目录等）以满足 RAG 构建，使用推理 LLM 辅助
|- **FR-009**: 系统必须对保留图表调用多模态模型完成图转 JSON
|- **FR-010**: 系统必须实现切块：结构 → 语义 → 句子 → 长度顺序切块
|- **FR-011**: 系统必须建设知识库：SQLite 元数据 + BM25 + ChromaDB 向量库，支持混合检索（LlamaIndex）
|- **FR-012**: 系统必须提供"图表 JSON → 绘制美观图表文件"的原子能力，基于 JSON 动态绘制
|- **FR-013**: 系统必须通过任务系统异步执行 ingest/generate 等长任务，并可查询状态
|- **FR-014**: 系统必须提供中台管理界面（Web UI），覆盖文档管理、LLM 配置、提示词管理与任务观测（Next.js + Shadcn UI）
|- **FR-015**: 系统必须提供用户自定义大纲的 LLM 润色功能，提升大纲结构化程度

### 关键实体

|- **SourceFile**: 源 PDF 文件元数据（状态、存储路径、hash、归档标记）
|- **Template**: 模板文件（custom/fixed、版本、锁定标记、存储路径）
|- **IntermediateArtifact**: 中间态产物（类型枚举、来源、可删除标记、存储路径）
|- **TargetFile**: 目标 HTML 文件（模板/知识库/任务关联、存储路径）
|- **LLMProvider/LLMModel/Capability**: Provider/模型配置与业务能力映射
|- **Prompt**: 提示词（scope、content、version、active）
|- **KnowledgeBase/Chunk**: 知识库与切块元数据（向量/文本/结构路径）
|- **Job**: 异步任务（kind/status/progress/error、request_id 贯穿）
|- **LLMCallSite**: LLM 调用点配置（绑定 Provider、能力、提示词范围、参数覆盖）

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

# Ingest（文档摄入）- T036
- POST   /v1/ingest               # 创建摄入任务（单文档/批量）
- GET    /v1/ingest/{job_id}      # 查询摄入任务状态

# Generate（内容生成）- T044
- POST   /v1/generate             # 创建生成任务（返回 job_id）
- GET    /v1/generate/{job_id}    # 查询生成任务状态
- POST   /v1/generate/polish-outline # 大纲润色（直接返回结果）

# LLM config
- GET    /v1/llm/providers
- POST   /v1/llm/providers
- PATCH  /v1/llm/providers/{provider_id}
- DELETE /v1/llm/providers/{provider_id}
- GET    /v1/llm/capabilities
- PATCH  /v1/llm/capabilities
- GET    /v1/llm/call-sites
- POST   /v1/llm/call-sites
- PATCH  /v1/llm/call-sites/{call_site_id}
- DELETE /v1/llm/call-sites/{call_site_id}

# Prompts
- GET    /v1/prompts
- POST   /v1/prompts
- PATCH  /v1/prompts/{prompt_id}
- DELETE /v1/prompts/{prompt_id}
```

### 测试脚本 API（非生产）

以下 API 仅用于测试脚本验证，不属于生产 API：

```yaml
# Test APIs（测试用）
- POST   /v1/test/mineru/ingest   # 测试 MinerU 清洗（直接调用服务层）
- POST   /v1/test/chart-extract   # 测试图转 JSON
- POST   /v1/test/kb/build        # 测试知识库构建
- POST   /v1/test/generate        # 测试内容生成
- POST   /v1/test/polish-outline  # 测试大纲润色
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

### LLM 配置规范

#### Provider 配置（通过中台管理界面配置）

```json
{
  "name": "openai-compatible",
  "type": "openai-compatible",
  "config_json": {
    "base_url": "https://api.example.com/v1",
    "api_key": "sk-xxx",
    "model": "gpt-4o",
    "max_tokens": 4096,
    "temperature": 0.7
  },
  "enabled": true
}
```

#### CallSite 配置（绑定 Provider）

```json
{
  "name": "doc_cleaning:clean_text",
  "provider_id": 1,
  "expected_model_kind": "chat",
  "prompt_scope": "doc_cleaning:clean_text",
  "enabled": true,
  "parameters": {
    "temperature": 0.3,
    "max_tokens": 2048
  }
}
```

---

## 测试规格

### 测试范围

| 测试编号 | 测试内容 | 测试方式 | 依赖 |
|---------|---------|---------|------|
| T093 | MinerU 清洗功能 | 测试脚本 | MinerU 在线服务连接 |
| T094 | 图转 JSON 功能 | 测试脚本 | 多模态 LLM 模型 |
| T095 | 知识库构建功能 | 测试脚本 | ChromaDB、Embedding 模型 |
| T096 | 生成白皮书功能 | 测试脚本 | 模板文件、知识库 |

### 测试通过标准

|- 每个测试脚本独立运行成功
|- 测试输出包含详细的验证结果
|- 错误信息清晰可追踪
|- 测试数据可复现

---

|**版本**: 1.2.1 | **创建**: 2026-01-16 | **最后更新**: 2026-01-24

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
- [x] T002 [P0] 初始化 Python 项目依赖（`pyproject.toml`），并提供 `env.example`
- [x] T003 [P0] 建立 FastAPI 最小服务（`/v1/health`）与统一错误响应格式
- [x] T004 [P0] 建立 SQLite 元数据与 `jobs` 表（最小字段）
- [x] T005 [P0] 建立 Celery + Redis Worker（占位任务）并接入 `/v1/jobs` 创建即入队
- [ ] T006 [P1] 规范化安装与运行命令（README：安装依赖、启动 API/Worker/Redis）

---

## 阶段2：文档/模板/目标文件管理（US2）

**目标**: 完成 CRUD + 归档 + 中间态观测/删除，形成平台治理能力基础。

- [ ] T010 [P1] [US2] 设计并实现 `source_files/templates/target_files/intermediate_artifacts` 数据表
- [ ] T011 [P1] [US2] 实现源文件上传/查询/更新/删除/归档 API（含文件落盘）
- [ ] T012 [P1] [US2] 实现模板上传/CRUD/锁定/预处理校验 API（预处理不使用 LLM）
- [ ] T013 [P2] [US2] 实现中间态列表/详情/删除 API
- [ ] T014 [P2] [US2] 实现目标文件查询/下载 API

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

**版本**: 1.0.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-16

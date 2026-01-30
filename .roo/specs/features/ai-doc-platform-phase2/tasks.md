---
template: Tasks
version: 1.1.0
lastUpdated: 2026-01-29
charter: ../constitution.md
---

# 任务：第二阶段 AI 文档生成平台

**输入**: 第二阶段实施计划（plan.md）、功能规格（spec.md）  
**先决条件**: `speckit/specs/features/ai-doc-platform-phase2/spec.md`、`plan.md`

---

## 阶段1：设置与基线校验（共享基础设施）

**目的**: 确认现有第一阶段能力稳定，为第二阶段改造建立基线。

- [x] T201 [P1] [P] [US1] 回归执行现有白皮书生成测试脚本，记录当前 Html 生成质量基线（重复率、空章节情况、图表漂移情况）  
  - 技术栈：pytest、现有 FastAPI 服务、LangChain/LlamaIndex 内容生成与检索链路

  **相关文件**:
  | 文件 | 用途 |
  |------|------|
  | [test_content_generation.py](../../../../tests/test_content_generation.py) | 白皮书生成 E2E 回归入口（T096） |
  | [t201-whitepaper-baseline.py](../../../../scripts/t201-whitepaper-baseline.py) | 基线统计脚本（重复段落/空章节/图表锚点缺失与可疑位置） |
  | [阶段1-白皮书生成质量基线.md](../../../../docs/process/ai-doc-platform-phase2/阶段1-白皮书生成质量基线.md) | 基线口径与运行方式说明 |
- [x] T202 [P1] [P] [US2] 抽样分析 `pic_to_json/chart_json` 与 Html 中图表位置的对应关系，形成现状文档（仅在 docs/process 中记录）  
  - 技术栈：Python（数据分析脚本）、现有 chart_json 规范与 Html 渲染结果

  **相关文件**:
  | 文件 | 用途 |
  |------|------|
  | [t202-chart-json-html-mapping.py](../../../../scripts/t202-chart-json-html-mapping.py) | 抽样导出 chart_id 在 HTML 中的章节位置，并反查 chart_json 路径 |
  | [阶段1-chart_json-与-HTML-图表位置对应关系.md](../../../../docs/process/ai-doc-platform-phase2/阶段1-chart_json-与-HTML-图表位置对应关系.md) | 现状链路与抽样分析方法 |
- [x] T203 [P1] [P] [US3] 盘点现有 LLM Provider/CallSite/Prompt 配置与中台页面，列出第二阶段涉及的调用点清单  
  - 技术栈：FastAPI + SQLAlchemy（Provider/CallSite/Prompt API）、Next.js 中台界面、LLMRuntimeService

  **相关文件**:
  | 文件 | 用途 |
  |------|------|
  | [llm.py](../../../../src/interfaces/api/routes/llm.py) | Provider/CallSite/Capability 配置 API |
  | [prompts.py](../../../../src/interfaces/api/routes/prompts.py) | Prompt 管理 API |
  | [llm_runtime_service.py](../../../../src/application/services/llm_runtime_service.py) | 运行时按 CallSite 构建模型的封装 |
  | [阶段1-LLM-Provider-CallSite-Prompt-盘点.md](../../../../docs/process/ai-doc-platform-phase2/阶段1-LLM-Provider-CallSite-Prompt-盘点.md) | 现状盘点与调用点清单 |

---

## 阶段2：Html 生成链路重构（内容链）

**目的**: 将单阶段 Html 生成逻辑重构为“章节生成 + 后处理 + 润色”的多阶段流水线。

### 测试与观测基础

- [x] T204 [P1] [P] [US1] 在 `tests/integration/` 中新增或扩展白皮书生成集成测试，捕获章节结构、重复段落与图表占位符布局  
  - 技术栈：pytest、FastAPI 测试客户端、现有 LangChain/LlamaIndex 生成链路
  
  **相关文件**:
  | 文件 | 用途 |
  |------|------|
  | [test_whitepaper_generation_pipeline.py](../../../../tests/integration/test_whitepaper_generation_pipeline.py) | 阶段2集成测试：章节结构/重复段落/图表占位符布局 |
  | [00-目录说明.md](../../../../tests/00-目录说明.md) | tests 目录索引更新 |

### 实施

- [x] T205 [P0] [US1] 在 `src/application/services/content_generation/` 下抽象章节生成管线（如 SectionGenerationPipeline），拆分章节级生成、后处理与润色阶段  
  - 技术栈：Python、现有 ContentGenerationService 模块、LangChain（SectionLLMGenerator/SectionStructuredGenerator）、LlamaIndex（上下文组装）
  
  **相关文件**:
  | 文件 | 用途 |
  |------|------|
  | [pipeline.py](../../../../src/application/services/content_generation/pipeline.py) | 章节生成多阶段管线（生成→后处理→可选润色） |
  | [postprocessors.py](../../../../src/application/services/content_generation/postprocessors.py) | 章节后处理工具（语义去重/空条目清理） |
  | [00-目录说明.md](../../../../src/application/services/content_generation/00-目录说明.md) | content_generation 目录索引更新 |

- [x] T206 [P0] [US1] 重构 `ContentGenerationService`，将现有 Html 生成逻辑迁移到新管线实现，确保对外 `/v1/jobs` 与 `/v1/targets` 接口保持兼容  
  - 技术栈：FastAPI、ContentGenerationService 导出层、现有任务系统与 TargetFileService
  
  **相关文件**:
  | 文件 | 用途 |
  |------|------|
  | [service.py](../../../../src/application/services/content_generation/service.py) | generate_content 迁移到管线编排，新增章节润色入口 |
  | [targets.py](../../../../src/interfaces/api/routes/targets.py) | 新增可选 polish_sections 参数并透传 |
  | [content_generation.py](../../../../src/application/schemas/content_generation.py) | /v1/generate 请求新增 polish_sections |
  | [generate.py](../../../../src/interfaces/api/routes/generate.py) | 透传 polish_sections 到服务层 |
  | [config.py](../../../../src/shared/config.py) | 白皮书生成默认参数：去重/清理/润色开关与阈值 |

- [x] T207 [P1] [P] [US1] 实现基于向量检索的语义去重工具，针对章节内与跨章节重复内容进行裁剪与合并（不改写事实）  
  - 技术栈：LlamaIndex/向量检索服务、现有向量存储服务（VectorStorageService）、Python 文本相似度处理
- [x] T208 [P1] [P] [US1] 实现空章节与“纲要残影”检测工具，消除仅有标题或纲要条目、无正文的小节  
  - 技术栈：Python（text_utils/Html 解析）、现有 Html 渲染结果
- [x] T209 [P1] [US1] 接入章节级语言润色阶段，限制只做措辞与衔接优化，不允许新增外部事实  
  - 技术栈：LangChain（LLM 调用链）、LLMRuntimeService、PromptService
- [x] T210 [P1] [P] [US1] 为新管线增加详细日志与覆盖率统计，便于对比第一阶段与第二阶段的生成质量  
  - 技术栈：现有结构化日志体系、coverage/统计脚本、FastAPI 任务日志
  
  **相关文件**:
  | 文件 | 用途 |
  |------|------|
  | [prompts.py](../../../../src/application/services/content_generation/prompts.py) | 新增章节润色提示词种子（content_generation:polish_section） |
  | [callsites.py](../../../../src/application/services/content_generation/callsites.py) | 新增章节润色与结构化生成 callsite 种子 |
  | [prompts.py](../../../../src/shared/constants/prompts.py) | 提示词注册表导出新 Scope |

---

## 阶段3：数据可视化模块重构（AntV 栈）

**目的**: 引入 AntV G2/InfoGraphic/S2，统一统计图、信息图与分析型表格的渲染能力与视觉风格。

### 测试与示例准备

- [x] T211 [P1] [P] [US2] 在 `tests/` 下整理 sankey、stacked_area、table 等典型 chart_json 示例，用于可视化重构回归测试  
  - 技术栈：Python/JSON 处理、现有 pic_to_json 输出规范

### 实施

- [x] T212 [P0] [US2] 设计并实现 `chart_json -> AntV G2/InfoGraphic/S2` 的映射层，将现有 `chart_type` 与字段映射为对应的 AntV 配置或 DSL，约定 `chart_id = chart_json` 文件名（去掉 `.json`）  
  - 技术栈：AntV G2、AntV InfoGraphic、AntV S2、TypeScript/JavaScript（可视化配置）、Python/TS 映射层
- [x] T213 [P1] [P] [US2] 在中台与后端封装 AntV 渲染服务，提供统一接口：输入 chart_json，输出 SVG/PNG 文件，并就地保存到 `data/intermediates/{id}/pic_to_json/chart_json/` 目录，文件名包含 `chart_id` 与当前主题（例如 `<chart_id>__whitepaper-default.svg`）  
  - 技术栈：AntV 可视化引擎（G2/InfoGraphic/S2）、Node.js/浏览器渲染环境、可能的无头浏览器（如 Playwright）用于 SVG/PNG 导出、FastAPI/任务系统
- [x] T214 [P1] [US2] 设计并实现“白皮书主题”，在 G2、InfoGraphic、S2 中应用  
  - 技术栈：AntV 主题系统、设计规范（颜色/字体/线条）
- [x] T215 [P1] [P] [US2] 在 `data/intermediates/{id}/pic_to_json/chart_json/` 目录中维护轻量级渲染元信息索引（如 `render_meta.json`），记录每个 `chart_id` 的 `json_hash`、主题、渲染版本与图片文件名，用于增量渲染决策  
  - 技术栈：Python/JSON 处理、文件系统操作
- [x] T216 [P1] [P] [US2] 将 Html 中现有图表渲染片段替换为预渲染静态图片的引用，在 Html 渲染阶段根据 `[Chart:chart_id]` 占位符与对应 intermediates 路径查找图片文件，并回退到清晰占位符与日志记录以处理缺失情况  
  - 技术栈：现有 HtmlRenderer（WhitepaperHtmlRenderer）、文件路径解析、FastAPI 目标文件写入

**相关文件**:
| 文件 | 用途 |
|------|------|
| [spec_mapper.py](../../../../src/application/services/antv_rendering/spec_mapper.py) | `chart_json -> AntV(G2/S2)` 映射层（含白皮书主题默认值） |
| [service.py](../../../../src/application/services/antv_rendering/service.py) | AntV 预渲染服务：输出 SVG/PNG + 增量渲染 |
| [render_meta.py](../../../../src/application/services/antv_rendering/render_meta.py) | `render_meta.json` 索引读写与哈希 |
| [render.mjs](../../../../scripts/antv/render.mjs) | Node 渲染入口（G2/S2/Infographic） |
| [antv_render.py](../../../../src/interfaces/api/routes/antv_render.py) | 预渲染 API：输入 pic_to_json，输出就地静态文件 |
| [app.py](../../../../src/interfaces/api/app.py) | 注册 AntV 预渲染路由 |
| [service.py](../../../../src/application/services/content_generation/service.py) | 生成时按 `[Chart:...]` 占位符触发增量预渲染 |
| [html_renderer.py](../../../../src/application/services/content_generation/html_renderer.py) | HTML 渲染时用静态图片替换图表占位符 |
| [auto_chart_resolver.py](../../../../src/application/services/content_generation/auto_chart_resolver.py) | `chart_id` 统一为 chart_json 文件名（stem） |
| [knowledge_base_service.py](../../../../src/application/services/knowledge_base_service.py) | 知识库 chart_id 统一为 chart_json 文件名（stem） |
| [package.json](../../../../package.json) | Node 依赖：@antv/g2、@antv/s2、@antv/infographic、playwright |
| [test_antv_spec_mapper_unit.py](../../../../tests/test_antv_spec_mapper_unit.py) | 映射层回归测试 |
| [chart_json fixtures](../../../../tests/fixtures/chart_json/00-目录说明.md) | sankey/stacked_area/table 样例 |

---

## 阶段4：中台 LLM 并发与 Prompt 管理增强

**目的**: 通过中台配置实现对 LLM 并发与提示词版本的统一治理。

### 测试与观测基础

- [x] T217 [P1] [P] [US3] 为 LLM 并发控制与 Prompt 管理增加基础集成测试，覆盖 Provider/CallSite/Prompt 的 CRUD 与配置生效路径  
  - 技术栈：pytest、FastAPI 测试客户端、现有 LLMProvider/LLMCallSite/Prompt API

### 实施

- [x] T218 [P0] [US3] 在 Provider/CallSite 的后端模型与前端配置页面中增加并发相关字段，并更新中台界面进行配置与展示  
  - 技术栈：SQLAlchemy（模型与迁移）、FastAPI（配置 API）、Next.js 中台界面
- [x] T219 [P1] [US3] 在 `LLMRuntimeService` 与相关业务服务中实现并发策略读取与执行逻辑，确保不会超过配置的并发上限  
  - 技术栈：LLMRuntimeService、LangChain（底层 LLM 调用）、Python 并发控制（asyncio）
- [x] T220 [P1] [P] [US3] 扩展 Prompt 管理界面与后端接口，支持提示词版本列表、差异查看、启用/停用与回滚  
  - 技术栈：PromptRepository、FastAPI Prompt API、Next.js Prompt 管理页面
- [x] T221 [P1] [US3] 确保第二阶段新增或改造的所有 LLM 调用点均通过统一的 Provider/CallSite/Prompt 配置路径访问  
  - 技术栈：LLMRuntimeService、调用点配置（CallSite）、现有 LangChain 集成

### 相关文件

| 文件 | 用途 |
|------|------|
| [llm_provider.py](../../../../src/domain/entities/llm_provider.py) | Provider 模型新增并发字段 |
| [llm_call_site.py](../../../../src/domain/entities/llm_call_site.py) | CallSite 模型新增并发字段 |
| [db.py](../../../../src/shared/db.py) | SQLite 轻量迁移新增并发列 |
| [llm.py](../../../../src/application/schemas/llm.py) | LLM 配置 API schema 增加并发字段 |
| [llm.py](../../../../src/interfaces/api/routes/llm.py) | LLM 配置 API 贯通并发字段 |
| [llm_runtime_service.py](../../../../src/application/services/llm_runtime_service.py) | 并发策略解析与门控（acquire_llm_slot） |
| [config.py](../../../../src/shared/config.py) | 默认并发配置项 |
| [providers/page.tsx](../../../../src/interfaces/admin-web/app/llm/providers/page.tsx) | Provider 并发字段配置与展示 |
| [call-sites/page.tsx](../../../../src/interfaces/admin-web/app/llm/call-sites/page.tsx) | CallSite 并发字段配置与展示 |
| [use-llm.ts](../../../../src/interfaces/admin-web/hooks/use-llm.ts) | 前端类型与请求体新增并发字段 |
| [llm_section.py](../../../../src/application/services/content_generation/llm_section.py) | 章节生成接入并发门控 |
| [section_structured.py](../../../../src/application/services/content_generation/section_structured.py) | 结构化生成接入并发门控 |
| [outline_polish_service.py](../../../../src/application/services/outline_polish/outline_polish_service.py) | 大纲润色接入并发门控 |
| [document_cleaning_service.py](../../../../src/application/services/document_cleaning_service.py) | 文档清洗并发读取与门控接入 |
| [prompt.py](../../../../src/application/schemas/prompt.py) | Prompt scope/diff 响应 schema |
| [prompt_repository.py](../../../../src/application/repositories/prompt_repository.py) | Prompt scope 聚合查询 |
| [prompts.py](../../../../src/interfaces/api/routes/prompts.py) | Prompt scopes/diff/get 接口 |
| [use-prompts.ts](../../../../src/interfaces/admin-web/hooks/use-prompts.ts) | Prompt hooks 支持 scopes/diff/patch |
| [prompts/page.tsx](../../../../src/interfaces/admin-web/app/prompts/page.tsx) | Prompt 版本列表/差异/回滚 UI |
| [conftest.py](../../../../tests/conftest.py) | Async 测试客户端与全量 API 模式 |
| [test_llm_config_api.py](../../../../tests/test_llm_config_api.py) | 阶段4 并发与 Prompt 集成测试 |

---

## 阶段5：完善与横切关注点

**目的**: 统一质量标准、补齐文档与长期可维护性。

- [x] T222 [P2] 更新 `docs/process/# 报告生成链路重构讨论记录（阶段一）.md` 及相关文档，反映第二阶段最终架构与实现决策  
  - 技术栈：Markdown 文档体系、现有 speckit 规范
- [x] T223 [P2] 在 `speckit/specs/changes/` 下记录与第二阶段相关的关键变更（Html 生成链、可视化栈、中台能力）  
  - 技术栈：change-manager 模板与规范
- [x] T224 [P3] [P] 针对重构后的模块进行必要的代码清理与重构，消除遗留 TODO/FIXME 与重复逻辑  
  - 技术栈：Python/TypeScript 代码库、项目章程中 P1 代码质量与 P8 禁止模式规范
- [x] T225 [P2] 对典型白皮书生成流程进行性能与稳定性评估，记录结论并给出后续优化建议  
  - 技术栈：pytest/基准测试脚本、现有日志与监控手段、LlamaIndex/AntV 渲染链路

### 相关文件

| 文件 | 用途 |
|------|------|
| [# 报告生成链路重构讨论记录（阶段一）.md](../../../../docs/process/%23%20%E6%8A%A5%E5%91%8A%E7%94%9F%E6%88%90%E9%93%BE%E8%B7%AF%E9%87%8D%E6%9E%84%E8%AE%A8%E8%AE%BA%E8%AE%B0%E5%BD%95%EF%BC%88%E9%98%B6%E6%AE%B5%E4%B8%80%EF%BC%89.md) | 阶段一讨论与阶段二落地事实对照补全 |
| [00-目录说明.md](../../../../docs/process/ai-doc-platform-phase2/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md) | 第二阶段过程文档目录索引（含阶段五评估文档） |
| [阶段1-白皮书生成质量基线.md](../../../../docs/process/ai-doc-platform-phase2/%E9%98%B6%E6%AE%B51-%E7%99%BD%E7%9A%AE%E4%B9%A6%E7%94%9F%E6%88%90%E8%B4%A8%E9%87%8F%E5%9F%BA%E7%BA%BF.md) | 基线文档补充对照链接与 uv 运行方式 |
| [阶段1-chart_json-与-HTML-图表位置对应关系.md](../../../../docs/process/ai-doc-platform-phase2/%E9%98%B6%E6%AE%B51-chart_json-%E4%B8%8E-HTML-%E5%9B%BE%E8%A1%A8%E4%BD%8D%E7%BD%AE%E5%AF%B9%E5%BA%94%E5%85%B3%E7%B3%BB.md) | 图表锚点/注入链路对照链接补充 |
| [阶段1-LLM-Provider-CallSite-Prompt-盘点.md](../../../../docs/process/ai-doc-platform-phase2/%E9%98%B6%E6%AE%B51-LLM-Provider-CallSite-Prompt-%E7%9B%98%E7%82%B9.md) | LLM 中台盘点文档对照链接补充 |
| [阶段5-白皮书生成性能与稳定性评估.md](../../../../docs/process/ai-doc-platform-phase2/%E9%98%B6%E6%AE%B55-%E7%99%BD%E7%9A%AE%E4%B9%A6%E7%94%9F%E6%88%90%E6%80%A7%E8%83%BD%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7%E8%AF%84%E4%BC%B0.md) | 阶段五评估口径、脚本与结论沉淀 |
| [t225-whitepaper-perf.py](../../../../scripts/t225-whitepaper-perf.py) | 白皮书生成性能评估脚本（输出 JSON+Markdown） |
| [2026-01-30-ai-doc-platform-phase2-key-changes.md](../../changes/2026-01/2026-01-30-ai-doc-platform-phase2-key-changes.md) | 第二阶段关键变更记录 |
| [00-目录说明.md](../../changes/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md) | changes 目录索引更新 |
| [00-目录说明.md](../../changes/2026-01/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md) | 2026-01 变更目录索引更新 |
| [document_cleaning_service.py](../../../../src/application/services/document_cleaning_service.py) | 修复清洗/图表写入方法覆盖并补齐 workspace_id 事实读取 |
| [llm_provider_service.py](../../../../src/application/services/llm_provider_service.py) | Provider 删除前绑定检查 |
| [targets.py](../../../../src/interfaces/api/routes/targets.py) | target 详情补全 template_name/kb_name |
| [use-documents.ts](../../../../src/interfaces/admin-web/hooks/use-documents.ts) | 清理前端遗留 TODO |

---

## 依赖关系与执行顺序（概要）

| 阶段 | 依赖 | 说明 |
|------|------|------|
| 阶段1：设置与基线 | 无 | 可立即开始，用于建立对比基线 |
| 阶段2：Html 重构 | 阶段1 完成基线记录 | 阻塞后续可视化与中台增强验收 |
| 阶段3：可视化重构 | 阶段1 完成基线记录 | 与阶段2可部分并行，但最终验收依赖新 Html 链路稳定 |
| 阶段4：中台增强 | 阶段1 完成清单盘点 | 可与阶段2/3 并行推进，注意避免引入额外不稳定因素 |
| 阶段5：完善 | 所有主要阶段完成 | 最后执行，用于文档、变更记录与整体评估 |

---

**版本**: 1.1.0 | **创建**: 2026-01-29

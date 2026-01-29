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

- [ ] T201 [P1] [P] [US1] 回归执行现有白皮书生成测试脚本，记录当前 Html 生成质量基线（重复率、空章节情况、图表漂移情况）  
  - 技术栈：pytest、现有 FastAPI 服务、LangChain/LlamaIndex 内容生成与检索链路
- [ ] T202 [P1] [P] [US2] 抽样分析 `pic_to_json/chart_json` 与 Html 中图表位置的对应关系，形成现状文档（仅在 docs/process 中记录）  
  - 技术栈：Python（数据分析脚本）、现有 chart_json 规范与 Html 渲染结果
- [ ] T203 [P1] [P] [US3] 盘点现有 LLM Provider/CallSite/Prompt 配置与中台页面，列出第二阶段涉及的调用点清单  
  - 技术栈：FastAPI + SQLAlchemy（Provider/CallSite/Prompt API）、Next.js 中台界面、LLMRuntimeService

---

## 阶段2：Html 生成链路重构（内容链）

**目的**: 将单阶段 Html 生成逻辑重构为“章节生成 + 后处理 + 润色”的多阶段流水线。

### 测试与观测基础

- [ ] T204 [P1] [P] [US1] 在 `tests/integration/` 中新增或扩展白皮书生成集成测试，捕获章节结构、重复段落与图表占位符布局  
  - 技术栈：pytest、FastAPI 测试客户端、现有 LangChain/LlamaIndex 生成链路

### 实施

- [ ] T205 [P0] [US1] 在 `src/application/services/content_generation/` 下抽象章节生成管线（如 SectionGenerationPipeline），拆分章节级生成、后处理与润色阶段  
  - 技术栈：Python、现有 ContentGenerationService 模块、LangChain（SectionLLMGenerator/SectionStructuredGenerator）、LlamaIndex（上下文组装）
- [ ] T206 [P0] [US1] 重构 `ContentGenerationService`，将现有 Html 生成逻辑迁移到新管线实现，确保对外 `/v1/jobs` 与 `/v1/targets` 接口保持兼容  
  - 技术栈：FastAPI、ContentGenerationService 导出层、现有任务系统与 TargetFileService
- [ ] T207 [P1] [P] [US1] 实现基于向量检索的语义去重工具，针对章节内与跨章节重复内容进行裁剪与合并（不改写事实）  
  - 技术栈：LlamaIndex/向量检索服务、现有向量存储服务（VectorStorageService）、Python 文本相似度处理
- [ ] T208 [P1] [P] [US1] 实现空章节与“纲要残影”检测工具，消除仅有标题或纲要条目、无正文的小节  
  - 技术栈：Python（text_utils/Html 解析）、现有 Html 渲染结果
- [ ] T209 [P1] [US1] 接入章节级语言润色阶段，限制只做措辞与衔接优化，不允许新增外部事实  
  - 技术栈：LangChain（LLM 调用链）、LLMRuntimeService、PromptService
- [ ] T210 [P1] [P] [US1] 为新管线增加详细日志与覆盖率统计，便于对比第一阶段与第二阶段的生成质量  
  - 技术栈：现有结构化日志体系、coverage/统计脚本、FastAPI 任务日志

---

## 阶段3：数据可视化模块重构（AntV 栈）

**目的**: 引入 AntV G2/InfoGraphic/S2，统一统计图、信息图与分析型表格的渲染能力与视觉风格。

### 测试与示例准备

- [ ] T211 [P1] [P] [US2] 在 `tests/` 下整理 sankey、stacked_area、table 等典型 chart_json 示例，用于可视化重构回归测试  
  - 技术栈：Python/JSON 处理、现有 pic_to_json 输出规范

### 实施

- [ ] T212 [P0] [US2] 设计并实现 `chart_json -> AntV G2/InfoGraphic/S2` 的映射层，将现有 `chart_type` 与字段映射为对应的 AntV 配置或 DSL，约定 `chart_id = chart_json` 文件名（去掉 `.json`）  
  - 技术栈：AntV G2、AntV InfoGraphic、AntV S2、TypeScript/JavaScript（可视化配置）、Python/TS 映射层
- [ ] T213 [P1] [P] [US2] 在中台与后端封装 AntV 渲染服务，提供统一接口：输入 chart_json，输出 SVG/PNG 文件，并就地保存到 `data/intermediates/{id}/pic_to_json/chart_json/` 目录，文件名包含 `chart_id` 与当前主题（例如 `<chart_id>__whitepaper-default.svg`）  
  - 技术栈：AntV 可视化引擎（G2/InfoGraphic/S2）、Node.js/浏览器渲染环境、可能的无头浏览器（如 Playwright）用于 SVG/PNG 导出、FastAPI/任务系统
- [ ] T214 [P1] [US2] 设计并实现“白皮书主题”，在 G2、InfoGraphic、S2 中应用  
  - 技术栈：AntV 主题系统、设计规范（颜色/字体/线条）
- [ ] T215 [P1] [P] [US2] 在 `data/intermediates/{id}/pic_to_json/chart_json/` 目录中维护轻量级渲染元信息索引（如 `render_meta.json`），记录每个 `chart_id` 的 `json_hash`、主题、渲染版本与图片文件名，用于增量渲染决策  
  - 技术栈：Python/JSON 处理、文件系统操作
- [ ] T216 [P1] [P] [US2] 将 Html 中现有图表渲染片段替换为预渲染静态图片的引用，在 Html 渲染阶段根据 `[Chart:chart_id]` 占位符与对应 intermediates 路径查找图片文件，并回退到清晰占位符与日志记录以处理缺失情况  
  - 技术栈：现有 HtmlRenderer（WhitepaperHtmlRenderer）、文件路径解析、FastAPI 目标文件写入

---

## 阶段4：中台 LLM 并发与 Prompt 管理增强

**目的**: 通过中台配置实现对 LLM 并发与提示词版本的统一治理。

### 测试与观测基础

- [ ] T217 [P1] [P] [US3] 为 LLM 并发控制与 Prompt 管理增加基础集成测试，覆盖 Provider/CallSite/Prompt 的 CRUD 与配置生效路径  
  - 技术栈：pytest、FastAPI 测试客户端、现有 LLMProvider/LLMCallSite/Prompt API

### 实施

- [ ] T218 [P0] [US3] 在 Provider/CallSite 的后端模型与前端配置页面中增加并发相关字段，并更新中台界面进行配置与展示  
  - 技术栈：SQLAlchemy（模型与迁移）、FastAPI（配置 API）、Next.js 中台界面
- [ ] T219 [P1] [US3] 在 `LLMRuntimeService` 与相关业务服务中实现并发策略读取与执行逻辑，确保不会超过配置的并发上限  
  - 技术栈：LLMRuntimeService、LangChain（底层 LLM 调用）、Python 并发控制（asyncio）
- [ ] T220 [P1] [P] [US3] 扩展 Prompt 管理界面与后端接口，支持提示词版本列表、差异查看、启用/停用与回滚  
  - 技术栈：PromptRepository、FastAPI Prompt API、Next.js Prompt 管理页面
- [ ] T221 [P1] [US3] 确保第二阶段新增或改造的所有 LLM 调用点均通过统一的 Provider/CallSite/Prompt 配置路径访问  
  - 技术栈：LLMRuntimeService、调用点配置（CallSite）、现有 LangChain 集成

---

## 阶段5：完善与横切关注点

**目的**: 统一质量标准、补齐文档与长期可维护性。

- [ ] T222 [P2] 更新 `docs/process/# 报告生成链路重构讨论记录（阶段一）.md` 及相关文档，反映第二阶段最终架构与实现决策  
  - 技术栈：Markdown 文档体系、现有 speckit 规范
- [ ] T223 [P2] 在 `speckit/specs/changes/` 下记录与第二阶段相关的关键变更（Html 生成链、可视化栈、中台能力）  
  - 技术栈：change-manager 模板与规范
- [ ] T224 [P3] [P] 针对重构后的模块进行必要的代码清理与重构，消除遗留 TODO/FIXME 与重复逻辑  
  - 技术栈：Python/TypeScript 代码库、项目章程中 P1 代码质量与 P8 禁止模式规范
- [ ] T225 [P2] 对典型白皮书生成流程进行性能与稳定性评估，记录结论并给出后续优化建议  
  - 技术栈：pytest/基准测试脚本、现有日志与监控手段、LlamaIndex/AntV 渲染链路

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

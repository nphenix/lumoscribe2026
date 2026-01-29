---
template: Plan
version: 1.0.0
lastUpdated: 2026-01-29
charter: ../constitution.md
---

# 实施计划：第二阶段 AI 文档生成平台

**分支**: `feature/phase2-ai-doc-platform` | **日期**: 2026-01-29 | **规格**: [spec.md](./spec.md)

**输入**: 第二阶段功能规格说明（Html 生成链路重构 + 基于 AntV 的数据可视化模块 + 中台 LLM 并发与提示词管理增强）

---

## 摘要

第二阶段在现有 ingest + 建库 + 报告生成能力的基础上，聚焦于三个方向：

- 重构 Html 生成链路，按章节拆分并引入“生成草稿 → 语义去重与结构校验 → 语言润色”的多阶段流水线，降低重复、补足空章节，稳定图表占位符；
- 重构数据可视化模块，以 AntV G2、InfoGraphic、S2 为核心栈，从图表 JSON 与原图出发，统一生成科研风格的统计图、信息图和分析型表格，支持 SVG/PNG 导出；
- 增强中台对 LLM 的工程治理能力，在 Provider/CallSite 层支持并发配置和提示词版本管理，所有调用统一通过 LLMRuntimeService 与 PromptService 读取配置。

---

## 技术上下文

**语言/版本**: Python 3.11、TypeScript/Next.js（中台 Web）、ECharts 现有栈（逐步迁移到 AntV）  
**主要依赖**: FastAPI、SQLAlchemy、LangChain/LlamaIndex、AntV G2/InfoGraphic/S2（新引入）、Celery/Ingest Pipeline  
**存储**: SQLite/关系型数据库（元数据）、BM25 索引、ChromaDB 向量库、文件系统（HTML/图表/中间态产物）  
**测试**: pytest（后端）、前端测试按现有中台规范执行  
**目标平台**: Windows 11 开发环境，部署目标为服务器环境（与第一阶段保持一致）  
**项目类型**: Web 平台（后端 API + 中台管理界面）  
**性能目标**:
- 生成单份白皮书 Html 的总耗时相较第一阶段不显著增加（在同等模型配置下），但重复率与结构质量明显提升；
- 图表渲染与导出在常规报告规模下（几十幅图表）可在可接受时间内完成；
- 中台并发配置生效后，确保不会超出 Provider 的并发上限。

**约束条件**:
- 保持第一阶段外部 API 兼容（/v1/jobs、/v1/targets 等），允许内部实现替换；
- 禁止在业务代码中硬编码提示词，必须通过统一 Prompt 管理机制；
- 遵守项目章程关于文件大小、分层与日志/错误处理规范。

**规模/范围**:
- 面向单工作区内的若干白皮书项目（数量级与第一阶段一致），优先保证一个典型白皮书完整链路的质量。

---

## 项目结构

### 文档（本功能）

```
speckit/specs/features/ai-doc-platform-phase2/
├── spec.md              # 第二阶段功能规格（新增/增强能力说明）
├── plan.md              # 本文件（架构/策略/决策，无任务清单）
└── tasks.md             # 第二阶段任务清单 + 验收标准
```

### 源代码（关联范围）

第二阶段主要涉及以下现有模块与新增模块：

- 内容生成链路：
  - `src/application/services/content_generation/` 子包
  - `src/application/services/content_generation_service.py` 导出层
- 图表与可视化：
  - `src/application/services/chart_renderer_service.py`
  - `data/intermediates/*/pic_to_json/chart_json/` 与 `images/` 相关处理逻辑
- 中台 LLM 配置与 Prompt 管理：
  - `src/application/services/llm_runtime_service.py`
  - `src/shared/constants/prompts.py`
  - `src/interfaces/admin-web/` 下的 LLM Provider/CallSite/Prompt 页面

---

## 结构决策

1. **Html 生成链路按章节与阶段拆分**
- 单一长链路拆分为：按章节生成草稿 → 基于检索的语义去重与结构校验 → 语言润色与连接；
- 每个阶段的输入/输出必须是 Markdown 级别的文本，便于日志与调试；
- 图表占位符 `[Chart:id]` 在生成阶段作为文本锚点，真正的图表渲染在 Html 渲染与可视化模块中完成。

2. **数据可视化统一采用 AntV 技术栈**
- 统计图采用 AntV G2，统一处理 stacked_area、bar、line 等科研型图表；
- 信息图/逻辑图采用 AntV InfoGraphic，将 sankey/价值链/政策清单等映射为信息图模板；
- 分析型表格采用 AntV S2（PivotSheet/TableSheet），并视需求选择在报告中以信息图或表格形式呈现；
- 通过统一主题配置保持图表、信息图和表格在同一视觉体系内。
- 图表渲染采用“预渲染静态图片”模式：以现有 `chart_json` 文件名（去掉 `.json`）为 chart_id，在 `data/intermediates/{id}/pic_to_json/chart_json/` 目录下就地生成 SVG/PNG 文件，并由 Html 渲染阶段按 `[Chart:chart_id]` 规则引用对应图片，而不是在生成 Html 时即时渲染图表。

3. **中台作为 LLM 配置与 Prompt 的唯一入口**
- 延续第一阶段的 Provider/CallSite/Prompt 管理模型；
- 增强 Provider/CallSite 配置以支持并发相关参数；
- 所有 Prompt 版本管理与启用状态由中台控制，运行时仅通过 PromptService 访问。

---

## 分阶段实施策略（概要）

> 详细任务将在 tasks.md 中分解，这里只记录阶段划分与高层策略。

### 阶段 A：Html 生成链路重构（内容链）

目标：在不改变对外 API 的前提下，用新链路替换现有单阶段 Html 生成逻辑。

- 从现有 `ContentGenerationService` 中抽取 Html 生成相关逻辑，形成可插拔的章节生成与后处理管线；
- 在章节级别引入“生成草稿 → 去重/结构校验 → 润色”的分阶段编排；
- 引入针对“空章节/纲要残影/重复内容”的专用后处理工具，利用现有向量检索基础设施；
- 保持 `/v1/jobs` 生成目标文件的接口形态不变，仅替换内部实现。

### 阶段 B：数据可视化模块重构（AntV 栈）

目标：用 AntV G2/InfoGraphic/S2 替代或包裹现有图表渲染路径，实现统一风格与出图能力。

- 分析现有 `chart_json` 结构与 Html 中图表插入方式，定义 `chart_type -> AntV spec/DSL` 映射规则；
- 引入 AntV 相关依赖与封装层，为后端/前端提供统一的“chart_json -> SVG/PNG 静态图片”能力，渲染产物就地存放在 `data/intermediates/{id}/pic_to_json/chart_json/` 目录下；
- 设计并实现“白皮书主题”，在 G2/InfoGraphic/S2 中应用；
- 在 Html 渲染阶段，根据正文中的 `[Chart:chart_id]` 占位符与对应 intermediates 路径，直接引用预渲染图片文件（例如 `<chart_id>__whitepaper-default.svg`），替换现有基于运行时 JS 的图表渲染片段。

### 阶段 C：中台 LLM 并发与 Prompt 管理增强

目标：让运维/配置人员通过中台控制第二阶段所需的 LLM 并发策略与提示词版本。

- 在 Provider/CallSite 的后端模型与前端配置页面中增加并发相关字段；
- 在 LLMRuntimeService 与相关业务服务中读取并执行并发限制策略；
- 扩展 Prompt 管理界面，支持版本列表、差异查看与启用/停用操作；
- 确保第二阶段新增的调用点全部遵守统一配置。

---

**版本**: 1.0.0 | **创建**: 2026-01-29

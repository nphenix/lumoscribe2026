## 现状核对（基于当前代码与文档）
- 白皮书/报告生成主链路已落在 [targets.py](file:///f:/lumoscribe2026/src/interfaces/api/routes/targets.py) 的 `/v1/targets/whitepaper/generate(_stream)`，核心编排在 [content_generation/pipeline.py](file:///f:/lumoscribe2026/src/application/services/content_generation/pipeline.py)。
- 质量/基线脚本已存在： [t201-whitepaper-baseline.py](file:///f:/lumoscribe2026/scripts/t201-whitepaper-baseline.py)（偏质量指标，不是性能压测框架）。
- T222 点名的过程文档在 [# 报告生成链路重构讨论记录（阶段一）.md](file:///f:/lumoscribe2026/docs/process/%23%20%E6%8A%A5%E5%91%8A%E7%94%9F%E6%88%90%E9%93%BE%E8%B7%AF%E9%87%8D%E6%9E%84%E8%AE%A8%E8%AE%BA%E8%AE%B0%E5%BD%95%EF%BC%88%E9%98%B6%E6%AE%B5%E4%B8%80%EF%BC%89.md)。
- `speckit/specs/changes/` 已有索引与 2026-01 月目录，但没有“phase2/第二阶段关键变更”专门记录：见 [changes/00-目录说明.md](file:///f:/lumoscribe2026/speckit/specs/changes/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md)。
- 代码侧 TODO/FIXME 主要集中在：
  - [document_cleaning_service.py](file:///f:/lumoscribe2026/src/application/services/document_cleaning_service.py)（IntermediateArtifact.workspace_id 被硬编码为 default）
  - [llm_provider_service.py](file:///f:/lumoscribe2026/src/application/services/llm_provider_service.py)（删除 Provider 前未做 CallSite/Capability 绑定检查）
  - [targets.py](file:///f:/lumoscribe2026/src/interfaces/api/routes/targets.py)（target 详情返回 template_name/kb_name 未填充）
  - [use-documents.ts](file:///f:/lumoscribe2026/src/interfaces/admin-web/hooks/use-documents.ts)（上传 sources workspace 固定 default）

## T222：更新阶段一讨论记录为“阶段二最终架构与决策”
- 修改 [# 报告生成链路重构讨论记录（阶段一）.md](file:///f:/lumoscribe2026/docs/process/%23%20%E6%8A%A5%E5%91%8A%E7%94%9F%E6%88%90%E9%93%BE%E8%B7%AF%E9%87%8D%E6%9E%84%E8%AE%A8%E8%AE%BA%E8%AE%B0%E5%BD%95%EF%BC%88%E9%98%B6%E6%AE%B5%E4%B8%80%EF%BC%89.md)：
  - 将“讨论草案/初版设想”的表述改为“第二阶段已落地实现”，并按事实对齐到当前代码：生成入口（targets 路由）、内容管线（pipeline）、后处理（去重/空章节处理等）、图表渲染（ChartRendererService/HTML 渲染器）。
  - 补齐关键默认策略/开关的来源（例如 [config.py](file:///f:/lumoscribe2026/src/shared/config.py) 中白皮书生成默认参数）。
- 同步更新/互链相关过程文档索引 [ai-doc-platform-phase2/00-目录说明.md](file:///f:/lumoscribe2026/docs/process/ai-doc-platform-phase2/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md) 与三份阶段1基线文档，使“阶段1基线 → 阶段2落地决策 → 阶段5评估结论”闭环。

## T223：记录第二阶段关键变更（Html 链/可视化栈/中台能力）
- 在 `speckit/specs/changes/2026-01/` 新增一份变更提案/记录（命名按约定，内容按模板 [change-template.md](file:///f:/lumoscribe2026/speckit/templates/change-template.md)）：
  - Delta 里覆盖：内容生成链（服务拆包、章节管线与后处理）、图表链路（占位符与渲染职责）、中台能力增强（LLM Provider/CallSite/Prompt/并发门控等第二阶段实际改动点）。
- 更新：
  - [changes/00-目录说明.md](file:///f:/lumoscribe2026/speckit/specs/changes/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md)
  - [changes/2026-01/00-目录说明.md](file:///f:/lumoscribe2026/speckit/specs/changes/2026-01/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md)

## T224：必要代码清理（消除遗留 TODO/FIXME、收敛重复逻辑）
- 以“可验证、无行为回归”为边界，优先处理影响正确性/可维护性的 TODO：
  - 文档清洗工件写库时的 workspace_id：在 `DocumentCleaningService._save_artifact()`（含 cleaned_doc 与 chart_json 两个实现）通过同一 DB session 查询 SourceFile.workspace_id，替代硬编码 default。
  - Provider 删除安全：在 `LLMProviderService.delete_provider()` 增加绑定检查（CallSite.provider_id、Capability.provider_id），存在引用则返回 409 并给出计数信息；必要时为仓储层补充按 provider_id 过滤的方法。
  - Target 详情关联信息：`/targets/{id}` 返回 `template_name`（TemplateRepository.get_by_id），`kb_name` 以当前事实填充为 `kb_id/collection_name`（若后续引入 KB 表再替换为真正名称）。
  - 前端 sources 上传 workspace：若当前 Admin Web 已有 workspace 概念（例如 query param/localStorage），则把 workspace 贯通；否则保留 default，但移除 TODO 并在 UI 层明确当前仅支持 default（避免“TODO 悬空”）。

## T225：白皮书生成性能与稳定性评估（产出结论+建议）
- 新增一个最小可运行的评估脚本（建议 `scripts/t225-whitepaper-perf.py`）：
  - 使用 `httpx` 调用 `/v1/targets/whitepaper/generate`（或 generate_stream 的 final 事件），循环多次；采集 `total_time_ms/total_tokens` 与各 section 的 `generation_time_ms/tokens_used`，统计 p50/p95、失败率、409（并发锁）等。
  - 输出：一份 JSON + 一份 Markdown 报告到 `docs/process/ai-doc-platform-phase2/`，并更新该目录的 [00-目录说明.md](file:///f:/lumoscribe2026/docs/process/ai-doc-platform-phase2/00-%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E.md)。
  - 报告包含：测试前置条件（需要已建库 collection、可用 LLM 配置）、运行命令、数据口径、结论与后续优化建议（例如：检索参数、并发门控、缓存/复用、分段策略等）。

## 任务收尾与验证（uv 环境）
- 更新任务勾选与“相关文件”列表： [tasks.md](file:///f:/lumoscribe2026/speckit/specs/features/ai-doc-platform-phase2/tasks.md)（完成 T222–T225 后按文件表补全）。
- 代码验证（按你要求用 uv）：
  - 单测：`uv run python -m pytest`
  - 评估脚本：`uv run python scripts/t225-whitepaper-perf.py ...`
  -（如需）对现有基线脚本：`uv run python scripts/t201-whitepaper-baseline.py ...`

如果你确认该方案，我将按以上顺序落地修改、补齐文档与脚本，并用 uv 运行测试/评估来验证结果。
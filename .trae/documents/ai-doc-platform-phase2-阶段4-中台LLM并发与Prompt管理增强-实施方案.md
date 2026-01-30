## 现状（基于代码事实）
- **LLM 配置模型**：`LLMProvider` 与 `LLMCallSite` 目前仅有连接/密钥/JSON 配置与 enabled 等字段，没有任何并发字段（见 [llm_provider.py](file:///f:/lumoscribe2026/src/domain/entities/llm_provider.py)、[llm_call_site.py](file:///f:/lumoscribe2026/src/domain/entities/llm_call_site.py)）。
- **LLM 配置 API**：`/v1/llm/providers`、`/v1/llm/call-sites` 仅读写现有字段（见 [llm.py](file:///f:/lumoscribe2026/src/interfaces/api/routes/llm.py)、[schemas/llm.py](file:///f:/lumoscribe2026/src/application/schemas/llm.py)）。
- **Prompt 管理**：后端已有版本自增 + active 切换（`PromptRepository.get_latest_version/deactivate_scope/get_active_prompt`；`PromptService.create_prompt/update_prompt`），API 支持 list/create/patch/delete（见 [prompt_repository.py](file:///f:/lumoscribe2026/src/application/repositories/prompt_repository.py)、[prompt_service.py](file:///f:/lumoscribe2026/src/application/services/prompt_service.py)、[prompts.py](file:///f:/lumoscribe2026/src/interfaces/api/routes/prompts.py)）。但：缺少 `GET /prompts/{id}`、缺少“scope 汇总/版本列表/差异”专用接口。
- **LLM 运行时**：`LLMRuntimeService` 按 CallSite 构建 LangChain runnable/模型（见 [llm_runtime_service.py](file:///f:/lumoscribe2026/src/application/services/llm_runtime_service.py)），目前没有统一并发门控。
- **业务调用点（需要纳入并发门控）**：
  - 章节生成：`SectionLLMGenerator` 直接 `runnable.astream/invoke`（见 [llm_section.py](file:///f:/lumoscribe2026/src/application/services/content_generation/llm_section.py)）。
  - 结构化章节：`SectionStructuredGenerator` 使用 `create_agent(...).ainvoke`，失败回退到 `model.invoke`（见 [section_structured.py](file:///f:/lumoscribe2026/src/application/services/content_generation/section_structured.py)）。
  - 大纲润色：`OutlinePolishService` 使用 `create_agent(...).ainvoke`（见 [outline_polish_service.py](file:///f:/lumoscribe2026/src/application/services/outline_polish/outline_polish_service.py)）。
  - 文档清洗：内部并发目前硬编码 `concurrency_limit = 1`，并且每段直接 `chain.astream`（见 [document_cleaning_service.py](file:///f:/lumoscribe2026/src/application/services/document_cleaning_service.py)）。
- **中台前端（Next.js admin-web）**：Provider/CallSite 配置页目前不包含并发字段；Prompt 页仅“创建新版本”，没有版本对比/启停/回滚（见 [providers/page.tsx](file:///f:/lumoscribe2026/src/interfaces/admin-web/app/llm/providers/page.tsx)、[call-sites/page.tsx](file:///f:/lumoscribe2026/src/interfaces/admin-web/app/llm/call-sites/page.tsx)、[prompts/page.tsx](file:///f:/lumoscribe2026/src/interfaces/admin-web/app/prompts/page.tsx)、[hooks/use-llm.ts](file:///f:/lumoscribe2026/src/interfaces/admin-web/hooks/use-llm.ts)、[hooks/use-prompts.ts](file:///f:/lumoscribe2026/src/interfaces/admin-web/hooks/use-prompts.ts)）。
- **DB 迁移方式**：项目未引入 Alembic，仅在 `src/shared/db.py::_apply_lightweight_sqlite_migrations` 做 SQLite add-column 补齐（见 [db.py](file:///f:/lumoscribe2026/src/shared/db.py)）。

## LangChain 并发“最佳实践”依据（联网结果）
- LangChain Runnable 接口中，`astream` 的默认实现会调用 `ainvoke`，如果组件支持真正的流式，需要覆盖 `astream`（这意味着并发门控应覆盖 `ainvoke/astream` 两条路径）([LangChain Reference - Runnables](https://reference.langchain.com/python/langchain_core/runnables/))。
- LangChain `RunnableConfig` 中包含 `max_concurrency` 字段，但其语义主要作用于批处理/并行执行场景，无法替代“跨业务调用点的全局并发上限治理”([langchain_core RunnableConfig 源码索引页](https://api.python.langchain.com/en/latest/_modules/langchain_core/runnables/config.html))。

## 目标（对应 tasks.md 阶段4：T217-T221）
1. **T218**：Provider/CallSite 增加并发字段，并贯通后端模型 + API + admin-web 配置与展示。
2. **T219**：在 `LLMRuntimeService` 提供统一并发门控能力；业务侧所有 LLM 调用点通过门控执行，保证不超过中台配置。
3. **T220**：Prompt 管理增强：版本列表、差异查看、启停、回滚（激活旧版本）。
4. **T221**：第二阶段新增/改造的所有 LLM 调用点，统一走 Provider/CallSite/Prompt 配置路径（并发门控也通过该路径）。
5. **T217**：补齐集成测试，覆盖 Provider/CallSite/Prompt CRUD 与“配置生效路径”（至少验证并发字段能影响运行时策略选择；不依赖真实外部 LLM）。

## 设计与实现方案

## 数据模型（DB）
- **新增列**：
  - `llm_providers.max_concurrency`：Provider 级并发上限（NULL 表示不限制/或走默认值）。
  - `llm_call_sites.max_concurrency`：CallSite 级并发上限（NULL 表示不限制/或走 Provider）。
- **SQLite 迁移**：在 [db.py](file:///f:/lumoscribe2026/src/shared/db.py) 的 `_apply_lightweight_sqlite_migrations` 中为 `llm_providers/llm_call_sites` 添加对应 `ALTER TABLE ... ADD COLUMN`。

## 后端 API / Schema
- 更新 `src/application/schemas/llm.py`：
  - Provider：Create/Update/Response 增加 `max_concurrency?: number | null`。
  - CallSite：Create/Update/Response 增加 `max_concurrency?: number | null`。
- 更新服务层：
  - `LLMProviderService.create_provider/update_provider` 写入/更新并发字段。
  - `LLMCallSiteService.create_call_site/update_call_site` 写入/更新并发字段。
- 更新路由层：`/llm/providers` 与 `/llm/call-sites` 的序列化/反序列化包含新字段。

## 运行时并发门控（LLMRuntimeService）
- 在 `LLMRuntimeService` 内新增“并发策略解析 + 门控”能力：
  - `resolve_effective_concurrency(callsite_key) -> {provider_limit, callsite_limit, effective}`：
    - `callsite_limit` 优先于 `provider_limit`。
    - 若两者均为空，则使用一个**可配置默认值**（建议在 `Settings` 增加 `llm_default_max_concurrency`，默认值取 4；仍可被中台字段覆盖）。
  - `asynccontextmanager acquire_llm_slot(callsite_key)`：
    - 内部按“CallSite semaphore + Provider semaphore（如果分别配置）”顺序 acquire，退出时 release。
    - semaphore 存放在**进程内**的 registry（类静态 dict），key 使用 `provider.id`/`callsite.id`，并将 `limit` 纳入 key 以便配置变化时自动重建。
- 为什么用 semaphore 而非 RunnableConfig：
  - 因为我们要控制的是“跨业务调用点、跨请求”的并发上限治理，而 RunnableConfig 的 `max_concurrency` 主要针对 batch/并行调用的内部策略，且对单次 invoke/astream 并不构成统一门控（见上面的 LangChain 引用）。

## 业务服务接入（确保 T219/T221）
- 全局策略：**所有调用 LLM 的地方，在真正触发 `invoke/ainvoke/astream` 前必须 `async with llm_runtime.acquire_llm_slot(callsite_key)`**。
- 覆盖点（已定位到的真实调用点）：
  - [llm_section.py](file:///f:/lumoscribe2026/src/application/services/content_generation/llm_section.py)：
    - `stream_tokens=True`：包住 `async for chunk in runnable.astream(payload)`。
    - `stream_tokens=False`：包住 `run_in_executor` 的 `runnable.invoke(payload)`。
  - [section_structured.py](file:///f:/lumoscribe2026/src/application/services/content_generation/section_structured.py)：
    - 包住 `agent.ainvoke(payload)` 与回退分支的 `model.invoke(parser_prompt)`。
  - [outline_polish_service.py](file:///f:/lumoscribe2026/src/application/services/outline_polish/outline_polish_service.py)：
    - 包住 `agent.ainvoke(payload)`（以及 sync fallback 分支）。
  - [document_cleaning_service.py](file:///f:/lumoscribe2026/src/application/services/document_cleaning_service.py)：
    - 将硬编码 `concurrency_limit=1` 替换为 `resolve_effective_concurrency(SCOPE_DOC_CLEANING)` 的结果（或允许 options 覆盖，但最终仍受门控）。
    - 在 `_llm_clean_text_segment_stream`/非流式清洗里包住 `chain.astream`。
  - 如在同文件还存在 `SCOPE_CHART_EXTRACTION` 等调用点，也一并接入门控。
- 接入后，**并发上限完全由中台 Provider/CallSite 字段决定**，业务侧仅负责“并行发起多少任务”，但不会突破门控。

## Prompt 管理增强（T220）

## 后端能力
- 在 Prompt API 增加：
  - `GET /prompts/{prompt_id}`：返回单条 Prompt（用于前端查看某版本详情/对比）。
  - `GET /prompts/scopes`（或 `GET /prompts/summary`）：按 scope 聚合返回：最新版本号、active 版本号、格式、版本数、updated_at（便于 UI 做“scope 列表/展开版本”）。
  - `GET /prompts/diff?from_id=&to_id=`：返回 unified diff（Python `difflib.unified_diff` 生成），便于前端直接展示“差异”。
  - 复用现有 `PATCH /prompts/{id}` 做启用/停用（active=true/false），回滚就是对旧版本 `active=true`。

## 前端（admin-web）
- Prompts 页面重构为两层：
  - Scope 列表（每行显示：scope、active 版本、最新版本、格式、版本数、更新时间）。
  - Scope 展开后展示版本列表（降序），提供：
    - 查看内容（text 或 chat messages）。
    - 选择两条版本做 diff（调用 `/prompts/diff`）。
    - 对某版本“启用/停用”（调用 `PATCH /prompts/{id}`）。
    - 对某历史版本“一键回滚/启用”。
- 同步修正 `use-prompts.ts`：
  - 增加 `usePrompt(id)` 对应新增 `GET /prompts/{id}`。
  - 将 `useUpdatePrompt` 从 `PUT` 改为 `PATCH`，字段改为 `active/description`（匹配后端真实 API）。

## 中台并发字段 UI（T218）
- Providers 页：新增 `max_concurrency` 数字输入与表格列展示（空=默认/不限制）。
- CallSites 页：编辑弹窗增加 `max_concurrency` 数字输入（作为 override；空=继承 Provider）。
- `hooks/use-llm.ts` 的 Provider/CallSite 类型与 Create/Update 接口同步增加字段。

## 测试与验证（T217）
- 新增/更新 pytest 用例（尽量复用现有 `api_client` fixture）：
  - Provider CRUD：创建/更新并发字段、列表返回字段。
  - CallSite CRUD：创建/更新并发字段、列表返回字段。
  - Prompt：创建多个版本、切换 active、保证 scope 内最多一个 active。
  - “配置生效路径”测试（不依赖真实 LLM）：
    - 通过仓储写入 Provider/CallSite 并发字段后，直接实例化 `LLMRuntimeService` 调用 `resolve_effective_concurrency(callsite_key)` 断言选择逻辑正确。
- 由于当前存在历史测试文件包含 `/v1/llm/models` 等已不存在的接口（见 [test_llm_config_api.py](file:///f:/lumoscribe2026/tests/test_llm_config_api.py)），会在本阶段同步**重写/拆分**为与现状匹配的测试，确保 `uv run pytest` 可通过。

## 交付与本地验证方式（uv 环境）
- 后端测试：`uv run pytest`（必要时指定：`uv run pytest -q`）。
- 关键点单测（如只跑并发与 prompts）：`uv run pytest -q tests/test_llm_concurrency_and_prompts.py`（文件名以最终落地为准）。

## 影响范围与兼容性
- DB：仅新增列（SQLite add-column），不破坏现有数据。
- 并发门控：进程内生效；多进程部署时每个进程独立限流（符合常见 ASGI 部署现实）。
- Prompt：新增只读接口与 diff 接口，不影响现有 `/prompts` list/create/patch/delete。

---

如果你确认该方案，我将按上述步骤开始实际开发，并在开发完成后：
- 更新 `speckit/specs/features/ai-doc-platform-phase2/tasks.md` 中阶段4 的勾选与“相关文件”列表（按项目规范）。
- 使用 `uv run ...` 运行测试验证通过。
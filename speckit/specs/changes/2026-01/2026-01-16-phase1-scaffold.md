---
id: phase1-scaffold
status: 已实施
created: 2026-01-16
updated: 2026-01-16
links:
  - ../../features/ai-doc-platform-phase1/spec.md
  - ../../features/ai-doc-platform-phase1/plan.md
  - ../../features/ai-doc-platform-phase1/tasks.md
---

# 变更提案：第一阶段骨架落地与流程纠偏

**变更ID**: `phase1-scaffold`  
**创建日期**: 2026-01-16  
**状态**: 已实施  
**关联功能**: [第一阶段 AI 文档生成平台 spec.md](../../features/ai-doc-platform-phase1/spec.md)

---

## 变更原因

为尽快形成“API 创建任务 → Redis 队列 → Worker 执行 → 状态回写 → API 查询”的最小闭环，先落地后端/worker 骨架；同时纠偏补齐 `spec/plan/tasks/change` 工件，恢复可追溯流程。

## 变更内容

- **后端骨架**: 新增 `src/` 分层骨架与 FastAPI 最小服务（health、统一错误格式、request_id）
- **任务系统**: 引入 Redis 队列（Celery），实现 `/v1/jobs` 创建任务即入队，占位任务回写 SQLite `jobs`
- **测试底座**: 增加基础 API 测试（health、422 错误格式），并将 pytest 作为默认依赖以避免环境缺失
- **文档工件**: 补齐 `speckit/specs/features/ai-doc-platform-phase1/{spec,plan,tasks}.md`

## Delta 操作

### 新增需求 (ADDED)

- **ADDED**: 任务队列采用 Redis（Celery），并以 SQLite 为任务状态权威来源
- **ADDED**: 第一阶段功能工件目录与 `spec/plan/tasks` 文档

### 修改需求 (MODIFIED)

- **MODIFIED**: 流程执行顺序从“直接编码”纠偏为“文档工件可追溯后继续推进”

### 移除需求 (REMOVED)

- 无

## 影响范围

**受影响的规格**: `speckit/specs/features/ai-doc-platform-phase1/spec.md`

**受影响的代码**（核心）:
- `pyproject.toml`（依赖：FastAPI/Celery/Redis/pytest）
- `src/interfaces/api/app.py`、`src/interfaces/api/routes/jobs.py`
- `src/interfaces/worker/celery_app.py`、`src/interfaces/worker/tasks.py`
- `src/shared/*`（config/logging/errors/db/storage/request_id）
- `tests/test_api_basic.py`

---

## 实施计划

### 阶段 1：准备

- [x] C001 审查并补齐第一阶段 spec/plan/tasks 工件
- [x] C002 创建变更记录（本文件）

### 阶段 2：实施

- [x] C003 落地 API/Worker 最小闭环骨架
- [x] C004 将 pytest 设为默认依赖，避免 `No module named pytest`

### 阶段 3：验证

- [ ] C005 执行 `python -m pip install -e .` 安装依赖
- [ ] C006 执行 `python -m pytest` 通过基础测试

---

**版本**: 1.0.0 | **创建**: 2026-01-16 | **最后更新**: 2026-01-16

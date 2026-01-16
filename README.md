# Lumoscribe2026

面向 AI 的文档生成平台（第一阶段）：**图片型 PDF → OCR/清洗 → 知识库（向量+BM25） → 按模板 RAG+LLM 生成单 HTML**，并提供 API 与中台管理能力。

## 当前进度

- ✅ 后端最小骨架（FastAPI + SQLite）
- ✅ 任务模型（Job）与基础接口（创建/查询）
- ✅ Worker（Celery + Redis）与流水线能力（OCR/清洗/切块/索引/生成）
- ⏳ Next.js 中台

---

## 快速安装与运行

### 前置要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.12 | 项目运行环境 |
| Redis | ≥ 6.0 | Celery 消息队列 |
| uv | ≥ 0.4 | 包管理工具（推荐）|

### 1) 安装依赖

```powershell
# 使用 uv 安装（推荐，速度快）
uv pip install -e .

# 或使用 pip 安装
pip install -e .
```

### 2) 启动 Redis

Redis 作为 Celery Worker 的消息队列，必须先于 Worker 启动。

```powershell
# 方式一：本地启动 Redis（默认端口 6379）
redis-server

# 方式二：使用 Docker（推荐）
docker run -d -p 6379:6379 --name lumoscribe-redis redis:7-alpine
```

### 3) 配置环境变量

复制根目录的 `env.example` 内容到你的本地环境变量（或复制为本地 `.env` 自行加载），至少需要：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LUMO_SQLITE_PATH` | SQLite 数据库文件路径 | `./data/lumoscribe.db` |
| `LUMO_STORAGE_ROOT` | 文件存储根目录 | `./storage` |
| `LUMO_REDIS_URL` | Redis 连接 URL | `redis://localhost:6379/0` |

**PowerShell 设置示例：**

```powershell
$LUMO_SQLITE_PATH = "$PWD\data\lumoscribe.db"
$LUMO_STORAGE_ROOT = "$PWD\storage"
$LUMO_REDIS_URL = "redis://localhost:6379/0"
```

### 4) 启动 API 服务

```powershell
python -m src.interfaces.api.main
```

API 服务默认运行在 `http://localhost:8000`。

**健康检查：**

```powershell
# 检查服务状态
Invoke-RestMethod -Uri "http://localhost:8000/v1/health"
```

### 5) 启动 Worker 服务

```powershell
# 在新终端中启动 Celery Worker
celery -A src.interfaces.worker.celery_app worker --loglevel=info
```

Worker 启动后会监听 Redis 队列中的任务。

### 6) 验证安装

完成上述步骤后，执行以下验证：

```powershell
# 1. 确认 Redis 运行
redis-cli ping
# 应返回: PONG

# 2. 确认 API 运行
Invoke-RestMethod -Uri "http://localhost:8000/v1/health"
# 应返回 JSON: {"status":"healthy"}

# 3. 创建测试任务
$body = @{ source_file = "test.pdf" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/v1/jobs" -Method POST -Body $body -ContentType "application/json"
```

---

## 目录结构

代码遵循分层放置：

- `src/domain/`：领域层
- `src/application/`：应用层
- `src/interfaces/`：接口层（API/Worker/AdminWeb）
- `src/shared/`：共享基础设施

---

## 开发命令速查

| 场景 | 命令 |
|------|------|
| 安装依赖 | `uv pip install -e .` |
| 启动 API | `python -m src.interfaces.api.main` |
| 启动 Worker | `celery -A src.interfaces.worker.celery_app worker --loglevel=info` |
| 运行测试 | `pytest tests/` |
| 代码检查 | `pytest tests/ --co -q` |

---

**最后更新**: 2026-01-16

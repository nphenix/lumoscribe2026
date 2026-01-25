# Lumoscribe2026 - 面向 AI 的文档生成平台

> **核心愿景**：图片型 PDF → OCR/清洗 → 知识库（向量+BM25）→ 按模板 RAG+LLM 生成单 HTML，并提供 API 与中台管理能力。

## 项目概览

| 指标 | 数值 |
|------|------|
| Python 文件数 | 82+ |
| 领域实体 | 8 个核心实体 |
| API 路由模块 | 9 个 |
| 服务层模块 | 15+ |
| LLM 调用点 | 15+ 种场景 |

## 技术栈

| 层级 | 技术选型 | 版本 |
|------|----------|------|
| 后端框架 | FastAPI | ≥ 0.115 |
| ORM | SQLAlchemy | ≥ 2.0 |
| 任务队列 | Celery + Redis | ≥ 5.4, ≥ 6.0 |
| 前端框架 | Next.js | ≥ 20.0 |
| 数据库 | SQLite | - |
| LLM 框架 | LangChain + LlamaIndex | ≥ 1.0, ≥ 0.14 |
| 向量数据库 | ChromaDB | ≥ 0.5 |
| Embedding | FlagEmbedding + Sentence-Transformers | ≥ 1.3.5, ≥ 3.0 |
| Python | - | 3.12 - 3.13 |

## 架构设计

采用分层架构（Clean Architecture）：

```
src/
├── domain/                 # 领域层（核心业务逻辑）
│   └── entities/          # 实体定义（8个核心实体）
├── application/           # 应用层（用例编排）
│   ├── services/          # 服务层（15+业务逻辑）
│   ├── repositories/      # 数据仓储
│   └── schemas/           # 数据模型（DTO）
├── interfaces/            # 接口层（外部交互）
│   ├── api/               # FastAPI 后端（9个路由模块）
│   │   └── routes/        # API 路由定义
│   ├── worker/            # Celery Worker
│   └── admin-web/         # Next.js 管理后台
└── shared/                # 共享基础设施
    ├── config.py          # 配置管理
    ├── db.py              # 数据库工具
    ├── logging.py         # 日志配置
    ├── storage.py         # 文件存储
    └── errors.py          # 错误定义
```

## 当前进度

### 核心功能 ✅
- 后端骨架：FastAPI + SQLite 完整架构
- 任务系统：Job 模型与 CRUD 接口
- 异步流水线：Celery Worker 支持多阶段处理
- 管理后台：Next.js Admin Dashboard

### 文档处理流水线 ✅
- OCR 解析：MinerU 在线服务集成（支持 400+ 语言）
- 文档清洗：智能文本清洗与结构化
- 语义分块：基于内容的智能切分
- 混合检索：向量检索（ChromaDB）+ BM25 双重机制

### RAG 生成系统 ✅
- 向量化引擎：FlagEmbedding + Sentence-Transformers
- LLM 运行时：OpenAI + Ollama 多提供商支持
- 内容生成：模板驱动的 RAG 生成
- 图表渲染：Mermaid + Vega 图表支持

### 核心领域实体 ✅
- SourceFile（源文件）、Template（模板）、TargetFile（目标文件）
- IntermediateArtifact（中间产物）、Prompt（提示词）
- LLMProvider、LLMCapability、LLMCallSite（LLM 调用管理）

---

## 快速安装与运行

### 前置要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.12 | 项目运行环境 |
| Node.js | ≥ 20.0 | 前端运行环境 |
| Redis | ≥ 6.0 | Celery 消息队列 |
| uv | ≥ 0.4 | 包管理工具（推荐）|

### 1) 安装依赖

**后端：**

本项目强制使用 [uv](https://github.com/astral-sh/uv) 进行包管理和环境隔离，以确保跨平台（Windows CUDA / macOS）的一致性。

```powershell
# 1. 初始化并同步环境 (自动安装 Python 3.12 和所有依赖)
# 注意：Windows 上会自动配置 CUDA 13.0 版 PyTorch，macOS 上会自动配置 MPS 版
uv sync

# 2. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**前端：**
```powershell
cd src/interfaces/admin-web
npm install
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

### 4) 启动服务

**终端 1: 启动 API 服务**
```powershell
python -m src.interfaces.api.main
# 默认端口: 7901
```

**终端 2: 启动 Worker 服务**
```powershell
celery -A src.interfaces.worker.celery_app worker --loglevel=info
```

**终端 3: 启动前端开发服务器**
```powershell
cd src/interfaces/admin-web
npm run dev
# 访问 http://localhost:7906
```

### 5) 验证安装

完成上述步骤后，执行以下验证：

1. **API**: 访问 `http://localhost:7901/v1/health` 应返回 `{"status":"healthy"}`
2. **Frontend**: 访问 `http://localhost:7906` 查看管理后台仪表盘

## 端口配置

本项目使用的所有端口配置汇总：

|| 端口 | 服务 | 说明 |
|------|------|------|
| **7901** | API 主服务 | 默认服务端口，`LUMO_API_PORT` |
| **7902** | KB Admin | 知识库建库/管理专用（独立部署） |
| **7903** | KB Query | 知识库查询专用（独立部署，hybrid + rerank） |
| **7904** | FlagEmbedding | 向量服务，`LUMO_LLM_FLAGEMBEDDING_HOST` |
| **7905** | MinerU | PDF OCR 服务，`LUMO_MINERU_API_URL` |
| **7906** | Next.js 前端 | 前端开发服务器 |
| **7907** | LLM 测试 Mock | 测试环境 Mock 服务 |
| **6379** | Redis | 消息队列/缓存（保持不变） |
| **11434** | Ollama | 本地 LLM 服务（保持不变） |

### 端口使用场景

```powershell
# 启动 API 主服务（终端 1）
$env:LUMO_API_PORT = "7901"
uv run python -m src.interfaces.api.main

# 启动 KB Admin 独立服务（独立终端）
$env:LUMO_API_MODE = "kb_admin"
$env:LUMO_API_PORT = "7902"
uv run python -m src.interfaces.api.main

# 启动 KB Query 独立服务（独立终端）
$env:LUMO_API_MODE = "kb_query"
$env:LUMO_API_PORT = "7903"
uv run python -m src.interfaces.api.main

# 启动前端开发服务器
cd src/interfaces/admin-web
npm run dev
# 访问 http://localhost:7906
```

---

## API 接口

### 基础信息
- **Base URL**: `http://localhost:7901/v1`
- **认证**: 暂无需认证（待完善）

### 核心接口

| 模块 | 方法 | 端点 | 说明 |
|------|------|------|------|
| **健康检查** | GET | `/health` | 服务状态 |
| **源文件管理** | POST | `/sources` | 上传源文件 |
| | GET | `/sources` | 列表查询 |
| | GET | `/sources/{id}` | 详情查询 |
| | DELETE | `/sources/{id}` | 删除文件 |
| **模板管理** | POST | `/templates` | 创建模板 |
| | GET | `/templates` | 列表查询 |
| | GET | `/templates/{id}` | 详情查询 |
| | PUT | `/templates/{id}` | 更新模板 |
| | DELETE | `/templates/{id}` | 删除模板 |
| **任务管理** | POST | `/jobs` | 创建任务 |
| | GET | `/jobs` | 列表查询 |
| | GET | `/jobs/{id}` | 详情查询 |
| | POST | `/jobs/{id}/cancel` | 取消任务 |
| **中间产物** | GET | `/intermediates/{id}/download` | 下载中间产物 |
| **提示词** | GET | `/prompts` | 列出可用提示词 |
| | POST | `/prompts/test` | 测试提示词 |
| **LLM 配置** | GET | `/llm/providers` | 列出 LLM 提供商 |
| | POST | `/llm/providers` | 添加提供商 |
| | GET | `/llm/callsites` | 列出调用点 |
| | PUT | `/llm/callsites/{id}/model` | 绑定模型 |

---

## 开发命令速查

| 场景 | 命令 |
|------|------|
| 安装后端依赖 | `uv pip install -e .` |
| 安装前端依赖 | `cd src/interfaces/admin-web; npm install` |
| 启动 API | `python -m src.interfaces.api.main` |
| 启动 Worker | `celery -A src.interfaces.worker.celery_app worker --loglevel=info` |
| 启动 Frontend | `cd src/interfaces/admin-web; npm run dev` |
| 运行测试 | `pytest tests/` |
| 代码检查 | `pytest tests/ --co -q` |

---

## 项目亮点

### 优势
- **架构清晰**：严格遵循分层架构，职责分离
- **AI 原生**：从设计就面向 LLM 应用（RAG、Prompt Engineering）
- **流水线设计**：支持文档处理的多阶段流水线
- **多 LLM 支持**：可配置 OpenAI/Ollama 等多种提供商
- **混合检索**：向量 + BM25 的混合检索策略
- **可观测性**：完善的日志和请求 ID 追踪
- **错误处理**：统一的错误响应格式

### 待完善
- 测试覆盖：当前测试目录存在但需补充
- 部署文档：缺少生产环境部署指南
- 监控告警：缺少健康检查和指标暴露
- 本地 MinerU：当前依赖在线服务，可考虑本地部署
- 权限管理：缺少用户认证和权限控制

---

## 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 项目快速开始指南
- [AGENTS.md](AGENTS.md) - 跨工具 Agent Skills 配置
- [docs/](docs/) - 过程文档和指南目录
- [00-目录说明.md](00-目录说明.md) - 项目根目录文件索引

---

**最后更新**: 2026-01-20

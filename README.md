# Lumoscribe2026 - 面向 AI 的文档生成平台

> **核心愿景**：图片型 PDF → OCR／清洗 → 知识库（向量＋BM25，多索引预建）→ 按模板 RAG＋LLM 生成单 HTML，并提供 API 与中台管理能力。

## 目录

- [项目概览](#项目概览)
- [技术栈](#技术栈)
- [架构设计](#架构设计)
- [前置要求](#前置要求)
- [开发环境部署](#开发环境部署)
- [生产环境部署](#生产环境部署)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [API 接口文档](#api-接口文档)
- [常见问题](#常见问题)
- [项目亮点](#项目亮点)
- [相关文档](#相关文档)

---

## 项目概览

| 指标 | 数值 |
|------|------|
| Python 文件数 | 82＋ |
| 领域实体 | 8 个核心实体 |
| API 路由模块 | 9 个 |
| 服务层模块 | 15＋ |
| LLM 调用点 | 15＋ 种场景 |

本项目是一个端到端的 AI 文档生成平台，支持从原始 PDF 文档到结构化输出的全流程自动化处理。核心功能包括 OCR 解析、文档清洗、语义分块、混合检索（向量＋BM25）、模板驱动的 RAG 生成，以及 Mermaid／Vega 图表渲染。

---

## 技术栈

| 层级 | 技术选型 | 版本要求 |
|------|----------|----------|
| 后端框架 | FastAPI | ≥ 0.115 |
| ORM | SQLAlchemy | ≥ 2.0 |
| 任务队列 | Celery ＋ Redis | ≥ 5.4，≥ 6.0 |
| 前端框架 | Next.js | ≥ 20.0 |
| 数据库 | SQLite | — |
| LLM 框架 | LangChain ＋ LlamaIndex | ≥ 1.0，≥ 0.14 |
| 向量数据库 | ChromaDB | ≥ 0.5 |
| Embedding | FlagEmbedding ＋ Sentence-Transformers | ≥ 1.3.5，≥ 3.0 |
| Python | — | 3.12 |
| Node.js | — | ≥ 20.0 |

---

## 架构设计

采用分层架构（Clean Architecture），各层职责清晰、依赖单向：

```
src/
├── domain/                      # 领域层：核心业务逻辑
│   └── entities/                # 实体定义（8 个核心实体）
├── application/                 # 应用层：用例编排
│   ├── services/               # 服务层（15＋ 业务逻辑）
│   ├── repositories/           # 数据仓储
│   └── schemas/                # 数据模型（DTO）
├── interfaces/                  # 接口层：外部交互
│   ├── api/                    # FastAPI 后端（9 个路由模块）
│   │   └── routes/             # API 路由定义
│   ├── worker/                 # Celery Worker
│   └── admin-web/              # Next.js 管理后台
└── shared/                      # 共享基础设施
    ├── config.py               # 配置管理
    ├── db.py                   # 数据库工具
    ├── logging.py              # 日志配置
    ├── storage.py              # 文件存储
    └── errors.py               # 错误定义
```

### 数据流概述

```
PDF 源文件 → MinerU OCR → 文本清洗 → 语义分块 → 向量化存储
                                                      ↓
生成请求 → 混合检索（向量＋BM25）→ Prompt 模板 → LLM 生成 → HTML 输出
```

---

## 前置要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.12 | 项目运行环境 |
| Node.js | ≥ 20.0 | 前端运行环境 |
| Redis | ≥ 6.0 | Celery 消息队列 |
| uv | ≥ 0.4 | 包管理工具（推荐） |
| Git | — | 代码版本控制 |

### 操作系统支持

| 操作系统 | GPU 支持 | 说明 |
|----------|----------|------|
| Windows 11 | CUDA 13.0 | 自动配置 PyTorch CUDA 版本 |
| macOS | Apple MPS | 自动配置 Metal Performance Shaders |
| Linux | CUDA | 手动配置 CUDA 版本 |

---

## 开发环境部署

### 步骤一：克隆项目

```powershell
git clone https://github.com/your-org/lumoscribe2026.git
cd lumoscribe2026
```

### 步骤二：安装后端依赖

本项目强制使用 [uv](https://github.com/astral-sh/uv) 进行包管理和环境隔离，以确保跨平台（Windows CUDA、macOS MPS）的一致性。

```powershell
# 1. 安装 uv（如果尚未安装）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 初始化并同步环境（自动安装 Python 3.12 和所有依赖）
uv sync

# 3. 激活虚拟环境
# Windows：
.venv\Scripts\activate

# macOS／Linux：
source .venv/bin/activate
```

> **说明**：Windows 上会自动配置 CUDA 13.0 版 PyTorch，macOS 上会自动配置 MPS 版。

### 步骤三：安装前端依赖

```powershell
cd src/interfaces/admin-web
npm install
cd ../../..
```

### 步骤四：启动 Redis

Redis 作为 Celery Worker 的消息队列，必须先于 Worker 启动。

```powershell
# 方式一：本地启动 Redis（默认端口 6379）
redis-server

# 方式二：使用 Docker（推荐）
docker run -d -p 6379:6379 --name lumoscribe-redis redis:7-alpine

# 方式三：使用 Docker Compose
docker compose up -d redis
```

### 步骤五：初始化数据库

```powershell
# 运行数据库初始化脚本
python scripts/init-db.py

# 验证数据库文件
ls -la data/
# 应看到 lumoscribe.db 文件
```

### 步骤六：配置环境变量

复制根目录的 `env.example` 内容到本地环境变量文件 `.env`：

```powershell
# 创建 .env 文件（从示例复制）
cp env.example .env

# 或直接在 PowerShell 中设置环境变量
$LUMO_SQLITE_PATH = "$PWD\data\lumoscribe.db"
$LUMO_STORAGE_ROOT = "$PWD\storage"
$LUMO_REDIS_URL = "redis://localhost:6379/0"
```

### 步骤七：启动服务

需要在不同的终端窗口中启动以下服务：

**终端一：启动 API 服务**

```powershell
# 默认端口：7901
python -m src.interfaces.api.main
```

**终端二：启动前端开发服务器**

```powershell
cd src/interfaces/admin-web
npm run dev
# 访问 http://localhost:7906
```

### 步骤八：验证安装

完成上述步骤后，访问以下地址验证安装是否成功：

| 服务 | 地址 | 预期结果 |
|------|------|----------|
| API 健康检查 | http://localhost:7901/v1/health | `{"status":"healthy"}` |
| 前端管理后台 | http://localhost:7906 | 显示管理仪表盘 |
| Redis 连接 | — | Worker 终端显示 "Connected to redis://localhost:6379/0" |

---

## 生产环境部署

### 方案一：Docker Compose 部署（推荐）

#### 1. 创建生产环境配置文件

创建 `docker-compose.prod.yml`：

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: lumoscribe-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: lumoscribe-api
    ports:
      - "7901:7901"
    environment:
      - LUMO_SQLITE_PATH=/app/data/lumoscribe.db
      - LUMO_STORAGE_ROOT=/app/storage
      - LUMO_REDIS_URL=redis://redis:6379/0
      - LUMO_API_PORT=7901
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
    depends_on:
      - redis
    restart: unless-stopped

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: lumoscribe-worker
    environment:
      - LUMO_SQLITE_PATH=/app/data/lumoscribe.db
      - LUMO_STORAGE_ROOT=/app/storage
      - LUMO_REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
    depends_on:
      - redis
    restart: unless-stopped

  frontend:
    build:
      context: ./src/interfaces/admin-web
      dockerfile: Dockerfile
    container_name: lumoscribe-frontend
    ports:
      - "7906:3000"
    restart: unless-stopped

volumes:
  redis_data:
```

#### 2. 创建 Dockerfile

**Dockerfile.api**：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml uv.lock* ./

# 安装 Python 依赖
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

# 复制应用代码
COPY src/ src/

# 创建必要目录
RUN mkdir -p /app/data /app/storage

EXPOSE 7901

CMD ["python", "-m", "src.interfaces.api.main"]
```

**Dockerfile.worker**：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY src/ src/

RUN mkdir -p /app/data /app/storage

CMD ["celery", "-A", "src.interfaces.worker.celery_app", "worker", "--loglevel=info"]
```

#### 3. 启动生产服务

```powershell
# 构建并启动所有服务
docker compose -f docker-compose.prod.yml up -d --build

# 查看日志
docker compose -f docker-compose.prod.yml logs -f

# 检查服务状态
docker compose -f docker-compose.prod.yml ps
```

### 方案二：手动部署

#### 1. 环境准备

```powershell
# 安装 Python 3.12
# 下载地址：https://www.python.org/downloads/

# 安装 Redis
# Windows：https://github.com/microsoftarchive/redis/releases
# Linux：sudo apt-get install redis-server
# macOS：brew install redis

# 安装 uv
pip install uv
```

#### 2. 部署后端

```powershell
# 克隆代码
git clone https://github.com/your-org/lumoscribe2026.git
cd lumoscribe2026

# 安装依赖（生产模式，不安装开发依赖）
uv sync --frozen --no-dev

# 创建必要目录
mkdir -p data storage logs

# 设置环境变量
export LUMO_SQLITE_PATH=/app/data/lumoscribe.db
export LUMO_STORAGE_ROOT=/app/storage
export LUMO_REDIS_URL=redis://localhost:6379/0
export LUMO_API_PORT=7901

# 初始化数据库
python scripts/init-db.py

# 使用 systemd 或 PM2 管理进程
# 推荐使用 supervisor 或 PM2
```

#### 3. 部署前端

```powershell
cd src/interfaces/admin-web

# 安装依赖
npm ci

# 构建生产版本
npm run build

# 使用 PM2 或 Nginx 托管
# npm install -g pm2
# pm2 start npm --name "lumoscribe-frontend" -- run start
```

#### 4. 配置 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name lumoscribe.example.com;

    # 前端静态资源
    location / {
        proxy_pass http://localhost:7906;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API 服务
    location /v1/ {
        proxy_pass http://localhost:7901/v1/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket 支持（如果有）
    location /ws/ {
        proxy_pass http://localhost:7901/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 方案三：Kubernetes 部署

项目提供 Kubernetes 部署配置，位于 `k8s/` 目录：

```
k8s/
├── namespace.yaml              # 命名空间配置
├── redis-deployment.yaml       # Redis 部署
├── api-deployment.yaml         # API 服务部署
├── worker-deployment.yaml      # Worker 部署
├── frontend-deployment.yaml    # 前端部署
├── service.yaml                # 服务暴露
└── ingress.yaml                # Ingress 配置
```

部署命令：

```powershell
kubectl apply -f k8s/
```

---

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `LUMO_SQLITE_PATH` | SQLite 数据库文件路径 | `./data/lumoscribe.db` | 是 |
| `LUMO_STORAGE_ROOT` | 文件存储根目录 | `./storage` | 是 |
| `LUMO_REDIS_URL` | Redis 连接 URL | `redis://localhost:6379/0` | 是 |
| `LUMO_API_PORT` | API 服务端口 | `7901` | 否 |
| `LUMO_API_MODE` | API 运行模式（`default`、`kb_admin`、`kb_query`） | `default` | 否 |
| `LUMO_LOG_LEVEL` | 日志级别（DEBUG、INFO、WARNING、ERROR） | `INFO` | 否 |
| `LUMO_MINERU_API_URL` | MinerU OCR 服务地址 | — | 否 |
| `LUMO_LLM_OPENAI_API_KEY` | OpenAI API Key | — | 否 |
| `LUMO_LLM_OPENAI_BASE_URL` | OpenAI Base URL | `https://api.openai.com/v1` | 否 |
| `LUMO_LLM_OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` | 否 |
| `LUMO_LLM_FLAGEMBEDDING_HOST` | FlagEmbedding 服务地址 | — | 否 |

### 端口配置

| 端口 | 服务 | 环境变量 | 说明 |
|------|------|----------|------|
| 7901 | API 主服务 | `LUMO_API_PORT` | 默认服务端口 |
| 7902 | KB Admin | `LUMO_API_PORT` | 知识库建库／管理专用（独立部署） |
| 7903 | KB Query | `LUMO_API_PORT` | 知识库查询专用（独立部署） |
| 7904 | FlagEmbedding | `LUMO_LLM_FLAGEMBEDDING_HOST` | 向量服务 |
| 7905 | MinerU | `LUMO_MINERU_API_URL` | PDF OCR 服务 |
| 7906 | Next.js 前端 | — | 前端开发服务器 |
| 7907 | LLM 测试 Mock | — | 测试环境 Mock 服务 |
| 6379 | Redis | — | 消息队列／缓存 |
| 11434 | Ollama | — | 本地 LLM 服务 |

### 启动命令示例

```powershell
# 启动 API 主服务
$env:LUMO_API_PORT = "7901"
uv run python -m src.interfaces.api.main

# 启动 KB Admin 独立服务
$env:LUMO_API_MODE = "kb_admin"
$env:LUMO_API_PORT = "7902"
uv run python -m src.interfaces.api.main

# 启动 KB Query 独立服务
$env:LUMO_API_MODE = "kb_query"
$env:LUMO_API_PORT = "7903"
uv run python -m src.interfaces.api.main

# 启动前端开发服务器
cd src/interfaces/admin-web
npm run dev
```

---

## 使用指南

### 完整使用流程

#### 步骤一：上传源文件

通过管理后台上传 PDF 源文件，或使用 API：

```bash
curl -X POST "http://localhost:7901/v1/sources" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

响应示例：

```json
{
  "id": "src_abc123",
  "filename": "document.pdf",
  "status": "pending",
  "created_at": "2026-01-27T10:00:00Z"
}
```

#### 步骤二：创建模板

在管理后台创建模板，或通过 API 上传模板文件。模板使用 Jinja2 语法，支持变量替换和条件逻辑。

模板示例：

```jinja2
# {{ title }}

## 概述
{{ overview }}

## 详细内容
{% for section in sections %}
### {{ section.title }}
{{ section.content }}
{% endfor %}

## 图表
{% if chart_data %}
```mermaid
{{ chart_data }}
```
{% endif %}
```

#### 步骤三：创建生成任务

```bash
curl -X POST "http://localhost:7901/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "src_abc123",
    "template_id": "tmpl_xyz789",
    "parameters": {
      "title": "技术文档",
      "overview": "这是一份技术文档"
    }
  }'
```

响应示例：

```json
{
  "id": "job_def456",
  "source_id": "src_abc123",
  "template_id": "tmpl_xyz789",
  "status": "processing",
  "created_at": "2026-01-27T10:05:00Z"
}
```

#### 步骤四：查询任务状态

```bash
curl "http://localhost:7901/v1/jobs/job_def456"
```

响应示例：

```json
{
  "id": "job_def456",
  "source_id": "src_abc123",
  "template_id": "tmpl_xyz789",
  "status": "completed",
  "progress": 100,
  "result": {
    "output_file_id": "tgt_ghi456"
  },
  "created_at": "2026-01-27T10:05:00Z",
  "completed_at": "2026-01-27T10:08:00Z"
}
```

#### 步骤五：下载生成结果

```bash
curl "http://localhost:7901/v1/targets/tgt_ghi456/download" \
  -o output.html
```

### 使用管理后台

1. 访问 http://localhost:7906
2. 左侧导航栏功能：
   - **文档管理**：上传和管理源文件
   - **模板管理**：创建和管理生成模板
   - **任务监控**：查看生成任务进度和状态
   - **LLM 配置**：配置 LLM 提供商和模型
   - **提示词管理**：编辑和测试提示词

---

## API 接口文档

### 基础信息

| 项目 | 值 |
|------|-----|
| Base URL | http://localhost:7901/v1 |
| 认证 | 暂无需认证（待完善） |
| 响应格式 | JSON |

### 错误响应格式

所有 API 错误返回统一的错误格式：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": [
      {
        "field": "file",
        "message": "文件大小超过限制"
      }
    ]
  }
}
```

### 接口列表

#### 健康检查

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康检查 |

**响应示例**：

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

#### 源文件管理

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/sources` | 上传源文件 |
| GET | `/sources` | 列表查询 |
| GET | `/sources/{id}` | 详情查询 |
| DELETE | `/sources/{id}` | 删除文件 |

**POST 请求示例**：

```bash
curl -X POST "http://localhost:7901/v1/sources" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**GET 列表请求示例**：

```bash
curl "http://localhost:7901/v1/sources?page=1&page_size=20&status=completed"
```

#### 模板管理

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/templates` | 创建模板 |
| GET | `/templates` | 列表查询 |
| GET | `/templates/{id}` | 详情查询 |
| PUT | `/templates/{id}` | 更新模板 |
| DELETE | `/templates/{id}` | 删除模板 |

**POST 请求示例**：

```bash
curl -X POST "http://localhost:7901/v1/templates" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "技术文档模板",
    "content": "# {{ title }}\n\n{{ content }}",
    "description": "用于生成技术文档的标准模板"
  }'
```

#### 任务管理

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/jobs` | 创建任务 |
| GET | `/jobs` | 列表查询 |
| GET | `/jobs/{id}` | 详情查询 |
| POST | `/jobs/{id}/cancel` | 取消任务 |

**POST 请求示例**：

```bash
curl -X POST "http://localhost:7901/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "src_abc123",
    "template_id": "tmpl_xyz789",
    "parameters": {
      "title": "我的文档",
      "overview": "文档概述"
    }
  }'
```

#### 中间产物

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/intermediates/{id}/download` | 下载中间产物 |
| GET | `/intermediates/{id}` | 查询中间产物信息 |

#### 提示词管理

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/prompts` | 列出可用提示词 |
| POST | `/prompts/test` | 测试提示词 |

#### LLM 配置

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/llm/providers` | 列出 LLM 提供商 |
| POST | `/llm/providers` | 添加提供商 |
| GET | `/llm/capabilities` | 列出模型能力 |
| GET | `/llm/callsites` | 列出调用点 |
| PUT | `/llm/callsites/{id}/model` | 绑定模型 |

---

## 常见问题

### Q1：启动 API 服务失败，提示端口已被占用

**问题**：启动 API 服务时提示 `Port 7901 is already in use`。

**解决方案**：

```powershell
# 查看占用端口的进程
netstat -ano | findstr :7901

# 或使用 PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 7901).OwningProcess

# 终止占用进程（根据实际 PID）
taskkill /PID <PID> /F

# 或修改端口后启动
$env:LUMO_API_PORT = "7902"
python -m src.interfaces.api.main
```

### Q2：Worker 无法连接 Redis

**问题**：Worker 启动后提示 `Connection refused` 或无法连接到 Redis。

**解决方案**：

```powershell
# 检查 Redis 服务状态
redis-cli ping
# 应返回 PONG

# 检查 Redis 连接 URL 配置
echo $LUMO_REDIS_URL

# 确认 Redis 服务已启动
# Windows：
redis-server.exe

# Docker 方式启动
docker run -d -p 6379:6379 redis:7-alpine
```

### Q3：上传文件失败，提示文件类型不支持

**问题**：上传文件时提示 `Unsupported file type`。

**解决方案**：

目前支持的源文件格式为 PDF（`.pdf`）。请确认文件格式正确：

```powershell
# 检查文件类型
file document.pdf
```

如果文件格式正确但仍失败，请检查文件是否损坏：

```powershell
# 使用 PDF 工具验证
# 安装 poppler-utils
# Windows：choco install poppler
# macOS：brew install poppler
# Linux：sudo apt-get install poppler-utils

pdftotext document.pdf - 2>&1 | head -n 10
```

### Q4：生成任务长时间处于 processing 状态

**问题**：任务创建后一直处于 processing 状态，没有进度更新。

**解决方案**：

1. 检查 Worker 服务是否正常运行
2. 检查 Redis 队列中是否有积压任务
3. 查看 Worker 日志获取详细错误信息

```powershell
# 查看 Worker 日志
# 如果使用 systemd
journalctl -u lumoscribe-worker -f

# 如果直接运行
# 查看 Worker 终端输出

# 检查 Redis 队列
redis-cli KEYS "*celery*"
```

### Q5：如何配置自定义 LLM 提供商

**问题**：需要使用除 OpenAI 和 Ollama 之外的其他 LLM 提供商。

**解决方案**：

1. 在管理后台的「LLM 配置」中添加提供商
2. 或通过 API 直接添加：

```bash
curl -X POST "http://localhost:7901/v1/llm/providers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_provider",
    "type": "openai_compatible",
    "base_url": "https://api.custom-provider.com/v1",
    "api_key": "your-api-key",
    "models": ["gpt-4", "gpt-3.5-turbo"]
  }'
```

### Q6：向量检索结果不准确

**问题**：检索结果与查询意图不匹配。

**解决方案**：

1. 调整检索参数（top_k、相似度阈值）
2. 检查文档分块是否合理
3. 优化 Prompt 模板
4. 考虑使用混合检索（向量＋BM25）

```bash
# 调整检索参数
curl -X POST "http://localhost:7901/v1/kb/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你的问题",
    "top_k": 10,
    "score_threshold": 0.5,
    "search_type": "hybrid"
  }'
```

### Q7：如何查看完整日志

**问题**：需要查看详细的运行日志进行问题排查。

**解决方案**：

```powershell
# 设置日志级别为 DEBUG
$env:LUMO_LOG_LEVEL = "DEBUG"

# 重新启动服务
python -m src.interfaces.api.main

# 日志会输出到控制台，可重定向到文件
python -m src.interfaces.api.main 2>&1 | Out-File -FilePath logs\api.log -Encoding UTF8
```

---

## 项目亮点

### 优势

- **架构清晰**：严格遵循分层架构，职责分离，便于维护和扩展
- **AI 原生**：从设计就面向 LLM 应用，内置 RAG、Prompt Engineering 支持
- **流水线设计**：支持文档处理的多阶段流水线，可灵活编排
- **多 LLM 支持**：可配置 OpenAI、Ollama 等多种提供商
- **混合检索**：向量＋BM25 的混合检索策略，支持多索引预建加载和去重
- **可观测性**：完善的日志和请求 ID 追踪，便于问题排查
- **错误处理**：统一的错误响应格式，便于客户端处理

### 待完善

| 功能 | 优先级 | 说明 |
|------|--------|------|
| 测试覆盖 | P1 | 补充单元测试和集成测试 |
| 监控告警 | P2 | 添加健康检查和指标暴露（Prometheus） |
| 本地 MinerU | P2 | 支持 MinerU 本地部署 |
| 权限管理 | P2 | 添加用户认证和权限控制 |
| 部署文档 | P2 | 完善生产环境部署指南（本文已补充） |

---

## 相关文档

| 文档 | 说明 |
|------|------|
| [QUICKSTART.md](QUICKSTART.md) | 项目快速开始指南 |
| [AGENTS.md](AGENTS.md) | 跨工具 Agent Skills 配置 |
| [docs/](docs/) | 过程文档和指南目录 |
| [00-目录说明.md](00-目录说明.md) | 项目根目录文件索引 |

---

**最后更新**：2026-01-27

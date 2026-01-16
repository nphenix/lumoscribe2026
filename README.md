# Lumoscribe2026

面向 AI 的文档生成平台（第一阶段）：**图片型 PDF → OCR/清洗 → 知识库（向量+BM25） → 按模板 RAG+LLM 生成单 HTML**，并提供 API 与中台管理能力。

## 当前进度

- ✅ 后端最小骨架（FastAPI + SQLite）
- ✅ 任务模型（Job）与基础接口（创建/查询）
- ⏳ Worker（Celery + Redis）与流水线能力（OCR/清洗/切块/索引/生成）
- ⏳ Next.js 中台

## 运行方式（开发态）

### 1) 配置环境变量

复制根目录的 `env.example` 内容到你的本地环境变量（或复制为本地 `.env` 自行加载），至少需要：

- `LUMO_SQLITE_PATH`
- `LUMO_STORAGE_ROOT`
- `LUMO_REDIS_URL`（Worker 阶段使用）

### 2) 启动 API

在仓库根目录执行：

```powershell
python -m src.interfaces.api.main
```

健康检查：

- `GET /v1/health`

## 目录结构

代码遵循分层放置：

- `src/domain/`：领域层
- `src/application/`：应用层
- `src/interfaces/`：接口层（API/Worker/AdminWeb）
- `src/shared/`：共享基础设施

---

最后更新：2026-01-16

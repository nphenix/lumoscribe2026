---
template: Plan
version: 1.0.0
lastUpdated: 2026-01-11
charter: ../constitution.md
---

# 实施计划：[功能名称]

**分支**: `feature/[issue-id]-[name]` | **日期**: [YYYY-MM-DD] | **规格**: [spec.md链接](./spec.md)

**输入**: 功能规格说明

---

## 摘要

[从功能规格中提取：主要需求 + 研究中的技术方法]

## 技术上下文

**语言/版本**: [例如：Python 3.11、Swift 5.9、Rust 1.75]  
**主要依赖**: [例如：FastAPI、UIKit、LLVM]  
**存储**: [例如：PostgreSQL、CoreData、文件] 或 `N/A`  
**测试**: [例如：pytest、XCTest、cargo test]  
**目标平台**: [例如：Linux 服务器、iOS 15+、WASM]  
**项目类型**: [单体/Web/移动]  
**性能目标**: [例如：1000 req/s、10k 行/秒、60 fps]  
**约束条件**: [例如：<200ms p95、<100MB 内存、离线可用]  
**规模/范围**: [例如：10k 用户、1M LOC、50 个屏幕]

## 项目结构

### 文档（本功能）

```
speckit/specs/features/[name]/
├── spec.md              # 功能规格（唯一入口，包含 API 章节）
├── plan.md              # 本文件（架构/策略/决策，无任务）
└── tasks.md             # 任务清单 + 验收标准
```

### 源代码（仓库根目录）

```
# 单个项目（默认）
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Web 应用（前端 + 后端）
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# 移动应用 + API
api/
└── [与后端结构相同]

ios/ 或 android/
└── [平台特定结构]
```

**结构决策**: [记录所选结构]

---

**版本**: 1.0.0 | **创建**: [YYYY-MM-DD]

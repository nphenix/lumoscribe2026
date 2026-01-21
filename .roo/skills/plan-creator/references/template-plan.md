# 计划模板参考

## 文件结构

```markdown
---
template: Plan
version: 1.0.0
lastUpdated: YYYY-MM-DD
charter: ../constitution.md
---

# 实施计划：[功能名称]

**分支**: `feature/[issue-id]-[name]`  
**日期**: [YYYY-MM-DD]  
**规格**: [spec.md链接](specs/feature/spec.md)

**输入**: 功能规格说明

---

## 摘要

[从功能规格中提取：主要需求 + 技术方法]

## 技术上下文

**语言/版本**: Python 3.11  
**主要依赖**: FastAPI 0.109、Pydantic 2.5  
**存储**: PostgreSQL 15、Redis 7  
**测试**: pytest 7.4  
**目标平台**: Linux 服务器  
**项目类型**: Web API  
**性能目标**: 1000 req/s  
**约束条件**: <200ms p95

## 项目结构

### 源代码

```
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/
```
```

## 技术字段

| 字段 | 说明 | 示例 |
|------|------|------|
| 语言/版本 | 编程语言及版本 | Python 3.11 |
| 主要依赖 | 关键框架和库 | FastAPI, Pydantic |
| 存储 | 数据库和存储方案 | PostgreSQL, Redis |
| 测试 | 测试框架 | pytest, XCTest |
| 目标平台 | 部署目标 | Linux, iOS 15+ |
| 项目类型 | 应用类型 | Web API, 移动应用 |
| 性能目标 | 性能指标 | 1000 req/s, 60 fps |
| 约束条件 | 限制条件 | <200ms p95 |

## 项目类型

| 类型 | 说明 | 目录结构 |
|------|------|----------|
| 单体 | 单个项目 | src/, tests/ |
| Web 应用 | 前端+后端 | backend/, frontend/ |
| 移动应用 | iOS 或 Android | api/, ios/ 或 android/ |

## 注意事项

- 技术选型要匹配团队能力
- 性能目标要可测量
- 约束条件要明确
- 结构决策要有理由

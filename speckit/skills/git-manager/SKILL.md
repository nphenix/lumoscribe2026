# Git Manager Skill

## 技能概述

**技能名称：** git-manager  
**版本：** 1.0.0  
**描述：** AI驱动的Git管理技能，自动生成规范的提交信息、管理分支、解决冲突  
**依赖：** MiniMax API (OpenAI兼容模式)

## 功能特性

### 核心能力

- **智能提交信息生成**：基于代码变更自动生成Conventional Commits格式的提交信息
- **分支管理**：创建、切换、删除、合并分支
- **变更分析**：分析未提交变更、对比分支差异
- **冲突解决**：辅助识别和解决合并冲突
- **最佳实践建议**：提供Git工作流优化建议

### 技术特性

- 支持MiniMax API (OpenAI兼容模式)
- 敏感数据自动过滤
- 本地处理优先
- 操作安全确认机制

## 使用方法

### 前置条件

```bash
# 安装依赖
npm install git-rewrite-commits openai
```

### 配置环境变量

```bash
# MiniMax API配置
export MINIMAX_API_KEY="your-api-key"
export MINIMAX_API_BASE="https://api.minimax.io/v1"

# 可选：配置默认模型
export GIT_AI_MODEL="MiniMax-M2.1"
```

### 集成到Git Hooks

```bash
# 安装Git hooks
npx git-rewrite-commits install-hooks

# 配置使用MiniMax API
git config hooks.commitProvider openai
git config hooks.apiKey ${MINIMAX_API_KEY}
```

## 配置文件

### .git-commit-ai.config.json

```json
{
  "provider": "openai",
  "apiBase": "https://api.minimax.io/v1",
  "model": "MiniMax-M2.1",
  "template": "type(scope): description",
  "language": "zh-CN",
  "safety": {
    "filterSensitive": true,
    "requireConfirmation": true,
    "autoBackup": true
  },
  "types": ["feat", "fix", "docs", "style", "refactor", "test", "chore"],
  "scopes": ["skill-generator", "plan-creator", "task-breakdown", "change-manager", "debug-diagnostic", "templates"]
}
```

## 贡献指南

### 提交信息格式

请遵循Conventional Commits规范：

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### 示例

```
feat(git-manager): add MiniMax API integration

- 支持MiniMax OpenAI兼容模式
- 自动生成规范的提交信息
- 敏感数据过滤

Closes #123
```

## 版本历史

- **1.0.0** (2026-01-12): 初始版本，支持提交信息生成和基础Git操作
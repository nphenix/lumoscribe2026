# 提示词管理与开发规范

> 本文档定义了项目中的提示词（Prompt）管理策略、开发流程与最佳实践。

## 1. 核心原则

### P11.1 禁止硬编码 (No Hard-coding)
- **严禁**在业务逻辑代码（Service/Controller）中直接拼接 Prompt 字符串。
- 所有 Prompt 必须通过配置化方式管理，至少应定义在常量文件中，最终目标是数据库化。

### P11.2 数据库为单一事实来源 (DB as Source of Truth)
- 运行时系统**必须**优先从数据库加载 Prompt。
- 代码中的常量仅作为**初始化种子 (Seed)** 或 **灾难回退 (Fallback)**。

### P11.3 变更可追溯 (Traceability)
- 提示词的修改必须产生新的版本号 (Version)。
- 生产环境禁止修改已发布的版本，只能新增版本。

## 2. 开发流程 (Code-First Seed 模式)

### 步骤 1：定义常量 (Distributed Definition)
在业务模块目录下创建 `prompts.py`，定义 Scope 和 Seed。

```python
# src/application/services/my_feature/prompts.py

SCOPE_MY_FEATURE = "my_feature:action"

PROMPTS = {
    SCOPE_MY_FEATURE: {
        "description": "功能说明",
        "format": "text",
        "content": "模板内容..."
    }
}
```

### 步骤 2：注册常量 (Centralized Registry)
在 `src/shared/constants/prompts.py` 中引入并聚合。

```python
# src/shared/constants/prompts.py
from src.application.services.my_feature.prompts import PROMPTS as MY_FEATURE_PROMPTS

DEFAULT_PROMPTS = {
    # ...
    **MY_FEATURE_PROMPTS,
}
```

### 步骤 3：播种数据库 (Seed)
在 `scripts/init-db.py` 的 `seed_prompts` 函数中（或自动同步脚本中），确保新定义的常量能写入数据库。

```python
# 现有的 seed_prompts 逻辑会自动处理
# 只需要确保常量被正确引入
```

### 步骤 3：业务调用 (Service Injection)
在业务 Service 中注入 `PromptService`，并使用它获取 Prompt。

```python
class MyService:
    def __init__(self, prompt_service: PromptService, ...):
        self.prompt_service = prompt_service

    async def run(self):
        # 1. 获取 Prompt
        prompt = self.prompt_service.get_active_prompt(SCOPE_MY_FEATURE)
        template = prompt.content if prompt else DEFAULT_PROMPTS[SCOPE_MY_FEATURE]["content"]
        
        # 2. 渲染 (推荐使用 replace 而非 format，避免 JSON 冲突)
        final_prompt = template.replace("{variable}", "value")
        
        # 3. 调用 LLM
        ...
```

## 3. 命名规范

Scope 命名采用 `module:action` 或 `module:submodule:action` 格式，小写，冒号分隔。

- `doc_cleaning:clean_text`
- `chart_extraction:extract_json`
- `kb:query:rewrite`

## 4. 维护与运营

- **日常调整**：直接在 Admin Web 界面“提示词管理”中编辑并发布新版本。
- **代码同步**：如果 Admin Web 上的调整经过验证效果更好，**建议**定期反向同步回 `src/shared/constants/prompts.py`，保持代码中的 Seed 也是较优版本（虽非强制，但推荐）。

## 5. Review Checklist

代码审查时请检查：
- [ ] 是否引入了新的 Prompt？如果是，是否添加到了 `constants/prompts.py`？
- [ ] 业务代码是否注入了 `PromptService`？
- [ ] 是否存在硬编码的 Prompt 字符串？

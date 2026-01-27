#!/usr/bin/env python
"""大纲智能识别功能完整冒烟测试脚本 - 调用 LLM 实际润色。"""

import asyncio
import sys
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, "f:/lumoscribe2026")


def test_service_instantiation():
    """测试服务实例化。"""
    print("=" * 60)
    print("测试 1: 服务实例化")
    print("=" * 60)
    
    from src.application.services.outline_polish.outline_polish_service import OutlinePolishService
    from src.application.services.llm_runtime_service import LLMRuntimeService
    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
    from src.application.repositories.llm_provider_repository import LLMProviderRepository
    from src.application.repositories.prompt_repository import PromptRepository
    from src.shared.db import make_engine, make_session_factory
    from src.shared.config import get_settings
    
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    
    session = session_factory()
    
    prompt_repo = PromptRepository(session)
    call_site_repo = LLMCallSiteRepository(session)
    capability_repo = LLMCapabilityRepository(session)
    provider_repo = LLMProviderRepository(session)
    
    llm_runtime = LLMRuntimeService(
        provider_repository=provider_repo,
        capability_repository=capability_repo,
        callsite_repository=call_site_repo,
        prompt_repository=prompt_repo,
    )
    
    service = OutlinePolishService(
        prompt_service=prompt_repo,
        llm_call_site_repository=call_site_repo,
        llm_runtime_service=llm_runtime,
    )
    
    print("✓ OutlinePolishService 实例化成功")
    return service


async def test_polish_with_llm(service):
    """使用 LLM 测试大纲润色。"""
    print("\n" + "=" * 60)
    print("测试 2: LLM 实际调用润色")
    print("=" * 60)
    
    from src.application.services.outline_polish.schema import OutlinePolishInput
    
    # 测试用例：混合章节和要求的大纲
    test_outline = """1. 行业发展背景
2. 市场竞争格局（请详细分析主要参与者）
3. 市场发展趋势
请确保：语言简洁专业，包含数据支撑"""
    
    input_data = OutlinePolishInput(
        outline=test_outline,
        industry="储能行业",
        report_type="市场研究报告",
    )
    
    print(f"输入大纲:\n{test_outline}\n")
    
    result = await service.polish_outline(input_data)
    
    if not result.success:
        print(f"❌ 润色失败: {result.error}")
        return False
    
    output = result.output
    print("=" * 60)
    print("润色结果")
    print("=" * 60)
    
    print(f"\n优化后大纲:\n{output.polished_outline}")
    
    print(f"\n修改摘要: {output.changes_summary}")
    print(f"结构完整: {output.structure_integrity}")
    print(f"关键词保留: {output.core_keywords_preserved}")
    print(f"识别出的要求: {output.recognized_requirements}")
    print(f"原始章节结构: {output.original_structure}")
    
    return True


def generate_markdown_report(result):
    """生成 Markdown 格式的测试报告。"""
    print("\n" + "=" * 60)
    print("测试 3: 生成 Markdown 报告")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# 大纲智能识别功能测试报告

> 生成时间: {timestamp}

## 测试结果

- **状态**: 🎉 通过
- **时间**: {timestamp}

## 输入大纲

```
{result.input.outline}
```

## 行业配置

- **行业**: {result.input.industry}
- **报告类型**: {result.input.report_type}
- **语言**: {result.input.language}
- **风格**: {result.input.style}

## 润色结果

### 优化后大纲

```markdown
{result.output.polished_outline}
```

### 修改摘要

{chr(10).join(f"- {item}" for item in result.output.changes_summary)}

### 检查结果

| 检查项 | 结果 |
|--------|------|
| 结构完整性 | {'✓' if result.output.structure_integrity else '✗'} |
| 核心关键词保留 | {'✓' if result.output.core_keywords_preserved else '✗'} |

### 智能识别结果

#### 识别出的要求

{chr(10).join(f"- {item}" for item in result.output.recognized_requirements)}

#### 原始章节结构

{chr(10).join(f"- {item}" for item in result.output.original_structure)}

---
*报告由冒烟测试脚本自动生成*
"""
    
    report_path = "test-outline-report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"✓ Markdown 报告已生成: {report_path}")
    return report_path


async def main():
    """主测试函数。"""
    print("\n" + "=" * 60)
    print("大纲智能识别功能 - 完整冒烟测试 (LLM 调用)")
    print("=" * 60 + "\n")
    
    try:
        # 测试 1: 服务实例化
        service = test_service_instantiation()
        
        # 测试 2: LLM 实际调用润色
        success = await test_polish_with_llm(service)
        if not success:
            print("\n❌ LLM 调用测试失败")
            return 1
        
        # 获取润色结果
        from src.application.services.outline_polish.schema import OutlinePolishInput
        
        test_outline = """1. 行业发展背景
2. 市场竞争格局（请详细分析主要参与者）
3. 市场发展趋势
请确保：语言简洁专业，包含数据支撑"""
        
        input_data = OutlinePolishInput(
            outline=test_outline,
            industry="储能行业",
            report_type="市场研究报告",
        )
        
        result = await service.polish_outline(input_data)
        
        # 测试 3: 生成 Markdown 报告
        report_path = generate_markdown_report(result)
        
        print("\n" + "=" * 60)
        print("🎉 完整冒烟测试通过！")
        print("=" * 60)
        print(f"\nMarkdown 报告: {report_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

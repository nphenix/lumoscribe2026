#!/usr/bin/env python
"""大纲智能识别功能冒烟测试脚本。"""

import asyncio
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, "f:/lumoscribe2026")

from src.application.services.outline_polish.schema import OutlinePolishInput


def test_schema_fields():
    """测试 Schema 字段是否正确扩展。"""
    print("=" * 60)
    print("测试 1: Schema 字段验证")
    print("=" * 60)
    
    # 测试 OutlinePolishOutput 包含新字段
    from src.application.services.outline_polish.schema import OutlinePolishOutput
    
    output = OutlinePolishOutput(
        polished_outline="# 优化后大纲\n## 章节1",
        changes_summary=["修改1"],
        structure_integrity=True,
        core_keywords_preserved=True,
        recognized_requirements=["要求1", "要求2"],
        original_structure=["1. 章节1", "2. 章节2"],
    )
    
    print(f"✓ polished_outline: {output.polished_outline[:30]}...")
    print(f"✓ recognized_requirements: {output.recognized_requirements}")
    print(f"✓ original_structure: {output.original_structure}")
    print("Schema 字段测试通过！\n")


def test_prompts_import():
    """测试提示词模板是否正确加载。"""
    print("=" * 60)
    print("测试 2: 提示词模板加载")
    print("=" * 60)
    
    from src.application.services.outline_polish.prompts import (
        SYSTEM_PROMPT_TEMPLATE,
        USER_PROMPT_TEMPLATE,
    )
    
    # 检查 SYSTEM_PROMPT_TEMPLATE 包含关键识别规则
    assert "行级语义识别" in SYSTEM_PROMPT_TEMPLATE, "缺少行级语义识别规则"
    assert "章节行识别" in SYSTEM_PROMPT_TEMPLATE, "缺少章节行识别规则"
    assert "要求行识别" in SYSTEM_PROMPT_TEMPLATE, "缺少要求行识别规则"
    assert "recognized_requirements" in SYSTEM_PROMPT_TEMPLATE, "缺少 recognized_requirements 字段说明"
    assert "original_structure" in SYSTEM_PROMPT_TEMPLATE, "缺少 original_structure 字段说明"
    
    print("✓ SYSTEM_PROMPT_TEMPLATE 包含智能识别规则")
    
    # 检查 USER_PROMPT_TEMPLATE 包含识别任务说明
    assert "行级语义分析" in USER_PROMPT_TEMPLATE, "缺少行级语义分析说明"
    assert "提取与分离" in USER_PROMPT_TEMPLATE, "缺少提取与分离说明"
    
    print("✓ USER_PROMPT_TEMPLATE 包含识别任务说明")
    print("提示词模板测试通过！\n")


def test_polished_outline_schema():
    """测试 PolishedOutline Schema 是否正确扩展。"""
    print("=" * 60)
    print("测试 3: PolishedOutline Schema 验证")
    print("=" * 60)
    
    from src.application.services.outline_polish.outline_polish_service import PolishedOutline
    
    # 测试使用新字段创建实例
    outline = PolishedOutline(
        polished_outline="# 优化后大纲\n## 章节1",
        changes_summary=["修改1"],
        structure_integrity=True,
        core_keywords_preserved=True,
        recognized_requirements=["要求1", "要求2"],
        original_structure=["1. 章节1", "2. 章节2"],
    )
    
    print(f"✓ polished_outline: {outline.polished_outline[:30]}...")
    print(f"✓ recognized_requirements: {outline.recognized_requirements}")
    print(f"✓ original_structure: {outline.original_structure}")
    print("PolishedOutline Schema 测试通过！\n")


def test_input_schema():
    """测试输入 Schema。"""
    print("=" * 60)
    print("测试 4: Input Schema 验证")
    print("=" * 60)
    
    input_data = OutlinePolishInput(
        outline="""1. 行业发展背景
2. 市场竞争格局（请详细分析主要参与者）
3. 市场发展趋势
请确保：语言简洁专业，包含数据支撑""",
        industry="储能行业",
        report_type="市场研究报告",
    )
    
    print(f"✓ outline: {input_data.outline[:50]}...")
    print(f"✓ industry: {input_data.industry}")
    print(f"✓ report_type: {input_data.report_type}")
    print("Input Schema 测试通过！\n")


async def main():
    """主测试函数。"""
    print("\n" + "=" * 60)
    print("大纲智能识别功能 - 冒烟测试")
    print("=" * 60 + "\n")
    
    try:
        test_schema_fields()
        test_prompts_import()
        test_polished_outline_schema()
        test_input_schema()
        
        print("=" * 60)
        print("🎉 所有冒烟测试通过！")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

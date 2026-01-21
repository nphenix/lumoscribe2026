"""T096: 生成白皮书功能测试脚本。

测试范围：
- 模板 section 解析完整性
- RAG 检索上下文质量
- LLM 生成内容与模板格式对齐
- 图表渲染正确性
- 大纲润色功能验证（polish_outline）

使用方法:
    pytest tests/test_content_generation.py -v --tb=short
"""

from __future__ import annotations

import os
import sys
fromfrom unittest.mock import pathlib import Path
 AsyncMock, MagicMock, Mock, patch

import pytest

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ContentGenerationTestConfig:
    """内容生成测试配置类。"""

    def __init__(self):
        self.template_dir = os.getenv("TEMPLATE_DIR", str(PROJECT_ROOT / "data" / "templates"))
        self.collection_name = os.getenv("COLLECTION_NAME", "default")


class TestContentGenerationService:
    """内容生成服务测试类。"""

    @pytest.fixture
    def sample_template(self, tmp_path):
        """创建示例模板文件。"""
        template_content = """# 产品白皮书

## 一、产品概述

{{product_overview}}

## 二、核心功能

### 2.1 功能一：智能文档处理

{{function_1_description}}

### 2.2 功能二：图表自动生成

{{function_2_description}}

## 三、技术架构

{{architecture_description}}

## 四、应用场景

{{use_cases}}

## 五、总结

{{conclusion}}
"""
        template_dir = tmp_path / "templates"
        template_dir.mkdir(exist_ok=True)
        
        template_path = template_dir / "test_template.md"
        template_path.write_text(template_content, encoding="utf-8")
        
        return template_path

    @pytest.fixture
    def mock_template(self, tmp_path, sample_template):
        """创建 Mock 模板实体。"""
        from src.domain.entities.template import Template, TemplateType

        # 创建模板目录
        template_dir = tmp_path / "template_storage"
        template_dir.mkdir(exist_ok=True)

        # 复制模板文件
        import shutil
        shutil.copy(sample_template, template_dir / "test_template.md")

        return Template(
            id="test-template-001",
            workspace_id="default",
            original_filename="test_template.md",
            storage_path=str(template_dir / "test_template.md"),
            type=TemplateType.CUSTOM,
            version=1,
            is_locked=False,
            preprocess_status="completed",
            preprocess_result={"placeholder_count": 6},
        )

    @pytest.fixture
    def sample_sections(self):
        """获取示例模板 sections。"""
        from src.application.services.content_generation_service import TemplateSection

        return [
            TemplateSection(
                section_id="section-1",
                title="产品概述",
                content="## 产品概述\n\n{{product_overview}}",
                placeholders=["product_overview"],
                chart_placeholders=[],
                order=0,
            ),
            TemplateSection(
                section_id="section-2",
                title="核心功能",
                content="## 核心功能\n\n### 2.1 功能一：智能文档处理\n\n{{function_1_description}}\n\n### 2.2 功能二：图表自动生成\n\n{{function_2_description}}",
                placeholders=["function_1_description", "function_2_description"],
                chart_placeholders=[],
                order=1,
            ),
        ]

    def test_template_section_parsing(self, mock_template, sample_template):
        """测试模板 section 解析。"""
        from src.application.services.content_generation_service import ContentGenerationService

        # Mock 所有依赖
        mock_template_service = Mock()
        mock_template_service.get_template = Mock(return_value=mock_template)

        mock_hybrid_search = Mock()
        mock_llm_runtime = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        sections = service.parse_template_sections(mock_template)

        assert len(sections) > 0

        # 验证 section 属性
        for section in sections:
            assert section.section_id is not None
            assert section.title is not None
            assert section.content is not None
            assert section.order >= 0

    def test_placeholder_extraction(self):
        """测试占位符提取。"""
        from src.application.services.content_generation_service import ContentGenerationService

        mock_template_service = Mock()
        mock_hybrid_search = Mock()
        mock_llm_runtime = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        content = "这是 {{placeholder1}} 和 {{placeholder2}} 的内容"
        placeholders = service._extract_placeholders(content)

        assert len(placeholders) == 2
        assert "placeholder1" in placeholders
        assert "placeholder2" in placeholders

    def test_chart_placeholder_extraction(self):
        """测试图表占位符提取。"""
        from src.application.services.content_generation_service import ContentGenerationService

        mock_template_service = Mock()
        mock_hybrid_search = Mock()
        mock_llm_runtime = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        content = "图表展示: {{chart:sales_chart}} 和 {{chart:revenue_chart}}"
        chart_placeholders = service._extract_chart_placeholders(content)

        assert len(chart_placeholders) == 2
        assert "sales_chart" in chart_placeholders
        assert "revenue_chart" in chart_placeholders

    @pytest.mark.asyncio
    async def test_rag_context_assembly(self, sample_sections):
        """测试 RAG 上下文组装。"""
        from src.application.services.content_generation_service import ContentGenerationService
        from src.application.services.hybrid_search_service import HybridSearchService
        from src.application.schemas.ingest import SearchResult, SearchStrategy

        mock_template_service = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        # Mock 混合检索服务
        mock_hybrid_search = Mock(spec=HybridSearchService)
        mock_hybrid_search.search = AsyncMock(return_value=SearchResult(
            chunk_id="context-1",
            content="这是检索到的上下文内容，包含产品的详细信息。",
            score=0.95,
            search_type=SearchStrategy.VECTOR,
            source_file_id=1,
            metadata={"source_file_id": 1},
            rank=0,
        ))

        mock_llm_runtime = Mock()

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        # 第一个 section 有一个占位符
        context = await service.assemble_context_for_section(
            section=sample_sections[0],
            collection_name="default"
        )

        assert context is not None
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_section_content_generation(self, sample_sections):
        """测试 section 内容生成。"""
        from src.application.services.content_generation_service import ContentGenerationService
        from src.application.schemas.chart_spec import ChartConfig

        mock_template_service = Mock()
        mock_hybrid_search = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        # Mock LLM 运行时
        mock_runnable = Mock()
        mock_runnable.invoke = Mock(return_value="这是生成的产品概述内容。")
        mock_llm_runtime = Mock()
        mock_llm_runtime.build_runnable_for_callsite = Mock(return_value=mock_runnable)

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        result = service.generate_section_content(
            section=sample_sections[0],
            context="产品概述上下文内容",
            chart_configs={},
            collection_name="default"
        )

        assert result.section_id == sample_sections[0].section_id
        assert result.title == sample_sections[0].title
        assert result.content is not None
        assert result.tokens_used >= 0

    @pytest.mark.asyncio
    async def test_outline_polishing(self):
        """测试大纲润色功能。"""
        from src.application.services.content_generation_service import ContentGenerationService

        mock_template_service = Mock()
        mock_hybrid_search = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        # Mock LLM 运行时
        mock_runnable = Mock()
        mock_runnable.invoke = Mock(return_value="""# 产品白皮书

## 一、概述
## 二、功能
## 三、技术
## 四、总结
""")
        mock_llm_runtime = Mock()
        mock_llm_runtime.build_runnable_for_callsite = Mock(return_value=mock_runnable)

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        raw_outline = """# 产品白皮书

## 概述

## 功能

## 技术

## 总结
"""

        polished = service.polish_outline(raw_outline)

        assert polished is not None
        assert len(polished) > 0

    def test_outline_polishing_empty_handling(self):
        """测试空大纲处理。"""
        from src.application.services.content_generation_service import (
            ContentGenerationService,
            ContentGenerationError,
        )

        mock_template_service = Mock()
        mock_hybrid_search = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=Mock(),
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        # 测试空大纲
        with pytest.raises(ContentGenerationError) as exc_info:
            service.polish_outline("")

        assert exc_info.value.code == "outline_empty"

    @pytest.mark.asyncio
    async def test_full_content_generation_flow(self, mock_template, sample_sections):
        """测试完整内容生成流程。"""
        from src.application.services.content_generation_service import ContentGenerationService
        from src.application.services.hybrid_search_service import HybridSearchService
        from src.application.schemas.ingest import SearchResult, SearchStrategy

        mock_template_service = Mock()
        mock_template_service.get_template = Mock(return_value=mock_template)
        mock_template_service.preprocess_template = Mock(return_value={
            "overall_status": "completed",
            "message": "预处理完成",
            "placeholders": ["product_overview", "function_1_description"]
        })

        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        # Mock 混合检索
        mock_hybrid_search = Mock(spec=HybridSearchService)
        mock_hybrid_search.search = AsyncMock(return_value=SearchResult(
            chunk_id="full-test-1",
            content="完整测试的检索内容。",
            score=0.9,
            search_type=SearchStrategy.VECTOR,
            source_file_id=1,
            metadata={},
            rank=0,
        ))

        # Mock LLM
        mock_runnable = Mock()
        mock_runnable.invoke = Mock(return_value="生成的内容")
        mock_llm_runtime = Mock()
        mock_llm_runtime.build_runnable_for_callsite = Mock(return_value=mock_runnable)

        service = ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

        # Mock section 解析
        with patch.object(service, "parse_template_sections", return_value=sample_sections):
            with patch.object(service, "assemble_context_for_section", return_value="上下文"):
                with patch.object(service, "generate_section_content", return_value=Mock(
                    section_id="test",
                    title="测试",
                    content="生成内容",
                    rendered_charts={},
                    sources=[],
                    tokens_used=100,
                    generation_time_ms=500,
                )):
                    result = service.generate_content(
                        template_id="test-template-001",
                        collection_name="default"
                    )

        assert result.template_id == "test-template-001"
        assert result.html_content is not None
        assert len(result.html_content) > 0


class TestMarkdownToHtml:
    """Markdown 转 HTML 测试类。"""

    @pytest.fixture
    def content_service(self):
        """获取内容生成服务（用于测试转换方法）。"""
        from src.application.services.content_generation_service import ContentGenerationService

        mock_template_service = Mock()
        mock_hybrid_search = Mock()
        mock_llm_runtime = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        return ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

    def test_heading_conversion(self, content_service):
        """测试标题转换。"""
        markdown = """# 一级标题

## 二级标题

### 三级标题
"""

        html = content_service._markdown_to_html(markdown)

        assert "<h1>一级标题</h1>" in html
        assert "<h2>二级标题</h2>" in html
        assert "<h3>三级标题</h3>" in html

    def test_bold_conversion(self, content_service):
        """测试粗体转换。"""
        markdown = "这是 **粗体文字** 和 __也是粗体__"

        html = content_service._markdown_to_html(markdown)

        assert "<strong>粗体文字</strong>" in html
        assert "<strong>也是粗体</strong>" in html

    def test_italic_conversion(self, content_service):
        """测试斜体转换。"""
        markdown = "这是 *斜体文字* 和 _也是斜体_"

        html = content_service._markdown_to_html(markdown)

        assert "<em>斜体文字</em>" in html
        assert "<em>也是斜体</em>" in html

    def test_list_conversion(self, content_service):
        """测试列表转换。"""
        markdown = """- 项目一
- 项目二
- 项目三
"""

        html = content_service._markdown_to_html(markdown)

        assert "<li>项目一</li>" in html

    def test_ordered_list_conversion(self, content_service):
        """测试有序列表转换。"""
        markdown = """1. 第一项
2. 第二项
3. 第三项
"""

        html = content_service._markdown_to_html(markdown)

        assert "<li>第二项</li>" in html

    def test_paragraph_conversion(self, content_service):
        """测试段落转换。"""
        markdown = """这是第一段。

这是第二段。
"""

        html = content_service._markdown_to_html(markdown)

        assert "<p>" in html


class TestHtmlRendering:
    """HTML 渲染测试类。"""

    @pytest.fixture
    def content_service(self):
        """获取内容生成服务。"""
        from src.application.services.content_generation_service import ContentGenerationService

        mock_template_service = Mock()
        mock_hybrid_search = Mock()
        mock_llm_runtime = Mock()
        mock_chart_renderer = Mock()
        mock_template_repo = Mock()

        return ContentGenerationService(
            template_service=mock_template_service,
            hybrid_search_service=mock_hybrid_search,
            llm_runtime_service=mock_llm_runtime,
            chart_renderer_service=mock_chart_renderer,
            template_repository=mock_template_repo
        )

    def test_default_styles(self, content_service):
        """测试默认样式生成。"""
        styles = content_service._get_default_styles()

        assert "body" in styles
        assert "font-family" in styles
        assert "h1" in styles or "h2" in styles

    def test_section_html_rendering(self, content_service):
        """测试 section HTML 渲染。"""
        from src.application.services.content_generation_service import SectionGenerationResult

        mock_result = SectionGenerationResult(
            section_id="test-section",
            title="测试章节",
            content="这是章节内容",
            rendered_charts={},
            sources=[],
            tokens_used=50,
            generation_time_ms=100,
        )

        html_parts = content_service._render_section_html(mock_result)

        assert any("<section" in part for part in html_parts)
        assert any("test-section" in part for part in html_parts)

    def test_final_html_rendering(self, content_service, mock_template):
        """测试最终 HTML 渲染。"""
        from src.application.services.content_generation_service import SectionGenerationResult
        from src.domain.entities.template import TemplateType

        mock_template.type = TemplateType.CUSTOM

        section_results = [
            SectionGenerationResult(
                section_id="section-1",
                title="第一章",
                content="第一章内容",
                rendered_charts={},
                sources=[],
                tokens_used=50,
                generation_time_ms=100,
            ),
            SectionGenerationResult(
                section_id="section-2",
                title="第二章",
                content="第二章内容",
                rendered_charts={},
                sources=[],
                tokens_used=60,
                generation_time_ms=120,
            ),
        ]

        html = content_service.render_final_html(mock_template, section_results)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "第一章" in html
        assert "第二章" in html


class TestChartRendering:
    """图表渲染测试类。"""

    @pytest.fixture
    def chart_renderer_service(self):
        """获取图表渲染服务。"""
        from src.application.services.chart_renderer_service import ChartRendererService

        return ChartRendererService()

    @pytest.fixture
    def chart_spec(self):
        """获取图表规格。"""
        from src.application.schemas.chart_spec import ChartConfig, ChartType

        return ChartConfig(
            chart_id="test_chart",
            chart_type=ChartType.BAR,
            title="测试图表",
            data={
                "x_axis": {"label": "类别", "values": ["A", "B", "C"]},
                "y_axis": {"label": "数值", "values": [10, 20, 30]},
            },
        )

    def test_echarts_rendering(self, chart_renderer_service, chart_spec):
        """测试 ECharts 渲染。"""
        result = chart_renderer_service.render_template_snippet(
            config=chart_spec,
            library="echarts"
        )

        assert result is not None
        assert result.container_html is not None
        assert result.script_html is not None
        assert "test_chart" in result.container_html or "test_chart" in result.script_html

    def test_chart_js_rendering(self, chart_renderer_service, chart_spec):
        """测试 Chart.js 渲染。"""
        result = chart_renderer_service.render_template_snippet(
            config=chart_spec,
            library="chartjs"
        )

        assert result is not None
        assert result.container_html is not None

    def test_different_chart_types(self, chart_renderer_service):
        """测试不同图表类型渲染。"""
        from src.application.schemas.chart_spec import ChartConfig, ChartType

        chart_types = [
            ChartType.BAR,
            ChartType.LINE,
            ChartType.PIE,
            ChartType.SCATTER,
        ]

        for chart_type in chart_types:
            config = ChartConfig(
                chart_id=f"test_{chart_type.value}",
                chart_type=chart_type,
                title=f"测试{chart_type.value}",
                data={
                    "x_axis": {"label": "X", "values": ["1", "2", "3"]},
                    "y_axis": {"label": "Y", "values": [1, 2, 3]},
                },
            )

            result = chart_renderer_service.render_template_snippet(
                config=config,
                library="echarts"
            )

            assert result is not None, f"Failed to render {chart_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

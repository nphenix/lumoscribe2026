"""文档清洗服务（T031）。

提供两层清洗策略：
1. 规则过滤：移除广告、噪声、重复内容
2. LLM 智能清洗：使用推理 LLM 保留结构，移除噪声
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.application.repositories.intermediate_artifact_repository import (
    IntermediateArtifactRepository,
)
from src.application.schemas.document_cleaning import (
    ChartType,
    ChartData,
    ChartJSONArtifact,
    ChartExtractionResult,
    CleaningOptions,
    CleaningResult,
    CleaningStats,
    CleanedDocumentArtifact,
)
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.prompt_service import PromptService
from src.domain.entities.intermediate_artifact import IntermediateType
from src.shared.constants.prompts import (
    DEFAULT_PROMPTS,
    SCOPE_CHART_EXTRACTION,
    SCOPE_DOC_CLEANING,
)
from src.shared.logging import logger


class DocumentCleaningService:
    """文档清洗服务。

    提供两层清洗策略，产出 cleaned_doc 中间产物。
    """

    # 广告特征模式
    AD_PATTERNS = [
        r"广告",  # 通用广告标记
        r"Sponsored",  # 赞助内容
        r"广告位",  # 广告位占位
        r"推广",  # 推广内容
        r"VIP",  # VIP 会员广告
        r"开通会员",  # 会员推广
        r"扫码关注",  # 二维码广告
        r"微信公众号",  # 公众号推广
        r"客服电话",  # 客服广告
        r"咨询热线",  # 咨询热线广告
    ]

    # 噪声模式（页眉页脚、页码、目录等）
    NOISE_PATTERNS = [
        r"第\s*\d+\s*页",  # 页码
        r"Page\s*\d+",  # 英文页码
        r"^\s*[\d\-]+\s*$",  # 纯数字行（可能是页码）
        r"^[-—=]{3,}$",  # 分隔线
        r"版权声明",  # 版权信息
        r"版权所有",  # 版权信息
        r"联系电话：\d+",  # 联系电话
        r"地址：",  # 地址信息
        r"^目\s*录\s*$",  # 目录标题
        r"^表\s*次\s*$",  # 目录标题变体
        r"^目\s*次\s*$",  # 目录标题变体
        r"^第[一二三四五六七八九十]+章",  # 章节标题（目录条目）
        r"^第[一二三四五六七八九十]+节",  # 小节标题（目录条目）
        r"^\d+[.、]\s+\S+.{0,30}$",  # 目录条目格式（数字+点+内容）
        r"^\s*\.{3,}\s*\d+\s*$",  # 目录页码引导线
    ]

    def __init__(
        self,
        llm_runtime: LLMRuntimeService,
        artifact_repository: IntermediateArtifactRepository,
        prompt_service: PromptService,
    ):
        self.llm_runtime = llm_runtime
        self.artifact_repository = artifact_repository
        self.prompt_service = prompt_service

    async def clean_document(
        self,
        text: str,
        source_file_id: int,
        options: CleaningOptions | None = None,
    ) -> CleaningResult:
        """清洗文档，返回清洗后的内容和元数据。

        Args:
            text: 原始文本
            source_file_id: 源文件 ID
            options: 清洗选项

        Returns:
            清洗结果
        """
        try:
            options = options or CleaningOptions()
            logger.info(f"开始清洗文档 source_file_id={source_file_id}")

            # 第一层：规则过滤
            rule_cleaned = self.rule_based_clean(text, options)
            rule_stats = self._calculate_stats(text, rule_cleaned)

            # 第二层：LLM 智能清洗
            llm_cleaned = await self.llm_clean(rule_cleaned, options)
            final_stats = self._calculate_stats(rule_cleaned, llm_cleaned)
            final_stats = self._merge_stats(rule_stats, final_stats)

            # 保存产物
            output_path = await self._save_artifact(
                source_file_id=source_file_id,
                cleaned_text=llm_cleaned,
                original_text=text,
                stats=final_stats,
            )

            # 构建产物数据
            artifact = CleanedDocumentArtifact(
                source_file_id=source_file_id,
                original_text=text,
                cleaned_text=llm_cleaned,
                cleaning_stats=final_stats,
                output_path=output_path,
            )

            # 写入文件
            self._write_artifact_file(output_path, artifact)

            logger.info(
                f"文档清洗完成 source_file_id={source_file_id}, "
                f"移除字符数={final_stats.removed_chars}"
            )

            return CleaningResult(
                success=True,
                cleaned_text=llm_cleaned,
                stats=final_stats,
                error=None,
            )

        except Exception as e:
            logger.error(f"文档清洗失败 source_file_id={source_file_id}: {e}")
            return CleaningResult(
                success=False,
                cleaned_text=None,
                stats=None,
                error=str(e),
            )

    def rule_based_clean(self, text: str, options: CleaningOptions) -> str:
        """规则过滤。

        Args:
            text: 原始文本
            options: 清洗选项

        Returns:
            规则过滤后的文本
        """
        result = text

        # 移除广告
        if options.remove_ads:
            for pattern in self.AD_PATTERNS:
                result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # 移除噪声
        if options.remove_noise:
            for pattern in self.NOISE_PATTERNS:
                result = re.sub(pattern, "", result, flags=re.MULTILINE)

        # 标准化空白字符
        if options.normalize_whitespace:
            # 将多个连续空白字符替换为单个空格
            result = re.sub(r"[ \t]+", " ", result)
            # 移除每行首尾空白
            result = "\n".join(line.strip() for line in result.split("\n"))
            # 将多个连续换行替换为单个换行
            result = re.sub(r"\n{3,}", "\n\n", result)

        # 移除重复内容
        if options.remove_duplicates:
            result = self._remove_duplicates(result)

        return result.strip()

    async def llm_clean(self, text: str, options: CleaningOptions) -> str:
        """LLM 智能清洗。

        使用 callsite `doc_cleaning:clean_text` 调用推理 LLM，保留文档结构，移除噪声。

        Args:
            text: 规则过滤后的文本
            options: 清洗选项

        Returns:
            LLM 清洗后的文本
        """
        try:
            # 构建提示词
            prompt_text = self._build_llm_clean_prompt(text, options)

            # 获取模型实例 (绕过 LLMRuntime 的固定 Prompt 逻辑)
            llm = self.llm_runtime.get_model_for_callsite(SCOPE_DOC_CLEANING)

            # 构建链: Prompt -> LLM -> OutputParser
            # 由于 prompt_text 已经是完整的提示词（包含指令和文本），直接作为输入
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt_template = ChatPromptTemplate.from_messages([("human", "{input}")])
            chain = prompt_template | llm | StrOutputParser()

            cleaned = await chain.ainvoke({"input": prompt_text})

            # 解析结果
            if isinstance(cleaned, str):
                return cleaned.strip()
            return str(cleaned).strip()

        except Exception as e:
            logger.warning(f"LLM 清洗失败，使用规则过滤结果: {e}")
            return text

    def _build_llm_clean_prompt(
        self, text: str, options: CleaningOptions
    ) -> str:
        """构建 LLM 清洗提示词。"""
        instructions = []

        if options.remove_ads:
            instructions.append("- 移除所有广告内容、推广信息")

        if options.remove_noise:
            instructions.append("- 移除页眉、页脚、页码等噪声")
            instructions.append("- 移除版权声明、联系方式等非正文内容")

        if options.remove_duplicates:
            instructions.append("- 移除重复的内容段落")

        if options.preserve_structure:
            instructions.append("- 保留文档的标题层级结构")
            instructions.append("- 保留段落、列表等格式")
            instructions.append("- 保持内容的逻辑连贯性")

        instructions_str = "\n".join(instructions)

        # 获取提示词模板
        prompt_entity = self.prompt_service.get_active_prompt(SCOPE_DOC_CLEANING)
        if prompt_entity and prompt_entity.content:
            template = prompt_entity.content
        else:
            template = DEFAULT_PROMPTS[SCOPE_DOC_CLEANING]["content"]

        # 渲染提示词
        # 简单替换 {instructions} 和 {text}，避免 format 报错
        prompt = template.replace("{instructions}", instructions_str).replace(
            "{text}", text
        )

        return prompt

    def _remove_duplicates(self, text: str) -> str:
        """移除重复内容。"""
        lines = text.split("\n")
        seen = set()
        unique_lines = []

        for line in lines:
            # 标准化行内容（去除首尾空白，转小写）
            normalized = line.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
            elif normalized and normalized in seen:
                # 跳过重复行
                pass

        return "\n".join(unique_lines)

    def _calculate_stats(
        self, original: str, cleaned: str
    ) -> CleaningStats:
        """计算清洗统计信息。"""
        original_chars = len(original)
        cleaned_chars = len(cleaned)

        original_paragraphs = len([p for p in original.split("\n") if p.strip()])
        cleaned_paragraphs = len([p for p in cleaned.split("\n") if p.strip()])

        return CleaningStats(
            original_chars=original_chars,
            cleaned_chars=cleaned_chars,
            removed_chars=max(0, original_chars - cleaned_chars),
            original_paragraphs=original_paragraphs,
            cleaned_paragraphs=cleaned_paragraphs,
            removed_paragraphs=max(0, original_paragraphs - cleaned_paragraphs),
            ads_removed=0,  # 简化估算
            noise_removed=0,
            duplicates_removed=0,
        )

    def _merge_stats(
        self, rule_stats: CleaningStats, llm_stats: CleaningStats
    ) -> CleaningStats:
        """合并两次清洗的统计信息。"""
        return CleaningStats(
            original_chars=rule_stats.original_chars,
            cleaned_chars=llm_stats.cleaned_chars,
            removed_chars=rule_stats.removed_chars + llm_stats.removed_chars,
            original_paragraphs=rule_stats.original_paragraphs,
            cleaned_paragraphs=llm_stats.cleaned_paragraphs,
            removed_paragraphs=rule_stats.removed_paragraphs
            + llm_stats.removed_paragraphs,
            ads_removed=rule_stats.ads_removed + llm_stats.ads_removed,
            noise_removed=rule_stats.noise_removed + llm_stats.noise_removed,
            duplicates_removed=rule_stats.duplicates_removed
            + llm_stats.duplicates_removed,
        )

    async def _save_artifact(
        self,
        source_file_id: int,
        cleaned_text: str,
        original_text: str,
        stats: CleaningStats,
    ) -> str:
        """保存中间产物到数据库。

        Returns:
            相对存储路径
        """
        import uuid
        from datetime import datetime

        batch_id = datetime.now().strftime("%Y%m%d")
        artifact_id = str(uuid.uuid4())
        relative_path = f"intermediates/{source_file_id}/cleaned_doc/{artifact_id}.json"

        # 保存到数据库
        artifact = await self.artifact_repository.create(
            id=artifact_id,
            workspace_id="default",  # TODO: 从上下文获取
            source_id=str(source_file_id),
            type=IntermediateType.CLEANED_DOC,
            storage_path=relative_path,
            deletable=True,
            extra_metadata=json.dumps(
                {
                    "batch_id": batch_id,
                    "original_chars": stats.original_chars,
                    "cleaned_chars": stats.cleaned_chars,
                },
                ensure_ascii=False,
            ),
        )

        return relative_path

    def _write_artifact_file(
        self, output_path: str, artifact: CleanedDocumentArtifact
    ) -> None:
        """写入产物文件。"""
        base_path = Path("data") / output_path
        base_path.parent.mkdir(parents=True, exist_ok=True)

        with open(base_path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2, ensure_ascii=False))


class ChartExtractionService:
    """图表提取与转换服务（T032）。

    从 MinerU 输出中提取图表，调用多模态模型将图表转为 JSON。
    """

    def __init__(
        self,
        llm_runtime: LLMRuntimeService,
        artifact_repository: IntermediateArtifactRepository,
        prompt_service: PromptService,
    ):
        self.llm_runtime = llm_runtime
        self.artifact_repository = artifact_repository
        self.prompt_service = prompt_service

    async def extract_charts(
        self,
        mineru_output: dict[str, Any],
        source_file_id: int,
    ) -> ChartExtractionResult:
        """提取图表并转换为 JSON。

        Args:
            mineru_output: MinerU 输出结果
            source_file_id: 源文件 ID

        Returns:
            图表提取结果
        """
        try:
            logger.info(f"开始提取图表 source_file_id={source_file_id}")

            # 从 MinerU 输出中提取图表信息
            charts = await self._extract_from_mineru(mineru_output, source_file_id)

            if not charts:
                logger.info(f"未发现图表 source_file_id={source_file_id}")
                return ChartExtractionResult(
                    success=True,
                    charts=[],
                    error=None,
                )

            # 调用多模态模型转换为 JSON
            json_charts = []
            for chart in charts:
                chart_json = await self.chart_to_json(
                    chart_image_path=chart["image_path"],
                    chart_type=chart["type"],
                )
                if chart_json:
                    json_charts.append(chart_json)

            # 保存产物
            output_path = await self._save_artifact(
                source_file_id=source_file_id,
                charts=json_charts,
            )

            # 写入文件
            artifact = ChartJSONArtifact(
                source_file_id=source_file_id,
                charts=json_charts,
                output_path=output_path,
            )
            self._write_artifact_file(output_path, artifact)

            logger.info(
                f"图表提取完成 source_file_id={source_file_id}, "
                f"提取图表数={len(json_charts)}"
            )

            return ChartExtractionResult(
                success=True,
                charts=json_charts,
                error=None,
            )

        except Exception as e:
            logger.error(f"图表提取失败 source_file_id={source_file_id}: {e}")
            return ChartExtractionResult(
                success=False,
                charts=None,
                error=str(e),
            )

    async def _extract_from_mineru(
        self, mineru_output: dict[str, Any], source_file_id: int
    ) -> list[dict[str, Any]]:
        """从 MinerU 输出中提取图表信息。"""
        charts = []

        # MinerU 输出结构：
        # - images: 页面图片列表
        # - markdown: 文本内容
        # - pdf_structure: PDF 结构信息

        images = mineru_output.get("images", [])
        pdf_structure = mineru_output.get("pdf_structure", {})

        # 从 images 中提取图表
        for idx, image_info in enumerate(images):
            image_path = image_info.get("path", "")
            image_type = image_info.get("type", "")

            # 判断是否为图表类型
            if self._is_chart_image(image_info):
                charts.append(
                    {
                        "chart_id": f"chart_{source_file_id}_{idx}",
                        "image_path": image_path,
                        "type": self._detect_chart_type(image_info),
                        "page_number": image_info.get("page", idx + 1),
                    }
                )

        # 从 pdf_structure 中提取表格和图表
        for element in pdf_structure.get("elements", []):
            if element.get("type") in ["table", "figure", "chart"]:
                charts.append(
                    {
                        "chart_id": f"chart_{source_file_id}_{element.get('index', 0)}",
                        "image_path": element.get("image_path", ""),
                        "type": element.get("type", "other"),
                        "page_number": element.get("page", 1),
                    }
                )

        return charts

    def _is_chart_image(self, image_info: dict[str, Any]) -> bool:
        """判断图片是否为图表。"""
        image_type = image_info.get("type", "").lower()
        chart_indicators = ["chart", "figure", "graph", "table", "plot"]

        # 检查类型字段
        if image_type in chart_indicators:
            return True

        # 检查文件名
        image_path = image_info.get("path", "").lower()
        if any(indicator in image_path for indicator in chart_indicators):
            return True

        # 检查宽度高度比例（图表通常有特定的宽高比）
        width = image_info.get("width", 0)
        height = image_info.get("height", 0)
        if width > 0 and height > 0:
            ratio = width / height
            # 图表通常不是正方形或极窄长
            if 0.3 < ratio < 5:
                return True

        return False

    def _detect_chart_type(self, image_info: dict[str, Any]) -> ChartType:
        """检测图表类型。"""
        image_type = image_info.get("type", "").lower()
        image_path = image_info.get("path", "").lower()

        if "table" in image_type or "table" in image_path:
            return ChartType.TABLE

        if "bar" in image_type or "bar" in image_path:
            return ChartType.BAR

        if "line" in image_type or "line" in image_path:
            return ChartType.LINE

        if "pie" in image_type or "pie" in image_path:
            return ChartType.PIE

        if "scatter" in image_type or "scatter" in image_path:
            return ChartType.SCATTER

        if "area" in image_type or "area" in image_path:
            return ChartType.AREA

        if "radar" in image_type or "radar" in image_path:
            return ChartType.RADAR

        if "heatmap" in image_type or "heatmap" in image_path:
            return ChartType.HEATMAP

        return ChartType.OTHER

    async def chart_to_json(
        self,
        chart_image_path: str,
        chart_type: str,
    ) -> ChartData | None:
        """图表转 JSON。

        调用多模态模型将图表图片转换为结构化 JSON。

        Args:
            chart_image_path: 图表图片路径
            chart_type: 图表类型

        Returns:
            图表 JSON 数据
        """
        try:
            # 检查文件是否存在
            if not Path(chart_image_path).exists():
                logger.warning(f"图表文件不存在: {chart_image_path}")
                return None

            # 构建提示词
            prompt_text = self._build_chart_prompt(chart_type)

            # 获取模型实例
            llm = self.llm_runtime.get_model_for_callsite(SCOPE_CHART_EXTRACTION)

            # 构建链
            # 注意：多模态模型通常需要特定的 Prompt 结构来传递图片
            # 这里假设 LLMRuntime 返回的 ChatProtocol 支持 standard LangChain message format
            from langchain_core.messages import HumanMessage

            # 构造多模态消息
            # LangChain 标准格式：content=[{"type": "text", ...}, {"type": "image_url", ...}]
            # 但具体的 LLM 实现可能有所不同，这里假设 LLMRuntimeService 会处理好适配
            # 实际上，如果 LLM 是 ChatOpenAI，它支持 base64 或 url
            # 为了简单起见，我们这里使用 LLMRuntimeService 提供的 build_runnable_for_capability
            # 但是它绑定了 Prompt... 且多模态的 Prompt 处理比较复杂
            
            # 回退方案：如果 capability 为 chart_ocr，我们期望 LLMRuntimeService 能正确处理
            # 但是我们的 prompt 是动态的 (chart_type)。
            
            # 我们手动构建多模态消息
            # 读取图片并编码为 base64
            import base64
            
            with open(chart_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}"
                        },
                    },
                ]
            )
            
            result = await llm.ainvoke([message])
            
            # 解析结果
            json_content = self._parse_chart_result(result.content if hasattr(result, "content") else result)

            return ChartData(
                chart_id=f"chart_{Path(chart_image_path).stem}",
                chart_type=ChartType(chart_type) if chart_type in [e.value for e in ChartType] else ChartType.OTHER,
                json_content=json_content,
                source_image_path=chart_image_path,
                confidence=0.9,  # TODO: 从模型响应中获取置信度
                page_number=None,
            )

        except Exception as e:
            logger.error(f"图表转 JSON 失败: {chart_image_path}, error: {e}")
            return None

    def _build_chart_prompt(self, chart_type: str) -> str:
        """构建图表识别提示词。"""
        type_descriptions = {
            "bar": "柱状图",
            "line": "折线图",
            "pie": "饼图",
            "table": "表格",
            "scatter": "散点图",
            "area": "面积图",
            "radar": "雷达图",
            "heatmap": "热力图",
            "other": "图表",
        }

        description = type_descriptions.get(chart_type, "图表")

        # 获取提示词模板
        prompt_entity = self.prompt_service.get_active_prompt(SCOPE_CHART_EXTRACTION)
        if prompt_entity and prompt_entity.content:
            template = prompt_entity.content
        else:
            template = DEFAULT_PROMPTS[SCOPE_CHART_EXTRACTION]["content"]

        # 渲染提示词
        prompt = template.replace("{description}", description).replace(
            "{chart_type}", chart_type
        )

        return prompt

    def _parse_chart_result(self, result: Any) -> dict[str, Any]:
        """解析图表识别结果。"""
        if isinstance(result, str):
            try:
                # 尝试解析 JSON
                import json

                # 清理可能的前缀/后缀
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:]
                if result.startswith("```"):
                    result = result[3:]
                if result.endswith("```"):
                    result = result[:-3]
                result = result.strip()

                return json.loads(result)
            except json.JSONDecodeError:
                # 如果解析失败，返回原始文本
                return {"raw_text": result, "error": "JSON 解析失败"}

        if isinstance(result, dict):
            return result

        return {"raw_result": str(result)}

    async def _save_artifact(
        self,
        source_file_id: int,
        charts: list[ChartData],
    ) -> str:
        """保存中间产物到数据库。

        Returns:
            相对存储路径
        """
        import uuid
        from datetime import datetime

        batch_id = datetime.now().strftime("%Y%m%d")
        artifact_id = str(uuid.uuid4())
        relative_path = f"intermediates/{source_file_id}/chart_json/{artifact_id}.json"

        # 保存到数据库
        await self.artifact_repository.create(
            id=artifact_id,
            workspace_id="default",  # TODO: 从上下文获取
            source_id=str(source_file_id),
            type=IntermediateType.CHART_JSON,
            storage_path=relative_path,
            deletable=True,
            extra_metadata=json.dumps(
                {
                    "batch_id": batch_id,
                    "chart_count": len(charts),
                },
                ensure_ascii=False,
            ),
        )

        return relative_path

    def _write_artifact_file(
        self, output_path: str, artifact: ChartJSONArtifact
    ) -> None:
        """写入产物文件。"""
        base_path = Path("data") / output_path
        base_path.parent.mkdir(parents=True, exist_ok=True)

        with open(base_path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2, ensure_ascii=False))

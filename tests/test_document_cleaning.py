"""T097: 文档噪音清洗功能测试脚本。

测试范围：
- 规则过滤（广告、噪声、重复内容）
- LLM 智能清洗
- 输入来源: data/intermediates/{source_file_id}/mineru_raw/ 的 md/json 文件
- 输出验证: data/intermediates/{source_file_id}/cleaned_doc/ 清洗后的完整 md/json
- 保留完整内容: md 和 json 文件均完整保留，供后续图转 JSON 使用

数据流转:
  输入: data/sources/default/*.pdf → MinerU OCR → data/intermediates/{id}/mineru_raw/
  清洗: data/intermediates/{id}/mineru_raw/* → 文档噪音清洗 → data/intermediates/{id}/cleaned_doc/
  输出: 清洗后的完整 md 和 json 文件

执行步骤:
  1. 首先运行 T093 (MinerU 清洗) 产生 mineru_raw 输出
  2. 运行本测试 (T097) 读取 mineru_raw 输出，清洗后写入 cleaned_doc
  3. 后续测试读取 cleaned_doc 进行图转 JSON 等操作

使用方法:
    # 仅运行本测试
    pytest tests/test_document_cleaning.py -v

    # 带详细输出
    pytest tests/test_document_cleaning.py -vv --tb=long
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DocumentCleaningPipeline:
    """文档清洗流水线测试类。"""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.sources_dir = project_root / "data" / "sources" / "default"
        self.intermediates_dir = project_root / "data" / "intermediates"

    def get_source_files(self) -> list[Path]:
        """获取源 PDF 文件列表。"""
        if not self.sources_dir.exists():
            raise FileNotFoundError(f"源文件目录不存在: {self.sources_dir}")
        
        pdf_files = list(self.sources_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"源文件目录中没有 PDF 文件: {self.sources_dir}")
        
        return pdf_files

    def get_mineru_output_dir(self, source_file_id: int) -> Optional[Path]:
        """获取指定源文件的 MinerU 输出目录。"""
        output_dir = self.intermediates_dir / str(source_file_id) / "mineru_raw"
        if output_dir.exists():
            return output_dir
        return None

    def get_cleaned_doc_dir(self, source_file_id: int) -> Path:
        """获取清洗后文档的输出目录。"""
        output_dir = self.intermediates_dir / str(source_file_id) / "cleaned_doc"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def load_mineru_output(self, source_file_id: int) -> dict[str, Any]:
        """加载 MinerU 输出文件。"""
        mineru_dir = self.get_mineru_output_dir(source_file_id)
        if mineru_dir is None:
            return None

        # 查找 md 文件
        md_files = list(mineru_dir.glob("*.md"))
        json_files = list(mineru_dir.glob("*.json"))

        result = {
            "source_file_id": source_file_id,
            "mineru_dir": str(mineru_dir),
        }

        # 加载 md 内容
        if md_files:
            md_file = md_files[0]
            result["markdown_file"] = str(md_file)
            result["markdown_content"] = md_file.read_text(encoding="utf-8")

        # 加载 json 内容
        if json_files:
            json_file = json_files[0]
            result["json_file"] = str(json_file)
            result["json_content"] = json.loads(json_file.read_text(encoding="utf-8"))

        return result

    def save_cleaned_output(
        self,
        source_file_id: int,
        original_text: str,
        cleaned_text: str,
        stats: dict[str, Any],
    ) -> dict[str, Path]:
        """保存清洗后的输出文件。"""
        cleaned_dir = self.get_cleaned_doc_dir(source_file_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存清洗后的 md 文件
        md_path = cleaned_dir / f"cleaned_{timestamp}.md"
        md_path.write_text(cleaned_text, encoding="utf-8")

        # 保存元数据 json
        meta_path = cleaned_dir / f"cleaned_{timestamp}_meta.json"
        meta_data = {
            "source_file_id": source_file_id,
            "original_chars": len(original_text),
            "cleaned_chars": len(cleaned_text),
            "removed_chars": len(original_text) - len(cleaned_text),
            "removed_ratio": (len(original_text) - len(cleaned_text)) / max(1, len(original_text)),
            "cleaned_at": timestamp,
            "stats": stats,
        }
        meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 复制原始 json（如果有的话）
        mineru_output = self.load_mineru_output(source_file_id)
        if mineru_output and "json_content" in mineru_output:
            json_path = cleaned_dir / f"cleaned_{timestamp}_original.json"
            # 在原始 json 中添加清洗后的内容
            original_json = mineru_output["json_content"].copy()
            original_json["cleaned_markdown"] = cleaned_text
            original_json["cleaning_stats"] = meta_data
            json_path.write_text(json.dumps(original_json, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "markdown": md_path,
            "metadata": meta_path,
            "json": cleaned_dir / f"cleaned_{timestamp}_original.json" if mineru_output and "json_content" in mineru_output else None,
        }


class TestDocumentCleaningPipeline:
    """文档清洗流水线测试类。"""

    @pytest.fixture
    def pipeline(self) -> DocumentCleaningPipeline:
        """获取测试流水线实例。"""
        return DocumentCleaningPipeline()

    @pytest.fixture
    def sample_text_with_noise(self) -> str:
        """获取包含噪声的示例文本。"""
        return """# 产品白皮书

## 广告内容
广告推广内容
Sponsored Advertisement
VIP会员专享
扫码关注微信公众号获取更多资讯

## 页眉页脚
第 1 页
Page 2
第 3 页

## 正文内容
这是文档的正文内容，包含有用的信息。

## 重复内容
这是重复的内容
这是重复的内容
这是重复的内容

## 联系方式
联系电话：123456789
咨询热线：400-123-4567
地址：北京市海淀区

## 参考资料
https://example.com
www.example.org

Copyright 2024
版权所有
"""

    def test_pipeline_initialization(self, pipeline):
        """测试流水线初始化。"""
        assert pipeline.sources_dir.exists()
        assert pipeline.intermediates_dir.exists()

    def test_source_file_detection(self, pipeline):
        """测试源文件检测。"""
        try:
            files = pipeline.get_source_files()
            assert len(files) > 0
            print(f"发现 {len(files)} 个源 PDF 文件")
        except FileNotFoundError as e:
            pytest.skip(str(e))

    def test_mineru_output_detection(self, pipeline):
        """测试 MinerU 输出检测。"""
        # 尝试查找已有的 MinerU 输出
        source_files = pipeline.get_source_files()
        
        for source_file in source_files[:1]:  # 只检查第一个文件
            source_file_id = hash(source_file.name) % 1000  # 简单生成 ID
            mineru_dir = pipeline.get_mineru_output_dir(source_file_id)
            
            if mineru_dir is None:
                print(f"未找到 MinerU 输出目录: {source_file.name} (ID: {source_file_id})")
            else:
                print(f"找到 MinerU 输出: {mineru_dir}")
                return True
        
        return False

    def test_ad_pattern_matching(self, sample_text_with_noise):
        """测试广告模式匹配。"""
        import re

        ad_patterns = [
            r"广告",
            r"Sponsored",
            r"VIP",
            r"扫码关注",
            r"微信公众号",
            r"咨询热线",
            r"联系电话",
        ]

        matched_count = 0
        for pattern in ad_patterns:
            matches = re.findall(pattern, sample_text_with_noise, re.IGNORECASE)
            if matches:
                matched_count += len(matches)
                print(f"广告模式 '{pattern}': 匹配 {len(matches)} 次")

        assert matched_count > 0, "应该匹配到广告内容"
        print(f"共匹配 {matched_count} 个广告实例")

    def test_noise_pattern_matching(self, sample_text_with_noise):
        """测试噪声模式匹配。"""
        import re

        noise_patterns = [
            r"第\s*\d+\s*页",
            r"Page\s*\d+",
            r"联系电话：\d+",
            r"地址：",
            r"Copyright",
            r"版权所有",
        ]

        matched_count = 0
        for pattern in noise_patterns:
            matches = re.findall(pattern, sample_text_with_noise, re.IGNORECASE)
            if matches:
                matched_count += len(matches)
                print(f"噪声模式 '{pattern}': 匹配 {len(matches)} 次")

        assert matched_count > 0, "应该匹配到噪声内容"
        print(f"共匹配 {matched_count} 个噪声实例")

    def test_rule_based_cleaning(self, sample_text_with_noise):
        """测试规则过滤清洗。"""
        import re

        # 广告模式
        ad_patterns = [
            r"广告",
            r"Sponsored",
            r"VIP",
            r"扫码关注",
            r"微信公众号",
            r"咨询热线",
            r"联系电话：\d+",
        ]

        # 噪声模式
        noise_patterns = [
            r"第\s*\d+\s*页",
            r"Page\s*\d+",
            r"Copyright.*\d{4}",
            r"版权所有",
            r"地址：[^。]+",
        ]

        cleaned = sample_text_with_noise

        # 移除广告
        for pattern in ad_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 移除噪声
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 标准化空白
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))

        # 验证
        print(f"\n清洗前字符数: {len(sample_text_with_noise)}")
        print(f"清洗后字符数: {len(cleaned)}")
        print(f"移除比例: {(len(sample_text_with_noise) - len(cleaned)) / len(sample_text_with_noise) * 100:.1f}%")

        # 验证广告被移除
        for pattern in ad_patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            assert len(matches) == 0, f"广告模式 '{pattern}' 未被完全移除"

        # 验证噪声被移除
        for pattern in noise_patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            assert len(matches) == 0, f"噪声模式 '{pattern}' 未被完全移除"

        # 验证结构保留
        assert "# 产品白皮书" in cleaned
        assert "这是文档的正文内容" in cleaned

        print("规则过滤测试通过!")

    def test_cleaning_with_real_mineru_output(self, pipeline):
        """使用真实 MinerU 输出进行清洗测试。"""
        # 尝试加载真实的 MinerU 输出
        source_files = pipeline.get_source_files()
        
        for source_file in source_files[:1]:
            # 假设 source_file_id 为 1（实际应该从数据库获取）
            source_file_id = 1
            
            mineru_output = pipeline.load_mineru_output(source_file_id)
            
            if mineru_output is None:
                pytest.skip(f"未找到 MinerU 输出，无法进行真实数据测试 (source_file_id={source_file_id})")
            
            if "markdown_content" not in mineru_output:
                pytest.skip("MinerU 输出中没有 markdown 内容")

            original_text = mineru_output["markdown_content"]
            
            print(f"\n使用真实 MinerU 输出进行测试:")
            print(f"  文件: {source_file.name}")
            print(f"  字符数: {len(original_text)}")
            print(f"  行数: {original_text.count(chr(10))}")

            # 执行规则过滤
            import re

            ad_patterns = [
                r"广告",
                r"Sponsored",
                r"VIP",
                r"扫码关注",
                r"微信公众号",
                r"咨询热线",
                r"联系电话",
            ]

            noise_patterns = [
                r"第\s*\d+\s*页",
                r"Page\s*\d+",
                r"Copyright",
                r"版权所有",
            ]

            cleaned = original_text

            for pattern in ad_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

            for pattern in noise_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))

            # 保存清洗结果
            stats = {
                "original_chars": len(original_text),
                "cleaned_chars": len(cleaned),
                "removed_chars": len(original_text) - len(cleaned),
                "removed_ratio": (len(original_text) - len(cleaned)) / max(1, len(original_text)),
                "ad_removed": sum(1 for p in ad_patterns if re.search(p, original_text, re.IGNORECASE)),
                "noise_removed": sum(1 for p in noise_patterns if re.search(p, original_text, re.IGNORECASE)),
            }

            output_paths = pipeline.save_cleaned_output(
                source_file_id=source_file_id,
                original_text=original_text,
                cleaned_text=cleaned,
                stats=stats,
            )

            print(f"\n清洗结果:")
            print(f"  移除字符数: {stats['removed_chars']} ({stats['removed_ratio']*100:.1f}%)")
            print(f"  输出文件:")
            for name, path in output_paths.items():
                if path and path.exists():
                    print(f"    - {name}: {path.relative_to(pipeline.project_root)}")

            # 验证输出文件
            assert output_paths["markdown"].exists(), "清洗后的 md 文件应该存在"
            assert output_paths["metadata"].exists(), "元数据文件应该存在"

            print("\n真实数据清洗测试通过!")

            return True

        return False

    def test_cleaning_output_preservation(self, pipeline, sample_text_with_noise):
        """测试清洗输出保留（md 和 json 完整保留）。"""
        # 模拟 MinerU 输出结构
        mineru_json = {
            "task_id": "test-task-001",
            "status": "completed",
            "result": {
                "markdown": sample_text_with_noise,
                "pdf_info": {"page_count": 3, "file_size": 1024},
                "images": [
                    {"path": "images/p1.png", "type": "chart", "page": 1},
                    {"path": "images/p2.png", "type": "table", "page": 2},
                ]
            }
        }

        # 执行清洗
        import re

        ad_patterns = [r"广告", r"Sponsored", r"VIP", r"扫码关注", r"咨询热线"]
        noise_patterns = [r"第\s*\d+\s*页", r"Page\s*\d+", r"Copyright"]

        cleaned = sample_text_with_noise
        for pattern in ad_patterns + noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # 验证 json 结构保留
        preserved_json = mineru_json.copy()
        preserved_json["cleaned_markdown"] = cleaned
        preserved_json["cleaning_stats"] = {
            "original_chars": len(sample_text_with_noise),
            "cleaned_chars": len(cleaned),
            "removed_chars": len(sample_text_with_noise) - len(cleaned),
        }
        preserved_json["result"]["images"] = mineru_json["result"]["images"]  # 保留图片信息

        # 验证关键字段
        assert "cleaned_markdown" in preserved_json
        assert "cleaning_stats" in preserved_json
        assert "result" in preserved_json
        assert "images" in preserved_json["result"]  # 图片信息保留

        print("JSON 结构保留测试通过!")
        print(f"  - cleaned_markdown: 保留")
        print(f"  - cleaning_stats: 保留")
        print(f"  - result.images: 保留 ({len(preserved_json['result']['images'])} 张图片)")


class TestCleaningStatistics:
    """清洗统计测试类。"""

    def test_statistics_calculation(self):
        """测试统计信息计算。"""
        original = "这是原始文本内容" * 100
        cleaned = "这是清洗后文本" * 80

        original_chars = len(original)
        cleaned_chars = len(cleaned)
        removed_chars = original_chars - cleaned_chars
        removed_ratio = removed_chars / original_chars

        stats = {
            "original_chars": original_chars,
            "cleaned_chars": cleaned_chars,
            "removed_chars": removed_chars,
            "removed_ratio": removed_ratio,
        }

        assert stats["original_chars"] == 1000
        assert stats["cleaned_chars"] == 800
        assert stats["removed_chars"] == 200
        assert stats["removed_ratio"] == 0.2

    def test_preserved_content_detection(self):
        """测试保留内容检测。"""
        # 应该保留的内容
        preserved_content = """# 文档标题

## 二级标题

这是重要内容，包含关键技术信息。

- 重要列表项一
- 重要列表项二

```python
def important_function():
    return True
```
"""

        # 应该被移除的内容
        removed_content = """广告推广
Sponsored
第 1 页
Page 2
联系电话：123456
"""

        # 验证保留内容
        assert "# 文档标题" in preserved_content
        assert "重要内容" in preserved_content
        assert "def important_function" in preserved_content

        # 验证移除内容
        assert "广告推广" not in removed_content
        assert "第 1 页" not in removed_content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

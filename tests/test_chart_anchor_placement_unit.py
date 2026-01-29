from __future__ import annotations

from src.application.services.content_generation.chart_anchor_postprocessor import place_chart_anchors


def test_place_chart_anchors_avoids_first_paragraph_when_no_hint():
    md = "\n\n".join(
        [
            "第一段很短。",
            "第二段更长一些，包含更多上下文信息，用于承载图表锚点。",
            "第三段也可以。",
        ]
    )
    out = place_chart_anchors(md, chart_ids=["A"], hint_by_chart_id={"A": ""}, min_paragraph_index=1)
    assert out.count("[Chart: A]") == 1
    assert "第一段很短。\n\n[Chart: A]" not in out


def test_place_chart_anchors_prefers_matching_paragraph():
    md = "\n\n".join(
        [
            "概述段落。",
            "本段讨论装机规模与累计装机规模趋势，并给出数据口径。",
            "结论段落。",
        ]
    )
    out = place_chart_anchors(md, chart_ids=["C1"], hint_by_chart_id={"C1": "累计装机规模"}, min_paragraph_index=1)
    assert out.count("[Chart: C1]") == 1
    assert "累计装机规模趋势，并给出数据口径。\n\n[Chart: C1]" in out

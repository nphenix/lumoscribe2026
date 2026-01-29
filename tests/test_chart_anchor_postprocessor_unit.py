from __future__ import annotations

from src.application.services.content_generation.chart_anchor_postprocessor import reduce_tail_pileup


def test_reduce_tail_pileup_moves_anchors_from_end():
    md = "\n\n".join(
        [
            "本段讨论收入增长与销售结构。",
            "本段讨论利润率与成本变化。",
            "\n".join(["[Chart: A]", "[Chart: B]"]),
        ]
    )
    out = reduce_tail_pileup(
        md,
        required_chart_ids=["A", "B"],
        hint_by_chart_id={"A": "收入 销售", "B": "利润 成本"},
        tail_window_lines=10,
    )
    assert out.count("[Chart: A]") == 1
    assert out.count("[Chart: B]") == 1
    assert "收入增长" in out
    assert "利润率" in out
    paras = [p.strip() for p in out.replace("\r\n", "\n").split("\n\n") if p.strip()]
    for p in paras:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if not lines:
            continue
        assert any(not ln.startswith("[Chart:") for ln in lines) is True

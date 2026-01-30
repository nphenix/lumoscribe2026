import json
from pathlib import Path

import pytest

from src.application.services.antv_rendering.spec_mapper import chart_json_to_antv_payload


@pytest.mark.parametrize(
    "name, expected_engine",
    [
        ("sankey", "g2"),
        ("stacked_area", "g2"),
        ("table", "s2"),
    ],
)
def test_chart_json_to_antv_payload_smoke(name: str, expected_engine: str):
    p = Path("tests") / "fixtures" / "chart_json" / f"{name}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    payload = chart_json_to_antv_payload(chart_id=name, chart_json=obj, theme="whitepaper-default")
    assert payload.engine == expected_engine
    assert payload.width > 0
    assert payload.height > 0
    assert isinstance(payload.spec, dict)
    assert payload.theme == "whitepaper-default"


def test_chart_json_to_antv_payload_sankey_has_links():
    p = Path("tests") / "fixtures" / "chart_json" / "sankey.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    payload = chart_json_to_antv_payload(chart_id="sankey", chart_json=obj, theme="whitepaper-default")
    assert payload.engine == "g2"
    assert payload.spec.get("type") == "sankey"
    links = ((payload.spec.get("data") or {}).get("value") or {}).get("links")
    assert isinstance(links, list)
    assert len(links) >= 5


def test_chart_json_to_antv_payload_table_has_columns_rows():
    p = Path("tests") / "fixtures" / "chart_json" / "table.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    payload = chart_json_to_antv_payload(chart_id="table", chart_json=obj, theme="whitepaper-default")
    assert payload.engine == "s2"
    assert isinstance(payload.spec.get("columns"), list)
    assert isinstance(payload.spec.get("rows"), list)


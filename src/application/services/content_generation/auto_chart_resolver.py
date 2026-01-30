"""自动图表反查与渲染配置抽取（T096）。

核心思想：
- 不依赖“模板占位符”
- 从 RAG 命中的 chunk 内容中提取图表线索：
  1) `![](images/<stem>...)` -> 反查 `chart_json/<stem>.json`
  2) 建库注入的 `[图表] <chart_name>` -> 在同 doc_dir 的 `chart_json/*.json` 中按名称匹配
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.application.schemas.chart_spec import ChartConfig
from src.application.schemas.ingest import SearchResult
from src.application.services.content_generation.chart_json_converter import ChartJsonConverter
from src.application.services.content_generation.text_utils import (
    CHART_SNIPPET_RE,
    CHART_ANCHOR_RE,
    norm_compact,
    read_text_best_effort,
)


class AutoChartResolver:
    """从检索命中反查 chart_json 并生成可渲染配置。"""

    def __init__(self, *, converter: ChartJsonConverter | None = None):
        self.converter = converter or ChartJsonConverter()

    def _resolve_doc_dir_rel(self, doc_rel_path: str) -> Path:
        rel = (doc_rel_path or "").strip().replace("\\", "/")
        rel = rel.split("#", 1)[0].split("?", 1)[0].strip()
        doc_dir_rel = Path(rel) if rel else Path(".")
        if doc_dir_rel.suffix:
            doc_dir_rel = doc_dir_rel.parent
        if str(doc_dir_rel) in {"", "."}:
            doc_dir_rel = Path(".")
        return doc_dir_rel

    def _resolve_chart_json_path(self, *, kb_root: Path, doc_rel_path: str, image_stem: str) -> Path:
        doc_dir_rel = self._resolve_doc_dir_rel(doc_rel_path)
        return (kb_root / doc_dir_rel / "chart_json" / f"{image_stem}.json").resolve()

    def _build_chart_name_index(self, chart_dir: Path) -> dict[str, Path]:
        """扫描 chart_dir，构建 {normalized_chart_name: json_path}。"""
        mapping: dict[str, Path] = {}
        if not chart_dir.exists():
            return mapping
        for jf in sorted(chart_dir.glob("*.json")):
            try:
                obj = json.loads(read_text_best_effort(jf))
            except Exception:
                continue
            if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                continue
            name = norm_compact(str(obj.get("_chart_name") or obj.get("chart_name") or ""))
            if name:
                mapping.setdefault(name, jf)
        return mapping

    def _build_chart_id_index(self, chart_dir: Path) -> dict[str, Path]:
        """扫描 chart_dir，构建 {chart_id: json_path}。"""
        mapping: dict[str, Path] = {}
        if not chart_dir.exists():
            return mapping
        for jf in sorted(chart_dir.glob("*.json")):
            try:
                obj = json.loads(read_text_best_effort(jf))
            except Exception:
                continue
            if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                continue
            mapping[jf.stem] = jf
        return mapping

    def _build_chart_candidate_index(self, chart_dir: Path) -> list[dict[str, Any]]:
        """扫描 chart_dir，构建弱匹配用的候选列表（名称/描述/类型）。"""
        out: list[dict[str, Any]] = []
        if not chart_dir.exists():
            return out
        for jf in sorted(chart_dir.glob("*.json")):
            try:
                obj = json.loads(read_text_best_effort(jf))
            except Exception:
                continue
            if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                continue
            name = str(obj.get("_chart_name") or obj.get("chart_name") or "").strip()
            desc = str(obj.get("description") or "").strip()
            ctype = str(obj.get("chart_type") or "").strip()
            text = norm_compact(f"{name} {desc} {ctype}".strip())
            if not text or len(text) < 4:
                continue
            out.append(
                {
                    "path": jf,
                    "chart_name": name,
                    "chart_type": ctype,
                    "description": desc,
                    "norm_text": text,
                }
            )
            # 兜底防止某些文档 chart_json 过多导致扫描过慢
            if len(out) >= 200:
                break
        return out

    def _match_chart_json_by_name(self, *, mapping: dict[str, Path], chart_name: str) -> Path | None:
        key = norm_compact(chart_name)
        if not key:
            return None
        if key in mapping:
            return mapping[key]
        # 子串弱匹配（用于处理轻微命名差异）
        for k, p in mapping.items():
            if key in k or k in key:
                return p
        return None

    def _ngram_sim(self, a: str, b: str, *, n: int = 2) -> float:
        """基于字符 n-gram 的相似度（对中文/英文都相对稳健）。"""
        aa = norm_compact(a)
        bb = norm_compact(b)
        if not aa or not bb:
            return 0.0

        def grams(s: str) -> set[str]:
            if len(s) <= n:
                return set(s)
            return {s[i : i + n] for i in range(0, len(s) - n + 1)}

        ga = grams(aa)
        gb = grams(bb)
        if not ga or not gb:
            return 0.0
        inter = ga.intersection(gb)
        union = ga.union(gb)
        return float(len(inter)) / float(len(union)) if union else 0.0

    def extract_auto_chart_configs(
        self,
        *,
        hits: list[SearchResult],
        kb_input_root: Path | None,
        md_image_re: Any,
        max_auto: int,
        seen_chart_json: set[str] | None = None,
        sources_set: set[str] | None = None,
        hint_text: str | None = None,
        allow_weak_match: bool = False,
        audit: list[dict[str, Any]] | None = None,
    ) -> list[ChartConfig]:
        """从命中 chunk 中抽取可渲染图表配置。"""
        if kb_input_root is None:
            return []
        seen_chart_json = seen_chart_json or set()
        sources_set = sources_set or set()

        chart_configs: list[ChartConfig] = []
        chart_name_index_cache: dict[str, dict[str, Path]] = {}
        chart_id_index_cache: dict[str, dict[str, Path]] = {}
        chart_candidate_index_cache: dict[str, list[dict[str, Any]]] = {}
        weak_matched_docs: set[str] = set()
        hint_norm = norm_compact(hint_text or "")

        for r in hits:
            if len(chart_configs) >= max_auto:
                break
            meta = r.metadata if isinstance(r.metadata, dict) else {}
            doc_rel = meta.get("doc_rel_path")
            if not doc_rel:
                continue
            if meta.get("chunk_type") == "chart":
                continue
            doc_rel_s = str(doc_rel)
            content_s = r.content or ""
            has_image_hint = md_image_re.search(content_s) is not None
            has_snippet_hint = CHART_SNIPPET_RE.search(content_s) is not None

            # 1) images/<stem> -> chart_json/<stem>.json
            for mimg in md_image_re.finditer(content_s):
                if len(chart_configs) >= max_auto:
                    break
                raw_path = (mimg.group(1) or "").strip().strip('"').strip("'")
                raw_path = raw_path.replace("\\", "/").split("#", 1)[0].split("?", 1)[0]
                if "images/" not in raw_path:
                    continue
                stem = Path(raw_path).stem
                if not stem:
                    continue
                cj = self._resolve_chart_json_path(kb_root=kb_input_root, doc_rel_path=doc_rel_s, image_stem=stem)
                key = str(cj).replace("\\", "/")
                if key in seen_chart_json:
                    continue
                seen_chart_json.add(key)
                if not cj.exists():
                    continue
                try:
                    obj = json.loads(read_text_best_effort(cj))
                except Exception:
                    continue
                if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                    continue
                cfgs = self.converter.chart_json_to_configs(obj)
                if cfgs:
                    chart_id = stem
                    cn = str(obj.get("_chart_name") or "").strip()
                    if cn:
                        sources_set.add(f"图表来源: {cn}")
                    for j, cfg in enumerate(cfgs):
                        if len(chart_configs) >= max_auto:
                            break
                        # 即使是 md_image 反查，也尽量绑定 chart_id，便于后续按 `[Chart: <id>]` 锚点插入
                        if chart_id:
                            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                            cfg = cfg.model_copy(
                                update={
                                    "extra": {
                                        **(extra or {}),
                                        "chart_anchor_id": chart_id,
                                        "chart_anchor_index": j,
                                    }
                                }
                            )
                        chart_configs.append(cfg)
                        if audit is not None:
                            rel_path = None
                            try:
                                rel_path = str(cj.relative_to(kb_input_root)).replace("\\", "/")
                            except Exception:
                                rel_path = str(cj).replace("\\", "/")
                            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                            audit.append(
                                {
                                    "type": "auto_chart",
                                    "chart_name": cn,
                                    "chart_type": cfg.chart_type,
                                    "doc_rel_path": doc_rel_s,
                                    "doc_chart_json_path": rel_path,
                                    "reason": f"md_image:{stem}",
                                    "cycle_detected": bool(extra.get("cycle_detected")) if extra else False,
                                    "original_chart_type": extra.get("original_chart_type") if extra else None,
                                }
                            )

            # 2) [图表] <chart_name> -> 在 doc_dir/chart_json 下反查
            if len(chart_configs) >= max_auto:
                break
            doc_dir_rel = self._resolve_doc_dir_rel(doc_rel_s)
            chart_dir = (kb_input_root / doc_dir_rel / "chart_json").resolve()
            idx_key = str(chart_dir).replace("\\", "/")

            # 0) [Chart: <id>] -> 精确 ID 匹配 (High Priority)
            if idx_key not in chart_id_index_cache:
                chart_id_index_cache[idx_key] = self._build_chart_id_index(chart_dir)
            id_mapping = chart_id_index_cache.get(idx_key) or {}

            chart_ids = meta.get("chart_ids")
            if isinstance(chart_ids, str) and chart_ids.strip():
                try:
                    parsed = json.loads(chart_ids)
                    chart_ids = parsed
                except Exception:
                    chart_ids = [x.strip() for x in chart_ids.split(",") if x.strip()]
            if isinstance(chart_ids, list) and chart_ids:
                for raw in chart_ids:
                    if len(chart_configs) >= max_auto:
                        break
                    raw_s = str(raw or "").strip()
                    if not raw_s:
                        continue
                    candidates = []
                    if raw_s.startswith("chart_"):
                        candidates.append(raw_s[6:])
                    candidates.append(raw_s)
                    path = None
                    used_chart_id = None
                    for cid in candidates:
                        p = id_mapping.get(cid)
                        if p is not None:
                            path = p
                            used_chart_id = cid
                            break
                    if path is None:
                        continue
                    key = str(path).replace("\\", "/")
                    if key in seen_chart_json:
                        continue
                    seen_chart_json.add(key)
                    try:
                        obj = json.loads(read_text_best_effort(path))
                    except Exception:
                        continue
                    if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                        continue
                    cfgs = self.converter.chart_json_to_configs(obj)
                    if cfgs:
                        cn = str(obj.get("_chart_name") or "").strip()
                        if cn:
                            sources_set.add(f"图表来源: {cn}")
                        for j, cfg in enumerate(cfgs):
                            if len(chart_configs) >= max_auto:
                                break
                            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                            cfg2 = cfg.model_copy(
                                update={
                                    "extra": {
                                        **extra,
                                        "chart_anchor_id": used_chart_id,
                                        "chart_anchor_index": j,
                                    }
                                }
                            )
                            chart_configs.append(cfg2)
                            if audit is not None:
                                rel_path = None
                                try:
                                    rel_path = str(path.relative_to(kb_input_root)).replace("\\", "/")
                                except Exception:
                                    rel_path = str(path).replace("\\", "/")
                                audit.append(
                                    {
                                        "type": "auto_chart",
                                        "chart_name": cn,
                                        "chart_type": cfg2.chart_type,
                                        "doc_rel_path": doc_rel_s,
                                        "doc_chart_json_path": rel_path,
                                        "reason": f"chart_ids_meta:{raw_s}",
                                        "cycle_detected": False,
                                        "original_chart_type": None,
                                    }
                                )

            if id_mapping:
                for m in CHART_ANCHOR_RE.finditer(content_s):
                    if len(chart_configs) >= max_auto:
                        break
                    chart_id = (m.group(1) or "").strip()
                    path = id_mapping.get(chart_id)
                    if path is None:
                        continue
                    key = str(path).replace("\\", "/")
                    if key in seen_chart_json:
                        continue
                    seen_chart_json.add(key)
                    try:
                        obj = json.loads(read_text_best_effort(path))
                    except Exception:
                        continue
                    if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                        continue
                    cfgs = self.converter.chart_json_to_configs(obj)
                    if cfgs:
                        cn = str(obj.get("_chart_name") or "").strip()
                        if cn:
                            sources_set.add(f"图表来源: {cn}")
                        for j, cfg in enumerate(cfgs):
                            if len(chart_configs) >= max_auto:
                                break
                            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                            cfg2 = cfg.model_copy(
                                update={
                                    "extra": {
                                        **extra,
                                        "chart_anchor_id": chart_id,
                                        "chart_anchor_index": j,
                                    }
                                }
                            )
                            chart_configs.append(cfg2)
                            if audit is not None:
                                rel_path = None
                                try:
                                    rel_path = str(path.relative_to(kb_input_root)).replace("\\", "/")
                                except Exception:
                                    rel_path = str(path).replace("\\", "/")
                                audit.append(
                                    {
                                        "type": "auto_chart",
                                        "chart_name": cn,
                                        "chart_type": cfg2.chart_type,
                                        "doc_rel_path": doc_rel_s,
                                        "doc_chart_json_path": rel_path,
                                        "reason": f"chart_anchor:{chart_id}",
                                        "cycle_detected": False,
                                        "original_chart_type": None,
                                    }
                                )

            if len(chart_configs) >= max_auto:
                break

            if idx_key not in chart_name_index_cache:
                chart_name_index_cache[idx_key] = self._build_chart_name_index(chart_dir)
            mapping = chart_name_index_cache.get(idx_key) or {}

            if mapping:
                for m in CHART_SNIPPET_RE.finditer(content_s):
                    if len(chart_configs) >= max_auto:
                        break
                    cn_raw = (m.group(1) or "").strip()
                    path = self._match_chart_json_by_name(mapping=mapping, chart_name=cn_raw)
                    if path is None:
                        continue
                    key = str(path).replace("\\", "/")
                    if key in seen_chart_json:
                        continue
                    seen_chart_json.add(key)
                    try:
                        obj = json.loads(read_text_best_effort(path))
                    except Exception:
                        continue
                    if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                        continue
                    cfgs = self.converter.chart_json_to_configs(obj)
                    if cfgs:
                        chart_id = str(obj.get("_chart_id") or "").strip()
                        if cn_raw:
                            sources_set.add(f"图表来源: {cn_raw}")
                        for j, cfg in enumerate(cfgs):
                            if len(chart_configs) >= max_auto:
                                break
                            if chart_id:
                                extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                                cfg = cfg.model_copy(
                                    update={
                                        "extra": {
                                            **(extra or {}),
                                            "chart_anchor_id": chart_id,
                                            "chart_anchor_index": j,
                                        }
                                    }
                                )
                            chart_configs.append(cfg)
                            if audit is not None:
                                rel_path = None
                                try:
                                    rel_path = str(path.relative_to(kb_input_root)).replace("\\", "/")
                                except Exception:
                                    rel_path = str(path).replace("\\", "/")
                                extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                                audit.append(
                                    {
                                        "type": "auto_chart",
                                        "chart_name": str(obj.get("_chart_name") or cn_raw or "").strip(),
                                        "chart_type": cfg.chart_type,
                                        "doc_rel_path": doc_rel_s,
                                        "doc_chart_json_path": rel_path,
                                        "reason": f"chart_snippet:{cn_raw}",
                                        "cycle_detected": bool(extra.get("cycle_detected")) if extra else False,
                                        "original_chart_type": extra.get("original_chart_type") if extra else None,
                                    }
                                )

            if (
                allow_weak_match
                and len(chart_configs) < max_auto
                and not has_image_hint
                and not has_snippet_hint
                and hint_norm
                and len(hint_norm) >= 4
                and idx_key not in weak_matched_docs
            ):
                weak_matched_docs.add(idx_key)
                if idx_key not in chart_candidate_index_cache:
                    chart_candidate_index_cache[idx_key] = self._build_chart_candidate_index(chart_dir)
                cand = chart_candidate_index_cache.get(idx_key) or []
                if cand:
                    best = None
                    best_score = 0.0
                    for it in cand:
                        path = it.get("path")
                        if not isinstance(path, Path):
                            continue
                        key = str(path).replace("\\", "/")
                        if key in seen_chart_json:
                            continue
                        score = self._ngram_sim(hint_norm, str(it.get("norm_text") or ""))
                        if score > best_score:
                            best_score = score
                            best = it
                    # 阈值：宁可不补也不引入噪音
                    if best is not None and best_score >= 0.22:
                        path = best["path"]
                        key = str(path).replace("\\", "/")
                        if key not in seen_chart_json:
                            try:
                                obj = json.loads(read_text_best_effort(path))
                            except Exception:
                                obj = None
                            if isinstance(obj, dict) and obj.get("is_chart") is True:
                                cfgs = self.converter.chart_json_to_configs(obj)
                                if cfgs:
                                    seen_chart_json.add(key)
                                    cn = str(obj.get("_chart_name") or best.get("chart_name") or "").strip()
                                    if cn:
                                        sources_set.add(f"图表来源: {cn}")
                                    for cfg in cfgs:
                                        if len(chart_configs) >= max_auto:
                                            break
                                        chart_configs.append(cfg)
                                        if audit is not None:
                                            rel_path = None
                                            try:
                                                rel_path = str(path.relative_to(kb_input_root)).replace("\\", "/")
                                            except Exception:
                                                rel_path = str(path).replace("\\", "/")
                                            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                                            audit.append(
                                                {
                                                    "type": "auto_chart",
                                                    "chart_name": cn,
                                                    "chart_type": cfg.chart_type,
                                                    "doc_rel_path": doc_rel_s,
                                                    "doc_chart_json_path": rel_path,
                                                    "reason": f"weak_match:{hint_text} (score={best_score:.2f})",
                                                    "cycle_detected": bool(extra.get("cycle_detected")) if extra else False,
                                                    "original_chart_type": extra.get("original_chart_type") if extra else None,
                                                }
                                            )

        return chart_configs


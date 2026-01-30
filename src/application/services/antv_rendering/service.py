from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.application.services.antv_rendering.render_meta import (
    get_entry,
    load_render_meta,
    save_render_meta,
    sha256_hex,
    upsert_entry,
)
from src.application.services.antv_rendering.spec_mapper import chart_json_to_antv_payload
from src.shared.errors import AppError
from src.shared.logging import get_logger, log_extra

log = get_logger(__name__)


class AntvRenderingService:
    RENDER_VERSION = "antv-v1"
    DEFAULT_THEME = "whitepaper-default"

    def _find_repo_root(self) -> Path:
        start = Path(__file__).resolve()
        for p in [start, *start.parents]:
            if (p / "pyproject.toml").exists() and (p / "scripts").exists():
                return p
        return start.parents[5] if len(start.parents) >= 6 else start.parent

    def _node_script_path(self) -> Path:
        repo = self._find_repo_root()
        return (repo / "scripts" / "antv" / "render.mjs").resolve()

    def _run_node_render(self, *, payload: dict[str, Any]) -> tuple[bytes, dict[str, Any] | None]:
        script = self._node_script_path()
        if not script.exists():
            raise AppError(
                code="missing_dependency",
                message="antv render script not found",
                status_code=500,
                details={"script": str(script)},
            )
        proc = subprocess.run(
            ["node", str(script)],
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout, None
        err_payload = None
        try:
            err_payload = json.loads(proc.stderr.decode("utf-8", errors="replace"))
        except Exception:
            err_payload = None
        return b"", err_payload or {"stderr": proc.stderr.decode("utf-8", errors="replace")}

    def render_from_chart_json_dir(
        self,
        *,
        kb_input_root: Path,
        chart_ids: list[str] | None = None,
        theme: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        root = Path(kb_input_root).resolve()
        chart_dir = (root / "chart_json").resolve()
        if not chart_dir.exists() or not chart_dir.is_dir():
            raise AppError(
                code="invalid_input_root",
                message="kb_input_root/chart_json not found",
                status_code=400,
                details={"kb_input_root": str(root)},
            )

        theme_name = (theme or self.DEFAULT_THEME).strip() or self.DEFAULT_THEME
        meta = load_render_meta(chart_dir)

        if chart_ids:
            ids = [str(x or "").strip() for x in chart_ids if str(x or "").strip()]
        else:
            ids = [
                p.stem
                for p in sorted(chart_dir.glob("*.json"))
                if p.name.lower() not in {"render_meta.json"}
            ]

        results: dict[str, Any] = {"theme": theme_name, "render_version": self.RENDER_VERSION, "items": []}

        for cid in ids:
            json_path = (chart_dir / f"{cid}.json").resolve()
            if not json_path.exists():
                results["items"].append({"chart_id": cid, "status": "missing_json"})
                continue

            raw_bytes = json_path.read_bytes()
            json_hash = sha256_hex(raw_bytes)
            prev = get_entry(meta, cid)
            prev_hash = str(prev.get("json_hash") or "") if isinstance(prev, dict) else ""
            prev_theme = str(prev.get("theme") or "") if isinstance(prev, dict) else ""
            prev_ver = str(prev.get("render_version") or "") if isinstance(prev, dict) else ""
            prev_files = prev.get("files") if isinstance(prev, dict) else None

            svg_name = f"{cid}__{theme_name}.svg"
            png_name = f"{cid}__{theme_name}.png"
            jpg_name = f"{cid}__{theme_name}.jpg"
            svg_path = (chart_dir / svg_name).resolve()
            png_path = (chart_dir / png_name).resolve()
            jpg_path = (chart_dir / jpg_name).resolve()

            already_ok = (
                (not force)
                and prev_hash == json_hash
                and prev_theme == theme_name
                and prev_ver == self.RENDER_VERSION
                and svg_path.exists()
                and png_path.exists()
                and jpg_path.exists()
            )
            if already_ok:
                results["items"].append(
                    {"chart_id": cid, "status": "skipped", "files": {"svg": svg_name, "png": png_name, "jpg": jpg_name}}
                )
                continue

            try:
                obj = json.loads(raw_bytes.decode("utf-8"))
            except Exception as exc:
                results["items"].append({"chart_id": cid, "status": "invalid_json", "error": str(exc)})
                continue

            try:
                antv_payload = chart_json_to_antv_payload(chart_id=cid, chart_json=obj, theme=theme_name)
            except Exception as exc:
                results["items"].append({"chart_id": cid, "status": "unsupported", "error": str(exc)})
                continue

            ok = True
            err_details: dict[str, Any] | None = None
            for fmt, out_path in [("svg", svg_path), ("png", png_path), ("jpg", jpg_path)]:
                bytes_out, err = self._run_node_render(
                    payload={
                        "engine": antv_payload.engine,
                        "width": antv_payload.width,
                        "height": antv_payload.height,
                        "format": fmt,
                        "theme": antv_payload.theme,
                        "spec": antv_payload.spec,
                    }
                )
                if err is not None:
                    ok = False
                    err_details = {"format": fmt, "error": err}
                    break
                out_path.write_bytes(bytes_out)

            if not ok:
                log.warning(
                    "antv_render.failed",
                    extra=log_extra(chart_id=cid, json_path=str(json_path), details=err_details),
                )
                results["items"].append({"chart_id": cid, "status": "failed", "error": err_details})
                continue

            payload_dict = asdict(antv_payload)
            upsert_entry(
                meta,
                chart_id=cid,
                json_hash=json_hash,
                theme=theme_name,
                render_version=self.RENDER_VERSION,
                files={"svg": svg_name, "png": png_name, "jpg": jpg_name},
                engine=payload_dict.get("engine"),
            )
            results["items"].append(
                {
                    "chart_id": cid,
                    "status": "rendered",
                    "engine": antv_payload.engine,
                    "files": {"svg": svg_name, "png": png_name, "jpg": jpg_name},
                }
            )

        save_render_meta(chart_dir, meta)
        return results

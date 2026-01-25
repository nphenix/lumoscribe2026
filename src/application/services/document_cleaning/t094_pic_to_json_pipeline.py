"""T094: intermediates_cleaned -> pic_to_json 批处理管道（脚本入口调用）。

说明（对齐项目要求）：
- T094 是“批处理/验收任务”，不应把 pipeline 逻辑长期塞进核心 Service 类里。
- 正式能力（图转 JSON / prompt / callsite / 严格解析）仍由 ChartExtractionService 承担；
  本模块只负责文件级工作流：复制、并发图转 JSON、删除非图表、清理引用、落盘报告。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.application.schemas.document_cleaning import ChartData
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.prompt_service import PromptService
from src.shared.constants.prompts import SCOPE_CHART_EXTRACTION


class T094PicToJsonPipeline:
    _MD_IMAGE_PATH_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")

    def __init__(
        self,
        *,
        llm_runtime: LLMRuntimeService,
        prompt_service: PromptService,
        chart_to_json: Callable[..., Awaitable[ChartData | None]],
    ):
        self.llm_runtime = llm_runtime
        self.prompt_service = prompt_service
        self._chart_to_json = chart_to_json

    def _normalize_ref_path(self, raw: str) -> str:
        s = (raw or "").strip().strip('"').strip("'")
        # 去掉 query/hash，避免误判
        s = s.split("#", 1)[0].split("?", 1)[0].strip()
        s = s.replace("\\", "/")
        if s.startswith("./"):
            s = s[2:]
        return s

    def _resolve_ref_to_abs(self, base_dir: Path, ref: str) -> Path | None:
        """将 markdown/json 中的路径引用解析为绝对路径（用于与 removed 集合对齐）。"""
        ref_norm = self._normalize_ref_path(ref)
        if not ref_norm:
            return None
        if "\x00" in ref_norm:
            return None
        lower = ref_norm.lower()
        if lower.startswith(("http://", "https://", "data:")):
            return None
        try:
            p = Path(ref_norm)
            if p.is_absolute():
                return p.resolve()
            return (base_dir / p).resolve()
        except (OSError, ValueError):
            return None

    def _extract_is_chart_strict(self, json_obj: dict[str, Any], *, image: Path) -> bool:
        if not isinstance(json_obj, dict):
            raise RuntimeError(f"图转 JSON 输出不是对象: image={image}")
        if "is_chart" not in json_obj:
            raise RuntimeError(f"图转 JSON 输出缺少 is_chart: image={image}")
        v = json_obj.get("is_chart")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in {"true", "yes", "y", "1"}:
                return True
            if vv in {"false", "no", "n", "0"}:
                return False
        raise RuntimeError(f"is_chart 字段类型非法: image={image}, value={v!r}")

    def _json_contains_removed(
        self, obj: Any, *, base_dir: Path, removed_abs: set[Path]
    ) -> bool:
        """判断任意 JSON 结构是否包含对 removed 图片的引用。"""
        if isinstance(obj, str):
            abs_path = self._resolve_ref_to_abs(base_dir, obj)
            return abs_path is not None and abs_path in removed_abs
        if isinstance(obj, list):
            return any(
                self._json_contains_removed(x, base_dir=base_dir, removed_abs=removed_abs)
                for x in obj
            )
        if isinstance(obj, dict):
            return any(
                self._json_contains_removed(v, base_dir=base_dir, removed_abs=removed_abs)
                for v in obj.values()
            )
        return False

    def _prune_json_obj(
        self,
        obj: Any,
        *,
        base_dir: Path,
        removed_abs: set[Path],
    ) -> tuple[Any, bool]:
        """递归移除 JSON 中对 removed 图片的引用。

        设计目标：不依赖固定 schema，尽可能小范围删除字段/列表项，同时确保不残留引用。
        """
        if isinstance(obj, str):
            abs_path = self._resolve_ref_to_abs(base_dir, obj)
            if abs_path is not None and abs_path in removed_abs:
                return None, True
            return obj, False

        if isinstance(obj, list):
            changed = False
            new_list: list[Any] = []
            for item in obj:
                # 列表元素若“包含”引用，倾向删除整个元素（常见：images 列表/元素对象）
                if self._json_contains_removed(item, base_dir=base_dir, removed_abs=removed_abs):
                    changed = True
                    continue
                new_item, item_changed = self._prune_json_obj(
                    item, base_dir=base_dir, removed_abs=removed_abs
                )
                if item_changed:
                    changed = True
                new_list.append(new_item)
            return new_list, changed

        if isinstance(obj, dict):
            changed = False
            new_dict: dict[str, Any] = {}
            for k, v in obj.items():
                if isinstance(v, str):
                    abs_path = self._resolve_ref_to_abs(base_dir, v)
                    if abs_path is not None and abs_path in removed_abs:
                        changed = True
                        continue
                    new_dict[k] = v
                    continue

                if isinstance(v, (dict, list)):
                    new_v, v_changed = self._prune_json_obj(
                        v, base_dir=base_dir, removed_abs=removed_abs
                    )
                    if v_changed:
                        changed = True
                    new_dict[k] = new_v
                    continue

                new_dict[k] = v
            return new_dict, changed

        return obj, False

    def _remove_md_refs(
        self, md_text: str, *, md_dir: Path, removed_abs: set[Path]
    ) -> tuple[str, int]:
        removed_count = 0

        def _repl(m: re.Match) -> str:
            nonlocal removed_count
            raw = m.group(1)
            abs_path = self._resolve_ref_to_abs(md_dir, raw)
            if abs_path is not None and abs_path in removed_abs:
                removed_count += 1
                return ""
            return m.group(0)

        new_text = self._MD_IMAGE_PATH_RE.sub(_repl, md_text or "")
        return new_text, removed_count

    def _extract_markdown_image_context(
        self, *, md_text: str, image_rel_posix: str
    ) -> tuple[str | None, str | None]:
        """从 Markdown 中提取指定图片的上下文（alt 文本 + 最近标题）。"""
        if not md_text:
            return None, None
        lines = md_text.replace("\r\n", "\n").split("\n")
        image_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        target = self._normalize_ref_path(image_rel_posix)

        for idx, line in enumerate(lines):
            m = image_re.search(line)
            if not m:
                continue
            alt = (m.group(1) or "").strip()
            ref = self._normalize_ref_path(m.group(2))
            if ref != target:
                continue

            heading = None
            for j in range(idx - 1, -1, -1):
                s = (lines[j] or "").strip()
                if not s:
                    continue
                if s.startswith("#"):
                    heading = s.lstrip("#").strip()
                    break
            return alt or None, heading or None

        return None, None

    def _extract_layout_image_name(self, *, doc_dir: Path, image_filename: str) -> str | None:
        """从 layout.json 中严格提取指定图片的标题（按 MinerU layout 结构）。"""
        layout_path = doc_dir / "layout.json"
        if not layout_path.exists():
            return None

        try:
            data = json.loads(layout_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        def _collect_span_contents(block: dict[str, Any]) -> str:
            v = block.get("content")
            if isinstance(v, str) and v.strip():
                return v.strip()

            parts: list[str] = []
            lines = block.get("lines")
            if isinstance(lines, list):
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    spans = line.get("spans")
                    if not isinstance(spans, list):
                        continue
                    for sp in spans:
                        if not isinstance(sp, dict):
                            continue
                        c = sp.get("content")
                        if isinstance(c, str) and c.strip():
                            parts.append(c.strip())
                            continue
                        t = sp.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
            return " ".join(parts).strip()

        def _image_body_has_path(block: dict[str, Any]) -> tuple[bool, int | None]:
            if block.get("type") != "image_body":
                return False, None
            group_id = block.get("group_id")
            group_id_int = int(group_id) if isinstance(group_id, (int, float)) else None
            lines = block.get("lines")
            if not isinstance(lines, list):
                return False, group_id_int
            for line in lines:
                if not isinstance(line, dict):
                    continue
                spans = line.get("spans")
                if not isinstance(spans, list):
                    continue
                for sp in spans:
                    if not isinstance(sp, dict):
                        continue
                    ip = sp.get("image_path")
                    if isinstance(ip, str) and ip.strip() == image_filename:
                        return True, group_id_int
            return False, group_id_int

        pdf_info = data.get("pdf_info")
        if not isinstance(pdf_info, list):
            return None

        for page in pdf_info:
            if not isinstance(page, dict):
                continue
            blocks = page.get("preproc_blocks")
            if not isinstance(blocks, list):
                continue

            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") != "image":
                    continue
                sub_blocks = b.get("blocks")
                if not isinstance(sub_blocks, list):
                    continue

                matched = False
                matched_group: int | None = None
                for sb in sub_blocks:
                    if not isinstance(sb, dict):
                        continue
                    ok, gid = _image_body_has_path(sb)
                    if ok:
                        matched = True
                        matched_group = gid
                        break
                if not matched:
                    continue

                captions: list[str] = []
                for sb in sub_blocks:
                    if not isinstance(sb, dict):
                        continue
                    sb_type = sb.get("type")
                    if sb_type not in {"image_caption", "image_title"}:
                        continue
                    gid = sb.get("group_id")
                    gid_int = int(gid) if isinstance(gid, (int, float)) else None
                    if matched_group is not None and gid_int is not None and gid_int != matched_group:
                        continue
                    txt = _collect_span_contents(sb)
                    if txt:
                        captions.append(txt)

                if captions:
                    seen: set[str] = set()
                    deduped: list[str] = []
                    for t in captions:
                        if t in seen:
                            continue
                        seen.add(t)
                        deduped.append(t)
                    return " ".join(deduped).strip() or None

                return None

        return None

    def _derive_chart_name(
        self,
        *,
        model_json: dict[str, Any],
        doc_dir: Path,
        image_rel_posix: str,
        per_doc_counter: dict[str, int],
    ) -> str:
        """生成用于 RAG 展示的图表名称（避免 uuid）。"""
        image_filename = Path(image_rel_posix).name

        layout_name = self._extract_layout_image_name(
            doc_dir=doc_dir, image_filename=image_filename
        )
        if layout_name:
            return layout_name

        for k in ("description", "title", "chart_title", "name"):
            v = model_json.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        md_files = sorted(doc_dir.glob("*.md"))
        for md_path in md_files:
            try:
                md_text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            alt, heading = self._extract_markdown_image_context(
                md_text=md_text, image_rel_posix=image_rel_posix
            )
            if alt:
                return alt
            if heading:
                return heading

        chart_type = model_json.get("chart_type")
        if isinstance(chart_type, str) and chart_type.strip():
            return f"图表（{chart_type.strip()}）"
        return "图表"

    def _validate_no_deleted_refs_in_md(
        self, md_text: str, *, md_dir: Path, removed_abs: set[Path]
    ) -> None:
        for m in self._MD_IMAGE_PATH_RE.finditer(md_text or ""):
            abs_path = self._resolve_ref_to_abs(md_dir, m.group(1))
            if abs_path is not None and abs_path in removed_abs:
                raise RuntimeError(f"Markdown 仍引用已删除图片: md_dir={md_dir}, ref={m.group(1)}")

    def _validate_no_deleted_refs_in_json(
        self, obj: Any, *, base_dir: Path, removed_abs: set[Path]
    ) -> None:
        if isinstance(obj, str):
            abs_path = self._resolve_ref_to_abs(base_dir, obj)
            if abs_path is not None and abs_path in removed_abs:
                raise RuntimeError(f"JSON 仍引用已删除图片: base_dir={base_dir}, ref={obj}")
            return
        if isinstance(obj, list):
            for x in obj:
                self._validate_no_deleted_refs_in_json(x, base_dir=base_dir, removed_abs=removed_abs)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                self._validate_no_deleted_refs_in_json(v, base_dir=base_dir, removed_abs=removed_abs)
            return

    async def run(
        self,
        *,
        input_root: str | Path = Path("data") / "intermediates_cleaned",
        output_root: str | Path = Path("data") / "pic_to_json",
        chart_json_dirname: str = "chart_json",
        max_images: int | None = None,
        concurrency: int = 1,
        strict: bool = True,
        progress_callback: Any | None = None,
        progress_every: int = 10,
        timeout_seconds: int = 300,
        resume: bool = False,
        source_id_filter: str | None = None,
    ) -> dict[str, Any]:
        """执行 T094 批处理。"""
        input_root_p = Path(input_root).resolve()
        output_root_p = Path(output_root).resolve()
        workspace_root = output_root_p

        if not input_root_p.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_root_p}")

        # 严格模式下，必须从 SQLite 获取激活 Prompt（不允许静默回退到代码 Seed）
        if strict:
            active_prompt = self.prompt_service.get_active_prompt(SCOPE_CHART_EXTRACTION)
            if (
                active_prompt is None
                or not getattr(active_prompt, "content", None)
                or not str(active_prompt.content).strip()
            ):
                raise RuntimeError(
                    f"未找到激活提示词: {SCOPE_CHART_EXTRACTION}（请先在 SQLite prompts 中发布激活版本，或运行 scripts/update-chart-extraction-prompt.py）"
                )

        # 默认按 ID 在 pic_to_json 下作为工作目录（有指定 source_id 时生效）
        if source_id_filter:
            sid_root = output_root_p / source_id_filter / "pic_to_json"
            try:
                sid_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            workspace_root = sid_root.resolve()

        # 1) 复制
        t0 = time.monotonic()
        should_copy = True
        if resume and output_root_p.exists():
            if source_id_filter:
                images_root_pre = (output_root_p / source_id_filter / "pic_to_json" / "images").resolve()
                has_files = False
                try:
                    if images_root_pre.exists() and images_root_pre.is_dir():
                        for _p in images_root_pre.rglob("*"):
                            if _p.is_file():
                                has_files = True
                                break
                except Exception:
                    has_files = False
                if has_files:
                    should_copy = False
                    if callable(progress_callback):
                        progress_callback(
                            {
                                "stage": "copy_start",
                                "ts": datetime.now().isoformat(),
                                "input_root": str(input_root_p),
                                "output_root": str(output_root_p),
                                "msg": "Resuming mode: images present, skipping copy",
                            }
                        )
                else:
                    should_copy = True
                    if callable(progress_callback):
                        progress_callback(
                            {
                                "stage": "copy_start",
                                "ts": datetime.now().isoformat(),
                                "input_root": str(input_root_p),
                                "output_root": str(output_root_p),
                                "msg": "Resuming mode: images missing, performing copy",
                            }
                        )
            else:
                should_copy = False
                if callable(progress_callback):
                    progress_callback(
                        {
                            "stage": "copy_start",
                            "ts": datetime.now().isoformat(),
                            "input_root": str(input_root_p),
                            "output_root": str(output_root_p),
                            "msg": "Resuming mode: skipping copy",
                        }
                    )

        if should_copy:
            if callable(progress_callback):
                progress_callback(
                    {
                        "stage": "copy_start",
                        "ts": datetime.now().isoformat(),
                        "input_root": str(input_root_p),
                        "output_root": str(output_root_p),
                    }
                )
            # 重要：当指定 source_id_filter（单文件/按文件并发）时，禁止删除全局 output_root，
            # 否则并发任务会互相 rmtree 导致图片“文件不存在”。
            # 仅清理当前 source_id 的 pic_to_json 子目录即可。
            if source_id_filter:
                output_root_p.mkdir(parents=True, exist_ok=True)
                sid_root = (output_root_p / source_id_filter / "pic_to_json").resolve()
                try:
                    if sid_root.exists():
                        shutil.rmtree(sid_root)
                except OSError:
                    # Windows 下可能存在文件句柄占用；失败时继续尝试覆盖拷贝
                    pass
            else:
                # 避免 input_root 与 output_root 相同导致误删或复制到自身
                if output_root_p.exists() and input_root_p != output_root_p:
                    shutil.rmtree(output_root_p)
                output_root_p.mkdir(parents=True, exist_ok=True)
            cleaned_dirs: list[tuple[Path, Path]] = []
            try:
                for sub in sorted(input_root_p.iterdir()):
                    if not sub.is_dir():
                        continue
                    if source_id_filter and sub.name != source_id_filter:
                        continue
                    cd = sub / "cleaned_doc"
                    if cd.exists() and cd.is_dir():
                        dest = output_root_p / sub.name / "pic_to_json"
                        cleaned_dirs.append((cd, dest))
            except OSError:
                cleaned_dirs = []
            if cleaned_dirs:
                for src_dir, dst_dir in cleaned_dirs:
                    dst_dir.parent.mkdir(parents=True, exist_ok=True)
                    # 清理旧 chart_json（仅限 pic_to_json 下，安全）
                    try:
                        cj = (dst_dir / chart_json_dirname).resolve()
                        if cj.exists():
                            shutil.rmtree(cj)
                    except OSError:
                        pass
                    # 若目录已存在（例如 Windows 下前序清理失败），允许覆盖；否则走普通 copytree
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            else:
                # 无 cleaned_doc 时不复制整个根目录；如果两者不同且确有需要再复制
                if input_root_p != output_root_p:
                    shutil.copytree(input_root_p, output_root_p, dirs_exist_ok=False)
            if callable(progress_callback):
                progress_callback(
                    {
                        "stage": "copy_done",
                        "ts": datetime.now().isoformat(),
                        "duration_s": round(time.monotonic() - t0, 3),
                    }
                )
            # 指定单一 ID 时，工作目录定位到该 ID 的 pic_to_json
            if source_id_filter:
                sid_root = output_root_p / source_id_filter / "pic_to_json"
                try:
                    sid_root.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
                workspace_root = sid_root.resolve()
                images_root_post = workspace_root / "images"
                if not images_root_post.exists() or not images_root_post.is_dir():
                    src_cd = (input_root_p / source_id_filter / "cleaned_doc").resolve()
                    if src_cd.exists() and src_cd.is_dir():
                        try:
                            shutil.copytree(src_cd, workspace_root, dirs_exist_ok=True)
                        except OSError:
                            pass
                stable = False
                for _ in range(3):
                    if images_root_post.exists() and images_root_post.is_dir():
                        any_file = False
                        try:
                            for _p in images_root_post.rglob("*"):
                                if _p.is_file():
                                    any_file = True
                                    break
                        except Exception:
                            any_file = False
                        if any_file:
                            stable = True
                            break
                    time.sleep(0.3)

        # 2) 扫描 images
        t1 = time.monotonic()
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        images: list[Path] = []
        if source_id_filter:
            images_root = workspace_root / "images"
            if not images_root.exists() or not images_root.is_dir():
                raise RuntimeError(f"images 目录不存在: {images_root}")
            for p in images_root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in exts:
                    continue
                images.append(p)
        else:
            for p in workspace_root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in exts:
                    continue
                if not any(part.lower() == "images" for part in p.parts):
                    continue
                if not any(part.lower() == "pic_to_json" for part in p.parts):
                    continue
                images.append(p)

        images.sort()
        if max_images is not None and max_images > 0:
            images = images[: int(max_images)]
        if not images:
            if source_id_filter:
                raise RuntimeError(f"images 目录下未找到任何图片: {workspace_root / 'images'}")
            raise RuntimeError(f"workspace 中未找到任何 images 图片: {workspace_root}")
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "scan_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t1, 3),
                    "total_images": len(images),
                }
            )
        if callable(progress_callback):
            try:
                per_id: dict[str, int] = {}
                if source_id_filter:
                    per_id[source_id_filter] = len(images)
                else:
                    for p in images:
                        rel_parts = p.resolve().relative_to(workspace_root).parts
                        if rel_parts:
                            sid = rel_parts[0]
                            per_id[sid] = per_id.get(sid, 0) + 1
                progress_callback(
                    {
                        "stage": "scan_summary",
                        "ts": datetime.now().isoformat(),
                        "by_id": per_id,
                    }
                )
            except Exception:
                pass

        # 2.5) 断点续跑状态文件（JSONL）
        state_path = (workspace_root / "t094_state.jsonl").resolve()

        def _image_sig(p: Path) -> dict[str, int] | None:
            try:
                st = p.stat()
                return {
                    "size": int(st.st_size),
                    "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
                }
            except OSError:
                return None

        def _load_state_latest(path: Path) -> dict[str, dict[str, Any]]:
            if not path.exists():
                return {}
            latest: dict[str, dict[str, Any]] = {}
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    src = obj.get("source_image")
                    if isinstance(src, str) and src.strip():
                        latest[src] = obj
            except OSError:
                return {}
            return latest

        async def _append_state(path: Path, record: dict[str, Any], *, lock: asyncio.Lock) -> None:
            line = json.dumps(record, ensure_ascii=False)
            async with lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

        state_latest: dict[str, dict[str, Any]] = _load_state_latest(state_path) if resume else {}
        state_lock = asyncio.Lock()

        if resume and not state_path.exists():
            for jf in sorted(workspace_root.rglob(f"{chart_json_dirname}/*.json")):
                try:
                    obj = json.loads(jf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                src = obj.get("_source_image")
                if not isinstance(src, str) or not src.strip():
                    continue
                img_abs = (workspace_root / Path(src)).resolve()
                sig = _image_sig(img_abs)
                rec = {
                    "ts": datetime.now().isoformat(),
                    "source_image": src,
                    "status": "chart",
                    "image_sig": sig,
                    "chart_json": jf.relative_to(workspace_root).as_posix(),
                    "chart_id": obj.get("_chart_id"),
                    "chart_name": obj.get("_chart_name"),
                    "chart_type": obj.get("chart_type"),
                }
                await _append_state(state_path, rec, lock=state_lock)
                state_latest[src] = rec

        # 3) 批量图转 JSON（按“源文件/目录”并发）
        started_at = datetime.now().isoformat()
        items: list[dict[str, Any]] = []
        non_chart_images: list[Path] = []
        # 并发单位：doc_dir（每个 source_id 的 pic_to_json 目录）
        doc_semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        results_lock = asyncio.Lock()
        done_count = 0
        last_unload_at = 0

        async def _one(image_path: Path, *, per_doc_counter: dict[str, int]) -> dict[str, Any]:
            rel_image = image_path.relative_to(workspace_root).as_posix()
            if image_path.parent.name.lower() != "images":
                raise RuntimeError(f"图片不在 images 目录下: {image_path}")
            doc_dir = image_path.parent.parent
            chart_json_dir = doc_dir / chart_json_dirname
            chart_json_dir.mkdir(parents=True, exist_ok=True)
            out_json_path = (chart_json_dir / f"{image_path.stem}.json").resolve()

            if resume:
                if out_json_path.exists():
                    try:
                        model_json = json.loads(out_json_path.read_text(encoding="utf-8"))
                        return {
                            "source_image": rel_image,
                            "result_json": out_json_path.relative_to(workspace_root).as_posix(),
                            "is_chart": True,
                            "status": "chart",
                            "should_delete": False,
                            "chart_id": model_json.get("_chart_id"),
                            "chart_name": model_json.get("_chart_name"),
                        }
                    except Exception:
                        try:
                            out_json_path.unlink()
                        except OSError:
                            pass
                sig_now = _image_sig(image_path)
                last = state_latest.get(rel_image)
                if isinstance(last, dict) and last.get("image_sig") == sig_now:
                    status = last.get("status")
                    if status == "chart":
                        rel_json = last.get("chart_json") or out_json_path.relative_to(workspace_root).as_posix()
                        if rel_json and (workspace_root / Path(rel_json)).exists():
                            return {
                                "source_image": rel_image,
                                "result_json": rel_json,
                                "is_chart": True,
                                "status": "chart",
                                "should_delete": False,
                                "chart_id": last.get("chart_id"),
                                "chart_name": last.get("chart_name"),
                            }
                    elif status == "non_chart":
                        return {
                            "source_image": rel_image,
                            "result_json": None,
                            "is_chart": False,
                            "status": "non_chart",
                            "should_delete": True,
                            "chart_id": None,
                            "chart_name": None,
                        }
                    elif status == "error":
                        return {
                            "source_image": rel_image,
                            "result_json": None,
                            "is_chart": False,
                            "status": "error",
                            "should_delete": False,
                            "error": last.get("error"),
                            "chart_id": None,
                            "chart_name": None,
                        }

            chart_data = await self._chart_to_json(
                chart_image_path=str(image_path),
                chart_type="other",
                strict_json=strict,
                timeout_seconds=timeout_seconds,
            )
            if chart_data is None:
                raise RuntimeError(f"图转 JSON 返回空: {image_path}")
            model_json = chart_data.json_content
            is_chart = (
                self._extract_is_chart_strict(model_json, image=image_path)
                if strict
                else bool(model_json.get("is_chart"))
            )

            chart_id = "chart_" + hashlib.sha1(rel_image.encode("utf-8")).hexdigest()[:12]
            chart_name = self._derive_chart_name(
                model_json=model_json,
                doc_dir=doc_dir,
                image_rel_posix=f"images/{image_path.name}",
                per_doc_counter=per_doc_counter,
            )

            model_json = {
                **model_json,
                "_chart_id": chart_id,
                "_chart_name": chart_name,
                "_source_image": rel_image,
            }

            if is_chart:
                out_json_path.write_text(
                    json.dumps(model_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result_json_rel = out_json_path.relative_to(workspace_root).as_posix()
                status = "chart"
                should_delete = False
            else:
                if out_json_path.exists():
                    out_json_path.unlink()
                result_json_rel = None
                status = "non_chart"
                should_delete = True

            sig_now = _image_sig(image_path)
            rec = {
                "ts": datetime.now().isoformat(),
                "source_image": rel_image,
                "status": status,
                "image_sig": sig_now,
                "chart_json": result_json_rel,
                "chart_id": chart_id,
                "chart_name": chart_name if is_chart else None,
                "chart_type": model_json.get("chart_type") if isinstance(model_json, dict) else None,
            }
            await _append_state(state_path, rec, lock=state_lock)
            state_latest[rel_image] = rec

            return {
                "source_image": rel_image,
                "result_json": result_json_rel,
                "is_chart": is_chart,
                "status": status,
                "should_delete": should_delete,
                "chart_id": chart_id,
                "chart_name": chart_name,
            }

        t2 = time.monotonic()

        async def _one_with_state(img: Path, *, per_doc_counter: dict[str, int]) -> dict[str, Any]:
            try:
                return await _one(img, per_doc_counter=per_doc_counter)
            except Exception as exc:
                rel_image = img.relative_to(workspace_root).as_posix()
                sig_now = _image_sig(img)
                rec = {
                    "ts": datetime.now().isoformat(),
                    "source_image": rel_image,
                    "status": "error",
                    "image_sig": sig_now,
                    "error": str(exc),
                }
                try:
                    await _append_state(state_path, rec, lock=state_lock)
                    state_latest[rel_image] = rec
                except Exception:
                    pass
                return {
                    "source_image": rel_image,
                    "result_json": None,
                    "is_chart": False,
                    "status": "error",
                    "should_delete": False,
                    "chart_id": None,
                    "chart_name": None,
                }

        # group images by doc_dir (pic_to_json directory)
        by_doc: dict[Path, list[Path]] = {}
        for img in images:
            doc_dir = img.parent.parent
            by_doc.setdefault(doc_dir, []).append(img)

        async def _process_doc(doc_dir: Path, imgs: list[Path]) -> None:
            nonlocal done_count, last_unload_at
            async with doc_semaphore:
                per_doc_counter: dict[str, int] = {}
                for img in imgs:
                    row = await _one_with_state(img, per_doc_counter=per_doc_counter)
                    async with results_lock:
                        items.append(row)
                        done_count += 1
                        if row.get("should_delete"):
                            non_chart_images.append(workspace_root / Path(row["source_image"]))

                        if callable(progress_callback) and (
                            done_count == 1
                            or done_count % max(1, int(progress_every)) == 0
                            or done_count == len(images)
                        ):
                            progress_callback(
                                {
                                    "stage": "chart_to_json_progress",
                                    "ts": datetime.now().isoformat(),
                                    "done": done_count,
                                    "total": len(images),
                                    "last_image": row.get("source_image"),
                                    "last_is_chart": row.get("is_chart"),
                                    "last_status": row.get("status"),
                                    "elapsed_s": round(time.monotonic() - t2, 3),
                                }
                            )

                        # Ollama only: periodically unload to free GPU memory
                        if done_count - last_unload_at >= 60:
                            last_unload_at = done_count
                            # outside lock: avoid blocking other updates
                            do_unload = True
                        else:
                            do_unload = False

                    if do_unload:
                        unloaded = await self.llm_runtime.unload_model_if_ollama(
                            SCOPE_CHART_EXTRACTION
                        )
                        if unloaded:
                            if callable(progress_callback):
                                progress_callback(
                                    {
                                        "stage": "ollama_unload",
                                        "ts": datetime.now().isoformat(),
                                        "msg": "Unloaded Ollama model to free GPU memory",
                                    }
                                )
                            await asyncio.sleep(2.0)

        tasks = [asyncio.create_task(_process_doc(d, imgs)) for d, imgs in by_doc.items()]
        await asyncio.gather(*tasks)

        removed_abs: set[Path] = {p.resolve() for p in non_chart_images}

        # 4) 删除非图表图片
        t3 = time.monotonic()
        deleted_count = 0
        for p in non_chart_images:
            try:
                if p.exists():
                    p.unlink()
                    deleted_count += 1
            except OSError as exc:
                raise RuntimeError(f"删除图片失败: {p}: {exc}") from exc
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "delete_non_charts_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t3, 3),
                    "deleted_images": deleted_count,
                }
            )

        # 5) 清理 md / json 引用
        t4 = time.monotonic()
        md_changed = 0
        md_refs_removed = 0
        for md_path in sorted(workspace_root.rglob("*.md")):
            if not any(part.lower() == "pic_to_json" for part in md_path.parts):
                continue
            original = md_path.read_text(encoding="utf-8")
            updated, removed_n = self._remove_md_refs(
                original, md_dir=md_path.parent, removed_abs=removed_abs
            )
            if removed_n > 0 and updated != original:
                md_path.write_text(updated, encoding="utf-8")
                md_changed += 1
                md_refs_removed += removed_n

        json_changed = 0
        json_refs_removed = 0
        for jf in sorted(workspace_root.rglob("*.json")):
            if not any(part.lower() == "pic_to_json" for part in jf.parts):
                continue
            raw = jf.read_text(encoding="utf-8")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON 文件解析失败: {jf}: {exc}") from exc

            new_obj, changed = self._prune_json_obj(obj, base_dir=jf.parent, removed_abs=removed_abs)
            if changed:
                jf.write_text(
                    json.dumps(new_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                json_changed += 1
                json_refs_removed += 1
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "prune_refs_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t4, 3),
                    "md_files_changed": md_changed,
                    "md_refs_removed": md_refs_removed,
                    "json_files_changed": json_changed,
                }
            )

        # 6) 一致性校验
        t5 = time.monotonic()
        for md_path in sorted(workspace_root.rglob("*.md")):
            if not any(part.lower() == "pic_to_json" for part in md_path.parts):
                continue
            self._validate_no_deleted_refs_in_md(
                md_path.read_text(encoding="utf-8"),
                md_dir=md_path.parent,
                removed_abs=removed_abs,
            )
        for jf in sorted(workspace_root.rglob("*.json")):
            if not any(part.lower() == "pic_to_json" for part in jf.parts):
                continue
            obj = json.loads(jf.read_text(encoding="utf-8"))
            self._validate_no_deleted_refs_in_json(obj, base_dir=jf.parent, removed_abs=removed_abs)
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "validate_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t5, 3),
                }
            )

        finished_at = datetime.now().isoformat()
        report = {
            "task": "T094",
            "input_root": str(input_root_p),
            "output_root": str(output_root_p),
            "workspace_root": str(workspace_root),
            "started_at": started_at,
            "finished_at": finished_at,
            "total_images_scanned": len(images),
            "total_results": len(items),
            "non_chart_images": [p.relative_to(workspace_root).as_posix() for p in non_chart_images],
            "deleted_images": deleted_count,
            "md_files_changed": md_changed,
            "md_refs_removed": md_refs_removed,
            "json_files_changed": json_changed,
            "json_refs_removed_estimate": json_refs_removed,
            "items": items,
        }
        (workspace_root / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (workspace_root / "chart_index.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report


from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _utc_ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")


def _round_ms(v: float | None) -> float | None:
    if v is None:
        return None
    return round(float(v), 2)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    xs = sorted(float(x) for x in values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


@dataclass
class RunResult:
    ok: bool
    status_code: int | None
    client_time_ms: float | None
    server_total_time_ms: float | None
    total_tokens: int | None
    html_length: int | None
    error: str | None


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_request_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "workspace_id": args.workspace_id,
    }
    if args.collection_name:
        payload["collection_name"] = args.collection_name
    if args.outline_filename:
        payload["outline_filename"] = args.outline_filename
    if args.polish_outline is not None:
        payload["polish_outline"] = args.polish_outline
    if args.polish_sections is not None:
        payload["polish_sections"] = args.polish_sections
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.rerank_top_n is not None:
        payload["rerank_top_n"] = args.rerank_top_n
    if args.score_threshold is not None:
        payload["score_threshold"] = args.score_threshold
    return payload


def _run_once(*, client: httpx.Client, url: str, payload: dict[str, Any]) -> RunResult:
    start = time.time()
    try:
        resp = client.post(url, json=payload)
    except Exception as e:
        return RunResult(
            ok=False,
            status_code=None,
            client_time_ms=_round_ms((time.time() - start) * 1000),
            server_total_time_ms=None,
            total_tokens=None,
            html_length=None,
            error=str(e),
        )

    client_ms = _round_ms((time.time() - start) * 1000)
    if resp.status_code != 200:
        err_txt = resp.text
        if len(err_txt) > 2000:
            err_txt = err_txt[:2000] + "…"
        return RunResult(
            ok=False,
            status_code=resp.status_code,
            client_time_ms=client_ms,
            server_total_time_ms=None,
            total_tokens=None,
            html_length=None,
            error=err_txt,
        )

    try:
        data = resp.json()
    except Exception as e:
        return RunResult(
            ok=False,
            status_code=resp.status_code,
            client_time_ms=client_ms,
            server_total_time_ms=None,
            total_tokens=None,
            html_length=None,
            error=f"invalid_json_response: {e}; body={resp.text[:2000]}",
        )

    coverage = data.get("coverage") or {}
    total_time_ms = coverage.get("total_time_ms")
    total_tokens = coverage.get("total_tokens")
    html_length = data.get("html_length")
    return RunResult(
        ok=True,
        status_code=resp.status_code,
        client_time_ms=client_ms,
        server_total_time_ms=_round_ms(total_time_ms),
        total_tokens=int(total_tokens) if isinstance(total_tokens, int) else None,
        html_length=int(html_length) if isinstance(html_length, int) else None,
        error=None,
    )


def _render_md_report(*, meta: dict[str, Any], results: list[RunResult]) -> str:
    ok_runs = [r for r in results if r.ok]
    client_ms = [float(r.client_time_ms) for r in ok_runs if r.client_time_ms is not None]
    server_ms = [float(r.server_total_time_ms) for r in ok_runs if r.server_total_time_ms is not None]
    tokens = [int(r.total_tokens) for r in ok_runs if r.total_tokens is not None]

    def _line(label: str, v: Any) -> str:
        return f"- {label}: {v}"

    lines: list[str] = [
        "# 阶段5：白皮书生成性能与稳定性评估（T225）",
        "",
        "## 运行信息",
        "",
        _line("生成时间", meta.get("generated_at")),
        _line("请求 URL", meta.get("url")),
        _line("请求参数", f"`{_safe_json(meta.get('request_payload'))}`"),
        _line("总轮次", meta.get("runs")),
        _line("成功轮次", len(ok_runs)),
        _line("失败轮次", len(results) - len(ok_runs)),
        "",
        "## 指标汇总（仅统计成功轮次）",
        "",
        _line("client_time_ms.p50", _round_ms(_percentile(client_ms, 50))),
        _line("client_time_ms.p95", _round_ms(_percentile(client_ms, 95))),
        _line("client_time_ms.max", _round_ms(_percentile(client_ms, 100))),
        _line("server_total_time_ms.p50", _round_ms(_percentile(server_ms, 50))),
        _line("server_total_time_ms.p95", _round_ms(_percentile(server_ms, 95))),
        _line("server_total_time_ms.max", _round_ms(_percentile(server_ms, 100))),
        _line("total_tokens.p50", _percentile([float(x) for x in tokens], 50) if tokens else None),
        _line("total_tokens.p95", _percentile([float(x) for x in tokens], 95) if tokens else None),
        "",
        "## 明细",
        "",
        "| # | ok | status | client_ms | server_ms | tokens | html_len | error |",
        "|---:|:--:|:------:|---------:|----------:|-------:|--------:|-------|",
    ]

    for idx, r in enumerate(results, start=1):
        err = (r.error or "").replace("\n", " ")
        if len(err) > 120:
            err = err[:120] + "…"
        lines.append(
            "| {idx} | {ok} | {status} | {client} | {server} | {tokens} | {hlen} | {err} |".format(
                idx=idx,
                ok="y" if r.ok else "n",
                status=r.status_code if r.status_code is not None else "-",
                client=r.client_time_ms if r.client_time_ms is not None else "-",
                server=r.server_total_time_ms if r.server_total_time_ms is not None else "-",
                tokens=r.total_tokens if r.total_tokens is not None else "-",
                hlen=r.html_length if r.html_length is not None else "-",
                err=err or "-",
            )
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `client_time_ms`：脚本侧端到端耗时（包含网络与服务端处理）。",
            "- `server_total_time_ms`：服务端返回的 `coverage.total_time_ms`（若服务端未返回则为空）。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API Base URL（不含 /v1）")
    parser.add_argument("--endpoint", default="/v1/targets/whitepaper/generate", help="生成端点路径")
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--outline-filename", default=None)
    parser.add_argument("--polish-outline", default=None, choices=["true", "false"])
    parser.add_argument("--polish-sections", default=None, choices=["true", "false"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank-top-n", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", default=str(Path("docs") / "process" / "ai-doc-platform-phase2"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    polish_outline = None
    if args.polish_outline is not None:
        polish_outline = args.polish_outline == "true"
    args.polish_outline = polish_outline

    polish_sections = None
    if args.polish_sections is not None:
        polish_sections = args.polish_sections == "true"
    args.polish_sections = polish_sections

    payload = _build_request_payload(args)
    url = args.base_url.rstrip("/") + args.endpoint
    ts = _utc_ts()
    out_dir = Path(args.output_dir)
    json_path = out_dir / f"t225-perf-{ts}.json"
    md_path = out_dir / f"t225-perf-{ts}.md"

    meta: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "url": url,
        "request_payload": payload,
        "runs": int(args.runs),
        "timeout_seconds": float(args.timeout),
    }

    if args.dry_run:
        report = {"meta": meta, "results": []}
        _write_json(json_path, report)
        _write_md(md_path, _render_md_report(meta=meta, results=[]))
        print(f"dry_run_written: {md_path}")
        return 0

    results: list[RunResult] = []
    with httpx.Client(timeout=httpx.Timeout(args.timeout)) as client:
        for _ in range(int(args.runs)):
            results.append(_run_once(client=client, url=url, payload=payload))

    report = {
        "meta": meta,
        "results": [
            {
                "ok": r.ok,
                "status_code": r.status_code,
                "client_time_ms": r.client_time_ms,
                "server_total_time_ms": r.server_total_time_ms,
                "total_tokens": r.total_tokens,
                "html_length": r.html_length,
                "error": r.error,
            }
            for r in results
        ],
    }
    _write_json(json_path, report)
    _write_md(md_path, _render_md_report(meta=meta, results=results))
    print(f"written: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


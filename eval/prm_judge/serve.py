"""为评估结果提供安全、支持视频拖动的本地 HTTP 服务。

用法示例：
    prm-judge serve --run-root eval/results/run_YYMMDD_HHMMSS --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse

from .metrics import recompute_record_metrics
from .visualize import (
    build_interactive_html_report,
    load_visualization_records,
    resolve_metric_config,
    resolve_prm_backend,
    resolve_success_threshold,
    select_video_value,
)


class InvalidRange(ValueError):
    """表示客户端提交了语法非法或越界的单区间 Range。"""


def _media_search_roots(run_root: Path) -> tuple[Path, ...]:
    """从运行元数据恢复相对媒体路径的原始解析上下文。"""
    roots: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if resolved not in roots:
            roots.append(resolved)

    params_path = run_root / "run_params.json"
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        params = {}

    working_directory = params.get("working_directory")
    if working_directory:
        add(Path(str(working_directory)))

    manifest_value = params.get("manifest")
    if manifest_value:
        manifest = Path(str(manifest_value)).expanduser()
        if manifest.is_absolute():
            add(manifest.parent)
        else:
            for candidate in (run_root, *run_root.parents):
                if (candidate / manifest).is_file():
                    add(candidate)
                    break

    add(run_root)
    add(Path.cwd())
    return tuple(roots)


def _local_video_path(
    value: str,
    run_root: Path,
    search_roots: tuple[Path, ...] | None = None,
) -> Path | None:
    """把 run 记录的视频值解析为本地路径；公网 URI 不进入媒体映射。"""
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    roots = search_roots or _media_search_roots(run_root)
    candidates = [(root / path).resolve() for root in roots]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _media_token(path: Path) -> str:
    """为绝对媒体路径生成不可猜目录结构的稳定令牌。"""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:32]


def _parse_byte_range(header: str, size: int) -> tuple[int, int]:
    """解析一个 RFC 7233 字节区间并返回闭区间端点。"""
    if size <= 0 or not header.startswith("bytes="):
        raise InvalidRange(header)
    value = header[6:].strip()
    if not value or "," in value or "-" not in value:
        raise InvalidRange(header)
    start_text, end_text = (part.strip() for part in value.split("-", 1))
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise InvalidRange(header)
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            if start < 0 or start >= size:
                raise InvalidRange(header)
            end = size - 1 if not end_text else int(end_text)
            if end < start:
                raise InvalidRange(header)
            end = min(end, size - 1)
    except ValueError as exc:
        raise InvalidRange(header) from exc
    return start, end


def _copy_range(source: BinaryIO, target: BinaryIO, start: int, length: int) -> None:
    """流式复制指定字节段，避免把大视频一次性读入内存。"""
    source.seek(start)
    remaining = length
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        target.write(chunk)
        remaining -= len(chunk)


def create_report_server(
    run_root: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """构建报告服务器；调用方可在测试中自行启动和关闭。"""
    run_root = run_root.expanduser().resolve()
    records = load_visualization_records(run_root)
    threshold = resolve_success_threshold(run_root)
    metric_config = resolve_metric_config(run_root, threshold)
    recompute_record_metrics(records, metric_config)
    prm_backend = resolve_prm_backend(run_root)
    media_search_roots = _media_search_roots(run_root)

    media: dict[str, Path] = {}
    external_sources: dict[int, str] = {}
    for record in records:
        raw = select_video_value(record)
        path = _local_video_path(raw, run_root, media_search_roots)
        if path is None:
            external_sources[id(record)] = raw
            continue
        media[_media_token(path)] = path

    def video_src(record: dict) -> str:
        raw = select_video_value(record)
        path = _local_video_path(raw, run_root, media_search_roots)
        if path is None:
            return external_sources.get(id(record), raw)
        return f"/media/{_media_token(path)}"

    report_html = build_interactive_html_report(
        records=records,
        run_root=run_root,
        success_threshold=threshold,
        prm_backend=prm_backend,
        video_src_resolver=video_src,
    ).encode("utf-8")

    class ReportHandler(BaseHTTPRequestHandler):
        """仅提供动态报告和令牌映射中的媒体文件。"""

        server_version = "PRMJudgeReport/1.5"

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(send_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._dispatch(send_body=False)

        def _dispatch(self, *, send_body: bool) -> None:
            parsed_path = urlparse(self.path).path
            if parsed_path in {"/", "/report.html"}:
                self._send_report(send_body)
                return
            prefix = "/media/"
            if parsed_path.startswith(prefix):
                token = parsed_path[len(prefix) :]
                if not token or "/" in token or token not in media:
                    self.send_error(404, "Unknown media token")
                    return
                self._send_media(media[token], send_body)
                return
            self.send_error(404, "Not found")

        def _send_report(self, send_body: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(report_html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(report_html)

        def _send_media(self, path: Path, send_body: bool) -> None:
            try:
                size = path.stat().st_size
            except OSError:
                self.send_error(404, "Recorded media is unavailable")
                return

            range_header = self.headers.get("Range")
            if range_header:
                try:
                    start, end = _parse_byte_range(range_header, size)
                except InvalidRange:
                    self.send_response(416)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
            else:
                start, end = 0, size - 1
                status = 200

            length = end - start + 1
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            try:
                with path.open("rb") as source:
                    _copy_range(source, self.wfile, start, length)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            print(f"[SERVE] {self.address_string()} - {format % args}")

    return ThreadingHTTPServer((host, port), ReportHandler)


def serve_run(run_root: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    """持续运行报告服务，直到用户按 Ctrl-C。"""
    server = create_report_server(run_root, host, port)
    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    print(f"[SERVE] Report: http://{display_host}:{bound_port}/report.html")
    print("[SERVE] Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVE] Stopped.")
    finally:
        server.server_close()

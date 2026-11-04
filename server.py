"""本地服务（只用 Python 标准库）：

    python3 server.py          # 然后浏览器打开 http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import mimetypes
import sys
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vectortool.pipeline import Params, decode_image, segment
from vectortool.trace import vectorize_segmentation

ROOT = Path(__file__).resolve().parent
PORT = 8765
_images: "OrderedDict[str, object]" = OrderedDict()
_lock = threading.Lock()


def vectorize(iid: str, params: dict):
    with _lock:
        img = _images.get(iid)
    if img is None:
        return 404, {"detail": "图片已过期，请重新上传"}
    t = time.time()
    p = Params.from_dict(params)
    seg = segment(img, p)
    svg, shapes, stats = vectorize_segmentation(seg, p)
    stats["total_ms"] = int((time.time() - t) * 1000)
    palette = [{"match": m, "out": c, "share": round(sh, 5)}
               for m, c, sh in zip(seg.match_hex, seg.colors_hex, seg.shares)]
    return 200, {"svg": svg, "shapes": shapes, "palette": palette, "stats": stats,
                 "width": seg.orig_size[0], "height": seg.orig_size[1]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/static/index.html"
        for prefix, folder in (("/static/", ROOT / "web"), ("/samples/", ROOT / "samples")):
            if path.startswith(prefix):
                f = (folder / path[len(prefix):]).resolve()
                if folder.resolve() in f.parents and f.is_file():
                    ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                    if ctype.startswith("text/") or ctype.endswith("javascript"):
                        ctype += "; charset=utf-8"
                    return self._send(200, f.read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            if self.path == "/api/upload":
                try:
                    img = decode_image(body)
                except Exception:
                    return self._json(400, {"detail": "无法读取这张图片"})
                iid = uuid.uuid4().hex[:12]
                with _lock:
                    _images[iid] = img
                    while len(_images) > 12:
                        _images.popitem(last=False)
                h, w = img.shape[:2]
                return self._json(200, {"id": iid, "width": w, "height": h})
            if self.path == "/api/vectorize":
                req = json.loads(body or b"{}")
                code, obj = vectorize(req.get("id", ""), req.get("params") or {})
                return self._json(code, obj)
            self._json(404, {"detail": "not found"})
        except Exception as e:  # 让前端看到错误而不是断开
            import traceback
            traceback.print_exc()
            self._json(500, {"detail": f"处理出错：{e}"})


def main():
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    port = int(nums[0]) if nums else PORT
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Vector Tool 已启动：{url}   （按 Ctrl+C 退出）")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

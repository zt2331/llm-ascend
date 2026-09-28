#!/usr/bin/env python
"""chat_relay.py —— 聊天页 + vLLM 流式反向代理（同一个端口）。

为什么需要它
------------
云 IDE（如华为云 online-*.huawei.com）把端口代理到 `/proxy/<port>/` 下，
而那层反向代理（通常是 nginx）**默认开启 proxy_buffering**：
它会把 vLLM 的 SSE 流**攒齐再一次性吐给浏览器**。

后果是页面上测出来的数字全是假的：
    TTFT      ≈ 整个生成耗时（因为第一个 chunk 直到最后才到）
    decode    ≈ 天文数字（因为所有 chunk 在同一瞬间到达）
实测到过 "decode 3207 tok/s / 生成阶段 319ms / 输出 1024 tokens" 这种值。

怎么解决
--------
nginx 认 **`X-Accel-Buffering: no`** 这个【响应头】——注意它是上游发的，
客户端设不了。所以这里放一个中继：

    浏览器 ──HTTPS──> 云IDE代理 ──HTTP──> chat_relay ──HTTP──> vLLM

中继从 vLLM 拿到 SSE 后**立刻逐块转发**，并在响应头里声明
`X-Accel-Buffering: no` / `Cache-Control: no-transform`，
让前置代理不再缓冲。同时它还把静态页一起托管，于是
**页面和 API 同源**，连 CORS 都不需要了。

用法
----
    python scripts/chat_relay.py                        # 页面+API 都在 :8080
    python scripts/chat_relay.py --port 8080 --vllm-port 8000
    # 然后浏览器打开  https://<host>/proxy/8080/
"""
import argparse
import http.client
import os
import socketserver
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
# 需要转发到 vLLM 的路径前缀
PROXY_PREFIXES = ("/v1/", "/metrics", "/health", "/docs", "/openapi.json")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon",
}

# 这些头由我们自己决定，不能从上游原样透传
DROP_RESP_HEADERS = {"transfer-encoding", "connection", "content-length",
                     "content-encoding", "keep-alive"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "chat-relay/1.0"
    vllm_host = "127.0.0.1"
    vllm_port = 8000

    # 默认日志太吵，关掉；出错时我们自己在 body 里给信息
    def log_message(self, fmt, *args):
        pass

    # ---------------- 静态页面 ----------------
    def _static(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", ""):
            path = "/chat.html"
        rel = os.path.normpath(path.lstrip("/"))
        full = os.path.join(WEB_DIR, rel)
        # 防目录穿越
        if not os.path.abspath(full).startswith(os.path.abspath(WEB_DIR)):
            return self._plain(403, "forbidden")
        if not os.path.isfile(full):
            return self._plain(404, f"not found: {path}\n"
                                    f"（本服务只托管 {WEB_DIR} 下的文件，"
                                    f"/v1/* 等路径会转发到 vLLM）")
        with open(full, "rb") as f:
            data = f.read()
        ext = os.path.splitext(full)[1].lower()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        # 页面本身绝不缓存，免得改了看不到
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _plain(self, code, text):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---------------- 反向代理（★流式转发） ----------------
    def _proxy(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None

        fwd = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in ("host", "connection", "keep-alive", "accept-encoding",
                      "transfer-encoding"):
                continue
            fwd[k] = v
        # ★ 不要 gzip：压缩会让代理倾向于攒够一块再发，破坏流式
        fwd["Accept-Encoding"] = "identity"
        fwd["Connection"] = "close"

        try:
            conn = http.client.HTTPConnection(self.vllm_host, self.vllm_port,
                                              timeout=1800)
            conn.request(method, self.path, body=body, headers=fwd)
            resp = conn.getresponse()
        except Exception as e:
            return self._plain(502, f"无法连接 vLLM ({self.vllm_host}:{self.vllm_port}): {e}\n"
                                    f"请先确认它已启动：\n"
                                    f"  curl -s http://127.0.0.1:{self.vllm_port}/v1/models")

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in DROP_RESP_HEADERS:
                continue
            self.send_header(k, v)
        # ★★ 关键两行：告诉前面的 nginx 不要缓冲，边收边转
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Cache-Control", "no-cache, no-transform")
        # 不设 Content-Length，用 Connection: close 界定 body ——
        # 这样浏览器能【边到边显示】，而不是等 EOF
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                chunk = resp.read1(16384)      # read1: 有数据就立刻返回
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()             # ★ 立即推给前端
        except (BrokenPipeError, ConnectionResetError):
            pass                               # 浏览器中途断开，正常
        finally:
            conn.close()

    # ---------------- 路由 ----------------
    def do_GET(self):
        if self.path.startswith(PROXY_PREFIXES):
            self._proxy("GET")
        else:
            self._static()

    def do_POST(self):
        if self.path.startswith(PROXY_PREFIXES):
            self._proxy("POST")
        else:
            self._plain(404, f"不支持的路径: {self.path}")

    def do_OPTIONS(self):
        # 同源部署其实用不到，但保留以防直连场景
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080, help="本服务端口（默认 8080）")
    ap.add_argument("--vllm-port", type=int, default=8000, help="vLLM 端口（默认 8000）")
    ap.add_argument("--vllm-host", default="127.0.0.1")
    args = ap.parse_args()

    Handler.vllm_host = args.vllm_host
    Handler.vllm_port = args.vllm_port

    if not os.path.isdir(WEB_DIR):
        print(f"[FAIL] 找不到 web 目录: {WEB_DIR}")
        return 1

    print("=" * 66)
    print(" 聊天中继已启动（静态页 + vLLM 流式反代，同端口）")
    print(f"   页面   : http://127.0.0.1:{args.port}/")
    print(f"   接口   : http://127.0.0.1:{args.port}/v1/*  →  "
          f"http://{args.vllm_host}:{args.vllm_port}")
    print("")
    print("   云 IDE 里请用同源代理地址访问页面，例如：")
    print(f"     https://<你的host>/proxy/{args.port}/")
    print("   此时页面与接口同源，不需要 CORS，也不会有 mixed content 问题。")
    print("")
    print("   本服务已对上游响应加 X-Accel-Buffering: no —— 这是让云 IDE")
    print("   那层 nginx 不缓冲 SSE、实现真流式的关键。")
    print("=" * 66)
    try:
        Server(("0.0.0.0", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())

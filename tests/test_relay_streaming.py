#!/usr/bin/env python
"""test_relay_streaming.py —— 验证 chat_relay 的透传是【真流式】。

为什么必须测：如果中继自己把 SSE 攒起来，页面上测出的 TTFT/decode 会全部
失真（实测出现过 "decode 3207 tok/s"）。所以这里用一个【按固定间隔吐 token】
的假 vLLM，再测量经由中继后各 chunk 的到达时刻：若中继是透传的，到达间隔
应当与假 vLLM 的发送间隔一致；若被缓冲，所有 chunk 会在末尾同时到达。

用法（项目根目录）:
    python tests/test_relay_streaming.py
"""
import http.client, json, subprocess, sys, threading, time, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CHUNK, GAP = 6, 0.20
FAKE_PORT, RELAY_PORT = 18001, 18080


N_CHUNK, GAP = 6, 0.20

class FakeVLLM(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for i in range(N_CHUNK):
            self.wfile.write(
                f'data: {json.dumps({"choices":[{"delta":{"content":"t%d"%i}}]})}\n\n'.encode())
            self.wfile.flush()
            time.sleep(GAP)
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
    def do_GET(self):                      # /v1/models
        b = json.dumps({"data":[{"id":"fake-model"}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)


def main():
    threading.Thread(
        target=ThreadingHTTPServer(("127.0.0.1", FAKE_PORT), FakeVLLM).serve_forever,
        daemon=True).start()
    relay = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "scripts", "chat_relay.py"),
         "--port", str(RELAY_PORT), "--vllm-port", str(FAKE_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    ok = True
    try:
        c = http.client.HTTPConnection("127.0.0.1", RELAY_PORT, timeout=30)
        c.request("GET", "/")
        r = c.getresponse(); body = r.read()
        print(f"① 静态页: HTTP {r.status}, {len(body)} 字节")
        ok &= r.status == 200 and b"<!DOCTYPE html>" in body

        c = http.client.HTTPConnection("127.0.0.1", RELAY_PORT, timeout=10)
        c.request("GET", "/v1/models")
        r = c.getresponse(); d = json.loads(r.read())
        print(f"② 反代 /v1/models: {d}")
        ok &= d["data"][0]["id"] == "fake-model"

        c = http.client.HTTPConnection("127.0.0.1", RELAY_PORT, timeout=30)
        c.request("POST", "/v1/chat/completions", body=b'{"stream":true}',
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        accel = r.getheader("X-Accel-Buffering")
        print(f"③ 响应头 X-Accel-Buffering = {accel!r}")
        ok &= accel == "no"

        t0 = time.time(); st = []
        while True:
            line = r.fp.readline()
            if not line: break
            if line.startswith(b"data:"): st.append(time.time() - t0)
        print(f"   SSE 到达时刻: {[round(s,3) for s in st]}")
        want = (len(st) - 1) * GAP
        spread = st[-1] - st[0]
        print(f"   首尾跨度 {spread:.3f}s（期望 ≈{want:.2f}s）")
        if spread > want * 0.6:
            print("   ✅ 透传是真流式")
        else:
            print("   ❌ 被缓冲：chunk 几乎同时到达"); ok = False
    finally:
        relay.terminate()
    print("\n" + ("全部通过 ✅" if ok else "存在失败 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

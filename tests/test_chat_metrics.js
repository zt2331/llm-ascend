// test_chat_metrics.js —— 用合成 SSE 流验证聊天页的度量口径
//
// 用法（在项目根目录）:
//     node tests/test_chat_metrics.js
//     node tests/test_chat_metrics.js <提取好的 script 文件>   # 可选
//
// 做法：把 web/chat.html 里的 <script> 抽出来，放进沙箱执行，
//       给它假的 DOM / fetch / 虚拟时钟，喂合成 SSE 流，
//       断言 TTFT、prefill、decode 的数值。
//
// 为什么需要它：页面上那些速度数字如果口径算错了，看起来依然"很合理"，
//       肉眼极难发现。本测试曾在实现里抓出一个真实缺陷 —— 生成阶段误把
//       流尾部的 usage/[DONE] 也算进去，导致 decode 速度被低估 20%。
const vm = require("vm");
const fs = require("fs");
const path = require("path");

const ROOT = path.dirname(__dirname);
let code;
if (process.argv[2]) {
  code = fs.readFileSync(process.argv[2], "utf8");
} else {
  const html = fs.readFileSync(path.join(ROOT, "web", "chat.html"), "utf8");
  const m = html.match(/<script>([\s\S]*?)<\/script>/);
  if (!m) { console.error("未能在 web/chat.html 里找到 <script> 块"); process.exit(2); }
  code = m[1];
}

// ---------- 虚拟时钟：完全可控，避免真实 sleep 导致的抖动 ----------
let VNOW = 0;
const performance = { now: () => VNOW };
const advance = ms => { VNOW += ms; };

// ---------- 极简 DOM ----------
const store = {};
function mkEl(id) {
  const el = {
    id, textContent: "", innerHTML: "", value: "", className: "",
    style: {}, disabled: false, scrollTop: 0, scrollHeight: 0,
    clientWidth: 340, width: 0, height: 0, onclick: null,
    _kids: [],
    appendChild(c) { this._kids.push(c); return c; },
    insertAdjacentHTML() {},
    addEventListener() {}, remove() {}, focus() {}, after() {},
    querySelector() { return mkEl("q"); },
    querySelectorAll() { return []; },
    getContext() { return new Proxy({}, { get: () => () => {} }); },
  };
  return el;
}
const document = {
  getElementById: id => (store[id] || (store[id] = mkEl(id))),
  createElement: () => mkEl("new"),
};
store.maxtok = mkEl("maxtok"); store.maxtok.value = "8";
store.temp   = mkEl("temp");   store.temp.value   = "0";
store.model  = mkEl("model");  store.model.value  = "qwen-test";
store.addr   = mkEl("addr");   store.addr.value   = "http://127.0.0.1:8000";
store.input  = mkEl("input");  store.input.value  = "";

// ---------- 合成 SSE ----------
// 每项 = [距上一次事件的虚拟毫秒数, 该事件的 SSE 文本]
let SSE_PLAN = [];
function makeFetch() {
  return async () => {
    let i = 0;
    return {
      ok: true,
      body: {
        getReader: () => ({
          async read() {
            if (i >= SSE_PLAN.length) return { done: true, value: undefined };
            const [dt, text] = SSE_PLAN[i++];
            advance(dt);
            return { done: false, value: new TextEncoder().encode(text) };
          },
        }),
      },
    };
  };
}

const sandbox = {
  document, performance, console, Math, JSON, Date, URL,
  location: { href: "http://127.0.0.1:8080/chat.html", protocol: "http:" },
  parseInt, parseFloat, isFinite, alert: () => {},
  TextEncoder, TextDecoder,
  setInterval, clearInterval,
  window: { devicePixelRatio: 1, addEventListener() {} },
  fetch: makeFetch(),
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

// 加载被测脚本，并暴露内部函数
vm.runInContext(code + `
;globalThis.__ask = ask;
globalThis.__setBase = v => { base_ = v; };
globalThis.__guessBase = guessBase;
globalThis.__diagnose = diagnose;
globalThis.__setHref = h => {
  location.href = h;
  location.protocol = new URL(h).protocol;
};
`, sandbox);
const ask = sandbox.__ask;
const guessBase = sandbox.__guessBase;
sandbox.__setBase("http://127.0.0.1:8000");

// ---------- 地址推断：云 IDE 代理是最容易踩的场景 ----------
function testGuessBase() {
  console.log("=== 地址自动推断（guessBase）===");
  const CASES = [
    // 页面所在的 URL                                 期望的默认 vLLM 地址
    ["http://127.0.0.1:8080/chat.html",
     "http://127.0.0.1:8000"],
    ["http://10.0.0.5:8080/chat.html",
     "http://10.0.0.5:8000"],
    // ★ 华为云 online IDE：端口被代理到 /proxy/<port>/ 下
    ["https://online-sz01.hicomp.huawei.com/proxy/8080/chat.html",
     "https://online-sz01.hicomp.huawei.com/proxy/8000"],
    // 代理前缀带更深路径
    ["https://x.example.com/a/b/proxy/8080/",
     "https://x.example.com/a/b/proxy/8000"],
    // 普通 https 站点：不能写 http 直连（会被按混合内容拦），退回同源
    ["https://example.com/chat.html",
     "https://example.com"],
  ];
  let bad = 0;
  for (const [href, want] of CASES) {
    sandbox.__setHref(href);
    const got = guessBase();
    const ok = got === want;
    console.log(`  ${ok ? "✅" : "❌"} ${href}\n       → ${got}${ok ? "" : `  (期望 ${want})`}`);
    ok ? pass++ : (fail++, bad++);
  }
  return bad;
}

// 诊断信息应当能识别出「127.0.0.1」与「混合内容」这两个真凶
function testDiagnose() {
  console.log("\n=== 失败诊断（diagnose）===");
  sandbox.__setHref("https://online-sz01.hicomp.huawei.com/proxy/8080/chat.html");
  const t1 = sandbox.__diagnose(new Error("Failed to fetch"), "http://127.0.0.1:8000");
  const checks = [
    ["识别出 127.0.0.1 不是服务器", t1.includes("127.0.0.1") && t1.includes("不是服务器")],
    ["识别出 HTTPS→HTTP 混合内容", t1.includes("混合内容")],
    ["给出自测地址", t1.includes("/v1/models")],
  ];
  for (const [name, ok] of checks) {
    console.log(`  ${ok ? "✅" : "❌"} ${name}`);
    ok ? pass++ : fail++;
  }
}

// ---------- 构造一个典型流：prefill 300ms，然后 6 个 token 每 50ms 一个 ----------
function buildPlan({ ttftMs, nTok, perTokMs, promptTokens }) {
  const plan = [];
  plan.push([0, `data: ${JSON.stringify({ choices: [{ delta: { role: "assistant" } }] })}\n\n`]);
  plan.push([ttftMs, `data: ${JSON.stringify({ choices: [{ delta: { content: "t0" } }] })}\n\n`]);
  for (let k = 1; k < nTok; k++) {
    plan.push([perTokMs, `data: ${JSON.stringify({ choices: [{ delta: { content: "t" + k } }] })}\n\n`]);
  }
  plan.push([perTokMs, `data: ${JSON.stringify({ choices: [{ delta: {} }], usage: {
      prompt_tokens: promptTokens, completion_tokens: nTok, total_tokens: promptTokens + nTok } })}\n\n`]);
  plan.push([0, "data: [DONE]\n\n"]);
  return plan;
}

const TOL = 0.05;   // 5% 容差（虚拟时钟是精确的，留一点余量给实现细节）
let pass = 0, fail = 0;
function check(name, got, want, tol = TOL) {
  const ok = want === 0 ? Math.abs(got) < 1e-6
                        : Math.abs(got - want) / Math.abs(want) <= tol;
  console.log(`  ${ok ? "✅" : "❌"} ${name}: got=${got.toFixed(3)} want=${want.toFixed(3)}`);
  ok ? pass++ : fail++;
}

(async () => {
  testGuessBase();
  testDiagnose();

  const CASES = [
    { ttftMs: 300, nTok: 6, perTokMs: 50, promptTokens: 512, label: "典型：512 in / 6 out" },
    { ttftMs: 800, nTok: 16, perTokMs: 25, promptTokens: 2048, label: "长 prompt、快速 decode" },
    { ttftMs: 120, nTok: 4, perTokMs: 100, promptTokens: 64, label: "短 prompt、慢 decode" },
  ];

  for (const c of CASES) {
    VNOW = 0;
    SSE_PLAN = buildPlan(c);
    console.log(`\n=== ${c.label} ===`);
    const m = await ask("hi", null, null);

    // 期望：TTFT = 首 token 到达时刻；生成阶段 = (nTok-1)*perTokMs
    check("TTFT (ms)", m.ttft, c.ttftMs);
    check("生成阶段 genMs (ms)", m.genMs, (c.nTok - 1) * c.perTokMs);
    check("prompt_tokens（来自 usage）", m.promptTokens, c.promptTokens);
    check("completion_tokens（来自 usage）", m.completionTokens, c.nTok);

    const prefill = m.promptTokens / (m.ttft / 1000);
    const decode  = (m.completionTokens - 1) / (m.genMs / 1000);
    check("prefill tok/s", prefill, c.promptTokens / (c.ttftMs / 1000));
    check("decode tok/s", decode, (c.nTok - 1) / ((c.nTok - 1) * c.perTokMs / 1000));
    console.log(`     文本拼接: "${m.text}" (${m.text.length} 字符)`);
    if (m.text !== Array.from({ length: c.nTok }, (_, k) => "t" + k).join("")) {
      console.log("  ❌ 文本拼接错误"); fail++;
    } else { console.log("  ✅ 文本拼接正确"); pass++; }
    if (!m.exact) { console.log("  ❌ 应识别为精确 usage"); fail++; }
  }

  console.log(`\n${"=".repeat(50)}\n通过 ${pass} / 失败 ${fail}`);
  process.exit(fail ? 1 : 0);
})();

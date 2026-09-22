#!/usr/bin/env python3
"""把内存中锚点(host state 槽位, 320 槽 x 147 MiB)打满 —— 密集短消息版。

原理:
  引擎在 prompt「最后 32 个消息边界」上放私有长锚点, 每条 continuation 至多保留
  --max-long-anchors-per-continuation=16 个(满则替换最浅的)。因此一个会话只要有
  16+ 条消息, 就能拿满 16 个锚点; 消息再短(约 1.5K token/条)也能在 ~30K token
  的 prompt 里产出 16 个锚点, 比整段 90K 大文本的 prefill 快 3 倍。

  20 条这样的会话 ≈ 320 个状态镜像槽位 → 打满。打满后的证明:
    1. request_log 的 context_cache.occupancy 显示 host_state_slots ≈ 320;
    2. 新会话到来时, 日志出现锚点 victim 的 CopyToHost/Drop(旧锚点被驱逐)。

用法:
  python3 fill_anchors.py                 # 打满 30000 端口的主服务
  python3 fill_anchors.py 30001 20        # 压力服务, 上限 20 条会话
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_paths import BENCH_MESSAGES, REQUEST_LOG  # noqa: E402

MANIFEST = Path(os.environ.get("NINFER_FILL_MANIFEST") or BENCH_MESSAGES / "long_niah_256k.json")
CHARS_PER_TOKEN = 32338 / 7680

N_MESSAGES = 20
MSG_TOKENS = 1500    # 每条消息 ~1.5K token; 20 条 ≈ 30K + system ≈ 32K
SYSTEM_TOKENS = 1000
MAX_SESSIONS = 30    # 安全上限(320 槽 / 17 ≈ 19, 留余量防部分会话锚点不足)


def load_corpus() -> str:
    msgs = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    body = msgs[1]["content"]
    return body[: body.index("</document>")]


def chunk(corpus: str, offset: int, tokens: int) -> str:
    n = int(tokens * CHARS_PER_TOKEN)
    return (corpus[offset % len(corpus):] + corpus * 2)[:n]


def build_session(session: int, corpus: str) -> list:
    system = "You are myai. Long shared test corpus follows. " + chunk(corpus, session * 311, SYSTEM_TOKENS)
    msgs = [{"role": "system", "content": system}]
    for i in range(N_MESSAGES):
        role = "user" if i % 2 == 0 else "assistant"
        salt = f"SESS-{session:02d}-MSG-{i:02d}-"
        msgs.append({"role": role, "content": salt + chunk(corpus, session * 971 + i * 4099, MSG_TOKENS)})
    msgs.append({"role": "user", "content": f"Reply with exactly: SESSION-{session:02d} ACK."})
    return msgs


def post(base: str, msgs: list, timeout: int = 900) -> dict:
    payload = json.dumps({
        "model": "myai", "messages": msgs, "stream": False,
        "max_completion_tokens": 1, "temperature": 0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    u = out.get("usage", {})
    return {"elapsed": time.time() - t0, "prompt": u.get("prompt_tokens"), "out": u.get("completion_tokens")}


def occupancy() -> dict:
    """读 request_log 最近一条带 occupancy 的记录(引擎在每轮 throughput 时写入)。"""
    try:
        lines = REQUEST_LOG.read_text(errors="replace").splitlines()
        for line in reversed(lines[-400:]):
            if '"occupancy"' in line:
                occ = json.loads(line)["context_cache"]["occupancy"]
                pinned = occ.get("device_state_slots", 0) + occ.get("host_state_slots", 0)
                return {"dev": occ.get("device_state_slots"), "host": occ.get("host_state_slots"),
                        "pinned": pinned, "kv_pages": occ.get("device_main_kv_pages"),
                        "host_kv_mb": round(occ.get("host_kv_bytes", 0) / 1e6, 1)}
    except Exception:
        pass
    return {}


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "30000"
    max_sessions = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_SESSIONS
    base = f"http://127.0.0.1:{port}"
    corpus = load_corpus()
    o = occupancy()
    print(f"start occupancy: host_state={o.get('host')}/{320} device_state={o.get('device')} "
          f"pinned={o.get('pinned')} kv_pages={o.get('kv_pages')} host_kv={o.get('host_kv_mb')}MB", flush=True)
    t0 = time.time()
    s = 0
    while s < max_sessions:
        s += 1
        msgs = build_session(s - 1, corpus)
        try:
            r = post(base, msgs)
        except Exception as e:
            print(f"[{s:2d}] FAIL {type(e).__name__}: {e}", flush=True)
            break
        o = occupancy()
        host = o.get("host")
        status = "FULL" if (host is not None and host >= 318) else ""
        print(f"[{s:2d}] prompt={r['prompt']:>6} {r['elapsed']:>5.1f}s | "
              f"host={host}/320 dev={o.get('device')} pinned={o.get('pinned')} "
              f"host_kv={o.get('host_kv_mb')}MB {status} | wall={time.time()-t0:.0f}s", flush=True)
        if status:
            print(f"== host state pool FULL (host_state_slots={host}/320) after {s} sessions ==")
            break
    print(f"DONE | final {occupancy()} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

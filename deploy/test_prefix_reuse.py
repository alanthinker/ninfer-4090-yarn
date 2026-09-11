#!/usr/bin/env python3
"""跨会话前缀复用测试：system+tools 相同、仅用户问题不同。

存在的理由：DSH 每个新会话都要发约 9.6K 的 system prompt + 工具定义，实测新会话
`cache 0 (0.0%)`、TTFT ~8s，而同会话追问 `cache 99.9%`、TTFT ~0.27s。要区分两种可能：

  A. 引擎不能跨会话复用「system+tools 之后」这个边界 → 引擎侧问题
  B. DSH 每次发的内容并非逐字节相同 → 客户端侧问题

本脚本发两次请求：system 与 tools 完全相同，只有最后的用户问题不同（模拟两个新会话）。
只要第二次出现约 9.6K 的 cache 命中，就说明引擎侧的跨会话复用是work的，问题在 B。

用法:
  python3 test_prefix_reuse.py            # 默认 system+tools 约 9.6K token
  python3 test_prefix_reuse.py 5000       # 指定 system+tools 目标 token 数
"""
import json
import sys
import urllib.request
from pathlib import Path

BENCH_CLI = Path("/root/ai/large_models/_ninfer_repos/ninfer-4090d-bench/examples/cli")
BASE = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "myai"
CHARS_PER_TOKEN = 32338 / 7680


def big_text(tokens: int) -> str:
    msgs = json.loads((BENCH_CLI / "messages" / "long_niah_256k.json").read_text(encoding="utf-8"))
    body = msgs[1]["content"]
    hay = body[: body.index("</document>")]
    reps = int(tokens * CHARS_PER_TOKEN) // len(hay) + 2
    return (hay * reps)[: int(tokens * CHARS_PER_TOKEN)]


def tools_block(n_tools: int = 6) -> list:
    """构造一组体积可观的工具定义（模拟 DSH 的工具注入）。"""
    filler = big_text(600)
    out = []
    for i in range(n_tools):
        out.append({
            "type": "function",
            "function": {
                "name": f"workspace_tool_{i}",
                "description": f"Tool {i}. {filler}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"arg_{j}": {"type": "string",
                                     "description": f"argument {j}: {filler[:200]}"}
                        for j in range(6)
                    },
                    "required": ["arg_0"],
                },
            },
        })
    return out


def send(label: str, system: str, tools: list, user_msg: str) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_msg}],
        "tools": tools,
        "enable_thinking": False,
        "max_completion_tokens": 32,
        "stream": False,
    }).encode()
    req = urllib.request.Request(BASE, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    t = d.get("timings", {})
    u = d.get("usage", {})
    print(f"  {label:<34} prompt={u.get('prompt_tokens'):>6}  "
          f"cache={t.get('cache_n', 0):>6}  "
          f"TTFT={t.get('ttft_ms', 0)/1000:>6.2f}s  "
          f"prefill={t.get('prompt_per_second', 0):>7.0f} tok/s")
    return t


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 9600
    # system + tools 挑大梁，用户问题很短
    sys_tokens = int(target * 0.55)
    system = ("You are a coding agent. " + big_text(sys_tokens) +
              "\nAlways answer concisely.")
    tools = tools_block()

    print(f"构造: system ≈ {sys_tokens} token, tools = {len(tools)} 个定义, "
          f"目标合计 ≈ {target} token\n")
    print("请求（system 与 tools 完全相同，只有用户问题不同）:")
    t1 = send("第1次 (新会话 A)", system, tools, "Say READY.")
    t2 = send("第2次 (新会话 B, 同前缀)", system, tools, "Say OK.")

    print()
    p1 = t1.get("prompt_tokens", 0)
    c2 = t2.get("cache_n", 0)
    print(f"  第1次 prompt={p1} cache={t1.get('cache_n',0)}")
    print(f"  第2次 prompt={t2.get('prompt_tokens',0)} cache={c2} "
          f"({100*c2/max(t2.get('prompt_tokens',1),1):.1f}%)")
    print()
    if c2 > 0.5 * p1:
        print(f"  >>> 跨会话复用【可用】: 第二次命中 {c2} token，"
              f"约为 system+tools 的长度。")
        print("      结论: 引擎侧没问题 → DSH 新会话的 9.6K 前缀并非逐字节相同。")
    elif c2 > 0:
        print(f"  >>> 部分复用: 命中 {c2} token ({100*c2/p1:.1f}%)，小于 system+tools 长度。")
    else:
        print("  >>> 跨会话复用【未生效】: 第二次仍是 0。引擎侧未利用该边界。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""多模态 + YaRN 回归测试。

存在的理由很具体：`text_context_impl.h` 的 prefill 路径里，带图的 chunk 会把 rope 位置存成
`[len,3]` 的 MRoPE 矩阵，而纯文本是 `[len,1]`。`scale_positions_yarn` 的校验只接受 `ne[1]==1`
的扁平向量，所以修复前**任何带图/视频的请求都会抛 invalid_argument**。这个脚本就是那次修复的
回归测试——它必须真的发一次图。

用法:
  python3 test_yarn_vision.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_paths import BENCH_CLI  # noqa: E402
BASE = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "myai"

# fixture 名 -> oracle。每条事实是一组**可接受的写法**（任一命中即算该条通过），因为 fixture
# 自带的是英文 oracle，而模型面对中文提问会用中文回答。
CASES = {
    "image_chart": [
        ["NIFER VISION 731"],
        ["three red circles", "3 个红色", "3个红色", "三个红色", "红色圆形 3", "3；", "；3；"],
    ],
    "image_natural": [
        ["mailbox 24", "邮箱 24", "邮箱**24**", "**24**", "24"],
        ["sun on right", "右侧", "右边", "right"],
    ],
}


def absolutize(content):
    """把 fixture 的内部媒体格式翻译成 HTTP 层真正接受的 OpenAI 格式。

    fixture 用的是 `{"type":"image","image":"<path>"}`（CLI 侧的内部形态），而 Chat
    Completions 端点只认 `image_url`，且 URL 必须是 data: URI 或 http(s)。
    本地路径不被接受，所以这里读文件并做 base64。
    """
    import base64

    if isinstance(content, str):
        return content
    out = []
    for part in content:
        part = dict(part)
        kind = part.get("type")
        if kind in ("image", "video"):
            path = BENCH_CLI.parent.parent / part["image"]
            mime = "video/mp4" if path.suffix == ".mp4" else "image/png"
            blob = base64.b64encode(path.read_bytes()).decode("ascii")
            field = "image_url" if kind == "image" else "video_url"
            out.append({"type": field, field: {"url": f"data:{mime};base64,{blob}"}})
        else:
            out.append(part)
    return out


def run_case(name: str, facts: list) -> bool:
    msgs = json.loads((BENCH_CLI / "messages" / f"{name}.json").read_text(encoding="utf-8"))
    for m in msgs:
        m["content"] = absolutize(m["content"])

    payload = json.dumps(
        {
            "model": MODEL,
            "messages": msgs,
            "enable_thinking": False,
            "max_completion_tokens": 128,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(BASE, data=payload, headers={"Content-Type": "application/json"})

    print(f"\n{'=' * 72}")
    print(f"多模态用例: {name}   期望事实: {facts}")
    print("=" * 72)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        print(f"  ✗ HTTP {exc.code}: {body}")
        return False
    except Exception as exc:  # noqa: BLE001 - report verbatim
        print(f"  ✗ 请求失败: {type(exc).__name__}: {exc}")
        return False
    wall = time.time() - started

    msg = out["choices"][0]["message"]
    answer = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    timings = out.get("timings", {})
    print(f"  prompt_tokens : {out.get('usage', {}).get('prompt_tokens')}")
    print(f"  TTFT          : {timings.get('ttft_ms', 0):.0f} ms")
    print(f"  decode        : {timings.get('predicted_per_second', 0):.0f} tok/s")
    print(f"  耗时          : {wall:.1f}s")
    print(f"  回答          : {(msg.get('content') or '')[:200]!r}")

    low = answer.lower()
    hits = [alts[0] for alts in facts if any(a.lower() in low for a in alts)]
    ok = len(hits) == len(facts)
    print(f"  事实命中      : {len(hits)}/{len(facts)}  {hits}")
    print(f"  >>> {'通过' if ok else '失败'}")
    return ok


def main() -> int:
    results = []
    for name, facts in CASES.items():
        results.append((name, run_case(name, facts)))
    print(f"\n{'=' * 72}")
    for name, ok in results:
        print(f"  {name:<20} {'通过' if ok else '失败'}")
    passed = sum(1 for _, ok in results if ok)
    print(f"  {passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

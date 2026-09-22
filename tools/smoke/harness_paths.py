#!/usr/bin/env python3
"""Shared support for the serve-level cache harnesses in this directory: locations and the few
protocol helpers every harness needs to read the Engine's answers the same way.

The harnesses drive a *running* Engine over HTTP and read the Engine's own artifacts: request dumps
(``reqdump/``), the JSONL request log, and the black-box bench fixtures maintained in the sibling
`ninfer-4090d-bench` checkout. None of those belong in the source tree, so every location resolves
through an environment variable with a repository-relative default::

    NINFER_HARNESS_DIR   Engine state written by tools/smoke/ninfer_service_*.sh
                         (default <repo>/build/harness, which is not tracked)
    NINFER_REQDUMP_DIR   request dumps to take a client shape from (default <harness>/reqdump)
    NINFER_REQUEST_LOG   JSONL request log to read counters from (default <harness>/request_log.jsonl)
    NINFER_SERVICE_LOG   service log carrying HTTP status and fatal lines
                         (default <harness>/ninfer_serve.log)
    NINFER_BENCH_DIR     ninfer-4090d-bench checkout (default <repo>/../ninfer-4090d-bench)
    NINFER_PORT          service port for harnesses that do not take one (default 30000)

An unset variable is not an error: the harnesses fall back to a synthesized client shape, skip the
counters, or report the missing fixture by path instead of failing obscurely.
"""

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

HARNESS_DIR: Path = Path(os.environ.get("NINFER_HARNESS_DIR") or REPO_ROOT / "build" / "harness")
REQDUMP_DIR: Path = Path(os.environ.get("NINFER_REQDUMP_DIR") or HARNESS_DIR / "reqdump")
REQUEST_LOG: Path = Path(os.environ.get("NINFER_REQUEST_LOG") or HARNESS_DIR / "request_log.jsonl")
SERVICE_LOG: Path = Path(os.environ.get("NINFER_SERVICE_LOG") or HARNESS_DIR / "ninfer_serve.log")
BENCH_DIR: Path = Path(os.environ.get("NINFER_BENCH_DIR") or REPO_ROOT.parent / "ninfer-4090d-bench")
BENCH_CLI: Path = BENCH_DIR / "examples" / "cli"
BENCH_MESSAGES: Path = BENCH_CLI / "messages"


def service_base(port: int | str | None = None) -> str:
    """OpenAI-compatible base URL of the service under test."""
    resolved = port if port is not None else os.environ.get("NINFER_PORT", "30000")
    return f"http://127.0.0.1:{resolved}"


def cached_prompt_tokens(usage: dict) -> int:
    """Prompt tokens the Engine served from cache, from an OpenAI-shaped ``usage`` object.

    The Engine reports this in ``usage.prompt_tokens_details.cached_tokens`` (the OpenAI Responses
    and Chat Completions schema). A harness that reads a top-level ``usage.cached_tokens`` instead
    reports 0 for every request and turns a 99.8 % hit into a false "VERDICT miss", so every
    harness reads it through this one helper.
    """
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)

# A/B evidence

Raw output behind [../端到端部署与测试报告.md](../端到端部署与测试报告.md). Kept in the tree because the
report's numbers are only useful if they can be checked.

| File | Produced by |
|---|---|
| `results_merged.json`, `run_merged.txt` | `ninfer-4090-merged:sm89` (`f7a6d215e272`) on port 18080 |
| `results_baseline.json`, `run_baseline.txt` | `ninfer-4090-merged:baseline` (`ee50c1cb930e`) on port 18081 |

Both runs used the same harness and the same server configuration; the only differences are the
image and the port. Harness: `/tmp/verify_merged/run_tests.py` and `run_tests_baseline.py`, which
differ only in the port and the output path. The committed equivalent is
[`deploy/verify_merged_e2e.py`](../../../deploy/verify_merged_e2e.py), and
[`deploy/verify_merged.sh ab`](../../../deploy/verify_merged.sh) reruns both sides and prints the
delta table.

## Provenance of the two images

The baseline must be the same tree without the port, or the delta means nothing. Verified by
scanning the built binaries for ported symbols:

```
symbol                     baseline   sm89
act_quant_g64                     0      5
int8_proj_launch                  0      5
int8_rowsplit_gemm                0      3
q4_q5_attn_input_int8             0      4
q4_linear_swiglu_int8             0      5
device_sm_count                   0      1
```

Zero on the baseline in every case, so the measured difference is attributable to the port.

## Server configuration (identical on both sides)

```
--max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4-e8
--max-concurrency 4 --max-pending-requests 16 --pending-timeout-ms 600000
--prefill-chunk 1024 --spec mtp --draft-tokens 3 --lm-head-draft
--preserve-thinking --vision --vision-max-tokens 32768
```

Cold start: weights 16.9 GiB / 3m13s, engine ready 3m25s, both images.

## Reading the numbers

All rates and latencies come from the server's own `timings` block, not from curl. See the report's
measurement-convention section: a streaming response emits SSE preamble bytes before prefill
completes, so `time_starttransfer` reports a meaningless ~0.1 s on a 70K-token prompt.

`prompt_n` is the tokenizer's count and is authoritative. The `prompt_chars` field is the raw
character count and is kept only to show why it should not be used for sizing: the repeated
Chinese boilerplate tokenizes at roughly 1.57 chars/token, so `110619` chars is `70221` tokens.

Note on the harness field `finish`: it holds the API's `finish_reason` value; the key was named
`finish` when these files were produced.

The two runs' `prompt_n` differ slightly (`70221` vs `70421`, and `144040` vs `145240`) because the
harness salts every paragraph header with fresh random hex, so the two prompts are not
byte-identical. That is intentional - it forces a genuine cold prefill - and it moves the token
count by 0.3-0.8%, which does not affect the comparison. The salt is also what makes `cache_n` zero
in cases b and d.

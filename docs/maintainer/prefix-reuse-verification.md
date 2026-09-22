# Prefix Cache Reuse Verification

## Purpose

Verify that the engine's state management (LRU device eviction + fair-share host
protection) correctly preserves idle conversation state, allowing quick restoration
when switching between conversations.

## The Problem This Protects Against

With 8 device state slots and potentially many concurrent conversations:

1. An active conversation's repeated activity fills all device slots
2. Idle conversations' device replicas get evicted (LRU)
3. **If the host replica is also lost** (evicted from 320 host slots, or never
   created because the state was DeviceOnly), the conversation is unrecoverable
   without full re-prefill (minutes for 100K+ token prompts)

The correct behavior:

- Device eviction is fine (100ms H2D reload)
- Host replicas of protected sessions must survive (fair-share)
- Switching to an idle conversation: LRU frees a device slot → H2D load → done

## Test Script

```bash
python3 tools/smoke/test_prefix_reuse_switch.py --port 30000
```

### What It Does

1. Creates 9 conversations, each with ~8K random tokens (unique per conversation)
2. Makes conversation 9 very active (12 rounds of back-and-forth)
3. Switches back to conversation 1
4. Checks if conversation 1 gets a cache hit

### Expected Results

**PASS (system working correctly):**
```
Conv1 (switch back): prompt=3100 cached=2900 (94%) time=0.9s
```
- Cache hit > 0 means the state survived (at least on host)
- TTFT < 2s means the H2D load was fast (not a full prefill)

**FAIL (system broken):**
```
Conv1 (switch back): prompt=3100 cached=0 (0%) time=7.5s
```
- Cache hit = 0 means the state was lost entirely
- Full prefill time proportional to prompt size

### Known Limitation: Digest Mismatch

If all cache candidates are rejected with `digest-mismatch` (visible in service
log under `reuse-diag`), the test will report FAIL even though the state
management is correct. This is a separate bug in prompt construction /
tokenization that causes the stored checkpoint digest to not match the
recomputed digest for the same input.

**How to distinguish:** Check the service log. If you see `REJECT digest-mismatch`
for the target conversation's checkpoint, the state IS present (it was found in
the catalog) but the content doesn't match. If you see `candidates=0`, the state
was truly lost (evicted from both device and host).

## Configuration Sensitivity

| Config | Effect on test |
|--------|---------------|
| `--device-state-slots 4` | 8 total device slots; more active rounds = more eviction pressure |
| `--host-state-slots 320` | Must be large enough for all conversations' host replicas |
| `--fair-share-buckets 8` | Protects 8 most recent idle sessions; with 9 convs, conv 1 may NOT be protected |
| `--auto-long-anchors N` (default 5) | Each conversation keeps up to N tail anchors, at least `--first-anchor-spacing` apart (more host slots consumed) |
| `--max-private-continuations 16` | Catalog capacity; must be >= number of conversations |

### Sizing for the Test

For 9 conversations with short prompts (~3K tokens each):
- Each conversation: ~2-3 checkpoints (endpoint + 1-2 anchors, since few message boundaries)
- Active conversation (12 rounds): ~15-20 checkpoints
- Total: ~40 checkpoints, well within 320 host slots

For 9 conversations with LONG prompts (100K+ tokens, many messages):
- Each conversation: ~40-50 checkpoints (endpoint + 32 tail + 8-10 spread)
- Total: ~400-450 checkpoints > 320 host slots
- **Will overflow**: some conversations' states will be evicted from host
- The test may FAIL for this reason (not a bug, just insufficient capacity)
- Fix: increase `--host-state-slots` or reduce `--auto-long-anchors`

## Full-Pool Acceptance: The Newest Request Publishes Its Own Anchor

The requirement this section verifies: **the request being sent right now has the highest priority,
so a saturated cache must still publish its own capture.** Otherwise the next message of the same
conversation has nothing to hit and pays a full prefill — the failure mode a user sees as "my
current message was not cached, so the next one can never hit".

A lightly loaded pool hides this: there is always free capacity to place a capture. The acceptance
is therefore always run with `host_state_slots` at its configured maximum.

```bash
# 1. Saturate the pool with *distinct* conversations (each round must use a new FILL_SESSION_BASE,
#    otherwise the round replays the previous round's conversations, hits cache, and adds nothing)
FILL_SESSION_BASE=0  python3 tools/smoke/fill_anchors.py 30000 30   # ... until it prints FULL
FILL_SESSION_BASE=30 python3 tools/smoke/fill_anchors.py 30000 30

# 2. The recorded sequences that failed in production, plus the live client shape
tools/smoke/pool_soak.sh 30000 /tmp/crash243

# 3. The real requests a client complained about, replayed at the saturated pool
python3 tools/smoke/replay_dump.py 30000 $(cat /tmp/userdumps.txt | tr '\n' ' ')
```

`pool_soak.sh` takes a byte offset in each of the Engine's two logs before every sequence and
reports that window with `report_window.py`, so per-request numbers and error counts belong to the
sequence that produced them. Both offsets are required and are not interchangeable: the service log
carries HTTP status and fatal lines, the request log the per-request record. A window passes when
its requests show 500/503/fatal `0`, and a cold request followed by a sibling message of the same
conversation shows the sibling reusing the cold request's own anchor.

Measured on 2026-09-22 at `host_state_slots 320/320` (`--host-state-slots 320`,
`--auto-long-anchors 5`, `--first-anchor-spacing 4096`), one-shot client shape of 9,480 tokens:

| Sequence | Result |
|---|---|
| cold request (developer prefix salted at its head, nothing to reuse) | `path=root`, 0 %, TTFT 7.9 s |
| sibling message of that request, 2 s later | `path=private_long_anchor`, 9,463/9,476 = 99.9 %, TTFT 0.16 s |
| the eight recorded client requests, right after a pool refill | first request re-prefills (4.4 s, 0 %); one sibling landed in a window gap (2,273/4,570 = 50 %, 1.9 s); the other six 99.5–100 % at 0.04–0.20 s |
| soak windows (fork sequence ×4, anchor probe, cold+siblings, 4-way bursts ×2) | 500/503/fatal `0` every window, cache 85.6–100 %, TTFT ≤ 0.31 s |

**A saturated pool publishes the newest boundary, not the whole window.** The request's own tail
anchor - the one its next message resumes from - is published even at `320/320`; the deeper members
of the last-N window are skipped with `capture: skip reason=pressure-no-private-baseline`, because
publishing them would have to evict another session's checkpoint and the per-request destructive
reclaim is spent on the newest frontier. The consequence is measurable: at `320/320` a fork into
the middle of a 4.8K conversation reused 801 of 3,883 tokens (21 %) and a fork into the middle of a
23K conversation reused 7,157 of 14,208 (50 %), while the same two shapes on a small pool with room
to publish reused 62 % and 63 %. Deeper coverage at saturation is therefore a capacity decision
(more `--host-state-slots`, or spending more than one destructive reclaim per cold request), not a
frontend one.

### Reading Cache Hits From a Client

The Engine reports reused prompt tokens in `usage.prompt_tokens_details.cached_tokens`, the OpenAI
Chat Completions and Responses schema. It does not emit a top-level `usage.cached_tokens` (that is
llama.cpp's shape); a harness that reads the top-level field reports `0` for every request and turns
a 99.8 % hit into a false "VERDICT miss". Every harness in `tools/smoke/` reads it through
`harness_paths.cached_prompt_tokens` for exactly that reason.

## When to Run

Run this test after any change to:
- `StateImageStore` (device/host slot allocation, eviction, LRU)
- `MaterializationPlanner` (pressure search, budget)
- Fair-share logic (bucket assignment, protection)
- `program_impl.h` state placement (freeze, move_checkpoint, fork)
- Host-to-device transfer path

## Related Tests

| Test | What it checks |
|------|---------------|
| `tests/ninfer_resource_manager_test` | Unit: state store allocation, eviction logic |
| `tests/ninfer_context_store_test` | Unit: context cache publish/consume |
| This script | Integration: end-to-end multi-conversation switch |

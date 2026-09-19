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
| `--auto-long-anchors 32` | Each conversation gets up to 32 tail anchors (more host slots consumed) |
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

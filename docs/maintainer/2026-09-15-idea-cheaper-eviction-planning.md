# Idea (not a design): cheaper eviction planning for cold / must-evict prompts

**Status:** parking lot. Do not turn this into a redesign unless the problem keeps recurring *and*
matters for the workload. Any real change to the eviction core needs a 100k+ prompt reuse A/B on a
box with a working GPU driver before shipping.

## Observation

For a prompt whose root build is infeasible — it must evict resident checkpoints to fit KV pages or
free a host-state slot — `MaterializationPlanner::plan` runs a budgeted branch-and-bound over the whole
checkpoint catalog. On this box each target evaluation is a host-side `ContextMachineCostModel` fold
(now/future-loss ns, offload bytes, copy ops); the throughput log shows `host ~300–344%` during it. The
budget (`per_candidate_assessment_ns = 0.5 s` floor, a 15 s floor, 90 s cap) forces the search to run
near-to-optimum and to seed every candidate's retention closure. So "evict until it fits" costs
seconds-to-minutes even though every snapshot already knows its owning session and recency.

## The retention policy is not the bottleneck

The "every session keeps 1, 8 fair-share buckets, most-recently-active gets more" policy is a cheap
*input* to the search (owner/checkpoint policies + protected slots). It narrows the victim domain but
does not price the combinations. The cost is in (a) the per-target cost-model fold being slow on this
host, and (b) evaluating many *combinations* (branch-and-bound, `kTargetBudget = 4096`), not single
snapshots. A few hundred snapshots is not a few hundred O(1) lookups.

## Possible direction (only if it becomes a real recurring pain)

Precompute a per-checkpoint "weighted cost per freed byte/slot" (usage-weighted future re-prefill
divided by size), sort, and greedily evict in that order until the request fits — O(N log N) with
O(1) per item. Fall back to the exact branch-and-bound only when the greedy result is close to
infeasible or the loss budget is tight. Keep the retention policy and the anti-regression seeding
guarantees intact.

## Why it is not done

- The exact branch-and-bound and the 0.5 s/candidate + 15 s floors were tuned to fix the 2026-09-11/12
  time-budget regressions (starved retention-closure seeding → spurious fair-share release → 0% reuse
  on 100k+ prompts). Changing the eviction core risks re-introducing them.
- This box's NVIDIA driver is broken, so a real 100k+ reuse A/B is not possible to confirm no regression.

## Related change already made

`materialization_planner.h` now scales the planning floor with prompt size (full 15 s for prompts at
or above `kPlanningFloorFullTokens`, tapering to ~0.25 s for tiny prompts). That addresses *small* cold
prompts. It deliberately does not touch the large-prompt / must-evict case above, on purpose.

## Trigger to revisit

If `ninfer_serve.log` shows repeated `queue`/planning times of ~10 s+ on prompts that must evict
(beyond the small-prompt floor already addressed), and it matters for the workload, scope the
greedy-with-fallback change and validate on a box with a working driver.

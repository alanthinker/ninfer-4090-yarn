// Special-case driver for `pressure KV replica changed before transfer` (reproduced-bug tooling).
//
// Service-level rounds cannot search this path: a fill prompt costs 25 s and the fatal needs a
// narrow shape, so a battery only hits it by accident. This driver builds each candidate shape
// inside ONE process with a tiny context-cache (host_state=4, private=8, kv=32768 tokens): the
// pools saturate in seconds and every pressure branch is reachable without HTTP.
//
// Why one case per process: the throw latches the engine (fail_all_locked -> 503 until restart),
// so a hit kills every later case in the same process. The parent shell runs the cases in a loop.
//
//   ninfer_pressure_prepare_cases <artifact.ninfer> <case>|all     (NINFER_TEST_ARTIFACT works too)
//
// The engine prints the two ends of the verdict: `[pressure] plan ... dev_off/host_on/writers/
// pins/active` when the plan is bound, and the failing condition inside the fatal message. A hit
// therefore names WHICH condition moved between those two lines.
//
// Cases:
//   shared-overlap    two owners' DemoteToHost runs both walk into the shared prefix (same
//                    LogicalKVPageHandle) -> the first work's prepare pins the pages and the
//                    second work's `source_pins != 0` validation throws
//   decode-vs-plan   one request decoding while another materializes a large root (production's
//                    shape: the plan is reserved in one worker iteration, prepared in a later one)
//   fill-big         saturate sequentially, then one large unique root (baseline shape)
//   twins            concurrent identical-prefix requests materializing at once
//   same-session-race  two continuations of the SAME conversation materialize concurrently
//                    (double-materialize / second bind while the first plans)
//   dense-drops      one session with a checkpoint per turn, then a root under saturation -
//                    pressure options that carry dropped_checkpoints
//   spill-race       two large roots at once while the pool is full (D2H in flight vs. plan)
//   host-full        host_state=2, so demote staging and the retire ladder run before a root
#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;

[[nodiscard]] ChatMessage msg(ChatRole role, std::string text) {
    ChatMessage message;
    message.role = role;
    MessagePart part;
    part.kind = MessagePartKind::Text;
    part.text = std::move(text);
    message.parts.push_back(std::move(part));
    return message;
}

[[nodiscard]] std::string repeat(const std::string& unit, int count) {
    std::string out;
    out.reserve(unit.size() * static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) { out += unit; }
    return out;
}

[[nodiscard]] RequestOptions small_output(std::uint32_t tokens) {
    RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    return options;
}

struct CaseResult {
    bool hit        = false;
    std::string tag;
    std::string detail;
};

// Sessions the SELECTED plans of this process destroyed, summed across every request the case ran.
// The retention contract - spill first, delete only what memory cannot absorb, delete the minimum -
// is what turns this number into a verdict, so a case whose pools provably have room asserts on it.
std::atomic<std::uint64_t> g_selected_evictions{0};

[[nodiscard]] CaseResult run(Engine& engine, std::vector<ChatMessage> messages,
                             std::uint32_t output_tokens, const char* tag) {
    CaseResult result;
    result.tag       = tag;
    const auto start = std::chrono::steady_clock::now();
    try {
        PromptInput input;
        input.messages = std::move(messages);
        PreparedPrompt prompt  = engine.prepare(std::move(input));
        GenerationResult gen   = engine.generate(std::move(prompt), small_output(output_tokens));
        const auto elapsed     = std::chrono::steady_clock::now() - start;
        const MaterializationDiagnostics& mat = gen.materialization;
        g_selected_evictions += mat.selected_owner_evictions;
        std::printf("  %-18s prompt=%-6u reuse=%-6u %.1fs stop=%d maxfb=%d degr=%-3u "
                    "evict=%-3u drops=%-3u replans=%u targets=%u closures=%u/%u\n",
                    tag, gen.prompt.prompt_tokens, gen.reused_prompt_tokens,
                    std::chrono::duration<double>(elapsed).count(),
                    static_cast<int>(mat.stop_reason), mat.selected_capped_fallback ? 1 : 0,
                    mat.selected_degradation_units, mat.selected_owner_evictions,
                    mat.selected_checkpoint_drops, mat.capacity_replans, mat.targets_evaluated,
                    mat.guided_closures_succeeded, mat.guided_closures_failed);
    } catch (const std::exception& error) {
        result.hit    = true;
        result.detail = error.what();
        std::printf("  %-18s HIT: %s\n", tag, error.what());
    } catch (...) {
        result.hit    = true;
        result.detail = "unknown exception";
        std::printf("  %-18s HIT: unknown exception\n", tag);
    }
    std::fflush(stdout);
    return result;
}

[[nodiscard]] CaseResult run_concurrent(Engine& engine,
                                        std::vector<std::vector<ChatMessage>> requests,
                                        std::uint32_t output_tokens, const char* tag) {
    std::vector<CaseResult> results(requests.size());
    std::vector<std::thread> threads;
    threads.reserve(requests.size());
    for (std::size_t index = 0; index < requests.size(); ++index) {
        threads.emplace_back([&, index] {
            std::string individual = std::string(tag) + "#" + std::to_string(index);
            results[index] = run(engine, std::move(requests[index]), output_tokens,
                                 individual.c_str());
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    CaseResult worst;
    for (const CaseResult& result : results) {
        if (result.hit && !worst.hit) { worst = result; }
    }
    return worst;
}

[[nodiscard]] std::vector<ChatMessage> conversation(const std::string& system, int turns,
                                                    int words) {
    std::vector<ChatMessage> out;
    out.push_back(msg(ChatRole::System, system));
    for (int turn = 0; turn < turns; ++turn) {
        out.push_back(msg(ChatRole::User, "turn " + std::to_string(turn) + " " +
                                              repeat("history token ", words)));
        out.push_back(msg(ChatRole::Assistant, "ack " + std::to_string(turn) + " " +
                                                   repeat("reply token ", words / 3)));
    }
    return out;
}

EngineOptions options_for(const std::filesystem::path& artifact, std::uint32_t host_state,
                          std::uint32_t kv_tokens, std::uint32_t fair_share_buckets,
                          std::uint32_t host_kv_mib, std::uint32_t max_private) {
    EngineOptions options;
    options.artifact_path       = artifact;
    options.max_context         = 16384;
    options.kv_capacity         = KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.max_concurrency     = 4;
    options.max_pending_requests = 16;
    options.context_cache.device_state_slots             = 0;
    options.context_cache.host_state_slots               = host_state;
    // Host KV must be large enough for a demote to FIT: with a128 MiB pool the projection reports
    // blocked_host=~660 MiB on every candidate, the planner never selects a non-evict KV option,
    // and the prepare path that throws is simply never entered (production runs32 GiB with
    // blocked_host=0). 4 GiB keeps the host axis open while staying far below production's pin,
    // but it is exactly the size of the device KV pool, so a case that fills the device also fills
    // the host arena; `fresh-dev-roomy` raises it so the host axis is never the binding one.
    options.context_cache.host_kv_capacity_bytes         = std::uint64_t{host_kv_mib} << 20U;
    options.context_cache.max_private_continuations      = max_private;
    options.context_cache.max_shared_prefixes            = 4;
    options.context_cache.max_long_anchors_per_continuation = 16;
    // 0 disables fair-share protection, which removes protected sessions from the pressure-owner
    // domain entirely (they can be neither evicted nor spilled). Used to A/B whether that
    // exclusion is what starves the spill path.
    options.context_cache.fair_share_buckets = fair_share_buckets;
    return options;
}

const std::string kShared = repeat("myai agent. shared stable prefix token ", 1000);  // ~7K
const std::string kRoot   = "standalone root. " + repeat("divergent token ", 3600);

// Fills the micro pool with retained sessions; returns nothing, hits are reported by the caller.
// Six sessions against private=8 leaves the catalog dense enough that the root has to buy its
// publication slot from a cache, which is the production shape (host.state320/320, catalog=512).
void saturate(Engine& engine, const char* tag) {
    for (int seed = 1; seed <= 6; ++seed) {
        (void)run(engine, conversation(kShared + " session " + std::to_string(seed), 5, 400), 8,
                  tag);
    }
}

// Production's actual shape (2026-09-25 09:01:32): one request decoding while another one
// materializes a large root. The plan is reserved in one worker iteration and prepared in a later
// one, so any unit that runs in between can move a page the plan already selected.
CaseResult case_decode_vs_plan(Engine& engine) {
    saturate(engine, "fill");
    CaseResult decoded;
    std::thread decoder([&] {
        decoded = run(engine, conversation(kShared + " decoding session", 3, 300), 1200,
                      "DECODER");
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    CaseResult planned =
        run(engine, {msg(ChatRole::System, kRoot), msg(ChatRole::User, "begin")}, 32, "BIG-ROOT");
    decoder.join();
    return planned.hit ? planned : decoded;
}

// Duplicate physical target across two pressure works. Shared-prefix pages are the SAME
// LogicalKVPageHandle in every owner that shares the prefix (logical_page_matches_prefix compares
// handles, not offsets), so two owners whose DemoteToHost runs both walk back into the shared
// region select the same pages. evaluate_pressure_target accepts that as one joint settle (the
//09-24 fix), but prepare_pressure_work still executes each work: HostKVExtentStore::prepare pins
// the first work's pages (host_kv_extent_store.h:129), and the second work's validation
// `source_pins != 0` throws - latching the engine. Forcing the overlap needs a deficit larger
// than ONE owner's private tail, so the walk-down reaches index 0 in more than one owner.
CaseResult case_shared_overlap(Engine& engine) {
    // NOTE: repeat(unit, n) repeats a multi-word unit - budget in words (n * unit_words * ~1.3
    // tokens), not in units. The first version overshot max_context5x and every request just
    // failed with `prepared prompt exceeds Engine max_context`.
    const std::string prefix  = repeat("shared stable system token ", 700);   // ~4.5K tokens
    const std::string history = repeat("divergent session history token ", 900);  // ~5.8K tokens
    for (int seed = 1; seed <= 6; ++seed) {
        std::vector<ChatMessage> session;
        session.push_back(msg(ChatRole::System, prefix));
        session.push_back(msg(ChatRole::User, "session " + std::to_string(seed) + " " + history));
        (void)run(engine, std::move(session), 8, "fill");
    }
    // A root larger than any single owner's private tail: every owner has to give pages back, so
    // several owners' runs extend past their own tail into the shared prefix.
    return run(engine,
               {msg(ChatRole::System, repeat("divergent root token ", 2600)),
                msg(ChatRole::User, "begin")},
               32, "BIG-ROOT");
}

// Deterministic duplicate target: CLONE sessions (identical system AND identical user text) map
// the same logical pages, so two owners' DemoteToHost runs select the IDENTICAL page range -
// exactly what the observed fatal needed (`begin=0 count=13` printed twice before it fired).
// Divergent tails make selections tile instead (0..125 / 126..128 / 129..210), which is why the
// shared-prefix-only shapes never hit.
CaseResult case_clone_overlap(Engine& engine) {
    std::vector<ChatMessage> clone;
    clone.push_back(msg(ChatRole::System, repeat("clone shared system token ", 700)));
    clone.push_back(msg(ChatRole::User, repeat("clone identical body token ", 900)));
    for (int seed = 0; seed < 6; ++seed) {
        std::vector<ChatMessage> copy = clone;
        (void)run(engine, std::move(copy), 8, "clone");
    }
    return run(engine,
               {msg(ChatRole::System, repeat("divergent root token ", 2600)),
                msg(ChatRole::User, "begin")},
               32, "BIG-ROOT");
}

// Reverse-engineered from the observed fatal: the two colliding works both printed
// `begin=0 count=13` - owners whose ENTIRE mapped range is ~13 pages (~830 tokens). Small owners
// all select their whole range under a large deficit, so any two of them are byte-identical
// selections over shared prefix pages. Clone sessions fail to model this because the engine
// collapses them into cache reuse (reuse=6448/6453) instead of creating independent owners.
CaseResult case_tiny_overlap(Engine& engine) {
    for (int seed = 1; seed <= 8; ++seed) {
        std::vector<ChatMessage> session;
        session.push_back(msg(ChatRole::System, repeat("tiny shared system token ", 100)));
        session.push_back(msg(ChatRole::User,
                              "owner " + std::to_string(seed) + " " +
                                  repeat("tiny body token ", 30)));
        (void)run(engine, std::move(session), 8, "tiny");
    }
    return run(engine,
               {msg(ChatRole::System, repeat("divergent root token ", 2600)),
                msg(ChatRole::User, "begin")},
               32, "BIG-ROOT");
}

// THE FAULT SHAPE (from production 2026-09-25): several brand-new, zero-reuse large prompts
// arrive in parallel while the old sessions still hold cached state. Each fresh prompt needs its
// whole KV on Device at once (cache 0% - nothing to borrow) and its own pages are being written,
// so they cannot be spilled; the only spillable pages left are the old sessions' leftovers. When
// the planner cannot cover the gap by spilling, it evicts whole old sessions - and their Host
// cache goes with them, which nobody then uses (production:162 slots +3.68 GiB freed,0 used).
//
// A pass means: no planner-driven eviction while Host KV still had room.
CaseResult case_parallel_fresh_roots(Engine& engine) {
    saturate(engine, "fill");  // old sessions now hold Device + Host cache
    std::vector<std::vector<ChatMessage>> requests;
    for (int index = 0; index < 3; ++index) {
        // Unique body per request: zero reuse, exactly like production's req#292.
        requests.push_back({msg(ChatRole::System, "fresh root " + std::to_string(index) + " " +
                                                      repeat("divergent token ", 2600)),
                            msg(ChatRole::User, "begin " + std::to_string(index))});
    }
    return run_concurrent(engine, std::move(requests), 64, "FRESH");
}

CaseResult case_fill_big(Engine& engine) {
    saturate(engine, "fill");
    return run(engine, {msg(ChatRole::System, kRoot), msg(ChatRole::User, "begin")}, 32,
               "BIG-ROOT");
}

CaseResult case_twins(Engine& engine) {
    saturate(engine, "fill");
    std::vector<std::vector<ChatMessage>> requests;
    for (int twin = 0; twin < 4; ++twin) {
        requests.push_back({msg(ChatRole::System,
                                std::string("shared twin brief. ") + repeat("twin payload ", 1200)),
                            msg(ChatRole::User, "begin twin " + std::to_string(twin))});
    }
    return run_concurrent(engine, std::move(requests), 16, "twin");
}

CaseResult case_same_session_race(Engine& engine) {
    // One conversation, two continuations submitted together: both materialize the SAME source,
    // so the second binds/activates its address space while the first is still planning.
    std::vector<ChatMessage> base = conversation(kShared + " race session", 6, 300);
    std::vector<std::vector<ChatMessage>> requests;
    for (int branch = 0; branch < 2; ++branch) {
        std::vector<ChatMessage> copy = base;
        copy.push_back(msg(ChatRole::User, "branch " + std::to_string(branch) + " " +
                                               repeat("tail token ", 900)));
        requests.push_back(std::move(copy));
    }
    saturate(engine, "fill");
    return run_concurrent(engine, std::move(requests), 24, "same-session");
}

CaseResult case_dense_drops(Engine& engine) {
    // One checkpoint per turn: the pressure option for this session can carry dropped_checkpoints
    // while it also carries KV actions, which is the prepare/plan divergence candidate.
    // Seven sessions is the count the observed fatal ran with (the original loop's turn=0 request
    // was rejected for having no user message, so only turns1..7 ever created owners); at eight
    // the pool shape changes and the duplicate target stops appearing.
    for (int turn = 1; turn <= 7; ++turn) {
        std::vector<ChatMessage> history = conversation(kShared + " dense session", turn, 250);
        (void)run(engine, std::move(history), 8, "dense-turn");
    }
    saturate(engine, "fill");
    return run(engine, {msg(ChatRole::System, kRoot), msg(ChatRole::User, "begin")}, 32,
               "BIG-ROOT");
}

CaseResult case_spill_race(Engine& engine) {
    saturate(engine, "fill");
    std::vector<std::vector<ChatMessage>> requests;
    for (int copy = 0; copy < 2; ++copy) {
        requests.push_back({msg(ChatRole::System, kRoot + " variant " + std::to_string(copy)),
                            msg(ChatRole::User, "begin")});
    }
    return run_concurrent(engine, std::move(requests), 64, "spill");
}

CaseResult case_host_full(Engine& engine) {
    for (int seed = 1; seed <= 4; ++seed) {
        (void)run(engine, conversation(kShared + " hostfull " + std::to_string(seed), 4, 300), 8,
                  "fill");
    }
    return run(engine, {msg(ChatRole::System, kRoot), msg(ChatRole::User, "begin")}, 32,
               "BIG-ROOT");
}

struct CaseEntry {
    const char* name;
    std::uint32_t host_state;
    // KV capacity in tokens. The overlap cases use a tight budget (16384) so one root plus the
    // fill actually overflows the device pool; the default 32768 leaves the other cases the
    // headroom they were sized against.
    std::uint32_t kv_tokens;
    // Fair-share buckets: the number of recently active sessions protected from the pressure
    // domain (0 = no protection, every retained session may be spilled).
    std::uint32_t fair_share_buckets;
    // Host KV arena in MiB. 4096 equals the tight device pool, so the host axis saturates as soon
    // as the device does; a roomy value makes the host axis provably non-binding, which is the
    // condition under which deleting cached sessions is never justified.
    std::uint32_t host_kv_mib;
    // Private catalog capacity (== Engine `max_private_continuations`). Production runs512 against
    //320 host slots, so the catalog can only fill when the host pool already has; the micro default
    // of8 against32 host slots inverts that and makes a publication slot, not memory, the binding
    // resource. `fresh-prod-shape` restores production's ratio.
    std::uint32_t max_private;
    // Upper bound on the sessions the selected plans of this case may destroy (-1 = unconstrained).
    // Only a case whose memory and catalog axes are all provably roomy can state it: there the
    // whole device deficit is payable by spilling, so any eviction is the regression under test.
    std::int64_t eviction_budget;
    CaseResult (*body)(Engine&);
};

const CaseEntry kCases[] = {
    {"parallel-fresh", 4, 32768, 8, 4096, 8, -1, case_parallel_fresh_roots},
    // Same shape with the Host slot pool deliberately ROOMY (32) and Device KV tight (16384):
    // now the only scarce resource is Device KV, which is production's situation. If the planner
    // still evicts whole sessions here, the fault is reproduced - Host cache dies while Host had
    // room for it. The 4 GiB host arena equals the device pool, so the host axis does saturate and
    // an eviction there is policy-legal; `fresh-dev-roomy` removes that ambiguity.
    {"fresh-dev-tight", 32, 16384, 8, 4096, 8, -1, case_parallel_fresh_roots},
    // The same body with BOTH memory axes provably roomy (host state32, host KV16 GiB against a
    //4 GiB device pool). Nothing but Device KV is scarce, so spilling is always sufficient in
    // principle; any planner eviction here is the defect under test.
    {"fresh-dev-roomy", 32, 16384, 8, 16384, 8, -1, case_parallel_fresh_roots},
    // production's ratios: host KV16 GiB, host state32, catalog51 (> host state, as512 > 320).
    // No axis except Device KV can be the reason to delete, so a zero-eviction spill plan exists
    // and the case asserts on it: this is the regression guard for "spill first, delete nothing
    // while memory has room".
    {"fresh-prod-shape", 32, 16384, 8, 16384, 51, 0, case_parallel_fresh_roots},
    {"tiny-overlap", 4, 16384, 8, 4096, 8, -1, case_tiny_overlap},
    {"clone-overlap", 4, 16384, 8, 4096, 8, -1, case_clone_overlap},
    {"shared-overlap", 4, 16384, 8, 4096, 8, -1, case_shared_overlap},
    {"decode-vs-plan", 4, 32768, 8, 4096, 8, -1, case_decode_vs_plan},
    {"fill-big", 4, 32768, 8, 4096, 8, -1, case_fill_big},
    {"twins", 4, 32768, 8, 4096, 8, -1, case_twins},
    {"same-session-race", 4, 32768, 8, 4096, 8, -1, case_same_session_race},
    {"dense-drops", 4, 32768, 8, 4096, 8, -1, case_dense_drops},
    // Same body with fair-share protection off: if the spill then suffices, the protected-owner
    // exclusion (not the spill rule) was starving the plan.
    {"dense-drops-np", 4, 32768, 0, 4096, 8, -1, case_dense_drops},
    {"spill-race", 4, 32768, 8, 4096, 8, -1, case_spill_race},
    {"host-full", 2, 32768, 8, 4096, 8, -1, case_host_full},
};

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path artifact = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path{};
    if (artifact.empty()) {
        if (const char* env = std::getenv("NINFER_TEST_ARTIFACT"); env != nullptr && *env != '\0') {
            artifact = env;
        }
    }
    if (artifact.empty() || !std::filesystem::exists(artifact)) {
        std::fprintf(stderr,
                     "usage: ninfer_pressure_prepare_cases <artifact.ninfer> <case>|list\n"
                     "       (or set NINFER_TEST_ARTIFACT); artifact=%s\n",
                     artifact.string().c_str());
        return 2;
    }
    const std::string selection = argc > 2 ? argv[2] : "list";
    if (selection == "list") {
        for (const CaseEntry& entry : kCases) { std::printf("%s\n", entry.name); }
        return 0;
    }

    bool ran = false;
    for (const CaseEntry& entry : kCases) {
        if (selection != "all" && selection != entry.name) { continue; }
        ran = true;
        std::printf("== case %s (host_state=%u) ==\n", entry.name, entry.host_state);
        std::fflush(stdout);
        Engine engine(options_for(artifact, entry.host_state, entry.kv_tokens,
                                entry.fair_share_buckets, entry.host_kv_mib, entry.max_private));
        g_selected_evictions.store(0, std::memory_order_relaxed);
        CaseResult result = entry.body(engine);
        if (result.hit) {
            std::printf("CASE %s HIT: %s\n", entry.name, result.detail.c_str());
            std::fflush(stdout);
            return 1;
        }
        const std::uint64_t evicted =
            g_selected_evictions.load(std::memory_order_relaxed);
        if (entry.eviction_budget >= 0 &&
            evicted > static_cast<std::uint64_t>(entry.eviction_budget)) {
            std::printf("CASE %s HIT: selected plans destroyed %llu session(s), budget %lld "
                        "(every memory and catalog axis in this case has room, so the whole "
                        "device deficit is payable by spilling)\n",
                        entry.name, static_cast<unsigned long long>(evicted),
                        static_cast<long long>(entry.eviction_budget));
            std::fflush(stdout);
            return 1;
        }
        std::printf("CASE %s clean (selected evictions=%llu)\n", entry.name,
                    static_cast<unsigned long long>(evicted));
        std::fflush(stdout);
    }
    if (!ran) {
        std::fprintf(stderr, "unknown case: %s (use `list`)\n", selection.c_str());
        return 2;
    }
    return 0;
}

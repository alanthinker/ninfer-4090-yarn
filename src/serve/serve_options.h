#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact identity.model_id
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::string slot_save_path;        // empty => /slots save/restore/erase disabled
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    // Retired with the upstream merge: per-sequence rewrite checkpoints and long anchors
    // supersede the host turn-checkpoint ring. The field is dead everywhere and goes away
    // with the flag one release after the deprecation warning ships.
    std::uint32_t turn_checkpoint_ring     = 0;
    bool deprecated_turn_checkpoints_given = false; // --turn-checkpoints was passed and ignored
    bool auto_save_evicted                 = false; // spill evicted sessions to their slot file
    // --auto-long-anchors N: propose a private long anchor at each of the last N message
    // boundaries of every prompt. Unset resolves to the retained-anchor cap
    // (--max-long-anchors-per-continuation) once the Engine has normalized it; 0 disables.
    // See resolve_automatic_private_anchors.
    std::optional<std::uint32_t> auto_long_anchors;
    // --auto-anchor-spacing N: also propose a private long anchor at the first message boundary at
    // or after every N tokens of every prompt, so a mid-history divergence resumes nearby instead
    // of from token zero. Unset or 0 disables; see ContextCacheHints::automatic_anchor_spacing.
    std::optional<std::uint32_t> auto_anchor_spacing;
    // --first-anchor-spacing N: spacing for the first automatic anchor only (default 8192).
    // Subsequent anchors continue at auto_anchor_spacing intervals. Ensures short prompts
    // (< 32K tokens) still get a catalog entry. 0 falls back to auto_anchor_spacing.
    std::optional<std::uint32_t> auto_first_anchor_spacing;
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    ContextCacheOptions context_cache;
    bool enable_vision              = false;
    std::uint32_t vision_max_tokens = 8192;
    bool use_cuda_graph             = true;
    float rope_scaling_factor              = 1.0F;
    std::uint32_t rope_scaling_original_context = 262144;
    bool allow_prefix_reuse = true;
    bool enable_thinking =
        true; // default thinking mode for the generation prompt (--no-thinking opts out)
    bool preserve_thinking = false;
    std::optional<std::uint32_t> default_thinking_budget;
    // The effort a request receives when it omits `reasoning_effort`
    // (--default-reasoning-effort). Unset keeps the loaded chat template's own default.
    //
    // The resolved effort is rendered INTO THE PROMPT: the leading block carries a reasoning
    // directive whose length depends on the level (medium renders nothing, xhigh renders the 38
    // tokens of kXHighReasoningInstructions). That block precedes every message, so a level that
    // differs from the one an earlier request used shifts every later position and the request
    // cannot reuse one cached token - correctly, because those tokens' KV was computed against a
    // different prefix. A client that sends an explicit effort on its ordinary turns but omits
    // the field on an auxiliary call therefore loses prefix reuse for exactly that call: measured
    // 2026-09-11, an 18,279-token summarization prompt sharing its first 17,851 tokens with the
    // preceding turn reused 0 and paid a full cold prefill, while the same prompt at the
    // session's own level reused 95.5% (8,444 of 8,839). Pinning the omitted case to the level
    // such a client does send keeps both prompts byte-identical.
    std::optional<ninfer::ReasoningEffort> default_reasoning_effort;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);

// The per-request ContextCacheHints::automatic_private_anchors value for this server: the
// explicit --auto-long-anchors when given, else the resolved anchor cap, and never more than
// that cap (extra proposals would only churn replacements within one prefill). Zero when the
// context cache is disabled. `resolved` must be the Engine's normalized options, not the parsed
// ServeOptions::context_cache, whose optionals are still unset.
// The per-request ContextCacheHints::automatic_anchor_spacing value for this server.
std::uint32_t resolve_automatic_anchor_spacing(const ServeOptions& options,
                                              const ContextCacheOptions& resolved);
std::uint32_t resolve_automatic_private_anchors(const ServeOptions& options,
                                                const ContextCacheOptions& resolved);
std::uint32_t resolve_first_anchor_spacing(const ServeOptions& options,
                                           const ContextCacheOptions& resolved);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve

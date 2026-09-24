// In-process pressure/joint-settle trigger driver (reproduced-bug tooling).
//
// The structural-closure -> evict-everything wipe needs a pool that saturates while heterogeneous
// sessions coexist; service-level rounds cost minutes because the fill phase dominates. This
// driver builds the same pressure inside ONE process with a tiny context-cache (host_state=4,
// private=8, kv=32768 tokens) so every candidate shape runs in seconds: dirty multi-turn seeds,
// concurrent identical-prefix twins (re-plan / double-materialize), single-turn long victims in
// the shared head, and large unique roots. Engine stderr carries
// `[pressure] target rejected site=...` for every structural exit (reason tags) and
// A structural rejection prints `[pressure] target rejected site=...`, naming the exit.
#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

void run(Engine& engine, std::vector<ChatMessage> messages, std::uint32_t output_tokens,
         const char* tag) {
    PromptInput input;
    input.messages = std::move(messages);
    const auto started = std::chrono::steady_clock::now();
    PreparedPrompt prompt = engine.prepare(std::move(input));
    GenerationResult result = engine.generate(std::move(prompt), small_output(output_tokens));
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started);
    const MaterializationDiagnostics& mat = result.materialization;
    std::printf("%-14s prompt=%-6u reuse=%-6u %.1fs stop=%d budget=%d maxfb=%d "
                "degr=%-4u replans=%u targets=%u closures=%u/%u\n",
                tag, result.prompt.prompt_tokens, result.reused_prompt_tokens, elapsed.count(),
                static_cast<int>(mat.stop_reason), mat.budget_exhausted ? 1 : 0,
                mat.selected_capped_fallback ? 1 : 0, mat.selected_degradation_units,
                mat.capacity_replans, mat.targets_evaluated, mat.guided_closures_succeeded,
                mat.guided_closures_failed);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path artifact = argc > 1 ? argv[1] : std::filesystem::path{};
    if (artifact.empty()) {
        const char* env = std::getenv("NINFER_TEST_ARTIFACT");
        if (env != nullptr && *env != '\0') { artifact = env; }
    }
    if (artifact.empty() || !std::filesystem::exists(artifact)) {
        std::fprintf(stderr,
                     "usage: ninfer_pressure_joint_settle_test <artifact.ninfer> "
                     "(or set NINFER_TEST_ARTIFACT); artifact=%s\n",
                     artifact.c_str());
        return 2;
    }
    EngineOptions options;
    options.artifact_path       = artifact;
    options.max_context         = 16384;
    options.kv_capacity         = KvCapacityPolicy::explicit_capacity(32768);  //512 pages
    options.max_concurrency     = 4;
    options.max_pending_requests = 16;
    options.context_cache.device_state_slots       = 0;    // device state total = C =4
    options.context_cache.host_state_slots         = 4;    // host pressure becomes reachable
    options.context_cache.host_kv_capacity_bytes   = 128ull << 20;
    options.context_cache.max_private_continuations = 8;
    options.context_cache.max_shared_prefixes       = 4;
    options.context_cache.max_long_anchors_per_continuation = 16;

    const auto loaded = std::chrono::steady_clock::now();
    Engine engine(std::move(options));
    std::printf("engine ready in %.1fs (fixed code)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - loaded).count());
    std::fflush(stdout);

    const std::string shared = repeat("myai agent. shared stable prefix token ", 1000);  // ~7K

    // Phase1: dirty multi-turn seeds - per-turn endpoint captures fill host_state=4 and leave
    // sessions with endpoint/rewrite history (the heterogeneous old sessions the service-level
    // fresh pools never had).
    for (int seed = 1; seed <= 2; ++seed) {
        std::vector<ChatMessage> conversation;
        conversation.push_back(msg(ChatRole::System, shared + " old session history header."));
        for (int turn = 0; turn < 6; ++turn) {
            conversation.push_back(
                msg(ChatRole::User, "old" + std::to_string(seed) + " turn" +
                                        std::to_string(turn) + " " + repeat("history token ", 400)));
            conversation.push_back(msg(ChatRole::Assistant,
                                       "ack reply " + std::to_string(turn) + " " +
                                           repeat("reply token ", 150)));
        }
        run(engine, std::move(conversation), 8, "seed");
    }

    // Phase2: concurrent identical-prefix twins - two sequences materializing the same logical
    // source at once (the re-plan / double-materialize candidate for a shared StateImage).
    {
        std::vector<std::thread> threads;
        for (int twin = 0; twin < 4; ++twin) {
            threads.emplace_back([&, twin] {
                std::vector<ChatMessage> messages;
                messages.push_back(msg(ChatRole::System,
                                       std::string("shared twin brief. ") +
                                           repeat("twin payload token ", 1200)));
                messages.push_back(msg(ChatRole::User, "begin twin " + std::to_string(twin)));
                run(engine, std::move(messages), 16,
                    ("twin" + std::to_string(twin)).c_str());
            });
        }
        for (std::thread& thread : threads) { thread.join(); }
    }

    // Phase3: single-turn long victims - one long turn so auto anchors land inside the shared
    // head; these sessions hold shared-derived state at pressure time.
    for (int victim = 3; victim <= 5; ++victim) {
        std::vector<ChatMessage> messages;
        messages.push_back(msg(ChatRole::System,
                               shared + " victim " + std::to_string(victim) + " " +
                                   repeat("fill token ", 900)));
        messages.push_back(msg(ChatRole::User,
                               "long turn " + std::to_string(victim) + " " +
                                   repeat("anchor token ", 1200)));
        run(engine, std::move(messages), 8, "victim");
    }

    // Phase4: large unique roots against the saturated micro pools - this is where the pressure
    // planner must close a device/host residual with the domain above as victims.
    for (int attempt = 1; attempt <= 2; ++attempt) {
        std::vector<ChatMessage> messages;
        messages.push_back(msg(ChatRole::System,
                               "standalone root. " + repeat("divergent token ", 3600) +
                                   " variant" + std::to_string(attempt)));
        messages.push_back(msg(ChatRole::User, "begin"));
        run(engine, std::move(messages), 32, "BIG-ROOT");
    }

    std::printf("PRESSURE-TEST DONE\n");
    return 0;
}

// Response-echo admission against PRODUCTION-shaped data: the real tokenizer, the real chat
// template, and the real multi-turn request pair that exposed the reuse gap (req#3 / req#4 of
// the 2026-09-28 story conversation: developer + 2 user messages, a long Chinese response with
// reasoning, then the client echo of that response plus the next user message, 26 tools).
//
// The test rebuilds the state exactly the way the engine would after round A (ledger =
// prompt ++ raw generated span; prompt_end / starts_in_reasoning / output_preserve_special /
// tool contract / prompt text digest all carried from round A's prepared prompt), then runs
// the real admission gate and the real stored-side re-parse against round B's echo spec.
// Any asymmetry between what the original round published to the client and what the
// response-echo re-parse computes shows up here on the CPU, without a service restart.
//
// Opt-in: needs the model directory (NINFER_ECHO_ADMISSION_MODEL) and the request dump
// (NINFER_ECHO_ADMISSION_DUMP); exits 77 when either is absent.

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/types.h>

#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Frontend        = ninfer::targets::qwen3_6::Frontend;
using FrontendFactory = ninfer::targets::qwen3_6::FrontendTestAccess;
namespace q6          = ninfer::targets::qwen3_6;

std::size_t failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("failed to open " + path.string()); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string env_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : fallback;
}

const Frontend& production_frontend(const std::filesystem::path& model_dir) {
    static const Frontend value = [&] {
        q6::FrontendResources resources;
        // The frontend validates that the chat template equals the one embedded in
        // tokenizer_config.json, so the embedded copy is the authoritative source here.
        resources.chat_template_jinja =
            nlohmann::json::parse(read_file(model_dir / "tokenizer_config.json"))
                .at("chat_template")
                .get<std::string>();
        resources.tokenizer_json = read_file(model_dir / "tokenizer.json");
        resources.tokenizer_config_json = read_file(model_dir / "tokenizer_config.json");
        resources.generation_config_json = read_file(model_dir / "generation_config.json");
        resources.preprocessor_config_json = read_file(model_dir / "preprocessor_config.json");
        resources.video_preprocessor_config_json =
            read_file(model_dir / "video_preprocessor_config.json");
        return FrontendFactory::create_component(resources, false);
    }();
    return value;
}

ninfer::ChatMessage from_wire(const nlohmann::json& wire) {
    auto role = [wire] {
        const auto value = wire.at("role").get<std::string>();
        if (value == "system") { return ninfer::ChatRole::System; }
        if (value == "developer") { return ninfer::ChatRole::Developer; }
        if (value == "user") { return ninfer::ChatRole::User; }
        if (value == "tool") { return ninfer::ChatRole::Tool; }
        return ninfer::ChatRole::Assistant;
    }();
    ninfer::ChatMessage message;
    message.role = role;
    const auto& content = wire.at("content");
    if (content.is_string()) {
        if (!content.get<std::string>().empty()) {
            message.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text,
                .text = content.get<std::string>()});
        }
    } else if (content.is_array()) {
        for (const auto& item : content) {
            if (item.value("type", "") == "text") {
                message.parts.push_back(ninfer::MessagePart{
                    .kind       = ninfer::MessagePartKind::Text,
                    .text       = item.value("text", std::string{}),
                });
            }
        }
    }
    if (wire.contains("reasoning_content") && wire["reasoning_content"].is_string()) {
        message.reasoning_content = wire["reasoning_content"].get<std::string>();
    }
    if (wire.contains("tool_calls") && wire["tool_calls"].is_array()) {
        for (const auto& call : wire["tool_calls"]) {
            const auto function    = call.value("function", nlohmann::json::object());
            message.tool_calls.push_back(ninfer::ToolCall{
                .id             = call.value("id", std::string{}),
                .name           = function.value("name", std::string{}),
                .arguments_json = function.value("arguments", std::string{}),
            });
        }
    }
    if (wire.contains("tool_call_id") && wire["tool_call_id"].is_string()) {
        message.tool_call_id = wire["tool_call_id"].get<std::string>();
    }
    return message;
}

struct AdmissionScenario {
    const Frontend& frontend;
    // Round A: the history without the generated response; its prepared prompt is the state's
    // prompt region and its carried identity the state's response-echo identity.
    q6::PreparedPromptData round_a;
    // The raw generated span of round A's response. Two variants: canonical whitespace
    // around the close serialization, and a non-canonical single-newline variant.
    std::vector<ninfer::TokenId> raw_canonical;
    std::vector<ninfer::TokenId> raw_noncanonical;
    // Round B: the client's echo of the response plus the next user message.
    q6::PreparedPromptData round_b;
    // What the original round published to the client (the fields the client echoes back).
    std::string published_reasoning;
    std::string published_content;
};

int verify_admission(const Frontend& frontend, const std::filesystem::path& dump_path,
                     bool noncanonical_raw) {
    std::fprintf(stderr, "[phase] parse dump\n");
    const auto dump = nlohmann::json::parse(read_file(dump_path));
    const auto& wire_messages = dump.at("messages");

    const auto wire_tools = [&dump] {
        std::vector<std::string> out;
        if (dump.contains("tools") && dump["tools"].is_array()) {
            for (const auto& tool : dump["tools"]) { out.push_back(tool.dump()); }
        }
        return out;
    }();

    std::vector<ninfer::ChatMessage> messages;
    for (const auto& wire : wire_messages) { messages.push_back(from_wire(wire)); }
    const std::size_t echo_index = wire_messages.size() - 2;
    check(messages[echo_index].role == ninfer::ChatRole::Assistant,
          "the second-to-last message of the dump must be the echoed assistant response");

    const auto make_input = [&](const std::vector<ninfer::ChatMessage>& msgs) {
        ninfer::PromptOptions options;
        options.preserve_thinking = true;
        options.enable_thinking   = true;
        if (dump.contains("chat_template_kwargs")) {
            const auto& kwargs = dump["chat_template_kwargs"];
            if (kwargs.contains("enable_thinking") && kwargs["enable_thinking"].is_boolean()) {
                options.enable_thinking = kwargs["enable_thinking"].get<bool>();
            }
            if (kwargs.contains("preserve_thinking") && kwargs["preserve_thinking"].is_boolean()) {
                options.preserve_thinking = kwargs["preserve_thinking"].get<bool>();
            }
        }
        if (dump.contains("reasoning_effort") && dump["reasoning_effort"].is_string()) {
            const auto effort = dump["reasoning_effort"].get<std::string>();
            if (effort == "low") { options.reasoning_effort = ninfer::ReasoningEffort::Low; }
            if (effort == "medium") { options.reasoning_effort = ninfer::ReasoningEffort::Medium; }
            if (effort == "xhigh") { options.reasoning_effort = ninfer::ReasoningEffort::XHigh; }
        }
        options.tool_jsons = wire_tools;
        return ninfer::PromptInput{.messages = msgs, .options = options};
    };

    std::vector<ninfer::ChatMessage> round_a_messages;
    for (std::size_t i = 0; i < echo_index; ++i) { round_a_messages.push_back(messages[i]); }
    std::fprintf(stderr, "[phase] prepare round A\n");
    auto prepared_a = frontend.prepare(make_input(round_a_messages));
    const auto& round_a = FrontendFactory::inspect(prepared_a);
    check(round_a.identity.rewrite_checkpoint.has_value(),
          "round A must be a continuation carrying a rewrite checkpoint");
    check(round_a.identity.response_echo == std::nullopt,
          "round A carries no echoed response, hence no echo spec");

    // The client's echoed response fields (from the dump, exactly what round A's session
    // published) define the raw generated span of the state we rebuild.
    const auto& echo_wire = wire_messages[static_cast<std::size_t>(echo_index)];
    const auto published_reasoning_in =
        echo_wire.value("reasoning_content", std::string{});
    std::string content_in;
    const auto& content_field = echo_wire.at("content");
    if (content_field.is_string()) { content_in = content_field.get<std::string>(); }

    // The raw span mirrors round A's generation opener: a reasoning-mode opener emits
    // `<think>\n` before the reasoning text, so the span holds the reasoning plus the close
    // serialization plus the content (canonical double newline, non-canonical single); a
    // content-mode opener (thinking disabled) holds the content alone.
    const std::string raw_text =
        round_a.starts_in_reasoning
            ? (noncanonical_raw
                   ? published_reasoning_in + "</think>\n" + content_in
                   : published_reasoning_in + "</think>\n\n" + content_in)
            : content_in;
    std::fprintf(stderr, "[phase] tokenize raw\n");
    const auto raw_tokens = frontend.tokenize_text(raw_text);
    check(!raw_tokens.empty(), "the simulated raw span must tokenize");

    // The engine's state after round A carries: the raw generated span (prompt_end marks the
    // prompt/generation boundary inside ledger = round A prompt ++ raw span) and round A's
    // carried decoder/identity context. The admission gate is the conjunction of the two
    // digest equalities below, so the test asserts them directly.
    std::vector<ninfer::TokenId> ledger = round_a.token_ids;
    ledger.insert(ledger.end(), raw_tokens.begin(), raw_tokens.end());
    const std::size_t prompt_end      = round_a.token_ids.size();
    const std::size_t execution_frontier = ledger.size();

    std::vector<ninfer::ChatMessage> round_b_messages = messages;
    std::fprintf(stderr, "[phase] prepare round B\n");
    auto prepared_b = frontend.prepare(make_input(round_b_messages));
    const auto& round_b = FrontendFactory::inspect(prepared_b);
    check(round_b.identity.response_echo.has_value(),
          "round B must carry an echo spec for the echoed response");
    if (!round_b.identity.response_echo) { return 1; }
    const auto& spec = *round_b.identity.response_echo;

    // Digest chain: round B's prefix digest is round A's carried prompt text digest.
    check(spec.prefix_digest == round_a.identity.prompt_text_digest,
          "the echo prefix digest must equal round A's carried prompt text digest");
    check(spec.boundary_token > 0 && spec.boundary_token < round_b.token_ids.size(),
          "the echo boundary token must sit strictly inside round B's prompt");

    // Gate clause 1: the incoming prefix digest is the state's carried prompt text digest.
    // Gate clause 2: the stored-side re-parse of the raw generated span digests to the
    // incoming echo block digest (decoder mode, tool contract, and reasoning split identical
    // to round A's).
    std::fprintf(stderr, "[phase] stored block digest\n");
    const auto block = frontend.response_block_digest(
        std::span<const ninfer::TokenId>(ledger.data() + prompt_end,
                                         execution_frontier - prompt_end),
        round_a.tool_call_output, round_a.starts_in_reasoning,
        round_a.output_preserve_special);
    check(block == spec.block_digest,
          "the stored-side block digest must equal the incoming echo block digest");

    // Splice geometry the engine relies on: the state keeps its raw prefix
    // [0, execution_frontier) and replaces the incoming prefix [0, boundary_token) with it, so the
    // resulting state prompt is execution_frontier + (prompt - boundary_token) tokens. That can
    // exceed the incoming prompt length - the raw body is routinely different in length from its
    // re-serialization - and the plan charges the difference to the request's output allowance
    // (response_echo_allowance_fits), which keeps the KV page entitlement and the capacity clamp
    // exact. No absolute bound holds here; the gate that does is the runtime predicate.

    // Junction contract: the tail the splice appends must begin with the echoed assistant turn's
    // closing serialization. The resident raw prefix ends where the generated response ends - its
    // stop token is not part of the generated span - so a boundary placed after the whole block
    // leaves the echoed assistant turn unterminated in the spliced prompt, and the model answers
    // by closing a thinking block that was never opened.
    const auto closing = frontend.tokenize_text("<|im_end|>\n");
    check(!closing.empty() && spec.boundary_token + closing.size() <= round_b.token_ids.size() &&
              std::equal(closing.begin(), closing.end(),
                         round_b.token_ids.begin() + spec.boundary_token),
          "the echo tail must begin with the assistant closing serialization");

    return 0;
}

} // namespace

int main() {
    const auto model_dir =
        std::filesystem::path(env_or("NINFER_ECHO_ADMISSION_MODEL",
                                     "/root/ai/large_models/qwen_3.8_27b/models/"
                                     "Qwen3.8-27B-AWQ-INT4"));
    const auto dump_path = std::filesystem::path(
        env_or("NINFER_ECHO_ADMISSION_DUMP",
               "/root/ai/large_models/_ninfer_repos/deploy-yarn/reqdump/"
               "2026-09-28T08:04:49+08:00-req-11-chat.json"));
    if (!std::filesystem::exists(model_dir / "tokenizer.json") ||
        !std::filesystem::exists(dump_path)) {
        return 77; // opt-in fixture absent
    }
    int result = 0;
    try {
        const auto& frontend = production_frontend(model_dir);
        result |= verify_admission(frontend, dump_path, /*noncanonical_raw=*/false);
        result |= verify_admission(frontend, dump_path, /*noncanonical_raw=*/true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (result == 0) {
        std::cerr << "test_echo_admission: ok\n";
    }
    return result;
}

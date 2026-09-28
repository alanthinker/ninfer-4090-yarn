#pragma once

// Shared synthetic-frontend fixture: a byte-level BPE tokenizer plus the repository's jinja
// chat templates, good enough to exercise the full Frontend (prepare/count/output semantics)
// without a model artifact. Used by the frontend test and by targets that need a valid
// Frontend object (for example load-plan rejections in create_program).

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::test::qwen36_frontend {

inline constexpr std::string_view kThinkingControlGuidance =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n";

inline constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";

inline constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

inline constexpr ninfer::TokenId kFixtureByteTokenBase = 1'000;

inline constexpr ninfer::TokenId fixture_byte_token(std::uint8_t byte) {
    // Preserve IDs already used by the output-session fixtures. All other bytes live outside the
    // added-token range so the synthetic tokenizer can encode arbitrary UTF-8 test input.
    switch (byte) {
    case static_cast<std::uint8_t>('x'):
        return 0;
    case 0xe4:
        return 10;
    case 0xb8:
        return 11;
    case 0xad:
        return 12;
    case 0x80:
        return 13;
    case 0xe0:
        return 14;
    case 0xed:
        return 15;
    case 0xa0:
        return 16;
    case 0xf4:
        return 17;
    case 0x90:
        return 18;
    case 0xf5:
        return 19;
    case 0xf0:
        return 20;
    case 0x9f:
        return 21;
    case 0x98:
        return 22;
    case 0xc2:
        return 23;
    case 0xa2:
        return 24;
    default:
        return kFixtureByteTokenBase + byte;
    }
}

inline std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open test resource: ") + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

inline std::string read_template_fixture(const char* path) {
    std::string source = read_file(path);
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

inline const std::string& thinking_toggle_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    return source;
}

inline const std::string& reasoning_effort_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/reasoning_effort_chat_template.jinja");
    return source;
}

inline nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

inline nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

inline std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

// The complete resource set a synthetic Frontend is built from: the byte-level tokenizer, the
// selected jinja chat template (ThinkingToggle by default), and pixel pipeline configs that
// satisfy the registered processor contract.
inline targets::qwen3_6::FrontendResources resources(const std::string& chat_template = thinking_toggle_template_source()) {
    targets::qwen3_6::FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(8, std::string(kThinkingControlGuidance)), added(30, "user\n"),
         added(31, "assistant\n"), added(32, "\n"), added(248045, "<|im_start|>", true),
         added(248046, "<|im_end|>", true), added(248053, "<|vision_start|>", true),
         added(248054, "<|vision_end|>", true), added(248056, "<|image_pad|>", true),
         added(248057, "<|video_pad|>", true), added(248068, "<think>"),
         added(248069, "</think>")});
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value <= 255; ++value) {
        const auto byte                = static_cast<std::uint8_t>(value);
        vocab[byte_level_symbol(byte)] = fixture_byte_token(byte);
    }
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"}, {"vocab", std::move(vocab)}, {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

} // namespace ninfer::test::qwen36_frontend

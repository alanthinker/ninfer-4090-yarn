#pragma once

#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

inline constexpr std::string_view kCanonicalReasoningCloseSerialization = "\n</think>\n\n";

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct MediaPlaceholderByteSpec {
    ByteSpan bytes;
    Modality modality      = Modality::Image;
    std::size_t item_index = 0;
};

struct MediaTokenRunByteSpec {
    ByteSpan bytes;
    Modality modality       = Modality::Image;
    std::size_t item_index  = 0;
    std::size_t frame_index = 0;
};

struct RenderedFragment {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
    [[nodiscard]] RenderedFragment
    rendered_content(bool add_vision_id = false, int* image_count = nullptr,
                     int* video_count = nullptr, std::size_t* media_count = nullptr,
                     std::vector<std::size_t>* part_boundaries = nullptr) const;
};

struct ChatRenderOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    // Internal renderer control used by frontend qualification. Product PromptInput always
    // selects either a new assistant turn or continuation of the final assistant.
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    std::vector<PromptCacheMarker> cache_markers;
};

struct RewriteCheckpointByteSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::size_t offset         = 0;
};

struct RenderedChat {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
    std::vector<MediaTokenRunByteSpec> media_token_runs;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    // Index n is the exact byte frontier after serializing the first n input messages. A missing
    // value means the template has no independent boundary there (for example, before a leading
    // instruction message folded into the system preamble).
    std::vector<std::optional<std::size_t>> message_boundaries;
    // One rendered byte boundary per requested cache marker.
    std::vector<std::optional<std::size_t>> cache_boundaries;
    // Byte frontier just after the LAST assistant message's body and before the assistant turn's
    // closing serialization (`<|im_end|>\n`) - or after the body of a continued final assistant
    // message, which has no closing serialization. Response-echo reuse splices the resident raw
    // prefix in at this frontier: the echo block's canonical render (render_echo_block) contains
    // both the opener and the closing serialization, so the tail replacing the echoed block must
    // begin before the closing serialization. Splicing at the end of the whole block instead
    // leaves the echoed assistant turn unterminated in the spliced prompt.
    std::optional<std::size_t> assistant_body_end;
};

enum class ChatTemplateSemantics : std::uint8_t {
    ThinkingToggle,
    ReasoningEffort,
};

// Split raw generated text into (reasoning, content) with the same semantics the jinja template
// uses when reasoning_content is not provided: reasoning is the text between the last <think>
// and the first </think> (surrounding newlines stripped); content is everything after the last
// </think> (leading newlines stripped). Without a </think> the whole text is content.
[[nodiscard]] std::pair<std::string, std::string> split_think(std::string_view raw);

// Render the canonical assistant block for one generated response: the exact bytes the prompt
// renderer emits for the same (reasoning, content, tool_calls) fields when a follow-up request
// echoes them back -
//
//   <|im_start|>assistant\n
//   <think>\n{trimmed reasoning}\n</think>\n\n
//   {content}{tool call blocks}
//   <|im_end|>\n
//
// `allow_empty_arguments` mirrors the template semantics (ReasoningEffort templates permit an
// empty argument object; ThinkingToggle templates do not). Response-echo reuse digests this
// block on both sides, so its bytes are part of the reuse contract.
[[nodiscard]] std::string render_echo_block(std::string_view reasoning, std::string_view content,
                                            const std::vector<ToolCall>& tool_calls,
                                            bool allow_empty_arguments);

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate resolve(std::string_view source);

    [[nodiscard]] PromptCapabilities capabilities() const noexcept;
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options = {}) const;

private:
    explicit CompiledChatTemplate(ChatTemplateSemantics semantics) noexcept
        : semantics_(semantics) {}

    ChatTemplateSemantics semantics_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal

// Response-echo reuse, CPU-qualified:
//   * the response-echo spec the frontend computes when preparing a continuation prompt
//     (text-space digests at the response boundary + boundary token);
//   * the digest chain: the spec's prefix digest of round N+1 equals the prompt text digest
//     carried by round N's state (the shared history a client re-sends byte-identically);
//   * the canonical response block: render_echo_block and the prompt renderer agree on the
//     bytes of an echoed assistant message, so the spec's block digest is a pure function of
//     the echoed fields;
//   * response_block_digest normalizes the model's RAW generated text (whitespace around the
//     thinking block) to the same canonical block digest, which is what lets an endpoint whose
//     raw tokens differ from the re-tokenized echo still match;
//   * the negative gates: a TurnClosure (non-preserve) continuation carries no echo spec, and
//     a tampered response block misses its digest.

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/frontend_fixture.h"
#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/digest.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <optional>
#include <vector>

namespace {

using Frontend        = ninfer::targets::qwen3_6::Frontend;
using FrontendFactory = ninfer::targets::qwen3_6::FrontendTestAccess;
namespace fi          = ninfer::targets::qwen3_6::frontend_internal;
namespace pf          = ninfer::targets::qwen3_6::detail;

std::size_t failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

const Frontend& fixture_frontend() {
    static const Frontend value =
        FrontendFactory::create_component(ninfer::test::qwen36_frontend::resources(), false);
    return value;
}

using PreparedPrompt = ninfer::targets::qwen3_6::PreparedPrompt;
using PreparedPromptData = ninfer::targets::qwen3_6::PreparedPromptData;

const PreparedPromptData& data_of(const PreparedPrompt& prepared) {
    return FrontendFactory::inspect(prepared);
}

ninfer::ChatMessage user_message(std::string text) {
    return ninfer::ChatMessage{.role  = ninfer::ChatRole::User,
                               .parts = {ninfer::MessagePart{
                                   .kind = ninfer::MessagePartKind::Text,
                                   .text = std::move(text)}}};
}

ninfer::ChatMessage assistant_message(std::string text, std::string reasoning) {
    return ninfer::ChatMessage{.role              = ninfer::ChatRole::Assistant,
                               .parts             = {ninfer::MessagePart{
                                   .kind = ninfer::MessagePartKind::Text,
                                   .text = std::move(text)}},
                               .reasoning_content = std::move(reasoning)};
}

ninfer::PromptInput conversation(std::vector<ninfer::ChatMessage> messages,
                                 bool preserve_thinking) {
    return ninfer::PromptInput{.messages = std::move(messages),
                               .options  = ninfer::PromptOptions{
                                   .preserve_thinking = preserve_thinking}};
}

int verify_spec_chain() {
    const Frontend& frontend = fixture_frontend();

    // Round 1: the user asks; with preserve_thinking the round ends in a ResponseReplay
    // rewrite checkpoint and carries its prompt text digest (the state's echo prefix key).
    auto prepared_first =
        frontend.prepare(conversation({user_message("x")}, true));
    const auto& first = data_of(prepared_first);
    check(first.identity.rewrite_checkpoint.has_value() &&
              first.identity.rewrite_checkpoint->kind ==
                  ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay,
          "preserve-thinking continuation must carry a ResponseReplay rewrite checkpoint");
    check(first.identity.response_echo == std::nullopt,
          "a prompt with no assistant response in its history must carry no echo spec");
    check(std::any_of(first.identity.prompt_text_digest.begin(),
                      first.identity.prompt_text_digest.end(),
                      [](std::uint8_t byte) { return byte != 0; }),
          "continuation prompt must carry a nonzero prompt text digest");
    const auto first_text_digest = first.identity.prompt_text_digest;

    // The canonical block of the response the engine will generate for round 1.
    const std::string reasoning = "r1";
    const std::string content   = "y";
    const std::string block = fi::render_echo_block(reasoning, content, {}, false);
    check(block.find("<think>\nr1") != std::string::npos, "canonical block misses reasoning");
    check(block.find("</think>\n\ny") != std::string::npos,
          "canonical block misses the canonical close serialization");

    // Round 2: the client echoes round 1's response and adds the next user message.
    auto history = std::vector<ninfer::ChatMessage>{user_message("x")};
    history.push_back(assistant_message(content, reasoning));
    history.push_back(user_message("z"));
    auto prepared_second = frontend.prepare(conversation(std::move(history), true));
    const auto& second = data_of(prepared_second);
    check(second.identity.response_echo.has_value(),
          "continuation prompt with an echoed response must carry an echo spec");
    if (!second.identity.response_echo) { return 1; }
    const auto& spec = *second.identity.response_echo;

    // Digest chain: round 2's prefix digest is round 1's carried prompt text digest.
    check(spec.prefix_digest == first_text_digest, "echo prefix digest breaks the round chain");
    // The digest round 2 carries advances with the new shared history.
    check(second.identity.prompt_text_digest != first_text_digest,
          "carried digest must advance with the new round");
    check(spec.boundary_token > 0 && spec.boundary_token < second.token_ids.size(),
          "echo boundary token must sit strictly inside the prompt");

    // The block digest is a pure function of the echoed fields: preparing the same history
    // again yields the same spec.
    auto repeated_history = std::vector<ninfer::ChatMessage>{user_message("x")};
    repeated_history.push_back(assistant_message(content, reasoning));
    repeated_history.push_back(user_message("z"));
    auto prepared_repeated = frontend.prepare(conversation(std::move(repeated_history), true));
    const auto& repeated  = data_of(prepared_repeated);
    check(repeated.identity.response_echo.has_value() &&
              repeated.identity.response_echo->block_digest == spec.block_digest &&
              repeated.identity.response_echo->prefix_digest == spec.prefix_digest &&
              repeated.identity.response_echo->boundary_token == spec.boundary_token,
          "echo spec must be deterministic for the same echoed fields");
    return failures == 0 ? 0 : 1;
}

int verify_block_digest_normalization() {
    const Frontend& frontend = fixture_frontend();
    const std::string reasoning = "r1";
    const std::string content   = "y";
    const auto canonical_digest =
        fi::sha256(fi::render_echo_block(reasoning, content, {}, false));

    // The raw generated span starts mid-reasoning: the prompt's generation opener already
    // emitted `<think>\n`, so the state ledger holds the reasoning text up to the close
    // serialization plus the content. A model whose whitespace around the close deviates from
    // the canonical serialization still parses to the same canonical block digest: the
    // response-echo match never depends on the model reproducing the template's exact
    // whitespace.
    const auto noncanonical_digest = frontend.response_block_digest(
        frontend.tokenize_text("r1\n</think>\ny"), nullptr, true, false);
    check(noncanonical_digest == canonical_digest,
          "non-canonical raw whitespace must normalize to the canonical block digest");

    // The canonical raw span (what a well-behaved model emits) digests to the same value.
    const auto roundtrip_digest =
        frontend.response_block_digest(frontend.tokenize_text("r1\n</think>\n\ny"), nullptr, true, false);
    check(roundtrip_digest == canonical_digest,
          "canonical raw text must round-trip to the canonical block digest");

    // A response the client did not echo does not match.
    const auto tampered_digest =
        frontend.response_block_digest(frontend.tokenize_text("r1\n</think>\n\nz"), nullptr, true, false);
    check(tampered_digest != canonical_digest, "tampered response must miss the block digest");

    // A thinking-disabled round starts its raw span in content, not reasoning.
    const auto no_think_digest =
        frontend.response_block_digest(frontend.tokenize_text("y"), nullptr, false, false);
    check(no_think_digest == fi::sha256(fi::render_echo_block("", "y", {}, false)),
          "content-only raw span must digest to the empty-reasoning canonical block");
    return failures == 0 ? 0 : 1;
}

int verify_splice_image_shape() {
    // A settled resident state's ledger, identity, and digest images all carry ONE entry beyond
    // its execution frontier: the committed pending token (ledger_frontier ==
    // execution_frontier + 1). The response-echo splice keeps only the raw prefix, so the copied
    // images must be truncated to the frontier before the re-tokenized tail is appended;
    // otherwise all three stay one entry ahead of the spliced ledger and every generated-token
    // commit reports a shape mismatch ("generated-prefix identity is not at its base or
    // committed extent").
    const Frontend& frontend = fixture_frontend();
    auto prepared = frontend.prepare(conversation({user_message("x")}, true));
    const auto& prompt = data_of(prepared);
    const std::size_t frontier = prompt.token_ids.size();
    check(frontier != 0, "the fixture prompt must tokenize");

    pf::ResidentPrefixIdentity identity;
    identity.assign(prompt);
    pf::PrefixShortlistDigests digests;
    digests.assign(prompt);
    check(identity.size() == frontier && digests.size() == frontier,
          "a freshly assigned image matches its token count");

    // One generated token was accepted beyond the prefill: the state is now settled.
    const std::span<const ninfer::TokenId> pending(prompt.token_ids);
    identity.append_generated(1, prompt.rope_delta);
    digests.append_generated(pending.subspan(frontier - 1, 1), prompt.rope_delta);
    check(identity.size() == frontier + 1 && digests.size() == frontier + 1,
          "a settled state's images carry one entry beyond its execution frontier");

    // The splice: raw prefix [0, frontier) plus a re-tokenized tail.
    const std::span<const ninfer::TokenId> tail = pending.subspan(frontier - 1, 1);
    pf::ResidentPrefixIdentity spliced_identity = identity;
    spliced_identity.truncate(frontier);
    spliced_identity.append_generated(tail.size(), prompt.rope_delta);
    pf::PrefixShortlistDigests spliced_digests = digests;
    spliced_digests.truncate(frontier);
    spliced_digests.append_generated(tail, prompt.rope_delta);
    check(spliced_identity.size() == frontier + tail.size() &&
              spliced_digests.size() == spliced_identity.size(),
          "the spliced prefix and tail keep the ledger, identity, and digest images aligned");
    return failures == 0 ? 0 : 1;
}

int verify_negative_shapes() {
    const Frontend& frontend = fixture_frontend();

    // A non-preserve continuation closes the turn (TurnClosure rewrite): it carries no echo
    // spec (nothing was generated yet), but its carried digest still anchors the next round's
    // prefix key.
    auto prepared_turn =
        frontend.prepare(conversation({user_message("x")}, false));
    const auto& turn = data_of(prepared_turn);
    check(turn.identity.response_echo == std::nullopt,
          "TurnClosure continuation must carry no echo spec");
    check(std::any_of(turn.identity.prompt_text_digest.begin(),
                      turn.identity.prompt_text_digest.end(),
                      [](std::uint8_t byte) { return byte != 0; }),
          "continuation with a rewrite checkpoint must carry its prompt text digest");

    // A prompt whose only assistant message is the trailing one (no following user message)
    // still carries the spec: the next generation continues the same turn.
    auto trailing = std::vector<ninfer::ChatMessage>{user_message("x"),
                                                     assistant_message("y", "r1")};
    auto prepared_trailing =
        frontend.prepare(conversation(std::move(trailing), true));
    const auto& trailing_data = data_of(prepared_trailing);
    check(trailing_data.identity.response_echo.has_value(),
          "trailing-assistant continuation must carry an echo spec");
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    int result = verify_spec_chain();
    result |= verify_splice_image_shape();
    result |= verify_block_digest_normalization();
    result |= verify_negative_shapes();
    if (result == 0) {
        std::cerr << "test_echo_reuse: ok\n";
    }
    return result;
}

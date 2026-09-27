// Regression tests for raw-request-dump retention ordering and name round-trips.
//
// Retention: before the fix, prune_dump_directory ordered directory entries by the leading
// request number in the file name. Request numbers restart at zero on every process
// restart, so in a directory that outlives a restart the newest dumps (small numbers) sat
// at the front of the deletion zone: every fresh write evicted the newest dumps while
// multi-day-old files (large numbers) survived indefinitely. Retention must order by
// capture time instead.
//
// Names: finalized names are time-leading "<time>-req-<number>-<route>.json" with the RFC 3339
// local form "YYYY-MM-DDTHH:MM:SS±HH:MM" (local wall clock plus its explicit UTC offset) at
// the front. The offset makes the parse pure arithmetic, so a name written on a UTC+8 host
// must mean the same instant on a UTC+14 host — the tests below verify that by switching the
// process timezone between write and parse. The number is the request's own pre-assigned id,
// identical in the pending name ("req-<unix-ms>-s<id>-<route>.json") and the finalized name,
// so same-second captures can never collide. Names written by older builds
// ("req-<number>-<time>-<route>.json" finalized, "req-<unix-ms>-<route>.json" pending) must
// keep parsing so a directory outlives the change.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include <cassert>

#include <nlohmann/json.hpp>

namespace ninfer::serve {
void prune_dump_directory(const std::string& directory, std::int64_t now_ms);
void dump_response_body(const char* route, std::uint64_t request_id,
                        const nlohmann::json& payload);
struct DumpFile {
    std::int64_t number = 0;
    std::int64_t time_ms = 0;
    std::filesystem::path path;
};
bool parse_dump_file_name(const std::string& name, DumpFile& out);
std::optional<std::int64_t> parse_dump_file_time(const std::string& name, std::int64_t number);
std::string format_local_timestamp(std::int64_t ms);
}

namespace {

std::string finalized_name(std::int64_t number, std::int64_t capture_ms) {
    const std::string stamp = ninfer::serve::format_local_timestamp(capture_ms);
    assert(!stamp.empty());
    return stamp + "-req-" + std::to_string(number) + "-chat.json";
}

std::string resp_name(std::int64_t number, std::int64_t write_ms) {
    const std::string stamp = ninfer::serve::format_local_timestamp(write_ms);
    assert(!stamp.empty());
    return stamp + "-resp-" + std::to_string(number) + "-chat.json";
}

std::string legacy_finalized_name(std::int64_t number, std::int64_t capture_ms) {
    const std::string stamp = ninfer::serve::format_local_timestamp(capture_ms);
    assert(!stamp.empty());
    return "req-" + std::to_string(number) + "-" + stamp + "-chat.json";
}

struct Harness {
    std::filesystem::path directory;

    Harness() {
        directory = std::filesystem::temp_directory_path() / "ninfer_dump_retention_test";
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        std::filesystem::create_directories(directory, error);
    }
    ~Harness() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    void add(std::int64_t number, std::int64_t capture_ms) {
        std::ofstream(directory / finalized_name(number, capture_ms)).put('x');
    }
    bool present(std::int64_t number, std::int64_t capture_ms) {
        return std::filesystem::exists(directory / finalized_name(number, capture_ms));
    }
    // Response dumps are written once under their final "<time>-resp-<number>-<route>.json"
    // name; they must be ordered and pruned by write time like any other dump.
    void add_resp(std::int64_t number, std::int64_t write_ms) {
        std::ofstream(directory / resp_name(number, write_ms)).put('x');
    }
    bool present_resp(std::int64_t number, std::int64_t write_ms) {
        return std::filesystem::exists(directory / resp_name(number, write_ms));
    }
    // A file named by an older build: it must be ordered and pruned like any other.
    void add_legacy(std::int64_t number, std::int64_t capture_ms) {
        std::ofstream(directory / legacy_finalized_name(number, capture_ms)).put('x');
    }
    bool present_legacy(std::int64_t number, std::int64_t capture_ms) {
        return std::filesystem::exists(directory / legacy_finalized_name(number, capture_ms));
    }
    void add_pending(std::int64_t capture_ms, std::uint64_t id) {
        // Current pending name: req-<unix-ms>-s<id>-<route>.json (the stamp is the number).
        std::ofstream(directory / ("req-" + std::to_string(capture_ms) + "-s" +
                                   std::to_string(id) + "-chat.json"))
            .put('x');
    }
    bool present_pending(std::int64_t capture_ms, std::uint64_t id) {
        return std::filesystem::exists(directory / ("req-" + std::to_string(capture_ms) + "-s" +
                                                    std::to_string(id) + "-chat.json"));
    }
    void add_legacy_pending(std::int64_t capture_ms) {
        // Older-build pending name: req-<unix-ms>-<route>.json.
        std::ofstream(directory / ("req-" + std::to_string(capture_ms) + "-chat.json")).put('x');
    }
    bool present_legacy_pending(std::int64_t capture_ms) {
        return std::filesystem::exists(directory / ("req-" + std::to_string(capture_ms) +
                                                    "-chat.json"));
    }
};

}  // namespace

int main() {
    // The retention knobs are process-wide statics read on first use: pin them before the
    // first prune call in this process.
    setenv("NINFER_DUMP_REQUESTS_LIMIT", "2", 1);
    setenv("NINFER_DUMP_REQUESTS_MAX_AGE_HOURS", "24", 1);

    const std::int64_t now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    const std::int64_t hour = 3600 * 1000;
    const std::int64_t day  = 24 * hour;

    // 1. Round trip in the process's own timezone: the RFC 3339 name a writer produces
    //    parses back to the same instant (whole seconds by construction).
    {
        const std::int64_t t = now_ms / 1000 * 1000;
        const std::optional<std::int64_t> parsed =
            ninfer::serve::parse_dump_file_time(ninfer::serve::format_local_timestamp(t) +
                                                    "-req-9-chat.json",
                                                9);
        assert(parsed.has_value() && *parsed == t);
        // The same instant named by an older build parses to the same value.
        const std::optional<std::int64_t> legacy =
            ninfer::serve::parse_dump_file_time("req-9-" +
                                                     ninfer::serve::format_local_timestamp(t) +
                                                     "-chat.json",
                                                9);
        assert(legacy.has_value() && *legacy == t);
    }

    // 2. The same name must mean the same instant on a host in another timezone: the
    //    embedded offset, not the reader's zone, decides the conversion.
    {
        const std::int64_t t = now_ms / 1000 * 1000;
        const std::string name = ninfer::serve::format_local_timestamp(t) + "-req-20-chat.json";
        const std::optional<std::int64_t> before =
            ninfer::serve::parse_dump_file_time(name, 20);
        const char* saved_tz = std::getenv("TZ");
        std::string saved = saved_tz == nullptr ? std::string() : saved_tz;
        setenv("TZ", "Etc/GMT-14", 1);  // POSIX flips the sign: UTC+14
        tzset();
        const std::optional<std::int64_t> after =
            ninfer::serve::parse_dump_file_time(name, 20);
        if (saved.empty()) { unsetenv("TZ"); } else { setenv("TZ", saved.c_str(), 1); }
        tzset();
        assert(before.has_value() && after.has_value());
        assert(*before == t && *after == t);
    }

    // 3. Retention orders by capture time, not by request number: one process generation
    //    before the restart holds large numbers and older dumps; the restarted process
    //    holds small numbers and the newest dumps.
    {
        Harness harness;
        harness.add(1000, now_ms - 2 * day);      // beyond the 24 h age limit
        harness.add(370, now_ms - 10 * hour);     // young, but outranked by the restart era
        harness.add(9, now_ms - 2 * hour);
        harness.add(20, now_ms - 1 * hour);
        harness.add_legacy(500, now_ms - 30 * hour);  // 26 h old: named by an older build
        harness.add_pending(now_ms - day - 12 * hour, 7);  // 36 h-old pending file, current name
        harness.add_legacy_pending(now_ms - day - 6 * hour);  // 30 h-old, older-build pending name

        ninfer::serve::prune_dump_directory(harness.directory.string(), now_ms);

        // Under time-ordered retention the two newest entries survive even though their
        // request numbers are the smallest in the directory. Under the old number-ordered
        // rule these two would have been the first files deleted.
        assert(harness.present(9, now_ms - 2 * hour));
        assert(harness.present(20, now_ms - 1 * hour));
        // Two days old: removed by the age rule.
        assert(!harness.present(1000, now_ms - 2 * day));
        // Ten hours old and limit 2: removed by the count rule.
        assert(!harness.present(370, now_ms - 10 * hour));
        // 26 h-old legacy-name file: the older format is still ordered by its embedded time.
        assert(!harness.present_legacy(500, now_ms - 30 * hour));
        // 36 h-old pending file (current name): removed through its unix-ms stamp.
        assert(!harness.present_pending(now_ms - day - 12 * hour, 7));
        // 30 h-old pending file (older build's name): still parsed and pruned.
        assert(!harness.present_legacy_pending(now_ms - day - 6 * hour));
    }

    // 4. Response dumps ("<time>-resp-<number>-<route>.json") parse like request dumps: the
    //    name round-trips to the write instant, the number is extracted, and the result is
    //    timezone-independent.
    {
        const std::int64_t t = now_ms / 1000 * 1000;
        const std::string name = resp_name(445, t);
        ninfer::serve::DumpFile file;
        const bool name_ok = ninfer::serve::parse_dump_file_name(name, file);
        assert(name_ok && file.number == 445);
        const std::optional<std::int64_t> parsed =
            ninfer::serve::parse_dump_file_time(name, 445);
        assert(parsed.has_value() && *parsed == t);

        // A bare "resp-…" file (no RFC 3339 time segment) is operator-owned: the parser must
        // not claim it, or pruning would start deleting files this code never wrote.
        ninfer::serve::DumpFile foreign;
        const bool foreign_ok = ninfer::serve::parse_dump_file_name("resp-123-s4-chat.json", foreign);
        assert(!foreign_ok);
    }

    // 5. Retention treats request and response dumps as one time-ordered pool: the response
    //    of a younger request outranks the request of an older one, and age prunes both kinds.
    {
        Harness harness;
        harness.add(1000, now_ms - 2 * day);        // req, beyond the 24 h age limit
        harness.add_resp(370, now_ms - 10 * hour);  // resp, young but outranked by the newest two
        harness.add_resp(9, now_ms - 2 * hour);     // resp, second newest
        harness.add(20, now_ms - 1 * hour);         // req, newest
        harness.add_resp(500, now_ms - 30 * hour);  // resp, 26 h old

        ninfer::serve::prune_dump_directory(harness.directory.string(), now_ms);

        // Under time-ordered retention the newest request dump and the second-newest response
        // dump survive even though the response number (9) is the smallest in the directory.
        assert(harness.present_resp(9, now_ms - 2 * hour));
        assert(harness.present(20, now_ms - 1 * hour));
        // Two days old: removed by the age rule.
        assert(!harness.present(1000, now_ms - 2 * day));
        // Ten hours old and limit 2: removed by the count rule.
        assert(!harness.present_resp(370, now_ms - 10 * hour));
        // 26 h old: removed by the age rule.
        assert(!harness.present_resp(500, now_ms - 30 * hour));
    }

    // 6. The response writer: disabled (env unset) it writes nothing; enabled it writes one
    //    "<time>-resp-<number>-<route>.json" file whose name round-trips through the retention
    //    parser and whose content is the exact payload. A second dump for a different id lands
    //    next to it, so a req/resp pair can be diffed in a plain listing. The writer's
    //    directory is a process-wide static read on first use, so the disabled case runs in a
    //    forked child (its first call initializes the static from the child's environment)
    //    while the parent keeps testing the enabled path.
    {
        const std::string directory =
            (std::filesystem::temp_directory_path() / "ninfer_dump_resp_write_test").string();
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        std::filesystem::create_directories(directory, error);

        nlohmann::json payload = {
            {"id", 445},
            {"protocol", "openai_chat_completions"},
            {"stream", true},
            {"finish_reason", "stop token"},
            {"usage",
             {{"prompt_tokens", 9441}, {"completion_tokens", 3336}, {"reasoning_tokens", 672}}},
            {"prefix_cache", {{"hit_tokens", 0}, {"reuse_path", "root"}}},
            {"id_slot", 1},
            {"session_digest", "abc123"},
            {"message",
             {{"role", "assistant"},
              {"content", "长故事的结尾。"},
              {"reasoning_content", "先想清楚结尾再收尾。"},
              {"tool_calls", nlohmann::json::array()}}},
        };

        // Disabled mode: the child inherits an unset variable, calls the writer (its first
        // call in its own process), and must leave the directory empty without throwing.
        {
            const pid_t child = ::fork();
            assert(child >= 0);
            if (child == 0) {
                unsetenv("NINFER_DUMP_REQUESTS");
                ninfer::serve::dump_response_body("chat", 445, payload);
                std::vector<std::filesystem::path> files;
                std::error_code scan_error;
                for (const auto& entry :
                     std::filesystem::directory_iterator(directory, scan_error)) {
                    if (scan_error) { break; }
                    files.push_back(entry.path());
                }
                std::_Exit(files.empty() ? 0 : 1);
            }
            int status = 0;
            ::waitpid(child, &status, 0);
            assert(status == 0);  // WIFEXITED(0) == 0 for a clean exit(0)
        }

        setenv("NINFER_DUMP_REQUESTS", directory.c_str(), 1);
        ninfer::serve::dump_response_body("chat", 445, payload);
        ninfer::serve::dump_response_body("chat", 446, payload);
        unsetenv("NINFER_DUMP_REQUESTS");

        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
            files.push_back(entry.path());
        }
        assert(files.size() == 2);

        std::vector<std::string> names;
        std::string file_445;
        for (const auto& file : files) {
            const std::string name = file.filename().string();
            names.push_back(name);
            ninfer::serve::DumpFile parsed;
            const bool parsed_ok = ninfer::serve::parse_dump_file_name(name, parsed);
            assert(parsed_ok);
            const std::optional<std::int64_t> written_time =
                ninfer::serve::parse_dump_file_time(name, parsed.number);
            assert(written_time.has_value());
            if (parsed.number == 445) { file_445 = name; }
        }
        assert(!file_445.empty());
        for (const auto& name : names) {
            // Time-leading so a plain directory listing reads in write order.
            assert(name.rfind("20", 0) == 0);
            assert(name.ends_with("-chat.json"));
            assert(name.find("-resp-") != std::string::npos);
        }

        const nlohmann::json written =
            nlohmann::json::parse(std::ifstream(directory + "/" + file_445));
        assert(written == payload);
        assert(written.at("message").at("content") == "长故事的结尾。");
        assert(written.at("message").at("reasoning_content") == "先想清楚结尾再收尾。");
        assert(written.at("message").at("tool_calls").is_array() &&
               written.at("message").at("tool_calls").empty());

        // Note: the writer's directory is a process-wide static (same pattern as
        // dump_request_body), so the failure-swallowing branch (fopen returning null inside
        // the try/catch) is not separately exercised here; both writers share it.
        std::filesystem::remove_all(directory, error);
    }
    std::printf("ok\n");
    return 0;
}

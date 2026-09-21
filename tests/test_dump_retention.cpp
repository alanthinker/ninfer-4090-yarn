// Regression tests for raw-request-dump retention ordering and name round-trips.
//
// Retention: before the fix, prune_dump_directory ordered directory entries by the leading
// request number in the file name. Request numbers restart at zero on every process
// restart, so in a directory that outlives a restart the newest dumps (small numbers) sat
// at the front of the deletion zone: every fresh write evicted the newest dumps while
// multi-day-old files (large numbers) survived indefinitely. Retention must order by
// capture time instead.
//
// Names: finalized names carry the RFC 3339 local form "YYYY-MM-DDTHH:MM:SS±HH:MM" (local
// wall clock plus its explicit UTC offset). The offset makes the parse pure arithmetic, so
// a name written on a UTC+8 host must mean the same instant on a UTC+14 host — the tests
// below verify that by switching the process timezone between write and parse.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <cassert>

namespace ninfer::serve {
void prune_dump_directory(const std::string& directory, std::int64_t now_ms);
std::optional<std::int64_t> parse_dump_file_time(const std::string& name, std::int64_t number);
std::string format_local_timestamp(std::int64_t ms);
}

namespace {

std::string finalized_name(std::int64_t number, std::int64_t capture_ms) {
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
    void add_pending(std::int64_t capture_ms) {
        // Pending files are named req-<unix-ms>-<route>.json: the stamp is the number.
        std::ofstream(directory / ("req-" + std::to_string(capture_ms) + "-chat.json")).put('x');
    }
    bool present_pending(std::int64_t capture_ms) {
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
            ninfer::serve::parse_dump_file_time("req-9-" +
                                                    ninfer::serve::format_local_timestamp(t) +
                                                    "-chat.json",
                                                9);
        assert(parsed.has_value() && *parsed == t);
    }

    // 2. The same name must mean the same instant on a host in another timezone: the
    //    embedded offset, not the reader's zone, decides the conversion.
    {
        const std::int64_t t = now_ms / 1000 * 1000;
        const std::string name = "req-20-" +
                                 ninfer::serve::format_local_timestamp(t) + "-chat.json";
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
        harness.add_pending(now_ms - day - 12 * hour);  // 36 h-old pending file

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
        // 36 h-old pending file: removed by the age rule through its unix-ms stamp.
        assert(!harness.present_pending(now_ms - day - 12 * hour));
    }
    std::printf("ok\n");
    return 0;
}

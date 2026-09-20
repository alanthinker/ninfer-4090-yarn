#pragma once

#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// StateIndex: in-memory hash table mapping prefix digest (at anchor frontier) to a StateImage
// that has a host replica. Provides discoverability for states whose catalog entry has been
// evicted: even when no ContinuationSlot references the state, the StateIndex retains a
// checkpoint_reference so the host slot data survives, and a later request with the same
// prefix can find and H2D-restore the state.
//
// Invariant: for HostOnly states (device_slot == none), the StateIndex is the only release
// path. drop_host_replica requires device_slot to exist; H2D uses keep_source_replica=true.
//
// Thread model: single-threaded (Program main loop). No synchronization needed.
//
// Hash table: open addressing with linear probing and tombstone deletion.
// Table size = 2x capacity (50% max load factor) to keep probe chains short.

using DigestPair = std::array<std::uint64_t, 2>;

class StateIndex {
public:
    struct Entry {
        DigestPair digest{};
        StateImageHandle state;
        std::uint32_t frontier = 0;
        std::uint64_t last_access_ns = 0;
    };

    explicit StateIndex(std::size_t capacity)
        : capacity_(capacity), slots_(capacity * 2U) {
        if (capacity_ == 0) {
            throw std::invalid_argument("StateIndex capacity must be nonzero");
        }
    }

    StateIndex(const StateIndex&)            = delete;
    StateIndex& operator=(const StateIndex&) = delete;
    StateIndex(StateIndex&&)                 = delete;
    StateIndex& operator=(StateIndex&&)      = delete;

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t occupied() const noexcept { return occupied_; }

    // Insert a state into the index. If the table is full, evicts the lowest-value entry
    // (lowest rebuild_cost * recency score) and returns the evicted entry so the caller
    // can release its checkpoint_reference. Returns nullopt only if insertion failed.
    // The caller MUST release the checkpoint_reference of the returned (evicted) entry.
    [[nodiscard]] std::optional<Entry> insert(const DigestPair& digest, StateImageHandle state,
                                              std::uint32_t frontier) {
        std::optional<Entry> evicted;
        if (occupied_ >= capacity_) {
            evicted = evict_lru();
            if (!evicted) { return std::nullopt; }
        }
        const std::size_t index = find_slot(digest);
        if (slots_[index].state == SlotState::Occupied) {
            slots_[index].state_handle   = state;
            slots_[index].frontier       = frontier;
            slots_[index].last_access_ns = now_ns();
            return evicted;
        }
        slots_[index].state          = SlotState::Occupied;
        slots_[index].digest         = digest;
        slots_[index].state_handle   = state;
        slots_[index].frontier       = frontier;
        slots_[index].last_access_ns = now_ns();
        ++occupied_;
        return evicted;
    }

    // Lookup by digest. Returns the entry by value if found (and marks it as accessed).
    // The caller must verify state handle validity before use.
    [[nodiscard]] std::optional<Entry> lookup(const DigestPair& digest) {
        const std::size_t index = find_slot(digest);
        if (slots_[index].state != SlotState::Occupied || slots_[index].digest != digest) {
            return std::nullopt;
        }
        slots_[index].last_access_ns = now_ns();
        return to_entry(slots_[index]);
    }

    // Touch an existing entry (e.g., when catalog path hits a state also in StateIndex).
    void touch(const DigestPair& digest) {
        const std::size_t index = find_slot(digest);
        if (slots_[index].state == SlotState::Occupied && slots_[index].digest == digest) {
            slots_[index].last_access_ns = now_ns();
        }
    }

    // Remove an entry by digest (marks as tombstone). Returns true if an entry was removed.
    bool erase(const DigestPair& digest) {
        const std::size_t index = find_slot(digest);
        if (slots_[index].state != SlotState::Occupied || slots_[index].digest != digest) {
            return false;
        }
        slots_[index].state = SlotState::Tombstone;
        --occupied_;
        return true;
    }

    // Evict the entry with the lowest eviction_score (least valuable: cheap to rebuild and/or
    // stale). Returns the evicted entry (caller must release its checkpoint_reference and state).
    [[nodiscard]] std::optional<Entry> evict_lru() {
        std::size_t best        = 0;
        double best_score       = std::numeric_limits<double>::max();
        bool best_occupied      = false;
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].state != SlotState::Occupied) { continue; }
            const double score = eviction_score(slots_[i]);
            if (!best_occupied || score < best_score) {
                best          = i;
                best_score    = score;
                best_occupied = true;
            }
        }
        if (!best_occupied) { return std::nullopt; }
        Entry entry = to_entry(slots_[best]);
        slots_[best].state = SlotState::Tombstone;
        --occupied_;
        return entry;
    }

    // Remove all entries that reference a specific state handle. Used when a state is released
    // through a non-StateIndex path (defensive cleanup).
    void erase_by_state(StateImageHandle state) {
        for (auto& slot : slots_) {
            if (slot.state == SlotState::Occupied && slot.state_handle == state) {
                slot.state = SlotState::Tombstone;
                --occupied_;
            }
        }
    }

private:
    enum class SlotState : std::uint8_t { Empty, Occupied, Tombstone };

    struct Slot {
        SlotState state = SlotState::Empty;
        DigestPair digest{};
        StateImageHandle state_handle;
        std::uint32_t frontier = 0;
        std::uint64_t last_access_ns = 0;
    };

    [[nodiscard]] static Entry to_entry(const Slot& s) noexcept {
        return Entry{.digest = s.digest, .state = s.state_handle,
                     .frontier = s.frontier, .last_access_ns = s.last_access_ns};
    }

    [[nodiscard]] std::size_t hash(const DigestPair& digest) const noexcept {
        std::uint64_t h = 1469598103934665603ULL;
        for (const std::uint64_t lane : digest) {
            for (std::uint32_t byte = 0; byte < 8; ++byte) {
                h ^= static_cast<std::uint8_t>(lane >> (8U * byte));
                h *= 1099511628211ULL;
            }
        }
        return static_cast<std::size_t>(h % slots_.size());
    }

    // Linear probing: stops at Empty (definitive not-found) or matching digest.
    // Tombstone slots are skipped (continue probing). Insert reuses the first Tombstone seen.
    [[nodiscard]] std::size_t find_slot(const DigestPair& digest) const {
        const std::size_t start = hash(digest);
        std::size_t first_tombstone = start;
        bool saw_tombstone = false;
        for (std::size_t probe = 0; probe < slots_.size(); ++probe) {
            const std::size_t candidate = (start + probe) % slots_.size();
            const SlotState ss = slots_[candidate].state;
            if (ss == SlotState::Empty) {
                // Definitive: key is not in the table. Use first tombstone if any, else here.
                return saw_tombstone ? first_tombstone : candidate;
            }
            if (ss == SlotState::Tombstone) {
                if (!saw_tombstone) {
                    first_tombstone = candidate;
                    saw_tombstone   = true;
                }
                continue;
            }
            // Occupied: check if it's our key.
            if (slots_[candidate].digest == digest) {
                return candidate;
            }
        }
        // Table is full of Occupied+Tombstone with no Empty — should not happen at 50% load.
        if (saw_tombstone) { return first_tombstone; }
        throw std::logic_error("StateIndex table is full");
    }

    [[nodiscard]] static double eviction_score(const Slot& slot) noexcept {
        // Higher score = more valuable (keep). Lower score = evict first.
        const double rebuild_cost = static_cast<double>(slot.frontier) / 1200.0;
        const double age_hours =
            static_cast<double>(now_ns() - slot.last_access_ns) / 3.6e12;
        const double recency = std::exp(-age_hours / 4.0);
        return rebuild_cost * recency;
    }

    [[nodiscard]] static std::uint64_t now_ns() noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    const std::size_t capacity_;
    std::size_t occupied_ = 0;
    std::vector<Slot> slots_;
};

} // namespace ninfer::targets::qwen3_6::detail

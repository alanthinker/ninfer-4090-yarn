#pragma once

#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_index.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// KVIndex: in-memory hash table mapping prefix digest (at anchor frontier) to a
// KVAddressSpaceHandle. Complements StateIndex: while StateIndex preserves the GDN state,
// KVIndex preserves the attention KV pages. Together they enable full reuse (state + KV)
// after catalog eviction.
//
// Thread model: single-threaded (Program main loop). No synchronization needed.

class KVIndex {
public:
    struct Entry {
        DigestPair digest{};
        KVAddressSpaceHandle text_address;
        std::optional<KVAddressSpaceHandle> backend_address;
        std::uint32_t frontier = 0;
        std::uint64_t last_access_ns = 0;
    };

    explicit KVIndex(std::size_t capacity)
        : capacity_(capacity), slots_(capacity * 2U) {
        if (capacity_ == 0) {
            throw std::invalid_argument("KVIndex capacity must be nonzero");
        }
    }

    KVIndex(const KVIndex&)            = delete;
    KVIndex& operator=(const KVIndex&) = delete;
    KVIndex(KVIndex&&)                 = delete;
    KVIndex& operator=(KVIndex&&)      = delete;

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t occupied() const noexcept { return occupied_; }

    // Insert into the index. If the table is full, evicts the lowest-value entry and
    // returns it so the caller can release its pins. Returns nullopt on failure.
    // The caller MUST release pins on the returned (evicted) entry.
    [[nodiscard]] std::optional<Entry> insert(const DigestPair& digest,
                                              KVAddressSpaceHandle text_address,
                                              std::optional<KVAddressSpaceHandle> backend_address,
                                              std::uint32_t frontier) {
        std::optional<Entry> evicted;
        if (occupied_ >= capacity_) {
            evicted = evict_lru();
            if (!evicted) { return std::nullopt; }
        }
        const std::size_t index = find_slot(digest);
        if (slots_[index].state == SlotState::Occupied) {
            slots_[index].text_address    = text_address;
            slots_[index].backend_address = backend_address;
            slots_[index].frontier        = frontier;
            slots_[index].last_access_ns  = now_ns();
            return evicted;
        }
        slots_[index].state           = SlotState::Occupied;
        slots_[index].digest          = digest;
        slots_[index].text_address    = text_address;
        slots_[index].backend_address = backend_address;
        slots_[index].frontier        = frontier;
        slots_[index].last_access_ns  = now_ns();
        ++occupied_;
        return evicted;
    }

    [[nodiscard]] std::optional<Entry> lookup(const DigestPair& digest) {
        const std::size_t index = find_slot(digest);
        if (slots_[index].state != SlotState::Occupied || slots_[index].digest != digest) {
            return std::nullopt;
        }
        slots_[index].last_access_ns = now_ns();
        return to_entry(slots_[index]);
    }

    void touch(const DigestPair& digest) {
        const std::size_t index = find_slot(digest);
        if (slots_[index].state == SlotState::Occupied && slots_[index].digest == digest) {
            slots_[index].last_access_ns = now_ns();
        }
    }

    bool erase(const DigestPair& digest) {
        const std::size_t index = find_slot(digest);
        if (slots_[index].state != SlotState::Occupied || slots_[index].digest != digest) {
            return false;
        }
        slots_[index].state = SlotState::Tombstone;
        --occupied_;
        return true;
    }

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

    void erase_by_address(KVAddressSpaceHandle address) {
        for (auto& slot : slots_) {
            if (slot.state == SlotState::Occupied && slot.text_address == address) {
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
        KVAddressSpaceHandle text_address;
        std::optional<KVAddressSpaceHandle> backend_address;
        std::uint32_t frontier = 0;
        std::uint64_t last_access_ns = 0;
    };

    [[nodiscard]] static Entry to_entry(const Slot& s) noexcept {
        return Entry{.digest = s.digest, .text_address = s.text_address,
                     .backend_address = s.backend_address,
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

    [[nodiscard]] std::size_t find_slot(const DigestPair& digest) const {
        const std::size_t start = hash(digest);
        std::size_t first_tombstone = start;
        bool saw_tombstone = false;
        for (std::size_t probe = 0; probe < slots_.size(); ++probe) {
            const std::size_t candidate = (start + probe) % slots_.size();
            const SlotState ss = slots_[candidate].state;
            if (ss == SlotState::Empty) {
                return saw_tombstone ? first_tombstone : candidate;
            }
            if (ss == SlotState::Tombstone) {
                if (!saw_tombstone) {
                    first_tombstone = candidate;
                    saw_tombstone   = true;
                }
                continue;
            }
            if (slots_[candidate].digest == digest) {
                return candidate;
            }
        }
        if (saw_tombstone) { return first_tombstone; }
        throw std::logic_error("KVIndex table is full");
    }

    [[nodiscard]] static double eviction_score(const Slot& slot) noexcept {
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

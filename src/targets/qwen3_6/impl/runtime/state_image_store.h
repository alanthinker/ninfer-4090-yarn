#pragma once

#include <ninfer/targets/qwen3_6/state_image.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class StateImageStore;

class StateImageHandle {
public:
    StateImageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(StateImageHandle, StateImageHandle) noexcept = default;

private:
    StateImageHandle(const StateImageStore* owner, std::uint32_t index,
                     std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const StateImageStore* owner_ = nullptr;
    std::uint32_t index_          = 0;
    std::uint32_t generation_     = 0;

    friend class StateImageStore;
};

enum class StateImageRole : std::uint8_t {
    Free,
    ActiveMutable,
    CheckpointImmutable,
    ReservedDestination,
};

enum class StateReplicaResidency : std::uint8_t {
    None,
    DeviceOnly,
    HostOnly,
    Both,
};

enum class StateTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    HostToFork,
};

struct StateImageSelectors {
    std::int32_t source      = -1;
    std::int32_t destination = -1;
};

class StateImageTransfer {
public:
    StateImageTransfer() noexcept = default;
    ~StateImageTransfer();

    StateImageTransfer(StateImageTransfer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), id_(std::exchange(other.id_, 0)),
          direction_(other.direction_), source_(other.source_), destination_(other.destination_) {}

    StateImageTransfer& operator=(StateImageTransfer&&)      = delete;
    StateImageTransfer(const StateImageTransfer&)            = delete;
    StateImageTransfer& operator=(const StateImageTransfer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] StateTransferDirection direction() const noexcept { return direction_; }

private:
    StateImageTransfer(StateImageStore& owner, std::uint64_t id, StateTransferDirection direction,
                       StateImageHandle source, StateImageHandle destination) noexcept
        : owner_(&owner), id_(id), direction_(direction), source_(source),
          destination_(destination) {}

    StateImageStore* owner_           = nullptr;
    std::uint64_t id_                 = 0;
    StateTransferDirection direction_ = StateTransferDirection::DeviceToHost;
    StateImageHandle source_;
    StateImageHandle destination_;

    friend class StateImageStore;
};

// Program-private logical StateImage ownership. Logical identity is independent from Device and
// Host replicas; published raw Tensor views are reconstructed only from an active Device binding.
class StateImageStore {
public:
    StateImageStore(qwen3_6::StateImageDevicePool& device, qwen3_6::HostStatePool* host,
                    std::uint32_t logical_capacity)
        : device_(&device), host_(host), objects_(logical_capacity),
          free_objects_(logical_capacity),
          free_device_slots_(static_cast<std::size_t>(device.slot_count())),
          free_object_count_(logical_capacity),
          free_device_count_(static_cast<std::uint32_t>(device.slot_count())) {
        if (logical_capacity == 0 || device.slot_count() <= 0 ||
            logical_capacity < static_cast<std::uint32_t>(device.slot_count()) ||
            (host != nullptr && logical_capacity < static_cast<std::uint32_t>(device.slot_count()) +
                                                       host->capacity())) {
            throw std::invalid_argument("StateImageStore capacity is inconsistent");
        }
        for (std::uint32_t index = 0; index < logical_capacity; ++index) {
            free_objects_[index] = logical_capacity - 1U - index;
        }
        for (std::uint32_t index = 0; index < free_device_slots_.size(); ++index) {
            free_device_slots_[index] =
                static_cast<std::int32_t>(free_device_slots_.size() - 1U - index);
        }
    }

    StateImageStore(const StateImageStore&)            = delete;
    StateImageStore& operator=(const StateImageStore&) = delete;
    StateImageStore(StateImageStore&&)                 = delete;
    StateImageStore& operator=(StateImageStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(objects_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept {
        return capacity() - free_object_count_;
    }

    [[nodiscard]] std::uint32_t device_occupied() const noexcept {
        return static_cast<std::uint32_t>(free_device_slots_.size()) - free_device_count_;
    }

    [[nodiscard]] std::uint32_t device_capacity() const noexcept {
        return static_cast<std::uint32_t>(free_device_slots_.size());
    }

    [[nodiscard]] std::uint32_t host_occupied() const noexcept {
        return host_ == nullptr ? 0U : host_->occupied();
    }

    [[nodiscard]] std::uint32_t device_free() const noexcept { return free_device_count_; }

    [[nodiscard]] std::uint32_t host_free() const noexcept {
        return host_ == nullptr ? 0U : host_->capacity() - host_->occupied();
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination, true);
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_logical_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination, false);
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_reset(cudaStream_t stream = nullptr) {
        std::optional<StateImageHandle> handle = allocate(StateImageRole::ActiveMutable, true);
        if (!handle) { return std::nullopt; }
        try {
            Object& object = require(*handle);
            device_->zero_slot(*object.device_slot, stream);
            object.content_epoch = next_epoch();
        } catch (...) {
            (void)release(*handle);
            throw;
        }
        return handle;
    }

    void activate_reset(StateImageHandle handle, cudaStream_t stream = nullptr) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ReservedDestination || !object.device_slot ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage reset reservation is not activatable");
        }
        device_->zero_slot(*object.device_slot, stream);
        object.content_epoch = next_epoch();
        object.role          = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool valid(StateImageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < objects_.size() &&
               objects_[handle.index_].role != StateImageRole::Free &&
               objects_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] std::uint32_t descriptor_index(StateImageHandle handle) const {
        (void)require(handle);
        return handle.index_;
    }

    [[nodiscard]] StateImageRole role(StateImageHandle handle) const {
        return require(handle).role;
    }

    [[nodiscard]] StateReplicaResidency residency(StateImageHandle handle) const {
        const Object& object = require(handle);
        if (object.device_slot && object.host_slot) { return StateReplicaResidency::Both; }
        if (object.device_slot) { return StateReplicaResidency::DeviceOnly; }
        if (object.host_slot) { return StateReplicaResidency::HostOnly; }
        return StateReplicaResidency::None;
    }

    [[nodiscard]] std::uint64_t content_epoch(StateImageHandle handle) const {
        return require(handle).content_epoch;
    }

    // Mark a state as recently touched (for LRU eviction ordering). Called when the state is
    // used as a reuse source, published as a new endpoint, or loaded from host to device.
    void touch(StateImageHandle handle) {
        Object& object = require(handle);
        object.last_touched_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();
    }

    // Mark this state as an endpoint (vs. anchor) for eviction priority purposes.
    void mark_endpoint(StateImageHandle handle) {
        require(handle).is_endpoint = true;
    }

    // How a retained checkpoint gives up one Device StateImage slot. Retention keeps a checkpoint
    // DeviceOnly (fast reuse) or HostOnly (slow reuse); an object holding both replicas exists only
    // while a transfer is in flight and is excluded as a victim, because its Device slot is already
    // committed to that transfer. A DeviceOnly checkpoint must therefore be copied to Host before
    // its slot can be reused, which the caller performs with begin_device_to_host and
    // publish_transfer.
    enum class SlotReleaseKind {
        DropDeviceReplica,  // victim already holds a Host replica: only the Device slot is freed
        CopyToHost,         // victim is DeviceOnly: copy to Host, then free the Device slot
        Drop,               // victim is unreferenced and unpinned: release it entirely
        // The victim holds a Host replica (possibly also a Device replica) and may be pinned:
        // dropping the Host replica frees a Host slot. Without this kind a full Host pool makes
        // the Device pool unrecoverable — CopyToHost needs a free Host slot and Drop refuses
        // pinned victims — so every new checkpoint (and every reuse that must restore one into
        // Device) is rejected and degrades to root prefill. Evicting the oldest anchor's Host
        // replica is what keeps the release ladder moving.
        EvictHostReplica,
    };

    // Select the least-valuable retained checkpoint that can release a slot at the given cost
    // (a Device slot for the first three kinds, a Host slot for EvictHostReplica). Anchors are
    // preferred over endpoints, because endpoints are likelier to be reused by the next request in
    // their session; among one class the least-recently-touched wins. Pinned victims are excluded
    // only for Drop (a full release would orphan live references); EvictHostReplica may drop the
    // Host replica of a pinned anchor, which makes exactly that checkpoint's state unavailable
    // until the owner republishes it. The veto callback receives the candidate handle so the
    // caller can apply ownership rules the store cannot see (a state a live sequence still binds
    // must never be evicted). Returns nullopt when no checkpoint qualifies.
    template <class VetoFn>
    [[nodiscard]] std::optional<StateImageHandle>
    select_slot_release_victim(SlotReleaseKind kind, VetoFn&& veto) const noexcept {
        std::optional<StateImageHandle> best;
        std::uint64_t best_touched = std::numeric_limits<std::uint64_t>::max();
        bool best_is_anchor        = true;
        for (std::uint32_t i = 0; i < objects_.size(); ++i) {
            const Object& obj = objects_[i];
            if (obj.role != StateImageRole::CheckpointImmutable) { continue; }
            if (kind == SlotReleaseKind::EvictHostReplica) {
                // Only a REDUNDANT Host replica may be dropped: the checkpoint must keep its Device
                // replica, or it would end with no published replica at all while its owner still
                // references it ('checkpoint StateImage has no published replica', 2026-09-21).
                // Freeing the last replica of a HostOnly checkpoint is the retirement step's job,
                // which retires the owner and its states together.
                if (!obj.host_slot || !obj.device_slot) { continue; }
            } else {
                if (!obj.device_slot) { continue; }
                if (kind == SlotReleaseKind::DropDeviceReplica && !obj.host_slot) { continue; }
                if (kind != SlotReleaseKind::DropDeviceReplica && obj.host_slot) { continue; }
                if (kind == SlotReleaseKind::Drop && obj.checkpoint_references != 0) {
                    continue;
                }
            }
            if (obj.source_pins != 0 || obj.destination_pinned || has_pending_replica(obj)) {
                continue;
            }
            if (veto(StateImageHandle(this, i, obj.generation))) { continue; }
            const bool is_anchor = !obj.is_endpoint;
            if (!best || (is_anchor && !best_is_anchor) ||
                (is_anchor == best_is_anchor && obj.last_touched_ns < best_touched)) {
                best           = StateImageHandle(this, i, obj.generation);
                best_touched   = obj.last_touched_ns;
                best_is_anchor = is_anchor;
            }
        }
        return best;
    }

    // How many StateImage slots the capacity-release ladder can still free, by mechanism.
    // `drop_device_replica`: Both-resident checkpoints whose Device slot frees without any Host
    // capacity. `demote_candidates`: DeviceOnly checkpoints, each of which needs one free Host
    // slot to demote. `host_evictable`: REDUNDANT Host replicas (Both-resident checkpoints) the
    // EvictHostReplica step can drop to create Host capacity; a HostOnly checkpoint is never
    // counted, because dropping its only replica would leave the checkpoint unpublished. All
    // counts honor the veto callback. A null Host pool means no Host-side mechanism is available.
    struct StateReliefCounts {
        std::uint32_t drop_device_replica = 0;
        std::uint32_t demote_candidates   = 0;
        std::uint32_t host_evictable      = 0;
    };

    template <class VetoFn>
    [[nodiscard]] StateReliefCounts count_state_relief(VetoFn&& veto) const noexcept {
        StateReliefCounts counts;
        if (host_ == nullptr) { return counts; }
        for (std::uint32_t i = 0; i < objects_.size(); ++i) {
            const Object& obj = objects_[i];
            if (obj.role != StateImageRole::CheckpointImmutable) { continue; }
            if (obj.source_pins != 0 || obj.destination_pinned || has_pending_replica(obj)) {
                continue;
            }
            if (veto(StateImageHandle(this, i, obj.generation))) { continue; }
            if (obj.host_slot && obj.device_slot) { ++counts.host_evictable; }
            if (obj.device_slot) {
                if (obj.host_slot) {
                    ++counts.drop_device_replica;
                } else {
                    ++counts.demote_candidates;
                }
            }
        }
        return counts;
    }

    [[nodiscard]] std::uint32_t source_pins(StateImageHandle handle) const {
        return require(handle).source_pins;
    }

    [[nodiscard]] bool destination_pinned(StateImageHandle handle) const {
        return valid(handle) && require(handle).destination_pinned;
    }

    // Replica residency, for callers that price what a release step would free (the planner's
    // capacity-relief credit) without walking the object table themselves.
    [[nodiscard]] bool has_device_replica(StateImageHandle handle) const {
        return valid(handle) && require(handle).device_slot.has_value();
    }

    [[nodiscard]] bool has_host_replica(StateImageHandle handle) const {
        return valid(handle) && require(handle).host_slot.has_value();
    }

    [[nodiscard]] std::uint32_t checkpoint_references(StateImageHandle handle) const {
        return require(handle).checkpoint_references;
    }

    void retain_checkpoint_reference(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable ||
            object.checkpoint_references == std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("StateImage checkpoint reference is not retainable");
        }
        ++object.checkpoint_references;
    }

    void release_checkpoint_reference(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable ||
            object.checkpoint_references == 0) {
            throw std::logic_error("StateImage checkpoint reference is not releasable");
        }
        --object.checkpoint_references;
    }

    // Diagnostic accessors: the object slot and generation behind a handle, for correlating
    // lifecycle logs across a reused slot.
    [[nodiscard]] std::uint32_t debug_index(StateImageHandle handle) const noexcept {
        return handle.index_;
    }

    [[nodiscard]] std::uint32_t debug_generation(StateImageHandle handle) const noexcept {
        return handle.generation_;
    }

    // Last time this state was touched, used to age out the least-valuable retained state when the
    // pools are exhausted.
    [[nodiscard]] std::uint64_t last_touched(StateImageHandle handle) const noexcept {
        return valid(handle) ? objects_[handle.index_].last_touched_ns
                             : std::numeric_limits<std::uint64_t>::max();
    }

    [[nodiscard]] std::int32_t physical_slot(StateImageHandle handle) const {
        const Object& object = require(handle);
        if (!object.device_slot) {
            throw std::logic_error("StateImage has no published Device replica");
        }
        return *object.device_slot;
    }

    // Fork-local (session persistence): read access to a published Host replica so a snapshot
    // can serialize a demoted StateImage without forcing it back onto the device.
    [[nodiscard]] qwen3_6::HostStateImageConstView host_view(StateImageHandle handle) const {
        const Object& object = require(handle);
        if (host_ == nullptr || !object.host_slot) {
            throw std::logic_error("StateImage has no published Host replica");
        }
        return host_->view(*object.host_slot);
    }

    // Fork-local (session persistence): mint a HostOnly CheckpointImmutable image directly from
    // host bytes - the shape a demoted checkpoint has - so a snapshot restore can land
    // checkpoints without a free Device slot. Returns nullopt when the Host pool is absent or
    // full.
    [[nodiscard]] std::optional<StateImageHandle>
    adopt_host_image(const qwen3_6::HostStateImageConstView& source) {
        if (host_ == nullptr || source.data == nullptr || source.layout == nullptr ||
            source.layout->image_bytes != host_->layout().image_bytes) {
            return std::nullopt;
        }
        std::optional<StateImageHandle> handle =
            allocate(StateImageRole::CheckpointImmutable, false);
        if (!handle) { return std::nullopt; }
        std::optional<qwen3_6::HostStateSlotHandle> slot = host_->allocate();
        if (!slot) {
            (void)release(*handle);
            return std::nullopt;
        }
        Object& object = require(*handle);
        std::memcpy(host_->writable_view(*slot).data, source.data, source.layout->image_bytes);
        object.host_slot     = *slot;
        object.content_epoch = next_epoch();
        return handle;
    }

    void move_checkpoint_to_active(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            object.checkpoint_references != 0 || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage checkpoint is not movable");
        }
        if (object.host_slot) {
            if (host_ == nullptr || !host_->release(*object.host_slot)) {
                throw std::logic_error("StateImage Host replica could not be consumed by Move");
            }
            object.host_slot.reset();
        }
        object.role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool drop_device_replica(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            !object.host_slot || object.source_pins != 0 || object.destination_pinned ||
            has_pending_replica(object)) {
            return false;
        }
        std::fprintf(stderr, "state-store: D2H demote handle=%u device_slot=%d -> host_only\n",
                     handle.index_, *object.device_slot);
        return_device_slot(*object.device_slot);
        object.device_slot.reset();
        return true;
    }

    [[nodiscard]] bool drop_host_replica(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.device_slot || !object.host_slot || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object) ||
            !host_->release(*object.host_slot)) {
            return false;
        }
        object.host_slot.reset();
        return true;
    }

    // Drops the Host replica of a retained checkpoint to free a Host StateImage slot, for the
    // EvictHostReplica step of the capacity-release ladder (see SlotReleaseKind). Unlike
    // drop_host_replica this also works on HostOnly states: a pinned HostOnly checkpoint becomes
    // a replica-less placeholder — its checkpoint is unavailable until the owner republishes it —
    // which is the documented "丢最老锚点" degradation, not data corruption: the object keeps its
    // references until the owner is released, and its KV pages stay independent. A Both-resident
    // checkpoint demotes to DeviceOnly and keeps serving Device-side reuse.
    [[nodiscard]] bool evict_host_replica(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.host_slot || !object.device_slot || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object) ||
            !host_->release(*object.host_slot)) {
            return false;
        }
        std::fprintf(stderr,
                     "state-store: anchor host evict handle=%u host_slot=%u refs=%u "
                     "device_only_after=%d\n",
                     handle.index_, object.host_slot->index, object.checkpoint_references,
                     object.device_slot ? 1 : 0);
        object.host_slot.reset();
        return true;
    }

    // Transfers the Device replica to a new logical identity while the old immutable identity
    // keeps its Host replica. This is the zero-copy preserving branch for a Both source.
    void split_device_replica_identity(StateImageHandle source, StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
            !source_object.device_slot || !source_object.host_slot ||
            source_object.source_pins != 0 || source_object.destination_pinned ||
            has_pending_replica(source_object) ||
            destination_object.role != StateImageRole::ReservedDestination ||
            destination_object.device_slot || destination_object.host_slot ||
            has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
            destination_object.destination_pinned) {
            throw std::logic_error("StateImage replica identity split is invalid");
        }
        destination_object.device_slot   = source_object.device_slot;
        destination_object.content_epoch = source_object.content_epoch;
        destination_object.role          = StateImageRole::ActiveMutable;
        source_object.device_slot.reset();
    }

    void freeze(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ActiveMutable || !object.device_slot ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage active image is not freezable");
        }
        object.content_epoch = next_epoch();
        object.role          = StateImageRole::CheckpointImmutable;
        // A newly published checkpoint is the most recent state in the pool, so it must sort after
        // every older one when the pools run out and the oldest entry has to be dropped.
        object.last_touched_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();
    }

    void thaw(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            object.checkpoint_references != 0 || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage checkpoint is not thawable");
        }
        object.role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool can_recycle_checkpoint_destination(StateImageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Object& object = objects_[handle.index_];
        return object.role == StateImageRole::CheckpointImmutable &&
               object.device_slot.has_value() && !object.host_slot &&
               object.checkpoint_references == 1 && object.source_pins == 0 &&
               !object.destination_pinned && !has_pending_replica(object);
    }

    [[nodiscard]] std::uint64_t recycle_checkpoint_destination(StateImageHandle handle) {
        if (!can_recycle_checkpoint_destination(handle)) {
            throw std::logic_error("StateImage checkpoint is not recyclable as a destination");
        }
        Object& object               = require(handle);
        const std::uint64_t epoch    = object.content_epoch;
        object.checkpoint_references = 0;
        object.role                  = StateImageRole::ReservedDestination;
        return epoch;
    }

    void restore_recycled_checkpoint(StateImageHandle handle, std::uint64_t content_epoch) {
        Object& object = require(handle);
        if (content_epoch == 0 || object.role != StateImageRole::ReservedDestination ||
            !object.device_slot || object.host_slot || object.checkpoint_references != 0 ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage recycled checkpoint is not restorable");
        }
        object.content_epoch         = content_epoch;
        object.checkpoint_references = 1;
        object.role                  = StateImageRole::CheckpointImmutable;
    }

    [[nodiscard]] StateImageSelectors begin_fork(StateImageHandle source,
                                                 StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
            !source_object.device_slot ||
            source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
            destination_object.role != StateImageRole::ReservedDestination ||
            !destination_object.device_slot || destination_object.source_pins != 0 ||
            destination_object.destination_pinned || has_pending_replica(destination_object)) {
            throw std::logic_error("StateImage fork binding is invalid");
        }
        ++source_object.source_pins;
        destination_object.destination_pinned = true;
        destination_object.role               = StateImageRole::ActiveMutable;
        destination_object.content_epoch      = source_object.content_epoch;
        return {.source      = *source_object.device_slot,
                .destination = *destination_object.device_slot};
    }

    void commit_fork(StateImageHandle source, StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source_object.role != StateImageRole::CheckpointImmutable ||
            source_object.source_pins == 0 ||
            destination_object.role != StateImageRole::ActiveMutable ||
            !destination_object.device_slot || !destination_object.destination_pinned) {
            throw std::logic_error("StateImage fork commit is invalid");
        }
        --source_object.source_pins;
        destination_object.destination_pinned = false;
    }

    void abort_fork(StateImageHandle source, StateImageHandle destination) {
        if (!can_abort_fork(source, destination)) {
            throw std::logic_error("StateImage fork abort is invalid");
        }
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        --source_object.source_pins;
        destination_object.destination_pinned = false;
        destination_object.role               = StateImageRole::ReservedDestination;
        destination_object.content_epoch      = 0;
    }

    [[nodiscard]] bool can_abort_fork(StateImageHandle source,
                                      StateImageHandle destination) const noexcept {
        if (!valid(source) || !valid(destination)) { return false; }
        const Object& source_object      = objects_[source.index_];
        const Object& destination_object = objects_[destination.index_];
        return source != destination && source_object.role == StateImageRole::CheckpointImmutable &&
               source_object.source_pins != 0 &&
               destination_object.role == StateImageRole::ActiveMutable &&
               destination_object.device_slot.has_value() && destination_object.destination_pinned;
    }

    [[nodiscard]] bool
    can_release_source_after_fork_abort(StateImageHandle source, StateImageHandle destination,
                                        std::uint32_t released_checkpoint_references) const {
        if (!can_abort_fork(source, destination)) { return false; }
        const Object& object = require(source);
        if (released_checkpoint_references > object.checkpoint_references) { return false; }
        if (object.checkpoint_references != released_checkpoint_references) { return true; }
        // abort_fork releases exactly the pin represented by this binding. Any other source pin
        // still protects the object and therefore prevents terminal owner settlement.
        if (object.source_pins != 1 || object.destination_pinned || has_pending_replica(object)) {
            return false;
        }
        if (object.host_slot) {
            if (host_ == nullptr) { return false; }
            (void)host_->view(*object.host_slot);
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_destination_after_fork_abort(StateImageHandle source, StateImageHandle destination,
                                             std::uint32_t released_checkpoint_references) const {
        if (!can_abort_fork(source, destination)) { return false; }
        const Object& object = require(destination);
        if (object.checkpoint_references != released_checkpoint_references ||
            object.source_pins != 0 || has_pending_replica(object)) {
            return false;
        }
        if (object.host_slot) {
            if (host_ == nullptr) { return false; }
            (void)host_->view(*object.host_slot);
        }
        return true;
    }

    [[nodiscard]] StateImageSelectors selectors(StateImageHandle source,
                                                StateImageHandle destination) const {
        const Object& source_object      = require(source);
        const Object& destination_object = require(destination);
        const bool inplace               = source == destination;
        if (!source_object.device_slot || !destination_object.device_slot ||
            (inplace && (source_object.role != StateImageRole::ActiveMutable ||
                         source_object.source_pins != 0 || source_object.destination_pinned)) ||
            (!inplace && (source_object.role != StateImageRole::CheckpointImmutable ||
                          source_object.source_pins == 0 ||
                          destination_object.role != StateImageRole::ActiveMutable ||
                          !destination_object.destination_pinned))) {
            throw std::logic_error("StateImage execution binding is invalid");
        }
        return {.source      = *source_object.device_slot,
                .destination = *destination_object.device_slot};
    }

    [[nodiscard]] std::optional<StateImageTransfer>
    reserve_device_to_host(StateImageHandle source) {
        Object& object = require(source);
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.device_slot || object.host_slot || has_pending_replica(object) ||
            object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        std::optional<qwen3_6::HostStateSlotHandle> target = host_->allocate();
        if (!target) { return std::nullopt; }
        const std::uint64_t transfer = next_transfer();
        object.pending_host_slot     = *target;
        object.transfer_id           = transfer;
        object.destination_pinned    = true;
        ++object.source_pins;
        return StateImageTransfer(*this, transfer, StateTransferDirection::DeviceToHost, source,
                                  source);
    }

    void enqueue_device_to_host(const StateImageTransfer& transfer, cudaStream_t stream = nullptr) {
        validate_transfer(transfer);
        Object& object = require(transfer.source_);
        if (transfer.direction_ != StateTransferDirection::DeviceToHost || host_ == nullptr ||
            !object.device_slot || !object.pending_host_slot) {
            throw std::logic_error("StateImage Device-to-Host transfer is not enqueueable");
        }
        device_->copy_to_host(*object.device_slot, host_->writable_view(*object.pending_host_slot),
                              stream);
    }

    [[nodiscard]] std::optional<StateImageTransfer>
    begin_device_to_host(StateImageHandle source, cudaStream_t stream = nullptr) {
        std::optional<StateImageTransfer> transfer = reserve_device_to_host(source);
        if (!transfer) { return std::nullopt; }
        try {
            enqueue_device_to_host(*transfer, stream);
        } catch (...) {
            abort_transfer(std::move(*transfer));
            throw;
        }
        Object& object = require(source);
        std::fprintf(stderr, "state-store: D2H copy handle=%u device_slot=%d -> host_slot=%u\n",
                     source.index_, object.device_slot ? *object.device_slot : -1,
                     object.pending_host_slot ? object.pending_host_slot->index : 0);
        return transfer;
    }

    [[nodiscard]] std::optional<StateImageTransfer>
    begin_host_to_device(StateImageHandle source, cudaStream_t stream = nullptr) {
        Object& object = require(source);
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.host_slot || object.device_slot || has_pending_replica(object) ||
            object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        const std::optional<std::int32_t> target = take_device_slot();
        if (!target) { return std::nullopt; }
        std::fprintf(stderr, "state-store: H2D load handle=%u host_slot=%u -> device_slot=%d\n",
                     source.index_, object.host_slot->index, *target);
        const std::uint64_t transfer = next_transfer();
        object.pending_device_slot   = *target;
        object.transfer_id           = transfer;
        object.destination_pinned    = true;
        ++object.source_pins;
        try {
            device_->copy_from_host(host_->view(*object.host_slot), *target, stream);
        } catch (...) {
            --object.source_pins;
            object.destination_pinned = false;
            object.transfer_id        = 0;
            object.pending_device_slot.reset();
            return_device_slot(*target);
            throw;
        }
        return StateImageTransfer(*this, transfer, StateTransferDirection::HostToDevice, source,
                                  source);
    }

    [[nodiscard]] std::optional<StateImageTransfer> begin_host_fork(StateImageHandle source,
                                                                    StateImageHandle destination,
                                                                    cudaStream_t stream = nullptr) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (host_ == nullptr || source == destination ||
            source_object.role != StateImageRole::CheckpointImmutable || !source_object.host_slot ||
            source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
            destination_object.role != StateImageRole::ReservedDestination ||
            destination_object.device_slot || destination_object.host_slot ||
            has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
            destination_object.destination_pinned) {
            return std::nullopt;
        }
        const std::optional<std::int32_t> target = take_device_slot();
        if (!target) { return std::nullopt; }
        const std::uint64_t transfer           = next_transfer();
        destination_object.pending_device_slot = *target;
        destination_object.transfer_id         = transfer;
        destination_object.destination_pinned  = true;
        destination_object.role                = StateImageRole::ActiveMutable;
        destination_object.content_epoch       = source_object.content_epoch;
        ++source_object.source_pins;
        try {
            device_->copy_from_host(host_->view(*source_object.host_slot), *target, stream);
        } catch (...) {
            --source_object.source_pins;
            destination_object.pending_device_slot.reset();
            destination_object.transfer_id        = 0;
            destination_object.destination_pinned = false;
            destination_object.role               = StateImageRole::ReservedDestination;
            destination_object.content_epoch      = 0;
            return_device_slot(*target);
            throw;
        }
        return StateImageTransfer(*this, transfer, StateTransferDirection::HostToFork, source,
                                  destination);
    }

    void publish_transfer(StateImageTransfer&& transfer, bool keep_source_replica) {
        validate_transfer(transfer);
        Object& source      = require(transfer.source_);
        Object& destination = require(transfer.destination_);
        switch (transfer.direction_) {
        case StateTransferDirection::DeviceToHost:
            source.host_slot = source.pending_host_slot;
            source.pending_host_slot.reset();
            if (!keep_source_replica) {
                return_device_slot(*source.device_slot);
                source.device_slot.reset();
            }
            --source.source_pins;
            source.destination_pinned = false;
            source.transfer_id        = 0;
            break;
        case StateTransferDirection::HostToDevice:
            source.device_slot = source.pending_device_slot;
            source.pending_device_slot.reset();
            if (!keep_source_replica) {
                if (host_ == nullptr || !host_->release(*source.host_slot)) {
                    throw std::logic_error("StateImage Host replica release failed at publication");
                }
                source.host_slot.reset();
            }
            --source.source_pins;
            source.destination_pinned = false;
            source.transfer_id        = 0;
            break;
        case StateTransferDirection::HostToFork:
            destination.device_slot = destination.pending_device_slot;
            destination.pending_device_slot.reset();
            --source.source_pins;
            destination.destination_pinned = false;
            destination.transfer_id        = 0;
            break;
        }
        consume(transfer);
    }

    void abort_transfer(StateImageTransfer&& transfer) noexcept {
        if (!valid_transfer(transfer)) {
            consume(transfer);
            return;
        }
        Object& source      = objects_[transfer.source_.index_];
        Object& destination = objects_[transfer.destination_.index_];
        if (source.source_pins != 0) { --source.source_pins; }
        if (transfer.direction_ == StateTransferDirection::DeviceToHost) {
            if (host_ != nullptr && source.pending_host_slot) {
                (void)host_->release(*source.pending_host_slot);
            }
            source.pending_host_slot.reset();
            source.destination_pinned = false;
            source.transfer_id        = 0;
        } else {
            if (destination.pending_device_slot) {
                return_device_slot(*destination.pending_device_slot);
            }
            destination.pending_device_slot.reset();
            destination.destination_pinned = false;
            destination.transfer_id        = 0;
            if (transfer.direction_ == StateTransferDirection::HostToFork) {
                destination.role          = StateImageRole::ReservedDestination;
                destination.content_epoch = 0;
            }
        }
        consume(transfer);
    }

    [[nodiscard]] bool release(StateImageHandle handle) noexcept {
        if (!can_release(handle)) { return false; }
        Object& object = objects_[handle.index_];
        std::fprintf(stderr,
                     "[state-store] release handle=%u gen=%u role=%d dev=%d host=%d "
                     "ckpt_refs=%u\n",
                     handle.index_, handle.generation_, static_cast<int>(object.role),
                     object.device_slot ? *object.device_slot : -1,
                     object.host_slot ? static_cast<int>(object.host_slot->index) : -1,
                     object.checkpoint_references);
        std::fflush(stderr);
        if (object.host_slot) {
            if (host_ == nullptr || !host_->release(*object.host_slot)) { return false; }
            object.host_slot.reset();
        }
        if (object.device_slot) {
            return_device_slot(*object.device_slot);
            object.device_slot.reset();
        }
        object.role          = StateImageRole::Free;
        object.content_epoch = 0;
        if (++object.generation == 0) { ++object.generation; }
        free_objects_[free_object_count_++] = handle.index_;
        return true;
    }

    [[nodiscard]] bool can_release(StateImageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Object& object = objects_[handle.index_];
        return object.checkpoint_references == 0 &&
               object.source_pins == 0 && !object.destination_pinned &&
               !has_pending_replica(object);
    }

    [[nodiscard]] bool
    can_release_after_checkpoint_references(StateImageHandle handle,
                                            std::uint32_t released_references) const {
        const Object& object = require(handle);
        if (released_references > object.checkpoint_references) { return false; }
        if (object.checkpoint_references != released_references) { return true; }
        if (object.source_pins != 0 || object.destination_pinned ||
            has_pending_replica(object)) {
            return false;
        }
        if (object.host_slot) {
            if (host_ == nullptr) { return false; }
            (void)host_->view(*object.host_slot);
        }
        return true;
    }

private:
    struct Object {
        std::uint32_t generation    = 1;
        std::uint64_t content_epoch = 0;
        std::optional<std::int32_t> device_slot;
        std::optional<qwen3_6::HostStateSlotHandle> host_slot;
        std::optional<std::int32_t> pending_device_slot;
        std::optional<qwen3_6::HostStateSlotHandle> pending_host_slot;
        std::uint64_t transfer_id           = 0;
        std::uint32_t checkpoint_references = 0;
        std::uint32_t source_pins           = 0;
        bool destination_pinned             = false;
        StateImageRole role                 = StateImageRole::Free;
        // LRU eviction metadata: last time this state was touched (used as a reuse source,
        // published as a new endpoint, or loaded from host to device). Used by evict_lru()
        // to select the least-recently-used state for eviction when device slots are full.
        std::uint64_t last_touched_ns = 0;
        // True if this state is a session endpoint or rewrite checkpoint (a reuse target for
        // the next request in the same session). False for long anchors (fallback reuse
        // points at intermediate positions). Endpoints are lower-priority eviction targets
        // than anchors because they are more likely to be reused next.
        bool is_endpoint = false;
    };

    [[nodiscard]] static bool has_pending_replica(const Object& object) noexcept {
        return object.pending_device_slot.has_value() || object.pending_host_slot.has_value() ||
               object.transfer_id != 0;
    }

    [[nodiscard]] std::optional<StateImageHandle> allocate(StateImageRole role,
                                                           bool with_device) noexcept {
        if (free_object_count_ == 0 || role == StateImageRole::Free ||
            (with_device && free_device_count_ == 0)) {
            std::fprintf(stderr,
                         "[state-store] allocate FAILED role=%d with_device=%d free_object=%u "
                         "free_device=%u\n",
                         static_cast<int>(role), with_device ? 1 : 0, free_object_count_,
                         free_device_count_);
            dump_device_slot_occupants();
            std::fflush(stderr);
            return std::nullopt;
        }
        const std::uint32_t index = free_objects_[--free_object_count_];
        Object& object            = objects_[index];
        object                    = Object{.generation = object.generation, .role = role};
        if (with_device) { object.device_slot = free_device_slots_[--free_device_count_]; }
        return StateImageHandle(this, index, object.generation);
    }

    [[nodiscard]] std::optional<std::int32_t> take_device_slot() noexcept {
        if (free_device_count_ == 0) {
            std::fprintf(stderr, "[state-store] take_device_slot: NONE FREE (dump follows)\n");
            dump_device_slot_occupants();
            std::fflush(stderr);
            return std::nullopt;
        }
        return free_device_slots_[--free_device_count_];
    }

    void dump_device_slot_occupants() const noexcept {
        std::fprintf(stderr,
                     "[state-store] dump: free_device=%u free_object=%u device_capacity=%zu\n",
                     free_device_count_, free_object_count_, objects_.size());
        for (std::size_t i = 0; i < objects_.size(); ++i) {
            const Object& obj = objects_[i];
            if (obj.role == StateImageRole::Free) { continue; }
            std::fprintf(stderr,
                         "[state-store]   #%zu role=%d dev=%d host=%d pend_dev=%d pend_host=%d "
                         "xfer=%llu ckpt_refs=%u src_pins=%u dst_pinned=%d "
                         "endpoint=%d evictable=%d\n",
                         i, static_cast<int>(obj.role),
                         obj.device_slot ? *obj.device_slot : -1,
                         obj.host_slot ? static_cast<int>(obj.host_slot->index) : -1,
                         obj.pending_device_slot ? *obj.pending_device_slot : -1,
                         obj.pending_host_slot ? static_cast<int>(obj.pending_host_slot->index)
                                               : -1,
                         static_cast<unsigned long long>(obj.transfer_id),
                         obj.checkpoint_references, obj.source_pins,
                         obj.destination_pinned ? 1 : 0, obj.is_endpoint ? 1 : 0,
                         (obj.role == StateImageRole::CheckpointImmutable && obj.device_slot &&
                          obj.host_slot && obj.source_pins == 0 && !obj.destination_pinned &&
                          !has_pending_replica(obj))
                             ? 1
                             : 0);
        }
    }

    void return_device_slot(std::int32_t slot) noexcept {
        free_device_slots_[free_device_count_++] = slot;
    }

    [[nodiscard]] std::uint64_t next_epoch() noexcept {
        if (++next_content_epoch_ == 0) { ++next_content_epoch_; }
        return next_content_epoch_;
    }

    [[nodiscard]] std::uint64_t next_transfer() noexcept {
        if (++next_transfer_id_ == 0) { ++next_transfer_id_; }
        return next_transfer_id_;
    }

    [[nodiscard]] Object& require(StateImageHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    [[nodiscard]] const Object& require(StateImageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    [[nodiscard]] bool valid_transfer(const StateImageTransfer& transfer) const noexcept {
        if (transfer.owner_ != this || transfer.id_ == 0 || !valid(transfer.source_) ||
            !valid(transfer.destination_)) {
            return false;
        }
        const Object& destination = objects_[transfer.destination_.index_];
        return destination.transfer_id == transfer.id_;
    }

    void validate_transfer(const StateImageTransfer& transfer) const {
        if (!valid_transfer(transfer)) {
            throw std::logic_error("StateImage transfer capability is stale");
        }
    }

    static void consume(StateImageTransfer& transfer) noexcept {
        transfer.owner_ = nullptr;
        transfer.id_    = 0;
    }

    qwen3_6::StateImageDevicePool* device_ = nullptr;
    qwen3_6::HostStatePool* host_          = nullptr;
    std::vector<Object> objects_;
    std::vector<std::uint32_t> free_objects_;
    std::vector<std::int32_t> free_device_slots_;
    std::uint32_t free_object_count_  = 0;
    std::uint32_t free_device_count_  = 0;
    std::uint64_t next_content_epoch_ = 0;
    std::uint64_t next_transfer_id_   = 0;
};

inline StateImageTransfer::~StateImageTransfer() {
    if (owner_ != nullptr) { owner_->abort_transfer(std::move(*this)); }
}

enum class StateReadOwnership : std::uint8_t {
    // The primary binding owns the direct StateImage lifetime.
    Primary,
    // The same active lineage retains the source through an optional checkpoint reference.
    LineageCheckpoint,
    // A retained private or shared owner outside this active lineage retains the source.
    ExternalOwner,
};

struct ActiveStateBinding {
    StateImageHandle read;
    StateImageHandle write;
    bool fork_pending                 = false;
    StateReadOwnership read_ownership = StateReadOwnership::Primary;

    // A non-primary read source is pinned only for the pending Fork. Its allocation belongs to a
    // surviving same-lineage checkpoint or external owner, never to the primary binding.
    [[nodiscard]] bool borrows_read() const noexcept {
        return read_ownership != StateReadOwnership::Primary;
    }

    // An external owner can be a retained private endpoint whose lifetime has no StateImage
    // checkpoint reference, so refcount zero alone does not authorize this sequence to release it.
    [[nodiscard]] bool read_has_external_owner() const noexcept {
        return read_ownership == StateReadOwnership::ExternalOwner;
    }
};

} // namespace ninfer::targets::qwen3_6::detail

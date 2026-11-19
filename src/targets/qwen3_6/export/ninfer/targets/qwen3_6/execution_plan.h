#pragma once

// Compile-time physical-layer -> execution-slot plan for the Qwen3.5-style hybrid family.
//
// The family owns one set of physical weights but may execute an ordered sequence of slots that
// revisits physical layers (SenseNova-Loop). Weights are indexed by physical layer; KV planes,
// recurrent state, checkpoints, graph state and the layer loop are indexed by slot. For ordinary
// variants the plan is the identity.

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6 {

template <std::int32_t Interval>
[[nodiscard]] constexpr bool is_full_attention_physical(std::int32_t physical) noexcept {
    return (physical + 1) % Interval == 0;
}

template <std::int32_t PhysicalLayers>
[[nodiscard]] constexpr std::array<std::int32_t, PhysicalLayers> identity_slot_physical() noexcept {
    std::array<std::int32_t, PhysicalLayers> slots{};
    for (std::int32_t index = 0; index < PhysicalLayers; ++index) { slots[index] = index; }
    return slots;
}

// Physical layers [Start,End) are executed Count additional times. The first visit stays in the
// identity position; the remaining visits are appended after the physical prefix in order.
template <std::int32_t PhysicalLayers, std::int32_t Start, std::int32_t End, std::int32_t Count>
[[nodiscard]] constexpr auto looped_slot_physical() noexcept {
    static_assert(Start >= 0 && Start < End && End <= PhysicalLayers);
    static_assert(Count >= 1);
    constexpr std::int32_t kLoopLayers = End - Start;
    constexpr std::int32_t kSlots      = PhysicalLayers + (Count - 1) * kLoopLayers;
    std::array<std::int32_t, kSlots> slots{};
    std::int32_t cursor = 0;
    for (std::int32_t physical = 0; physical < Start; ++physical) {
        slots[cursor++] = physical;
    }
    for (std::int32_t visit = 0; visit < Count; ++visit) {
        for (std::int32_t physical = Start; physical < End; ++physical) {
            slots[cursor++] = physical;
        }
    }
    for (std::int32_t physical = End; physical < PhysicalLayers; ++physical) {
        slots[cursor++] = physical;
    }
    return slots;
}

template <std::int32_t PhysicalLayers, std::int32_t Slots, std::int32_t PhysicalFull,
          std::int32_t PhysicalGdn, std::array<std::int32_t, Slots> SlotPhysical,
          std::int32_t Interval = 4>
struct ExecutionPlan {
    static_assert(PhysicalLayers > 0 && Slots > 0);
    static_assert(PhysicalFull > 0 && PhysicalGdn > 0);
    static_assert(PhysicalFull + PhysicalGdn == PhysicalLayers);
    static_assert(PhysicalFull == (PhysicalLayers / Interval));

    static constexpr std::int32_t physical_layers = PhysicalLayers;
    static constexpr std::int32_t slots           = Slots;
    static constexpr std::int32_t physical_full   = PhysicalFull;
    static constexpr std::int32_t physical_gdn    = PhysicalGdn;
    static constexpr std::int32_t interval        = Interval;
    static constexpr std::array<std::int32_t, Slots> slot_physical = SlotPhysical;

    [[nodiscard]] static constexpr std::int32_t slot_physical_layer(std::int32_t slot) noexcept {
        return slot_physical[static_cast<std::size_t>(slot)];
    }

    [[nodiscard]] static constexpr bool physical_is_full(std::int32_t physical) noexcept {
        return is_full_attention_physical<Interval>(physical);
    }

    [[nodiscard]] static constexpr bool slot_is_full(std::int32_t slot) noexcept {
        return physical_is_full(slot_physical_layer(slot));
    }

    [[nodiscard]] static constexpr std::int32_t physical_full_index(
        std::int32_t physical) noexcept {
        std::int32_t index = 0;
        for (std::int32_t layer = 0; layer < physical; ++layer) {
            index += physical_is_full(layer) ? 1 : 0;
        }
        return index;
    }

    [[nodiscard]] static constexpr std::int32_t physical_gdn_index(
        std::int32_t physical) noexcept {
        std::int32_t index = 0;
        for (std::int32_t layer = 0; layer < physical; ++layer) {
            index += physical_is_full(layer) ? 0 : 1;
        }
        return index;
    }

    [[nodiscard]] static constexpr std::int32_t slot_full_index(std::int32_t slot) noexcept {
        std::int32_t index = 0;
        for (std::int32_t visit = 0; visit < slot; ++visit) {
            index += slot_is_full(visit) ? 1 : 0;
        }
        return index;
    }

    [[nodiscard]] static constexpr std::int32_t slot_gdn_index(std::int32_t slot) noexcept {
        std::int32_t index = 0;
        for (std::int32_t visit = 0; visit < slot; ++visit) {
            index += slot_is_full(visit) ? 0 : 1;
        }
        return index;
    }

    static constexpr std::int32_t full_slots = [] {
        std::int32_t count = 0;
        for (const std::int32_t physical : slot_physical) {
            count += physical_is_full(physical) ? 1 : 0;
        }
        return count;
    }();

    static constexpr std::int32_t gdn_slots = Slots - full_slots;
};

} // namespace ninfer::targets::qwen3_6

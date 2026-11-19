#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_cost.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/sensenova_loop_9b/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
using SenseNovaLoop9B = sensenova_loop_9b::Package;

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan, DeviceContext& device,
                        const StartupObserver& startup_observer);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    LoadedQwen3_6_35BA3B(std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           Qwen3_6_35BA3B::SequencePlan sequence_plan, DeviceContext& device,
                           const StartupObserver& startup_observer);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};


struct LoadedSenseNovaLoop9B {
    std::unique_ptr<SenseNovaLoop9B::LoadedModel> model;
    SenseNovaLoop9B::Frontend frontend;

    LoadedSenseNovaLoop9B(std::unique_ptr<SenseNovaLoop9B::LoadedModel> stable_model,
                          const EngineOptions& options);
    ~LoadedSenseNovaLoop9B();

    LoadedSenseNovaLoop9B(const LoadedSenseNovaLoop9B&)            = delete;
    LoadedSenseNovaLoop9B& operator=(const LoadedSenseNovaLoop9B&) = delete;
};

struct SenseNovaLoop9BInstance {
    using Package = SenseNovaLoop9B;

    std::unique_ptr<LoadedSenseNovaLoop9B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<SenseNovaLoop9B::Program> program;

    SenseNovaLoop9BInstance(std::unique_ptr<LoadedSenseNovaLoop9B> stable_loaded,
                            runtime::KvCapacityResolution resolution,
                            SenseNovaLoop9B::SequencePlan sequence_plan, DeviceContext& device,
                            const StartupObserver& startup_observer);
    ~SenseNovaLoop9BInstance();

    SenseNovaLoop9BInstance(const SenseNovaLoop9BInstance&)            = delete;
    SenseNovaLoop9BInstance& operator=(const SenseNovaLoop9BInstance&) = delete;
};

using ActiveTarget =
    std::variant<std::unique_ptr<Qwen3_6_27BInstance>, std::unique_ptr<Qwen3_6_35BA3BInstance>,
                 std::unique_ptr<SenseNovaLoop9BInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    runtime::ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer

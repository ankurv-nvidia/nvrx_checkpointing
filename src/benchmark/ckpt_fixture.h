#pragma once

#include "byte_storage.h"
#include "state.h"
#include "tensor.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

using FixtureFunc = void (*)(StateTree& tree, std::filesystem::path const& ckptDir);

struct CheckpointTestContext
{
    bool                                           outofproc{false};
    uint16_t                                       nodes{};
    std::unique_ptr<StateTree>                     globalRegistry;
    std::unordered_set<std::unique_ptr<StateTree>> localRegistries;

    std::mutex              mutex;
    std::atomic<bool>       stopRequested{false};
    uint32_t                completedRanks{0};
    Rank                    worldSize{0};
    FixtureFunc             fixtureFunc{nullptr};
    std::filesystem::path   ckptDir;
    std::exception_ptr      workerException;    // first exception captured from any worker
    std::condition_variable cv;
};

// Walks the registry once (with placeholder rank/worldSize) to find the maximum
// rank present in any ShardedTensor node, then returns that value + 1 as worldSize.
Rank InferWorldSize(StateTree const& tree);

// Builds a per-rank StateTree from `ctx.globalRegistry`, allocating zero-filled
// device memory and tiling each shard with its FQN as the test pattern.
std::unique_ptr<StateTree> BuildPerRankStateTree(CheckpointTestContext& ctx, RankTopology topo);

// Loads a serialized state registry from `path` and stores it in `ctx.globalRegistry`,
// then sets `ctx.worldSize` via InferWorldSize and populates `ctx.localRegistries`.
void LoadStateRegistryFromPath(CheckpointTestContext& ctx, std::filesystem::path const& path);

// Per-rank driver loop. Waits on `ctx.fixtureFunc`; calls it on each iteration; bumps
// `ctx.completedRanks` and notifies once all ranks have finished. Exits when `ctx.stopRequested`.
void RunCheckpointClient(CheckpointTestContext& ctx, StateTree& tree);

// Walks `localState` and verifies each shard's bytes in `blob` still match the FQN
// pattern that BuildPerRankStateTree wrote. Throws std::runtime_error on mismatch.
void ValidateRoundTrip(StateTree const& localState, ByteStorage const& tensorStorage);

// Drives a save: SaveCheckpoint -> WaitForCheckpointSave. No validation — used by
// the benchmark timing path so the round-trip walk doesn't dominate the measurement.
void SaveCheckpointOnly(StateTree& tree, std::filesystem::path const& ckptDir);

// SaveCheckpointOnly + a QueryServerSessions-driven ValidateRoundTrip for this rank's
// session. Used by the Catch2 test. Throws std::runtime_error on round-trip mismatch.
void SaveCheckpointAndValidate(StateTree& tree, std::filesystem::path const& ckptDir);

// Spins up one driver thread per rank in `ctx.localRegistries`, each running
// RunCheckpointClient. Caller must later call JoinClientThreads.
std::vector<std::thread> StartClientThreads(CheckpointTestContext& ctx);

// Sets stopRequested, notifies all waiters, joins every thread.
void JoinClientThreads(CheckpointTestContext& ctx, std::vector<std::thread>& tasks);

// Synchronously runs `func` on every driver thread: sets ctx.fixtureFunc / ctx.ckptDir,
// notifies, then waits for all ranks to finish.
void RunOnAllRanks(CheckpointTestContext& ctx, FixtureFunc func, std::filesystem::path const& ckptDir);

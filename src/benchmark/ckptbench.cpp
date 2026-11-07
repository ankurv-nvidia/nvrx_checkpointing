#include "CommonMacros.h"
#include "checkpoint.h"
#include "ckpt_fixture.h"
#include "server.h"

#include <benchmark/benchmark.h>
#include <boost/program_options.hpp>

#include <fmt/format.h>
#include <fmt/std.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace bpo = boost::program_options;

namespace
{
SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
CheckpointTestContext Ctx;                         // NOLINT
bool                  CheckpointsSaved = false;    // NOLINT
SUPPRESS_WARNINGS_END

constexpr std::string_view KDefaultCkptDir = "/tmp/checkpoint_bench";

void BM_SaveCheckpoint(benchmark::State& benchState)    // NOLINT(readability-identifier-naming,llvm-prefer-static-over-anonymous-namespace)
{
    for (auto _ : benchState)
    {
        RunOnAllRanks(Ctx, &SaveCheckpointOnly, KDefaultCkptDir);
        CheckpointsSaved = true;
    }
}

SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
BENCHMARK(BM_SaveCheckpoint)->Unit(benchmark::kMicrosecond)->Iterations(1);
SUPPRESS_WARNINGS_END
}    // namespace

int main(int argc, char** argv)
try
{
    bpo::options_description desc("Allowed options");
    std::filesystem::path    stateRegistry;

    desc.add_options()("help", "produce help message")                                                         //
        ("ranks-per-node", bpo::value<uint16_t>(&Ctx.ranksPerNode), "number of ranks per node")                //
        ("outproc", bpo::bool_switch(&Ctx.outofproc), "run the checkpoint server out-of-process")              //
        ("stateregistry", bpo::value<std::filesystem::path>(&stateRegistry), "path to the state registry");    //

    bpo::variables_map vm;
    auto               parsed = bpo::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    bpo::store(parsed, vm);
    bpo::notify(vm);

    auto                     unrecognized = bpo::collect_unrecognized(parsed.options, bpo::include_positional);
    std::vector<std::string> benchArgStrings;
    benchArgStrings.reserve(1 + unrecognized.size());
    benchArgStrings.emplace_back(argv[0]);
    benchArgStrings.insert(benchArgStrings.end(), unrecognized.begin(), unrecognized.end());

    if (vm.contains("help"))
    {
        desc.print(std::cerr);
        std::cerr << '\n';
        int   argcForHelp   = 1;
        char* argvForHelp[] = {argv[0], const_cast<char*>("--help")};    // NOLINT
        benchmark::Initialize(&argcForHelp, argvForHelp);                // NOLINT
        return 1;
    }

    if (!vm.contains("stateregistry")) { throw std::runtime_error("State registry path must be provided with --stateregistry"); }

    LoadStateRegistryFromPath(Ctx, stateRegistry);

    std::vector<std::unique_ptr<ServiceReference>> references;
    if (Ctx.outofproc)
    {
        references.reserve(Ctx.localRegistries.size());
        for (auto const& tree : Ctx.localRegistries)
        {
            references.push_back(std::make_unique<ServiceReference>(tree->SessionId(), tree->Topo()));
        }
    }

    auto tasks = StartClientThreads(Ctx);

    std::vector<char*> benchArgv;
    benchArgv.reserve(benchArgStrings.size());
    for (auto& s : benchArgStrings) { benchArgv.push_back(s.data()); }
    int benchArgc = static_cast<int>(benchArgv.size());
    benchmark::Initialize(&benchArgc, benchArgv.data());
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    JoinClientThreads(Ctx, tasks);
    if (CheckpointsSaved)
    {
        for (uint32_t snapshot = 1; snapshot <= NumSnapshots; ++snapshot)
        {
            WaitForCkptFinalize(static_cast<uint32_t>(Ctx.globalRegistry->SessionId()), snapshot);
        }
    }
    if (!Ctx.outofproc)
    {
        // Keep disk I/O outside BM_SaveCheckpoint while ensuring the in-process
        // daemon is not torn down with background DCP writes still in flight.
        WaitForInprocServerWrites();
        StopInprocServer();
    }

    return 0;
} catch (std::exception const& e)
{
    if (!Ctx.outofproc) { StopInprocServer(); }
    fmt::print(stderr, "Exception: {}\n", e.what());
    return 1;
}

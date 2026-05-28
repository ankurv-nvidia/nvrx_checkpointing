#include "SetThreadName.h"
#include "blob.h"
#include "messages.h"

#include <boost/asio/io_context.hpp>
#include <boost/process.hpp>
#include <boost/process/v2/popen.hpp>
#include <boost/program_options.hpp>

#include <fmt/std.h>

#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string_view>

namespace bpo = boost::program_options;
using namespace checkpoint::ipc;
static constexpr char const* QueueName          = "shm_ipc_benchmark";
static constexpr size_t      DefaultIter        = 128;
static constexpr size_t      DefaultMessageSize = D2HBufferSize;
static constexpr size_t      DefaultRanks       = 2;

static std::atomic<PerfMetrics::Clock::time_point> StartTime{};    // NOLINT
static std::atomic<PerfMetrics::Clock::duration>   Duration{};     // NOLINT
using namespace std::chrono_literals;

template <> struct fmt::formatter<Message::LocalStateData> : fmt::formatter<std::string_view>
{
    auto format(Message::LocalStateData const& obj, fmt::format_context& ctx) const    // NOLINT
    { return fmt::format_to(ctx.out(), "(sessionId={}, messageSize={}, offset={})", obj.sessionId, obj.messageSize, obj.offset); }
};
static void Server(size_t ranks, size_t msgCount, size_t memorySize, bool validate, bool debug)
{
    SetThreadName("ncs:server");
    if (debug) { fmt::print("[s] starting\n"); }
    auto msgQueue = MsgQueue::Create(QueueName);
    if (debug) { fmt::print("[s] created queue {}\n", QueueName); }
    auto totalBytes    = msgCount * memorySize * ranks;
    auto totalMessages = totalBytes / D2HBufferSize;

    Blob buffer(D2HBufferSize * D2HBufferCount);

    for (size_t i = 0; i < totalBytes; i += D2HBufferSize)
    {
        auto msgIndex                        = i / D2HBufferSize;
        auto const [stopRequested, hdr, buf] = msgQueue.BeginRead();
        if (stopRequested) { throw std::runtime_error("Stop requested"); }
        auto  hdr1    = std::get<Message::LocalStateData>(hdr);
        auto* destptr = buffer[i % buffer.size];
        if (debug) { fmt::print("[s] recv msg[{}/{}] {} msgQ = {}\n", msgIndex, totalMessages, hdr1, msgQueue); }
        std::memcpy(destptr, buf.data(), hdr1.messageSize);
        if (validate)
        {
            if (hdr.index() != static_cast<size_t>(Message::Type::LocalStateData))
            {
                throw std::runtime_error(fmt::format("Validation failed for msg {}. Expected message type {} but got {}",
                                                     msgIndex,
                                                     static_cast<uint32_t>(Message::Type::LocalStateData),
                                                     hdr.index()));
            }
            if (hdr1.messageSize != D2HBufferSize)
            {
                throw std::runtime_error(fmt::format(
                    "Validation failed for msg {}. Expected message size {} but got {}", msgIndex, D2HBufferSize, hdr1.messageSize));
            }
            for (size_t j = 0; j < hdr1.messageSize; j++)
            {
                auto expectedData = static_cast<uint8_t>(hdr1.globalRank);
                // = ((ranks * (hdr1.partitionStart + j)) +) % std::numeric_limits<uint8_t>::max();
                auto data = (reinterpret_cast<uint8_t const*>(destptr))[j];    // NOLINT
                if (data != expectedData) { throw std::runtime_error(fmt::format("Validation failed at index {}", j)); }
            }
        }
        msgQueue.EndRead();
    }
    fmt::print("[s] finished {}\n", msgQueue);
    msgQueue.Stop();
}

static void Client(uint32_t rank, uint32_t /* worldSize */, size_t msgCount, size_t memorySize, bool debug)
{
    std::string threadName = fmt::format("nv:client{}", rank);
    SetThreadName(threadName.c_str());

    if (memorySize % D2HBufferSize != 0) { throw std::runtime_error("Memory size must be a multiple of D2H buffer size"); }

    if (debug) { fmt::print("[c{}] starting\n", rank); }
    auto msgQueue = MsgQueue::Open(QueueName);
    if (debug) { fmt::print("[c{}] opened queue\n", rank); }

    // auto data = std::vector<size_t>(std::from_range, std::views::iota(size_t{0}, memorySize / sizeof(size_t)));
    std::vector<uint8_t> data;
    data.reserve(memorySize);
    auto totalBytes    = msgCount * memorySize;
    auto totalMessages = totalBytes / D2HBufferSize;

    for (size_t j = 0; j < memorySize; ++j) { data.push_back(static_cast<uint8_t>(rank)); }
    StartTime = PerfMetrics::Clock::now();

    for (size_t i = 0; i < totalBytes; i += D2HBufferSize)
    {
        auto  msgIndex   = i / D2HBufferSize;
        auto* srcptr     = data.data() + (i % memorySize);
        auto [slot, buf] = msgQueue.AllocSlot(Message::LocalStateData{.sessionId   = SessionId{0},
                                                                      .snapshotId  = SnapshotId{0},
                                                                      .globalRank  = Rank{rank},
                                                                      .messageSize = D2HBufferSize,
                                                                      .offset      = i * memorySize});

        std::memcpy(buf, srcptr, D2HBufferSize);
        msgQueue.CommitSlot(slot);
        if (debug) { fmt::print("[c{}] sent msg[{}/{}] {} msgQ = {}\n", rank, msgIndex, totalMessages, slot, msgQueue); }
    }

    auto stopTime    = PerfMetrics::Clock::now();
    Duration         = std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime - StartTime.load());
    double bandwidth = (static_cast<double>(msgCount * memorySize) * 1e9 / 1024 / 1024 / 1024)
                       / static_cast<double>(Duration.load().count());    // NOLINT
    fmt::print("[c{}] finished sending {} messages in {}ms Bandwidth = {:.4f}GB/s. msgQ = {}\n",
               rank,
               msgCount,
               std::chrono::duration_cast<std::chrono::microseconds>(Duration.load()),
               bandwidth,
               msgQueue);
}

static void MemcpyBenchmark(size_t ranks, size_t msgCount, size_t memorySize)
{
    auto bufSize = D2HBufferSize / 64;
    Blob src{bufSize};
    Blob dst{bufSize};
    for (size_t j = 0; j < bufSize; ++j) { src.At<uint8_t>(j) = 1u; }

    auto totalBytes    = msgCount * memorySize * ranks;
    StartTime          = PerfMetrics::Clock::now();
    auto totalMessages = totalBytes / D2HBufferSize;

    for (size_t i = 0; i < totalBytes; i += bufSize) { memcpy(dst.data, src.data, bufSize); }

    Duration = std::chrono::duration_cast<std::chrono::nanoseconds>(PerfMetrics::Clock::now() - StartTime.load());
    double bandwidth
        = (static_cast<double>(totalBytes) * 1e9 / 1024 / 1024 / 1024) / static_cast<double>(Duration.load().count());    // NOLINT

    fmt::print("Baseline memcpy Finished ranks={} msgcount={} msgsize={}  {} memcpy of size {} bytes in {}us Bandwidth = {:.4f}GB/s\n",
               ranks,
               msgCount,
               memorySize,
               totalMessages,
               totalBytes,
               std::chrono::duration_cast<std::chrono::microseconds>(Duration.load()),
               bandwidth);
}

static void MemcpyBounceBufferBenchmark(size_t ranks, size_t msgCount, size_t memorySize)
{
    Blob dst{memorySize};
    Blob src{memorySize};
    Blob buf{D2HBufferSize * D2HBufferCount};

    for (size_t j = 0; j < memorySize; ++j) { src.At<uint8_t>(j) = 1u; }

    auto totalBytes    = msgCount * memorySize;
    auto totalMessages = totalBytes / D2HBufferSize;

    StartTime = PerfMetrics::Clock::now();
    for (size_t rank = 0; rank < ranks; rank++)
    {
        uint64_t currentSlot = 0;
        for (size_t i = 0; i < totalBytes; i += D2HBufferSize, currentSlot++)
        {
            auto  slot    = currentSlot % D2HBufferCount;
            auto* slotptr = buf[slot * D2HBufferSize];
            auto* srcptr  = src[i % memorySize];
            auto* dstptr  = dst[i % memorySize];
            std::memcpy(slotptr, srcptr, D2HBufferSize);
            std::memcpy(dstptr, slotptr, D2HBufferSize);
            //  buffers.push_back(std::move(buffer));
            //  if (buffers.size() == D2HBufferCount) { buffers.clear(); }
        }
    }

    Duration = std::chrono::duration_cast<std::chrono::nanoseconds>(PerfMetrics::Clock::now() - StartTime.load());
    double bandwidth
        = (static_cast<double>(totalBytes * ranks) * 1e9 / 1024 / 1024 / 1024) / static_cast<double>(Duration.load().count());    // NOLINT

    fmt::print("Baseline memcpy bouncebuffer Finished {} memcpy of size {} bytes in {}us Bandwidth = {:.4f}GB/s\n",
               totalMessages * ranks,
               totalBytes * ranks,
               std::chrono::duration_cast<std::chrono::microseconds>(Duration.load()),
               bandwidth);
}

int main(int argc, char* argv[])
try
{
    bpo::options_description desc("Allowed options");
    bool                     client       = false;
    bool                     server       = false;
    bool                     launchclient = false;
    bool                     launchserver = false;
    bool                     validate     = false;
    bool                     debug        = false;
    bool                     nocloneproc  = false;
    size_t                   ranks        = DefaultRanks;
    size_t                   iter         = DefaultIter;
    size_t                   size         = DefaultMessageSize * 8;

    desc.add_options()("help", "produce help message")                              //
        ("client", bpo::bool_switch(&client), "run as client")                      //
        ("server", bpo::bool_switch(&server), "run as server")                      //
        ("validate", bpo::bool_switch(&validate), "validate messages")              //
        ("debug", bpo::bool_switch(&debug), "enable debug logging")                 //
        ("iter", bpo::value<size_t>(&iter), "number of iterations")                 //
        ("size", bpo::value<size_t>(&size), "Transfer Chunk Size")                  //
        ("nocloneproc", bpo::bool_switch(&nocloneproc), "disable clone process")    //
        ("ranks", bpo::value<size_t>(&ranks), "number of ranks");                   //

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    if (vm.contains("help"))
    {
        desc.print(std::cerr);
        return 1;
    }

    MemcpyBenchmark(ranks, iter, size);
    MemcpyBounceBufferBenchmark(ranks, iter, size);

    std::vector<std::thread> tasks;

    if (server && client)
    {
        // Both are inproc
    }
    else
    {
        if (server && !nocloneproc) { launchclient = true; }
        else if (client && !nocloneproc) { launchserver = true; }
        else if (!server && !client && !nocloneproc)
        {
            // default no args scenario.
            client       = true;
            launchserver = true;
        }
    }

    if (server)
    {
        /*
        auto threads = std::views::iota(size_t{0}, ranks) | std::views::transform([&](size_t rank) {
                           return std::async(std::launch::async | std::launch::deferred, Server, rank, iter, size, validate);
                       })
                       | std::ranges::to<std::vector>();
        */
        tasks.emplace_back(Server, ranks, iter, size, validate, debug);
#ifdef LAUNCH_FROM_SERVER
        if (!server && !client) { child.wait(); }
#endif
    }

    if (client)
    {
        /*
        auto threads = std::views::iota(size_t{0}, ranks) | std::views::transform([&](size_t rank) {
                           return std::async(std::launch::async | std::launch::deferred, Client, rank, iter, size);
                       })
                       | std::ranges::to<std::vector>();
        */
        if (debug) { fmt::print("Launching {} client threads\n", ranks); }
        tasks.reserve(ranks);
        for (size_t rank = 0; rank < ranks; ++rank) { tasks.emplace_back(Client, rank, ranks, iter, size, debug); }
    }

    if (launchclient || launchserver)
    {
        tasks.emplace_back([&]() {
            SetThreadName(launchclient ? "ClientLauncher" : "ServerLauncher");
            if (debug) { fmt::print("Launching {} thread\n", launchclient ? "client" : "server"); }

            boost::asio::io_context ctx;
            boost::process::process child(ctx,
                                          argv[0],    // NOLINT
                                          {launchclient ? "--client" : "--server",
                                           "--nocloneproc",
                                           fmt::format("--ranks={}", ranks),
                                           fmt::format("--iter={}", iter),
                                           fmt::format("--size={}", size),
                                           fmt::format("{}", validate ? "--validate" : ""),
                                           fmt::format("{}", debug ? "--debug" : "")},
                                          boost::process::process_stdio{.in = {}, .out = stdout, .err = stderr});
            child.wait();
        });
    }

    for (auto& t : tasks) { t.join(); }
    return 0;
} catch (std::exception const& e)
{
    fmt::print(stderr, "Exception: {}\n", e.what());
    return 1;
}

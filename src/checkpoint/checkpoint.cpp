#include "CommonMacros.h"
#include "SetThreadName.h"

#include "checkpoint.h"
#include "config.h"
#include "messages.h"
#include "numa_utils.h"    // IWYU pragma: keep
#include "service_client.h"
#include "state.h"
#include "stateserdes.h"
#include "tensor.h"

#include <boost/asio/read_until.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/empty_body.hpp>
#include <boost/beast/http/field.hpp>
#include <boost/beast/http/file_body.hpp>
#include <boost/beast/version.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/process.hpp>
#include <boost/process/execute.hpp>
#include <hwloc.h>

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>

using namespace std::chrono_literals;
namespace cipc = checkpoint::ipc;

template <typename TStream> static void TryCleanShutdown(TStream& stream)
{
    boost::system::error_code ec;
    ec = stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    if (ec && ec != boost::system::errc::not_connected)
    {    //
        fmt::print(stderr, "[native_ckpt:cli][ERROR] Shutdown Error: {}\n", ec.what());
    }
}

void SvcPing()
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::mutex PingMutex;
    SUPPRESS_WARNINGS_END

    std::scoped_lock lock(PingMutex);
    auto             selfLoc   = boost::dll::this_line_location();
    auto             svcExeLoc = selfLoc.parent_path() / "checkpoint_server";
    if (TryConnectToCheckpointService()) { return; }
    if (checkpoint::config::GetConfigValue<checkpoint::config::SvcDisableAutoLaunch>())
    {
        throw std::runtime_error("Externally managed checkpoint service is unavailable");
    }

    try
    {
        boost::asio::io_context    ctx;
        boost::asio::readable_pipe output(ctx);
        boost::process::process    proc(ctx, svcExeLoc, {}, boost::process::process_stdio{.in = {}, .out = output, .err = stderr});
        std::string                line;
        boost::system::error_code  ec;
        boost::asio::read_until(output, boost::asio::dynamic_buffer(line), '\n', ec);
        proc.detach();
    } catch (std::exception const& ex)
    {
        fmt::print(stderr,
                   "[native_ckpt:cli][ERROR] Exception in SvcPing from {} svcbin={}: {}\n",
                   selfLoc.string(),
                   svcExeLoc.string(),
                   ex.what());
        throw;
    }

    // Both the elected daemon and losing candidates write READY only after the
    // elected service answers a ping. The two short attempts inside TryConnect
    // cover a final scheduling edge without a second readiness policy here.
    if (!TryConnectToCheckpointService()) { throw std::runtime_error("Failed to start or connect to checkpoint service"); }
}

static void SvcRegisterLocalState(std::shared_ptr<StateTree> registry)
{
    SvcPing();

    auto const topo        = registry->Topo();
    auto const serverIndex = (static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer) % topo.peersPerNode;
    auto const port        = cipc::Message::Port + serverIndex;
    auto       send        = [port](boost::beast::http::verb method, std::string const& target, std::string const& body) {
        static constexpr int      MaxAttempts = 5;
        boost::system::error_code ec;
        for (int attempt = 0; attempt < MaxAttempts; ++attempt)
        {
            ec.clear();
            boost::asio::io_context        ioc;
            boost::beast::tcp_stream       stream(ioc);
            boost::asio::ip::tcp::resolver resolver(ioc);
            auto                           results = resolver.resolve("localhost", std::to_string(port), ec);
            if (ec) { continue; }
            [[maybe_unused]] auto endpoint = stream.connect(results, ec);
            if (ec) { continue; }

            static constexpr int                                         HttpVersion11 = 11;
            boost::beast::http::request<boost::beast::http::string_body> req{method, target, HttpVersion11};
            req.set(boost::beast::http::field::host, fmt::format("{}:{}", "localhost", port));
            req.set(boost::beast::http::field::content_type, "application/octet-stream");
            req.body() = body;
            req.prepare_payload();
            boost::beast::http::write(stream, req, ec);
            if (ec) { continue; }

            boost::beast::flat_buffer                                     buffer;
            boost::beast::http::response<boost::beast::http::string_body> res;
            boost::beast::http::read(stream, buffer, res, ec);
            if (ec) { continue; }
            TryCleanShutdown(stream);
            return res;
        }
        throw boost::system::system_error(ec);
    };

    auto requireSuccess = [port](auto const& response, std::string_view operation) {
        if (response.result() != boost::beast::http::status::ok && response.result() != boost::beast::http::status::accepted)
        {
            throw std::runtime_error(
                fmt::format("{} failed on port {}: HTTP {}: {}", operation, port, response.result_int(), response.body()));
        }
    };

    std::stringstream activationStream;
    serdes::WriteBlobWithSize<uint16_t>(activationStream, "");
    StateSerDes::WriteTopology(activationStream, topo);
    auto response
        = send(boost::beast::http::verb::post, fmt::format("/api/activatesession/{}", cipc::SessionId::Default), activationStream.str());
    requireSuccess(response, "session activation");

    std::stringstream registryStream;
    StateSerDes::Write(registryStream, *registry, StateNode::Handle::Root);
    auto registryBody = registryStream.str();
    registry.reset();
    response = send(boost::beast::http::verb::post, fmt::format("/api/localstateregistry/{}", cipc::SessionId::Default), registryBody);
    requireSuccess(response, "local registry upload");
}

static uint8_t ResolveClientQueueCount(RankTopology const& topo)
{
    auto configuredQueueCount = checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferNumQueues>();
    if (configuredQueueCount == 0 && checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferSize>() != 0)
    {
        configuredQueueCount = cipc::ExplicitSizeDefaultQueueCount;
    }
    return cipc::ResolveQueueCount(topo.ranksPerPeer, configuredQueueCount);
}

void SvcStop()
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::mutex PingMutex;
    SUPPRESS_WARNINGS_END

    std::scoped_lock lock(PingMutex);

    if (!TryConnectToCheckpointService()) { return; }

    boost::system::error_code ec;

    boost::asio::io_context        ioc;
    boost::beast::tcp_stream       stream(ioc);
    boost::asio::ip::tcp::resolver resolver(ioc);

    auto results = resolver.resolve("localhost", std::to_string(cipc::Message::Port));
    // Try connecting to localhost before we send a request to check and fail-fast
    [[maybe_unused]] auto endpoint = stream.connect(results, ec);
    if (ec) { return; }

    // Create HTTP GET request with buffer_body (avoids copying state data)
    static constexpr int                                         HttpVersion11 = 11;
    boost::beast::http::request<boost::beast::http::buffer_body> req{boost::beast::http::verb::get, "/api/stop", HttpVersion11};
    req.set(boost::beast::http::field::host, fmt::format("{}:{}", "localhost", cipc::Message::Port));
    req.body().data = nullptr;
    req.body().size = 0;
    req.body().more = false;
    req.prepare_payload();

    // Send the request (Beast sends directly from buffer with Content-Length)
    boost::beast::http::write(stream, req);
    // Read the response
    boost::beast::flat_buffer                                     buffer;
    boost::beast::http::response<boost::beast::http::string_body> res;
    boost::beast::http::read(stream, buffer, res);

    if (res.result() != boost::beast::http::status::ok) { return; }
    TryCleanShutdown(stream);
}

void SvcSpinDown()
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::mutex SpinDownMutex;
    SUPPRESS_WARNINGS_END

    std::scoped_lock lock(SpinDownMutex);

    if (!TryConnectToCheckpointService()) { return; }

    boost::system::error_code ec;

    boost::asio::io_context        ioc;
    boost::beast::tcp_stream       stream(ioc);
    boost::asio::ip::tcp::resolver resolver(ioc);

    auto results = resolver.resolve("localhost", std::to_string(cipc::Message::Port));
    // Fail fast if the service is already gone.
    [[maybe_unused]] auto endpoint = stream.connect(results, ec);
    if (ec) { return; }

    auto target = std::string("/api/spindown");

    static constexpr int                                         HttpVersion11 = 11;
    boost::beast::http::request<boost::beast::http::buffer_body> req{boost::beast::http::verb::get, target, HttpVersion11};
    req.set(boost::beast::http::field::host, fmt::format("{}:{}", "localhost", cipc::Message::Port));
    req.body().data = nullptr;
    req.body().size = 0;
    req.body().more = false;
    req.prepare_payload();

    boost::beast::http::write(stream, req);

    // The server holds this connection open with chunked heartbeats and only completes the
    // response once spin-down finishes. No stream timeout is set, so the read blocks across
    // the heartbeats for the full (possibly long) spin-down rather than timing out.
    boost::beast::flat_buffer                                            buffer;
    boost::beast::http::response_parser<boost::beast::http::string_body> parser;
    parser.body_limit(std::numeric_limits<std::uint64_t>::max());
    boost::beast::http::read(stream, buffer, parser, ec);
    if (ec && ec != boost::beast::http::error::end_of_stream) { return; }

    TryCleanShutdown(stream);
}

void WaitForCkptFinalize(uint32_t sessionId, int64_t iteration)
{
    boost::system::error_code ec;

    boost::asio::io_context        ioc;
    boost::beast::tcp_stream       stream(ioc);
    boost::asio::ip::tcp::resolver resolver(ioc);

    auto results = resolver.resolve("localhost", std::to_string(cipc::Message::Port), ec);
    if (ec) { throw std::runtime_error(fmt::format("wait-for-ckpt-finalize: resolve failed: {}", ec.message())); }
    [[maybe_unused]] auto endpoint = stream.connect(results, ec);
    if (ec) { throw std::runtime_error(fmt::format("wait-for-ckpt-finalize: connect failed: {}", ec.message())); }

    auto                 target        = fmt::format("/api/wait-for-ckpt-finalize/{}/{}", sessionId, iteration);
    static constexpr int HttpVersion11 = 11;
    boost::beast::http::request<boost::beast::http::buffer_body> req{boost::beast::http::verb::get, target, HttpVersion11};
    req.set(boost::beast::http::field::host, fmt::format("{}:{}", "localhost", cipc::Message::Port));
    req.body().data = nullptr;
    req.body().size = 0;
    req.body().more = false;
    req.prepare_payload();
    boost::beast::http::write(stream, req, ec);
    if (ec) { throw std::runtime_error(fmt::format("wait-for-ckpt-finalize: request write failed: {}", ec.message())); }

    // The server holds the connection open with chunked heartbeats and only completes the response
    // once the snapshot is durable. No stream timeout is set, so the read blocks across heartbeats
    // for the full (possibly long) finalize rather than timing out.
    boost::beast::flat_buffer                                            buffer;
    boost::beast::http::response_parser<boost::beast::http::string_body> parser;
    parser.body_limit(std::numeric_limits<std::uint64_t>::max());
    boost::beast::http::read(stream, buffer, parser, ec);
    if (ec && ec != boost::beast::http::error::end_of_stream)
    {
        throw std::runtime_error(fmt::format("wait-for-ckpt-finalize: response read failed: {}", ec.message()));
    }
    if (parser.get().result() != boost::beast::http::status::ok)
    {
        throw std::runtime_error(fmt::format("wait-for-ckpt-finalize: server returned HTTP {}", parser.get().result_int()));
    }
    TryCleanShutdown(stream);
}

namespace
{
struct CheckpointWriter
{
    using Clock     = cipc::PerfMetrics::Clock;
    using TimePoint = Clock::time_point;
    using Duration  = Clock::duration;

    StateTree const& root;    // NOLINT

    // Resolve this rank's channel and mmap its shm queue (cheap; logs the open time).
    static cipc::MsgQueue OpenMsgQueue(RankTopology const& topo)
    {
        auto       localRank  = static_cast<uint32_t>(topo.rank) % topo.ranksPerPeer;
        auto const queueCount = ResolveClientQueueCount(topo);
        // Distribute consecutive local ranks as evenly as possible over the
        // configured queues. Daemon-side sizing uses the same mapping.
        auto const index       = cipc::QueueIndex(localRank, topo.ranksPerPeer, queueCount);
        auto const peerIndex   = (static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer) % topo.peersPerNode;
        auto const channelName = cipc::Message::RequestChannelName(static_cast<uint16_t>(peerIndex), index);
        return cipc::MsgQueue::Open(channelName);
    }

    static uint8_t D2HConcurrencyForQueue(RankTopology const& topo, cipc::MsgQueue const& queue)
    {
        auto const queueCount    = ResolveClientQueueCount(topo);
        auto const localRank     = static_cast<uint32_t>(topo.rank) % topo.ranksPerPeer;
        auto const queueIndex    = cipc::QueueIndex(localRank, topo.ranksPerPeer, queueCount);
        auto const producerCount = cipc::QueueProducerCount(queueIndex, topo.ranksPerPeer, queueCount);
        return cipc::QueueD2HConcurrency(queue.SlotCount(), producerCount);
    }

    // Note: a queue can be shared by multiple ranks
    struct QueueState
    {
        std::unique_ptr<cipc::MsgQueue> queue;
        std::shared_future<void>        ready;
        int                             numaNode{-1};
    };

    struct QueueRegistry
    {
        std::mutex                                            mu;
        std::unordered_map<Rank, std::shared_ptr<QueueState>> states;
    };
    static QueueRegistry& Registry()
    {
        SUPPRESS_WARNINGS_START
        SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
        static QueueRegistry Registry;
        SUPPRESS_WARNINGS_END
        return Registry;
    }

    static void SendStateRegistry(StateTree const& root)
    {
        auto const& topo = root.Topo();
        if (topo.ranksPerPeer == 0 || topo.peersPerNode == 0) { throw std::invalid_argument("peer topology must be nonzero"); }
        auto&            reg = Registry();
        std::scoped_lock lock(reg.mu);
        if (reg.states.contains(topo.rank)) { return; }
        auto state         = std::make_shared<QueueState>();
        auto readyP        = std::make_shared<std::promise<void>>();
        auto registryClone = root.Clone();
        state->ready       = readyP->get_future().share();
        reg.states.emplace(topo.rank, state);
        int numa = -1;
#ifdef PHOENIX_HAVE_CUDA
        // Need to determine numa for current device since once cudaHostRegister has faulted+pinned the pages, a later mbind is a no-op
        // pinned pages won't migrate.
        numa = DeviceContext::CudaGetCurrentDeviceNumaNode();
#endif
        state->numaNode = numa;
        std::thread([registryCopy = std::move(registryClone), numa, state, readyP] {
            SetThreadName("ncc:qinit");
            try
            {
                auto const queueTopo = registryCopy->Topo();
                SvcRegisterLocalState(registryCopy);
                state->queue = std::make_unique<cipc::MsgQueue>(OpenMsgQueue(queueTopo));
#ifdef PHOENIX_HAVE_CUDA
                auto region = state->queue->GetShmDataRegion();
                BindRegionToNumaNode(numa, region.data(), region.size());
                DeviceContext::CudaRegisterHostBuffer(region.data(), region.size());
#else
                (void)numa;
#endif
            } catch (...)
            {
                readyP->set_exception(std::current_exception());
                return;
            }
            readyP->set_value();
        }).detach();
    }

    static std::shared_ptr<QueueState> GetQueueState(StateTree const& root)
    {
        std::shared_ptr<QueueState> state;
        {
            auto&            reg = Registry();
            std::scoped_lock lock(reg.mu);
            state = reg.states.at(root.Topo().rank);
        }
        state->ready.get();    // rethrows if queue initialization failed
        return state;
    }

    static cipc::MsgQueue& GetMsgQueue(RankTopology const& topo)
    {
        std::shared_ptr<QueueState> state;
        {
            auto&            reg = Registry();
            std::scoped_lock lock(reg.mu);
            auto const       it = reg.states.find(topo.rank);
            if (it == reg.states.end()) { throw std::logic_error("checkpoint queue was not initialized from a state registry"); }
            state = it->second;
        }
        state->ready.get();
        return *state->queue;
    }

    std::shared_ptr<QueueState> queueState;
    cipc::MsgQueue&             msgQueue;
    int const                   numaNode;
    size_t const                slotSize;    // runtime shm slot capacity, set by the server
    uint8_t const               d2hConcurrency;
    std::atomic<uint32_t>       totalD2HRequests{0};
    std::atomic<uint32_t>       smallD2HRequests{0};
    std::atomic<uint64_t>       totalWrittenBytes{0};

    TimePoint const start = Clock::now();

    explicit CheckpointWriter(StateTree const& rootIn LFTBND) :
        root(rootIn),
        queueState(GetQueueState(root)),
        msgQueue(*queueState->queue),
        numaNode(queueState->numaNode),
        slotSize(msgQueue.SlotSize()),
        d2hConcurrency(D2HConcurrencyForQueue(root.Topo(), msgQueue))
    {}
    ~CheckpointWriter() {}    // NOLINT(modernize-use-equals-default) causes GCC warnings
    CLASS_DELETE_COPY_AND_MOVE(CheckpointWriter);

    struct PartitionContext
    {
        static constexpr size_t ExpectedPendingD2HRequestsPerBuffer = 1024;

        struct D2HRequest
        {
            DeviceId device;
            uint64_t src;
            uint64_t size;
        };

        CheckpointWriter* const that           = nullptr;
        uint8_t const           index          = 0;
        uint64_t const          partitionStart = 0;
        uint64_t const          partitionEnd   = 0;
        cipc::SnapshotId const  snapshotId{cipc::SnapshotId::Invalid};

        std::unordered_map<DeviceId, std::unique_ptr<DeviceContext>> deviceContext;

        std::vector<D2HRequest> pendingD2HRequests;
        uint64_t                sendOffset{};
        uint64_t                readOffset{};
        size_t                  bufBytesRemaining;    // remaining free space in the current slot; starts at a full slot

        PartitionContext(CheckpointWriter* thatIn LFTBND,
                         uint8_t                  indexIn,
                         uint64_t                 partitionStartIn,
                         uint64_t                 partitionEndIn,
                         cipc::SnapshotId const&  snapshotIdIn) :
            that(thatIn),
            index(indexIn),
            partitionStart(partitionStartIn),
            partitionEnd(partitionEndIn),
            snapshotId(snapshotIdIn),
            sendOffset(partitionStartIn),
            bufBytesRemaining(thatIn->slotSize)
        { pendingD2HRequests.reserve(ExpectedPendingD2HRequestsPerBuffer); }

        CLASS_DELETE_COPY_AND_MOVE(PartitionContext);

        ~PartitionContext()
        {
            assert(bufBytesRemaining == that->slotSize);
            if (pendingD2HRequests.capacity() > ExpectedPendingD2HRequestsPerBuffer)
            {
                fmt::print(stderr,
                           "[native_ckpt:cli][{}:{}][WARN]  pending D2H requests which exceeds the expected count of "
                           "{}. Consider increasing the buffer size or expected count.\n",
                           static_cast<uint32_t>(that->root.GlobalRank()),
                           index,
                           pendingD2HRequests.size(),
                           ExpectedPendingD2HRequestsPerBuffer);
            }
        }

        private:
        void ProcessTensor_(StateNode::Handle /* handle */, DeviceId const device, uint64_t tensorSize, uint64_t tensorHandle)
        {
            if (readOffset >= partitionEnd) { return; }
            if (tensorSize == 0) { return; }

            auto next = readOffset + tensorSize;
            if (next <= partitionStart)
            {
                readOffset = next;
                return;
            }
            size_t startOffset = 0;
            if (readOffset < partitionStart)
            {
                startOffset = partitionStart - readOffset;
                readOffset  = partitionStart;
            }
            size_t bytesRemaining = std::min(tensorSize - startOffset, partitionEnd - readOffset);
            while (bufBytesRemaining < bytesRemaining)
            {
                pendingD2HRequests.emplace_back(device, tensorHandle + startOffset, bufBytesRemaining);
                startOffset += bufBytesRemaining;
                bytesRemaining -= bufBytesRemaining;
                bufBytesRemaining = 0;
                FlushBytes_(that->slotSize);
            }
            if (bytesRemaining > 0)
            {
                pendingD2HRequests.emplace_back(device, tensorHandle + startOffset, bytesRemaining);
                bufBytesRemaining -= bytesRemaining;
            }
            if (bufBytesRemaining == 0) { FlushBytes_(that->slotSize); }
            readOffset = next;
        }

        void FlushBytes_(size_t sendBytes)
        {
            auto [slot, slotbuf]                                 = that->msgQueue.AllocSlot(cipc::Message::LocalStateData{
                .sessionId   = cipc::SessionId{0},
                .snapshotId  = snapshotId,
                .globalRank  = that->root.GlobalRank(),
                .messageSize = static_cast<uint32_t>(sendBytes),
                .offset      = sendOffset,
            });
            auto*                              ptr               = slotbuf;
            DeviceContext*                     lastDeviceContext = nullptr;
            std::unordered_set<DeviceContext*> usedDeviceContexts;
            for (auto const& req : pendingD2HRequests)
            {
                if (lastDeviceContext == nullptr || lastDeviceContext->device != req.device)
                {
                    lastDeviceContext = deviceContext.contains(req.device)
                                            ? deviceContext[req.device].get()
                                            : deviceContext.insert({req.device, DeviceContext::CreateDeviceContext(req.device, index)})
                                                  .first->second.get();
                    lastDeviceContext->StreamAcquire();
                    usedDeviceContexts.insert(lastDeviceContext);
                }
                lastDeviceContext->AddD2H(req.src, ptr, req.size);
                that->totalWrittenBytes.fetch_add(req.size);
                ptr += req.size;    // NOLINT
            }
            for (auto* ctx : usedDeviceContexts) { ctx->Flush(); }

            that->msgQueue.CommitSlot(slot);
            pendingD2HRequests.clear();
            //  Send data to checkpoint server after D2H completes
            that->totalD2HRequests.fetch_add(1);
            sendOffset += sendBytes;

            assert(sendBytes + bufBytesRemaining == that->slotSize);
            bufBytesRemaining = that->slotSize;
        }

        // Flush pending D2H data and release the stream lock.
        // Callers that intend to continue submitting work on this context must call
        // deviceContext->StreamAcquire() afterwards (FlushFullBuffer_ does this internally).
        void FlushRemaining_()
        {
            assert(bufBytesRemaining <= that->slotSize);
            if (bufBytesRemaining == that->slotSize) { return; }
            if (pendingD2HRequests.empty()) { return; }
            that->smallD2HRequests.fetch_add(1);
            FlushBytes_(that->slotSize - bufBytesRemaining);
        }

        public:
        void WriteTensors()
        {
            that->root.WalkNodes([&](StateNode const& node) {
                auto type = static_cast<StateNode::Type>(node.data.index());
                if (type != StateNode::Type::TensorShard && type != StateNode::Type::ShardedTensor) { return; }
                try
                {
                    if (type == StateNode::Type::TensorShard)
                    {
                        auto const& tensor = std::get<StateNode::TensorShard>(node.data);
                        ProcessTensor_(node.handle, tensor.device, tensor.ShardSizeBytes(), tensor.handle);
                    }
                    else if (type == StateNode::Type::ShardedTensor)
                    {
                        auto const& tensor = std::get<StateNode::ShardedTensor>(node.data);
                        for (auto const& shard : tensor.shards)
                        {
                            ProcessTensor_(node.handle, shard.device, shard.ShardSizeBytes(tensor.dtype), shard.handle);
                        }
                    }
                } catch (std::exception& e)
                {
                    fmt::print(stderr,
                               "[native_ckpt:cli][{}:{}][ERROR] Failed to write partition [{} - {}): offset={} error={}\n",
                               that->root.GlobalRank(),
                               index,
                               partitionStart,
                               partitionEnd,
                               readOffset,
                               e.what());
                    throw;
                }
            });
            FlushRemaining_();
            if (sendOffset != partitionEnd)
            {
                throw std::runtime_error(
                    fmt::format("[native_ckpt:cli][{}:{}][ERROR] Mismatch in sent bytes offset = {} and partition-size={} ([{}:{}])\n",
                                that->root.GlobalRank(),
                                index,
                                sendOffset,
                                (partitionEnd - partitionStart),
                                partitionStart,
                                partitionEnd));
            }
        }
    };

    void WriteTensors(cipc::SnapshotId const& snapshotId)
    {
        auto totalTensorStateSizeInBytes = root.GetTotalTensorStateSizeInBytes();
        if (totalTensorStateSizeInBytes == 0) { return; }

        std::array<std::future<void>, cipc::D2HConcurrency> writerTasks{};

        auto const concurrentBufferBytes     = slotSize * d2hConcurrency;
        auto const buffersChunksPerPartition = (totalTensorStateSizeInBytes + concurrentBufferBytes - 1) / concurrentBufferBytes;
        // std::views::enumerate not yet available in libc++ 21.1.7 as of (2026-01-05)
        // https://github.com/llvm/llvm-project/pull/73617 for (auto [index, task] : std::views::enumerate(writerTasks))

        for (uint8_t index = 0; index < d2hConcurrency; ++index)
        {
            auto&          task                = writerTasks[index];    // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            uint64_t const partitionBufferSize = (buffersChunksPerPartition * slotSize);
            uint64_t const partitionStart      = partitionBufferSize * index;
            uint64_t const partitionEnd        = std::min(partitionStart + partitionBufferSize, totalTensorStateSizeInBytes);
            if (partitionStart >= partitionEnd) { break; }
            task = std::async(std::launch::async | std::launch::deferred, [this, index, partitionStart, partitionEnd, snapshotId] {
                SetThreadName(fmt::format("ncc:wrtn:{}", index).c_str());
                PartitionContext ctx(this, index, partitionStart, partitionEnd, snapshotId);
                // On scope exit free the buffer;
                ctx.WriteTensors();
            });
        }
        for (auto& task : writerTasks)
        {
            if (task.valid()) { task.get(); }
        }
    }

    void SendToServer(cipc::SnapshotId const& snapshotId, std::filesystem::path const& ckptPath)
    {
        SetThreadName("ncc:sendsvr");

        auto totalTensorStateSizeInBytes = root.GetTotalTensorStateSizeInBytes();
        auto numTensors                  = root.GetTotalNumTensors();
        auto stop1                       = Clock::now();

        {
            // The CUDA current device is thread-local. Reuse the node captured while
            // opening this rank's queue instead of querying again from an async writer
            // thread, where the current device may have fallen back to device 0.
            std::stringstream sstrm;
            serdes::WriteBlobWithSize<uint16_t>(sstrm, std::filesystem::path(ckptPath).string());
            uint32_t numBlobs   = 0;
            uint32_t numScalars = 0;
            root.WalkNodes([&](StateNode const& node) {
                auto const type = static_cast<StateNode::Type>(node.data.index());
                if (type == StateNode::Type::Blob) { ++numBlobs; }
                else if (type == StateNode::Type::Scalar) { ++numScalars; }
            });
            serdes::WriteTrivial(sstrm, numBlobs);
            root.WalkNodes([&](StateNode const& node) {
                auto const type = static_cast<StateNode::Type>(node.data.index());
                if (type == StateNode::Type::Blob)
                {
                    auto const& blob = std::get<StateNode::Blob>(node.data);
                    serdes::WriteBlobWithSize<uint32_t>(sstrm, std::string_view(blob.data(), blob.size()));
                }
            });
            serdes::WriteTrivial(sstrm, numScalars);
            root.WalkNodes([&](StateNode const& node) {
                auto const type = static_cast<StateNode::Type>(node.data.index());
                if (type == StateNode::Type::Scalar) { serdes::WriteTrivial(sstrm, std::get<StateNode::Scalar>(node.data)); }
            });
            auto msg = sstrm.str();
            msgQueue.Push({cipc::Message::ActivateSnapshot{
                              .sessionId    = cipc::SessionId{0},
                              .snapshotId   = snapshotId,
                              .globalRank   = root.GlobalRank(),
                              .snapshotSize = totalTensorStateSizeInBytes,
                              .messageSize  = static_cast<uint32_t>(msg.size()),
                              .numaNode     = numaNode,
                              .startedAt    = root.startedAt,
                          }},
                          msg.data(),
                          msg.size());
        }
        WriteTensors(snapshotId);
        auto const stop2       = Clock::now();
        auto const d2hRequests = totalD2HRequests.load();
        auto const smallD2H    = smallD2HRequests.load();

        if (smallD2H > 16)
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:cli][{}][ERROR] Too many small d2h requests={} for total tensors={} total_d2h={}",
                            root.GlobalRank(),
                            smallD2H,
                            numTensors,
                            d2hRequests));
        }

        if (totalWrittenBytes.load() != totalTensorStateSizeInBytes)
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:cli][{}][ERROR] Total written bytes does not match total tensor size: written={} expected={}",
                            root.GlobalRank(),
                            totalWrittenBytes.load(),
                            totalTensorStateSizeInBytes));
        }

        msgQueue.Push({cipc::Message::FinalizeSnapshot{
                          .sessionId          = cipc::SessionId{0},
                          .globalRank         = root.GlobalRank(),
                          .snapshotId         = snapshotId,
                          .snapshotDuration   = stop2 - start,
                          .d2hDuration        = stop2 - stop1,
                          .d2hRequests        = totalD2HRequests.load(),
                          .totalSnapshotBytes = totalWrittenBytes.load(),
                      }},
                      nullptr,
                      0);
    }
};
}    // namespace

void SaveCheckpoint(StateTree const& root, cipc::SnapshotId const& snapshotId, std::filesystem::path const& ckptPath)
{
    CheckpointWriter writer{root};
    writer.SendToServer(snapshotId, ckptPath);
}

void WaitForCheckpointSave(RankTopology const& topo)
{
    auto& msgQueue = CheckpointWriter::GetMsgQueue(topo);
    for (size_t i = 0; i < msgQueue.SlotCount(); ++i) { msgQueue.Push({cipc::Message::Ping{}}, nullptr, 0); }
}

void FinalizeStateRegistry(StateTree& root)
{
    if (root.IsFinalized()) { return; }
    root.FinalizeRegistry();
    CheckpointWriter::SendStateRegistry(root);
}

// For now, we don't have any deferred tensors to save, so this is a no-op. In the future, we can add logic here to save any remaining
// tensors that were deferred during registration.
[[noreturn]] static void LoadCheckpointforStateTreeFromStream(StateTree& /* tree */, std::istream& /* stream */)
{ throw std::logic_error("TODO"); }

[[noreturn]] void LoadCheckpoint(StateTree& root, std::filesystem::path const& ckptPath)
{
    auto          fname = ckptPath / fmt::format("rank_{}.ckpt", root.GlobalRank());
    std::ifstream ifs(fname, std::ios::binary);
    if (!ifs) { throw std::runtime_error(fmt::format("Failed to open checkpoint file for writing: {}", fname.string())); }
    LoadCheckpointforStateTreeFromStream(root, ifs);
}

std::unique_ptr<DeviceContext> DeviceContext::CreateDeviceContext(DeviceId device, uint8_t streamIdx)
{
    using namespace std::literals;
    using namespace std::string_view_literals;
    if (device.IsType("cpu"sv)) { return DeviceContext::CreateDeviceContextCPU(device, streamIdx); }

#ifdef PHOENIX_HAVE_CUDA
    if (device.IsType("cuda"sv)) { return DeviceContext::CreateDeviceContextCUDA(device, streamIdx); }
#endif
#ifdef PHOENIX_HAVE_METAL
    if (device.IsType("mps"sv)) { return DeviceContext::CreateDeviceContextMTL(device, streamIdx); }
#endif
#if !defined(PHOENIX_HAVE_CUDA) && defined(PHOENIX_HAVE_METAL)
    if (device.IsType("cuda"sv)) { return DeviceContext::CreateDeviceContextMTL(device, streamIdx); }
#endif
    // Falling through to CPU on an unmatched device type silently maps a
    // CUDA / MPS device address to CPUContext::AddD2H, which does
    // `memcpy(host, (void*)deviceAddr, size)` and segfaults the moment a
    // real device pointer arrives. Better to fail loudly: misconfigurations
    // (e.g. the wheel built without PHOENIX_BUILD_CUDA but loaded into a
    // CUDA training run) now surface as a clear runtime error at the
    // factory call instead of an opaque SIGSEGV deep inside the writer.
    throw std::runtime_error(fmt::format("Unsupported device type for DeviceContext creation: {} — "
                                         "is this build missing PHOENIX_BUILD_CUDA / PHOENIX_BUILD_METAL?",
                                         device));
}

void SvcDecrRef()
{}

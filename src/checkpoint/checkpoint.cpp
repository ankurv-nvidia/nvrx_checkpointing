#include "CommonMacros.h"
#include "SetThreadName.h"

#include "checkpoint.h"
#include "config.h"
#include "messages.h"
#include "quant.h"
#include "server.h"
#include "service_client.h"
#include "service_reference.h"
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

#include <chrono>
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
    if (checkpoint::config::GetConfigValue<checkpoint::config::ServerInproc>())
    {
        static constexpr auto StartTimeout = 900s;
        auto const            deadline     = std::chrono::steady_clock::now() + StartTimeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (TryConnectToCheckpointService()) { return; }
            std::this_thread::sleep_for(100ms);
        }
        throw std::runtime_error("Timed out waiting for the local-rank-0 checkpoint service");
    }

    bool retry = false;
    try
    {
        boost::asio::io_context    ctx;
        boost::asio::readable_pipe output(ctx);
        boost::process::process    proc(ctx, svcExeLoc, {}, boost::process::process_stdio{.in = {}, .out = output, .err = stderr});
        std::string                line;
        boost::system::error_code  ec;
        boost::asio::read_until(output, boost::asio::dynamic_buffer(line), '\n', ec);
        retry = line == "RETRY\n";
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

    if (retry) { throw boost::system::system_error(boost::asio::error::try_again, "Checkpoint service election is pending"); }

    // Both the elected daemon and losing candidates write READY only after the
    // elected service answers a ping. The two short attempts inside TryConnect
    // cover a final scheduling edge without a second readiness policy here.
    if (!TryConnectToCheckpointService()) { throw std::runtime_error("Failed to start or connect to checkpoint service"); }
}

static uint32_t ServerIndex(StateTree const& registry)
{
    auto const& topo        = registry.Topo();
    auto const  serverIndex = (static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer) % topo.peersPerNode;
    return serverIndex;
}

static auto LocalServiceRequest(uint32_t serverIndex, std::string const& target, std::string const& body)
{
    auto const                port        = cipc::Message::Port + serverIndex;
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
        boost::beast::http::request<boost::beast::http::string_body> req{boost::beast::http::verb::post, target, HttpVersion11};
        req.set(boost::beast::http::field::host, fmt::format("localhost:{}", port));
        req.set(boost::beast::http::field::content_type, "application/octet-stream");
        req.body() = body;
        req.prepare_payload();
        boost::beast::http::write(stream, req, ec);
        if (ec) { throw boost::system::system_error(ec); }

        boost::beast::flat_buffer                                            buffer;
        boost::beast::http::response_parser<boost::beast::http::string_body> parser;
        parser.body_limit(std::numeric_limits<uint64_t>::max());
        boost::beast::http::read(stream, buffer, parser, ec);
        if (ec) { throw boost::system::system_error(ec); }
        auto response = parser.release();
        if (response.result() != boost::beast::http::status::ok)
        {
            throw std::runtime_error(
                fmt::format("{} failed on port {}: HTTP {}: {}", target, port, response.result_int(), response.body()));
        }
        TryCleanShutdown(stream);
        return response;
    }
    throw boost::system::system_error(ec);
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
    LocalServiceRequest(0, "/api/stop", {});
}

void SvcSpinDown()
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::mutex SpinDownMutex;
    SUPPRESS_WARNINGS_END

    std::scoped_lock lock(SpinDownMutex);

    if (!TryConnectToCheckpointService()) { return; }

    LocalServiceRequest(0, "/api/spindown", {});
    auto const deadline = std::chrono::steady_clock::now() + 30s;
    while (TryConnectToCheckpointService())
    {
        if (std::chrono::steady_clock::now() >= deadline) { throw std::runtime_error("Checkpoint service shutdown timed out"); }
        std::this_thread::sleep_for(20ms);
    }
}

void WaitForCkptFinalize(uint32_t sessionId, int64_t iteration)
{
    auto response = LocalServiceRequest(0, fmt::format("/api/wait-for-ckpt-finalize/{}/{}", sessionId, iteration), {});
    if (response.result() != boost::beast::http::status::ok)
    {
        throw std::runtime_error(
            fmt::format("wait-for-ckpt-finalize: server returned HTTP {}: {}", response.result_int(), response.body()));
    }
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
    static auto OpenMsgQueue(RankTopology const& topo)
    {
        auto       localRank  = static_cast<uint32_t>(topo.rank) % topo.ranksPerPeer;
        auto const queueCount = ResolveClientQueueCount(topo);
        // Distribute consecutive local ranks as evenly as possible over the
        // configured queues. Daemon-side sizing uses the same mapping.
        auto const index       = cipc::QueueIndex(localRank, topo.ranksPerPeer, queueCount);
        auto const peerIndex   = (static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer) % topo.peersPerNode;
        auto const channelName = cipc::Message::RequestChannelName(static_cast<uint16_t>(peerIndex), index);
        return cipc::MsgQueue::OpenUniquePtr(channelName);
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
        std::shared_ptr<StateTree>      snapshotRegistry;
        std::mutex                      snapshotMutex;
        uint64_t                        generation{};
        bool                            registered{false};

        ~QueueState()
        {
#ifdef PHOENIX_HAVE_CUDA
            if (registered) try
                {
                    DeviceContext::CudaUnregisterHostBuffer(queue->GetShmDataRegion().data());
                } catch (std::exception const& error)
                {
                    fmt::print(stderr, "[native_ckpt:cli][ERROR] Queue unregistration failed: {}\n", error.what());
                }
#endif
        }
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

    static void ObserveGeneration(Rank rank, uint64_t generation)
    {
        auto&            reg = Registry();
        std::scoped_lock lock(reg.mu);
        auto&            state = reg.states[rank];
        if (state && state->generation == generation) { return; }
        state             = std::make_shared<QueueState>();
        state->generation = generation;
    }

    static void SendStateRegistry(StateTree const& registry)
    {
        auto const& topo = registry.Topo();
        if (topo.ranksPerPeer == 0 || topo.peersPerNode == 0) { throw std::invalid_argument("peer topology must be nonzero"); }
        auto&            reg = Registry();
        std::unique_lock lock(reg.mu);
        auto&            entry = reg.states[topo.rank];
        if (entry && entry->ready.valid()) { return; }
        if (!entry) { entry = std::make_shared<QueueState>(); }
        auto               state = entry;
        std::promise<void> readyPromise;
        state->ready = readyPromise.get_future().share();
        lock.unlock();
        int numa = -1;
#ifdef PHOENIX_HAVE_CUDA
        int deviceIndex = -1;
        registry.ForEachPayload([&](StateNode::Handle, Tensor::DType, DeviceId device, uint64_t, uint64_t) {
            if (deviceIndex < 0 && device.IsType("cuda")) { deviceIndex = static_cast<int>(device.index); }
        });
        if (deviceIndex >= 0) { numa = DeviceContext::CudaGetDeviceNumaNode(deviceIndex); }
#endif
        try
        {
            SvcPing();
            std::ostringstream ostr;
            serdes::WriteBlobWithSize<uint16_t>(ostr, "");
            StateSerDes::WriteTopology(ostr, topo);
            LocalServiceRequest(ServerIndex(registry), fmt::format("/api/activatesession/{}", registry.SessionId()), ostr.str());
            ostr.str({});
            StateSerDes::Write(ostr, registry, StateNode::Handle::Root);
            LocalServiceRequest(ServerIndex(registry), fmt::format("/api/localstateregistry/{}", registry.SessionId()), ostr.str());

            auto response = LocalServiceRequest(
                ServerIndex(registry), fmt::format("/api/snapshot-plan/{}/{}", registry.SessionId(), registry.GlobalRank()), {});
            std::istringstream input(response.body(), std::ios::binary);
            auto               snapshotRegistry = StateSerDes::Read(input);
            if (snapshotRegistry->SessionId() != registry.SessionId() || snapshotRegistry->GlobalRank() != registry.GlobalRank())
            {
                throw std::runtime_error("Snapshot plan does not match the requesting session and rank");
            }
            snapshotRegistry->FinalizeRegistry();
            ostr.str({});
            serdes::WriteTrivial<uint64_t>(ostr, snapshotRegistry->GetTotalSnapshotStateSizeInBytes());
            serdes::WriteTrivial<int32_t>(ostr, numa);
            LocalServiceRequest(
                ServerIndex(registry), fmt::format("/api/shmqueue/{}/{}", registry.SessionId(), registry.GlobalRank()), ostr.str());
            auto const& queueTopo = registry.Topo();
            state->queue          = OpenMsgQueue(queueTopo);
#ifdef PHOENIX_HAVE_CUDA
            auto region = state->queue->GetShmDataRegion();
            DeviceContext::CudaRegisterHostBuffer(region.data(), region.size());
            state->registered = true;
#endif
            state->snapshotRegistry = std::move(snapshotRegistry);
        } catch (...)
        {
            readyPromise.set_exception(std::current_exception());
            throw;
        }
        readyPromise.set_value();
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

    std::shared_ptr<QueueState> queueState;
    cipc::MsgQueue&             msgQueue;
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
        void AppendDeviceBytes_(DeviceId const device, uint64_t source, uint64_t size)
        {
            while (bufBytesRemaining < size)
            {
                pendingD2HRequests.push_back({.device = device, .src = source, .size = bufBytesRemaining});
                source += bufBytesRemaining;
                size -= bufBytesRemaining;
                bufBytesRemaining = 0;
                FlushBytes_(that->slotSize);
            }
            if (size > 0)
            {
                pendingD2HRequests.push_back({.device = device, .src = source, .size = size});
                bufBytesRemaining -= size;
            }
            if (bufBytesRemaining == 0) { FlushBytes_(that->slotSize); }
        }

        void ProcessTensor_(StateNode::Handle /* handle */,
                            Tensor::DType  snapshotDtype,
                            DeviceId const device,
                            uint64_t       tensorSize,
                            uint64_t       tensorHandle)
        {
            if (readOffset >= partitionEnd) { return; }
            if (tensorSize == 0) { return; }

            auto next = readOffset + tensorSize;
            if (next <= partitionStart)
            {
                readOffset = next;
                return;
            }
            uint64_t startOffset = 0;
            if (readOffset < partitionStart)
            {
                startOffset = partitionStart - readOffset;
                readOffset  = partitionStart;
            }
            uint64_t const bytesRemaining = std::min(tensorSize - startOffset, partitionEnd - readOffset);
            if (snapshotDtype == Tensor::DType::Invalid) { throw std::runtime_error("Snapshot dtype is invalid"); }
            if (snapshotDtype == Tensor::DType::FP8E4M3 || snapshotDtype == Tensor::DType::MXFP8 || snapshotDtype == Tensor::DType::NVFP4)
            {
                checkpoint::quant::SerializeSnapshotSegments(
                    tensorHandle, startOffset, bytesRemaining, [&](DeviceId segmentDevice, uint64_t source, uint64_t size) {
                        AppendDeviceBytes_(segmentDevice, source, size);
                    });
            }
            else
            {
                AppendDeviceBytes_(device, tensorHandle + startOffset, bytesRemaining);
            }
            readOffset = next;
        }

        void FlushBytes_(size_t sendBytes)
        {
            auto [slot, slotbuf]                                 = that->msgQueue.AllocSlot(cipc::Message::SnapshotData{
                .sessionId   = that->root.SessionId(),
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
                    if (usedDeviceContexts.insert(lastDeviceContext).second) { lastDeviceContext->StreamAcquire(); }
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
            that->root.ForEachPayload([&](StateNode::Handle node, Tensor::DType dtype, DeviceId device, uint64_t handle, uint64_t size) {
                try
                {
                    ProcessTensor_(node, dtype, device, size, handle);
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
        auto totalTensorStateSizeInBytes = root.GetTotalPayloadSizeInBytes();
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

        auto totalTensorStateSizeInBytes = root.GetTotalPayloadSizeInBytes();
        auto numTensors                  = root.GetTotalNumTensors();
        auto stop1                       = Clock::now();

        {
            std::stringstream sstrm;
            serdes::WriteBlobWithSize<uint16_t>(sstrm, std::filesystem::path(ckptPath).string());
            StateSerDes::Write(sstrm, root, StateNode::Handle::Root);
            auto msg = sstrm.str();
            if (msg.size() > std::numeric_limits<uint32_t>::max()) { throw std::length_error("Snapshot metadata exceeds wire limit"); }
            cipc::Message::ActivateSnapshot header{
                .sessionId    = root.SessionId(),
                .snapshotId   = snapshotId,
                .globalRank   = root.GlobalRank(),
                .snapshotSize = totalTensorStateSizeInBytes,
                .messageSize  = static_cast<uint32_t>(msg.size()),
                .startedAt    = root.startedAt,
            };
            std::ostringstream activation;
            serdes::WriteTrivial(activation, header);
            activation.write(msg.data(), static_cast<std::streamsize>(msg.size()));
            LocalServiceRequest(ServerIndex(root),
                                fmt::format("/api/activate-snapshot/{}/{}/{}", root.SessionId(), root.GlobalRank(), snapshotId),
                                activation.str());
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
                          .sessionId          = root.SessionId(),
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

checkpoint::ServiceReference SvcAcquireReference(StateTree const& root)
{
    auto const& topo = root.Topo();
    if (topo.ranksPerPeer == 0 || topo.peersPerNode == 0) { throw std::invalid_argument("peer topology must be nonzero"); }
    static constexpr auto AcquireTimeout = 900s;
    auto const            deadline       = std::chrono::steady_clock::now() + AcquireTimeout;
    while (true)
    {
        try
        {
            if (checkpoint::config::GetConfigValue<checkpoint::config::ServerInproc>()
                && !checkpoint::config::GetConfigValue<checkpoint::config::SvcDisableAutoLaunch>()
                && static_cast<uint32_t>(topo.rank) % (topo.ranksPerPeer * topo.peersPerNode) == 0)
            {
                StartOwnedInprocServer();
            }
            else
            {
                SvcPing();
            }
            auto reference = checkpoint::ServiceReferences::AcquireProcess(root.SessionId(), topo);
            CheckpointWriter::ObserveGeneration(topo.rank, reference.Generation());
            return reference;
        } catch (boost::system::system_error const&)
        {
            if (std::chrono::steady_clock::now() >= deadline) { throw; }
            std::this_thread::sleep_for(100ms);
        }
    }
}

void FlushCheckpointQueue(StateTree const& root)
{
    std::shared_ptr<CheckpointWriter::QueueState> state;
    {
        auto&            reg = CheckpointWriter::Registry();
        std::scoped_lock lock(reg.mu);
        auto             it = reg.states.find(root.Topo().rank);
        if (it == reg.states.end() || !it->second->ready.valid()) { return; }
        state = it->second;
    }
    state->ready.get();
    state->queue->Flush();
}

std::shared_ptr<StateTree> GetSnapshotRegistry(StateTree const& localRegistry)
{ return CheckpointWriter::GetQueueState(localRegistry)->snapshotRegistry; }

void SaveCheckpoint(StateTree const& snapshotRegistry, cipc::SnapshotId const& snapshotId, std::filesystem::path const& ckptPath)
{
    auto             state = CheckpointWriter::GetQueueState(snapshotRegistry);
    std::scoped_lock lock(state->snapshotMutex);
    snapshotRegistry.ForEachPayload([&](StateNode::Handle node, Tensor::DType, DeviceId, uint64_t handle, uint64_t storageSize) {
        if (storageSize != 0 && handle == 0)
        {
            throw std::runtime_error(fmt::format("Snapshot state '{}' has no current source data", snapshotRegistry.GetNodeId(node)));
        }
    });
    CheckpointWriter writer{snapshotRegistry};
    writer.SendToServer(snapshotId, ckptPath);
}

void WaitForCheckpointSave(StateTree const& snapshotRegistry)
{
    auto& msgQueue = *CheckpointWriter::GetQueueState(snapshotRegistry)->queue;
    for (size_t i = 0; i < msgQueue.SlotCount(); ++i) { msgQueue.Push({cipc::Message::Ping{}}, nullptr, 0); }
}

void FinalizeStateRegistry(StateTree& localRegistry)
{
    if (!localRegistry.IsFinalized()) { localRegistry.FinalizeRegistry(); }
    CheckpointWriter::SendStateRegistry(localRegistry);
}

// For now, we don't have any deferred tensors to save, so this is a no-op. In the future, we can add logic here to save any remaining
// tensors that were deferred during registration.
[[noreturn]] static void LoadCheckpointforStateTreeFromStream(StateTree& /* tree */, std::istream& /* stream */)
{ throw std::logic_error("TODO"); }

[[noreturn]] void LoadCheckpoint(StateTree& localRegistry, std::filesystem::path const& ckptPath)
{
    auto          fname = ckptPath / fmt::format("rank_{}.ckpt", localRegistry.GlobalRank());
    std::ifstream ifs(fname, std::ios::binary);
    if (!ifs) { throw std::runtime_error(fmt::format("Failed to open checkpoint file for writing: {}", fname.string())); }
    LoadCheckpointforStateTreeFromStream(localRegistry, ifs);
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

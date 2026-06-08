#include "CommonMacros.h"
#include "SetThreadName.h"

#include "checkpoint.h"
#include "messages.h"
#include "numa_utils.h"
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
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;
namespace cipc = checkpoint::ipc;

template <typename TStream> static void TryCleanShutdown(TStream& stream)
{
    boost::system::error_code ec;
    ec = stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    if (ec && ec != boost::system::errc::not_connected)
    {    //
        fmt::print(stderr, "Shutdown Error: {}\n", ec.what());
    }
}

static bool TryConnectToCheckpointService()
{
    boost::system::error_code ec;

    boost::asio::io_context        ioc;
    boost::beast::tcp_stream       stream(ioc);
    boost::asio::ip::tcp::resolver resolver(ioc);

    auto results = resolver.resolve("localhost", std::to_string(cipc::Message::Port));
    // Try connecting to localhost before we send a request to check and fail-fast
    [[maybe_unused]] auto endpoint = stream.connect(results, ec);
    if (ec) { return false; }

    // Create HTTP GET request with buffer_body (avoids copying state data)
    static constexpr int                                         HttpVersion11 = 11;
    boost::beast::http::request<boost::beast::http::buffer_body> req{boost::beast::http::verb::get, "/api/ping", HttpVersion11};
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

    if (res.result() != boost::beast::http::status::ok) { return false; }
    TryCleanShutdown(stream);
    return true;
}

static auto GetEnv(char const* envVar, std::string_view defaultVal) -> std::string_view
{
    auto* val = getenv(envVar);
    return val != nullptr ? std::string_view(val) : defaultVal;
};

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
    if (GetEnv("CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH", "0") == "1") { std::this_thread::sleep_for(1s); }
    else
    {
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
    }
    if (!TryConnectToCheckpointService()) { throw std::runtime_error("Failed to start or connect to checkpoint service"); }
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

namespace
{
struct CheckpointWriter
{
    using Clock     = cipc::PerfMetrics::Clock;
    using TimePoint = Clock::time_point;
    using Duration  = Clock::duration;

    StateTree const&      root;    // NOLINT
    static cipc::MsgQueue InitMsgQueue(RankTopology const& topo)
    {
        int numa = GetCurrentNumaNode();
#ifdef PHOENIX_HAVE_CUDA
        // The GPU, fixed by the rank, decides where the D2H copy lands fastest.
        if (int gpuNuma = DeviceContext::CudaGetCurrentDeviceNumaNode(); gpuNuma >= 0) { numa = gpuNuma; }
#endif
        auto localRank = static_cast<uint32_t>(topo.rank) % topo.ranksPerPeer;
        // Map consecutive same-socket ranks to one queue so each shm buffer stays
        // within a single NUMA domain.
        auto ranksPerQueue = topo.ranksPerPeer / cipc::MaxRanksPerNode;
        if (ranksPerQueue == 0) { ranksPerQueue = 1; }
        auto index     = (localRank / ranksPerQueue) % cipc::MaxRanksPerNode;
        auto peerIndex = (static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer) % topo.peersPerNode;
        auto channelName = cipc::Message::RequestChannelName(static_cast<uint16_t>(peerIndex), static_cast<uint8_t>(index));
        fmt::print(stderr, "[native_ckpt:cli][{}][INFO][perf][NUMA#{}] Using channel={}\n", topo.rank, numa, channelName);
        auto msgQueue = cipc::MsgQueue::Open(channelName);
#ifdef PHOENIX_HAVE_CUDA
        auto region = msgQueue.GetShmDataRegion();
        // Bind the data pages to the rank-local node before cudaHostRegister faults
        // them in; the queue's Create initializes only the control header. Single
        // node only — the rank->domain mapping isn't correct for multi-peer layouts.
        if (topo.peersPerNode == 1) { BindRegionToNumaNode(numa, region.data(), region.size()); }
        DeviceContext::CudaRegisterHostBuffer(region.data(), region.size());
#endif
        return msgQueue;
    }

    static cipc::MsgQueue& GetMsgQueue(RankTopology const& topo)
    {
        SUPPRESS_WARNINGS_START
        SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
        // TODO_PINNED_BUFFERS
        // This cannot be thread_local because it uses pinned buffers which are registered once per process
        static /* thread_local */ std::unordered_map<Rank, std::unique_ptr<cipc::MsgQueue>> MsgQueueMap;
        SUPPRESS_WARNINGS_END

        auto it = MsgQueueMap.find(topo.rank);
        if (it == MsgQueueMap.end())
        {
            auto msgQueue = InitMsgQueue(topo);
            it            = MsgQueueMap.emplace(topo.rank, std::make_unique<cipc::MsgQueue>(std::move(msgQueue))).first;
        }
        return *(it->second);
    }

    cipc::MsgQueue&     msgQueue = GetMsgQueue(root.Topo());
    std::atomic<size_t> totalD2HRequests{0};
    std::atomic<size_t> smallD2HRequests{0};
    std::atomic<size_t> totalWrittenBytes{0};

    TimePoint const start = Clock::now();

    CheckpointWriter(StateTree const& rootIn) : root(rootIn) {}
    ~CheckpointWriter() {}    // NOLINT(modernize-use-equals-default) causes GCC warnings
    CLASS_DELETE_COPY_AND_MOVE(CheckpointWriter);

    struct PartitionContext
    {
        static constexpr size_t ExpectedPendingD2HRequestsPerBuffer = 256;

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
        size_t                  bufBytesRemaining{cipc::D2HBufferSize};

        PartitionContext(CheckpointWriter*       thatIn,
                         uint8_t                 indexIn,
                         uint64_t                partitionStartIn,
                         uint64_t                partitionEndIn,
                         cipc::SnapshotId const& snapshotIdIn) :
            that(thatIn),
            index(indexIn),
            partitionStart(partitionStartIn),
            partitionEnd(partitionEndIn),
            snapshotId(snapshotIdIn),
            sendOffset(partitionStartIn)
        { pendingD2HRequests.reserve(ExpectedPendingD2HRequestsPerBuffer); }

        CLASS_DELETE_COPY_AND_MOVE(PartitionContext);

        ~PartitionContext()
        {
            assert(bufBytesRemaining == cipc::D2HBufferSize);
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
                FlushBytes_(cipc::D2HBufferSize);
            }
            if (bytesRemaining > 0)
            {
                pendingD2HRequests.emplace_back(device, tensorHandle + startOffset, bytesRemaining);
                bufBytesRemaining -= bytesRemaining;
            }
            if (bufBytesRemaining == 0) { FlushBytes_(cipc::D2HBufferSize); }
            readOffset = next;
        }

        void FlushBytes_(uint32_t sendBytes)
        {
            auto [slot, slotbuf] = that->msgQueue.AllocSlot(cipc::Message::LocalStateData{.sessionId   = cipc::SessionId{0},
                                                                                          .snapshotId  = snapshotId,
                                                                                          .globalRank  = that->root.GlobalRank(),
                                                                                          .messageSize = sendBytes,
                                                                                          .offset      = sendOffset});
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

            assert(sendBytes + bufBytesRemaining == cipc::D2HBufferSize);
            bufBytesRemaining = cipc::D2HBufferSize;
        }

        // Flush pending D2H data and release the stream lock.
        // Callers that intend to continue submitting work on this context must call
        // deviceContext->StreamAcquire() afterwards (FlushFullBuffer_ does this internally).
        void FlushRemaining_()
        {
            assert(bufBytesRemaining <= cipc::D2HBufferSize);
            if (bufBytesRemaining == cipc::D2HBufferSize) { return; }
            if (pendingD2HRequests.empty()) { return; }
            that->smallD2HRequests.fetch_add(1);
            FlushBytes_(static_cast<uint32_t>(cipc::D2HBufferSize - bufBytesRemaining));
        }

        public:
        void WriteTensors()
        {

            auto startTime = std::chrono::system_clock::now();
            fmt::print(stderr,
                       "[native_ckpt:cli][{}][INFO] Starting worker {} for partition [{} - {})\n",
                       that->root.GlobalRank(),
                       index,
                       partitionStart,
                       partitionEnd);

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

            fmt::print(stderr,
                       "[native_ckpt:cli][{}:{}][INFO] Finished partition [{}:{}] in {}us {}\n",
                       that->root.GlobalRank(),
                       index,
                       partitionStart,
                       partitionEnd,
                       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - startTime).count(),
                       that->msgQueue);
        }
    };

    void WriteTensors(cipc::SnapshotId const& snapshotId)
    {
        auto totalTensorStateSizeInBytes = root.GetTotalTensorStateSizeInBytes();
        if (totalTensorStateSizeInBytes == 0) { return; }

        std::array<std::future<void>, cipc::D2HNumConcurrency> writerTasks{};

        auto buffersChunksPerPartition = (totalTensorStateSizeInBytes + (cipc::D2HBufferSize * cipc::D2HNumConcurrency) - 1)
                                         / (cipc::D2HBufferSize * cipc::D2HNumConcurrency);
        // std::views::enumerate not yet available in libc++ 21.1.7 as of (2026-01-05)
        // https://github.com/llvm/llvm-project/pull/73617 for (auto [index, task] : std::views::enumerate(writerTasks))

        for (uint8_t index = 0; index < cipc::D2HNumConcurrency; ++index)
        {
            auto&          task                = writerTasks[index];    // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            uint64_t const partitionBufferSize = (buffersChunksPerPartition * cipc::D2HBufferSize);
            uint64_t const partitionStart      = partitionBufferSize * index;
            uint64_t const partitionEnd        = std::min(partitionStart + partitionBufferSize, totalTensorStateSizeInBytes);
            if (partitionStart >= partitionEnd) { break; }
            task = std::async(std::launch::async | std::launch::deferred, [this, index, partitionStart, partitionEnd, snapshotId]() {
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
            std::stringstream sstrm;
            serdes::WriteBlobWithSize<uint16_t>(sstrm, std::filesystem::path(ckptPath).string());
            auto msg = sstrm.str();
            msgQueue.Push({cipc::Message::ActivateSnapshot{.sessionId    = cipc::SessionId{0},
                                                           .snapshotId   = snapshotId,
                                                           .globalRank   = root.GlobalRank(),
                                                           .snapshotSize = totalTensorStateSizeInBytes,
                                                           .messageSize  = static_cast<uint32_t>(msg.size())}},
                          msg.data(),
                          msg.size());
        }
        WriteTensors(snapshotId);
        auto stop2  = Clock::now();
        auto wrTime = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop2 - stop1).count()) / 1e9;    // NOLINT
        auto bandwidth = static_cast<double>(totalTensorStateSizeInBytes) / (1024 * 1024) / wrTime;
        fmt::print(stderr,
                   "[native_ckpt:cli][{}][INFO][perf] Finished bw_mbps={:.2f} "
                   "| Time(us): dev_buf_init_us={} wr_tnsr_us={} total_us={} "
                   "| Count: num_tensors={} small_d2h={} total_d2h={} "
                   "| Bytes: total_written_bytes={} avg_d2h_rq={} avg_tnsr={}\n",
                   root.GlobalRank(),
                   bandwidth,
                   std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start).count(),
                   std::chrono::duration_cast<std::chrono::microseconds>(stop2 - stop1).count(),
                   std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start).count(),
                   numTensors,
                   smallD2HRequests.load(),
                   totalD2HRequests.load(),
                   totalWrittenBytes.load(),
                   totalTensorStateSizeInBytes / totalD2HRequests,
                   totalTensorStateSizeInBytes / std::max(numTensors, size_t{1u}));

        if (totalWrittenBytes.load() != totalTensorStateSizeInBytes)
        {
            throw std::runtime_error("Total written bytes does not match total tensor size");
        }
        msgQueue.Push(
            {cipc::Message::FinalizeSnapshot{.sessionId = cipc::SessionId{0}, .globalRank = root.GlobalRank(), .snapshotId = snapshotId}},
            nullptr,
            0);
    }
};
}    // namespace

void SaveCheckpoint(StateTree const& root, cipc::SnapshotId const& snapshotId, std::filesystem::path const& ckptPath)
{
    StateSerDes::DebugDumpLocalStateTreeKVP(ckptPath / fmt::format("cli_metadata_{}.txt", root.GlobalRank()), root);
    fmt::print(stderr, "[native_ckpt:cli][{}][INFO] Saving Checkpoint to dir {}\n", root.GlobalRank(), ckptPath.string());
    CheckpointWriter writer{root};
    writer.SendToServer(snapshotId, ckptPath);
}

void WaitForCheckpointSave(RankTopology const& topo)
{
    auto& msgQueue = CheckpointWriter::GetMsgQueue(topo);
    for (size_t i = 0; i < cipc::D2HBufferCount; ++i) { msgQueue.Push({cipc::Message::Ping{}}, nullptr, 0); }
}

void FinalizeStateRegistry(StateTree& self)
{
    fmt::print(stderr, "[native_ckpt:cli][{}][INFO] Finalizing State Registry\n", self.Topo().rank);
    SvcPing();    // TODO: Get rid of this
    SetThreadName("ncc:finst");
    auto const& topo     = self.Topo();
    auto&       msgQueue = CheckpointWriter::GetMsgQueue(topo);

    self.FinalizeRegistry();
    {
        std::stringstream sstrm;
        serdes::WriteBlobWithSize<uint16_t>(sstrm, "");
        StateSerDes::WriteTopology(sstrm, topo);
        auto msg = sstrm.str();
        msgQueue.Push({cipc::Message::ActivateSession{.sessionId = cipc::SessionId{0}, .messageSize = static_cast<uint32_t>(msg.size())}},
                      msg.data(),
                      msg.size());
    }
    {
        std::stringstream sstrm;

        StateSerDes::Write(sstrm, self, StateNode::Handle::Root);
        auto msg = sstrm.str();
        msgQueue.Push(cipc::Message::LocalStateRegistry{.sessionId = cipc::SessionId{0}, .messageSize = static_cast<uint32_t>(msg.size())},
                      msg.data(),
                      msg.size());
        fmt::print(stderr, "[native_ckpt:cli][{}][INFO] Send LocalStateRegistry MsgSize = {}\n", self.GlobalRank(), msg.size());
    }
}

// For now, we don't have any deferred tensors to save, so this is a no-op. In the future, we can add logic here to save any remaining
// tensors that were deferred during registration.
[[noreturn]] static void LoadCheckpointforStateTreeFromStream(StateTree& /* tree */, std::istream& /* stream */)
{
    throw std::logic_error("TODO");
#ifdef TODO_LOAD_CHECKPOINT
    StateSerDes::ReadIntoNode(stream, tree, 0);
    for (auto& node : tree.nodes)
    {
        auto& handler = tree.handlers[node.handle];
        switch (static_cast<StateNode::Type>(node.data.index()))
        {
        case StateTree::StateType::Scalar:
        {
            handler->SetScalarState(std::get<StateNode::Scalar>(node.data));
            break;
        }
        case StateTree::StateType::Blob:
        {
            handler->SetBlob(std::string_view(std::get<StateNode::Blob>(node.data)));
            break;
        }
        case StateTree::StateType::Tensor:
        {
            handler->SetTensorState(std::get<StateNode::Tensor>(node.data));
            break;
        }
        case StateTree::StateType::Dict:
        {
            break;
        }
        case StateTree::StateType::Unknown: [[fallthrough]];
        case StateTree::StateType::Invalid: [[fallthrough]];
        default: throw std::runtime_error("Unsupported state type in checkpoint");
        }
    }
#endif
}

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
    return DeviceContext::CreateDeviceContextCPU(device, streamIdx);
    // throw std::runtime_error(fmt::format("Unsupported device type : {} for DeviceContext creation", deviceStr));
}

void SvcDecrRef()
{}

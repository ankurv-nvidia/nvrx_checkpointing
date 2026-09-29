#pragma once
#include "CommonMacros.h"
#include "config.h"

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <fmt/chrono.h>
#include <fmt/std.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace checkpoint::ipc
{
static constexpr size_t  MinShmSize                    = 32ull * 1024ull * 1024ull;    // 32MB
static constexpr uint8_t D2HConcurrency                = 4;
static constexpr size_t  TargetShmSlotSize             = 2ull * 1024ull * 1024ull * 1024ull;
static constexpr uint8_t ExplicitSizeDefaultQueueCount = 4;
static constexpr size_t  ExplicitSizeDefaultSlotCount  = 4;

#ifdef PHOENIX_BUILD_FOR_VALGRIND
// Valgrind tracks V-bits/A-bits for every mapped byte and keeps a fixed-size
// segment table (aspacem). The default sizes below produce 1 GiB of shm per
// rank plus many 8 MiB writer-thread stacks, which exhausts valgrind's
// shadow memory / segment table on multi-rank runs. Shrink aggressively when
// building specifically for a valgrind run.
static constexpr size_t MaxShmSize = 128ull * 1024ull * 1024ull;
#else
static constexpr size_t MaxShmSize = 256ull * 1024ull * 1024ull * 1024ull;    // 256GB
#endif
static constexpr size_t MinSnapshotShmSize = MinShmSize / 4;

inline size_t RoundSnapshotShmSize(size_t snapshotSize)
{
    if (snapshotSize == 0) { return MinSnapshotShmSize; }
    snapshotSize        = std::min(snapshotSize, MaxShmSize);
    auto const half     = std::max<size_t>(snapshotSize / 2, 1);
    auto const quantum  = std::max<size_t>(std::bit_floor(half), 1ull << 20u);
    auto const withRoom = snapshotSize + ((snapshotSize + 3) / 4);
    auto       rounded  = std::max(((withRoom + quantum - 1) / quantum) * quantum, MinSnapshotShmSize);
    if (snapshotSize >= TargetShmSlotSize)
    {
        auto const dataSlots        = std::max<size_t>(D2HConcurrency, 1 + ((snapshotSize - 1) / TargetShmSlotSize));
        auto const slotsWithControl = std::min(dataSlots + 1, MaxShmSize / TargetShmSlotSize);
        rounded                     = std::max(rounded, slotsWithControl * TargetShmSlotSize);
    }
    return std::min(rounded, MaxShmSize);
}

inline uint8_t ResolveQueueCount(uint32_t ranksPerPeer, uint8_t configuredQueueCount)
{
    if (ranksPerPeer == 0 || ranksPerPeer > std::numeric_limits<uint8_t>::max())
    {
        throw std::invalid_argument("ranks per peer must fit in the queue index");
    }
    return configuredQueueCount == 0 ? static_cast<uint8_t>(ranksPerPeer)
                                     : static_cast<uint8_t>(std::min<uint32_t>(ranksPerPeer, configuredQueueCount));
}

inline uint8_t QueueIndex(uint32_t localRank, uint32_t ranksPerPeer, uint8_t queueCount)
{
    if (localRank >= ranksPerPeer || queueCount == 0 || queueCount > ranksPerPeer)
    {
        throw std::invalid_argument("rank and queue counts are inconsistent");
    }
    return static_cast<uint8_t>((static_cast<uint64_t>(localRank) * queueCount) / ranksPerPeer);
}

inline size_t QueueProducerCount(uint8_t queueIndex, uint32_t ranksPerPeer, uint8_t queueCount)
{
    if (queueCount == 0 || queueCount > ranksPerPeer || queueIndex >= queueCount)
    {
        throw std::invalid_argument("queue index must be less than the active queue count");
    }
    size_t producers = 0;
    for (uint32_t localRank = 0; localRank < ranksPerPeer; ++localRank)
    {
        if (QueueIndex(localRank, ranksPerPeer, queueCount) == queueIndex) { ++producers; }
    }
    return producers;
}

inline uint8_t QueueD2HConcurrency(size_t slotCount, size_t producerCount)
{
    if (slotCount == 0 || producerCount == 0) { throw std::invalid_argument("slot and producer counts must be nonzero"); }
    return std::min(
        D2HConcurrency,
        std::max<uint8_t>(1u, static_cast<uint8_t>(std::min<size_t>(std::numeric_limits<uint8_t>::max(), slotCount / producerCount))));
}

namespace bip = boost::interprocess;

struct StopRequestedException : public std::runtime_error
{
    StopRequestedException() : std::runtime_error("Stop requested") {}
};

struct PerfMetrics
{
    std::string_view name;
    using Clock = std::chrono::high_resolution_clock;
    Clock::rep      count{0};
    Clock::duration max{};
    Clock::duration total{};
    Clock::duration min{Clock::duration::max()};

#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS

    bool Record(Clock::duration dur)
    {
        count++;
        total += dur;
        max            = std::max(max, dur);
        min            = std::min(min, dur);
        auto threshold = (total / count) * 4;
        if (dur > threshold)
        {
            fmt::print("[shm_buf][perf][WARN] Perf Abnormality {} took {} > {} avg = {}. Count = {}\n",
                       name,
                       std::chrono::duration_cast<std::chrono::microseconds>(dur),
                       std::chrono::duration_cast<std::chrono::microseconds>(threshold),
                       std::chrono::duration_cast<std::chrono::microseconds>(total) / count,
                       count);
            return true;
        }
        return false;
    }
#endif
    auto format_to(fmt::format_context& ctx) const    // NOLINT
    {
        return fmt::format_to(ctx.out(),
                              "{}: cnt={} avg={}, max={}, min={}",
                              name,
                              count,
                              count == 0 ? std::chrono::nanoseconds{0} : total / count,
                              std::chrono::duration_cast<std::chrono::nanoseconds>(max),
                              std::chrono::duration_cast<std::chrono::nanoseconds>(min));
    }
};
}    // namespace checkpoint::ipc

template <> struct fmt::formatter<checkpoint::ipc::PerfMetrics> : fmt::formatter<std::string_view>
{
    auto format(checkpoint::ipc::PerfMetrics const& obj, fmt::format_context& ctx) const    // NOLINT
    {
        return obj.format_to(ctx);    //
    }
};

namespace checkpoint::ipc
{
// Multi-producer single-consumer ring buffer in shared memory
template <typename THdr> struct ShmRingBuffer
{
    public:
    static constexpr auto HearBeatInterval = std::chrono::microseconds(1);

    PerfMetrics sendPerf{"SendPerf"}, recvPerf{"RecvPerf"}, rlatency{"RdLatency"}, wlatency{"WrLatency"};    // NOLINT
    struct Control
    {
        using Clock = std::chrono::high_resolution_clock;
        // With a single consumer and single producer this is good enough
        // The consumer can read all memory from _rbeg to _rend
        // The producer can write all memory from _wbeg to _wend
        // As the consumer consumes the data, _wbeg and _wend are advanced to signal producer
        // As the producer produces the data, _rbeg and _rend are advanced to signal consumer
        std::atomic<Clock::time_point> rtimestamp{Clock::now()};
        std::atomic<Clock::time_point> wtimestamp{Clock::now()};
        std::atomic<std::size_t>       rbeg{0};
        std::atomic<std::size_t>       rend{0};
        std::atomic<std::size_t>       wbeg{0};
        std::atomic<std::size_t>       wend{0};

        uint64_t shmSize{0};
        uint64_t slotSize{0};
        uint64_t slotCount{0};

        bip::interprocess_mutex     mutex;
        bip::interprocess_condition canRead;     // Signaled when data is available to read
        bip::interprocess_condition canWrite;    // Signaled when space is available to write
    };

    auto format_to(fmt::format_context& ctx) const    // NOLINT
    {
        return fmt::format_to(ctx.out(),
                              "ShmRingBuffer[{}]: rbeg={} rend={} wbeg={} wend={}{}{}{}",
                              _slotCount,
                              _ctrl->rbeg,
                              _ctrl->rend,
                              _ctrl->wbeg,
                              _ctrl->wend,
                              sendPerf.count == 0 ? "" : fmt::format("{}", sendPerf),
                              recvPerf.count == 0 ? "" : fmt::format("{}", recvPerf),
                              rlatency.count == 0 ? "" : fmt::format("{}", rlatency),
                              wlatency.count == 0 ? "" : fmt::format("{}", wlatency));
    }

    static constexpr size_t AlignBoundary    = 4096;
    static constexpr size_t TargetSlotSize   = TargetShmSlotSize;
    static constexpr size_t DefaultSlotCount = 32;
    static constexpr size_t MaxSlotCount
        = (std::numeric_limits<size_t>::max() - sizeof(Control) - (AlignBoundary - 1)) / (sizeof(THdr) + AlignBoundary);

    static constexpr size_t HeaderAllocationSize(size_t slotCount)
    {
        if (slotCount == 0 || slotCount > MaxSlotCount) { throw std::invalid_argument("Invalid shared-memory slot count"); }
        auto const capacity = (sizeof(THdr) * slotCount) + sizeof(Control);
        return ((capacity + AlignBoundary - 1) / AlignBoundary) * AlignBoundary;
    }

    // Automatic geometry uses roughly 2 GiB transfers. Small queues retain
    // four data slots and two control slots per producer so small checkpoints
    // can still use the full D2H concurrency without delaying finalization.
    static constexpr size_t SlotCountForShmSize(size_t totalShmSize, size_t maximumSlotCount, size_t producerCount)
    {
        if (maximumSlotCount == 0 || maximumSlotCount > MaxSlotCount || producerCount == 0)
        {
            throw std::invalid_argument("Invalid automatic shared-memory slot geometry");
        }
        auto const minimumSlotCount = std::min(maximumSlotCount, producerCount * (D2HConcurrency + 2));
        auto const headerSize       = HeaderAllocationSize(maximumSlotCount);
        if (totalShmSize <= headerSize) { return minimumSlotCount; }
        auto const dataSize           = totalShmSize - headerSize;
        auto const slotsForTargetSize = 1 + ((dataSize - 1) / TargetSlotSize);
        if (slotsForTargetSize < 2 * producerCount) { return minimumSlotCount; }
        return std::min(slotsForTargetSize, maximumSlotCount);
    }

    static constexpr size_t MinimumShmSize(size_t slotCount) { return HeaderAllocationSize(slotCount) + (slotCount * AlignBoundary); }

    static ShmRingBuffer Create(std::string name, size_t totalShmSize, size_t slotCount)
    { return {std::move(name), true, totalShmSize, slotCount}; }
    static ShmRingBuffer Open(std::string name) { return {std::move(name), false, 0, 0}; }

    // Find the largest per-node budget the shared-memory filesystem accepts:
    // start from maxShmSize and halve on ENOSPC. The probe is a dedicated
    // throwaway object, truncated and removed - a live channel must never be
    // resized to probe capacity, because a concurrently opening client maps and
    // pins whatever size it observes. Filesystems that accept sparse truncation
    // report the full cap; the ring writes pages lazily, and openers wait out
    // the initial pinning, so an optimistic answer is safe.
    static size_t DetectShmSize(std::string const& name, size_t maxShmSize, size_t minShmSize, size_t slotCount)
    {
        size_t const floor     = std::max(minShmSize, MinimumShmSize(slotCount));
        size_t       allocSize = std::max(maxShmSize, floor);
        auto const   probeName = name + ".sizeprobe";
        while (true)
        {
            try
            {
                auto shm = bip::shared_memory_object(bip::open_or_create, probeName.c_str(), bip::read_write);
                shm.truncate(static_cast<bip::offset_t>(allocSize));    // NOLINT
                bip::shared_memory_object::remove(probeName.c_str());
                fmt::print(stderr, "[shm_ring_buffer] '{}': shm budget {} bytes (cap {}, floor {})\n", name, allocSize, maxShmSize, floor);
                return allocSize;
            } catch (bip::interprocess_exception const& e)
            {
                bip::shared_memory_object::remove(probeName.c_str());
                if (e.get_native_error() != ENOSPC) { throw; }
                if (allocSize <= floor)
                {
                    throw std::runtime_error(
                        fmt::format("Cannot allocate shared memory '{}': no space for the {} byte minimum", name, floor));
                }
                size_t const next = std::max(allocSize / 2, floor);
                fmt::print(stderr, "[shm_ring_buffer] '{}': {} bytes not allocatable (ENOSPC), retrying with {}\n", name, allocSize, next);
                allocSize = next;
            }
        }
    }

    [[nodiscard]] size_t SlotSize() const noexcept { return _slotSize; }
    [[nodiscard]] size_t SlotCount() const noexcept { return _slotCount; }
    [[nodiscard]] size_t HeaderSize() const noexcept { return HeaderAllocationSize(_slotCount); }

    CLASS_DELETE_COPY_DEFAULT_MOVE(ShmRingBuffer);
    ~ShmRingBuffer()
    {
        if (_resetShm)
        {
            Stop();
            bip::shared_memory_object::remove(_name.c_str());
        }
        else
        {
            Flush();
        }
    }

    void Stop()
    {
        // Removing this can potentially cause a deadlock
        bip::scoped_lock<bip::interprocess_mutex> lock(_ctrl->mutex);
        _stopRequested = true;
        _ctrl->canWrite.notify_all();
        _ctrl->canRead.notify_all();
    }

    void Push(THdr const& hdr, char const* data, size_t size)
    {
#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS
        auto start = PerfMetrics::Clock::now();
#endif
        if (size > _slotSize) { throw std::runtime_error("Message exceeds shared buffer slot capacity"); }
        auto slot   = StartWriting_();
        _hdrs[slot] = hdr;
        std::memcpy(SlotPtr_(slot), data, size);
        FinishWriting_(slot);
#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS
        sendPerf.Record(PerfMetrics::Clock::now() - start);
#endif
    }

    std::pair<size_t, char*> AllocSlot(THdr const& hdr)
    {
        auto slot   = StartWriting_();
        _hdrs[slot] = hdr;
        return {slot, SlotPtr_(slot)};
    }

    void CommitSlot(size_t slot) { FinishWriting_(slot); }

    std::tuple<bool, THdr const&, std::span<char const>> BeginRead()
    {
#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS
        readstart = PerfMetrics::Clock::now();
#endif
        auto [stopRequested, slot] = StartReading_();
        return {stopRequested, _hdrs[slot], std::span<char const>{SlotPtr_(slot), _slotSize}};
    }

    void EndRead()
    {
        FinishReading_();
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS and SHM_RING_BUFFER_ENABLE_PERF_METRICS == 2
        if (recvPerf.Record(PerfMetrics::Clock::now() - readstart))
        {
            // Print Read-Head of queue for debugging.
            fmt::print("[shm_buf][perf][VERBOSE] Queue: {} Hdr: {}\n", *this, _hdrs[_ctrl->rbeg % _slotCount]);
        }
#endif
    }

    void Heartbeat() { _ctrl->rtimestamp = Control::Clock::now(); }

    // The extent the ring actually uses (control + headers + slots), for callers that
    // NUMA-bind or cudaHostRegister the buffer — it must never exceed the published
    // geometry, which bounds what a client may touch and pin.
    std::span<char> GetShmDataRegion()
    {
        return {reinterpret_cast<char*>(_region.get_address()), HeaderSize() + (_slotCount * _slotSize)};    // NOLINT
    }

    void Flush()
    {
        auto start      = PerfMetrics::Clock::now();
        auto rtimestamp = _ctrl->rtimestamp.load();
        while (AvailableForWrite_() < _slotCount)
        {
            std::this_thread::sleep_for(HearBeatInterval);
            if (_ctrl->rtimestamp.load() != rtimestamp)
            {
                rtimestamp = _ctrl->rtimestamp.load();
                start      = PerfMetrics::Clock::now();
            }
            else if (PerfMetrics::Clock::now() - start > _writeWaitTimeout)
            {
                throw std::runtime_error(fmt::format("Flush: No heartbeat from reader, possible deadlock Queue: {} Queue-Head-Hdr: {}\n",
                                                     *this,
                                                     _hdrs[_ctrl->wbeg % _slotCount]));
            }
        }
    }

    private:
    size_t AvailableForRead_() { return _ctrl->rend - _ctrl->rbeg; }
    size_t AvailableForWrite_() { return _ctrl->wend - _ctrl->wbeg; }
    char*  SlotPtr_(size_t slot) { return _dataBase + (slot * _slotSize); }    // NOLINT

    size_t StartWriting_()
    {

        bip::scoped_lock<bip::interprocess_mutex> lock(_ctrl->mutex);
        while (!_stopRequested && AvailableForWrite_() == 0)
        {
#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS
            auto start = PerfMetrics::Clock::now();
#if SHM_RING_BUFFER_ENABLE_PERF_METRICS == 2
            // Print Write-Head of queue for debugging.
            fmt::print("[shm_buf][perf][VERBOSE] Queue Full: {} Queue-Head-Hdr: {}\n", *this, _hdrs[_ctrl->wbeg % _slotCount]);
#endif
#endif

            auto rtimestamp = _ctrl->rtimestamp.load();
            // This feels like a hack but is necessary because we dont always notify the reader after every write op
            // So its important to do so right before we wait
            _ctrl->canRead.notify_all();
            auto valid = _ctrl->canWrite.wait_for(lock, _writeWaitTimeout, [this] { return AvailableForWrite_() > 0 || _stopRequested; });
#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS
            wlatency.Record(PerfMetrics::Clock::now() - start);
#endif
            // if heartbeat stopped then raise exception
            if (!valid && _ctrl->rtimestamp.load() == rtimestamp)
            {
                throw std::runtime_error(fmt::format(
                    "No heartbeat from reader, possible deadlock Queue: {} Queue-Head-Hdr: {}\n", *this, _hdrs[_ctrl->wbeg % _slotCount]));
            }

            rtimestamp = _ctrl->rtimestamp.load();
        }
        if (_stopRequested) { throw StopRequestedException(); }

        size_t offset = _ctrl->wbeg;
        _ctrl->wbeg++;
        return offset % _slotCount;
    }

    void FinishWriting_(size_t slot)
    {
        // bool readFull = false;
        {
            bip::scoped_lock<bip::interprocess_mutex> lock(_ctrl->mutex);
            while ((_ctrl->rend % _slotCount) != slot)
            {
                lock.unlock();
                std::this_thread::yield();
                lock.lock();
            }    // TODO : something better than spin wait
            // readFull = AvailableForRead_() == 0;
            if (AvailableForRead_() == 0)
            {
                _ctrl->wtimestamp = Control::Clock::now();
                _ctrl->canRead.notify_all();
            }

            _ctrl->rend++;
        }
    }

    std::tuple<bool, size_t> StartReading_()
    {
        bip::scoped_lock<bip::interprocess_mutex> lock(_ctrl->mutex);
        while (AvailableForRead_() == 0)
        {
            if (_stopRequested) { return {true, 0}; }
            // This feels like a hack but is necessary because we dont always notify the writer on every read op
            // So its important to do so right before we wait
            _ctrl->rtimestamp = Control::Clock::now();
            _ctrl->canWrite.notify_all();
            _ctrl->canRead.wait_for(lock, HearBeatInterval, [this] { return AvailableForRead_() > 0 || _stopRequested; });
#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS
            if (AvailableForRead_() > 0)
            {
                rlatency.Record(Control::Clock::now() - _ctrl->wtimestamp.load());    //
            }
#endif
        }
        // Flush the queue before stopping to avoid leaving the other side hanging
        // if (_stopRequested) { throw StopRequestedException(); }
        _ctrl->rtimestamp = Control::Clock::now();
        size_t offset     = _ctrl->rbeg;
        _ctrl->rbeg++;
        return {false, offset % _slotCount};
    }

    void FinishReading_()
    {
        // bool writeFull = false;
        {
            bip::scoped_lock<bip::interprocess_mutex> lock(_ctrl->mutex);
            //_ctrl->timestamp = static_cast<uint64_t>(
            //    std::chrono::duration_cast<std::chrono::nanoseconds>(PerfMetrics::Clock::now().time_since_epoch()).count());

            if (AvailableForWrite_() == 0)
            {
                //_ctrl->wtimestamp = Control::Clock::now();
                _ctrl->canWrite.notify_all();
            }
            _ctrl->wend++;
        }
    }

    ShmRingBuffer(std::string&& nameIn, bool resetState, size_t allocSize, size_t slotCount) :
        _name(std::move(nameIn)), _slotCount(slotCount), _resetShm(resetState)
    {
        if (_resetShm)
        {
            auto const minimumShmSize = MinimumShmSize(_slotCount);
            if (allocSize < minimumShmSize)
            {
                throw std::invalid_argument(
                    fmt::format("Shared memory '{}': requested {} bytes, need at least {}", _name, allocSize, minimumShmSize));
            }
            _shm = bip::shared_memory_object(bip::open_or_create, _name.c_str(), bip::read_write);
            _shm.truncate(static_cast<bip::offset_t>(allocSize));

            _region    = bip::mapped_region(_shm, bip::read_write);
            auto* base = static_cast<char*>(_region.get_address());
            _ctrl      = reinterpret_cast<Control*>(base);                                                // NOLINT
            _hdrs      = std::span<THdr>(reinterpret_cast<THdr*>(base + sizeof(Control)), _slotCount);    // NOLINT
            _slotSize  = ((allocSize - HeaderSize()) / _slotCount) & ~(AlignBoundary - 1);
            _dataBase  = base + HeaderSize();    // NOLINT
            assert(_slotSize >= AlignBoundary);
            assert(_region.get_size() >= HeaderSize() + (_slotCount * _slotSize));

            new (&_ctrl->mutex) bip::interprocess_mutex();
            new (&_ctrl->canRead) bip::interprocess_condition();
            new (&_ctrl->canWrite) bip::interprocess_condition();

            _ctrl->rbeg       = 0;
            _ctrl->rend       = 0;
            _ctrl->wbeg       = 0;
            _ctrl->wend       = _slotCount;
            _ctrl->shmSize    = allocSize;
            _ctrl->slotSize   = _slotSize;
            _ctrl->slotCount  = _slotCount;
            _ctrl->rtimestamp = Control::Clock::now();
            _ctrl->wtimestamp = Control::Clock::now();
        }
        else
        {
            auto                  start           = PerfMetrics::Clock::now();
            static constexpr auto MaxWaitDuration = std::chrono::seconds(10);

            while (true)
            {
                auto iterationStart = PerfMetrics::Clock::now();
                if (iterationStart - start > MaxWaitDuration)
                {
                    throw std::runtime_error(
                        fmt::format("Failed to open shared memory after multiple attempts over {} seconds",
                                    std::chrono::duration_cast<std::chrono::seconds>(PerfMetrics::Clock::now() - start).count()));
                }

                _shm                     = bip::shared_memory_object(bip::open_or_create, _name.c_str(), bip::read_write);
                bip::offset_t regionSize = 0;
                if (!_shm.get_size(regionSize) || std::cmp_less(regionSize, sizeof(Control)))
                {
                    std::this_thread::sleep_for(HearBeatInterval);
                    continue;
                }
                _region    = bip::mapped_region(_shm, bip::read_write);
                auto* base = static_cast<char*>(_region.get_address());
                _ctrl      = reinterpret_cast<Control*>(base);    // NOLINT
                // Published queue geometry is the readiness signal for concurrent initialization.
                while (true)
                {
                    [[maybe_unused]] auto const seen = _ctrl->rtimestamp.load();    // acquire: orders the geometry reads below
                    if (_ctrl->slotSize != 0 && _ctrl->shmSize != 0 && _ctrl->slotCount != 0) { break; }
                    if (PerfMetrics::Clock::now() - start >= MaxWaitDuration) { break; }
                    std::this_thread::yield();
                }
                if (PerfMetrics::Clock::now() - start > MaxWaitDuration) { continue; }

                AdoptPublishedGeometry_();
                break;
            }
        }
    }

    // Adopt the slot geometry the creator published in Control, remapping to exactly
    // the creator's shmSize. The file size observed at open time is not authoritative
    // — only the published geometry bounds what this process may map, touch, and pin.
    // Throws logic_error on any inconsistency.
    void AdoptPublishedGeometry_()
    {
        uint64_t const shmSize   = _ctrl->shmSize;
        uint64_t const slotSize  = _ctrl->slotSize;
        uint64_t const slotCount = _ctrl->slotCount;
        if (slotSize == 0 || (slotSize % AlignBoundary) != 0)
        {
            throw std::logic_error(fmt::format("Shared memory '{}': invalid slot size {} reported by creator", _name, slotSize));
        }
        if (slotCount == 0 || slotCount > MaxSlotCount)
        {
            throw std::logic_error(fmt::format("Shared memory '{}': invalid slot count {} reported by creator", _name, slotCount));
        }
        _slotCount = static_cast<size_t>(slotCount);
        if (shmSize < HeaderSize() + (_slotCount * slotSize))
        {
            throw std::logic_error(
                fmt::format("Shared memory '{}': creator-reported size {} too small for {} slots of {} bytes plus {} byte header",
                            _name,
                            shmSize,
                            _slotCount,
                            slotSize,
                            HeaderSize()));
        }
        bip::offset_t fileSize = 0;
        if (!_shm.get_size(fileSize) || std::cmp_less(fileSize, shmSize))
        {
            throw std::logic_error(
                fmt::format("Shared memory '{}': file size {} is less than creator-reported shmSize {}", _name, fileSize, shmSize));
        }
        if (_region.get_size() != shmSize) { _region = bip::mapped_region(_shm, bip::read_write, 0, shmSize); }
        auto* base = static_cast<char*>(_region.get_address());
        _ctrl      = reinterpret_cast<Control*>(base);    // NOLINT
        _slotCount = slotCount;
        _slotSize  = slotSize;
        _hdrs      = std::span<THdr>(reinterpret_cast<THdr*>(base + sizeof(Control)), _slotCount);    // NOLINT
        _dataBase  = base + HeaderSize();                                                             // NOLINT
    }

    std::string               _name;
    bip::shared_memory_object _shm;
    bip::mapped_region        _region;
    Control*                  _ctrl{};
    char*                     _dataBase{};    // start of the slot data region
    size_t                    _slotSize{};    // runtime per-slot stride, from Control::slotSize
    size_t                    _slotCount{};
    std::chrono::seconds      _writeWaitTimeout{checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferWriteTimeoutSeconds>()};
    std::span<THdr>           _hdrs;
    bool                      _stopRequested{false};
    bool                      _resetShm{false};
#ifdef SHM_RING_BUFFER_ENABLE_PERF_METRICS
    PerfMetrics::Clock::time_point readstart;
#endif
};
}    // namespace checkpoint::ipc

template <typename THdr> struct fmt::formatter<checkpoint::ipc::ShmRingBuffer<THdr>> : fmt::formatter<std::string_view>
{
    auto format(checkpoint::ipc::ShmRingBuffer<THdr> const& obj, fmt::format_context& ctx) const    // NOLINT
    {
        return obj.format_to(ctx);    //
    }
};

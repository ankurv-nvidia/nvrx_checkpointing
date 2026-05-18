#pragma once
#include "CommonMacros.h"

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
#include <chrono>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

namespace checkpoint::ipc
{
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

#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS

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
template <typename THdr, size_t SlotSize, size_t NSlots> struct ShmRingBuffer
{
    public:
    static constexpr auto HearBeatInterval = std::chrono::microseconds(1);
#ifdef PHOENIX_BUILD_FOR_VALGRIND
    static constexpr auto WriteWaitTimeout = std::chrono::minutes(5);
#else
    static constexpr auto WriteWaitTimeout = std::chrono::seconds(1);
#endif

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
        std::atomic<std::size_t>       wend{NSlots};    // Initially set to NSlots

        bip::interprocess_mutex     mutex;
        bip::interprocess_condition canRead;     // Signaled when data is available to read
        bip::interprocess_condition canWrite;    // Signaled when space is available to write
    };

    auto format_to(fmt::format_context& ctx) const    // NOLINT
    {
        return fmt::format_to(ctx.out(),
                              "ShmRingBuffer[{}]: rbeg={} rend={} wbeg={} wend={}{}{}{}",
                              NSlots,
                              _ctrl->rbeg,
                              _ctrl->rend,
                              _ctrl->wbeg,
                              _ctrl->wend,
                              sendPerf.count == 0 ? "" : fmt::format("{}", sendPerf),
                              recvPerf.count == 0 ? "" : fmt::format("{}", recvPerf),
                              rlatency.count == 0 ? "" : fmt::format("{}", rlatency),
                              wlatency.count == 0 ? "" : fmt::format("{}", wlatency));
    }

    static constexpr size_t AlignBoundary = 4096;
    static_assert(SlotSize % AlignBoundary == 0, "SlotSize must be multiple of AlignBoundary");

    static constexpr size_t DataCapacity = NSlots * SlotSize;
    static constexpr size_t HdrCapacity  = (sizeof(THdr) * NSlots) + sizeof(Control);
    static constexpr size_t HdrAllocSize = ((HdrCapacity + AlignBoundary - 1) / AlignBoundary) * AlignBoundary;

    static constexpr size_t TotalShmSize = HdrAllocSize + DataCapacity;

    static ShmRingBuffer Create(std::string name) { return {std::move(name), true}; }

    static ShmRingBuffer Open(std::string name) { return {std::move(name), false}; }

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
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS
        auto start = PerfMetrics::Clock::now();
#endif
        if (size > SlotSize) { throw std::runtime_error("Message exceeds shared buffer slot capacity"); }
        auto slot   = StartWriting_();
        _hdrs[slot] = hdr;
        std::memcpy(_data[slot].data(), data, size);
        FinishWriting_(slot);
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS
        sendPerf.Record(PerfMetrics::Clock::now() - start);
#endif
    }

    std::pair<size_t, char*> AllocSlot(THdr const& hdr)
    {
        auto slot   = StartWriting_();
        _hdrs[slot] = hdr;
        return {slot, _data[slot].data()};
    }

    void CommitSlot(size_t slot) { FinishWriting_(slot); }

    std::tuple<bool, THdr const&, std::array<char, SlotSize> const&> BeginRead()
    {
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS
        readstart = PerfMetrics::Clock::now();
#endif
        auto [stopRequested, slot] = StartReading_();
        return {stopRequested, _hdrs[slot], _data[slot]};
    }

    void EndRead()
    {
        FinishReading_();
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS and SHM_RING_BUFFER_ENABLE_PERF_METRICS == 2
        if (recvPerf.Record(PerfMetrics::Clock::now() - readstart))
        {
            // Print Read-Head of queue for debugging.
            fmt::print("[shm_buf][perf][VERBOSE] Queue: {} Hdr: {}\n", *this, _hdrs[_ctrl->rbeg % NSlots]);
        }
#endif
    }

    void Heartbeat() { _ctrl->rtimestamp = Control::Clock::now(); }

    // In case the user wants to pin the buffer to devices like cuda
    std::span<char> GetShmDataRegion()
    {
        return {reinterpret_cast<char*>(_region.get_address()), _region.get_size()};    // NOLINT
    }

    void Flush()
    {
        while (AvailableForWrite_() < NSlots) { std::this_thread::sleep_for(HearBeatInterval); }
    }

    private:
    size_t AvailableForRead_() { return _ctrl->rend - _ctrl->rbeg; }
    size_t AvailableForWrite_() { return _ctrl->wend - _ctrl->wbeg; }

    size_t StartWriting_()
    {

        bip::scoped_lock<bip::interprocess_mutex> lock(_ctrl->mutex);
        while (!_stopRequested && AvailableForWrite_() == 0)
        {
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS
            auto start = PerfMetrics::Clock::now();
#if SHM_RING_BUFFER_ENABLE_PERF_METRICS == 2
            // Print Write-Head of queue for debugging.
            fmt::print("[shm_buf][perf][VERBOSE] Queue Full: {} Queue-Head-Hdr: {}\n", *this, _hdrs[_ctrl->wbeg % NSlots]);
#endif
#endif

            auto rtimestamp = _ctrl->rtimestamp.load();
            // This feels like a hack but is necessary because we dont always notify the reader after every write op
            // So its important to do so right before we wait
            _ctrl->canRead.notify_all();
            auto valid = _ctrl->canWrite.wait_for(lock, WriteWaitTimeout, [this]() { return AvailableForWrite_() > 0 || _stopRequested; });
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS
            wlatency.Record(PerfMetrics::Clock::now() - start);
#endif
            // if heartbeat stopped then raise exception
            if (!valid && _ctrl->rtimestamp.load() == rtimestamp)
            {
                throw std::runtime_error(fmt::format(
                    "No heartbeat from reader, possible deadlock Queue: {} Queue-Head-Hdr: {}\n", *this, _hdrs[_ctrl->wbeg % NSlots]));
            }

            rtimestamp = _ctrl->rtimestamp.load();
        }
        if (_stopRequested) { throw StopRequestedException(); }

        size_t offset = _ctrl->wbeg;
        _ctrl->wbeg++;
        return offset % NSlots;
    }

    void FinishWriting_(size_t slot)
    {
        // bool readFull = false;
        {
            bip::scoped_lock<bip::interprocess_mutex> lock(_ctrl->mutex);
            while ((_ctrl->rend % NSlots) != slot)
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::nanoseconds(10));
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
            _ctrl->canRead.wait_for(lock, HearBeatInterval, [this]() { return AvailableForRead_() > 0 || _stopRequested; });
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS
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
        return {false, offset % NSlots};
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

    ShmRingBuffer(std::string&& nameIn, bool resetState) : _name(std::move(nameIn)), _resetShm(resetState)
    {
        if (_resetShm)
        {

            _shm = bip::shared_memory_object(bip::open_or_create, _name.c_str(), bip::read_write);
            _shm.truncate(static_cast<bip::offset_t>(TotalShmSize));
            _region    = bip::mapped_region(_shm, bip::read_write);
            auto* base = static_cast<char*>(_region.get_address());
            _ctrl      = reinterpret_cast<Control*>(base);                                                       // NOLINT
            _hdrs      = std::span<THdr>(reinterpret_cast<THdr*>(base + sizeof(Control)), NSlots);               // NOLINT
            _data      = std::span<std::array<char, SlotSize>>(reinterpret_cast<std::array<char, SlotSize>*>(    // NOLINT
                                                                   base + HdrAllocSize),
                                                               NSlots);
            assert(NSlots == std::min<std::size_t>(NSlots, _region.get_size() - sizeof(Control)));

            new (&_ctrl->mutex) bip::interprocess_mutex();
            new (&_ctrl->canRead) bip::interprocess_condition();
            new (&_ctrl->canWrite) bip::interprocess_condition();

            _ctrl->rbeg       = 0;
            _ctrl->rend       = 0;
            _ctrl->wbeg       = 0;
            _ctrl->wend       = NSlots;
            _ctrl->rtimestamp = Control::Clock::now();
            _ctrl->wtimestamp = Control::Clock::now();
        }
        else
        {
            auto                  start           = PerfMetrics::Clock::now();
            static constexpr auto MaxWaitDuration = std::chrono::seconds(5);

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
                if (!_shm.get_size(regionSize) || regionSize <= 0)
                {
                    std::this_thread::sleep_for(HearBeatInterval);
                    continue;
                }
                if (static_cast<size_t>(regionSize) < TotalShmSize)
                {
                    throw std::logic_error(
                        fmt::format("Invalid state for shared_memory. expected-size({}) != actual-size({})", TotalShmSize, regionSize));
                }
                _region         = bip::mapped_region(_shm, bip::read_write);
                auto* base      = static_cast<char*>(_region.get_address());
                _ctrl           = reinterpret_cast<Control*>(base);    // NOLINT
                auto rtimestamp = _ctrl->rtimestamp.load();

                _hdrs = std::span<THdr>(reinterpret_cast<THdr*>(base + sizeof(Control)), NSlots);               // NOLINT
                _data = std::span<std::array<char, SlotSize>>(reinterpret_cast<std::array<char, SlotSize>*>(    // NOLINT
                                                                  base + HdrAllocSize),
                                                              NSlots);
                assert(NSlots == std::min<std::size_t>(NSlots, _region.get_size() - sizeof(Control)));
                while (_ctrl->rtimestamp.load() == rtimestamp && PerfMetrics::Clock::now() - start < MaxWaitDuration)
                {
                    std::this_thread::yield();
                }
                if (PerfMetrics::Clock::now() - start > MaxWaitDuration) { continue; }
                break;
            }
        }
    }

    std::string                           _name;
    bip::shared_memory_object             _shm;
    bip::mapped_region                    _region;
    Control*                              _ctrl{};
    std::span<std::array<char, SlotSize>> _data;
    std::span<THdr>                       _hdrs;
    bool                                  _stopRequested{false};
    bool                                  _resetShm{false};
#if defined SHM_RING_BUFFER_ENABLE_PERF_METRICS
    PerfMetrics::Clock::time_point readstart;
#endif
};
}    // namespace checkpoint::ipc

template <typename THdr, size_t SlotSize, size_t NSlots>
struct fmt::formatter<checkpoint::ipc::ShmRingBuffer<THdr, SlotSize, NSlots>> : fmt::formatter<std::string_view>
{
    auto format(checkpoint::ipc::ShmRingBuffer<THdr, SlotSize, NSlots> const& obj, fmt::format_context& ctx) const    // NOLINT
    {
        return obj.format_to(ctx);    //
    }
};

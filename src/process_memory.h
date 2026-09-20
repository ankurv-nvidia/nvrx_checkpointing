#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

struct ProcessMemUsageInfo
{
    uint64_t rssBytes{};
    uint64_t peakRssBytes{};
    uint64_t anonymousRssBytes{};
    uint64_t fileRssBytes{};
    uint64_t shmemRssBytes{};
    uint64_t lockedBytes{};
    uint64_t pinnedBytes{};
    uint64_t swapBytes{};

    static auto Get()
    {
        ProcessMemUsageInfo usage{};

        std::ifstream status{"/proc/self/status"};
        if (!status) { return usage; }
        std::string line;
        while (std::getline(status, line))
        {
            auto const colon = line.find(':');
            if (colon == std::string_view::npos) { continue; }

            auto const key   = std::string_view{line}.substr(0, colon);
            auto const value = std::string_view{line}.substr(colon + 1);
            if (key == "VmRSS") { usage.rssBytes = ParseBytes_(value); }
            else if (key == "VmHWM") { usage.peakRssBytes = ParseBytes_(value); }
            else if (key == "RssAnon") { usage.anonymousRssBytes = ParseBytes_(value); }
            else if (key == "RssFile") { usage.fileRssBytes = ParseBytes_(value); }
            else if (key == "RssShmem") { usage.shmemRssBytes = ParseBytes_(value); }
            else if (key == "VmLck") { usage.lockedBytes = ParseBytes_(value); }
            else if (key == "VmPin") { usage.pinnedBytes = ParseBytes_(value); }
            else if (key == "VmSwap") { usage.swapBytes = ParseBytes_(value); }
        }
        return usage;
    }

    private:
    static uint64_t ParseBytes_(std::string_view const& value)
    {
        auto const begin = value.find_first_not_of(" \t");
        if (begin == std::string_view::npos) { throw std::invalid_argument("missing process-memory value"); }
        auto const end = value.find_first_not_of("0123456789", begin);
        uint64_t   amount{};
        auto const result
            = std::from_chars(value.data() + begin, value.data() + (end == std::string_view::npos ? value.size() : end), amount);
        if (result.ec != std::errc{}) { throw std::invalid_argument("invalid process-memory value"); }
        auto const unitStart = end == std::string_view::npos ? value.size() : value.find_first_not_of(" \t", end);
        if (unitStart != std::string_view::npos && value.substr(unitStart) != "kB")
        {
            throw std::invalid_argument("unexpected process-memory unit");
        }
        static constexpr uint64_t BytesPerKiB = 1024;
        if (amount > std::numeric_limits<uint64_t>::max() / BytesPerKiB) { throw std::overflow_error("process-memory value overflow"); }
        return amount * BytesPerKiB;
    }
};

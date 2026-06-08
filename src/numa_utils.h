#pragma once
#include "CommonMacros.h"

#include <hwloc.h>

#include <fmt/format.h>

#include <array>
#include <string>

// Process-lifetime hwloc topology, loaded once. hwloc_topology_load scans /sys,
// so it must not run per-operation (e.g. on the checkpoint server's per-snapshot
// hot path). Query/membind calls on a loaded topology are read-only and safe to
// use concurrently from multiple threads. Never destroyed (singleton).
inline hwloc_topology_t HwlocTopology()
{
    // The cache is per-process; a separate copy in the client .so and the server
    // executable is fine (both read-only), so the duplication warning is benign.
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wunique-object-duplication")
    static hwloc_topology_t topo = [] {
        hwloc_topology_t t{};
        hwloc_topology_init(&t);
        hwloc_topology_load(t);
        return t;
    }();
    SUPPRESS_WARNINGS_END
    return topo;
}

inline uint8_t DetectNumaNodeCount()
{
    int count = hwloc_get_nbobjs_by_type(HwlocTopology(), HWLOC_OBJ_NUMANODE);
    return count > 0 ? static_cast<uint8_t>(count) : 1;
}

// Bind a mapped range to `node` so its pages land there regardless of which
// thread/process first touches them (per-VMA; best-effort).
inline void BindRegionToNumaNode(int node, void* addr, size_t size)
{
    if (node < 0 || addr == nullptr || size == 0) { return; }
    hwloc_topology_t topo = HwlocTopology();
    if (hwloc_obj_t obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, static_cast<unsigned>(node));
        obj != nullptr && obj->nodeset != nullptr)
    {
        (void)hwloc_set_area_membind(topo, addr, size, obj->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
    }
}

inline int GetCurrentNumaNode()
{
    hwloc_topology_t topo      = HwlocTopology();
    auto             numaCount = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    if (numaCount <= 1) { return 0; }
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    auto           cleanup = [&] { hwloc_bitmap_free(cpuset); };
    auto getCpusetStr = [&]() {
        std::array<char, 256> buf{};
        hwloc_bitmap_list_snprintf(buf.data(), buf.size(), cpuset);
        return std::string(buf.data());
    };
    if (hwloc_get_last_cpu_location(topo, cpuset, HWLOC_CPUBIND_THREAD) != 0)
    {
        auto err = fmt::format(
            "GetCurrentNumaNode: hwloc_get_last_cpu_location failed. numaCount={} errno={} cpuset={}", numaCount, errno, getCpusetStr());
        cleanup();
        throw std::logic_error(err);
    }
    auto        cpusetStr = getCpusetStr();
    hwloc_obj_t obj       = nullptr;
    while ((obj = hwloc_get_next_obj_by_type(topo, HWLOC_OBJ_NUMANODE, obj)) != nullptr)
    {
        if (obj->cpuset != nullptr && (hwloc_bitmap_intersects(cpuset, obj->cpuset) != 0))
        {
            auto node = static_cast<int>(obj->os_index);
            cleanup();
            return node;
        }
    }
    cleanup();
    throw std::logic_error(
        fmt::format("GetCurrentNumaNode: No NUMA node intersects current CPU location. numaCount={} cpuset={}", numaCount, cpusetStr));
}

#pragma once
#include "CommonMacros.h"

#include <hwloc.h>

#include <fmt/format.h>

#include <array>
#include <string>
#include <vector>

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

// Resolve a NUMA node by its OS index. `node` is an OS index (e.g. the GPU's
// /sys numa_node), which differs from hwloc's logical index when GPU-memory NUMA
// nodes are present (GB200), so hwloc_get_obj_by_type (logical) must not be used.
inline hwloc_obj_t NumaNodeByOsIndex(hwloc_topology_t topo, int node)
{
    hwloc_obj_t obj = nullptr;
    while ((obj = hwloc_get_next_obj_by_type(topo, HWLOC_OBJ_NUMANODE, obj)) != nullptr)
    {
        if (obj->os_index == static_cast<unsigned>(node)) { return obj; }
    }
    return nullptr;
}

// Pin the calling thread's CPU to NUMA node `node` (OS index, best-effort). The
// checkpoint server pins each queue's dequeue thread to the domain of the ranks
// it serves so the shm->blob memcpy runs on-socket.
inline void BindThreadCpuToNumaNode(int node)
{
    if (node < 0) { return; }
    hwloc_topology_t topo = HwlocTopology();
    if (hwloc_obj_t obj = NumaNodeByOsIndex(topo, node); obj != nullptr && obj->cpuset != nullptr)
    {
        (void)hwloc_set_cpubind(topo, obj->cpuset, HWLOC_CPUBIND_THREAD);
    }
}

// Restore the calling thread to run on all cores (undo a prior cpu pin).
inline void UnbindThreadCpu()
{
    hwloc_topology_t topo = HwlocTopology();
    if (hwloc_obj_t root = hwloc_get_root_obj(topo); root != nullptr && root->cpuset != nullptr)
    {
        (void)hwloc_set_cpubind(topo, root->cpuset, HWLOC_CPUBIND_THREAD);
    }
}

// Return NUMA nodes that can both run a CPU thread and back host allocations.
// A NUMA object's cpuset expresses CPU locality, so it may be non-empty even
// when that memory target has no local capacity.
inline std::vector<int> DetectHostMemoryNumaNodes()
{
    hwloc_topology_t topo = HwlocTopology();
    std::vector<int> nodes;
    int              fallbackNode = 0;
    bool             hasFallback  = false;
    hwloc_obj_t      obj          = nullptr;
    while ((obj = hwloc_get_next_obj_by_type(topo, HWLOC_OBJ_NUMANODE, obj)) != nullptr)
    {
        bool const hasCpus = obj->cpuset != nullptr && hwloc_bitmap_iszero(obj->cpuset) == 0;
        if (!hasCpus) { continue; }
        if (!hasFallback)
        {
            fallbackNode = static_cast<int>(obj->os_index);
            hasFallback  = true;
        }
        if (obj->attr != nullptr && obj->attr->numanode.local_memory > 0) { nodes.push_back(static_cast<int>(obj->os_index)); }
    }
    if (nodes.empty()) { nodes.push_back(fallbackNode); }
    return nodes;
}

inline uint8_t DetectNumaNodeCount()
{ return static_cast<uint8_t>(DetectHostMemoryNumaNodes().size()); }

// Migrate a mapped range to `node` (OS index) so its pages are local to the GPU
// that copies into it, regardless of which thread/process first touched them.
// MIGRATE moves already-resident pages (the shm region is server-created), which
// a bind-only policy would leave in place.
inline void BindRegionToNumaNode(int node, void* addr, size_t size)
{
    if (node < 0 || addr == nullptr || size == 0) { return; }
    // Single NUMA domain: pages are already local, so the migrate is pure cost.
    if (DetectNumaNodeCount() <= 1) { return; }
    hwloc_topology_t topo = HwlocTopology();
    if (hwloc_obj_t obj = NumaNodeByOsIndex(topo, node); obj != nullptr && obj->nodeset != nullptr)
    {
        (void)hwloc_set_area_membind(topo, addr, size, obj->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_MIGRATE);
    }
}

inline int GetCurrentNumaNode()
{
    hwloc_topology_t topo      = HwlocTopology();
    auto             numaCount = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    if (numaCount <= 1) { return 0; }
    hwloc_cpuset_t cpuset       = hwloc_bitmap_alloc();
    auto           cleanup      = [&] { hwloc_bitmap_free(cpuset); };
    auto           getCpusetStr = [&]() {
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

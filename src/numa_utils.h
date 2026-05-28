#pragma once
#include "CommonMacros.h"

#include <hwloc.h>

#include <fmt/format.h>

#include <array>
#include <string>

inline uint8_t DetectNumaNodeCount()
{
    hwloc_topology_t topo{};
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int count = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    hwloc_topology_destroy(topo);
    return count > 0 ? static_cast<uint8_t>(count) : 1;
}

inline int GetCurrentNumaNode()
{
    struct Scope
    {
        Scope()
        {
            hwloc_topology_init(&topo);
            hwloc_topology_load(topo);
        }
        ~Scope()
        {
            hwloc_topology_destroy(topo);
            if (cpuset != nullptr) hwloc_bitmap_free(cpuset);
        }
        CLASS_DELETE_COPY_AND_MOVE(Scope);
        hwloc_topology_t topo{};
        hwloc_cpuset_t   cpuset{};
    };

    Scope scope{};
    auto  numaCount = hwloc_get_nbobjs_by_type(scope.topo, HWLOC_OBJ_NUMANODE);
    if (numaCount <= 1) { return 0; }
    scope.cpuset      = hwloc_bitmap_alloc();
    auto getCpusetStr = [&]() {
        std::array<char, 256> buf{};
        hwloc_bitmap_list_snprintf(buf.data(), buf.size(), scope.cpuset);
        return std::string(buf.data());
    };
    if (hwloc_get_last_cpu_location(scope.topo, scope.cpuset, HWLOC_CPUBIND_THREAD) != 0)
    {
        throw std::logic_error(fmt::format(
            "GetCurrentNumaNode: hwloc_get_last_cpu_location failed. numaCount={} errno={} cpuset={}", numaCount, errno, getCpusetStr()));
    }
    auto        cpusetStr = getCpusetStr();
    hwloc_obj_t obj       = nullptr;
    while ((obj = hwloc_get_next_obj_by_type(scope.topo, HWLOC_OBJ_NUMANODE, obj)) != nullptr)
    {
        if (obj->cpuset != nullptr && (hwloc_bitmap_intersects(scope.cpuset, obj->cpuset) != 0)) { return static_cast<int>(obj->os_index); }
    }
    throw std::logic_error(
        fmt::format("GetCurrentNumaNode: No NUMA node intersects current CPU location. numaCount={} cpuset={}", numaCount, cpusetStr));
}

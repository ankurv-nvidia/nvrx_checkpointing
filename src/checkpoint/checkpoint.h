#pragma once
#include "CommonMacros.h"

#include "messages.h"
#include "state.h"
#include "tensor.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

void SvcIncrRef();
void SvcDecrRef();

[[noreturn]] void LoadCheckpoint(StateTree& root, std::filesystem::path const& ckptPath);

void SaveCheckpoint(StateTree const&                         root,
                    checkpoint::ipc::SnapshotId const&       snapshotId,
                    std::filesystem::path const&             ckptPath,
                    std::span<StateTree::SnapshotBlob const> snapshotBlobs = {});
void FinalizeStateRegistry(StateTree& self);
void WaitForCheckpointSave(RankTopology const& topo);

// Warm up the MsgQueue pinned buffers for CUDA D2H
void WarmUpCheckpoint(RankTopology const& topo);
void SvcPing();
void SvcStop();

// Graceful shutdown: ask the server to stop once every session is idle, blocking until the
// spin-down completes.
void SvcSpinDown();
struct DeviceContext
{
    DeviceContext(DeviceId deviceIn, uint8_t streamIdxIn) : device(deviceIn), streamIdx(streamIdxIn) {}
    virtual ~DeviceContext() = default;
    CLASS_DELETE_COPY_AND_MOVE(DeviceContext);

    virtual uint64_t Allocate(uint64_t size)            = 0;
    virtual void     Free(uint64_t deviceAddr)          = 0;
    virtual void*    AllocHostPinnedBuffer(size_t size) = 0;
    virtual void     FreeHostPinnedBuffer(void* ptr)    = 0;

    virtual void StreamAcquire()                                                        = 0;
    virtual void AddD2H(uint64_t srcDeviceAddr, void* dstHostBuffer, size_t size)       = 0;
    virtual void AddH2D(void const* srcHostBuffer, uint64_t dstDeviceAddr, size_t size) = 0;
    virtual void Flush()                                                                = 0;

    DeviceId const device;
    uint8_t const  streamIdx{};

    static std::unique_ptr<DeviceContext> CreateDeviceContext(DeviceId device, uint8_t streamIdx);

    static std::unique_ptr<DeviceContext> CreateDeviceContextCPU(DeviceId device, uint8_t streamIdx);

#ifdef PHOENIX_HAVE_CUDA
    static std::unique_ptr<DeviceContext> CreateDeviceContextCUDA(DeviceId device, uint8_t streamIdx);

    static void CudaRegisterHostBuffer(void* ptr, size_t size);
    static void CudaUnregisterHostBuffer(void* ptr);

    // NUMA node of the current CUDA device (from its PCI bus), or -1 if unknown.
    static int CudaGetCurrentDeviceNumaNode();
#endif

#ifdef PHOENIX_HAVE_METAL
    static std::unique_ptr<DeviceContext> CreateDeviceContextMTL(DeviceId device, uint8_t streamIdx);
#endif
};

#pragma once
#include "CommonMacros.h"

#include "messages.h"
#include "state.h"
#include "tensor.h"

#include <cstdint>
#include <filesystem>

void SvcIncrRef();
void SvcDecrRef();

[[noreturn]] void LoadCheckpoint(StateTree& root, std::filesystem::path const& ckptPath);

void SaveCheckpoint(StateTree const& root, checkpoint::ipc::SnapshotId const& snapshotId, std::filesystem::path const& ckptPath);
void FinalizeStateRegistry(StateTree& self);
void WaitForCheckpointSave(RankTopology const& topo);
void SvcPing();
void SvcStop();
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

#if defined(PHOENIX_HAVE_CUDA)
    static std::unique_ptr<DeviceContext> CreateDeviceContextCUDA(DeviceId device, uint8_t streamIdx);

    static void CudaRegisterHostBuffer(void* ptr, size_t size);
    static void CudaUnregisterHostBuffer(void* ptr);
#endif

#if defined(PHOENIX_HAVE_METAL)
    static std::unique_ptr<DeviceContext> CreateDeviceContextMTL(DeviceId device, uint8_t streamIdx);
#endif
};

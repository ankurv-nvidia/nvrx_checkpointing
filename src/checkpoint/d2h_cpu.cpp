#include "checkpoint.h"
#include <cstdint>
#include <cstring>
namespace
{
struct CPUContext : public DeviceContext
{
    CPUContext(DeviceId deviceIn, uint8_t streamIdxIn) : DeviceContext(deviceIn, streamIdxIn) {}
    ~CPUContext() override = default;
    CLASS_DELETE_COPY_AND_MOVE(CPUContext);

    void StreamAcquire() override {}
    void Flush() override {}

    void AddD2H(uint64_t srcDeviceAddr, void* dstHostBuffer, size_t size) override
    {
        std::memcpy(dstHostBuffer, reinterpret_cast<void*>(srcDeviceAddr), size);    // NOLINT
    }
    void AddH2D(void const* srcHostBuffer, uint64_t dstDeviceAddr, size_t size) override
    {
        std::memcpy(reinterpret_cast<void*>(dstDeviceAddr), srcHostBuffer, size);    // NOLINT
    }

    uint64_t Allocate(uint64_t size) override
    {
        void* ptr = std::malloc(size);    // NOLINT
        if (ptr == nullptr) { throw std::runtime_error(fmt::format("Failed to allocate device memory of size {}", size)); }
        return reinterpret_cast<uint64_t>(ptr);    // NOLINT
    }
    void Free(uint64_t deviceAddr) override
    {
        std::free(reinterpret_cast<void*>(deviceAddr));    // NOLINT
    }

    void FreeHostPinnedBuffer(void* ptr) override
    {
        std::free(ptr);    // NOLINT
    }

    void* AllocHostPinnedBuffer(size_t size) override
    {
        return std::malloc(size);    // NOLINT
    }
};
}    // namespace

std::unique_ptr<DeviceContext> DeviceContext::CreateDeviceContextCPU(DeviceId device, uint8_t streamIdx)
{
    return std::make_unique<CPUContext>(device, streamIdx);    //
}

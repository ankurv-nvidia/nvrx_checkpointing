#include "CommonMacros.h"
#include "checkpoint.h"
#include "tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cctype>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

// CUDA error checking macros
#define CUDA_CHECK(call, ...)                                                                                                         \
    do /*NOLINT*/                                                                                                                     \
    {                                                                                                                                 \
        cudaError_t error = call;                                                                                                     \
        if (error != cudaSuccess)                                                                                                     \
        {                                                                                                                             \
            int deviceIdx = -1;                                                                                                       \
            cudaGetDevice(&deviceIdx);                                                                                                \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + ": "                 \
                                     + "deviceidx=" + std::to_string(deviceIdx)                                                       \
                                     + " threadid=" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ": " \
                                     + cudaGetErrorString(error) + " " + fmt::format(__VA_ARGS__));                                   \
        }                                                                                                                             \
    } while (0)

struct CudaContext : public DeviceContext
{
    struct CudaStream
    {
        std::mutex   mutex;
        cudaStream_t stream{nullptr};
        int          deviceRankIdx{-1};
        CudaStream(int deviceRankIdxIn) : deviceRankIdx(deviceRankIdxIn)
        { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Failed to create stream"); }

        ~CudaStream()
        {
            if (stream != nullptr) { cudaStreamDestroy(stream); }
        }

        CLASS_DELETE_COPY_AND_MOVE(CudaStream);
        static std::shared_ptr<CudaStream> GetInstance(DeviceId deviceIn, uint8_t streamIdx)
        {
            int deviceCount = 0;
            CUDA_CHECK(cudaGetDeviceCount(&deviceCount), "Failed to get device count");
            if (deviceCount == 0) { throw std::runtime_error("No CUDA devices available"); }
            auto deviceRankIdx = static_cast<int>(deviceIn.index) % deviceCount;
            CUDA_CHECK(cudaSetDevice(deviceRankIdx), "Failed to set device to index {}", deviceRankIdx);

            uint64_t                    hash = deviceIn.index << 8 | streamIdx;
            std::lock_guard<std::mutex> lock(Mutex);
            auto                        it = Instances.find(hash);
            if (it == Instances.end()) { it = Instances.emplace(hash, std::make_shared<CudaStream>(deviceRankIdx)).first; }
            return it->second;
        }

        SUPPRESS_WARNINGS_START
        SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
        SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
        static inline std::mutex                                                Mutex;
        static inline std::unordered_map<uint64_t, std::shared_ptr<CudaStream>> Instances;
        SUPPRESS_WARNINGS_END
    };

    CudaContext(DeviceId deviceIn, uint8_t streamIdxIn) :
        DeviceContext(deviceIn, streamIdxIn), stream(CudaStream::GetInstance(deviceIn, streamIdxIn))
    {
        lock = std::unique_lock<std::mutex>(stream->mutex);
        lock.unlock();
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "Failed to create CUDA event");
    }

    void StreamAcquire() override { lock.lock(); }

    void AddD2H(uint64_t srcDeviceAddr, void* dstHostBuffer, size_t size) override    // NOLINT
    {
        CUDA_CHECK(cudaMemcpyAsync(dstHostBuffer, reinterpret_cast<void*>(srcDeviceAddr), size, cudaMemcpyDeviceToHost, stream->stream),
                   "device={} Failed to copy data from device to host at device address {:x} to host buffer {:p} of size {}",
                   device,
                   srcDeviceAddr,
                   dstHostBuffer,
                   size);
    }

    void Flush() override
    {
        CUDA_CHECK(cudaEventRecord(event, stream->stream), "Failed to record CUDA event");
        lock.unlock();
        CUDA_CHECK(cudaEventSynchronize(event), "Failed to synchronize CUDA event");
    }

    void AddH2D(void const* srcHostBuffer, uint64_t dstDeviceAddr, size_t size) override    // NOLINT
    {
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dstDeviceAddr), srcHostBuffer, size, cudaMemcpyHostToDevice, stream->stream),
                   "dev={} Failed to copy data from host buffer {:p} to device at address {:x} of size {}",
                   device,
                   srcHostBuffer,
                   dstDeviceAddr,
                   size);
    }

    uint64_t Allocate(uint64_t size) override
    {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size), "Failed to allocate device memory of size {}", size);
        return reinterpret_cast<uint64_t>(ptr);    // NOLINT
    }
    void Free(uint64_t deviceAddr) override
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(deviceAddr)), "Failed to free device memory at address {:x}", deviceAddr);    //
    }

    void* AllocHostPinnedBuffer(size_t size) override
    {
        void* ptr = nullptr;
        CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault), "Failed to allocate host pinned buffer of size {}", size);
        return ptr;
    }

    void FreeHostPinnedBuffer(void* ptr) override { CUDA_CHECK(cudaFreeHost(ptr), "Failed to free host pinned buffer at {:p}", ptr); }

    std::unique_lock<std::mutex> lock;
    cudaEvent_t                  event{nullptr};
    std::shared_ptr<CudaStream>  stream;
};

void DeviceContext::CudaRegisterHostBuffer(void* ptr, size_t size)
{
    CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault), "Failed to register host buffer at {:p} of size {}", ptr, size);    //
}

void DeviceContext::CudaUnregisterHostBuffer(void* ptr)
{
    CUDA_CHECK(cudaHostUnregister(ptr), "Failed to unregister host buffer at {:p}", ptr);    //
}

int DeviceContext::CudaGetCurrentDeviceNumaNode()
{
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) { return -1; }
    std::array<char, 32> busId{};
    if (cudaDeviceGetPCIBusId(busId.data(), static_cast<int>(busId.size()), device) != cudaSuccess) { return -1; }
    // sysfs PCI paths are lowercase; cudaDeviceGetPCIBusId yields uppercase hex.
    for (auto& ch : busId) { ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }
    std::ifstream f(std::string("/sys/bus/pci/devices/") + busId.data() + "/numa_node");
    int           node = -1;
    if (f >> node) { return node; }    // a value of -1 here means "no NUMA affinity"
    return -1;
}

std::unique_ptr<DeviceContext> DeviceContext::CreateDeviceContextCUDA(DeviceId device, uint8_t streamIdx)
{
    return std::make_unique<CudaContext>(device, streamIdx);    //
}

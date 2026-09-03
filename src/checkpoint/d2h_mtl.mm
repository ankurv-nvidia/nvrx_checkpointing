#include "CommonMacros.h"
#include "checkpoint.h"

#include <memory>
#include <stdexcept>

// On Apple Silicon, PyTorch's MPS backend allocates tensors with
// MTLStorageModeShared (unified memory). This means tnsr.data_ptr() returns a
// plain CPU-accessible pointer — NOT an id<MTLBuffer>.  The handle stored in
// TensorShard is therefore a CPU pointer, so AddD2H / AddH2D can be
// implemented as plain memcpy calls.
//
// Callers are responsible for ensuring all pending MPS GPU work is complete
// before invoking AddD2H (e.g. via torch.mps.synchronize() on the Python
// side).  On Apple Silicon unified memory there is no separate CPU<->GPU copy;
// any in-flight GPU writes to the buffer are visible to the CPU as soon as the
// corresponding command buffer completes.

struct MetalContext : public DeviceContext { // NOLINT
  MetalContext(DeviceId deviceIn, uint8_t streamIdxIn)
      : DeviceContext(deviceIn, streamIdxIn) {}

  ~MetalContext() override = default;

  CLASS_DELETE_COPY_AND_MOVE(MetalContext);
  void StreamAcquire() override {}
  void Flush() override {}

  void AddD2H(uint64_t srcDeviceAddr, void *dstHostBuffer,
              size_t size) override {
    // srcDeviceAddr == tnsr.data_ptr(): already a CPU-accessible address.
    std::memcpy(dstHostBuffer,
                reinterpret_cast<const void *>(srcDeviceAddr), // NOLINT
                size);
  }

  void AddH2D(void const *srcHostBuffer, uint64_t dstDeviceAddr,
              size_t size) override {
    // dstDeviceAddr == tnsr.data_ptr(): already a CPU-accessible address.
    std::memcpy(reinterpret_cast<void *>(dstDeviceAddr), // NOLINT
                srcHostBuffer, size);
  }

  uint64_t Allocate(uint64_t size) override {
    // Allocate unified memory buffer that's CPU-accessible.
    void *ptr = std::malloc(size); // NOLINT
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return reinterpret_cast<uint64_t>(ptr); // NOLINT
  }

  void Free(uint64_t deviceAddr) override {
    std::free(reinterpret_cast<void *>(deviceAddr)); // NOLINT
  }

  void FreeHostPinnedBuffer(void *ptr) override {
    std::free(ptr); // NOLINT
  }

  void *AllocHostPinnedBuffer(size_t size) override {
    return std::malloc(size); // NOLINT
  }
};

std::unique_ptr<DeviceContext>
DeviceContext::CreateDeviceContextMTL(DeviceId device, uint8_t streamIdx) {
  return std::make_unique<MetalContext>(device, streamIdx);
}

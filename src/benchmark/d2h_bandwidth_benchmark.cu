/*
 * Device-to-Host (D2H) Bandwidth Benchmark
 * 
 * A standalone utility to measure GPU to CPU memory transfer bandwidth.
 * 
 * Compile:
 *   nvcc -O3 -std=c++17 d2h_bandwidth_benchmark.cu -o d2h_benchmark
 * 
 * Usage:
 *   ./d2h_benchmark [device_id]
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(error));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

struct BenchmarkResult {
    size_t bufferSize;
    double bandwidthGBps;
    double timeMs;
    bool usePinnedMemory;
};

// Format byte size in human-readable format
std::string formatSize(size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024) {
        return std::to_string(bytes / (1024.0 * 1024.0 * 1024.0)).substr(0, 4) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024.0 * 1024.0)).substr(0, 5) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024.0).substr(0, 5) + " KB";
    } else {
        return std::to_string(bytes) + " B";
    }
}

// Run benchmark for a single configuration
BenchmarkResult benchmarkSingleConfig(size_t bufferSize, int numIterations, 
                                      int warmupIterations, bool usePinned) {
    // Allocate device memory
    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, bufferSize));
    CUDA_CHECK(cudaMemset(devicePtr, 0xAB, bufferSize));

    // Allocate host memory
    void* hostPtr = nullptr;
    if (usePinned) {
        CUDA_CHECK(cudaHostAlloc(&hostPtr, bufferSize, cudaHostAllocDefault));
    } else {
        hostPtr = malloc(bufferSize);
        if (!hostPtr) {
            CUDA_CHECK(cudaFree(devicePtr));
            throw std::runtime_error("Failed to allocate host memory");
        }
    }

    // Create CUDA stream and events
    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Warmup iterations
    for (int i = 0; i < warmupIterations; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(hostPtr, devicePtr, bufferSize, 
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark iterations
    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    for (int i = 0; i < numIterations; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(hostPtr, devicePtr, bufferSize, 
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    auto elapsedMs = elapsed.count() / 1000.0;
    // Calculate elapsed time
    //float elapsedMs = 0.0f;
    //CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    // Calculate average time and bandwidth
    double avgTimeMs = static_cast<double>(elapsedMs) / numIterations;
    double avgTimeSec = avgTimeMs / 1000.0;
    double bytesPerSec = bufferSize / avgTimeSec;
    double bandwidthGBps = bytesPerSec / (1024.0 * 1024.0 * 1024.0);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(devicePtr));
    
    if (usePinned) {
        CUDA_CHECK(cudaFreeHost(hostPtr));
    } else {
        free(hostPtr);
    }

    BenchmarkResult result;
    result.bufferSize = bufferSize;
    result.bandwidthGBps = bandwidthGBps;
    result.timeMs = avgTimeMs;
    result.usePinnedMemory = usePinned;

    return result;
}

// Run full benchmark suite
std::vector<BenchmarkResult> runBenchmark(int deviceId) {
    std::vector<BenchmarkResult> results;

    // Set device
    CUDA_CHECK(cudaSetDevice(deviceId));

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    printf("\n");
    printf("========================================================\n");
    printf("     Device-to-Host (D2H) Bandwidth Benchmark\n");
    printf("========================================================\n");
    printf("Device %d: %s\n", deviceId, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("PCIe Bus ID: %d:%d.%d\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    printf("========================================================\n\n");

    // Default buffer sizes (1KB to 256MB)
    std::vector<size_t> bufferSizes = {
        1024,          // 1 KB
        4096,          // 4 KB
        16384,         // 16 KB
        65536,         // 64 KB
        262144,        // 256 KB
        1048576,       // 1 MB
        4194304,       // 4 MB
       // 16777216,      // 16 MB
       // 67108864,      // 64 MB
       // 268435456,      // 256 MB
       // 536870912,     // 512 MB
       // 1073741824,    // 1 GB
       // 2147483648,    // 2 GB
       // 4294967296,    // 4 GB
       // 8589934592,    // 8 GB
       // 17179869184,   // 16 GB
       // 34359738368,   // 32 GB
       // 68719476736,   // 64 GB
    };

    int numIterations = 1;
    int warmupIterations = 0;

    // Run benchmarks
    for (size_t bufferSize : bufferSizes) {
        // Test pinned memory
        try {
            auto result = benchmarkSingleConfig(bufferSize, numIterations, 
                                               warmupIterations, true);
            results.push_back(result);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error testing pinned memory for size %zu: %s\n", 
                    bufferSize, e.what());
        }

        // Test pageable memory
        try {
            auto result = benchmarkSingleConfig(bufferSize, numIterations, 
                                               warmupIterations, false);
            results.push_back(result);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error testing pageable memory for size %zu: %s\n", 
                    bufferSize, e.what());
        }
    }

    return results;
}

// Print results in formatted table
void printResults(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) {
        printf("No benchmark results to display.\n");
        return;
    }

    // Print header
    printf("%15s | %12s | %15s | %12s\n", 
           "Buffer Size", "Memory Type", "Bandwidth (GB/s)", "Time (ms)");
    printf("----------------+-------------+-----------------+-------------\n");

    // Print results
    for (const auto& result : results) {
        std::string memType = result.usePinnedMemory ? "Pinned" : "Pageable";
        printf("%15s | %12s | %15.2f | %12.6f\n",
               formatSize(result.bufferSize).c_str(),
               memType.c_str(),
               result.bandwidthGBps,
               result.timeMs);
    }

    // Find and print peak bandwidth
    auto maxBandwidth = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.bandwidthGBps < b.bandwidthGBps;
        });

    printf("\n");
    printf("========================================================\n");
    printf("Peak Bandwidth: %.2f GB/s (%s, %s)\n",
           maxBandwidth->bandwidthGBps,
           formatSize(maxBandwidth->bufferSize).c_str(),
           maxBandwidth->usePinnedMemory ? "Pinned" : "Pageable");
    printf("========================================================\n");
}

int main(int argc, char** argv) {
    int deviceId = 0;

    // Parse command line arguments
    if (argc > 1) {
        deviceId = atoi(argv[1]);
    }

    // Check if device is available
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return EXIT_FAILURE;
    }

    if (deviceId < 0 || deviceId >= deviceCount) {
        fprintf(stderr, "Invalid device ID %d. Available devices: 0-%d\n", 
                deviceId, deviceCount - 1);
        return EXIT_FAILURE;
    }

    // Run benchmark
    try {
        auto results = runBenchmark(deviceId);
        printResults(results);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Benchmark failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
}


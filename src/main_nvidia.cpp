#include <cuda_runtime.h>
#include <iostream>

int getCudaCoresPerSM(int major, int minor) {
    // This mapping is based on NVIDIA's CUDA documentation
    struct SMtoCores {
        int SM;
        int cores;
    };
    SMtoCores gpuArchCores[] = {
        { 2, 32 },  // Fermi
        { 3, 192 }, // Kepler
        { 5, 128 }, // Maxwell
        { 6, 64 },  // Pascal (6.1 has 128 cores per SM)
        { 7, 64 },  // Volta (7.5 has 128 cores per SM)
        { 8, 64 },  // Ampere (8.6 has 128 cores per SM)
        { 9, 128 }, // Hopper
    };

    for (const auto& entry : gpuArchCores) {
        if (entry.SM == major) {
            return entry.cores;
        }
    }
    return 64; // Default fallback
}

void printCudaCoreCount() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int smCount = prop.multiProcessorCount;
    int cudaCoresPerSM = getCudaCoresPerSM(prop.major, prop.minor);
    int totalCudaCores = smCount * cudaCoresPerSM;

    std::cout << "GPU: " << prop.name << "\n"
              << "Compute Capability: " << prop.major << "." << prop.minor << "\n"
              << "Streaming Multiprocessors: " << smCount << "\n"
              << "CUDA Cores per SM: " << cudaCoresPerSM << "\n"
              << "Total CUDA Cores: " << totalCudaCores << "\n";
}

int main() {
    printCudaCoreCount();
    return 0;
}



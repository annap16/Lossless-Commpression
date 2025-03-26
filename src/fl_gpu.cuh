#pragma once

#include <stdint.h>

int FLGPUCompression(uint8_t *data, int dataSize, uint8_t *&compressedData, uint8_t *&bitsPerDataTable, int dataFrameSize);
void FLGPUDecompression(uint8_t *compressedData, uint8_t *bitsPerDataTable, int compressedDataSize, int dataSize, int dataFrameSize, uint8_t *decompressedData);

#ifdef IMPLEMENTATION

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

__global__ void kernelBitsPerFrame(uint8_t *data, int *dataSize, int *dataFrameSize, uint8_t *bitsPerDataTable);
__global__ void kernelCompressDataFL(uint8_t *data, int *dataSize, int *dataFrameSize, uint8_t *bitsPerDataTable, int *writeIndices, uint8_t *compressedData);
__global__ void kernelDecompressDataFL(uint8_t *compressedData, uint8_t *bitsPerDataTable, int *dataSize, int *dataFrameSize, int *readIndices, uint8_t *decompressedData);
__host__ __device__ uint8_t getBitSize(uint8_t data);

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::string errorMessage = "GPU Error: " + std::string(cudaGetErrorString(code)) +
                                   " in file " + std::string(file) +
                                   " at line " + std::to_string(line);
        throw std::runtime_error(errorMessage);
    }
}

__device__ uint8_t atomicOrUint8(uint8_t *address, uint8_t val)
{
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int sel = selectors[(size_t)address & 3];
    unsigned int old, assumed, or_, new_;
    old = *base_address;
    do
    {
        assumed = old;
        or_ = val | (uint8_t)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440);
        new_ = __byte_perm(old, or_, sel);
        if (new_ == old)
            break;
        old = atomicCAS(base_address, assumed, new_);

    } while (assumed != old);
    return old;
}

// Compression with FL algorithm on GPU
int FLGPUCompression(uint8_t *data, int dataSize, uint8_t *&compressedData, uint8_t *&bitsPerDataTable, int dataFrameSize)
{
    // Initialize and allocate memory
    int framesCount = std::ceil(dataSize / static_cast<float>(dataFrameSize));
    uint8_t *d_data = nullptr;
    uint8_t *d_compressedData = nullptr;
    int *d_dataSize = nullptr;
    int *d_dataFrameSize = nullptr;

    bitsPerDataTable = new uint8_t[framesCount];
    if (!bitsPerDataTable)
    {
        printf("bitsPerTable new error in function FLGPUCompression\n");
        exit(1);
    }

    try
    {
        printf("\tMemory allocation...\n");
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Memory allocation on GPU and copying data from CPU
        gpuErrchk(cudaMalloc((void **)&d_data, sizeof(uint8_t) * dataSize));
        gpuErrchk(cudaMalloc((void **)&d_dataSize, sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_dataFrameSize, sizeof(int)));

        gpuErrchk(cudaMemcpy(d_data, data, sizeof(uint8_t) * dataSize, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_dataSize, &dataSize, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_dataFrameSize, &dataFrameSize, sizeof(int), cudaMemcpyHostToDevice));

        thrust::device_vector<uint8_t> d_bitsPerDataTable(framesCount);
        thrust::device_vector<int> d_bitsPerDataTableScan(framesCount + 1);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("\tPerforming compression..\n");
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Kernel invocation: calculating the smallest number of bits needed to represent a given frame
        int threadsPerBlocks = 1024;
        int blocksCount = std::ceil(static_cast<float>(framesCount) / 1024.0);
        kernelBitsPerFrame<<<blocksCount, threadsPerBlocks>>>(d_data, d_dataSize, d_dataFrameSize, thrust::raw_pointer_cast(d_bitsPerDataTable.data()));

        // Checking errors after kernel invocation
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("CUDA kernel execution failed: %s \n", cudaGetErrorString(err));
        }

        // Calculating the total number of bits required for the compressed data.
        thrust::transform(thrust::device, d_bitsPerDataTable.begin(), d_bitsPerDataTable.end() - 1, d_bitsPerDataTableScan.begin() + 1, [d_dataFrameSize] __device__(int val)
                          { return val * (*d_dataFrameSize); });
        d_bitsPerDataTableScan[d_bitsPerDataTableScan.size() - 1] = (dataSize - dataFrameSize * (framesCount - 1)) * d_bitsPerDataTable[d_bitsPerDataTable.size() - 1];
        thrust::inclusive_scan(thrust::device, d_bitsPerDataTableScan.begin() + 1, d_bitsPerDataTableScan.end(), d_bitsPerDataTableScan.begin() + 1);
        d_bitsPerDataTableScan[0] = 0;

        int bitsCompressedData = d_bitsPerDataTableScan[d_bitsPerDataTableScan.size() - 1];
        int bytesCompressedData = std::ceil((bitsCompressedData) / 8.0);

        // Memory allocation for compressed data on CPU and GPU
        compressedData = new uint8_t[bytesCompressedData];
        gpuErrchk(cudaMalloc((void **)&d_compressedData, sizeof(uint8_t) * bytesCompressedData));
        gpuErrchk(cudaMemset(d_compressedData, 0, sizeof(uint8_t) * bytesCompressedData));

        // Kernel invocation: compressing data using the FL algorithm with one thread assigned per frame.
        kernelCompressDataFL<<<blocksCount, threadsPerBlocks>>>(d_data, d_dataSize, d_dataFrameSize, thrust::raw_pointer_cast(d_bitsPerDataTable.data()),
                                                                thrust::raw_pointer_cast(d_bitsPerDataTableScan.data()), d_compressedData);

        // Checking errors after kernel invocation
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("CUDA kernel execution failed: %s \n", cudaGetErrorString(err));
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("\tCopying results to CPU tables..\n");
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Copying results to CPU tables
        thrust::copy(d_bitsPerDataTable.begin(), d_bitsPerDataTable.end(), bitsPerDataTable);
        gpuErrchk(cudaMemcpy(compressedData, d_compressedData, sizeof(uint8_t) * bytesCompressedData, cudaMemcpyDeviceToHost));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_data);
        cudaFree(d_dataSize);
        cudaFree(d_dataFrameSize);
        cudaFree(d_compressedData);

        return bytesCompressedData;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        if (bitsPerDataTable)
        {
            delete[] bitsPerDataTable;
            bitsPerDataTable = nullptr;
        }
        if (compressedData)
        {
            delete[] compressedData;
            compressedData = nullptr;
        }
        if (d_data)
        {
            cudaFree(d_data);
            d_data = nullptr;
        }
        if (d_dataSize)
        {
            cudaFree(d_dataSize);
            d_dataSize = nullptr;
        }
        if (d_dataFrameSize)
        {
            cudaFree(d_dataFrameSize);
            d_dataFrameSize = nullptr;
        }
        if (d_compressedData)
        {
            cudaFree(d_compressedData);
            d_compressedData = nullptr;
        }
        exit(1);
    }
}

// Decompression with FL algorithm on GPU
void FLGPUDecompression(uint8_t *compressedData, uint8_t *bitsPerDataTable, int compressedDataSize, int dataSize, int dataFrameSize, uint8_t *decompressedData)
{
    // Initialize and allocate memory
    int framesCount = std::ceil(dataSize / static_cast<float>(dataFrameSize));
    int *d_dataFrameSize = nullptr;
    uint8_t *d_compressedData = nullptr;
    uint8_t *d_bitsPerDataTable = nullptr;
    int *d_dataSize = nullptr;
    uint8_t *d_decompressedData = nullptr;

    try
    {
        printf("\tMemory allocation...\n");
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Memory allocation on GPU and copying data from CPU
        gpuErrchk(cudaMalloc((void **)&d_compressedData, sizeof(uint8_t) * compressedDataSize));
        gpuErrchk(cudaMalloc((void **)&d_bitsPerDataTable, sizeof(uint8_t) * framesCount));
        gpuErrchk(cudaMalloc((void **)&d_dataSize, sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_dataFrameSize, sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_decompressedData, sizeof(uint8_t) * dataSize));

        gpuErrchk(cudaMemcpy(d_compressedData, compressedData, sizeof(uint8_t) * compressedDataSize, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_bitsPerDataTable, bitsPerDataTable, sizeof(uint8_t) * framesCount, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_dataSize, &dataSize, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_dataFrameSize, &dataFrameSize, sizeof(int), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemset(d_decompressedData, 0, sizeof(uint8_t) * dataSize));

        thrust::device_vector<int> d_bitsPerDataTableScan(framesCount);
        thrust::copy(bitsPerDataTable, bitsPerDataTable + framesCount, d_bitsPerDataTableScan.begin());

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("\tPerforming decompression..\n");
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Assigning indices for each thread to access its designated portion of compressed data
        thrust::transform(thrust::device, d_bitsPerDataTableScan.begin(), d_bitsPerDataTableScan.end(), d_bitsPerDataTableScan.begin(), [d_dataFrameSize] __device__(int val)
                          { return val * (*d_dataFrameSize); });
        thrust::exclusive_scan(thrust::device, d_bitsPerDataTableScan.begin(), d_bitsPerDataTableScan.end(), d_bitsPerDataTableScan.begin(), 0);

        // Kernel invocation: decompressing data algorithm with one thread assigned per frame.
        int threadsPerBlocks = 1024;
        int blocksCount = std::ceil(static_cast<float>(framesCount) / 1024.0);
        kernelDecompressDataFL<<<blocksCount, threadsPerBlocks>>>(d_compressedData, d_bitsPerDataTable, d_dataSize, d_dataFrameSize, thrust::raw_pointer_cast(d_bitsPerDataTableScan.data()), d_decompressedData);

        // Checking errors after kernel invocation
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("CUDA kernel execution failed: %s \n", cudaGetErrorString(err));
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("\tCopying results to CPU table..\n");
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Copying results to CPU table
        gpuErrchk(cudaMemcpy(decompressedData, d_decompressedData, sizeof(uint8_t) * dataSize, cudaMemcpyDeviceToHost));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_compressedData);
        cudaFree(d_bitsPerDataTable);
        cudaFree(d_dataSize);
        cudaFree(d_dataFrameSize);
        cudaFree(d_decompressedData);
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;

        if (decompressedData)
        {
            delete[] decompressedData;
            decompressedData = nullptr;
        }
        if (compressedData)
        {
            delete[] compressedData;
            compressedData = nullptr;
        }
        if (bitsPerDataTable)
        {
            delete[] bitsPerDataTable;
            bitsPerDataTable = nullptr;
        }
        if (d_compressedData)
        {
            cudaFree(d_compressedData);
            d_compressedData = nullptr;
        }
        if (d_bitsPerDataTable)
        {
            cudaFree(d_bitsPerDataTable);
            d_bitsPerDataTable = nullptr;
        }
        if (d_dataSize)
        {
            cudaFree(d_dataSize);
            d_dataSize = nullptr;
        }
        if (d_dataFrameSize)
        {
            cudaFree(d_dataFrameSize);
            d_dataFrameSize = nullptr;
        }
        if (d_decompressedData)
        {
            cudaFree(d_decompressedData);
            d_decompressedData = nullptr;
        }
        exit(1);
    }
}

//  Calculating the minimum number of bits required to represent a frame
__global__ void kernelBitsPerFrame(uint8_t *data, int *dataSize, int *dataFrameSize, uint8_t *bitsPerDataTable)
{
    // Calculating indices for the given thread and terminating any additional threads
    int _dataSize = *dataSize;
    int _dataFrameSize = *dataFrameSize;
    int writeIndx = blockIdx.x * 1024 + threadIdx.x;
    int readIndx = writeIndx * _dataFrameSize;
    if (readIndx > _dataSize)
    {
        return;
    }

    // Calculating the minimum number of bits required to represent the frame assigned to the thread
    uint8_t maxBitLength = getBitSize(data[readIndx]);
    uint8_t pomLength;
    for (int i = readIndx + 1; i < readIndx + _dataFrameSize && i < _dataSize; i++)
    {
        pomLength = getBitSize(data[i]);
        if (pomLength > maxBitLength)
        {
            maxBitLength = pomLength;
        }
    }

    bitsPerDataTable[writeIndx] = maxBitLength;
}

// Compressing data by frames using the FL algorithm
__global__ void kernelCompressDataFL(uint8_t *data, int *dataSize, int *dataFrameSize, uint8_t *bitsPerDataTable,
                                     int *writeIndices, uint8_t *compressedData)
{
    // Calculating indices for the given thread and terminating any additional threads
    int frameIndx = blockIdx.x * 1024 + threadIdx.x;
    int _dataFrameSize = *dataFrameSize;
    int _dataSize = *dataSize;
    if (frameIndx * _dataFrameSize > _dataSize - 1)
    {
        return;
    }

    int writeIndx = writeIndices[frameIndx];
    int startIndx, offsetInCell;
    uint8_t bitsPerData = bitsPerDataTable[frameIndx];
    uint8_t tempData;

    // Compressing a frame assigned to a running thread
    for (int i = frameIndx * (*dataFrameSize); i < (frameIndx + 1) * (*dataFrameSize) && i < (*dataSize); i++)
    {
        startIndx = writeIndx / 8;
        offsetInCell = writeIndx % 8;

        // Data compression when it fits in one cell
        if (bitsPerData <= 8 - offsetInCell)
        {
            tempData = data[i] << (8 - bitsPerData);
            tempData = tempData >> offsetInCell;
            atomicOrUint8(&compressedData[startIndx], tempData);
        }
        // Data compression when data has to be divided into two cells
        else
        {
            tempData = data[i] >> (bitsPerData - (8 - offsetInCell));
            atomicOrUint8(&compressedData[startIndx], tempData);
            tempData = data[i] << (16 - bitsPerData - offsetInCell);
            atomicOrUint8(&compressedData[startIndx + 1], tempData);
        }

        writeIndx += bitsPerData;
    }
}

// Decompressing frame data previously compressed using the FL algorithm
__global__ void kernelDecompressDataFL(uint8_t *compressedData, uint8_t *bitsPerDataTable, int *dataSize, int *dataFrameSize, int *readIndices, uint8_t *decompressedData)
{
    // Calculating indices for the given thread and terminating any additional threads
    int frameIndx = blockIdx.x * 1024 + threadIdx.x;
    int _dataFrameSize = *dataFrameSize;
    int _dataSize = *dataSize;
    if (frameIndx * _dataFrameSize > _dataSize - 1)
    {
        return;
    }

    int readIndex = readIndices[frameIndx];
    int startIndx, offsetInCell;
    uint8_t tempData;

    // Decompressing a frame assigned to a running thread
    for (int i = frameIndx * _dataFrameSize; i < (frameIndx + 1) * _dataFrameSize && i < _dataSize; i++)
    {
        startIndx = readIndex / 8;
        offsetInCell = readIndex % 8;

        // Data decompression when it fits in one cell
        if (bitsPerDataTable[frameIndx] <= 8 - offsetInCell)
        {
            tempData = compressedData[startIndx] << offsetInCell;
            tempData = tempData >> (8 - bitsPerDataTable[frameIndx]);
            decompressedData[i] = tempData;
        }
        // Data decompression when data has to be divided into two cells
        else
        {
            tempData = compressedData[startIndx] << offsetInCell;
            tempData = tempData >> (8 - bitsPerDataTable[frameIndx]);
            uint8_t tempData2 = compressedData[startIndx + 1] >> (16 - bitsPerDataTable[frameIndx] - offsetInCell);
            decompressedData[i] = tempData | tempData2;
        }
        readIndex += bitsPerDataTable[frameIndx];
    }
}

// Calculate the smallest number of bits needed to represent a given integer in binary form
__host__ __device__ uint8_t getBitSize(uint8_t data)
{
    if (data == 0)
        return 1;
    if (data == 255)
        return 8;
    return static_cast<uint8_t>(ceilf(log2f(data + 1)));
}
#endif

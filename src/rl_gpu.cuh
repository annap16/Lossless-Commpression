#pragma once
#include <stdint.h>

int RLGPUCompression(uint8_t *data, int dataSize, uint8_t *&compressedData, uint8_t *&sequencesLength);
void RLGPUDecompression(uint8_t *compressedData, uint8_t *sequencesLength, int compressedDataSize, int decompressedDataSize, uint8_t *decompressedData);

#ifdef IMPLEMENTATION
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

__global__ void kernelFixOverflow(uint8_t *data, int *sequencesLength, int *writeIndx, int *size, uint8_t *compressedData, uint8_t *fixedSequencesLength);
__global__ void kernelDecompressionRL(uint8_t *compressedData, uint8_t *sequencesLength, int *compressedDataSize, int *writeIndx, uint8_t *decompressedData);

// Compression with RL algorithm on GPU
int RLGPUCompression(uint8_t *data, int dataSize, uint8_t *&compressedData, uint8_t *&sequencesLength)
{
    uint8_t *d_fixedSequencesLength = nullptr;
    uint8_t *d_compressedData = nullptr;
    try
    {
        printf("\tMemory allocation...\n");
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Vector used to calculate the number of threads needed for processing
        thrust::device_vector<int> d_mask(dataSize);

        // Length of a sequence in run-length encoding can exceed 255
        thrust::device_vector<int> d_count(dataSize);

        // Data after reduction by key, used by threads to identify the portions of data to process
        thrust::device_vector<uint8_t> d_reducedKeys(dataSize);

        // Size of the output array (temporary byproduct)
        thrust::device_vector<int> d_compressedSize(dataSize);

        // Number of threads needed for processing
        thrust::device_vector<int> d_threadCount(dataSize);

        // Temporary vector for intermediate calculations
        thrust::device_vector<int> d_ones(dataSize);

        // Size of the output array for the compressed data after scanning
        thrust::device_vector<int> d_compressedSizeScan(dataSize);

        // Data for compression copied to the GPU
        thrust::device_vector<uint8_t> d_data(data, data + dataSize);

        thrust::fill(thrust::device, d_mask.begin(), d_mask.end(), 0);
        thrust::fill(thrust::device, d_ones.begin(), d_ones.end(), 1);

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

        // Calculating threads needed for processing
        thrust::transform(thrust::device, d_data.begin() + 1, d_data.end(), d_data.begin(), d_mask.begin() + 1, thrust::not_equal_to<uint8_t>());
        d_mask[0] = 1;
        thrust::inclusive_scan(thrust::device, d_mask.begin(), d_mask.end(), d_threadCount.begin(), thrust::plus<int>());

        // Calculating the length of a sequence in run-length encoding, can exceed 255
        thrust::reduce_by_key(thrust::device, d_data.begin(), d_data.end(), d_ones.begin(), d_reducedKeys.begin(), d_count.begin(), thrust::equal_to<int>());

        // Calculating the length of output array
        thrust::transform(thrust::device, d_count.begin(), d_count.end(), d_compressedSize.begin(), [] __device__(int x)
                          { return static_cast<int>(ceilf(x / 255.0f)); });
        thrust::inclusive_scan(thrust::device, d_compressedSize.begin(), d_compressedSize.end(), d_compressedSizeScan.begin(), thrust::plus<int>());

        // Determining write positions within the kernel for each thread
        thrust::device_vector<int> d_writeIndx(d_threadCount[d_threadCount.size() - 1]);
        thrust::exclusive_scan(thrust::device, d_compressedSize.begin(), d_compressedSize.begin() + d_writeIndx.size(), d_writeIndx.begin(), 0, thrust::plus<int>());

        int threadsPerBlocks = 1024;
        int blocksCount = std::ceil(static_cast<float>(d_threadCount[d_threadCount.size() - 1]) / 1024.0);

        // Allocate memory for compressed data and sequence lengths on GPU
        gpuErrchk(cudaMalloc((void **)&d_compressedData, sizeof(uint8_t) * d_compressedSizeScan[d_compressedSizeScan.size() - 1]));
        gpuErrchk(cudaMalloc((void **)&d_fixedSequencesLength, sizeof(uint8_t) * d_compressedSizeScan[d_compressedSizeScan.size() - 1]));

        // Kernel invocation: The kernel processes sequences by splitting those with lengths greater than 255 into the appropriate number of smaller units
        kernelFixOverflow<<<blocksCount, threadsPerBlocks>>>(thrust::raw_pointer_cast(d_reducedKeys.data()), thrust::raw_pointer_cast(d_count.data()),
                                                             thrust::raw_pointer_cast(d_writeIndx.data()), thrust::raw_pointer_cast(&d_threadCount[d_threadCount.size() - 1]),
                                                             d_compressedData, d_fixedSequencesLength);
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

        // Allocating memory for compressed data on CPU
        compressedData = new uint8_t[d_compressedSizeScan[d_compressedSizeScan.size() - 1]];
        if (!compressedData)
        {
            printf("new error compressedData in function RLGPUCompression\n");
            exit(1);
        }
        sequencesLength = new uint8_t[d_compressedSizeScan[d_compressedSizeScan.size() - 1]];
        if (!sequencesLength)
        {
            printf("new error sequencesLength in function RLGPUCompression\n");
            exit(1);
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
        gpuErrchk(cudaMemcpy(compressedData, d_compressedData, sizeof(uint8_t) * d_compressedSizeScan[d_compressedSizeScan.size() - 1], cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(sequencesLength, d_fixedSequencesLength, sizeof(uint8_t) * d_compressedSizeScan[d_compressedSizeScan.size() - 1], cudaMemcpyDeviceToHost));

        int compressedDataSize;
        gpuErrchk(cudaMemcpy(&compressedDataSize, thrust::raw_pointer_cast(&d_compressedSizeScan[d_compressedSizeScan.size() - 1]), sizeof(int), cudaMemcpyDeviceToHost));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_compressedData);
        cudaFree(d_fixedSequencesLength);
        return compressedDataSize;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;

        if (compressedData)
        {
            delete[] compressedData;
            compressedData = nullptr;
        }
        if (sequencesLength)
        {
            delete[] sequencesLength;
            sequencesLength = nullptr;
        }
        if (d_compressedData)
        {
            cudaFree(d_compressedData);
            d_compressedData = nullptr;
        }
        if (d_fixedSequencesLength)
        {
            cudaFree(d_fixedSequencesLength);
            d_fixedSequencesLength = nullptr;
        }
        exit(1);
    }
}

// Decompression with RL algorithm on GPU
void RLGPUDecompression(uint8_t *compressedData, uint8_t *sequencesLength, int compressedDataSize, int decompressedDataSize, uint8_t *decompressedData)
{
    uint8_t *d_compressedData = nullptr;
    int *d_compressedDataSize = nullptr;
    uint8_t *d_decompressedData = nullptr;

    try
    {
        printf("\tMemory allocation...\n");
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Allocating memory on GPU
        gpuErrchk(cudaMalloc((void **)&d_compressedData, sizeof(uint8_t) * compressedDataSize));
        gpuErrchk(cudaMalloc((void **)&d_compressedDataSize, sizeof(int)));
        gpuErrchk(cudaMalloc((void **)&d_decompressedData, sizeof(uint8_t) * decompressedDataSize));

        gpuErrchk(cudaMemcpy(d_compressedData, compressedData, sizeof(uint8_t) * compressedDataSize, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_compressedDataSize, &compressedDataSize, sizeof(int), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemset(d_decompressedData, 0, sizeof(uint8_t) * decompressedDataSize));

        thrust::device_vector<uint8_t> d_sequencesLength(sequencesLength, sequencesLength + compressedDataSize);
        thrust::device_vector<int> d_writeIndx(compressedDataSize);
        thrust::device_vector<int> d_calcWriteIndx(compressedDataSize);
        thrust::transform(d_sequencesLength.begin(), d_sequencesLength.end(), d_calcWriteIndx.begin(), thrust::identity<int>());

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

        // Determining write positions within the kernel for each thread
        thrust::exclusive_scan(thrust::device, d_calcWriteIndx.begin(), d_calcWriteIndx.end(), d_writeIndx.begin(), 0, thrust::plus<int>());

        // Kernel invocation: Decompressing each data sequence, with one thread processing each entry (one cell in the table per thread)
        int threadsPerBlocks = 1024;
        int blocksCount = std::ceil(static_cast<float>(compressedDataSize) / 1024.0);
        kernelDecompressionRL<<<blocksCount, threadsPerBlocks>>>(d_compressedData, thrust::raw_pointer_cast(d_sequencesLength.data()), d_compressedDataSize, thrust::raw_pointer_cast(d_writeIndx.data()), d_decompressedData);

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
        gpuErrchk(cudaMemcpy(decompressedData, d_decompressedData, sizeof(uint8_t) * decompressedDataSize, cudaMemcpyDeviceToHost));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("\tExecution time: %f milliseconds\n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_compressedData);
        cudaFree(d_compressedDataSize);
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
        if (sequencesLength)
        {
            delete[] sequencesLength;
            sequencesLength = nullptr;
        }
        if (d_compressedData)
        {
            cudaFree(d_compressedData);
            d_compressedData = nullptr;
        }
        if (d_compressedDataSize)
        {
            cudaFree(d_compressedDataSize);
            d_compressedDataSize = nullptr;
        }
        if (d_decompressedData)
        {
            cudaFree(d_decompressedData);
            d_decompressedData = nullptr;
        }
        exit(1);
    }
}

// Processing sequences by splitting those with lengths greater than 255 into the appropriate number of smaller units
__global__ void kernelFixOverflow(uint8_t *data, int *sequencesLength, int *writeIndx, int *size, uint8_t *compressedData, uint8_t *fixedSequencesLength)
{
    // Calculating indices for the given thread and terminating any additional threads
    int indx = blockIdx.x * 1024 + threadIdx.x;
    if (indx > *(size)-1)
    {
        return;
    }

    int sequenceLengthLeft = sequencesLength[indx];
    int myWriteIndx = writeIndx[indx];
    uint8_t myData = data[indx];

    // Writing compressed data, splitting sequences longer than 255 into smaller chunks and updating lengths
    while (sequenceLengthLeft > 0)
    {
        compressedData[myWriteIndx] = myData;

        if (sequenceLengthLeft > 255)
        {
            fixedSequencesLength[myWriteIndx] = 255;
            sequenceLengthLeft -= 255;
        }
        else
        {
            fixedSequencesLength[myWriteIndx] = sequenceLengthLeft;
            sequenceLengthLeft = 0;
        }

        myWriteIndx++;
    }
}

// Decompressing each data sequence, with one thread processing each entry (one cell in the table per thread)
__global__ void kernelDecompressionRL(uint8_t *compressedData, uint8_t *sequencesLength, int *compressedDataSize, int *writeIndx, uint8_t *decompressedData)
{
    // Calculating indices for the given thread and terminating any additional threads
    int indx = blockIdx.x * 1024 + threadIdx.x;
    if (indx > (*compressedDataSize) - 1)
    {
        return;
    }

    int myWriteIndx = writeIndx[indx];
    uint8_t myData = compressedData[indx];

    // Writing decompressed data to the output based on the sequence length.
    for (int i = 0; i < sequencesLength[indx]; i++)
    {
        decompressedData[myWriteIndx] = myData;
        myWriteIndx++;
    }
}
#endif
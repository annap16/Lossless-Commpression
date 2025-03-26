#pragma once

void readAndCompressFL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU);
void readAndCompressRL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU);
void decompressAndWriteFL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU);
void decompressAndWriteRL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU);

#ifdef IMPLEMENTATION

#include "fl_cpu.h"
#include "rl_cpu.h"
// Reading the input file and compressing it using the FL algorithm
void readAndCompressFL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU)
{
    printf("Reading from input file...\n");

    // Recording the initial time for data loading
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *fileRead = fopen(filenameInput, "r");
    if (fileRead == NULL)
    {
        printf("The provided file for reading is not valid.\n");
        exit(1);
    }

    // Calculating data size
    fseek(fileRead, 0, SEEK_END);
    int dataSize = ftell(fileRead);
    if (dataSize == 0)
    {
        printf("The file is empty. No data to compress. Empty file created\n");
        return;
    }

    // Reading data
    fseek(fileRead, 0, SEEK_SET);
    uint8_t *fileData = new uint8_t[dataSize];
    fread(fileData, sizeof(uint8_t), dataSize, fileRead);

    fclose(fileRead);

    // Recording the final time for data loading and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    uint8_t *compressedData = nullptr;
    uint8_t *bitsPerDataTable = nullptr;
    int dataFrameSize = 32;

    printf("Starting compression...\n");

    // Recording the initial time for compression
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Compressing data using the FL compression
    int compressedDataSize;
    if (calcOnGPU)
    {
        compressedDataSize = FLGPUCompression(fileData, dataSize, compressedData, bitsPerDataTable, dataFrameSize);
    }
    else
    {
        compressedDataSize = FLCompression(fileData, dataSize, compressedData, bitsPerDataTable, dataFrameSize);
    }

    // Recording the final time for compression and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Compression process completed. Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Writing to the output file...\n");

    // Recording the initial time writing to the output file
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int framesCount = std::ceil(dataSize / static_cast<float>(dataFrameSize));
    FILE *fileWrite = fopen(filenameOutput, "wb");
    if (fileWrite == NULL)
    {
        printf("The provided file for writing is not valid.\n");
        delete[] fileData;
        delete[] compressedData;
        delete[] bitsPerDataTable;
        exit(1);
    }

    // Writing the result to the output file:
    // Header: dataSizeBeforeCompression, dataSizeAfterCompression, frameSizeUsedDuringCompression
    // Body: compressedData (size: dataSizeAfterCompression), bitsRequiredPerFrame (size: ceil(dataSize / dataFrameSize))
    fwrite(&dataSize, sizeof(int), 1, fileWrite);
    fwrite(&compressedDataSize, sizeof(int), 1, fileWrite);
    fwrite(&dataFrameSize, sizeof(int), 1, fileWrite);

    fwrite(compressedData, sizeof(uint8_t), compressedDataSize, fileWrite);
    fwrite(bitsPerDataTable, sizeof(uint8_t), framesCount, fileWrite);
    fclose(fileWrite);

    // Recording the final time for writing to the output file and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] compressedData;
    delete[] bitsPerDataTable;
    delete[] fileData;
}

// Reading the input file and compressing it using the RL algorithm
void readAndCompressRL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU)
{
    // Reading from input file
    printf("Reading from input file...\n");

    // Recording the initial time for data loading
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *fileRead = fopen(filenameInput, "r");
    if (fileRead == NULL)
    {
        printf("The provided file for reading is not valid.\n");
        exit(1);
    }

    // Calculating data size
    fseek(fileRead, 0, SEEK_END);
    int dataSize = ftell(fileRead);
    if (dataSize == 0)
    {
        printf("The file is empty. No data to compress. Empty file created\n");
        return;
    }

    // Reading data
    fseek(fileRead, 0, SEEK_SET);
    uint8_t *fileData = new uint8_t[dataSize];
    fread(fileData, sizeof(uint8_t), dataSize, fileRead);
    fclose(fileRead);

    // Recording the final time for data loading and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    uint8_t *compressedData = nullptr;
    uint8_t *sequencesLength = nullptr;

    // Starting compression
    printf("Starting compression...\n");

    // Recording the initial time for compression
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Compressing data using the RL compression
    int compressedDataSize;
    if (calcOnGPU)
    {
        compressedDataSize = RLGPUCompression(fileData, dataSize, compressedData, sequencesLength);
    }
    else
    {
        std::vector<uint8_t> compressedDataVec;
        std::vector<uint8_t> sequencesLengthVec;

        compressedDataSize = RLCompression(fileData, dataSize, compressedDataVec, sequencesLengthVec);

        compressedData = new uint8_t[compressedDataVec.size()];
        sequencesLength = new uint8_t[sequencesLengthVec.size()];

        std::copy(compressedDataVec.begin(), compressedDataVec.end(), compressedData);
        std::copy(sequencesLengthVec.begin(), sequencesLengthVec.end(), sequencesLength);
    }

    // Recording the final time for compression and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Compression process completed. Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Writing to the output file
    printf("Writing to the output file...\n");

    // Recording the initial time for writing to the output file
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *fileWrite = fopen(filenameOutput, "wb");
    if (fileWrite == NULL)
    {
        printf("The provided file for writing is not valid.\n");
        delete[] fileData;
        delete[] compressedData;
        delete[] sequencesLength;
        exit(1);
    }

    // Writing the result to the output file:
    // Header: dataSizeBeforeCompression, dataSizeAfterCompression
    // Body: compressedData (size: dataSizeAfterCompression), sequencesLength (size: dataSizeAfterCompression)
    fwrite(&dataSize, sizeof(int), 1, fileWrite);
    fwrite(&compressedDataSize, sizeof(int), 1, fileWrite);
    fwrite(compressedData, sizeof(uint8_t), compressedDataSize, fileWrite);
    fwrite(sequencesLength, sizeof(uint8_t), compressedDataSize, fileWrite);
    fclose(fileWrite);

    // Recording the final time for writing to the output file and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleaning up memory
    delete[] compressedData;
    delete[] sequencesLength;
    delete[] fileData;
}

// Decompressing the file using the FL algorithm and writing the result to the output file
void decompressAndWriteFL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU)
{
    // Reading from input file
    printf("Reading from input file...\n");

    // Recording the initial time for data loading
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *fileRead = fopen(filenameInput, "rb");
    if (fileRead == NULL)
    {
        printf("The provided file for reading is not valid.\n");
        exit(1);
    }

    fseek(fileRead, 0, SEEK_END);
    int fileSize = ftell(fileRead);
    if (fileSize == 0)
    {
        printf("The file is empty. No data to compress. Empty file created\n");
        return;
    }
    fseek(fileRead, 0, SEEK_SET);

    // Reading the header of the input file
    int dataSize, compressedDataSize, dataFrameSize;
    fread(&dataSize, sizeof(int), 1, fileRead);
    fread(&compressedDataSize, sizeof(int), 1, fileRead);
    fread(&dataFrameSize, sizeof(int), 1, fileRead);

    int framesCount = std::ceil(dataSize / static_cast<float>(dataFrameSize));

    // Reading the body of the input file
    uint8_t *compressedData = new uint8_t[compressedDataSize];
    uint8_t *bitsPerDataTable = new uint8_t[framesCount];
    fread(compressedData, sizeof(uint8_t), compressedDataSize, fileRead);
    fread(bitsPerDataTable, sizeof(uint8_t), framesCount, fileRead);
    fclose(fileRead);

    // Recording the final time for data loading and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    uint8_t *decompressedData = new uint8_t[dataSize];

    // Starting decompression
    printf("Starting decompression...\n");

    // Recording the initial time for decompression
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Decompressing the data using FL algorithm
    if (calcOnGPU)
    {
        FLGPUDecompression(compressedData, bitsPerDataTable, compressedDataSize, dataSize, dataFrameSize, decompressedData);
    }
    else
    {
        FLDecompression(compressedData, bitsPerDataTable, dataSize, dataFrameSize, decompressedData);
    }

    // Recording the final time for decompression and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Decompression process completed. Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Writing to the output file
    printf("Writing to the output file...\n");

    // Recording the initial time for writing to the output file
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *fileWrite = fopen(filenameOutput, "wb");
    if (fileWrite == NULL)
    {
        printf("The provided file for writing is not valid.\n");
        delete[] compressedData;
        delete[] bitsPerDataTable;
        delete[] decompressedData;
        exit(1);
    }

    // Writing decompressed data to the output file
    fwrite(decompressedData, sizeof(uint8_t), dataSize, fileWrite);
    fclose(fileWrite);

    // Recording the final time for writing to the output file and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleaning up memory
    delete[] compressedData;
    delete[] bitsPerDataTable;
    delete[] decompressedData;
}

// Decompressing the file using the RL algorithm and writing the result to the output file
void decompressAndWriteRL(const char *filenameInput, const char *filenameOutput, bool calcOnGPU)
{
    // Reading from input file
    printf("Reading from input file...\n");

    // Recording the initial time for data loading
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *fileRead = fopen(filenameInput, "rb");
    if (fileRead == NULL)
    {
        printf("The provided file for reading is not valid.\n");
        exit(1);
    }

    fseek(fileRead, 0, SEEK_END);
    int fileSize = ftell(fileRead);
    if (fileSize == 0)
    {
        printf("The file is empty. No data to compress. Empty file created\n");
        return;
    }
    fseek(fileRead, 0, SEEK_SET);

    // Reading the header of the input file
    int dataSize, compressedDataSize;
    fread(&dataSize, sizeof(int), 1, fileRead);
    fread(&compressedDataSize, sizeof(int), 1, fileRead);

    uint8_t *compressedData = new uint8_t[compressedDataSize];
    uint8_t *sequencesLength = new uint8_t[compressedDataSize];

    // Reading the body of the input file
    fread(compressedData, sizeof(uint8_t), compressedDataSize, fileRead);
    fread(sequencesLength, sizeof(uint8_t), compressedDataSize, fileRead);
    fclose(fileRead);

    // Recording the final time for data loading and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    uint8_t *decompressedData = new uint8_t[dataSize];

    // Starting decompression
    printf("Starting decompression...\n");

    // Recording the initial time for decompression
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Decompressing the data using RL algorithm
    if (calcOnGPU)
    {
        RLGPUDecompression(compressedData, sequencesLength, compressedDataSize, dataSize, decompressedData);
    }
    else
    {
        RLDecompression(compressedData, sequencesLength, compressedDataSize, decompressedData);
    }

    // Recording the final time for decompression and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Decompression process completed. Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Writing to the output file
    printf("Writing to the output file...\n");

    // Recording the initial time for writing to the output file
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *fileWrite = fopen(filenameOutput, "wb");
    if (fileWrite == NULL)
    {
        printf("The provided file for writing is not valid.\n");
        delete[] compressedData;
        delete[] sequencesLength;
        delete[] decompressedData;
        exit(1);
    }

    // Writing decompressed data to the output file
    fwrite(decompressedData, sizeof(uint8_t), dataSize, fileWrite);
    fclose(fileWrite);

    // Recording the final time for writing to the output file and the difference between start and stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time: %f milliseconds\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleaning up memory
    delete[] compressedData;
    delete[] sequencesLength;
    delete[] decompressedData;
}

#endif
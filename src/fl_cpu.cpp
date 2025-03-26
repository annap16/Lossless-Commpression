
#include <fl_cpu.h>
#include <cmath>
#include <stdio.h>

// Decompression with FL algorithm on CPU
void FLDecompression(uint8_t *compressedData, uint8_t *bitsPerDataTable, int dataSize, int dataFrameSize, uint8_t *decompressedData)
{
    int framesCount = std::ceil(dataSize / static_cast<float>(dataFrameSize));

    int occupiedSpace = 0;
    for (int i = 0; i < framesCount; i++)
    {
        decompressFrame(compressedData, bitsPerDataTable, i, dataFrameSize, decompressedData, dataSize, occupiedSpace);
        occupiedSpace += bitsPerDataTable[i] * dataFrameSize;
    }
}

// Helper function for decompressing single frame (FL) on CPU
void decompressFrame(uint8_t *compressedData, uint8_t *bitsPerDataTable, int currFrame, int dataFrameSize, uint8_t *decompressedData, int dataSize, int occupiedSpace)
{
    int startIndx, offsetInCell;
    uint8_t tempData;

    for (int i = currFrame * dataFrameSize; i < (currFrame + 1) * dataFrameSize && i < dataSize; i++)
    {
        startIndx = occupiedSpace / 8;
        offsetInCell = occupiedSpace % 8;

        // Data decompression when it fits in one cell
        if (bitsPerDataTable[currFrame] <= 8 - offsetInCell)
        {
            tempData = compressedData[startIndx] << offsetInCell;
            tempData = tempData >> (8 - bitsPerDataTable[currFrame]);
            decompressedData[i] = tempData;
        }
        // Data decompression when data has to be divided into two cells
        else
        {
            tempData = compressedData[startIndx] << offsetInCell;
            tempData = tempData >> (8 - bitsPerDataTable[currFrame]);
            uint8_t tempData2 = compressedData[startIndx + 1] >> (16 - bitsPerDataTable[currFrame] - offsetInCell);
            decompressedData[i] = tempData | tempData2;
        }

        occupiedSpace += bitsPerDataTable[currFrame];
    }
}

// Compression with FL algorithm on CPU
int FLCompression(uint8_t *data, int dataSize, uint8_t *&compressedData, uint8_t *&bitsPerDataTable, int dataFrameSize)
{
    int framesCount = std::ceil(dataSize / static_cast<float>(dataFrameSize));

    bitsPerDataTable = new uint8_t[framesCount];
    if (!bitsPerDataTable)
    {
        printf("bitsPerDataTable new error in function FLCompression\n");
        exit(1);
    }

    int bitsCountAfterCompression = 0;

    // Calculate the total size of compressed data in bits and determine the number of bits required for each frame
    for (int i = 0; i < framesCount - 1; i++)
    {
        uint8_t bitsPerData = getMaxLength(data, dataSize, i, dataFrameSize);
        bitsCountAfterCompression += (bitsPerData * dataFrameSize);
        bitsPerDataTable[i] = bitsPerData;
    }
    int bitsPerData = getMaxLength(data, dataSize, framesCount - 1, dataFrameSize);
    bitsCountAfterCompression += (bitsPerData * (dataSize - ((framesCount - 1) * dataFrameSize)));
    bitsPerDataTable[framesCount - 1] = bitsPerData;

    // Calculate compressed data size in bytes and allocate memory
    int afterCompressionBytesSize = std::ceil(bitsCountAfterCompression / 8.0);
    compressedData = new uint8_t[afterCompressionBytesSize];
    if (!compressedData)
    {
        printf("compressedData new errorin function FLCompression\n");
        exit(1);
    }

    // Initialize data
    for (int i = 0; i < afterCompressionBytesSize; i++)
    {
        compressedData[i] = 0;
    }

    // Compress each frame
    int occupiedSpace = 0;
    for (int i = 0; i < framesCount; i++)
    {
        compressFrame(data, dataSize, i, dataFrameSize, compressedData, bitsPerDataTable, occupiedSpace);
        occupiedSpace += bitsPerDataTable[i] * dataFrameSize;
    }

    return afterCompressionBytesSize;
}

// Helper function for compressing single frame (FL) on CPU
void compressFrame(uint8_t *data, int dataSize, int currFrame, int dataFrameSize, uint8_t *compressedData, uint8_t *bitsPerDataTable, int occupiedSpace)
{
    for (int i = currFrame * dataFrameSize; i < dataSize && i < (currFrame + 1) * dataFrameSize; i++)
    {
        int startIndx = occupiedSpace / 8;
        int offsetInCell = occupiedSpace % 8;
        uint8_t tempData;
        // Data compression when it fits in one cell
        if (bitsPerDataTable[currFrame] <= 8 - offsetInCell)
        {
            tempData = data[i] << (8 - bitsPerDataTable[currFrame]);
            tempData = tempData >> offsetInCell;
            compressedData[startIndx] = compressedData[startIndx] | tempData;
        }
        // Data compression when data has to be divided into two cells
        else
        {
            tempData = data[i] >> (bitsPerDataTable[currFrame] - (8 - offsetInCell));
            compressedData[startIndx] = compressedData[startIndx] | tempData;
            tempData = data[i] << (16 - bitsPerDataTable[currFrame] - offsetInCell);
            compressedData[startIndx + 1] = compressedData[startIndx + 1] | tempData;
        }

        occupiedSpace += bitsPerDataTable[currFrame];
    }
}

// Calculate the smallest number of bits needed to represent a given frame in FL algorithm
int getMaxLength(uint8_t *data, int dataSize, int currFrame, int dataFrameSize)
{
    int maxLength = getBitSizeCPU(data[currFrame * dataFrameSize]);
    for (int i = currFrame * dataFrameSize + 1; i < dataSize && i < (currFrame + 1) * (dataFrameSize); i++)
    {
        int tempLength = getBitSizeCPU(data[i]);
        if (tempLength > maxLength)
        {
            maxLength = tempLength;
        }
    }
    return maxLength;
}

// Calculate the smallest number of bits needed to represent a given integer in binary form
uint8_t getBitSizeCPU(uint8_t data)
{
    if (data == 0)
        return 1;
    if (data == 255)
        return 8;
    return static_cast<uint8_t>(ceilf(log2f(data + 1)));
}
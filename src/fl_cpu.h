#pragma once

#include <stdint.h>

int FLCompression(uint8_t *data, int dataSize, uint8_t *&compressedData, uint8_t *&bitsPerDataTable, int dataFrameSize);
void FLDecompression(uint8_t *compressedData, uint8_t *bitsPerDataTable, int dataSize, int dataFrameSize, uint8_t *decompressedData);
void compressFrame(uint8_t *data, int dataSize, int currFrame, int dataFrameSize, uint8_t *compressedData, uint8_t *bitsPerDataTable, int occupiedSpace);
void decompressFrame(uint8_t *compressedData, uint8_t *bitsPerDataTable, int currFrame, int dataFrameSize, uint8_t *decompressedData, int dataSize, int occupiedSpace);
int getMaxLength(uint8_t *data, int dataSize, int currFrame, int dataFrameSize);
uint8_t getBitSizeCPU(uint8_t data);

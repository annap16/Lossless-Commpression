#pragma once

#include <stdint.h>
#include <vector>

int RLCompression(uint8_t *data, int dataSize, std::vector<uint8_t> &compressedData, std::vector<uint8_t> &sequencesLength);
void RLDecompression(uint8_t *compressedData, uint8_t *sequencesLength, int compressedDataLength, uint8_t *decompressedData);
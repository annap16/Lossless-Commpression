#include "rl_cpu.h"

// Compression with RL algorithm on CPU
int RLCompression(uint8_t *data, int dataSize, std::vector<uint8_t> &compressedData, std::vector<uint8_t> &sequencesLength)
{
    uint8_t counter = 1;

    for (int i = 0; i + 1 < dataSize; i++)
    {
        if (data[i] == data[i + 1] && counter < 255)
        {
            counter++;
        }
        else
        {
            compressedData.push_back(data[i]);
            sequencesLength.push_back(counter);
            counter = 1;
        }
    }

    compressedData.push_back(data[dataSize - 1]);
    sequencesLength.push_back(counter);
    return compressedData.size();
}

// Decompression with RL algorithm on CPU
void RLDecompression(uint8_t *compressedData, uint8_t *sequencesLength, int compressedDataLength, uint8_t *decompressedData)
{
    int decompressedDataLength = 0;

    for (int i = 0; i < compressedDataLength; i++)
    {
        decompressedDataLength += sequencesLength[i];
    }

    int writeIndx = 0;

    for (int i = 0; i < compressedDataLength; i++)
    {
        for (int j = 0; j < sequencesLength[i]; j++)
        {
            decompressedData[writeIndx] = compressedData[i];
            writeIndx++;
        }
    }
}
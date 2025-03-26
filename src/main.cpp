#include <stdio.h>
#include <stdint.h>

#include <vector>
#include <cstring>
#include "fl_cpu.h"
#include "rl_cpu.h"
#include "fl_gpu.cuh"
#include "rl_gpu.cuh"
#include "file_manager.cuh"

int main(int argc, char **argv)
{
    // Verifying the correctness of the input provided from the console.
    if (argc != 6)
    {
        printf("Wrong number of input parameters. The invocation must be as follows: cpu/gpu operation method input_file output_file\n");
        return 1;
    }
    const char *device = argv[1];
    if (std::strcmp(device, "cpu") != 0 && std::strcmp(device, "gpu") != 0)
    {
        printf("Choose device: CPU or GPU\n");
        return 1;
    }
    const char *operation = argv[2];
    if (std::strcmp(operation, "c") != 0 && std::strcmp(operation, "d") != 0)
    {
        printf("Wrong operation type. Operation type must be: 'c' for compressing or 'd' for decompressing\n");
        return 1;
    }

    const char *method = argv[3];
    if (std::strcmp(method, "fl") != 0 && std::strcmp(method, "rl") != 0)
    {
        printf("Wrong method. Mathod must be: 'fl' or 'rl'\n");
        return 1;
    }

    const char *filenameInput = argv[4];
    FILE *fileRead = fopen(filenameInput, "r");
    if (fileRead == NULL)
    {
        printf("The provided file for reading is not valid.\n");
        return 1;
    }
    fclose(fileRead);

    const char *filenameOutput = argv[5];
    FILE *fileWrite = fopen(filenameOutput, "wb");
    if (fileWrite == NULL)
    {
        printf("The provided file for writing is not valid.\n");
        return 1;
    }
    fclose(fileWrite);

    bool calcOnGPU = true;
    if (std::strcmp(device, "cpu") == 0)
    {
        calcOnGPU = false;
    }

    // Invoking the appropriate function based on the input provided from the console
    if (std::strcmp(operation, "c") == 0)
    {
        if (std::strcmp(method, "fl") == 0)
        {
            readAndCompressFL(filenameInput, filenameOutput, calcOnGPU);
        }
        else
        {
            readAndCompressRL(filenameInput, filenameOutput, calcOnGPU);
        }
    }
    else
    {
        if (std::strcmp(method, "fl") == 0)
        {
            decompressAndWriteFL(filenameInput, filenameOutput, calcOnGPU);
        }
        else
        {
            decompressAndWriteRL(filenameInput, filenameOutput, calcOnGPU);
        }
    }

    return 0;
}

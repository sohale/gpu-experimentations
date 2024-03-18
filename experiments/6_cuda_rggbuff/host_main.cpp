#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

// Error checking for CUDA
#define cudaCheckError() { cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) { printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE); } }

// Dummy function to load an RGB image (you need to implement this)
void loadImage(const char* filename, unsigned char** image, int* width, int* height);

// Dummy function to save an RGB image (you need to implement this)
void saveImage(const char* filename, unsigned char* image, int width, int height);

// Declare the function defined in .cu (kernel)
__global__ void apply_contrast(unsigned char* image, int width, int height, float contrast);


// Main function
int main() {
    const char* inputFile = "m1.bmp";
    const char* outputFile = "out.bmp";

    int width, height;
    unsigned char* hostImage = nullptr;
    loadImage(inputFile, &hostImage, &width, &height);

    // devImage is a pointer to a location on GPU RAM
    unsigned char* devImage;
    size_t pixelSize = 3 * sizeof(unsigned char); // 3 channels (RGB)
    size_t imageSize = width * height * pixelSize;
    cudaMalloc(&devImage, imageSize);
    cudaMemcpy(devImage, hostImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    float contrast = 1.2f; // Adjust contrast level
    apply_contrast<<<gridSize, blockSize>>>(devImage, width, height, contrast);
    cudaCheckError();

    cudaMemcpy(hostImage, devImage, imageSize, cudaMemcpyDeviceToHost);
    saveImage(outputFile, hostImage, width, height);

    cudaFree(devImage);
    delete[] hostImage;

    return 0;
}

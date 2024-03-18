__global__ void apply_contrast(unsigned char* image, int width, int height, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 3 is for RGB channels
        int idx = (y * width + x) * 3;


        for (int color = 0; color < 3; color++) {
            int pixel = image[idx + color];
            pixel = 128 + static_cast<int>(contrast * (pixel - 128));
            // Clamp the value in [0,255]
            image[idx + color] = max(0, min(255, pixel));
        }
    }
}

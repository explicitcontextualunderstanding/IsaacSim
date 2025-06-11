
template <typename T>
__global__ void textureCopyToRawBufferKernel(cudaTextureObject_t srcTexObj,
                                             T* dstBuffer,
                                             unsigned int dstWidth,
                                             unsigned int dstHeight)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    const float tu = (x + 0.5f) / dstWidth;
    const float tv = (y + 0.5f) / dstHeight;

    dstBuffer[y * dstWidth + x] = tex2D<T>(srcTexObj, tu, tv);
}

template <typename T>
void textureCopyToRawBuffer(cudaTextureObject_t srcTexObj,
                            unsigned char* dstBuffer,
                            unsigned int dstWidth,
                            unsigned int dstHeight,
                            cudaStream_t stream)
{
    const dim3 dimBlock(32, 32);
    const dim3 dimGrid((dstWidth + dimBlock.x - 1) / dimBlock.x, (dstHeight + dimBlock.y - 1) / dimBlock.y);
    textureCopyToRawBufferKernel<T>
        <<<dimGrid, dimBlock, 0, stream>>>(srcTexObj, reinterpret_cast<T*>(dstBuffer), dstWidth, dstHeight);
}

extern "C" void textureFloatCopyToRawBuffer(cudaTextureObject_t srcTexObj,
                                            unsigned char* dstBuffer,
                                            unsigned int dstWidth,
                                            unsigned int dstHeight,
                                            cudaStream_t stream)
{
    if (dstWidth * dstHeight > 0 && dstBuffer != nullptr)
    {
        textureCopyToRawBuffer<float>(srcTexObj, dstBuffer, dstWidth, dstHeight, stream);
    }
}
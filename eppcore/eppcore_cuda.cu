#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <math_constants.h>

namespace {

}

__global__ void epp_inflation_cuda_kernel(
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> instance,
    torch::PackedTensorAccessor<float,5,torch::RestrictPtrTraits,size_t> infdst,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> infsrc,
    const int height,
    const int width,
    const int bs,
    const int infheight,
    const int infwidth
    ) {
    int m;
    int n;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;
        if (instance[blockIdx.x][0][m][n] > -1){
            for(int p1 = 0; p1 < infheight; p1++){
                for(int p2 = 0; p2 < infwidth; p2++){
                    infdst[blockIdx.x][m][n][p1][p2] = infsrc[blockIdx.x][instance[blockIdx.x][0][m][n]][p1][p2];
                }
            }
        }
    }
    return;

    }

__global__ void epp_compressio_cuda_kernel(
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> instance,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> compdst,
    torch::PackedTensorAccessor<float,5,torch::RestrictPtrTraits,size_t> compsrc,
    const int height,
    const int width,
    const int bs,
    const int compheight,
    const int compwidth
    ) {
    int m;
    int n;

    float inc = 0.1f;
    float tmpsum = 0;


    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;
        // if (instance[blockIdx.x][0][m][n] > -1){
            for(int p1 = 0; p1 < compheight; p1++){
                for(int p2 = 0; p2 < compwidth; p2++){
                    // atomicAdd((float*)&compdst[blockIdx.x][instance[blockIdx.x][0][m][n]][p1][p2], compsrc[blockIdx.x][m][n][p1][p2]);
                    // compsrc[blockIdx.x][m][n][p1][p2] = -1;
                    // atomicAdd((float*)&compdst[0][0][0][0], (float)compsrc[blockIdx.x][m][n][p1][p2]);
                    // atomicAdd((float*)&compdst[0][0][0][0], 1.0f);
                    // atomicAdd((float*)&compdst[blockIdx.x][instance[blockIdx.x][0][m][n]][p1][p2], 0.000002f);
                    // atomicAdd((float*)&compdst[0][0][0][0], inc);
                    atomicAdd((float*)&compdst[0][0][0][0], 0.1f);
                    atomicAdd((float*)&compdst[1][0][0][0], 1.0f);
                }
            }
        // }
    }
    return;

    }

void epp_inflation_cuda(
    torch::Tensor instance,
    torch::Tensor infdst,
    torch::Tensor infsrc,
    int height,
    int width,
    int bs,
    int infheight,
    int infwidth
    ){
      const int threads = 512;

      epp_inflation_cuda_kernel<<<bs, threads>>>(
            instance.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            infdst.packed_accessor<float,5,torch::RestrictPtrTraits,size_t>(),
            infsrc.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            infheight,
            infwidth
            );
    return;
    }

void epp_compression_cuda(
    torch::Tensor instance,
    torch::Tensor compdst,
    torch::Tensor compsrc,
    int height,
    int width,
    int bs,
    int compheight,
    int compwidth
    ){
      const int threads = 512;

      epp_compressio_cuda_kernel<<<bs, threads>>>(
            instance.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            compdst.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            compsrc.packed_accessor<float,5,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            compheight,
            compwidth
            );
    return;
    }
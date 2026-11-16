#ifdef GOMC_CUDA

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>

#include "BoxDimensions.h"
#include "CUDAMemoryManager.cuh"
#include "CalculatePMECUDAKernel.cuh"
#include "CalculateMinImageCUDAKernel.cuh"
#include "ConstantDefinitionsCUDAKernel.cuh"
#include "MoveSettings.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void CallUpdateVectorsAndRecipTermsGPU(VariablesCUDA *vars)
{

#ifndef NDEBUG
  checkLastErrorCUDA(__FILE__, __LINE__);
#endif

    int threadsPerBlock = 32;
    int blocksPerGrid = 4;
    UpdateVectorsAndRecipTermsKernel<<<blocksPerGrid, threadsPerBlock>>>();

#ifndef NDEBUG
  checkLastErrorCUDA(__FILE__, __LINE__);
#endif
}

void CallUpdateGreenFunctionGPU(VariablesCUDA *vars, int Kx, int Ky, int Kz, int halfKz, 
                                double Lx, double Ly, double Lz, double* gf_out, double vol, 
                                double pre, std::vector<double> bx, std::vector<double> by,
                                std::vector<double>bz)
{
//cudaMemcpy required for bx, by, bz, and gf_out
//Check for correct {&, *} in passing pointer var gf_out
//'SIZE' is a placeholder

    cudaMemcpy(vars->gpu_bx, bx.data(), bx.size() * sizeof(double), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(vars->gpu_by, by.data(), by.size() * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(vars->gpu_bz, bz.data(), bz.size() * sizeof(double),
               cudaMemcpyHostToDevice);
//I think gf_out only contains one item
    cudaMemcpy(vars->gpu_finalVal, gf_out, (sizeof(gf_out) / sizeof(gf_out[0])) * sizeof(double), 
               cudaMemcpyHostToDevice);

#ifndef NDEBUG
  checkLastErrorCUDA(__FILE__, __LINE__);
#endif

    dim3 threadsPerBlock(8,8,8);
    dim3 blocksPerGrid(
      (Kx  + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (Ky + threadsPerBlock.y - 1) / threadsPerBlock.y,
      (Kz  + threadsPerBlock.z - 1) / threadsPerBlock.z);
    UpdateGreenFunctionKernel<<<blocksPerGrid, threadsPerBlock>>>(
	vars->gpu_bx, vars->gpu_by, vars->gpu_bz, vars->gpu_finalVal,
	Kx, Ky, Kz, halfKz, Lx, Ly, Lz, vol, pre);

#ifndef NDEBUG
  checkLastErrorCUDA(__FILE__, __LINE__);
#endif

    cudaMemcpy(gf_out, vars->gpu_finalVal, (sizeof(gf_out) / sizeof(gf_out[0])) * sizeof(double),
               cudaMemcpyDeviceToHost);
}

void CallSumMeshEnergyGPU(VariablesCUDA *vars)
{
#ifndef NDEBUG
  checkLastErrorCUDA(__FILE__, __LINE__);
#endif

    int threadsPerBlock = 32;
    int blocksPerGrid = 4;
    SumMeshEnergyKernel<<<blocksPerGrid, threadsPerBlock>>>();

#ifndef NDEBUG
  checkLastErrorCUDA(__FILE__, __LINE__);
#endif
}

__global__ void UpdateVectorsAndRecipTermsKernel()
{

}

__global__ void UpdateGreenFunctionKernel(const double *__restrict__ gpu_bx, const double *__restrict__ gpu_by,
    const double *__restrict__ gpu_bz, const double *__restrict__ gpu_finalVal,
    int Kx, int Ky, int Kz, int halfKz, double Lx, double Ly, double Lz,
    double vol, double pre)
{
  //This might be correct..?
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= Kx || y >= Ky || z >= halfKz) {return;}

  int kx_int = (x <= Kx / 2) ? x : x - Kx;
  double kx = 2.0 * M_PI * kx_int / Lx;

  int ky_int = (y <= Ky / 2) ? y : y - Ky;
  double ky = 2.0 * M_PI * ky_int / Ly;

  double kz = 2.0 * M_PI * z / Lz;
  double kSq = kx * kx + ky * ky + kz * kz;
  int idx = (x * Ky + y) * halfKz + z;

  if (kSq == 0) {gpu_finalVal[idx] = 0.0;}
  else{
    gpu_finalVal[idx] = num::qqFact * (4.0 * M_PI / (vol * kSq)) *
                  exp(-pre * kSq) * gpu_bx[x] * gpu_by[y] * gpu_bz[z];
  }
}

__global__ void SumMeshEnergyKernel()
{

}


#endif

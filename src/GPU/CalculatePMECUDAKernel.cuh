#ifndef CALCULATE_PME_CUDA_KERNEL_H
#define CALCULATE_PME_CUDA_KERNEL_H

#ifdef GOMC_CUDA
#include "VariablesCUDA.cuh"
#include "XYZArray.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

void CallUpdateVectorsAndRecipTermsGPU(VariablesCUDA *vars);

void CallUpdateGreenFunctionGPU(VariablesCUDA *vars, int Kx, int Ky, int Kz, int halfKz, 
                                double Lx, double Ly, double Lz, double* gf_out, double vol, 
                                double pre, std::vector<double> bx, std::vector<double> by,
                                std::vector<double>bz);

void CallSumMeshEnergyGPU(VariablesCUDA *vars);


__global__ void UpdateVectorsAndRecipTermsKernel();

__global__ void UpdateGreenFunctionKernel(
    const double *__restrict__ gpu_bx, const double *__restrict__ gpu_by,
    const double *__restrict__ gpu_bz, const double *__restrict__ gpu_finalVal,
    int Kx, int Ky, int Kz, int halfKz, double Lx, double Ly, double Lz,
    double vol, double pre);


__global__ void SumMeshEnergyKernel();


#endif /*GOMC_CUDA*/
#endif /*CALCULATE_PME_CUDA_KERNEL_H*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

cudaError_t cudaDotProduct(int *c, const int *a, const int *b, unsigned int size);
cudaError_t cuda2GPUDotProduct(int *c, const int *a, const int *b, unsigned int N);
int* allocAndAssignMat(int size);

__global__ void dot(int *c, const int *a, const int *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] += a[i] * b[i];
}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	const int N = 10000;  // this is the sqrt of the total elements or the len of one side of the square matrix
	const int* a = allocAndAssignMat(N * N);
	const int* b = allocAndAssignMat(N * N);
	int* c = (int*)malloc((N * N) * sizeof(int));

	for (int i = 0; i < N * N; i++) {
		c[i] = 0;
	}
    
    int mySum = 0;

	cudaEventRecord(start);

    // Add vectors in parallel.
    cudaError_t cudaStatus = cuda2GPUDotProduct(c, a, b, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDotProduct failed!");
        return 1;
    }
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	for (int i = 0; i < N*N; i++) {
		//printf("%d ", c[i]);
		mySum += c[i];
	}

    //Results
	printf("Size of N*N: %d \nResult: %d \nTime in kernel %f", N * N, mySum, milliseconds);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t cuda2GPUDotProduct(int *c, const int *a, const int *b, unsigned int N)
{
    int size = N*N
    int subSize = N*(N/2)

    int* dev_c_0;
    int* dev_a_0;
    int* dev_c_1;
    int* dev_a_1;
    int* dev_b;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0)

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c_0, subSize * sizeof(int));  // allocating the space on the gpu
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a_0, subSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpyAsync(dev_a_0, &a[0], subSize * sizeof(int), cudaMemcpyHostToDevice);  // moving the data to the gpu counterpart not c as that is results
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dot<<<(subSize+255)/256, 256>>>(dev_c_0, dev_a_0, dev_b);  // execution configuration - 

    cudaStatus = cudaSetDevice(1);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c_1, subSize * sizeof(int));  // allocating the space on the gpu
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a_1, subSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpyAsync(dev_a_1, &a[subSize], subSize * sizeof(int), cudaMemcpyHostToDevice);  // moving the data to the gpu counterpart not c as that is results
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dot<<<(subSize+255)/256, 256>>>(dev_c_1, dev_a_1, dev_b);  // execution configuration - 

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpyAsync(&c[0], dev_c_0, subSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpyAsync(&c[subSize], dev_c, subSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after getting result!\n", cudaStatus);
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t cudaDotProduct(int *c, const int *a, const int *b, unsigned int N)
{
    int size = N * N;
    int *dev_a; 
    int *dev_b; 
    int *dev_c;

    cudaError_t cudaStatus;
    

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));  // allocating the space on the gpu
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);  // moving the data to the gpu counterpart not c as that is results
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dot<<<(size+255)/256, 256>>>(dev_c, dev_a, dev_b);  // execution configuration - 

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpyAsync(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

int* allocAndAssignMat(int size) {
	/*
		This function takes in the size of the matrix (N*N) and returns a pointer with appropriate memory allocated as well as filled with values

		@params: int size
		@returns: int* ptr
	*/
	int* ptr = (int*)malloc(size * sizeof(int));
	for (int i = 0; i < size; i++) {
		ptr[i] = 2;
	}
	return ptr;
}

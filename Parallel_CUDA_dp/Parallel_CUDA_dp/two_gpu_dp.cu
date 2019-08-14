#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void dot( int *a, int *b, int *c, int Ns);
int* allocAndAssignMat(int size);

//===========================================
__global__ void dot( int *a, int *b, int *c, int Ns){

    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    //c[i] += a[i] * b[i];

    int tid = threadIdx.x + blockIdx.x * blockDim.x; 

    while(tid < Ns){
        c[tid] += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

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

// Note: It is assumed that machine has 2 GPUs
int main( void ) {
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int mySum = 0;

    const int N = 10000;

    int *a, *b, *c;
    //double *dev_a, *dev_b, *dev_c;

    // allocate the memory on the CPU
    a=allocAndAssignMat(N * N);
    b=allocAndAssignMat(N * N);;
    c=(int*)malloc((N * N) * sizeof(int));
    for (int i = 0; i < N * N; i++) {
		c[i] = 0;
	}

    // There's 2 GPUs on this machine
    int *dev_a[2], *dev_b[2], *dev_c[2];
    const int Ns[2] = {N/2, N-(N/2)};

    // Allocate the memory on the GPUs
    for(int dev=0; dev<2; dev++) {
        cudaSetDevice(dev);
        cudaMalloc( (void**)&dev_a[dev], Ns[dev] * sizeof(int) );
        cudaMalloc( (void**)&dev_b[dev], Ns[dev] * sizeof(int) );
        cudaMalloc( (void**)&dev_c[dev], Ns[dev] * sizeof(int) );
    }

    // Copy a and b to GPUs
    for(int dev=0,pos=0; dev<2; pos+=Ns[dev], dev++) {
        cudaSetDevice(dev);
        cudaMemcpy( dev_a[dev], a+pos, Ns[dev] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( dev_b[dev], b+pos, Ns[dev] * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Time
    cudaEventRecord(start);
    for(int i=0;i<10000;++i) {
        for(int dev=0; dev<2; dev++) {
            cudaSetDevice(dev);
            dot<<<((N*N)+255)/256, 256>>>( dev_a[dev], dev_b[dev], dev_c[dev], Ns[dev] );
        }
    }
    
    // Copy c back from the GPU to the CPU
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	for (int i = 0; i < N*N; i++) {
		mySum += c[i];
	}

    //Results
	printf("Size of N*N: %d \nResult: %d \nTime in kernel %f \n", N * N, mySum, milliseconds);

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    // free the memory allocated on the CPU
    free(a);
    free(b);
    free(c);
 
    return 0;
}
 







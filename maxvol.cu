#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CUSOLVER_CALL(x) do { if((x) != CUSOLVER_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

__device__ double calculate_determinant(double *matrix, int n) {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int *devInfo;
    CUDA_CALL(cudaMalloc((void**)&devInfo, sizeof(int)));
    double *d_work;
    int lwork;
    cusolverDnDgetrf_bufferSize(cusolverH, n, n, matrix, n, &lwork);
    CUDA_CALL(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    double *d_tau;
    CUDA_CALL(cudaMalloc((void**)&d_tau, n * sizeof(double)));

    CUSOLVER_CALL(cusolverDnDgetrf(cusolverH, n, n, matrix, n, d_work, d_tau, devInfo));

    int h_info;
    CUDA_CALL(cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        printf("Matrix is singular\n");
        return 0.0;
    }

    double determinant = 1.0;
    for (int i = 0; i < n; i++) {
        determinant *= matrix[i * n + i];
    }

    cusolverDnDestroy(cusolverH);
    CUDA_CALL(cudaFree(devInfo));
    CUDA_CALL(cudaFree(d_work));
    CUDA_CALL(cudaFree(d_tau));

    return determinant;
}

__global__ void find_maxvol(double *A, int m, int n, int r, double *max_values, int *indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int i = idx / n;
        int j = idx % n;

        double submatrix[r * r];
        for (int k = 0; k < r; k++) {
            for (int l = 0; l < r; l++) {
                submatrix[k * r + l] = A[(i + k) * n + (j + l)];
            }
        }

        double det = calculate_determinant(submatrix, r);

        max_values[idx] = det;
        indices[idx] = idx;
    }
}

void initialize_random_indices(int *indices, int m) {
    for (int i = 0; i < m; i++) {
        indices[i] = i;
    }
    for (int i = 0; i < m; i++) {
        int j = rand() % m;
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

int main() {
    const int m = 10;
    const int r = 5;
    const double epsilon = 0.01;

    double *h_A = (double*)malloc(m * r * sizeof(double));
    int *h_indices = (int*)malloc(m * sizeof(int));

    for (int i = 0; i < m * r; i++) {
        h_A[i] = (double)rand() / RAND_MAX;
    }

    initialize_random_indices(h_indices, m);

    double *d_A;
    int *d_indices;
    CUDA_CALL(cudaMalloc((void**)&d_A, m * r * sizeof(double)));
    CUDA_CALL(cudaMalloc((void**)&d_indices, m * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_A, h_A, m * r * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_indices, h_indices, m * sizeof(int), cudaMemcpyHostToDevice));

    double *d_max_values;
    CUDA_CALL(cudaMalloc((void**)&d_max_values, m * r * sizeof(double)));

    find_maxvol<<<(m * r + 255) / 256, 256>>>(d_A, m, r, r, d_max_values, d_indices);

    thrust::device_ptr<double> d_max_values_ptr(d_max_values);
    thrust::device_ptr<int> d_indices_ptr(d_indices);

    int max_idx = thrust::max_element(d_max_values_ptr, d_max_values_ptr + m * r) - d_max_values_ptr;
    int i = max_idx / r;
    int j = max_idx % r;

    CUDA_CALL(cudaMemcpy(h_indices, d_indices, m * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Indices of the rows that form the maximum volume submatrix:\n");
    for (int i = 0; i < r; i++) {
        printf("%d ", h_indices[i]);
    }
    printf("\n");

    free(h_A);
    free(h_indices);
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_indices));
    CUDA_CALL(cudaFree(d_max_values));

    return EXIT_SUCCESS;
}

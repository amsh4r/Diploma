#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void contraction_kernel(double *G, double *w, int *dims, int d, int rank, double *v, int current_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rank) {
        double sum = 0.0;
        int stride = 1;
        for (int k = d - 1; k >= current_dim + 1; --k) {
            stride *= dims[k];
        }
        int base_idx = idx * stride;
        for (int i = 0; i < dims[current_dim]; ++i) {
            sum += G[base_idx + i * rank] * w[i * d + current_dim];
        }
        v[idx] = sum;
    }
}

void tensor_contraction(double **G, double *w, int *dims, int d, int rank, double *result) {
    double **d_G = (double**)malloc(d * sizeof(double*));
    double *d_w;
    for (int i = 0; i < d; ++i) {
        CUDA_CALL(cudaMalloc(&d_G[i], rank * dims[i] * sizeof(double)));
        CUDA_CALL(cudaMemcpy(d_G[i], G[i], rank * dims[i] * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CALL(cudaMalloc(&d_w, d * sizeof(double)));
    CUDA_CALL(cudaMemcpy(d_w, w, d * sizeof(double), cudaMemcpyHostToDevice));

    double *d_v;
    CUDA_CALL(cudaMalloc(&d_v, rank * sizeof(double)));

    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    CUBLAS_CALL(cublasDgemv(handle, CUBLAS_OP_T, dims[0], rank, 1.0, d_G[0], dims[0], d_w, 1, 0.0, d_v, 1));

    for (int k = 1; k < d; ++k) {
        double *d_W;
        CUDA_CALL(cudaMalloc(&d_W, rank * dims[k] * sizeof(double)));

        contraction_kernel<<<(rank + 255) / 256, 256>>>(d_G[k], d_w, dims, d, rank, d_v, k);

        CUDA_CALL(cudaFree(d_W));
    }

    CUDA_CALL(cudaMemcpy(result, d_v, rank * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < d; ++i) {
        CUDA_CALL(cudaFree(d_G[i]));
    }
    CUDA_CALL(cudaFree(d_w));
    CUDA_CALL(cudaFree(d_v));

    CUBLAS_CALL(cublasDestroy(handle));

    free(d_G);
}

double compute_integral(double **G, double *w, int *dims, int d, int rank) {
    double result;
    tensor_contraction(G, w, dims, d, rank, &result);
    return result;
}

void read_tensor_data(const char *filename, double ***G, int **dims, int *d, int *rank) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    if (fread(d, sizeof(int), 1, file) != 1) {
        printf("Error: Unable to read number of dimensions from file %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(rank, sizeof(int), 1, file) != 1) {
        printf("Error: Unable to read rank from file %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    *dims = (int*)malloc(*d * sizeof(int));
    if (fread(*dims, sizeof(int), *d, file) != *d) {
        printf("Error: Unable to read dimensions from file %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    *G = (double**)malloc(*d * sizeof(double*));
    for (int i = 0; i < *d; ++i) {
        (*G)[i] = (double*)malloc((*rank) * (*dims)[i] * sizeof(double));
        if (fread((*G)[i], sizeof(double), (*rank) * (*dims)[i], file) != (*rank) * (*dims)[i]) {
            printf("Error: Unable to read tensor data for dimension %d from file %s\n", i, filename);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

void read_w_data(const char *filename, double **w, int *dims, int d) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    *w = (double*)malloc(d * sizeof(double));
    if (fread(*w, sizeof(double), d, file) != d) {
        printf("Error: Unable to read weight data from file %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fclose(file);
}

int main() {
    int *dims, d, rank;
    double **G;
    double *w;

    const char *tensor_filename = "tensor_data.bin";
    const char *weight_filename = "w_data.bin";

    read_tensor_data(tensor_filename, &G, &dims, &d, &rank);
    read_w_data(weight_filename, &w, dims, d);

    double integral = compute_integral(G, w, dims, d, rank);
    printf("Computed integral: %f\n", integral);

    for (int i = 0; i < d; ++i) {
        free(G[i]);
    }
    free(G);
    free(dims);
    free(w);

    return 0;
}


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


void cross_approximation(double *tensor, int *dims, int n_dims, int rank, double epsilon, double **U, double **V) {
    int rows = dims[0];
    int cols = 1;
    for (int i = 1; i < n_dims; ++i) {
        cols *= dims[i];
    }

    double *d_tensor, *d_matrix;
    CUDA_CALL(cudaMalloc(&d_tensor, rows * cols * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_matrix, rows * cols * sizeof(double)));
    CUDA_CALL(cudaMemcpy(d_tensor, tensor, rows * cols * sizeof(double), cudaMemcpyHostToDevice));

    reshape_tensor_to_matrix<<<(rows * cols + 255) / 256, 256>>>(d_tensor, d_matrix, dims, n_dims, rows, cols);


    int *indices = (int*)malloc(rows * sizeof(int));
    initialize_random_indices(indices, rows);

    double *d_A0;
    CUDA_CALL(cudaMalloc(&d_A0, rows * cols * sizeof(double)));
    CUDA_CALL(cudaMemset(d_A0, 0, rows * cols * sizeof(double)));

    double *d_R;
    CUDA_CALL(cudaMalloc(&d_R, rank * cols * sizeof(double)));
    for (int i = 0; i < rank; ++i) {
        CUDA_CALL(cudaMemcpy(d_R + i * cols, d_matrix + indices[i] * cols, cols * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    int *d_I;
    CUDA_CALL(cudaMalloc(&d_I, rank * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_I, indices, rank * sizeof(int), cudaMemcpyHostToDevice));

    while (norm > epsilon) {
        double *d_QR, *d_Q;
        CUDA_CALL(cudaMalloc(&d_QR, rank * rank * sizeof(double)));
        CUDA_CALL(cudaMemcpy(d_QR, d_R, rank * rank * sizeof(double), cudaMemcpyDeviceToDevice));

        cusolverDnHandle_t cusolverH;
        cusolverDnCreate(&cusolverH);

        int *d_info;
        CUDA_CALL(cudaMalloc((void**)&d_info, sizeof(int)));
        double *d_tau;
        CUDA_CALL(cudaMalloc((void**)&d_tau, rank * sizeof(double)));
        int lwork;
        cusolverDnDgeqrf_bufferSize(cusolverH, rank, rank, d_QR, rank, &lwork);
        double *d_work;
        CUDA_CALL(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

        CUSOLVER_CALL(cusolverDnDgeqrf(cusolverH, rank, rank, d_QR, rank, d_tau, d_work, lwork, d_info));

        int h_info;
        CUDA_CALL(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            printf("QR decomposition failed\n");
            return;
        }

        CUSOLVER_CALL(cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, rank, rank, rank, d_QR, rank, d_tau, d_R, rank, d_work, lwork, d_info));

        CUDA_CALL(cudaMalloc(&d_Q, rank * rank * sizeof(double)));
        CUDA_CALL(cudaMemcpy(d_Q, d_QR, rank * rank * sizeof(double), cudaMemcpyDeviceToDevice));

        cusolverDnDestroy(cusolverH);
        CUDA_CALL(cudaFree(d_info));
        CUDA_CALL(cudaFree(d_tau));
        CUDA_CALL(cudaFree(d_work));
        CUDA_CALL(cudaFree(d_QR));

        find_maxvol<<<(rows + 255) / 256, 256>>>(d_Q, rank, rank,rank, d_max_values, d_indices);

        double *h_max_values = (double*)malloc(rows * sizeof(double));
        int *h_indices = (int*)malloc(rows * sizeof(int));
        CUDA_CALL(cudaMemcpy(h_max_values, d_max_values, rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_indices, d_indices, rows * sizeof(int), cudaMemcpyDeviceToHost));

        int max_idx = thrust::max_element(thrust::device, d_max_values, d_max_values + rows) - d_max_values;

        CUDA_CALL(cudaMemcpy(d_R, d_matrix + max_idx * cols, rank * cols * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CALL(cudaMemcpy(d_matrix_sub, d_Q + max_idx * rank, rank * rank * sizeof(double), cudaMemcpyDeviceToDevice));

        norm = 0.0;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasDnrm2(handle, rank * rank, d_max_values, 1, &norm);
        cublasDestroy(handle);

        free(h_max_values);
        free(h_indices);
        CUDA_CALL(cudaFree(d_max_values));
        CUDA_CALL(cudaFree(d_indices));
        CUDA_CALL(cudaFree(d_Q));
    }

    *U = (double*)malloc(rows * rank * sizeof(double));
    *V = (double*)malloc(rank * cols * sizeof(double));

    CUDA_CALL(cudaMemcpy(*U, d_matrix, rows * rank * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(*V, d_R, rank * cols * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_tensor));
    CUDA_CALL(cudaFree(d_matrix));
    CUDA_CALL(cudaFree(d_A0));
    CUDA_CALL(cudaFree(d_R));
    CUDA_CALL(cudaFree(d_I));
}


void initialize_random_indices(int *indices, int n) {
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    std::random_shuffle(indices, indices + n);
}

void reshape_tensor_to_matrix(double *tensor, double *matrix, int *dims, int n_dims, int rows, int cols) {
    int total_threads = rows * cols;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    reshape_tensor_to_matrix_kernel<<<num_blocks, threads_per_block>>>(tensor, matrix, dims, n_dims, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void reshape_tensor_to_matrix_kernel(double *tensor, double *matrix, int *dims, int n_dims, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;

        int tensor_idx = 0;
        int stride = 1;
        for (int i = n_dims - 1; i >= 0; --i) {
            int dim_idx = col / stride % dims[i];
            tensor_idx += dim_idx * stride;
            stride *= dims[i];
        }

        matrix[idx] = tensor[tensor_idx * rows + row];
    }
}


void tt_cross(double *tensor, int *dims, int n_dims, int rank, double epsilon, double **cores) {
    double *U, *V;
    int N = 1;
    for (int i = 1; i < n_dims; ++i) {
        N *= dims[i];
    }

    double *reshaped_tensor = (double*)malloc(dims[0] * N * sizeof(double));
    reshape_tensor_to_matrix(tensor, reshaped_tensor, dims, n_dims, dims[0], N);

    cross_approximation(reshaped_tensor, dims, n_dims, rank, epsilon, &U, &V);

    cores[0] = U;

    for (int i = 1; i < n_dims - 1; ++i) {
        N /= dims[i];
        double *reshaped_V = (double*)malloc(rank * N * sizeof(double));
        reshape_tensor_to_matrix(V, reshaped_V, dims + i, n_dims - i, rank, N);

        cross_approximation(reshaped_V, dims + i, n_dims - i, rank, epsilon, &U, &V);

        cores[i] = (double*)malloc(dims[i] * rank * rank * sizeof(double));
        reshape_matrix_to_tensor(U, cores[i], dims + i, n_dims - i, dims[i], rank);

        free(reshaped_V);
    }

    cores[n_dims - 1] = V;
    free(reshaped_tensor);
}

void save_tensor_to_file(double *tensor, int *dims, int n_dims, const char *filename) {
    FILE *file = fopen(filename, "wb");
    fwrite(&n_dims, sizeof(int), 1, file);
    fwrite(dims, sizeof(int), n_dims, file);
    int size = 1;
    for (int i = 0; i < n_dims; ++i) {
        size *= dims[i];
    }
    fwrite(tensor, sizeof(double), size, file);
    fclose(file);
}

double* load_tensor_from_file(int *dims, int *n_dims, const char *filename) {
    FILE *file = fopen(filename, "rb");
    fread(n_dims, sizeof(int), 1, file);
    fread(dims, sizeof(int), *n_dims, file);
    int size = 1;
    for (int i = 0; i < *n_dims; ++i) {
        size *= dims[i];
    }
    double *tensor = (double*)malloc(size * sizeof(double));
    fread(tensor, sizeof(double), size, file);
    fclose(file);
    return tensor;
}

void save_tt_to_file(double *tensor, int *dims, int *ranks, int n_dims, const char *filename) {
    FILE *file = fopen(filename, "wb");
    fwrite(&n_dims, sizeof(int), 1, file);
    fwrite(dims, sizeof(int), n_dims, file);
    fwrite(ranks, sizeof(int), n_dims + 1, file);
    int size = 1;
    for (int i = 0; i < n_dims + 1; ++i) {
        size *= ranks[i];
    }
    fwrite(tensor, sizeof(double), size, file);
    fclose(file);
}

double* load_tt_from_file(int *dims, int *ranks, int *n_dims, const char *filename) {
    FILE *file = fopen(filename, "rb");
    fread(n_dims, sizeof(int), 1, file);
    fread(dims, sizeof(int), *n_dims, file);
    fread(ranks, sizeof(int), *n_dims + 1, file);
    int size = 1;
    for (int i = 0; i < *n_dims + 1; ++i) {
        size *= ranks[i];
    }
    double *tensor = (double*)malloc(size * sizeof(double));
    fread(tensor, sizeof(double), size, file);
    fclose(file);
    return tensor;
}

int main() {
    int loaded_n_dims;
    int *loaded_dims = NULL;
    double *loaded_tensor = load_tensor_from_file(&loaded_dims, &loaded_n_dims, "tensor.bin");

    if (loaded_dims == NULL || loaded_tensor == NULL) {
        printf("Error: Failed to load tensor from file\n");
        return EXIT_FAILURE;
    }

    int *ranks = (int*)malloc((loaded_n_dims + 1) * sizeof(int));
    if (ranks == NULL) {
        printf("Error: Memory allocation failed\n");
        free(loaded_dims);
        free(loaded_tensor);
        return EXIT_FAILURE;
    }

    ranks[0] = 1;
    for (int i = 0; i < loaded_n_dims; ++i) {
        ranks[i] = 5;
    }
    ranks[loaded_n_dims] = 1;

    double **cores = (double**)malloc(loaded_n_dims * sizeof(double*));
    if (cores == NULL) {
        printf("Error: Memory allocation failed\n");
        free(loaded_dims);
        free(loaded_tensor);
        free(ranks);
        return EXIT_FAILURE;
    }

    tt_cross(loaded_tensor, loaded_dims, loaded_n_dims, ranks, cores);

    save_tt_to_file(loaded_dims, loaded_n_dims, ranks, cores, "tensor_tt.bin");

    free(loaded_dims);
    free(loaded_tensor);
    free(ranks);

    for (int i = 0; i < loaded_n_dims; ++i) {
        free(cores[i]);
    }

    free(cores);

    return 0;
}

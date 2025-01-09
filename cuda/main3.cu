#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>



// Définition des tailles
#define RAW_DATA_SIZE 32
#define KERNEL_SIZE 5
#define NUM_KERNELS 6
#define C1_SIZE 28
#define S1_SIZE 14


void MatrixInit(float *M, int n, int p,int d, int q) {
    
    if (q == 0){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
    }
    if (q == 1){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  1;
        }
    }
    else if (q == 2){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
        for (int k = 0; k < d; k++){
            M[k * (n * p) + 12] = 2;
        }
    }
    else{
        //Valeurs entre 0 et 1
        for (int i = 0; i < n * p * d; i++){
            M[i] = (float)rand() / RAND_MAX;
             
        }
    }
}
//Pour Print les matrices

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", M[i * p + j]);
        }
        printf("\n");
    }
}
//plot les matrices
void charBckgrndPrint(const char *str, int rgb[3]) {
    printf("\033[48;2;%d;%d;%dm%s\033[0m", rgb[0], rgb[1], rgb[2], str);
}

void imgColorPrint(int height, int width, float *matrix) {
    int row, col;
    char *str = "  ";
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            int val = (int)(matrix[row * width + col] * 255);
            int color[3] = {val, val, val}; // Grayscale
            charBckgrndPrint(str, color);
        }
        printf("\n");
    }
}


//Fonction de convolution        

__global__ void cudaConvolution2D(float* input, float* kernel, float* output, int input_ligne, int input_colonne, int kernel_size, int number_kernel, int output_ligne, int output_colonne){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s;

    if (lig < output_ligne && col < output_colonne){
        
        int tot_kernel = kernel_size * kernel_size;
        int tot_output = output_ligne * output_colonne;
        
        for (int n_k = 0; n_k < number_kernel; n_k++){
            s = 0.0;
            
            for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                    
                    s += input[(lig + kernel_lig) * input_colonne + (col + kernel_col)] * kernel[kernel_lig * kernel_size + kernel_col + n_k * tot_kernel];
                    
                }
            }
            
            output[lig * output_colonne + col + n_k * tot_output] = s;
        }
    }   
}

//Fonction de sous-échantillonnage
__global__ void cudaSubsampling(float *input, float *output, int inputSize, int outputSize) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputRow < outputSize && outputCol < outputSize) {
        int inputRow = outputRow * 2;
        int inputCol = outputCol * 2;

        float sum = 0.0f;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                sum += input[(inputRow + i) * inputSize + (inputCol + j)];
            }
        }
        output[blockIdx.z * outputSize * outputSize + outputRow * outputSize + outputCol] = sum / 4.0f;
    }
}
//Fonction d'activation
__device__ float* activation_tanh(float* M, int M_ligne, int M_colonne, int M_prof) {
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < M_ligne && col < M_colonne){
        
        int tot_M = M_ligne * M_colonne;
        
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            M[lig * M_colonne + col + n_prof * tot_M] = tanh(M[lig * M_colonne + col + n_prof * tot_M]);
        }
            
    }
            
    return M;
}

__global__ void cudaTanh(float* M, int M_ligne, int M_colonne, int M_prof){
    activation_tanh(M, M_ligne, M_colonne, M_prof);
}


//Main 

int main() {
    // Déclaration des matrices
    float *raw_data = (float *)malloc(RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float));
    float *C1_kernel = (float *)malloc(NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *C1_data = (float *)malloc(NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));
    float *S1_data = (float *)malloc(NUM_KERNELS * S1_SIZE * S1_SIZE * sizeof(float));

    // Initialisation des matrices
    MatrixInit(raw_data, RAW_DATA_SIZE, RAW_DATA_SIZE, 1, 3);
    MatrixInit(C1_kernel, KERNEL_SIZE, KERNEL_SIZE, NUM_KERNELS, 2);
    MatrixInit(C1_data, C1_SIZE, C1_SIZE, NUM_KERNELS, 0);
    MatrixInit(S1_data, S1_SIZE, S1_SIZE, NUM_KERNELS, 0);

    // Déclaration des matrices CUDA
    float *d_raw_data, *d_C1_kernel, *d_C1_data, *d_S1_data;

    // Allocation mémoire sur GPU
    cudaMalloc((void **)&d_raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C1_data, NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));
    cudaMalloc((void **)&d_S1_data, NUM_KERNELS * S1_SIZE * S1_SIZE * sizeof(float));

    // Copie des données vers le GPU
    cudaMemcpy(d_raw_data, raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, NUM_KERNELS * S1_SIZE * S1_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Configuration des grilles et des blocs
    dim3 block_size(32, 32);
    dim3 grid_size(1, 1);

    // Lancement des kernels
    cudaConvolution2D<<<grid_size, block_size>>>(d_raw_data, d_C1_kernel, d_C1_data, RAW_DATA_SIZE, RAW_DATA_SIZE, KERNEL_SIZE, NUM_KERNELS, C1_SIZE, C1_SIZE);
    cudaDeviceSynchronize();

    cudaTanh<<<grid_size, block_size>>>(d_C1_data, C1_SIZE, C1_SIZE, NUM_KERNELS);
    cudaDeviceSynchronize();

    cudaConvolution2D<<<grid_size, block_size>>>(d_C1_data, d_C1_kernel, d_S1_data, C1_SIZE, C1_SIZE, KERNEL_SIZE, NUM_KERNELS, S1_SIZE, S1_SIZE);
    cudaDeviceSynchronize();

    // Copie des résultats sur le CPU
    cudaMemcpy(C1_data, d_C1_data, NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, NUM_KERNELS * S1_SIZE * S1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Affichage des matrices
    printf("\nMatrice de base raw_data:\n");
    MatrixPrint(raw_data, RAW_DATA_SIZE, RAW_DATA_SIZE);
    printf("\nNoyau de convolution C1_kernel:\n");
    MatrixPrint(C1_kernel, KERNEL_SIZE, KERNEL_SIZE);
    printf("\nMatrice résultante de la convolution et de la fonction d'activation:\n");
    MatrixPrint(C1_data, C1_SIZE, C1_SIZE);
    printf("\nMatrice résultante du MeanPooling:\n");
    MatrixPrint(S1_data, S1_SIZE, S1_SIZE);


    // Affichage des matrices sous forme de graphiques
    printf("Matrice de base raw_data:\n");
    imgColorPrint(RAW_DATA_SIZE, RAW_DATA_SIZE, raw_data);

    printf("\nMatrice résultante de la convolution et de la fonction d'activation:\n");
    imgColorPrint(C1_SIZE, C1_SIZE, C1_data);

    printf("\nMatrice résultante du MeanPooling:\n");
    imgColorPrint(S1_SIZE, S1_SIZE, S1_data);

    // Libération des mémoires GPU et CPU
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);

    free(raw_data);
    free(C1_kernel);
    free(C1_data);
    free(S1_data);

    return 0;
}

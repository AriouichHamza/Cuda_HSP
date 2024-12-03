#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define RAW_DATA_SIZE 32
#define C1_SIZE 28
#define S1_SIZE 14
#define KERNEL_SIZE 5
#define NUM_KERNELS 6
#define WIDTH 28
#define HEIGHT 28

__device__ float activation_tanh(float x) {
    return tanhf(x);
}

// Fonction d'activation pour le CPU
float activation_tanh_cpu(float x) {
    return tanhf(x);
}

// Initialisation des matrices avec des valeurs aléatoires entre 0 et 1
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX);
    }
}

// Convolution 2D sur CPU
void convolution2D_CPU(float *input, float *kernel, float *output, int inputSize, int outputSize, int kernelSize) {
    for (int k = 0; k < NUM_KERNELS; k++) { // Pour chaque noyau
        for (int i = 0; i < outputSize; i++) { // Parcourir les lignes de sortie
            for (int j = 0; j < outputSize; j++) { // Parcourir les colonnes de sortie
                float sum = 0.0f;
                for (int ki = 0; ki < kernelSize; ki++) { // Parcourir les lignes du noyau
                    for (int kj = 0; kj < kernelSize; kj++) { // Parcourir les colonnes du noyau
                        int inputRow = i + ki;
                        int inputCol = j + kj;
                        sum += input[inputRow * inputSize + inputCol] * kernel[k * kernelSize * kernelSize + ki * kernelSize + kj];
                    }
                }
                output[k * outputSize * outputSize + i * outputSize + j] = activation_tanh_cpu(sum);
            }
        }
    }
}

// Convolution 2D sur GPU
__global__ void cudaConvolution2D(float *input, float *kernel, float *output, int inputSize, int outputSize, int kernelSize) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputRow < outputSize && outputCol < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                int inputRow = outputRow + i;
                int inputCol = outputCol + j;
                sum += input[inputRow * inputSize + inputCol] * kernel[blockIdx.z * kernelSize * kernelSize + i * kernelSize + j];
            }
        }
        output[(blockIdx.z * outputSize * outputSize) + (outputRow * outputSize + outputCol)] = activation_tanh(sum);
    }
}

void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
  }
}


int main() {
    // Allocation des matrices
    float *raw_data = (float *)malloc(RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float));
    float *C1_kernel = (float *)malloc(NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *C1_data_cpu = (float *)malloc(NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));
    float *C1_data_gpu = (float *)malloc(NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));


    // Initialisation des matrices
    int i, j;
    int ***img;
    int color[3]={255,0,0};
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;

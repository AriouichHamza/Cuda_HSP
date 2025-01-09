#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Définitions des tailles
#define IMG_SIZE 28
#define C1_KERNEL_SIZE 5
#define C1_OUTPUT_SIZE 28
#define S2_SIZE 14
#define C3_KERNEL_SIZE 5
#define C3_OUTPUT_SIZE 10
#define S4_SIZE 5
#define FC1_SIZE 120
#define FC2_SIZE 84
#define OUTPUT_SIZE 10

// Charger un fichier binaire dans un tableau
void LoadBinaryFile(const char* filename, float* array, int size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier : %s\n", filename);
        exit(1);
    }
    fread(array, sizeof(float), size, file);
    fclose(file);
}

// CUDA Kernel pour la convolution
__global__ void Convolution2D(float* input, float* kernel, float* bias, float* output, int inputSize, int kernelSize, int outputSize, int numKernels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernelIdx = blockIdx.z;

    if (x < outputSize && y < outputSize && kernelIdx < numKernels) {
        float sum = bias[kernelIdx];
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int inputX = x + j;
                int inputY = y + i;
                sum += input[inputY * inputSize + inputX] * kernel[kernelIdx * kernelSize * kernelSize + i * kernelSize + j];
            }
        }
        output[(kernelIdx * outputSize + y) * outputSize + x] = tanh(sum); // Activation Tanh
    }
}

// CUDA Kernel pour le sous-échantillonnage (Average Pooling)
__global__ void AveragePooling(float* input, float* output, int inputSize, int outputSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                sum += input[(y * 2 + i) * inputSize + (x * 2 + j)];
            }
        }
        output[y * outputSize + x] = sum / 4.0f;
    }
}

// CUDA Kernel pour les couches entièrement connectées
__global__ void FullyConnected(float* input, float* weights, float* bias, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputSize) {
        float sum = bias[idx];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[idx * inputSize + i];
        }
        output[idx] = tanh(sum); // Activation Tanh
    }
}

int main() {
    // Charger l'image MNIST
    float* image = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/mnist_image.bin", image, IMG_SIZE * IMG_SIZE);

    // Charger les poids et biais
    float* C1_weights = (float*)malloc(6 * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));
    float* C1_bias = (float*)malloc(6 * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_0_weights.bin", C1_weights, 6 * C1_KERNEL_SIZE * C1_KERNEL_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_0_bias.bin", C1_bias, 6);

    float* C3_weights = (float*)malloc(16 * C3_KERNEL_SIZE * C3_KERNEL_SIZE * sizeof(float));
    float* C3_bias = (float*)malloc(16 * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_2_weights.bin", C3_weights, 16 * C3_KERNEL_SIZE * C3_KERNEL_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_2_bias.bin", C3_bias, 16);

    float* FC1_weights = (float*)malloc(FC1_SIZE * S4_SIZE * S4_SIZE * 16 * sizeof(float));
    float* FC1_bias = (float*)malloc(FC1_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_5_weights.bin", FC1_weights, FC1_SIZE * S4_SIZE * S4_SIZE * 16);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_5_bias.bin", FC1_bias, FC1_SIZE);

    float* FC2_weights = (float*)malloc(FC2_SIZE * FC1_SIZE * sizeof(float));
    float* FC2_bias = (float*)malloc(FC2_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_6_weights.bin", FC2_weights, FC2_SIZE * FC1_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_6_bias.bin", FC2_bias, FC2_SIZE);

    float* Output_weights = (float*)malloc(OUTPUT_SIZE * FC2_SIZE * sizeof(float));
    float* Output_bias = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_7_weights.bin", Output_weights, OUTPUT_SIZE * FC2_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/ressources pour le TP-20241129/Cuda_HSP/python/weights/layer_7_bias.bin", Output_bias, OUTPUT_SIZE);

    // Libération de la mémoire
    free(image);
    free(C1_weights);
    free(C1_bias);
    free(C3_weights);
    free(C3_bias);
    free(FC1_weights);
    free(FC1_bias);
    free(FC2_weights);
    free(FC2_bias);
    free(Output_weights);
    free(Output_bias);

    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>


void MatrixInit(float *M, int n, int p){
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      M[i * p + j] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
  }
}

void MatrixPrint(float *M, int n, int p){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < p; j++){
      printf("%f ", M[i * p + j]);
    }
    printf("\n");
  }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }

}



int main() {

   
int n = 2, p = 2;
float M1[4];
float M2[4];
float Mout[4];
MatrixInit(M1, n, p);
MatrixInit(M2, n, p);
MatrixAdd(M1, M2, Mout, n, p);

printf("\n");
MatrixPrint(M1, n, p);
printf("\n");
MatrixPrint(M2, n, p);
printf("\n");
MatrixPrint(Mout, n, p);
printf("\n");


return 0;

}
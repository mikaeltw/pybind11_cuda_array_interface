__global__
void saxpy_kernel(float *s, float *x, float *y, int a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
         s[i] = a * x[i] + y[i];
    }
}

void call_saxpy(float *s, float *x, float *y, int a, int n) {
    saxpy_kernel<<<1, 1>>>(s, x, y, a, n);
}
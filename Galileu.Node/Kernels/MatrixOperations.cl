__kernel void matmul_forward(
    __global const double* A,
    __global const double* B,
    __global double* C,
    const int M,
    const int K,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__kernel void elementwise_add_forward(
    __global const double* A,
    __global const double* B,
    __global double* C,
    const int total_elements)
{
    int i = get_global_id(0);
    if (i < total_elements)
    {
        C[i] = A[i] + B[i];
    }
}

__kernel void elementwise_add_broadcast_forward(
    __global const double* A,
    __global const double* B,
    __global double* C,
    const int M,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[col];
    }
}

__kernel void elementwise_multiply(
    __global const double* A,
    __global const double* B,
    __global double* C,
    const int total_elements)
{
    int i = get_global_id(0);
    if (i < total_elements)
    {
        C[i] = A[i] * B[i];
    }
}

__kernel void sigmoid_forward(
    __global const double* input,
    __global double* output,
    const int total_elements)
{
    int i = get_global_id(0);
    if (i < total_elements)
    {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

__kernel void tanh_forward(
    __global const double* input,
    __global double* output,
    const int total_elements)
{
    int i = get_global_id(0);
    if (i < total_elements)
    {
        output[i] = tanh(input[i]);
    }
}

// Lembre-se de adicionar o kernel exp_forward que usamos em NeuralNetworkLSTM.cs
__kernel void exp_forward(
    __global const double* input,
    __global double* output,
    const int total_elements)
{
    int i = get_global_id(0);
    if (i < total_elements)
    {
        output[i] = exp(input[i]);
    }
}
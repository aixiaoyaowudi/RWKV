#include <algorithm>

#define MIN_VALUE (-1e38)

// external compile_flag T_LEN defined

template<typename F>
__global__ void wkv_v4_kernel_forward_cuda_kernel(const int T, const int C,
    const F* __restrict__ const k, const F* __restrict__ const v, const F* __restrict__ const w, const F* __restrict__ const u,  F* __restrict__ const output)
{
    const int sp = blockIdx.y * T * C + blockIdx.x * blockDim.x + threadIdx.x;
    const int cp = blockIdx.x * blockDim.x + threadIdx.x;
    const F w_coef = w[cp];
    const F u_coef = u[cp];
    // F res = 0;
    F numerator = 0, denominator = 0;
    F max_order = MIN_VALUE;
    for(int i = 0; i < T; ++i)
    {
        const int cur_pos = sp + i * C;
        const F wkv_t_norm = max(max_order, u_coef + k[cur_pos]);
        const F exp_wkv_term_1 = expf(max_order - wkv_t_norm);
        const F exp_wkv_term_2 = expf(u_coef + k[cur_pos] - wkv_t_norm);
        output[cur_pos] = (exp_wkv_term_1 * numerator + exp_wkv_term_2 * v[cur_pos])/(exp_wkv_term_1 * denominator + exp_wkv_term_2);
        // output[cur_pos] = float(cp);
        const F new_max_order = max(max_order - w_coef, k[cur_pos]);
        const F trans_term_1 = expf(max_order - w_coef - new_max_order);
        const F trans_term_2 = expf(k[cur_pos] - new_max_order);
        numerator = trans_term_1 * numerator + trans_term_2 * v[cur_pos];
        denominator = trans_term_1 * denominator + trans_term_2;
        max_order = new_max_order;
    }
}
template<typename F>
__global__ void wkv_v4_kernel_forward_cuda_kernel_less_exp(const int T, const int C,
    const F* __restrict__ const k, const F* __restrict__ const v, const F* __restrict__ const w, const F* __restrict__ const u,  F* __restrict__ const output)
{
    const int sp = blockIdx.y * T * C + blockIdx.x * blockDim.x + threadIdx.x;
    const int cp = blockIdx.x * blockDim.x + threadIdx.x;
    const F w_coef = w[cp];
    const F u_coef = u[cp];
    const F exp_diff = exp(u_coef - w_coef), rev_exp_diff = 1/exp_diff;
    // F res = 0;
    F numerator = 0, denominator = 0;
    F max_order = MIN_VALUE;
    for(int i = 0; i < T; ++i)
    {
        const int cur_pos = sp + i * C;
        F exp_wkv_term_1, exp_wkv_term_2;
        F tmp = expf(max_order - u_coef - k[cur_pos]), rtmp = 1/tmp;
        if(max_order < u_coef + k[cur_pos]) exp_wkv_term_1 = tmp, exp_wkv_term_2 = 1;
        else exp_wkv_term_2 = rtmp, exp_wkv_term_1 = 1;
        output[cur_pos] = (exp_wkv_term_1 * numerator + exp_wkv_term_2 * v[cur_pos])/(exp_wkv_term_1 * denominator + exp_wkv_term_2);
        F trans_term_1, trans_term_2;
        tmp *= exp_diff, rtmp *= rev_exp_diff;
        if(max_order - w_coef < k[cur_pos]) trans_term_1 = tmp, trans_term_2 = 1, max_order = k[cur_pos];
        else trans_term_2 = rtmp, trans_term_1 = 1, max_order = max_order - w_coef;
        numerator = trans_term_1 * numerator + trans_term_2 * v[cur_pos];
        denominator = trans_term_1 * denominator + trans_term_2;
    }
}
template<typename F>
__global__ void wkv_v4_kernel_backward_cuda_kernel(const int T, const int C, const F* __restrict__ const k, const F* __restrict__ const v,
    const F* __restrict__ const w, const F* __restrict__ const u, const F* __restrict__ const grad_output, F* __restrict__ const grad_k, F* __restrict__ const grad_v,
    F* __restrict__ const grad_w, F* __restrict__ const grad_u)
{
    const int sp = blockIdx.y * T * C + blockIdx.x * blockDim.x + threadIdx.x, vp = blockIdx.y * C + blockIdx.x * blockDim.x + threadIdx.x;
    const int cp = blockIdx.x * blockDim.x + threadIdx.x;
    const F w_coef = w[cp];
    const F u_coef = u[cp];
    F w_grad = 0, u_grad = 0;
    F output[T_LEN], normalized_denominator[T_LEN], normalized_order[T_LEN]/*, normalized_u_coef[T_LEN]*/;
    F numerator = 0, denominator = 0, partial_numerator = 0, partial_denominator = 0;
    F max_order = MIN_VALUE;
    for(int i = 0; i < T; ++i)
    {
        const int cur_pos = sp + i * C;
        const F wkv_t_norm = max(max_order, u_coef + k[cur_pos]);
        const F exp_wkv_term_1 = expf(max_order - wkv_t_norm);
        const F exp_wkv_term_2 = expf(u_coef + k[cur_pos] - wkv_t_norm);
        // normalized_u_coef[i] = exp_wkv_term_2;
        F nume = (exp_wkv_term_1 * numerator + exp_wkv_term_2 * v[cur_pos]), deno = 1/(exp_wkv_term_1 * denominator + exp_wkv_term_2);
        // output[i] = (exp_wkv_term_1 * numerator + exp_wkv_term_2 * v[cur_pos])/(exp_wkv_term_1 * denominator + exp_wkv_term_2);
        F _output = nume * deno;output[i] = _output;
        w_grad += grad_output[cur_pos] * deno * exp_wkv_term_1 * (partial_numerator - _output * partial_denominator);
        u_grad += exp_wkv_term_2 * grad_output[cur_pos] * deno * (v[cur_pos] - _output);
        normalized_denominator[i] = deno;
        normalized_order[i] = wkv_t_norm;
        // output[cur_pos] = float(cp);
        const F new_max_order = max(max_order - w_coef, k[cur_pos]);
        const F trans_term_1 = expf(max_order - w_coef - new_max_order);
        const F trans_term_2 = expf(k[cur_pos] - new_max_order);
        partial_numerator = (partial_numerator + numerator) * trans_term_1;
        numerator = trans_term_1 * numerator + trans_term_2 * v[cur_pos];
        partial_denominator = (partial_denominator + denominator) * trans_term_1;
        denominator = trans_term_1 * denominator + trans_term_2;
        max_order = new_max_order;
    }
    max_order = MIN_VALUE;
    F accumulated_deno = 0, accumulated_output_with_deno = 0;
    for(int i = T - 1; i >= 0; --i)
    {
        const int cur_pos = sp + i * C;
        const F normalized_u_coef = expf(u_coef + k[cur_pos] - normalized_order[i]);
        const F cur_u_coef = normalized_u_coef/*[i]*/ * normalized_denominator[i] * grad_output[cur_pos];
        const F grad_trans_term = expf(max_order + k[cur_pos]/* - normalized_order[i]*/);
        grad_v[cur_pos] = accumulated_deno * grad_trans_term + cur_u_coef;
        grad_k[cur_pos] = (accumulated_deno * v[cur_pos] - accumulated_output_with_deno) * grad_trans_term +
                          cur_u_coef * (v[cur_pos] - output[i]);
        const F new_max_order = max(max_order - w_coef, -normalized_order[i]);
        const F trans_accumulated_1 = expf(max_order - w_coef - new_max_order);
        const F trans_accumulated_2 = expf(-normalized_order[i] - new_max_order);
        const F constant = normalized_denominator[i] * trans_accumulated_2 * grad_output[cur_pos];
        accumulated_deno = accumulated_deno * trans_accumulated_1 + constant;
        accumulated_output_with_deno = accumulated_output_with_deno * trans_accumulated_1 + constant * output[i];
        max_order = new_max_order;
    }
    grad_w[vp] = - w_grad * w_coef;
    grad_u[vp] = u_grad;
}
void wkv_v4_kernel_forward_cuda(int B, int T, int C, float* k, float* v, float* w, float* u, float* output)
{
    dim3 threads(std::min(C,32)), blocks(C/std::min(C,32), B);
    wkv_v4_kernel_forward_cuda_kernel<<<blocks, threads>>>(T, C, k, v, w, u, output);
}
void wkv_v4_kernel_forward_cuda_less_exp(int B, int T, int C, float* k, float* v, float* w, float* u, float* output)
{
    dim3 threads(std::min(C,32)), blocks(C/std::min(C,32), B);
    wkv_v4_kernel_forward_cuda_kernel_less_exp<<<blocks, threads>>>(T, C, k, v, w, u, output);
}
void wkv_v4_kernel_backward_cuda(int B, int T, int C, float* k, float* v, float* w, float* u, float* grad_output, float *gk, float *gv, float *gw, float *gu)
{
    dim3 threads(std::min(C,32)), blocks(C/std::min(C,32), B);
    wkv_v4_kernel_backward_cuda_kernel<<<blocks, threads>>>(T, C, k, v, w, u, grad_output, gk, gv, gw, gu);
}
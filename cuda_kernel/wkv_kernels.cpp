#include <torch/extension.h>
#include <iostream>

void wkv_v4_kernel_forward_cuda(int B, int T, int C, float* k, float* v, float* w, float* u, float* output);
void wkv_v4_kernel_forward_cuda_less_exp(int B, int T, int C, float* k, float* v, float* w, float* u, float* output);
void wkv_v4_kernel_backward_cuda(int B, int T, int C, float* k, float* v, float* w, float* u, float* grad_output, float *gk, float *gv, float *gw, float *gu);

void wkv_v4_kernel_forward(torch::Tensor k, torch::Tensor v, torch::Tensor w, torch::Tensor u, torch::Tensor output)
{
    int B, T, C;
    B = k.sizes()[0];
    T = k.sizes()[1];
    C = k.sizes()[2];
    // for(int i(0);i<B*T*C;++i) std::cerr<<output.data_ptr<float>()[i]<<",";std::cerr<<std::endl;
    wkv_v4_kernel_forward_cuda(B, T, C, k.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), u.data_ptr<float>(), output.data_ptr<float>());
}
void wkv_v4_kernel_forward_less_exp(torch::Tensor k, torch::Tensor v, torch::Tensor w, torch::Tensor u, torch::Tensor output)
{
    int B, T, C;
    B = k.sizes()[0];
    T = k.sizes()[1];
    C = k.sizes()[2];
    // for(int i(0);i<B*T*C;++i) std::cerr<<output.data_ptr<float>()[i]<<",";std::cerr<<std::endl;
    wkv_v4_kernel_forward_cuda_less_exp(B, T, C, k.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), u.data_ptr<float>(), output.data_ptr<float>());
}
void wkv_v4_kernel_backward(torch::Tensor k, torch::Tensor v, torch::Tensor w, torch::Tensor u, torch::Tensor grad_output, torch::Tensor grad_k,
                            torch::Tensor grad_v, torch::Tensor grad_w, torch::Tensor grad_u)
{
    int B, T, C;
    B = k.sizes()[0];
    T = k.sizes()[1];
    C = k.sizes()[2];
    // for(int i(0);i<B*T*C;++i) std::cerr<<output.data_ptr<float>()[i]<<",";std::cerr<<std::endl;
    wkv_v4_kernel_backward_cuda(B, T, C, k.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), u.data_ptr<float>(), grad_output.data_ptr<float>(),
                                grad_k.data_ptr<float>(), grad_v.data_ptr<float>(), grad_w.data_ptr<float>(), grad_u.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &wkv_v4_kernel_forward, "wkv v4 forward");
    m.def("backward", &wkv_v4_kernel_backward, "wkv v4 backward");
    m.def("forward_less_exp", &wkv_v4_kernel_forward_less_exp, "wkv v4 forward(less exp)");
}

TORCH_LIBRARY(wkv_v4, m)
{
    m.def("forward", &wkv_v4_kernel_forward);
    m.def("backward", &wkv_v4_kernel_backward);
    m.def("forward_less_exp", &wkv_v4_kernel_forward_less_exp);
}

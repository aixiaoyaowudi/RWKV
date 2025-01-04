import torch
from torch.utils.cpp_extension import load
from math import *
import timeit
wkv_v4=load(name="wkv_v4", sources=["wkv_kernels.cpp", "wkv_kernels.cu"],verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3',
                                                                                                          '-Xptxas -O3', '-DT_LEN=1024'])
wkv_v4_std=load(name="wkv_v4_std", sources=["../../RWKV-LM-main/RWKV-LM-main/RWKV-v4/cuda/wkv_op.cpp",
                                            "../../RWKV-LM-main/RWKV-LM-main/RWKV-v4/cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-DTmax=1024'])
emb_len = 768
batch_size = 12
d = 4 * emb_len
w = torch.ones(emb_len, device = 'cuda').contiguous()
u = torch.ones(emb_len, device = 'cuda').contiguous()

L = 12
l = 0

for i in range(emb_len):
    w[i] = -5 + 8 * pow(i / (d-1), 0.7 + 1.3 * l / (L-1))
    u[i] = 0.5 * ((i + 1) % 3-1) + log(0.3)

w = torch.exp(w)
ctx_len = 1024
k = torch.rand((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
v = torch.rand((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
output = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
output_std = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
output_grad = torch.rand((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
grad_k = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
grad_v = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
grad_k_std = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
grad_v_std = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
grad_w = torch.empty((batch_size, emb_len), device = 'cuda').contiguous()
grad_u = torch.empty((batch_size, emb_len), device = 'cuda').contiguous()
grad_w_std = torch.empty((batch_size, emb_len), device = 'cuda').contiguous()
grad_u_std = torch.empty((batch_size, emb_len), device = 'cuda').contiguous()

def output_mse(name, result, result_std):
    print(f"MSE of {name}:", sqrt(torch.sum(torch.square(result - result_std))/result.numel()))

wkv_v4_std.forward(batch_size, ctx_len, emb_len, -w, u, k, v, output_std)
wkv_v4.forward(k, v, w, u, output)
wkv_v4.backward(k, v, w, u, output_grad, grad_k, grad_v, grad_w, grad_u)
wkv_v4_std.backward(batch_size, ctx_len, emb_len, -w, u, k, v, output_grad, grad_w_std, grad_u_std, grad_k_std, grad_v_std)
output_mse("forward result", output, output_std)
output_mse("gradient w", grad_w, grad_w_std)
output_mse("gradient u", grad_u, grad_u_std)
output_mse("gradient k", grad_k, grad_k_std)
output_mse("gradient v", grad_v, grad_v_std)
print("Mine time consumption:", timeit.repeat(
    stmt = "wkv_v4.backward(k, v, w, u, output_grad, grad_k, grad_v, grad_w, grad_u)",
    repeat = 4, number = 100000, globals = globals()))
print("Std time consumption:", timeit.repeat(
    stmt = "wkv_v4_std.backward(batch_size, ctx_len, emb_len, -w, u, k, v, output_grad, grad_w_std, grad_u_std, grad_k_std, grad_v_std)",
    repeat = 4, number = 100000, globals = globals()))


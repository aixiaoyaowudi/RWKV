import torch
from torch.utils.cpp_extension import load
from math import *
import timeit
wkv_v4=load(name="wkv_v4", sources=["wkv_kernels.cpp", "wkv_kernels.cu"],verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3'])
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

def test_accuracy():
    ctx_len = 32
    k = torch.rand((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
    v = torch.rand((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
    output = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
    output_less = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
    right_output = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
    wkv_v4.forward(k, v, w, u, output)
    wkv_v4.forward_less_exp(k, v, w, u, output_less)
    for bi in range(batch_size):
        numerator = torch.zeros((emb_len, ), device = 'cuda')
        denominator = torch.zeros((emb_len, ), device = 'cuda')
        p = torch.zeros((emb_len, ), device = 'cuda')
        q = torch.zeros((emb_len, ), device = 'cuda')
        for ci in range(ctx_len):
            q = torch.max(p, u + k[bi][ci])
            right_output[bi][ci] = (numerator * torch.exp(p - q) + torch.exp(u + k[bi][ci] - q) * v[bi][ci]) / (denominator + torch.exp(u + k[bi][ci] - q))
            q = torch.max(p - w, k[bi][ci])
            numerator = numerator * torch.exp(p - w - q) + torch.exp(k[bi][ci] - q) * v[bi][ci];
            denominator = denominator * torch.exp(p - w -q) + torch.exp(k[bi][ci] - q)
            p = q
    # print(right_output)
    # print(output)
    # print(output_less)
    print("MSE of normal kernel:", sqrt(torch.sum(torch.square(output - right_output))/output.numel()))
    print("MSE of kernel with less exp:", sqrt(torch.sum(torch.square(output_less - right_output))/output.numel()))
    # print("MSE between:", sqrt(torch.sum(torch.square(output_less - output))/output.numel()))
    # print(output)


ctx_len = 1024
k = torch.rand((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
v = torch.rand((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
output = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
output_less = torch.empty((batch_size, ctx_len, emb_len), device = 'cuda').contiguous()
# timeit.repeat("wkv_v4.forward(k, v, w, u, output)", repeat = 4, number = 768)
wkv_v4.forward(k, v, w, u, output)
wkv_v4.forward_less_exp(k, v, w, u, output_less)
print("MSE between:", sqrt(torch.sum(torch.square(output_less - output))/output.numel()))
print("Normal time consumption (test):", timeit.repeat(stmt = "wkv_v4.forward(k, v, w, u, output_less)", repeat = 4, number = 4, globals = globals()))
print("Normal time consumption:", timeit.repeat(stmt = "wkv_v4.forward(k, v, w, u, output_less)", repeat = 4, number = 50000, globals = globals()))
print("Less exp time consumption:", timeit.repeat(stmt = "wkv_v4.forward_less_exp(k, v, w, u, output_less)", repeat = 4, number = 50000, globals = globals()))

# if __name__ == '__main__':
#     test_accuracy()
#     # test_speed()
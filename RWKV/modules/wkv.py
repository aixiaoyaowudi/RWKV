import torch
from torch.utils.cpp_extension import load
from .config import *
import os

# print(__file__)

wkv_v4_kernel = load(
    name = "wkv_v4",
    sources = [os.path.realpath(os.path.join(os.path.dirname(__file__), '../../cuda_kernel/wkv_kernels.cpp')),
               os.path.realpath(os.path.join(os.path.dirname(__file__), '../../cuda_kernel/wkv_kernels.cu'))],
    verbose = True,
    extra_cuda_cflags = ['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DT_LEN={context_len}']
)

class WKV_v4(torch.autograd.Function):
    @staticmethod
    def forward(context, k, v, w, u):
        w = torch.exp(w)
        k, v, w, u = k.contiguous().cuda(), v.contiguous().cuda(), w.contiguous().cuda(), u.contiguous().cuda()
        context.save_for_backward(k, v, w, u)
        output = torch.empty(k.shape, device = 'cuda', memory_format = torch.contiguous_format)
        wkv_v4_kernel.forward(k, v, w, u, output)
        return output
    @staticmethod
    def backward(context, grad_output):
        k, v, w, u = context.saved_tensors
        B, T, C = k.shape
        grad_k = torch.empty((B, T, C), device = 'cuda', memory_format = torch.contiguous_format)
        grad_v = torch.empty((B, T, C), device = 'cuda', memory_format = torch.contiguous_format)
        grad_w = torch.empty((B, C), device = 'cuda', memory_format = torch.contiguous_format)
        grad_u = torch.empty((B, C), device = 'cuda', memory_format = torch.contiguous_format)
        wkv_v4_kernel.backward(k, v, w, u, grad_output.contiguous().cuda(), grad_k, grad_v, grad_w, grad_u)
        return (grad_k, grad_v, grad_w, grad_u)

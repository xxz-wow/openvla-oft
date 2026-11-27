"""Fused training RMSNorm for the frozen BF16 Llama norm weights in OpenVLA LoRA."""
import torch
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import LlamaRMSNorm


@triton.jit
def _norm_forward(X, W, Out, Inv, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0).to(tl.float32)
    weight = tl.load(W + col, mask, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    # Transformers casts back to the input dtype before multiplying the weight.
    normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + row * N + col, normalized * weight, mask)
    tl.store(Inv + row, inv)


@triton.jit
def _norm_backward(DY, X, W, Inv, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0).to(tl.float32)
    dy = tl.load(DY + row * N + col, mask, other=0).to(tl.float32)
    weight = tl.load(W + col, mask, other=0).to(tl.float32)
    inv = tl.load(Inv + row)
    # This is the BF16 gradient at the cast in the original computation.
    g = (dy * weight).to(tl.bfloat16).to(tl.float32)
    grad_variance = (-0.5 * tl.sum(g * x, axis=0)) * ((inv * inv) * inv)
    dx = g * inv + (grad_variance / N) * (2.0 * x)
    tl.store(DX + row * N + col, dx, mask)


class _FrozenNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, eps):
        n = hidden.shape[-1]
        rows = hidden.numel() // n
        out = torch.empty_like(hidden)
        inv = torch.empty(rows, device=hidden.device, dtype=torch.float32)
        block = triton.next_power_of_2(n)
        _norm_forward[(rows,)](hidden, weight, out, inv, n, eps, block,
                               num_warps=8, enable_fp_fusion=False)
        ctx.save_for_backward(hidden, weight, inv)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dy):
        hidden, weight, inv = ctx.saved_tensors
        if torch.is_grad_enabled():
            x = hidden.float()
            inverse = (x.square().mean(dim=-1, keepdim=True) + ctx.eps).rsqrt()
            reference = weight * (x * inverse).to(hidden.dtype)
            dx, = torch.autograd.grad(reference, hidden, dy, create_graph=True)
            return dx, None, None
        n = hidden.shape[-1]
        dx = torch.empty_like(hidden)
        _norm_backward[(inv.numel(),)](dy.contiguous(), hidden, weight, inv, dx, n,
                                        triton.next_power_of_2(n), num_warps=8, enable_fp_fusion=False)
        return dx, None, None


def enable_musa_rmsnorm(model, mode='fast'):
    if mode != 'fast':
        raise ValueError(f'Unknown RMSNorm mode {mode!r}')
    layers = [layer for layer in model.modules() if type(layer) is LlamaRMSNorm]
    for layer in layers:
        _install(layer)
    return len(layers)


def _install(layer):
    original = layer.forward

    def forward(hidden):
        weight = layer.weight
        if (not layer.training or weight.requires_grad or hidden.device.type != 'musa'
                or hidden.dtype != torch.bfloat16 or weight.dtype != hidden.dtype
                or not hidden.is_contiguous() or not weight.is_contiguous()
                or hidden.numel() == 0 or not 0 < hidden.shape[-1] <= 16384):
            return original(hidden)
        return _FrozenNorm.apply(hidden, weight, layer.variance_epsilon)

    layer.forward = forward

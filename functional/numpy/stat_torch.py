import torch
from torch.utils.cpp_extension import load

thistdd_gpu = load(
    name="tbhistdd",
    sources=["src/tbhistdd.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--generate-line-info"],  # , '--use_fast_math'],
    verbose=True,
)


class BatchedFunctionalHistogram2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bins, ranges, bweights):
        bhist, indices, remainder, rsteps = thistdd_gpu.tbhist2d(
            input, bins, ranges, bweights
        )
        ctx.save_for_backward(indices, remainder, rsteps, bweights)
        return bhist

    @staticmethod
    def backward(ctx, bgrad_hist):
        indices, remainder, rsteps, bweights = ctx.saved_tensors
        grad_input, grad_bweights = thistdd_gpu.tbhist2d_bwd(
            bgrad_hist, indices, remainder, bweights, rsteps
        )
        return grad_input, None, None, grad_bweights


class BatchedFunctionalBilinearHistogram2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bins, ranges, bweights):
        bhist, indices, remainder, rsteps = thistdd_gpu.tbbilinearhist2d(
            input, bins, ranges, bweights
        )
        ctx.save_for_backward(indices, remainder, rsteps, bweights)
        return bhist

    @staticmethod
    def backward(ctx, bgrad_hist):
        indices, remainder, rsteps, bweights = ctx.saved_tensors
        grad_input, grad_bweights = thistdd_gpu.tbbilinearhist2d_bwd(
            bgrad_hist, indices, remainder, bweights, rsteps
        )
        return grad_input, None, None, grad_bweights


class BatchedFunctionalHistogram3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bins, ranges, bweights):
        bhist, indices, remainder, rsteps = thistdd_gpu.tbhist3d(
            input, bins, ranges, bweights
        )
        ctx.save_for_backward(indices, remainder, rsteps, bweights)
        return bhist

    @staticmethod
    def backward(ctx, bgrad_hist):
        indices, remainder, rsteps, bweights = ctx.saved_tensors
        grad_input, grad_bweights = thistdd_gpu.tbhist3d_bwd(
            bgrad_hist, indices, remainder, bweights, rsteps
        )
        return grad_input, None, None, grad_bweights

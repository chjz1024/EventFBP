"""
Containing legacy torch implementations that enable higher-order AD but with worse performance
"""

import torch


# @torch.compile
def batchedthistdd(
    input: torch.Tensor, bins: torch.Tensor, ranges: torch.Tensor, weights: torch.Tensor
):
    """
    Args:
        input: Tensor[B,N,D]
        bins: Tensor[D]
        ranges: Tensor[D,2]
        weights: Tensor[B,N]

    Returns:
        hist: Tensor[B,W1,W2,...,WD]
        indices: Tensor[B,N,D]
        remainder: Tensor[B,N,D]
        steps: Tensor[D]
    """
    device = input.device
    B, N, D = input.shape
    assert bins.shape == (D,)
    assert ranges.shape == (D, 2)
    assert weights.shape == (B, N)
    hist = torch.zeros(*(B, *(bins + 2)), device=device)  # (B,W1,W2,...,WD)
    steps = (ranges[:, 1] - ranges[:, 0]) / bins  # (D,)
    coord = (input - ranges[:, 0]) / steps  # (B,N,D)
    indices = coord.floor()  # (B,N,D)
    remainder = coord - indices  # (B,N,D)
    safe_indices = torch.clamp(
        indices + 1, torch.zeros_like(bins), bins + 1
    ).long()  # (B,N,D)
    return (
        hist.index_put(
            (
                torch.arange(B, device=device).repeat_interleave(N),
                *safe_indices.reshape(-1, D).T,
            ),
            weights.reshape(-1),
            accumulate=True,
        )[(slice(None),) + (slice(1, -1),) * D],
        indices,
        remainder,
        steps,
    )


# @torch.compile
def batchedthistdd_bwd(
    grad_hist: torch.Tensor,
    indices: torch.Tensor,
    remainder: torch.Tensor,
    weights: torch.Tensor,
    steps: torch.Tensor,
):
    """
    Args:
        grad_hist: Tensor[B,W1,W2,...,WD]
        indices: Tensor[B,N,D]
        remainder: Tensor[B,N,D]
        weights: Tensor[B,N]
        steps: Tensor[D]

    Returns:
        gradx: Tensor[B,N,D]
        gradw: Tensor[B,N]
    """
    device = grad_hist.device
    B, N, D = indices.shape
    assert indices.shape == (B, N, D)
    assert remainder.shape == (B, N, D)
    assert weights.shape == (B, N)
    assert steps.shape == (D,)
    grad_hist_padded = torch.nn.functional.pad(grad_hist, (1, 1) * D + (0, 0))

    # (B,N,D,3)
    remainder_weights_kernel = torch.stack(
        [(1 - remainder) ** 2 / 2, remainder - remainder**2 + 1 / 2, remainder**2 / 2],
        dim=3,
    )
    remainder_weights_dkernel = (
        torch.stack([remainder - 1, 1 - 2 * remainder, remainder], dim=3)
        / steps[None, None, :, None]
    )

    k_combinations = torch.cartesian_prod(
        *[torch.arange(3, device=device) for _ in range(D)]
    ).reshape(-1, D)
    # (B,N,3^D,D)
    safe_expanded_indices = torch.clip(
        indices[:, :, None, :] + k_combinations[None, None, :, :],
        torch.zeros(D, dtype=torch.int64, device=device),
        torch.tensor(grad_hist.shape[1:], device=device) + 1,
    ).long()
    # (B,N,3^D)
    grad_vals = grad_hist_padded[
        (
            torch.arange(B, device=device)
            .repeat_interleave(N * 3**D)
            .reshape(B, N, 3**D),
            *safe_expanded_indices.permute(3, 0, 1, 2),
        )
    ]

    # 3x performance degradation without loop
    # # (B,N,D,3^D)
    # kernel_vals = remainder_weights_kernel[torch.arange(B,device=device)[:,None,None,None],torch.arange(N,device=device)[None,:,None,None], torch.arange(D,device=device)[None,None,:,None], k_combinations.T[None,None,:,:]]
    # dkernel_vals = remainder_weights_dkernel[torch.arange(B,device=device)[:,None,None,None],torch.arange(N,device=device)[None,:,None,None], torch.arange(D,device=device)[None,None,:,None], k_combinations.T[None,None,:,:]]

    # # (B,N,3^D)
    # weights_prod = torch.prod(kernel_vals, dim=2)

    # # (B,N,D,3^D)
    # factor = torch.where(kernel_vals==0, 0, dkernel_vals / kernel_vals)

    # # (B,N,D,3^D)
    # dweights_prod = weights_prod[:,:,None,:] * weights[:,:,None,None] * factor

    # return torch.sum(dweights_prod * grad_vals[:,:,None,:], dim = 3),torch.sum(weights_prod * grad_vals, dim = 2)

    # (B,N,3^D)
    weights_prod = torch.prod(
        remainder_weights_kernel[
            torch.arange(B, device=device)[:, None, None, None],
            torch.arange(N, device=device)[None, :, None, None],
            torch.arange(D, device=device)[None, None, :, None],
            k_combinations.T[None, None, :, :],
        ],
        dim=2,
    )

    # (B,N)
    gradw = torch.sum(weights_prod * grad_vals, dim=2)

    gradx_list = []

    # gradx = torch.zeros((N,D))
    for d in range(D):
        hybrid_weights = torch.where(
            torch.arange(D, device=device)[None, None, :, None] == d,
            remainder_weights_dkernel,
            remainder_weights_kernel,
        )
        dweights_prod = (
            torch.prod(
                hybrid_weights[
                    torch.arange(B, device=device)[:, None, None, None],
                    torch.arange(N, device=device)[None, :, None, None],
                    torch.arange(D, device=device)[None, None, :, None],
                    k_combinations.T[None, None, :, :],
                ],
                dim=2,
            )
            * weights[:, :, None]
        )
        # gradx[:,d] = torch.sum(dweights_prod * grad_vals, dim=1)
        gradx_list.append(torch.sum(dweights_prod * grad_vals, dim=2))

    return torch.stack(gradx_list, dim=2), gradw


# 2x performance degradation
# @torch.compile
def batchedthistddloop(
    input: torch.Tensor, bins: torch.Tensor, ranges: torch.Tensor, weights: torch.Tensor
):
    """
    Args:
        input: Tensor[B,N,D]
        bins: Tensor[D]
        ranges: Tensor[D,2]
        weights: Tensor[B,N]
    """
    B, N, D = input.shape
    assert bins.shape == (D,)
    assert ranges.shape == (D, 2)
    assert weights.shape == (B, N)
    hist = torch.zeros(*(bins + 2))  # (B,W1,W2,...,WD)
    steps = (ranges[:, 1] - ranges[:, 0]) / bins  # (D,)
    coord = (input - ranges[:, 0]) / steps  # (B,N,D)
    indices = coord.floor()  # (B,N,D)
    remainder = coord - indices  # (B,N,D)
    safe_indices = torch.clamp(
        indices + 1, torch.zeros_like(bins), bins + 1
    ).long()  # (B,N,D)

    hist = torch.stack(
        [
            hist.index_put(tuple(safe_indices[b].T), weights[b], accumulate=True)[
                (slice(1, -1),) * D
            ]
            for b in range(B)
        ]
    )
    return hist, indices, remainder, steps


# @torch.compile
def thistdd(
    input: torch.Tensor, bins: torch.Tensor, ranges: torch.Tensor, weights: torch.Tensor
):
    """
    Args:
        input: Tensor[N,D]
        bins: Tensor[D]
        ranges: Tensor[D,2]
        weights: Tensor[N]

    Returns:
        hist: Tensor[W1,W2,...,WD]
        indices: Tensor[N,D]
        remainder: Tensor[N,D]
        steps: Tensor[D]
    """
    device = input.device
    N, D = input.shape
    hist = torch.zeros(*(bins + 2), device=device)
    steps = (ranges[:, 1] - ranges[:, 0]) / bins
    coord = (input - ranges[:, 0]) / steps
    indices = coord.floor()
    remainder = coord - indices
    safe_indices = torch.clamp(indices + 1, torch.zeros_like(bins), bins + 1).long()
    return (
        hist.index_put(tuple(safe_indices.T), weights, accumulate=True)[
            (slice(1, -1),) * D
        ],
        indices,
        remainder,
        steps,
    )


# @torch.compile
def thistdd_bwd(
    grad_hist: torch.Tensor,
    indices: torch.Tensor,
    remainder: torch.Tensor,
    weights: torch.Tensor,
    steps: torch.Tensor,
):
    """
    Args:
        grad_hist: Tensor[W1,W2,...,WD]
        indices: Tensor[N,D]
        remainder: Tensor[N,D]
        weights: Tensor[N]
        steps: Tensor[D]

    Returns:
        gradx: Tensor[N,D]
        gradw: Tensor[N]
    """
    device = grad_hist.device
    N, D = indices.shape
    grad_hist_padded = torch.nn.functional.pad(grad_hist, (1, 1) * D)

    remainder_weights_kernel = torch.stack(
        [(1 - remainder) ** 2 / 2, remainder - remainder**2 + 1 / 2, remainder**2 / 2],
        dim=2,
    )
    remainder_weights_dkernel = (
        torch.stack([remainder - 1, 1 - 2 * remainder, remainder], dim=2)
        / steps[None, :, None]
    )

    k_combinations = torch.cartesian_prod(
        *[torch.arange(3, device=device) for _ in range(D)]
    ).reshape(-1, D)
    safe_expanded_indices = torch.clip(
        indices[:, None, :] + k_combinations[None, :, :],
        torch.zeros(D, dtype=torch.int64, device=device),
        torch.tensor(grad_hist.shape, device=device) + 1,
    ).long()
    grad_vals = grad_hist_padded[tuple(safe_expanded_indices.permute(2, 0, 1))]

    # kernel_vals = remainder_weights_kernel[torch.arange(N,device=device)[:,None,None], torch.arange(D,device=device)[None, :, None], k_combinations.T[None, :, :]]
    # dkernel_vals = remainder_weights_dkernel[torch.arange(N,device=device)[:,None,None], torch.arange(D,device=device)[None, :, None], k_combinations.T[None, :, :]]

    # weights_prod = torch.prod(kernel_vals, dim=1)

    # factor = torch.where(kernel_vals==0, 0, dkernel_vals / kernel_vals)

    # dweights_prod = weights_prod[:,None,:] * weights[:, None, None] * factor

    # return torch.sum(dweights_prod * grad_vals[:, None, :], dim = 2),torch.sum(weights_prod * grad_vals, dim = 1)

    weights_prod = torch.prod(
        remainder_weights_kernel[
            torch.arange(N, device=device)[:, None, None],
            torch.arange(D, device=device)[None, :, None],
            k_combinations.T[None, :, :],
        ],
        dim=1,
    )

    gradw = torch.sum(weights_prod * grad_vals, dim=1)

    gradx_list = []

    # gradx = torch.zeros((N,D))
    for d in range(D):
        hybrid_weights = torch.where(
            torch.arange(D, device=device)[None, :, None] == d,
            remainder_weights_dkernel,
            remainder_weights_kernel,
        )
        dweights_prod = (
            torch.prod(
                hybrid_weights[
                    torch.arange(N, device=device)[:, None, None],
                    torch.arange(D, device=device)[None, :, None],
                    k_combinations.T[None, :, :],
                ],
                dim=1,
            )
            * weights[:, None]
        )
        # gradx[:,d] = torch.sum(dweights_prod * grad_vals, dim=1)
        gradx_list.append(torch.sum(dweights_prod * grad_vals, dim=1))

    return torch.stack(gradx_list, dim=1), gradw


# @torch.compile
def thistdd_fwd(
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    indices: torch.Tensor,
    remainder: torch.Tensor,
    weights: torch.Tensor,
    steps: torch.Tensor,
    bins: torch.Tensor,
):
    """
    grad_input: Tensor[NxD]
    grad_weight: Tensor[N]
    indices: Tensor[NxD]
    remainder: Tensor[NxD]
    weights: Tensor[N]
    steps: Tensor[D]
    """
    N, D = indices.shape
    grad_hist_padded = torch.zeros(*(bins + 2))

    remainder_weights_kernel = torch.stack(
        [(1 - remainder) ** 2 / 2, remainder - remainder**2 + 1 / 2, remainder**2 / 2],
        dim=2,
    )
    remainder_weights_dkernel = (
        torch.stack([remainder - 1, 1 - 2 * remainder, remainder], dim=2)
        / steps[None, :, None]
    )

    k_combinations = torch.cartesian_prod(*[torch.arange(3) for _ in range(D)]).reshape(
        -1, D
    )
    safe_indices = torch.clip(
        indices[:, None, :] + k_combinations[None, :, :],
        torch.zeros_like(bins),
        bins + 1,
    ).long()
    weights_prod = (
        torch.prod(
            remainder_weights_kernel[
                torch.arange(N)[:, None, None],
                torch.arange(D)[None, :, None],
                k_combinations.T[None, :, :],
            ],
            dim=1,
        )
        * grad_weight[:, None]
    )

    grad_hist_padded = grad_hist_padded.index_put(
        tuple(safe_indices.view(-1, D).T), weights_prod.view(-1), accumulate=True
    )

    for d in range(D):
        hybrid_weights = torch.where(
            torch.arange(D)[None, :, None] == d,
            remainder_weights_dkernel,
            remainder_weights_kernel,
        )
        dweights_prod = (
            torch.prod(
                hybrid_weights[
                    torch.arange(N)[:, None, None],
                    torch.arange(D)[None, :, None],
                    k_combinations.T[None, :, :],
                ],
                dim=1,
            )
            * grad_input[:, d : d + 1]
            * weights[:, None]
        )
        grad_hist_padded = grad_hist_padded.index_put(
            tuple(safe_indices.view(-1, D).T), dweights_prod.view(-1), accumulate=True
        )

    return grad_hist_padded[(slice(1, -1),) * D]


# @torch.compile
def tbilinearhistdd(
    input: torch.Tensor, bins: torch.Tensor, ranges: torch.Tensor, weights: torch.Tensor
):
    """
    input: Tensor[NxD]
    bins: Tensor[D]
    ranges: Tensor[Dx2]
    weights: Tensor[N]
    """
    N, D = input.shape
    hist = torch.zeros(*(bins + 2))
    steps = (ranges[:, 1] - ranges[:, 0]) / bins
    coord = (input - ranges[:, 0]) / steps
    indices = coord.floor()
    remainder = coord - indices  # NxD

    k_combinations = torch.cartesian_prod(*[torch.arange(2) for _ in range(D)]).reshape(
        -1, D
    )
    safe_expanded_indices = torch.clip(
        indices[:, None, :] + k_combinations[None, :, :] + 1,
        torch.zeros_like(bins),
        bins + 1,
    ).long()
    weights_kernel = torch.stack([1 - remainder, remainder], dim=2)
    weights_prod = (
        torch.prod(
            weights_kernel[
                torch.arange(N)[:, None, None],
                torch.arange(D)[None, :, None],
                k_combinations.T[None, :, :],
            ],
            dim=1,
        )
        * weights[:, None]
    )

    return (
        hist.index_put(
            tuple(safe_expanded_indices.view(-1, D).T),
            weights_prod.view(-1),
            accumulate=True,
        )[(slice(1, -1),) * D],
        indices,
        remainder,
        steps,
    )


# @torch.compile
def tbilinearhistdd_bwd(
    grad_hist: torch.Tensor,
    indices: torch.Tensor,
    remainder: torch.Tensor,
    weights: torch.Tensor,
    steps: torch.Tensor,
):
    """
    grad_hist: Tensor[W1xW2x...xWD]
    indices: Tensor[NxD]
    remainder: Tensor[NxD]
    weights: Tensor[N]
    steps: Tensor[D]
    """
    N, D = indices.shape
    grad_hist_padded = torch.nn.functional.pad(grad_hist, (1, 1) * D)

    remainder_weights_kernel = torch.stack(
        [
            (1 - remainder) ** 3 / 6,
            (3 * remainder**3 - 6 * remainder**2 + 4) / 6,
            (-3 * remainder**3 + 3 * remainder**2 + 3 * remainder + 1) / 6,
            remainder**3 / 6,
        ],
        dim=2,
    )
    remainder_weights_dkernel = (
        torch.stack(
            [
                -((1 - remainder) ** 2) / 2,
                (3 * remainder**2 - 4 * remainder) / 2,
                (-3 * remainder**2 + 2 * remainder + 1) / 2,
                remainder**2 / 2,
            ],
            dim=2,
        )
        / steps[None, :, None]
    )

    k_combinations = torch.cartesian_prod(*[torch.arange(4) for _ in range(D)]).reshape(
        -1, D
    )
    safe_expanded_indices = torch.clip(
        indices[:, None, :] + k_combinations[None, :, :],
        torch.zeros(D, dtype=torch.int64),
        torch.tensor(grad_hist.shape) + 1,
    ).long()
    grad_vals = grad_hist_padded[tuple(safe_expanded_indices.permute(2, 0, 1))]
    weights_prod = torch.prod(
        remainder_weights_kernel[
            torch.arange(N)[:, None, None],
            torch.arange(D)[None, :, None],
            k_combinations.T[None, :, :],
        ],
        dim=1,
    )

    gradw = torch.sum(weights_prod * grad_vals, dim=1)

    gradx_list = []

    for d in range(D):
        hybrid_weights = torch.where(
            torch.arange(D)[None, :, None] == d,
            remainder_weights_dkernel,
            remainder_weights_kernel,
        )
        dweights_prod = (
            torch.prod(
                hybrid_weights[
                    torch.arange(N)[:, None, None],
                    torch.arange(D)[None, :, None],
                    k_combinations.T[None, :, :],
                ],
                dim=1,
            )
            * weights[:, None]
        )
        gradx_list.append(torch.sum(dweights_prod * grad_vals, dim=1))

    return torch.stack(gradx_list, dim=1), gradw


# @torch.compile
def tbilinearhistdd_fwd(
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    indices: torch.Tensor,
    remainder: torch.Tensor,
    weights: torch.Tensor,
    steps: torch.Tensor,
    bins: torch.Tensor,
):
    """
    grad_input: Tensor[NxD]
    grad_weight: Tensor[N]
    indices: Tensor[NxD]
    remainder: Tensor[NxD]
    weights: Tensor[N]
    steps: Tensor[D]
    """
    N, D = indices.shape
    grad_hist_padded = torch.zeros(*(bins + 2))

    remainder_weights_kernel = torch.stack(
        [
            (1 - remainder) ** 3 / 6,
            (3 * remainder**3 - 6 * remainder**2 + 4) / 6,
            (-3 * remainder**3 + 3 * remainder**2 + 3 * remainder + 1) / 6,
            remainder**3 / 6,
        ],
        dim=2,
    )
    remainder_weights_dkernel = (
        torch.stack(
            [
                -((1 - remainder) ** 2) / 2,
                (3 * remainder**2 - 4 * remainder) / 2,
                (-3 * remainder**2 + 2 * remainder + 1) / 2,
                remainder**2 / 2,
            ],
            dim=2,
        )
        / steps[None, :, None]
    )

    k_combinations = torch.cartesian_prod(*[torch.arange(4) for _ in range(D)]).reshape(
        -1, D
    )
    safe_expanded_indices = torch.clip(
        indices[:, None, :] + k_combinations[None, :, :],
        torch.zeros_like(bins),
        bins + 1,
    ).long()
    weights_prod = (
        torch.prod(
            remainder_weights_kernel[
                torch.arange(N)[:, None, None],
                torch.arange(D)[None, :, None],
                k_combinations.T[None, :, :],
            ],
            dim=1,
        )
        * grad_weight[:, None]
    )

    grad_hist_padded = grad_hist_padded.index_put(
        tuple(safe_expanded_indices.view(-1, D).T),
        weights_prod.view(-1),
        accumulate=True,
    )

    for d in range(D):
        hybrid_weights = torch.where(
            torch.arange(D)[None, :, None] == d,
            remainder_weights_dkernel,
            remainder_weights_kernel,
        )
        dweights_prod = (
            torch.prod(
                hybrid_weights[
                    torch.arange(N)[:, None, None],
                    torch.arange(D)[None, :, None],
                    k_combinations.T[None, :, :],
                ],
                dim=1,
            )
            * grad_input[:, d : d + 1]
            * weights[:, None]
        )
        grad_hist_padded = grad_hist_padded.index_put(
            tuple(safe_expanded_indices.view(-1, D).T),
            dweights_prod.view(-1),
            accumulate=True,
        )

    return grad_hist_padded[(slice(1, -1),) * D]


# @torch.compile
def tbilinearhistdd_bwd_naive(
    grad_hist: torch.Tensor,
    indices: torch.Tensor,
    remainder: torch.Tensor,
    weights: torch.Tensor,
    steps: torch.Tensor,
):
    """
    grad_hist: Tensor[W1xW2x...xWD]
    indices: Tensor[NxD]
    remainder: Tensor[NxD]
    weights: Tensor[N]
    steps: Tensor[D]
    """
    N, D = indices.shape
    grad_hist_padded = torch.nn.functional.pad(grad_hist, (1, 1) * D)

    remainder_weights_kernel = torch.stack([1 - remainder, remainder], dim=2)
    remainder_weights_dkernel = (
        torch.stack([-torch.ones_like(remainder), torch.ones_like(remainder)], dim=2)
        / steps[None, :, None]
    )

    k_combinations = torch.cartesian_prod(*[torch.arange(2) for _ in range(D)]).reshape(
        -1, D
    )
    # expanded_indices = indices[:, None, :] - 1 + k_combinations[None, :, :]
    safe_expanded_indices = torch.clip(
        indices[:, None, :] + k_combinations[None, :, :] + 1,
        torch.zeros(D, dtype=torch.int64),
        torch.tensor(grad_hist.shape) + 1,
    ).long()
    grad_vals = grad_hist_padded[tuple(safe_expanded_indices.permute(2, 0, 1))]
    weights_prod = torch.prod(
        remainder_weights_kernel[
            torch.arange(N)[:, None, None],
            torch.arange(D)[None, :, None],
            k_combinations.T[None, :, :],
        ],
        dim=1,
    )

    gradw = torch.sum(weights_prod * grad_vals, dim=1)

    gradx_list = []

    for d in range(D):
        hybrid_weights = torch.where(
            torch.arange(D)[None, :, None] == d,
            remainder_weights_dkernel,
            remainder_weights_kernel,
        )
        dweights_prod = (
            torch.prod(
                hybrid_weights[
                    torch.arange(N)[:, None, None],
                    torch.arange(D)[None, :, None],
                    k_combinations.T[None, :, :],
                ],
                dim=1,
            )
            * weights[:, None]
        )
        gradx_list.append(torch.sum(dweights_prod * grad_vals, dim=1))

    return torch.stack(gradx_list, dim=1), gradw


# @torch.compile
def thist2d_bwd(
    grad_hist: torch.Tensor,
    indices: torch.Tensor,
    remainder: torch.Tensor,
    weights: torch.Tensor,
    rsteps: torch.Tensor,
):
    """
    Args:
        grad_hist: Tensor[W1,W2]
        indices: Tensor[N,D]
        remainder: Tensor[N,D]
        weights: Tensor[N]
        steps: Tensor[D]

    Returns:
        gradx: Tensor[N,D]
        gradw: Tensor[N]
    """
    device = grad_hist.device
    N, D = indices.shape
    assert D == 2
    grad_hist_padded = torch.nn.functional.pad(grad_hist, (3, 3) * D)

    remainder_weights_kernel = [
        (1 - remainder) ** 2 / 2,
        remainder - remainder**2 + 1 / 2,
        remainder**2 / 2,
    ]
    remainder_weights_dkernel = [
        (remainder - 1) * rsteps,
        (1 - 2 * remainder) * rsteps,
        remainder * rsteps,
    ]

    gradw = torch.zeros_like(weights)
    gradx = torch.zeros_like(indices)
    safe_indices = tuple(
        torch.clamp(
            indices + 3,
            torch.ones_like(rsteps),
            torch.tensor(grad_hist.shape, device=device) + 4,
        )
        .long()
        .T
    )
    for k0 in range(3):
        for k1 in range(3):
            grad_vals = grad_hist_padded[
                (safe_indices[0] + k0 - 1, safe_indices[1] + k1 - 1)
            ]

            w = remainder_weights_kernel[k0][:, 0] * remainder_weights_kernel[k1][:, 1]
            wd0 = (
                remainder_weights_dkernel[k0][:, 0]
                * remainder_weights_kernel[k1][:, 1]
                * weights
            )
            wd1 = (
                remainder_weights_kernel[k0][:, 0]
                * remainder_weights_dkernel[k1][:, 1]
                * weights
            )

            gradw += w * grad_vals
            gradx[:, 0] += wd0 * grad_vals
            gradx[:, 1] += wd1 * grad_vals

    return gradx, gradw


class FunctionalHistogramdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bins, range, weights):
        """
        Args:
            ctx: torch context
            input: Tensor[N,D]
            bins: Tensor[D]
            ranges: Tensor[D,2]
            weights: Tensor[N]

        Returns:
            hist: Tensor[W1,W2,...,WD]
        """

        hist, indices, remainder, steps = thistdd(input, bins, range, weights)
        ctx.save_for_backward(indices, remainder, weights, steps)
        return hist

    @staticmethod
    def backward(ctx, grad_hist):
        indices, remainder, weights, steps = ctx.saved_tensors
        gx, gw = thistdd_bwd(grad_hist, indices, remainder, weights, steps)
        return gx, None, None, gw


class BatchedFunctionalHistogramdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bins, ranges, weights):
        """
        Args:
            ctx: torch context
            input: Tensor[B,N,D]
            bins: Tensor[D]
            ranges: Tensor[D,2]
            weights: Tensor[B,N]

        Returns:
            hist: Tensor[B,W1,W2,...,WD]
        """
        hist, indices, remainder, steps = batchedthistdd(input, bins, ranges, weights)
        ctx.save_for_backward(indices, remainder, weights, steps)
        return hist

    @staticmethod
    def backward(ctx, grad_hist):
        indices, remainder, weights, steps = ctx.saved_tensors
        gx, gw = batchedthistdd_bwd(grad_hist, indices, remainder, weights, steps)
        return gx, None, None, gw


class ComposableFunctionalHistogramdd(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input, bins, range, weights):
        """
        Args:
            input: Tensor[N,D]
            bins: Tensor[D]
            range: Tensor[D,2]
            weights: Tensor[N]

        Returns:
            hist: Tensor[W1,W2,...,WD]
            indices: Tensor[N,D]
            remainder: Tensor[N,D]
            steps: Tensor[D]
        """
        return thistdd(input, bins, range, weights)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        _0, bins, _1, weights = inputs
        _2, indices, remainder, steps = outputs
        ctx.save_for_backward(indices, remainder, weights, steps)
        ctx.save_for_forward(indices, remainder, weights, steps, bins)

    @staticmethod
    def backward(ctx, grad_hist, _0, _1, _2):  # vjp
        indices, remainder, weights, steps = ctx.saved_tensors
        gx, gw = thistdd_bwd(grad_hist, indices, remainder, weights, steps)
        return gx, None, None, gw

    @staticmethod
    def jvp(ctx, grad_input, _0, _1, grad_weight):  # forward autodiff
        indices, remainder, weights, steps, bins = ctx.saved_tensors
        return (
            thistdd_fwd(
                grad_input, grad_weight, indices, remainder, weights, steps, bins
            ),
            torch.zeros_like(indices),
            torch.zeros_like(remainder),
            torch.zeros_like(steps),
        )


class ComposableFunctionalBilinearHistogramdd(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input, bins, range, weights):
        """
        Args:
            input: Tensor[N,D]
            bins: Tensor[D]
            range: Tensor[D,2]
            weights: Tensor[N]

        Returns:
            hist: Tensor[W1,W2,...,WD]
            indices: Tensor[N,D]
            remainder: Tensor[N,D]
            steps: Tensor[D]
        """
        return tbilinearhistdd(input, bins, range, weights)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        _0, bins, _1, weights = inputs
        _2, indices, remainder, steps = outputs
        ctx.save_for_backward(indices, remainder, weights, steps)
        ctx.save_for_forward(indices, remainder, weights, steps, bins)

    @staticmethod
    def backward(ctx, grad_hist, _0, _1, _2):  # vjp
        indices, remainder, weights, steps = ctx.saved_tensors
        gx, gw = tbilinearhistdd_bwd(grad_hist, indices, remainder, weights, steps)
        return gx, None, None, gw

    @staticmethod
    def jvp(ctx, grad_input, _0, _1, grad_weight):  # forward autodiff
        indices, remainder, weights, steps, bins = ctx.saved_tensors
        return (
            tbilinearhistdd_fwd(
                grad_input, grad_weight, indices, remainder, weights, steps, bins
            ),
            torch.zeros_like(indices),
            torch.zeros_like(remainder),
            torch.zeros_like(steps),
        )
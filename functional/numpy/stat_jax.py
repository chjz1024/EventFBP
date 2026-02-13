import jax
import jax.numpy as jnp
import numpy as np


def jhistdd_fixedrange_linear(bins: np.ndarray, ranges: jax.Array):
    """
    Rect binning kernel with linear recon kernel.

    Enabling arbitrary input dimension
    """

    steps = (ranges[:, 1] - ranges[:, 0]) / bins

    @jax.custom_jvp
    def jhistdd_fixedrange_(input: jax.Array, weights: jax.Array):
        # N, D = input.shape
        hist: jax.Array = jnp.zeros(bins)
        indices = ((input - ranges[:, 0]) / steps).astype(int)
        return hist.at[tuple(indices.T)].add(weights, wrap_negative_indices=False)
        # return jnp.histogramdd(input, bins, ranges, weights)[0]

    @jhistdd_fixedrange_.defjvp
    def jhistdd_fixedrange_jvp(primals, tangents):
        input, weights = primals
        input_dot, weights_dot = tangents

        N, D = input.shape
        hist = jhistdd_fixedrange_(input, weights)

        grad_hist = jnp.zeros_like(hist)
        # steps = (ranges[:,1]-ranges[:,0])/jnp.array(bins)
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        indices = indices.astype(int)
        remainder /= steps

        remainder_weights_kernel = jnp.stack(
            [
                (1 - remainder) ** 2 / 2,
                remainder - remainder**2 + 1 / 2,
                remainder**2 / 2,
            ],
            axis=2,
        )
        remainder_weights_dkernel = (
            jnp.stack([remainder - 1, 1 - 2 * remainder, remainder], axis=2)
            / steps[None, :, None]
        )

        k_combinations = jnp.stack(
            jnp.meshgrid(
                *[
                    jnp.arange(3),
                ]
                * D,
                indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, D)
        expanded_indices = indices[:, None, :] + k_combinations[None, :, :] - 1
        weights_prod = (
            jnp.prod(
                remainder_weights_kernel[
                    jnp.arange(N)[:, None, None],
                    jnp.arange(D)[None, :, None],
                    k_combinations.T[None, :, :],
                ],
                axis=1,
            )
            * weights_dot[:, None]
        )

        grad_hist = grad_hist.at[tuple(expanded_indices.reshape(-1, D).T)].add(
            weights_prod.reshape(-1), wrap_negative_indices=False
        )

        for d in range(D):
            hybrid_weights = jnp.where(
                jnp.arange(D)[None, :, None] == d,
                remainder_weights_dkernel,
                remainder_weights_kernel,
            )
            dweights_prod = (
                jnp.prod(
                    hybrid_weights[
                        jnp.arange(N)[:, None, None],
                        jnp.arange(D)[None, :, None],
                        k_combinations.T[None, :, :],
                    ],
                    axis=1,
                )
                * input_dot[:, d : d + 1]
                * weights[:, None]
            )
            grad_hist = grad_hist.at[tuple(expanded_indices.reshape(-1, D).T)].add(
                dweights_prod.reshape(-1), wrap_negative_indices=False
            )

        return hist, grad_hist

    return jhistdd_fixedrange_


def jtruncgausshist2d_fixedrange_linear(
    bins: np.ndarray, ranges: jax.Array, return_func: bool = True
):
    """
    Gaussian binning kernel with linear recon kernel.
    """

    steps = (ranges[:, 1] - ranges[:, 0]) / bins

    def jtruncgausshist2d_fixedrange_fwd(input: jax.Array, weights: jax.Array):
        """
        input: Array[Nx2]
        bins: Array[D]
        ranges: Array[Dx2]
        weights: Array[N]
        """
        N, D = input.shape
        assert D == 2
        hist = jnp.zeros(bins)
        steps = (ranges[:, 1] - ranges[:, 0]) / bins
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        indices = tuple(indices.astype(int).T)
        remainder /= steps
        remainder -= 0.5

        remainder_weights_kernel = (
            jnp.exp(
                -jnp.stack([remainder + 1, remainder, remainder - 1], axis=2) ** 2 / 2
            )
            / (2 * jnp.pi) ** 0.5
        )

        for k0 in range(3):
            for k1 in range(3):
                hist = hist.at[(indices[0] + k0 - 1, indices[1] + k1 - 1)].add(
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                    * weights,
                    wrap_negative_indices=False,
                )

        return hist

    @jax.custom_jvp
    def jtruncgausshist2d_fixedrange_(input: jax.Array, weights: jax.Array):
        return jtruncgausshist2d_fixedrange_fwd(input, weights)

    @jtruncgausshist2d_fixedrange_.defjvp
    def jtruncgausshist2d_fixedrange_jvp(primals, tangents):
        input, weights = primals
        input_dot, weights_dot = tangents

        # N, D = input.shape
        hist = jtruncgausshist2d_fixedrange_(input, weights)

        grad_hist = jnp.zeros_like(hist)
        # steps = (ranges[:,1]-ranges[:,0])/jnp.array(bins)
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        # tuple(D of Array[N])
        indices = tuple(indices.astype(int).T)

        remainder /= steps
        remainder -= 0.5

        disturb = jnp.stack([remainder + 1, remainder, remainder - 1], axis=2)
        expdisturb = jnp.exp(-(disturb**2) / 2)
        erfdisturbd2 = jax.scipy.special.erf(disturb / 2**0.5) / 2
        erf9d8sd2 = jax.scipy.special.erf(3 / 2**1.5) / 2
        dr1 = erf9d8sd2 + erfdisturbd2[..., 2]
        dr2 = -erf9d8sd2 + erfdisturbd2[..., 1] - 2 * erfdisturbd2[..., 2]
        dr3 = erfdisturbd2[..., 0] + erfdisturbd2[..., 2] - 2 * erfdisturbd2[..., 1]
        dr4 = erf9d8sd2 + erfdisturbd2[..., 1] - 2 * erfdisturbd2[..., 0]
        dr5 = -erf9d8sd2 + erfdisturbd2[..., 0]
        rs2pi = (2 * jnp.pi) ** -0.5
        rexp9d8s2pi = jnp.exp(-9 / 8) * rs2pi
        r1 = (
            -rexp9d8s2pi
            + jnp.exp(-disturb[..., 2] ** 2 / 2) * rs2pi
            + erf9d8sd2 * disturb[..., 2]
            + erfdisturbd2[..., 2] * disturb[..., 2]
        )

        r2 = (
            rexp9d8s2pi
            - 2 * jnp.exp(-disturb[..., 2] ** 2 / 2) * rs2pi
            + jnp.exp(-disturb[..., 1] ** 2 / 2) * rs2pi
            + erf9d8sd2 * (1 - disturb[..., 2])
            - 2 * disturb[..., 2] * erfdisturbd2[..., 2]
            + disturb[..., 1] * erfdisturbd2[..., 1]
        )

        r3 = (
            (expdisturb[..., 0] + expdisturb[..., 2] - 2 * expdisturb[..., 1])
            / (2 * jnp.pi) ** 0.5
            + disturb[..., 0] * erfdisturbd2[..., 0]
            + disturb[..., 2] * erfdisturbd2[..., 2]
            - 2 * disturb[..., 1] * erfdisturbd2[..., 1]
        )

        r4 = (
            rexp9d8s2pi
            - 2 * jnp.exp(-disturb[..., 0] ** 2 / 2) * rs2pi
            + jnp.exp(-disturb[..., 1] ** 2 / 2) * rs2pi
            + erf9d8sd2 * (1 + disturb[..., 0])
            - 2 * disturb[..., 0] * erfdisturbd2[..., 0]
            + disturb[..., 1] * erfdisturbd2[..., 1]
        )

        r5 = (
            -rexp9d8s2pi
            + jnp.exp(-disturb[..., 0] ** 2 / 2) * rs2pi
            - erf9d8sd2 * disturb[..., 0]
            + erfdisturbd2[..., 0] * disturb[..., 0]
        )

        remainder_weights_kernel = jnp.stack(
            [r5, r4, r3, r2, r1],
            axis=2,
        )

        remainder_weights_dkernel = (
            jnp.stack(
                [dr5, dr4, dr3, dr2, dr1],
                axis=2,
            )
            / steps[None, :, None]
        )

        for k0 in range(5):
            for k1 in range(5):
                w = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                )
                wd0 = (
                    remainder_weights_dkernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                    * weights
                )
                wd1 = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_dkernel[:, 1, k1]
                    * weights
                )
                grad_hist = grad_hist.at[
                    (indices[0] + k0 - 2, indices[1] + k1 - 2)
                ].add(
                    wd0 * input_dot[:, 0] + wd1 * input_dot[:, 1] + w * weights_dot,
                    wrap_negative_indices=False,
                )

        return hist, grad_hist

    return (
        jtruncgausshist2d_fixedrange_
        if return_func
        else jtruncgausshist2d_fixedrange_fwd
    )


def jhist2d_fixedrange_lanzcos(
    bins: np.ndarray, ranges: jax.Array, return_func: bool = True
):
    """
    Rect binning kernel with lanzcos recon kernel.
    """

    steps = (ranges[:, 1] - ranges[:, 0]) / bins

    def jhist2d_fixedrange_fwd(input: jax.Array, weights: jax.Array):
        N, D = input.shape
        assert D == 2
        hist = jnp.zeros(bins)
        indices = ((input - ranges[:, 0]) / steps).astype(int)
        return hist.at[tuple(indices.T)].add(weights, wrap_negative_indices=False)

    @jax.custom_jvp
    def jhist2d_fixedrange_(input: jax.Array, weights: jax.Array):
        return jhist2d_fixedrange_fwd(input, weights)

    @jhist2d_fixedrange_.defjvp
    def jhist2d_fixedrange_jvp(primals, tangents):
        input, weights = primals
        input_dot, weights_dot = tangents

        # N, D = input.shape
        hist = jhist2d_fixedrange_(input, weights)

        grad_hist = jnp.zeros_like(hist)
        # steps = (ranges[:,1]-ranges[:,0])/jnp.array(bins)
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        # tuple(D of Array[N])
        indices = tuple(indices.astype(int).T)

        remainder /= steps

        r1 = (
            4 * jnp.sin(jnp.pi * remainder / 2) * jnp.sinc(remainder - 2)
            - jax.scipy.special.sici(jnp.pi)[0]
            + 3 * jax.scipy.special.sici(3 * jnp.pi)[0]
            - jax.scipy.special.sici(jnp.pi * (remainder - 2) / 2)[0]
            + 3 * jax.scipy.special.sici(3 * jnp.pi * (remainder - 2) / 2)[0]
        ) / (2 * jnp.pi)
        r2 = (
            4 * jnp.cos(jnp.pi * remainder / 2) * jnp.sinc(remainder - 1)
            + 2 * jnp.sin(jnp.pi * remainder) * jnp.sinc((remainder - 2) / 2)
            - jax.scipy.special.sici(jnp.pi * (remainder - 1) / 2)[0]
            + 3 * jax.scipy.special.sici(3 * jnp.pi * (remainder - 1) / 2)[0]
            + jax.scipy.special.sici(jnp.pi * (remainder - 2) / 2)[0]
            - 3 * jax.scipy.special.sici(3 * jnp.pi * (remainder - 2) / 2)[0]
        ) / (2 * jnp.pi)
        r3 = (
            -4 * jnp.sin(jnp.pi * remainder / 2) * jnp.sinc(remainder)
            - 2 * jnp.sin(jnp.pi * remainder) * jnp.sinc((remainder - 1) / 2)
            - jax.scipy.special.sici(jnp.pi * remainder / 2)[0]
            + 3 * jax.scipy.special.sici(3 * jnp.pi * remainder / 2)[0]
            + jax.scipy.special.sici(jnp.pi * (remainder - 1) / 2)[0]
            - 3 * jax.scipy.special.sici(3 * jnp.pi * (remainder - 1) / 2)[0]
        ) / (2 * jnp.pi)
        r4 = (
            -4 * jnp.cos(jnp.pi * remainder / 2) * jnp.sinc(remainder + 1)
            + 2 * jnp.sin(jnp.pi * remainder) * jnp.sinc(remainder / 2)
            - jax.scipy.special.sici(jnp.pi * (remainder + 1) / 2)[0]
            + 3 * jax.scipy.special.sici(3 * jnp.pi * (remainder + 1) / 2)[0]
            + jax.scipy.special.sici(jnp.pi * remainder / 2)[0]
            - 3 * jax.scipy.special.sici(3 * jnp.pi * remainder / 2)[0]
        ) / (2 * jnp.pi)
        r5 = (
            -2 * jnp.sin(jnp.pi * remainder) * jnp.sinc((remainder + 1) / 2)
            - jax.scipy.special.sici(jnp.pi)[0]
            + 3 * jax.scipy.special.sici(3 * jnp.pi)[0]
            + jax.scipy.special.sici(jnp.pi * (remainder + 1) / 2)[0]
            - 3 * jax.scipy.special.sici(3 * jnp.pi * (remainder + 1) / 2)[0]
        ) / (2 * jnp.pi)

        # dr1 = -jnp.cos(jnp.pi * remainder / 2) * jnp.sinc(jnp.pi * (remainder - 2)/2)**2
        dr1 = (
            2
            * jnp.cos(jnp.pi * remainder / 2)
            * (jnp.cos(jnp.pi * remainder) - 1)
            / (jnp.pi * (remainder - 2)) ** 2
        )
        dr2 = (
            jnp.sinc((remainder - 1) / 2) * jnp.sinc((remainder - 1))
            + jnp.cos(jnp.pi * remainder / 2) * jnp.sinc((remainder - 2) / 2) ** 2
        )
        dr3 = (
            jnp.sinc(remainder / 2) * jnp.sinc(remainder)
            - jnp.sin(jnp.pi * remainder / 2) * jnp.sinc((remainder - 1) / 2) ** 2
        )
        dr4 = (
            jnp.sinc((remainder + 1) / 2) * jnp.sinc((remainder + 1))
            - jnp.cos(jnp.pi * remainder / 2) * jnp.sinc(remainder / 2) ** 2
        )
        dr5 = (
            2
            * jnp.sin(jnp.pi * remainder / 2)
            * (jnp.cos(jnp.pi * remainder) + 1)
            / (jnp.pi * (remainder + 1)) ** 2
        )

        remainder_weights_kernel = jnp.stack(
            [r5, r4, r3, r2, r1],
            axis=2,
        )

        remainder_weights_dkernel = (
            jnp.stack(
                [dr5, dr4, dr3, dr2, dr1],
                axis=2,
            )
            / steps[None, :, None]
        )

        # (N, D, 3)

        for k0 in range(5):
            for k1 in range(5):
                w = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                )
                wd0 = (
                    remainder_weights_dkernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                    * weights
                )
                wd1 = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_dkernel[:, 1, k1]
                    * weights
                )
                grad_hist = grad_hist.at[
                    (indices[0] + k0 - 2, indices[1] + k1 - 2)
                ].add(
                    wd0 * input_dot[:, 0] + wd1 * input_dot[:, 1] + w * weights_dot,
                    wrap_negative_indices=False,
                )

        return hist, grad_hist

    return jhist2d_fixedrange_ if return_func else jhist2d_fixedrange_fwd


def jhist2d_fixedrange_linear(bins: np.ndarray, ranges: jax.Array, return_func: bool = True):
    """
    Rect binning kernel with linear recon kernel.
    """

    steps = (ranges[:, 1] - ranges[:, 0]) / bins

    def jhist2d_fixedrange_fwd(input: jax.Array, weights: jax.Array):
        N, D = input.shape
        assert D == 2
        hist = jnp.zeros(bins)
        indices = ((input - ranges[:, 0]) / steps).astype(int)
        return hist.at[tuple(indices.T)].add(weights, wrap_negative_indices=False)

    @jax.custom_jvp
    def jhist2d_fixedrange_(input: jax.Array, weights: jax.Array):
        return jhist2d_fixedrange_fwd(input, weights)

    @jhist2d_fixedrange_.defjvp
    def jhist2d_fixedrange_jvp(primals, tangents):
        input, weights = primals
        input_dot, weights_dot = tangents

        # N, D = input.shape
        hist = jhist2d_fixedrange_(input, weights)

        grad_hist = jnp.zeros_like(hist)
        # steps = (ranges[:,1]-ranges[:,0])/jnp.array(bins)
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        # tuple(D of Array[N])
        indices = tuple(indices.astype(int).T)

        remainder /= steps

        # (N, D, 3)
        remainder_weights_kernel = jnp.stack(
            [
                (1 - remainder) ** 2 / 2,
                remainder - remainder**2 + 1 / 2,
                remainder**2 / 2,
            ],
            axis=2,
        )
        remainder_weights_dkernel = (
            jnp.stack([remainder - 1, 1 - 2 * remainder, remainder], axis=2)
            / steps[None, :, None]
        )

        for k0 in range(3):
            for k1 in range(3):
                w = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                )
                wd0 = (
                    remainder_weights_dkernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                    * weights
                )
                wd1 = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_dkernel[:, 1, k1]
                    * weights
                )
                grad_hist = grad_hist.at[
                    (indices[0] + k0 - 1, indices[1] + k1 - 1)
                ].add(
                    wd0 * input_dot[:, 0] + wd1 * input_dot[:, 1] + w * weights_dot,
                    wrap_negative_indices=False,
                )

        return hist, grad_hist

    return jhist2d_fixedrange_ if return_func else jhist2d_fixedrange_fwd


def jbilinearhist2d_fixedrange_linear(
    bins: np.ndarray, ranges: np.ndarray, return_func: bool = True
):
    """
    Linear binning kernel with linear recon kernel.
    """    

    steps = (ranges[:, 1] - ranges[:, 0]) / bins

    def jbilinearhist2d_fixedrange_fwd(input: jax.Array, weights: jax.Array):
        """
        input: Array[Nx2]
        bins: Array[D]
        ranges: Array[Dx2]
        weights: Array[N]
        """
        N, D = input.shape
        assert D == 2
        hist = jnp.zeros(bins)
        steps = (ranges[:, 1] - ranges[:, 0]) / bins
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        indices = tuple(indices.astype(int).T)
        remainder /= steps

        # (N, D, 2)
        remainder_weights_kernel = jnp.stack(
            [1 - remainder, remainder],
            axis=2,
        )

        for k0 in range(2):
            for k1 in range(2):
                w = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                )

                hist = hist.at[indices[0] + k0, indices[1] + k1].add(
                    w * weights,
                    wrap_negative_indices=False,
                )

        return hist

    @jax.custom_jvp
    def jbilinearhist2d_fixedrange_(input: jax.Array, weights: jax.Array):
        return jbilinearhist2d_fixedrange_fwd(input, weights)

    @jbilinearhist2d_fixedrange_.defjvp
    def jbilinearhist2d_fixedrange_jvp(primals, tangents):
        input, weights = primals
        input_dot, weights_dot = tangents

        # N, D = input.shape
        hist = jbilinearhist2d_fixedrange_(input, weights)

        grad_hist = jnp.zeros_like(hist)
        # steps = (ranges[:,1]-ranges[:,0])/jnp.array(bins)
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        # tuple(D of Array[N])
        indices = tuple(indices.astype(int).T)

        remainder /= steps

        # (N, D, 4)
        remainder_weights_kernel = jnp.stack(
            [
                (1 - remainder) ** 3 / 6,
                (3 * remainder**3 - 6 * remainder**2 + 4) / 6,
                (-3 * remainder**3 + 3 * remainder**2 + 3 * remainder + 1) / 6,
                remainder**3 / 6,
            ],
            axis=2,
        )
        remainder_weights_dkernel = (
            jnp.stack(
                [
                    -((1 - remainder) ** 2) / 2,
                    (3 * remainder**2 - 4 * remainder) / 2,
                    (-3 * remainder**2 + 2 * remainder + 1) / 2,
                    remainder**2 / 2,
                ],
                axis=2,
            )
            / steps[None, :, None]
        )

        for k0 in range(4):
            for k1 in range(4):
                w = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                )
                wd0 = (
                    remainder_weights_dkernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                    * weights
                )
                wd1 = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_dkernel[:, 1, k1]
                    * weights
                )
                grad_hist = grad_hist.at[
                    (indices[0] + k0 - 1, indices[1] + k1 - 1)
                ].add(
                    wd0 * input_dot[:, 0] + wd1 * input_dot[:, 1] + w * weights_dot,
                    wrap_negative_indices=False,
                )

        return hist, grad_hist

    return (
        jbilinearhist2d_fixedrange_ if return_func else jbilinearhist2d_fixedrange_fwd
    )


def jhist2d_fixedrange_cubic(bins: np.ndarray, ranges: jax.Array, a=-0.5):
    """
    Rect binning kernel with cubic recon kernel.
    """

    steps = (ranges[:, 1] - ranges[:, 0]) / bins

    def jhist2d_fixedrange_fwd(input: jax.Array, weights: jax.Array):
        N, D = input.shape
        assert D == 2
        hist = jnp.zeros(bins)
        indices = ((input - ranges[:, 0]) / steps).astype(int)
        return hist.at[tuple(indices.T)].add(weights, wrap_negative_indices=False)

    @jax.custom_jvp
    def jhist2d_fixedrange_(input: jax.Array, weights: jax.Array):
        return jhist2d_fixedrange_fwd(input, weights)

    @jhist2d_fixedrange_.defjvp
    def jhist2d_fixedrange_jvp(primals, tangents):
        input, weights = primals
        input_dot, weights_dot = tangents

        # N, D = input.shape
        hist = jhist2d_fixedrange_(input, weights)

        grad_hist = jnp.zeros_like(hist)
        # steps = (ranges[:,1]-ranges[:,0])/jnp.array(bins)
        indices, remainder = jnp.divmod(input - ranges[:, 0], steps)
        # tuple(D of Array[N])
        indices = tuple(indices.astype(int).T)

        remainder /= steps

        y = remainder

        # (N, D, 3)
        remainder_weights_kernel = jnp.stack(
            [
                (-a * (-1 + y) ** 3 * (1 + 3 * y)) / 12,
                (-6 * (-1 + y) ** 3 * (1 + y) + a * (-1 + 6 * y**2 - 4 * y**3)) / 12,
                1 / 2 + y + (-2 + y) * y**3 + a * (-1 + y * (-1 + y) ** 2 * y**2) / 12,
                (a - 6 * a * y**2 + 4 * (3 + a) * y**3 - 6 * y**4) / 12,
                a * (4 - 3 * y) * y**3 / 12,
            ],
            axis=2,
        )
        remainder_weights_dkernel = (
            jnp.stack(
                [
                    -a * (-1 + y) ** 2 * y,
                    -(-1 + y) * (-1 + y * (-1 + a + 2 * y)),
                    1 + y * (a - 3 * (2 + a) * y + 2 * (2 + a) * y**2),
                    y * (a * (-1 + y) + (3 - 2 * y) * y),
                    -a * (-1 + y) * y**2,
                ],
                axis=2,
            )
            / steps[None, :, None]
        )

        for k0 in range(5):
            for k1 in range(5):
                w = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                )
                wd0 = (
                    remainder_weights_dkernel[:, 0, k0]
                    * remainder_weights_kernel[:, 1, k1]
                    * weights
                )
                wd1 = (
                    remainder_weights_kernel[:, 0, k0]
                    * remainder_weights_dkernel[:, 1, k1]
                    * weights
                )
                grad_hist = grad_hist.at[
                    (indices[0] + k0 - 2, indices[1] + k1 - 2)
                ].add(
                    wd0 * input_dot[:, 0] + wd1 * input_dot[:, 1] + w * weights_dot,
                    wrap_negative_indices=False,
                )

        return hist, grad_hist

    return jhist2d_fixedrange_

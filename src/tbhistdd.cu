#include <torch/extension.h>

using namespace at::indexing;

__host__ __device__ __forceinline__ float clamp(float x, const float min,
                                                const float max) {
  return ::min(::max(x, min), max);
}

__global__ void tbhist2d_kernel(
    // input
    const int64_t w0, const int64_t w1, const float l0, const float l1,
    const float rs0, const float rs1,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> input_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> weights_a,
    // output
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> bhist_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> indices_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> remainder_a) {
  auto bn = blockIdx.x * blockDim.x + threadIdx.x;
  if (bn < weights_a.size(0) * weights_a.size(1)) {
    // auto i = bi / weights_a.size(0);
    auto n = bn % weights_a.size(1);
    // auto b = bi % weights_a.size(0);
    auto b = bn / weights_a.size(1);
    auto x0 = (input_a[b][n][0] - l0) * rs0;
    auto x1 = (input_a[b][n][1] - l1) * rs1;
    int32_t i0 = indices_a[b][n][0] = floor(x0);
    int32_t i1 = indices_a[b][n][1] = floor(x1);
    remainder_a[b][n][0] = x0 - i0;
    remainder_a[b][n][1] = x1 - i1;
    atomicAdd(&bhist_a[b][clamp(i0 + 1, 0, w0 + 1)][clamp(i1 + 1, 0, w1 + 1)],
              weights_a[b][n]);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tbhist2d(
    at::Tensor input, at::Tensor bins, at::Tensor ranges, at::Tensor weights) {
  TORCH_CHECK(input.ndimension() == 3, "Input dimension should be 3");
  TORCH_CHECK(bins.ndimension() == 1, "Bins dimension should be 1");
  TORCH_CHECK(ranges.ndimension() == 2, "Ranges dimension should be 2");
  TORCH_CHECK(weights.ndimension() == 2, "Weights dimension should be 2");
  TORCH_CHECK(input.size(0) == weights.size(0), "Batch dimension mismatch");
  TORCH_CHECK(input.size(1) == weights.size(1),
              "Point number dimension mismatch");
  TORCH_CHECK(input.size(2) == 2,
              "The last dimension of input should be 2 for 2d histogram");
  TORCH_CHECK(bins.size(0) == 2,
              "Bins should contain two components representing the dimension");
  TORCH_CHECK(ranges.size(0) == 2,
              "Ranges should contain 2 bounds for 2d histogram");
  TORCH_CHECK(ranges.size(1) == 2,
              "Ranges bound should contain two components as [low,high)");

  auto w0(bins[0].item<int64_t>());
  auto w1(bins[1].item<int64_t>());
  auto l0(ranges[0][0].item<float>());
  auto r0(ranges[0][1].item<float>());
  auto l1(ranges[1][0].item<float>());
  auto r1(ranges[1][1].item<float>());
  auto rs0(w0 / (r0 - l0));
  auto rs1(w1 / (r1 - l1));

  at::Tensor bhist =
      at::zeros({input.size(0), w0 + 2, w1 + 2}, input.options());
  at::Tensor indices = at::empty_like(input);
  at::Tensor remainder = at::empty_like(input);
  at::Tensor rsteps = at::tensor({rs0, rs1}, input.options());

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tbhist2d", [&] {
    tbhist2d_kernel<<<ceil(weights.numel() / 256.), 256>>>(
        w0, w1, l0, l1, rs0, rs1,
        input.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        bhist.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        remainder.packed_accessor32<float, 3, at::RestrictPtrTraits>());
  });

  return {bhist.index({Slice(), Slice(1, -1), Slice(1, -1)}), indices,
          remainder, rsteps};
}

template <size_t D>
__global__ void tbhistdd_kernel(
    // input
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> ranges_a,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> bins_a,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> rsteps_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> input_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> weights_a,
    // output
    at::PackedTensorAccessor32<float, D + 1, at::RestrictPtrTraits> bhist_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> indices_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> remainder_a) {
  auto bn = blockIdx.x * blockDim.x + threadIdx.x;
  if (bn < weights_a.size(0) * weights_a.size(1)) {
    // auto i = bi / weights_a.size(0);
    auto n = bn % weights_a.size(1);
    // auto b = bi % weights_a.size(0);
    auto b = bn / weights_a.size(1);
    uint32_t offset = b * bhist_a.stride(0);
    for (int i = 0; i < D; i++) {
      auto xi = (input_a[b][n][i] - ranges_a[i][0]) * rsteps_a[i];
      int32_t ii = indices_a[b][n][i] = floor(xi);
      remainder_a[b][n][i] = xi - ii;
      offset += bhist_a.stride(i + 1) * clamp(ii + 1, 0, bins_a[i] + 1);
    }
    atomicAdd(bhist_a.data() + offset, weights_a[b][n]);
  }
}

template <size_t D>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tbhistdd(
    at::Tensor input, at::Tensor bins, at::Tensor ranges, at::Tensor weights) {

  TORCH_CHECK(input.ndimension() == 3, "Input dimension should be 3");
  TORCH_CHECK(bins.ndimension() == 1, "Bins dimension should be 1");
  TORCH_CHECK(ranges.ndimension() == 2, "Ranges dimension should be 2");
  TORCH_CHECK(weights.ndimension() == 2, "Weights dimension should be 2");
  TORCH_CHECK(input.size(0) == weights.size(0), "Batch dimension mismatch");
  TORCH_CHECK(input.size(1) == weights.size(1),
              "Point number dimension mismatch");
  TORCH_CHECK(bins.size(0) == ranges.size(0),
              "Bins should contain the same number of components representing "
              "the dimensions");
  TORCH_CHECK(ranges.size(1) == 2,
              "Ranges bound should contain two components as [low,high)");
  TORCH_CHECK(bins.size(0) == D, "Bins number should be D as instantiated");
  std::vector<int64_t> dims;
  dims.emplace_back(input.size(0));
  for (int i = 0; i < D; i++)
    dims.emplace_back(bins[i].item<int64_t>() + 2);

  at::Tensor bhist = at::zeros(dims, input.options());

  at::Tensor rsteps =
      bins / (ranges.index({Slice(), 1}) - ranges.index({Slice(), 0}));

  at::Tensor indices = at::empty_like(input);
  at::Tensor remainder = at::empty_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tbhistdd", [&] {
    tbhistdd_kernel<D><<<ceil(weights.numel() / 256.), 256>>>(
        ranges.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        bins.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
        rsteps.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        input.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        bhist.packed_accessor32<float, D + 1, at::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        remainder.packed_accessor32<float, 3, at::RestrictPtrTraits>());
  });

  std::vector<TensorIndex> slices;
  slices.emplace_back(Slice());
  for (int i = 0; i < D; i++)
    slices.emplace_back(Slice(1, -1));

  return {bhist.index(slices), indices, remainder, rsteps};
  // return {bhist, indices, remainder, rsteps};
}

__global__ void tbbilinearhist2d_kernel(
    // input
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> ranges_a,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> bins_a,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> rsteps_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> input_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> weights_a,
    // output
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> bhist_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> indices_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> remainder_a) {
  auto bn = blockIdx.x * blockDim.x + threadIdx.x;
  if (bn < weights_a.size(0) * weights_a.size(1)) {
    // auto i = bi / weights_a.size(0);
    auto n = bn % weights_a.size(1);
    // auto b = bi % weights_a.size(0);
    auto b = bn / weights_a.size(1);
    float remainder_weights_kernel[2][2];  // [D][2]
    int32_t idx[2];
    for (int i = 0; i < 2; i++) {
      auto xi = (input_a[b][n][i] - ranges_a[i][0]) * rsteps_a[i];
      idx[i] = indices_a[b][n][i] = floor(xi);
      const float rw = xi - idx[i];
      remainder_a[b][n][i] = rw;
      remainder_weights_kernel[i][0] = 1 - rw;
      remainder_weights_kernel[i][1] = rw;
    }
    for (int k0 = 0; k0 < 2; k0++) {
      for (int k1 = 0; k1 < 2; k1++) {
        atomicAdd(&bhist_a[b][clamp(idx[0] + 1 + k0, 0, bins_a[0] + 1)]
                          [clamp(idx[1] + 1 + k1, 0, bins_a[1] + 1)],
                  remainder_weights_kernel[0][k0] *
                      remainder_weights_kernel[1][k1] * weights_a[b][n]);
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tbbilinearhist2d(
    at::Tensor input, at::Tensor bins, at::Tensor ranges, at::Tensor weights) {

  TORCH_CHECK(input.ndimension() == 3, "Input dimension should be 3");
  TORCH_CHECK(bins.ndimension() == 1, "Bins dimension should be 1");
  TORCH_CHECK(ranges.ndimension() == 2, "Ranges dimension should be 2");
  TORCH_CHECK(weights.ndimension() == 2, "Weights dimension should be 2");
  TORCH_CHECK(input.size(0) == weights.size(0), "Batch dimension mismatch");
  TORCH_CHECK(input.size(1) == weights.size(1),
              "Point number dimension mismatch");
  TORCH_CHECK(bins.size(0) == ranges.size(0),
              "Bins should contain the same number of components representing "
              "the dimensions");
  TORCH_CHECK(ranges.size(1) == 2,
              "Ranges bound should contain two components as [low,high)");
  TORCH_CHECK(bins.size(0) == 2, "Bins number should be 2 as instantiated");
  std::vector<int64_t> dims;
  dims.emplace_back(input.size(0));
  for (int i = 0; i < 2; i++)
    dims.emplace_back(bins[i].item<int64_t>() + 2);

  at::Tensor bhist = at::zeros(dims, input.options());

  at::Tensor rsteps =
      bins / (ranges.index({Slice(), 1}) - ranges.index({Slice(), 0}));

  at::Tensor indices = at::empty_like(input);
  at::Tensor remainder = at::empty_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tbbilinearhist2d", [&] {
    tbbilinearhist2d_kernel<<<ceil(weights.numel() / 256.), 256>>>(
        ranges.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        bins.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
        rsteps.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        input.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        bhist.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        remainder.packed_accessor32<float, 3, at::RestrictPtrTraits>());
  });

  std::vector<TensorIndex> slices;
  slices.emplace_back(Slice());
  for (int i = 0; i < 2; i++)
    slices.emplace_back(Slice(1, -1));

  return {bhist.index(slices), indices, remainder, rsteps};
}

__global__ void tbbilinearhist2d_bwd_kernel(
    // input
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits>
        bgrad_hist_padded_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> indices_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> remainder_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> weights_a,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> rsteps_a,
    // output
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> gradx_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> gradw_a) {
  auto bn = blockIdx.x * blockDim.x + threadIdx.x;
  if (bn < weights_a.size(0) * weights_a.size(1)) {
    // auto i = bi / weights_a.size(0);
    const auto n = bn % weights_a.size(1);
    // auto b = bi % weights_a.size(0);
    const auto b = bn / weights_a.size(1);

    float remainder_weights_kernel[2][4];
    float remainder_weights_dkernel[2][4];

    for (int i = 0; i < 2; i++) {
      const auto rw = remainder_a[b][n][i];
      remainder_weights_kernel[i][0] = (1 - rw) * (1 - rw) * (1 - rw) / 6;
      remainder_weights_kernel[i][1] = (3 * rw * rw * rw - 6 * rw * rw + 4) / 6;
      remainder_weights_kernel[i][2] =
          (-3 * rw * rw * rw + 3 * rw * rw + 3 * rw + 1) / 6;
      remainder_weights_kernel[i][3] = rw * rw * rw / 6;
      remainder_weights_dkernel[i][0] = rsteps_a[i] * -(1 - rw) * (1 - rw) / 2;
      remainder_weights_dkernel[i][1] =
          rsteps_a[i] * (3 * rw * rw - 4 * rw) / 2;
      remainder_weights_dkernel[i][2] =
          rsteps_a[i] * (-3 * rw * rw + 2 * rw + 1) / 2;
      remainder_weights_dkernel[i][3] = rsteps_a[i] * rw * rw / 2;
    }

    float gradw_weight = 0;
    float gradx_weight[2]{};

    for (int k0 = 0; k0 < 4; k0++) {
      for (int k1 = 0; k1 < 4; k1++) {
        const auto grad_val =
            bgrad_hist_padded_a[b][clamp(indices_a[b][n][0] + k0, 0,
                                         bgrad_hist_padded_a.size(1) - 1)]
                               [clamp(indices_a[b][n][1] + k1, 0,
                                      bgrad_hist_padded_a.size(2) - 1)];
        const float w =
            remainder_weights_kernel[0][k0] * remainder_weights_kernel[1][k1];
        const float wd0 = remainder_weights_dkernel[0][k0] *
                          remainder_weights_kernel[1][k1] * weights_a[b][n];
        const float wd1 = remainder_weights_kernel[0][k0] *
                          remainder_weights_dkernel[1][k1] * weights_a[b][n];
        gradw_weight += w * grad_val;
        gradx_weight[0] += wd0 * grad_val;
        gradx_weight[1] += wd1 * grad_val;
      }
    }
    atomicAdd(&gradw_a[b][n], gradw_weight);
    atomicAdd(&gradx_a[b][n][0], gradx_weight[0]);
    atomicAdd(&gradx_a[b][n][1], gradx_weight[1]);
  }
}

std::pair<at::Tensor, at::Tensor> tbbilinearhist2d_bwd(at::Tensor bgrad_hist,
                                                       at::Tensor indices,
                                                       at::Tensor remainder,
                                                       at::Tensor weights,
                                                       at::Tensor rsteps) {

  TORCH_CHECK(bgrad_hist.ndimension() == 3, "Grad dimension should be ", 3);
  TORCH_CHECK(weights.ndimension() == 2, "Weights dimension should be 2");
  TORCH_CHECK(indices.ndimension() == 3, "Indices dimension should be 3");
  TORCH_CHECK(indices.size(2) == 2, "Indices should have ", 2, " components");
  TORCH_CHECK(rsteps.ndimension() == 1 && rsteps.size(0) == 2,
              "Steps should be a 1d tensor with D elements");
  TORCH_CHECK(indices.sizes() == remainder.sizes(),
              "Sizes of indices and remainder should be the same");
  TORCH_CHECK(bgrad_hist.size(0) == weights.size(0) &&
                  weights.size(0) == indices.size(0),
              "Batch dimension mismatch");
  TORCH_CHECK(indices.size(1) == weights.size(1),
              "Point number dimension mismatch");

  std::vector<int64_t> pad(2 * 2, 1);

  at::Tensor bgrad_hist_padded = at::pad(bgrad_hist, pad);
  at::Tensor gradw = at::zeros_like(weights);
  at::Tensor gradx = at::zeros_like(remainder);

  AT_DISPATCH_FLOATING_TYPES(
      bgrad_hist.scalar_type(), "tbbilinearhistdd bwd", [&] {
        tbbilinearhist2d_bwd_kernel<<<ceil(weights.numel() / 256.), 256>>>(
            bgrad_hist_padded
                .packed_accessor32<float, 3, at::RestrictPtrTraits>(),
            indices.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
            remainder.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
            weights.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            rsteps.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
            gradx.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
            gradw.packed_accessor32<float, 2, at::RestrictPtrTraits>());
      });

  return {gradx, gradw};
}

__global__ void tbhist2d_bwd_kernel(
    // input
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits>
        bgrad_hist_padded_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> indices_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> remainder_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> weights_a,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> rsteps_a,
    // output
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> gradx_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> gradw_a) {
  auto bn = blockIdx.x * blockDim.x + threadIdx.x;
  if (bn < weights_a.size(0) * weights_a.size(1)) {
    // auto i = bi / weights_a.size(0);
    auto n = bn % weights_a.size(1);
    // auto b = bi % weights_a.size(0);
    auto b = bn / weights_a.size(1);
    const auto rw0 = remainder_a[b][n][0];
    const auto rw1 = remainder_a[b][n][1];
    const float remainder_weights_kernel[3][2]{(1 - rw0) * (1 - rw0) / 2,
                                               (1 - rw1) * (1 - rw1) / 2,
                                               rw0 - rw0 * rw0 + 0.5f,
                                               rw1 - rw1 * rw1 + 0.5f,
                                               rw0 * rw0 / 2,
                                               rw1 * rw1 / 2};
    const float remainder_weights_dkernel[3][2]{
        (rw0 - 1) * rsteps_a[0],     (rw1 - 1) * rsteps_a[1],
        (1 - 2 * rw0) * rsteps_a[0], (1 - 2 * rw1) * rsteps_a[1],
        rw0 * rsteps_a[0],           rw1 * rsteps_a[1]};

    float gradw_weight = 0;
    float gradx_weight = 0;
    float grady_weight = 0;

    const uint32_t x0 =
        clamp(indices_a[b][n][0] + 3, 1, bgrad_hist_padded_a.size(1) - 2);
    const uint32_t x1 =
        clamp(indices_a[b][n][1] + 3, 1, bgrad_hist_padded_a.size(2) - 2);

    for (int k0 = 0; k0 < 3; k0++) {
      for (int k1 = 0; k1 < 3; k1++) {
        const auto grad_val = bgrad_hist_padded_a[b][x0 + k0 - 1][x1 + k1 - 1];
        const float w =
            remainder_weights_kernel[k0][0] * remainder_weights_kernel[k1][1];
        const float wd0 = remainder_weights_dkernel[k0][0] *
                          remainder_weights_kernel[k1][1] * weights_a[b][n];
        const float wd1 = remainder_weights_kernel[k0][0] *
                          remainder_weights_dkernel[k1][1] * weights_a[b][n];
        gradw_weight += w * grad_val;
        gradx_weight += wd0 * grad_val;
        grady_weight += wd1 * grad_val;
      }
    }
    atomicAdd(&gradw_a[b][n], gradw_weight);
    atomicAdd(&gradx_a[b][n][0], gradx_weight);
    atomicAdd(&gradx_a[b][n][1], grady_weight);
  }
}

std::pair<at::Tensor, at::Tensor> tbhist2d_bwd(at::Tensor bgrad_hist,
                                               at::Tensor indices,
                                               at::Tensor remainder,
                                               at::Tensor weights,
                                               at::Tensor rsteps) {

  TORCH_CHECK(bgrad_hist.ndimension() == 3, "Grad dimension should be 3");
  TORCH_CHECK(weights.ndimension() == 2, "Weights dimension should be 2");
  TORCH_CHECK(indices.ndimension() == 3, "Indices dimension should be 3");
  TORCH_CHECK(rsteps.ndimension() == 1 && rsteps.size(0) == 2,
              "Steps should be a 1d tensor with 2 elements");
  TORCH_CHECK(indices.sizes() == remainder.sizes(),
              "Sizes of indices and remainder should be the same");
  TORCH_CHECK(indices.size(2) == 2, "Indices should have 2 components");
  TORCH_CHECK(bgrad_hist.size(0) == weights.size(0) &&
                  weights.size(0) == indices.size(0),
              "Batch dimension mismatch");
  TORCH_CHECK(indices.size(1) == weights.size(1),
              "Point number dimension mismatch");

  at::Tensor bgrad_hist_padded = at::pad(bgrad_hist, {3, 3, 3, 3});
  at::Tensor gradw = at::zeros_like(weights);
  at::Tensor gradx = at::zeros_like(remainder);

  AT_DISPATCH_FLOATING_TYPES(bgrad_hist.scalar_type(), "bhist2d bwd", [&] {
    tbhist2d_bwd_kernel<<<ceil(weights.numel() / 256.), 256>>>(
        bgrad_hist_padded.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        remainder.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        rsteps.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        gradx.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        gradw.packed_accessor32<float, 2, at::RestrictPtrTraits>());
  });

  return {gradx, gradw};
}

template <size_t D>
__global__ void tbhistdd_bwd_kernel(
    // input
    at::PackedTensorAccessor32<float, D + 1, at::RestrictPtrTraits>
        bgrad_hist_padded_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> indices_a,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> remainder_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> weights_a,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> rsteps_a,
    // output
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> gradx_a,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> gradw_a) {
  auto bn = blockIdx.x * blockDim.x + threadIdx.x;
  if (bn < weights_a.size(0) * weights_a.size(1)) {
    // auto i = bi / weights_a.size(0);
    const auto n = bn % weights_a.size(1);
    // auto b = bi % weights_a.size(0);
    const auto b = bn / weights_a.size(1);

    float remainder_weights_kernel[3][D];
    float remainder_weights_dkernel[3][D];

    int total_combinations = 1;
    for (int i = 0; i < D; i++) {
      const auto rw = remainder_a[b][n][i];
      remainder_weights_kernel[0][i] = (1 - rw) * (1 - rw) / 2;
      remainder_weights_kernel[1][i] = rw - rw * rw + 0.5f;
      remainder_weights_kernel[2][i] = rw * rw / 2;
      remainder_weights_dkernel[0][i] = (rw - 1) * rsteps_a[i];
      remainder_weights_dkernel[1][i] = (1 - 2 * rw) * rsteps_a[i];
      remainder_weights_dkernel[2][i] = rw * rsteps_a[i];
      total_combinations *= 3;
    }

    float gradw_weight = 0;
    float gradx_weight[D]{};

    uint32_t base_offset = b * bgrad_hist_padded_a.stride(0);

    for (int tc = 0; tc < total_combinations; tc++) {
      int offset = base_offset;
      int idx = tc;
      int ki[D];
      float w = 1;
      float wd[D];
      for (int i = 0; i < D; i++) {
        ki[i] = idx % 3;
        idx /= 3;
        wd[i] = weights_a[b][n];
        const uint32_t x = clamp(indices_a[b][n][i] + 3, 1,
                                 bgrad_hist_padded_a.size(i + 1) - 2);
        offset += (x + ki[i] - 1) * bgrad_hist_padded_a.stride(i + 1);
      }

      const auto grad_val = bgrad_hist_padded_a.data()[offset];

      for (int i = 0; i < D; i++) {
        w *= remainder_weights_kernel[ki[i]][i];
        for (int j = 0; j < D; j++) {
          if (j == i) {
            wd[i] *= remainder_weights_dkernel[ki[j]][j];
          } else {
            wd[i] *= remainder_weights_kernel[ki[j]][j];
          }
        }
      }
      gradw_weight += w * grad_val;
      for (int i = 0; i < D; i++) {
        gradx_weight[i] += wd[i] * grad_val;
      }
    }

    atomicAdd(&gradw_a[b][n], gradw_weight);
    for (int i = 0; i < D; i++) {
      atomicAdd(&gradx_a[b][n][i], gradx_weight[i]);
    }
  }
}

template <size_t D>
std::pair<at::Tensor, at::Tensor> tbhistdd_bwd(at::Tensor bgrad_hist,
                                               at::Tensor indices,
                                               at::Tensor remainder,
                                               at::Tensor weights,
                                               at::Tensor rsteps) {

  TORCH_CHECK(bgrad_hist.ndimension() == D + 1, "Grad dimension should be ",
              D + 1);
  TORCH_CHECK(weights.ndimension() == 2, "Weights dimension should be 2");
  TORCH_CHECK(indices.ndimension() == 3, "Indices dimension should be 3");
  TORCH_CHECK(indices.size(2) == D, "Indices should have ", D, " components");
  TORCH_CHECK(rsteps.ndimension() == 1 && rsteps.size(0) == D,
              "Steps should be a 1d tensor with D elements");
  TORCH_CHECK(indices.sizes() == remainder.sizes(),
              "Sizes of indices and remainder should be the same");
  TORCH_CHECK(bgrad_hist.size(0) == weights.size(0) &&
                  weights.size(0) == indices.size(0),
              "Batch dimension mismatch");
  TORCH_CHECK(indices.size(1) == weights.size(1),
              "Point number dimension mismatch");

  std::vector<int64_t> pad(2 * D, 3);

  at::Tensor bgrad_hist_padded = at::pad(bgrad_hist, pad);
  at::Tensor gradw = at::zeros_like(weights);
  at::Tensor gradx = at::zeros_like(remainder);

  AT_DISPATCH_FLOATING_TYPES(bgrad_hist.scalar_type(), "bhistdd bwd", [&] {
    tbhistdd_bwd_kernel<D><<<ceil(weights.numel() / 256.), 256>>>(
        bgrad_hist_padded
            .packed_accessor32<float, D + 1, at::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        remainder.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        rsteps.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
        gradx.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
        gradw.packed_accessor32<float, 2, at::RestrictPtrTraits>());
  });

  return {gradx, gradw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tbbilinearhist2d", &tbbilinearhist2d);
  m.def("tbhist2d", &tbhistdd<2>);
  m.def("tbhist3d", &tbhistdd<3>);
  m.def("tbbilinearhist2d_bwd", &tbbilinearhist2d_bwd);
  m.def("tbhist2d_bwd", &tbhistdd_bwd<2>);
  m.def("tbhist3d_bwd", &tbhistdd_bwd<3>);
}

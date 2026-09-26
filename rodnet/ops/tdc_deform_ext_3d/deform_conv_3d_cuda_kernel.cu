#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <iostream>
using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#ifndef DEBUG_INFO                      //normal mode
const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
#else                                   //debug mode
const int CUDA_NUM_THREADS = 16;
const int kMaxGridNum = 1;
#endif

inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_trilinear(
    const scalar_t *bottom_data,
    const int data_time,
    const int data_height,
    const int data_width,
    const int time,
    const int height,
    const int width,
    scalar_t t,
    scalar_t h,
    scalar_t w)
{
  if (t <= -1 || h <= -1 || w <= -1 || t >= time || h >= height || w >= width)
  {
    return static_cast<scalar_t>(0);
  }
  int t_low = floor(t);
  int h_low = floor(h);
  int w_low = floor(w);
  int t_high = t_low + 1;
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  scalar_t lt = t - t_low;
  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t ht = 1 - lt;
  scalar_t hh = 1 - lh;
  scalar_t hw = 1 - lw;
  scalar_t val = static_cast<scalar_t>(0);
  for (int dz = 0; dz <= 1; ++dz)
  {
    int tt = dz == 0 ? t_low : t_high;
    scalar_t wt = dz == 0 ? ht : lt;
    if (tt < 0 || tt >= time)
      continue;
    for (int dy = 0; dy <= 1; ++dy)
    {
      int yy = dy == 0 ? h_low : h_high;
      scalar_t wh = dy == 0 ? hh : lh;
      if (yy < 0 || yy >= height)
        continue;
      for (int dx = 0; dx <= 1; ++dx)
      {
        int xx = dx == 0 ? w_low : w_high;
        scalar_t ww = dx == 0 ? hw : lw;
        if (xx < 0 || xx >= width)
          continue;
        scalar_t v = bottom_data[tt * data_height * data_width + yy * data_width + xx];
        val += wt * wh * ww * v;
      }
    }
  }
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight_3d(
    scalar_t argmax_t,
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int t,
    const int h,
    const int w,
    const int time,
    const int height,
    const int width)
{
  if (argmax_t <= -1 || argmax_t >= time ||
      argmax_h <= -1 || argmax_h >= height ||
      argmax_w <= -1 || argmax_w >= width)
  {
    return static_cast<scalar_t>(0);
  }
  int argmax_t_low = floor(argmax_t);
  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_t_high = argmax_t_low + 1;
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  scalar_t weight_t = static_cast<scalar_t>(0);
  scalar_t weight_h = static_cast<scalar_t>(0);
  scalar_t weight_w = static_cast<scalar_t>(0);
  if (t == argmax_t_low)
    weight_t = argmax_t_high - argmax_t;
  else if (t == argmax_t_high)
    weight_t = argmax_t - argmax_t_low;
  if (h == argmax_h_low)
    weight_h = argmax_h_high - argmax_h;
  else if (h == argmax_h_high)
    weight_h = argmax_h - argmax_h_low;
  if (w == argmax_w_low)
    weight_w = argmax_w_high - argmax_w;
  else if (w == argmax_w_high)
    weight_w = argmax_w - argmax_w_low;
  return weight_t * weight_h * weight_w;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight_3d(
    scalar_t argmax_t,
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int time,
    const int height,
    const int width,
    const scalar_t *im_data,
    const int data_height,
    const int data_width,
    const int bp_dir)
{
  if (argmax_t <= -1 || argmax_t >= time ||
      argmax_h <= -1 || argmax_h >= height ||
      argmax_w <= -1 || argmax_w >= width)
  {
    return static_cast<scalar_t>(0);
  }
  int t_low = floor(argmax_t);
  int h_low = floor(argmax_h);
  int w_low = floor(argmax_w);
  int t_high = t_low + 1;
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  scalar_t lt = argmax_t - t_low;
  scalar_t lh = argmax_h - h_low;
  scalar_t lw = argmax_w - w_low;
  scalar_t ht = 1 - lt;
  scalar_t hh = 1 - lh;
  scalar_t hw = 1 - lw;
  scalar_t weight = static_cast<scalar_t>(0);
  for (int dz = 0; dz <= 1; ++dz)
  {
    int tt = (dz == 0) ? t_low : t_high;
    scalar_t wt = (dz == 0) ? ht : lt;
    scalar_t dwt = (dz == 0) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(1);
    if (tt < 0 || tt >= time)
      continue;
    for (int dy = 0; dy <= 1; ++dy)
    {
      int yy = (dy == 0) ? h_low : h_high;
      scalar_t wh = (dy == 0) ? hh : lh;
      scalar_t dwh = (dy == 0) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(1);
      if (yy < 0 || yy >= height)
        continue;
      for (int dx = 0; dx <= 1; ++dx)
      {
        int xx = (dx == 0) ? w_low : w_high;
        scalar_t ww = (dx == 0) ? hw : lw;
        scalar_t dww = (dx == 0) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(1);
        if (xx < 0 || xx >= width)
          continue;
        scalar_t v = im_data[tt * data_height * data_width + yy * data_width + xx];
        if (bp_dir == 0)
        {
          weight += dwt * wh * ww * v;
        }
        else if (bp_dir == 1)
        {
          weight += wt * dwh * ww * v;
        }
        else if (bp_dir == 2)
        {
          weight += wt * wh * dww * v;
        }
      }
    }
  }
  return weight;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel_3d(
    const int n,
    const scalar_t *data_im,
    const scalar_t *data_offset,
    const int time,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int num_channels,
    const int deformable_group,
    const int time_col,
    const int height_col,
    const int width_col,
    scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int t_col = (index / width_col / height_col) % time_col;
    const int b_col = (index / width_col / height_col / time_col) % batch_size;
    const int c_im = (index / width_col / height_col / time_col) / batch_size;
    const int c_col = c_im * kernel_t * kernel_h * kernel_w;
    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int t_in = t_col * stride_t - pad_t;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    scalar_t *data_col_ptr = data_col + ((((c_col * batch_size + b_col) * time_col + t_col) * height_col + h_col) * width_col + w_col);
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * time * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 3 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;
    for (int k = 0; k < kernel_t; ++k)
    {
      for (int i = 0; i < kernel_h; ++i)
      {
        for (int j = 0; j < kernel_w; ++j)
        {
          const int kernel_idx = (k * kernel_h + i) * kernel_w + j;
          const int data_offset_t_ptr = (((3 * kernel_idx) * time_col + t_col) * height_col + h_col) * width_col + w_col;
          const int data_offset_h_ptr = (((3 * kernel_idx + 1) * time_col + t_col) * height_col + h_col) * width_col + w_col;
          const int data_offset_w_ptr = (((3 * kernel_idx + 2) * time_col + t_col) * height_col + h_col) * width_col + w_col;
          const scalar_t offset_t = data_offset_ptr[data_offset_t_ptr];
          const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
          const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
          const scalar_t t_im = static_cast<scalar_t>(t_in + k * dilation_t) + offset_t;
          const scalar_t h_im = static_cast<scalar_t>(h_in + i * dilation_h) + offset_h;
          const scalar_t w_im = static_cast<scalar_t>(w_in + j * dilation_w) + offset_w;
          scalar_t val = static_cast<scalar_t>(0);
          if (t_im > -1 && h_im > -1 && w_im > -1 &&
              t_im < time && h_im < height && w_im < width)
          {
            val = deformable_im2col_trilinear(
                data_im_ptr,
                time,
                height,
                width,
                time,
                height,
                width,
                t_im,
                h_im,
                w_im);
          }
          *data_col_ptr = val;
          data_col_ptr += batch_size * time_col * height_col * width_col;
        }
      }
    }
  }
}

void deformable_im2col_3d(
    const at::Tensor data_im, const at::Tensor data_offset, const int channels,
    const int time, const int height, const int width,
    const int ksize_t, const int ksize_h, const int ksize_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, at::Tensor data_col)
{
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int time_col = (time + 2 * pad_t - (dilation_t * (ksize_t - 1) + 1)) / stride_t + 1;
  int num_kernels = channels * height_col * width_col * time_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;
  const int expected_offset_channels = deformable_group * 3 * ksize_t * ksize_h * ksize_w;
  TORCH_CHECK(data_offset.dim() == 5,
              "data_offset must be 5D [B, C_offset, T_out, H_out, W_out], got dim=",
              data_offset.dim());

  TORCH_CHECK(data_offset.size(1) == expected_offset_channels,
              "offset channel mismatch: got ", data_offset.size(1),
              ", expected ", expected_offset_channels,
              " = deformable_group * 3 * ksize_t * ksize_h * ksize_w");

  TORCH_CHECK(data_offset.size(2) == time_col,
              "offset time mismatch: got ", data_offset.size(2),
              ", expected ", time_col);

  TORCH_CHECK(data_offset.size(3) == height_col,
              "offset height mismatch: got ", data_offset.size(3),
              ", expected ", height_col);

  TORCH_CHECK(data_offset.size(4) == width_col,
              "offset width mismatch: got ", data_offset.size(4),
              ", expected ", width_col);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        deformable_im2col_gpu_kernel_3d<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, data_offset_, time, height, width,
            ksize_t, ksize_h, ksize_w,
            pad_t, pad_h, pad_w,
            stride_t, stride_h, stride_w,
            dilation_t, dilation_h, dilation_w,
            channel_per_deformable_group,
            parallel_imgs,
            channels,
            deformable_group,
            time_col,
            height_col,
            width_col,
            data_col_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col_3d: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel_3d(
    const int n,
    const scalar_t *data_col,
    const scalar_t *data_offset,
    const int channels,
    const int time,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int deformable_group,
    const int time_col,
    const int height_col,
    const int width_col,
    scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / time_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / time_col / batch_size / kernel_w) % kernel_h;
    const int k = (index / width_col / height_col / time_col / batch_size / kernel_w / kernel_h) % kernel_t;
    const int c = index / width_col / height_col / time_col / batch_size / kernel_w / kernel_h / kernel_t;
    const int deformable_group_index = c / channel_per_deformable_group;
    const int w_out = index % width_col;
    const int h_out = (index / width_col) % height_col;
    const int t_out = (index / width_col / height_col) % time_col;
    const int b = (index / width_col / height_col / time_col) % batch_size;
    const int w_in = w_out * stride_w - pad_w;
    const int h_in = h_out * stride_h - pad_h;
    const int t_in = t_out * stride_t - pad_t;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 3 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;
    const int kernel_idx = (k * kernel_h + i) * kernel_w + j;
    const int data_offset_t_ptr = (((3 * kernel_idx) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_h_ptr = (((3 * kernel_idx + 1) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((3 * kernel_idx + 2) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_t = data_offset_ptr[data_offset_t_ptr];
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_t_data = static_cast<scalar_t>(t_in + k * dilation_t) + offset_t;
    const scalar_t cur_inv_h_data = static_cast<scalar_t>(h_in + i * dilation_h) + offset_h;
    const scalar_t cur_inv_w_data = static_cast<scalar_t>(w_in + j * dilation_w) + offset_w;
    const scalar_t cur_top_grad = data_col[index];
    const int cur_t = floor(cur_inv_t_data);
    const int cur_h = floor(cur_inv_h_data);
    const int cur_w = floor(cur_inv_w_data);
    for (int dz = 0; dz <= 1; ++dz)
    {
      for (int dy = 0; dy <= 1; ++dy)
      {
        for (int dx = 0; dx <= 1; ++dx)
        {
          const int tt = cur_t + dz;
          const int yy = cur_h + dy;
          const int xx = cur_w + dx;
          if (tt >= 0 && tt < time &&
              yy >= 0 && yy < height &&
              xx >= 0 && xx < width)
          {
            const int cur_bottom_grad_pos =
                (((b * channels + c) * time + tt) * height + yy) * width + xx;
            scalar_t weight = get_gradient_weight_3d(
                cur_inv_t_data,
                cur_inv_h_data,
                cur_inv_w_data,
                tt,
                yy,
                xx,
                time,
                height,
                width);
            gpuAtomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
          }
        }
      }
    }
  }
}

void deformable_col2im_3d(
    const at::Tensor data_col, const at::Tensor data_offset, const int channels,
    const int time, const int height, const int width,
    const int ksize_t, const int ksize_h, const int ksize_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group,
    at::Tensor grad_im)
{
  int time_col = (time + 2 * pad_t - (dilation_t * (ksize_t - 1) + 1)) / stride_t + 1;
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_t * ksize_h * ksize_w * time_col * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();
        deformable_col2im_gpu_kernel_3d<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_kernels,
            data_col_,
            data_offset_,
            channels,
            time,
            height,
            width,
            ksize_t,
            ksize_h,
            ksize_w,
            pad_t,
            pad_h,
            pad_w,
            stride_t,
            stride_h,
            stride_w,
            dilation_t,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            parallel_imgs,
            deformable_group,
            time_col,
            height_col,
            width_col,
            grad_im_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_3d: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel_3d(
    const int n,
    const scalar_t *data_col,
    const scalar_t *data_im,
    const scalar_t *data_offset,
    const int channels,
    const int time,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int offset_channels,
    const int deformable_group,
    const int time_col,
    const int height_col,
    const int width_col,
    scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = static_cast<scalar_t>(0);
    const int w = index % width_col;
    const int h = (index / width_col) % height_col;
    const int t = (index / width_col / height_col) % time_col;
    const int c = (index / width_col / height_col / time_col) % offset_channels;
    const int b = (index / width_col / height_col / time_col) / offset_channels;
    const int deformable_group_index = c / (3 * kernel_t * kernel_h * kernel_w);
    const int col_step = kernel_t * kernel_h * kernel_w;
    const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col * time_col;
    const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_t / kernel_h / kernel_w * time * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 3 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;
    const int offset_c = c - deformable_group_index * 3 * kernel_t * kernel_h * kernel_w;
    for (int col_c = (offset_c / 3); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * time_col + t) * height_col + h) * width_col + w;
      const int bp_dir = offset_c % 3;
      const int j = (col_pos / width_col / height_col / time_col / batch_size) % kernel_w;
      const int i = (col_pos / width_col / height_col / time_col / batch_size / kernel_w) % kernel_h;
      const int k = (col_pos / width_col / height_col / time_col / batch_size / kernel_w / kernel_h) % kernel_t;
      const int w_out = col_pos % width_col;
      const int h_out = (col_pos / width_col) % height_col;
      const int t_out = (col_pos / width_col / height_col) % time_col;
      const int w_in = w_out * stride_w - pad_w;
      const int h_in = h_out * stride_h - pad_h;
      const int t_in = t_out * stride_t - pad_t;
      const int kernel_idx = (k * kernel_h + i) * kernel_w + j;
      const int data_offset_t_ptr = (((3 * kernel_idx) * time_col + t_out) * height_col + h_out) * width_col + w_out;
      const int data_offset_h_ptr = (((3 * kernel_idx + 1) * time_col + t_out) * height_col + h_out) * width_col + w_out;
      const int data_offset_w_ptr = (((3 * kernel_idx + 2) * time_col + t_out) * height_col + h_out) * width_col + w_out;
      const scalar_t offset_t = data_offset_ptr[data_offset_t_ptr];
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_t = static_cast<scalar_t>(t_in + k * dilation_t) + offset_t;
      scalar_t inv_h = static_cast<scalar_t>(h_in + i * dilation_h) + offset_h;
      scalar_t inv_w = static_cast<scalar_t>(w_in + j * dilation_w) + offset_w;
      if (inv_t <= -1 || inv_h <= -1 || inv_w <= -1 ||
          inv_t >= time || inv_h >= height || inv_w >= width)
      {
        inv_t = static_cast<scalar_t>(-2);
        inv_h = static_cast<scalar_t>(-2);
        inv_w = static_cast<scalar_t>(-2);
      }
      const int im_channel_idx = col_c / col_step;
      const scalar_t weight = get_coordinate_weight_3d(
          inv_t,
          inv_h,
          inv_w,
          time,
          height,
          width,
          data_im_ptr + im_channel_idx * time * height * width,
          height,
          width,
          bp_dir);
      val += weight * data_col_ptr[col_pos];
    }
    grad_offset[index] = val;
  }
}

void deformable_col2im_coord_3d(
    const at::Tensor data_col,
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int time,
    const int height,
    const int width,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int deformable_group,
    at::Tensor grad_offset)
{
  int time_col = (time + 2 * pad_t - (dilation_t * (ksize_t - 1) + 1)) / stride_t + 1;
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  const int offset_channels = 3 * ksize_t * ksize_h * ksize_w * deformable_group;
  int num_kernels = time_col * height_col * width_col * offset_channels * parallel_imgs;
  int channel_per_deformable_group = channels * ksize_t * ksize_h * ksize_w / deformable_group;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();
        deformable_col2im_coord_gpu_kernel_3d<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_kernels,
            data_col_,
            data_im_,
            data_offset_,
            channels,
            time,
            height,
            width,
            ksize_t,
            ksize_h,
            ksize_w,
            pad_t,
            pad_h,
            pad_w,
            stride_t,
            stride_h,
            stride_w,
            dilation_t,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            parallel_imgs,
            offset_channels,
            deformable_group,
            time_col,
            height_col,
            width_col,
            grad_offset_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_coord_3d: %s\n", cudaGetErrorString(err));
  }
}
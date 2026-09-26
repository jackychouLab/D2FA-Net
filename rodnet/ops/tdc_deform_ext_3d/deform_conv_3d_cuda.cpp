#include <torch/extension.h>
#include <ATen/DeviceGuard.h>
#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif
#ifndef AT_ERROR
#define AT_ERROR(...) TORCH_CHECK(false, __VA_ARGS__)
#endif
#include <cmath>
#include <vector>
#include <iostream>
void deformable_im2col_3d(const at::Tensor data_im, const at::Tensor data_offset,
                       const int channels, const int time, const int height, const int width,
                       const int ksize_t, const int ksize_h, const int ksize_w,
                       const int pad_t, const int pad_h, const int pad_w,
                       const int stride_t, const int stride_h, const int stride_w,
                       const int dilation_t, const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       at::Tensor data_col);

void deformable_col2im_3d(const at::Tensor data_col, const at::Tensor data_offset,
                       const int channels, const int time, const int height, const int width,
                       const int ksize_t, const int ksize_h, const int ksize_w,
                       const int pad_t, const int pad_h, const int pad_w,
                       const int stride_t, const int stride_h, const int stride_w,
                       const int dilation_t, const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deformable_group,
                       at::Tensor grad_im);

void deformable_col2im_coord_3d(
    const at::Tensor data_col, const at::Tensor data_im,
    const at::Tensor data_offset, const int channels,
    const int time, const int height, const int width,
    const int ksize_t, const int ksize_h, const int ksize_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int parallel_imgs,
    const int deformable_group, at::Tensor grad_offset);

void shape_check(at::Tensor input, at::Tensor offset, at::Tensor *gradOutput,
                 at::Tensor weight, int kH, int kW, int kT, int dH, int dW, int dT,
                 int padH, int padW, int padT, int dilationH, int dilationW, int dilationT,
                 int group, int deformable_group) {
  AT_CHECK(weight.ndimension() == 5,
           "5D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
           "but got: %s",
           weight.ndimension());

  AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  AT_CHECK(kW > 0 && kH > 0 && kT > 0,
           "kernel size should be greater than zero, but got kH: %d kW: %d kT: %d", kH,
           kW, kT);

  AT_CHECK((weight.size(2) == kT && weight.size(3) == kH && weight.size(4) == kW),
           "kernel size should be consistent with weight, ",
           "but got kH: %d kW: %d kT: %d weight.size(2): %d, weight.size(3): %d, weight.size(4): %d", kH,
           kW, kT, weight.size(2), weight.size(3), weight.size(4));

  AT_CHECK(dW > 0 && dH > 0 && dT > 0,
           "stride should be greater than zero, but got dH: %d dW: %d dT: %d", dH, dW, dT);

  AT_CHECK(
      dilationW > 0 && dilationH > 0 && dilationT > 0,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d dilationT: %d",
      dilationH, dilationW, dilationT);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;
  if (ndim == 5) {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }
  AT_CHECK(ndim == 4 || ndim == 5, "4D or 5D input tensor expected but got: %s", ndim);
  long nInputPlane = weight.size(1) * group;
  long inputTime = input.size(dimt);
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  AT_CHECK(nInputPlane % deformable_group == 0, "input channels must divide deformable group size");
  if (outputTime < 1 || outputHeight < 1 || outputWidth < 1)
    AT_ERROR(
        "Given input size: (%ld x %ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld x %ld). Output size is too small",
        nInputPlane, inputTime, inputHeight, inputWidth,
        nOutputPlane, outputTime, outputHeight, outputWidth);

  AT_CHECK(input.size(1) == nInputPlane,
           "invalid number of input planes, expected: %d, but got: %d",
           nInputPlane, input.size(1));

  AT_CHECK((inputHeight >= kH && inputWidth >= kW && inputTime >= kT),
           "input data is smaller than kernel");

  AT_CHECK(offset.ndimension() == 5,
           "5D offset tensor expected, but got: %d", offset.ndimension());

  AT_CHECK((offset.size(2) == outputTime && offset.size(3) == outputHeight && offset.size(4) == outputWidth),
           "invalid spatial size of offset, expected time: %d height: %d width: %d, but "
           "got time: %d height: %d width: %d",
           outputTime, outputHeight, outputWidth, offset.size(2), offset.size(3), offset.size(4));

  AT_CHECK((offset.size(1) == deformable_group * 3 * kH * kW * kT),
           "invalid number of channels of offset, expected: %d, but got: %d",
           deformable_group * 3 * kH * kW * kT, offset.size(1));

  if (gradOutput != NULL) {
    AT_CHECK(gradOutput->size(dimf) == nOutputPlane,
             "invalid number of gradOutput planes, expected: %d, but got: %d",
             nOutputPlane, gradOutput->size(dimf));
    AT_CHECK((gradOutput->size(dimt) == outputTime &&
              gradOutput->size(dimh) == outputHeight &&
              gradOutput->size(dimw) == outputWidth),
             "invalid size of gradOutput, expected time: %d height: %d width: %d, but "
             "got time: %d height: %d width: %d",
             outputTime, outputHeight, outputWidth,
             gradOutput->size(dimt), gradOutput->size(dimh), gradOutput->size(dimw));
  }
}

int deform_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor offset, at::Tensor output,
                             at::Tensor columns, at::Tensor ones,
                             int kW, int kH, int kT, int dW, int dH, int dT,
                             int padW, int padH, int padT,
                             int dilationW, int dilationH, int dilationT,
                             int group, int deformable_group, int im2col_step) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  #ifdef DEBUG_INFO
  std::cout << "[cpp]deform_conv_forward_cuda: forward start" << "\n";
  #endif
  shape_check(input, offset, NULL, weight, kH, kW, kT, dH, dW, dT, padH, padW, padT, dilationH, dilationW, dilationT, group, deformable_group);
  at::DeviceGuard guard(input.device());
  #ifdef DEBUG_INFO
  std::cout << "[cpp]deform_conv_forward_cuda: finish shape_check()" << "\n";
  #endif
  input = input.contiguous();
  offset = offset.contiguous();
  weight = weight.contiguous();
  int batch = 1;
  if (input.ndimension() == 4) {
    batch = 0;
    input.unsqueeze_(0);
    offset.unsqueeze_(0);
  }

  // todo: assert batchsize dividable by im2col_step
  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputTime = input.size(2);
  long inputHeight = input.size(3);
  long inputWidth = input.size(4);
  long nOutputPlane = weight.size(0);
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  AT_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");
  output = output.view({batchSize / im2col_step, im2col_step, nOutputPlane, outputTime, outputHeight, outputWidth});
  #ifdef DEBUG_INFO
  std::cout << "[cpp]deform_conv_forward_cuda: columns.size=" << nInputPlane * kW * kH * kT << " " << im2col_step * outputHeight * outputWidth * outputTime << std::endl;
  #endif
  columns = at::zeros({nInputPlane * kW * kH * kT, im2col_step * outputHeight * outputWidth * outputTime}, input.options());
  #ifdef DEBUG_INFO
  std::cout << "[cpp]deform_conv_forward_cuda: finish build columns" << "\n";
  #endif
  if (ones.ndimension() != 3 || ones.size(0) * ones.size(1) * ones.size(2) < outputTime * outputHeight * outputWidth)
  {
    ones = at::ones({outputTime, outputHeight, outputWidth}, input.options());
  }
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime, inputHeight, inputWidth});
  offset = offset.view({batchSize / im2col_step, im2col_step, deformable_group * 3 * kH * kW * kT, outputTime, outputHeight, outputWidth});
  #ifdef DEBUG_INFO
  std::cout << "[cpp]deform_conv_forward_cuda: finish build input & offset" << "\n";
  #endif
  at::Tensor output_buffer = at::zeros({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight, outputWidth}, output.options());
  // TODO: dim different from original mmdet: flatten(1) following ???TO CHECK???
  output_buffer = output_buffer.view( {output_buffer.size(0), group, output_buffer.size(1) / group, output_buffer.size(2), output_buffer.size(3), output_buffer.size(4), output_buffer.size(5)});
  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col_3d(input[elt], offset[elt], nInputPlane, inputTime, inputHeight, inputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW, dilationT, dilationH, dilationW, im2col_step, deformable_group, columns);
    #ifdef DEBUG_INFO
    std::cout << "[cpp]deform_conv_forward_cuda: finish deformable_im2col()" << "\n";
    #endif
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3), weight.size(4)});
    for (int g = 0; g < group; g++) {
      output_buffer[elt][g] = output_buffer[elt][g].flatten(1).addmm_(weight[g].flatten(1), columns[g]).view_as(output_buffer[elt][g]);
    }
    #ifdef DEBUG_INFO
    std::cout << "[cpp]deform_conv_forward_cuda: finish calculate output_buffer" << "\n";
    #endif
  }
  output_buffer = output_buffer.view({output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2), output_buffer.size(3), output_buffer.size(4), output_buffer.size(5), output_buffer.size(6)});
  output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight, outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputTime, outputHeight, outputWidth});
  input = input.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  offset = offset.view({batchSize, deformable_group * 3 * kH * kW * kT, outputTime, outputHeight, outputWidth});
  if (batch == 0) {
    output = output.view({nOutputPlane, outputTime, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputTime, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
  }
  return 1;
}

int deform_conv_backward_input_cuda(at::Tensor input, at::Tensor offset,
                                    at::Tensor gradOutput, at::Tensor gradInput,
                                    at::Tensor gradOffset, at::Tensor weight,
                                    at::Tensor columns,
                                    int kW, int kH, int kT, int dW, int dH, int dT,
                                    int padW, int padH, int padT,
                                    int dilationW, int dilationH, int dilationT,
                                    int group, int deformable_group, int im2col_step) {
  shape_check(input, offset, &gradOutput, weight, kH, kW, kT, dH, dW, dT, padH, padW, padT,
              dilationH, dilationW, dilationT, group, deformable_group);
  at::DeviceGuard guard(input.device());
  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();
  weight = weight.contiguous();
  int batch = 1;
  if (input.ndimension() == 4) {
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2), input.size(3)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2), offset.size(3)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3)});
  }
  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputTime = input.size(2);
  long inputHeight = input.size(3);
  long inputWidth = input.size(4);
  long nOutputPlane = weight.size(0);
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  AT_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");
  gradInput = gradInput.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  columns = at::zeros( {nInputPlane * kW * kH * kT, im2col_step * outputTime * outputHeight * outputWidth}, input.options());
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step, nOutputPlane, outputTime, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);
  gradInput = gradInput.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime, inputHeight, inputWidth});
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime, inputHeight, inputWidth});
  gradOffset = gradOffset.view({batchSize / im2col_step, im2col_step, deformable_group * 3 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  offset = offset.view({batchSize / im2col_step, im2col_step, deformable_group * 3 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3), weight.size(4)});
    gradOutput = gradOutput.view({gradOutput.size(0), group, gradOutput.size(1) / group, gradOutput.size(2), gradOutput.size(3), gradOutput.size(4), gradOutput.size(5)});
    for (int g = 0; g < group; g++) {
      columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose(0, 1), gradOutput[elt][g].flatten(1), 0.0f, 1.0f);
    }
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradOutput = gradOutput.view( {gradOutput.size(0), gradOutput.size(1) * gradOutput.size(2), gradOutput.size(3), gradOutput.size(4), gradOutput.size(5), gradOutput.size(6)});
    deformable_col2im_coord_3d(columns, input[elt], offset[elt], nInputPlane, inputTime, inputHeight, inputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW, dilationT, dilationH, dilationW, im2col_step, deformable_group, gradOffset[elt]);
    deformable_col2im_3d(columns, offset[elt], nInputPlane, inputTime, inputHeight, inputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW, dilationT, dilationH, dilationW, im2col_step, deformable_group, gradInput[elt]);
  }
  gradOutput.transpose_(1, 2);
  gradOutput = gradOutput.view({batchSize, nOutputPlane, outputTime, outputHeight, outputWidth});
  gradInput = gradInput.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  gradOffset = gradOffset.view({batchSize, deformable_group * 3 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  offset = offset.view({batchSize, deformable_group * 3 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputTime, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputTime, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputTime, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
    gradOffset = gradOffset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
  }
  return 1;
}

int deform_conv_backward_parameters_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor gradOutput,
    at::Tensor gradWeight,
    at::Tensor columns, at::Tensor ones,
    int kW, int kH, int kT, int dW, int dH, int dT,
    int padW, int padH, int padT,
    int dilationW, int dilationH, int dilationT,
    int group,
    int deformable_group, float scale, int im2col_step) {
  // todo: transpose and reshape outGrad
  // todo: reshape columns
  // todo: add im2col_step as input
  shape_check(input, offset, &gradOutput, gradWeight, kH, kW, kT, dH, dW, dT, padH, padW, padT, dilationH, dilationW, dilationT, group, deformable_group);
  at::DeviceGuard guard(input.device());
  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();
  int batch = 1;
  if (input.ndimension() == 4) {
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2), input.size(3)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3)});
  }
  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputTime = input.size(2);
  long inputHeight = input.size(3);
  long inputWidth = input.size(4);
  long nOutputPlane = gradWeight.size(0);
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  AT_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");
  columns = at::zeros({nInputPlane * kW * kH * kT, im2col_step * outputHeight * outputWidth * outputTime}, input.options());
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step, nOutputPlane, outputTime, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);
  at::Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight, outputWidth});
  gradOutputBuffer.copy_(gradOutput);
  gradOutput.transpose_(1, 2);
  gradOutput = gradOutput.view({batchSize, nOutputPlane, outputTime, outputHeight, outputWidth});
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime, inputHeight, inputWidth});
  offset = offset.view({batchSize / im2col_step, im2col_step, deformable_group * 3 * kH * kW * kT, outputTime, outputHeight, outputWidth});
  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col_3d(input[elt], offset[elt], nInputPlane, inputTime, inputHeight, inputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW, dilationT, dilationH, dilationW, im2col_step, deformable_group, columns);
    gradOutputBuffer = gradOutputBuffer.view({gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) / group, gradOutputBuffer.size(2), gradOutputBuffer.size(3), gradOutputBuffer.size(4), gradOutputBuffer.size(5)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    gradWeight = gradWeight.view({group, gradWeight.size(0) / group, gradWeight.size(1), gradWeight.size(2), gradWeight.size(3), gradWeight.size(4)});
    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g].flatten(1).addmm_(gradOutputBuffer[elt][g].flatten(1), columns[g].transpose(1, 0), 1.0, scale).view_as(gradWeight[g]);
    }
    gradOutputBuffer = gradOutputBuffer.view({gradOutputBuffer.size(0), gradOutputBuffer.size(1) * gradOutputBuffer.size(2), gradOutputBuffer.size(3), gradOutputBuffer.size(4), gradOutputBuffer.size(5), gradOutputBuffer.size(6)});
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradWeight = gradWeight.view({gradWeight.size(0) * gradWeight.size(1), gradWeight.size(2), gradWeight.size(3), gradWeight.size(4), gradWeight.size(5)});
  }
  input = input.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  offset = offset.view({batchSize, deformable_group * 3 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputTime, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputTime, inputHeight, inputWidth});
  }
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv_forward_cuda", &deform_conv_forward_cuda,
        "deform forward (CUDA)");
  m.def("deform_conv_backward_input_cuda", &deform_conv_backward_input_cuda,
        "deform_conv_backward_input (CUDA)");
  m.def("deform_conv_backward_parameters_cuda",
        &deform_conv_backward_parameters_cuda,
        "deform_conv_backward_parameters (CUDA)");
}

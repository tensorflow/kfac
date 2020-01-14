/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <vector>

#include "third_party/tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "third_party/tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "third_party/tensorflow/compiler/xla/client/xla_builder.h"
#include "third_party/tensorflow/compiler/xla/literal_util.h"
#include "third_party/tensorflow/compiler/xla/shape.h"
#include "third_party/tensorflow/compiler/xla/shape_util.h"
#include "third_party/tensorflow/compiler/xla/util.h"
#include "third_party/tensorflow/compiler/xla/window_util.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/lib/core/errors.h"

namespace deepmind {
namespace tensorflow {
namespace kfac {
namespace {

using ::tensorflow::errors::InvalidArgument;

class PatchesSecondMomentXlaOp : public ::tensorflow::XlaOpKernel {
 public:
  explicit PatchesSecondMomentXlaOp(::tensorflow::OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("kernel_shape", &kernel_shape_));
    OP_REQUIRES(ctx, kernel_shape_.size() == 2,
                InvalidArgument("Kernel shape must be of length 2"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stride", &stride_));
    OP_REQUIRES(ctx, stride_.size() == 2,
                InvalidArgument("Stride must be of length 2"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    OP_REQUIRES(ctx, padding_.size() == 2,
                InvalidArgument("Kernel padding must be of length 2"));
  }

  // The gereneral idea is that each [1,1,C,1,1,C] element in the matrix output
  // tensor of [kH,kW,C,kH,kW,C] can be computed independently using a dot
  // product with the not participating elements sliced (or masked) out. We
  // iterate over the first kH,kW dimensions using a while loop and then use a
  // convolution to iterate over the second pair of kH,kW dimensions.
  void Compile(::tensorflow::XlaOpKernelContext* ctx) override {
    const auto input = ctx->Input(0);
    const ::tensorflow::TensorShape input_shape = ctx->InputShape(0);
    const int64 batch_size = input_shape.dim_size(0);
    const int64 image_height = input_shape.dim_size(1);
    const int64 image_width = input_shape.dim_size(2);
    const int64 num_channels = input_shape.dim_size(3);
    const int64 output_size =
        kernel_shape_[0] * kernel_shape_[1] * num_channels;

    // We will pad the image to the closest multiple of stride so we will have
    // evenly sized slices inside the while loop.
    const int64 padded_height = xla::RoundUpToNearest(
        image_height + 2 * padding_[0], static_cast<int64>(stride_[0]));
    const int64 padded_width = xla::RoundUpToNearest(
        image_width + 2 * padding_[1], static_cast<int64>(stride_[1]));

    // Compute the shape of the while loop what will be used for the nested
    // parameters.
    const xla::Shape padded_image_shape = xla::ShapeUtil::MakeShape(
        xla::F32, {batch_size, padded_height, padded_width, num_channels});
    const xla::Shape matrix_result_shape = xla::ShapeUtil::MakeShape(
        xla::F32, {kernel_shape_[0], kernel_shape_[1], num_channels,
                   kernel_shape_[0], kernel_shape_[1], num_channels});
    const xla::Shape vector_result_shape = xla::ShapeUtil::MakeShape(
        xla::F32, {kernel_shape_[0], kernel_shape_[1], num_channels});
    const xla::Shape loop_shape = xla::ShapeUtil::MakeTupleShape(
        {xla::ShapeUtil::MakeShape(xla::S32, {}), padded_image_shape,
         matrix_result_shape, vector_result_shape});

    auto build_body = [&]() {
      auto builder = ctx->builder()->CreateSubBuilder("body");
      auto constant_s32 = [&](int32 x) { return ConstantR0(builder.get(), x); };
      auto constant_f32 = [&](float x) { return ConstantR0(builder.get(), x); };

      auto param = Parameter(builder.get(), 0, loop_shape, "");
      auto index = GetTupleElement(param, 0);
      auto padded_image = GetTupleElement(param, 1);
      auto matrix_result = GetTupleElement(param, 2);
      auto vector_result = GetTupleElement(param, 3);

      // We use a single while loop to iterate over kH and kW for simpler
      // generated HLO code so we have to convert the linear index to a 2D
      // index.
      auto index_h = Rem(index, constant_s32(kernel_shape_[0]));
      auto index_w = Div(index, constant_s32(kernel_shape_[0]));

      // Introduce new `stride` sized dimensions so we can do a strided-slice
      // via the subsequent dynamic-slice op.
      auto rhs_padded_image = Reshape(
          padded_image, {batch_size, padded_height / stride_[0], stride_[0],
                         padded_width / stride_[1], stride_[1], num_channels});

      // Compute the size of the RHS required for the convolution to contain all
      // of the elements what are in bound for the current iteration and then
      // dynamic-slice the current RHS out of the full image based on the
      // iteration indices. We rely on the R6 shape to correctly slice out along
      // the striding dimension as well.
      const int64 rhs_height =
          (image_height + 2 * padding_[0] - kernel_shape_[0]) / stride_[0] + 1;
      const int64 rhs_width =
          (image_width + 2 * padding_[1] - kernel_shape_[1]) / stride_[1] + 1;
      auto conv_rhs = DynamicSlice(
          rhs_padded_image,
          {constant_s32(0), Div(index_h, constant_s32(stride_[0])),
           Rem(index_h, constant_s32(stride_[0])),
           Div(index_w, constant_s32(stride_[1])),
           Rem(index_w, constant_s32(stride_[1])), constant_s32(0)},
          {batch_size, rhs_height, 1, rhs_width, 1, num_channels});

      // Introduce new trivial dimensions for the activation of the convolution
      // to match the rank with the rank of the kernel what is R6 due to the
      // code required to handle striding.
      auto conv_lhs = Reshape(padded_image, {batch_size, padded_height, 1,
                                             padded_width, 1, num_channels});

      // The activation can have some extra elements coming from the requirement
      // that the padded image have to be a multiple of stride so we need some
      // (negative) pad high on the convolution to remove it.
      const int64 height_pad_high =
          kernel_shape_[0] -
          (padded_height -
           xla::window_util::DilatedBound(rhs_height, stride_[0]) + 1);
      const int64 width_pad_high =
          kernel_shape_[1] -
          (padded_width -
           xla::window_util::DilatedBound(rhs_width, stride_[1]) + 1);

      // Use a convolution to calculate [1,1,C,kH,kW,C] elements of the result.
      // The original stride is turned into a window dilation to restore the
      // original kernel size after the strided slice and end up with a kernel
      // when only every stride'th element is non-zero.
      xla::ConvolutionDimensionNumbers dnums;
      dnums.set_input_batch_dimension(5);
      dnums.set_input_feature_dimension(0);
      dnums.add_input_spatial_dimensions(1);
      dnums.add_input_spatial_dimensions(2);
      dnums.add_input_spatial_dimensions(3);
      dnums.add_input_spatial_dimensions(4);
      dnums.set_kernel_input_feature_dimension(0);
      dnums.set_kernel_output_feature_dimension(5);
      dnums.add_kernel_spatial_dimensions(1);
      dnums.add_kernel_spatial_dimensions(2);
      dnums.add_kernel_spatial_dimensions(3);
      dnums.add_kernel_spatial_dimensions(4);
      dnums.set_output_batch_dimension(5);
      dnums.set_output_feature_dimension(0);
      dnums.add_output_spatial_dimensions(1);
      dnums.add_output_spatial_dimensions(2);
      dnums.add_output_spatial_dimensions(3);
      dnums.add_output_spatial_dimensions(4);
      auto matrix_value = Reshape(
          xla::ConvGeneralDilated(
              conv_lhs, conv_rhs, {1, 1, 1, 1},
              {{0, height_pad_high}, {0, 0}, {0, width_pad_high}, {0, 0}},
              {1, 1, 1, 1}, {stride_[0], 1, stride_[1], 1}, dnums),
          {1, 1, num_channels, kernel_shape_[0], kernel_shape_[1],
           num_channels});

      // Update the slice of the output matrix what was computed by the above
      // convolution in the result matrix.
      auto new_matrix_result = DynamicUpdateSlice(
          matrix_result, matrix_value,
          {index_h, index_w, constant_s32(0), constant_s32(0), constant_s32(0),
           constant_s32(0)});

      // The corresponding slice of the vector output is just the sum of the
      // elements in the strided kernel.
      auto vector_value = BroadcastInDim(
          Reduce(conv_rhs, constant_f32(0.0),
                 *ctx->GetOrCreateAdd(input_type(0)), {0, 1, 2, 3, 4}),
          {1, 1, num_channels}, {2});

      // Update the slize of the output vector what was computed by the above
      // reduce in the result vector.
      auto new_vector_result = DynamicUpdateSlice(
          vector_result, vector_value, {index_h, index_w, constant_s32(0)});

      // Update loop index for the next iteration of the loop.
      auto new_index = Add(index, constant_s32(1));
      auto new_param = Tuple(
          builder.get(),
          {new_index, padded_image, new_matrix_result, new_vector_result});
      return builder->Build(new_param).ValueOrDie();
    };

    auto build_cond = [&]() {
      auto builder = ctx->builder()->CreateSubBuilder("cond");
      auto param = Parameter(builder.get(), 0, loop_shape, "");
      auto index = GetTupleElement(param, 0);
      auto result = Lt(index, ConstantR0(builder.get(),
                                         kernel_shape_[0] * kernel_shape_[1]));
      return builder->Build(result).ValueOrDie();
    };

    auto zero_f32 = ConstantR0(ctx->builder(), 0.0f);

    // Pad the image first by the specified padding amount followed by some
    // extra padding to ensure that the image size is a multiple of stride so we
    // can use the reshape trick for executing a strided slice via a
    // dynamic-slice.
    xla::PaddingConfig pad_config;
    pad_config.add_dimensions();
    auto pad_height_dim = pad_config.add_dimensions();
    pad_height_dim->set_edge_padding_low(padding_[0]);
    pad_height_dim->set_edge_padding_high(padded_height - image_height -
                                          padding_[0]);
    auto pad_width_dim = pad_config.add_dimensions();
    pad_width_dim->set_edge_padding_low(padding_[1]);
    pad_width_dim->set_edge_padding_high(padded_width - image_width -
                                         padding_[1]);
    pad_config.add_dimensions();
    auto padded_image = Pad(input, zero_f32, pad_config);

    // Build the while loop iterating over the first pair of kH,KW dimensions.
    auto param = Tuple(ctx->builder(),
                       {ConstantR0(ctx->builder(), 0), padded_image,
                        Broadcast(zero_f32, matrix_result_shape.dimensions()),
                        Broadcast(zero_f32, vector_result_shape.dimensions())});
    auto while_result = While(build_cond(), build_body(), param);

    // Reshape the outputs to R2 and R1 respectively from the R6 and R3 tensors
    // produced by the while loop.
    auto matrix_out =
        Reshape(GetTupleElement(while_result, 2), {output_size, output_size});
    auto vector_out = Reshape(GetTupleElement(while_result, 3), {output_size});
    ctx->SetOutput(0, matrix_out);
    ctx->SetOutput(1, vector_out);
  }

 private:
  std::vector<int32> kernel_shape_;
  std::vector<int32> stride_;
  std::vector<int32> padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(PatchesSecondMomentXlaOp);
};

REGISTER_XLA_OP(Name("PatchesSecondMoment"), PatchesSecondMomentXlaOp);

}  // namespace
}  // namespace kfac
}  // namespace tensorflow
}  // namespace deepmind

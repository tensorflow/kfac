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

#include "third_party/tensorflow/core/framework/op.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"

namespace deepmind {
namespace tensorflow {
namespace kfac {
namespace {

using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("PatchesSecondMoment")
    .Input("image: float")
    .Output("output_matrix: float")
    .Output("output_vector: float")
    .Attr("kernel_shape: list(int)")
    .Attr("stride: list(int)")
    .Attr("padding: list(int)")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle image_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &image_shape));
      // Assumes NHWC layout.
      DimensionHandle num_channels_dim = c->Dim(image_shape, 3);
      std::vector<int32> kernel_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_shape", &kernel_shape));
      if (kernel_shape.size() != 2) {
        return ::tensorflow::errors::InvalidArgument(
            "kernel_shape must be a list of two integers");
      }
      DimensionHandle kernel_size_dim =
          c->MakeDim(kernel_shape[0] * kernel_shape[1]);
      DimensionHandle output_size;
      TF_RETURN_IF_ERROR(
          c->Multiply(num_channels_dim, kernel_size_dim, &output_size));
      c->set_output(0, c->MakeShape({output_size, output_size}));
      c->set_output(1, c->MakeShape({output_size}));
      return ::tensorflow::Status::OK();
    });

}  // namespace
}  // namespace kfac
}  // namespace tensorflow
}  // namespace deepmind

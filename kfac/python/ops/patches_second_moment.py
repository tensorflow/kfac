# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Custom op for computing the second moment of the patches to a conv layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from kfac.cc.ops import gen_patches_second_moment_ops


def patches_second_moment(image, kernel_shape, stride=1, padding="VALID"):
  """Computes the second moment of the patch vectors to a conv layer.

  Computes the sum and the sum of self outer products over all patch vectors
  of the input image/feature-map, given a kernel shape, stride and padding.
  This should be much more memory efficient than simply extracting the patches
  and doing a matrix multiply.

  Naive NumPy implementation (ignoring padding):

  ```python
  output_size = kernel_height * kernel_width * num_channels
  output_matrix = np.zeros([output_size, output_size])
  output_vector = np.zeros([output_size])
  # Iterate over image patches.
  for x in xrange(0, image.shape[1] - kernel_height + 1, stride):
    for y in xrange(0, image.shape[2] - kernel_width + 1, stride):
      # Extract image patch and flatten.
      patch = np.reshape(
          image[:, x:x+kernel_height, y:y+kernel_width, :], [batch_size, -1])
      output_matrix += np.matmul(np.transpose(patch), patch)
      output_vector += np.sum(patch, axis=0, keepdims=False)  # Sum over batch.
  ```

  Args:
    image: The 4D input image, in NHWC layout.
    kernel_shape: List of two integers describing the shape of the kernel. Can
        be a scalar if the kernel is square.
    stride: List of two integers describing the shape of the kernel. Can be a
        scalar if the stride is equal in both dimensions.
    padding: Either a list of two integers (height and width padding), or a
        string ('VALID' or 'SAME').

  Returns:
    A 2D `Tensor` of size NxN containing the sum of outer products of the
    patch vectors (that each have dimension N), and a 1D `Tensor` of size N
    containing the sum over these vectors, where
    N = kernel_height * kernel_width * num_channels.

  Raises:
    ValueError: If the parameters don't conform to the specification.
  """
  if not isinstance(kernel_shape, (list, tuple)):
    kernel_shape = [kernel_shape, kernel_shape]
  if len(kernel_shape) != 2:
    raise ValueError("kernel_shape must be a list of two integers or a scalar.")

  if not isinstance(stride, (list, tuple)):
    stride = [stride, stride]
  if len(stride) != 2:
    raise ValueError("stride must be a list of two integers or a scalar.")

  if isinstance(padding, str):
    if padding == "VALID":
      padding = [0, 0]
    elif padding == "SAME":
      padding = [(kernel_shape[0] - 1) / 2, (kernel_shape[1] - 1) / 2]
  if len(padding) != 2:
    raise ValueError("padding must be a list of two integers or a string "
                     "('VALID' or 'SAME').")

  return gen_patches_second_moment_ops.patches_second_moment(
      image, kernel_shape, stride, padding)

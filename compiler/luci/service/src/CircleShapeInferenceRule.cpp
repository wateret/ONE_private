/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci/Service/CircleShapeInferenceRule.h"
#include "Check.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleDialect.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <oops/InternalExn.h>

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace
{

// Call this for CircleAvgPool2D and CircleMaxPool2D only
template <class Pool2DType> loco::NodeShape infer_pool_2d_shape(const Pool2DType *node)
{
  LUCI_ASSERT(loco::shape_known(node->value()), "Shape must be known");

  auto ifm_shape = loco::shape_get(node->value()).template as<loco::TensorShape>();
  assert(ifm_shape.rank() == 4);

  uint32_t input_height = ifm_shape.dim(1).value();
  uint32_t input_width = ifm_shape.dim(2).value();
  uint32_t stride_height = node->stride()->h();
  uint32_t stride_width = node->stride()->w();
  uint32_t window_height = node->filter()->h();
  uint32_t window_width = node->filter()->w();
  uint32_t dilation_height = 1; // dilation for CircleAvgPool2D and CircleMaxPool2D is 1
  uint32_t dilation_width = 1;
  uint32_t effective_window_height = dilation_height * (window_height - 1) + 1;
  uint32_t effective_window_width = dilation_width * (window_width - 1) + 1;

  uint32_t output_height = 0;
  uint32_t output_width = 0;

  if (node->padding() == luci::Padding::VALID)
  {
    output_height = (input_height + stride_height - effective_window_height) / stride_height;
    output_width = (input_width + stride_width - effective_window_width) / stride_width;
  }
  else if (node->padding() == luci::Padding::SAME)
  {
    output_height = (input_height + stride_height - 1) / stride_height;
    output_width = (input_width + stride_width - 1) / stride_width;
  }
  else
    LUCI_ASSERT(false, "Wrong padding type");

  loco::TensorShape ofm_shape;
  ofm_shape.rank(4);
  ofm_shape.dim(0) = ifm_shape.dim(0);
  ofm_shape.dim(1) = output_height;
  ofm_shape.dim(2) = output_width;
  ofm_shape.dim(3) = ifm_shape.dim(3);

  return loco::NodeShape{ofm_shape};
}

/**
 * @brief Create a higher-rank TensorShape following NumPy broadcasting semantics
 *
 * HOW TO USE:
 *
 *   auto expanded_tensor_shape = expand(tensor_shape).to(N);
 */
class TensorShapeExpander
{
public:
  TensorShapeExpander(const loco::TensorShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  loco::TensorShape to(uint32_t output_rank)
  {
    auto const &input_shape = _shape;
    uint32_t const input_rank = input_shape.rank();

    assert(input_rank <= output_rank && "Cannot shrink rank");
    uint32_t const axis_shift = output_rank - input_rank;

    loco::TensorShape output_shape;

    output_shape.rank(output_rank);
    for (uint32_t axis = 0; axis < output_rank; ++axis)
    {
      output_shape.dim(axis) = (axis < axis_shift) ? 1 : input_shape.dim(axis - axis_shift);
    }

    return output_shape;
  }

private:
  const loco::TensorShape _shape;
};

/**
 * @breif  Expand shape x and y to same rank by align right and filling with 1
 */
void expand_rank(loco::TensorShape &x, loco::TensorShape &y)
{
  auto x_rank = x.rank();
  auto y_rank = y.rank();

  if (x_rank == y_rank)
    return;

  TensorShapeExpander x_exp(x);
  TensorShapeExpander y_exp(y);

  auto xy_rank = std::max(x_rank, y_rank);

  x = x_rank > y_rank ? x : x_exp.to(xy_rank);
  y = y_rank > x_rank ? y : y_exp.to(xy_rank);
}

/**
 * @breif  Returns shape of expanded dimension of input x and y having same rank
 */
loco::TensorShape expand_dimension(const loco::TensorShape &x, const loco::TensorShape &y)
{
  assert(x.rank() == y.rank());

  auto rank = x.rank();

  loco::TensorShape output_shape;

  output_shape.rank(rank);
  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    assert(x.dim(axis).known() && y.dim(axis).known());

    auto x_dim = x.dim(axis).value();
    auto y_dim = y.dim(axis).value();

    // each dimension of x and y should be same or one must be 1 if different
    if (!((x_dim == y_dim) || (x_dim == 1 || y_dim == 1)))
      INTERNAL_EXN("Cannot produce expand_dimension of two shapes");

    output_shape.dim(axis) = std::max(x_dim, y_dim);
  }

  return output_shape;
}

loco::TensorShape broadcast_shape(const loco::TensorShape &x, const loco::TensorShape &y)
{
  auto x_match = x;
  auto y_match = y;

  expand_rank(x_match, y_match);

  auto output_shape = expand_dimension(x_match, y_match);

  return output_shape;
}

/**
 * @brief Class to infer the shape of CircleNode
 *
 * @note All CircleNode's inputs and outputs are always loco::Domain::Tensor
 */
class ShapeInferenceAlgorithm final : public luci::CircleNodeVisitor<loco::NodeShape>
{
public:
  loco::NodeShape visit(const luci::CircleAbs *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    return loco::NodeShape{x_shape};
  }

  loco::NodeShape visit(const luci::CircleAdd *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleArgMax *node) final
  {
    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();
    auto dimension_shape = loco::shape_get(node->dimension()).as<loco::TensorShape>();

    int64_t select_axis = 0;
    {
      LUCI_ASSERT(node->dimension(), "2nd input dimension() should not be nullptr");

      // Only support node's shape() is CircleConst with S32/S64
      // Support S32 for now.
      auto const_shape_node = dynamic_cast<luci::CircleConst *>(node->dimension());
      LUCI_ASSERT(const_shape_node, "Only support CircleConst for shape of CircleArgMax");
      LUCI_ASSERT(const_shape_node->dtype() == loco::DataType::S32,
                  "Only support int32 CircleConst for CircleArgMax");

      if (const_shape_node->rank() > 1)
        INTERNAL_EXN_V("Only support rank 0/1 CircleConst",
                       oops::to_uint32(const_shape_node->rank()));

      select_axis = const_shape_node->scalar<loco::DataType::S32>();
    }
    assert(select_axis < input_shape.rank());
    assert(select_axis >= 0); // TODO support minus of this breaks

    // NOTE select_axis is removed
    loco::TensorShape shape_output;
    uint32_t rank = input_shape.rank();
    uint32_t shrink = static_cast<uint32_t>(select_axis);
    assert(rank > 0);
    shape_output.rank(rank - 1);
    for (uint32_t r = 0, d = 0; r < rank; ++r)
    {
      if (r == shrink)
        continue;
      shape_output.dim(d++) = input_shape.dim(r);
    }
    return loco::NodeShape{shape_output};
  }

  loco::NodeShape visit(const luci::CircleAveragePool2D *node) final
  {
    return infer_pool_2d_shape(node);
  }

  loco::NodeShape visit(const luci::CircleConcatenation *node) final
  {
    // TODO Support when CircleConcatenation has 0 input
    assert(node->numValues() > 0);

    auto first_shape = loco::shape_get(node->values(0)).as<loco::TensorShape>();
    auto axis = node->axis();
    if (axis < 0)
      axis += first_shape.rank();

    assert(0 <= axis);
    assert(first_shape.rank() > static_cast<uint32_t>(axis));

    loco::TensorShape output_shape;

    output_shape.rank(first_shape.rank());
    for (uint32_t i = 0; i < output_shape.rank(); ++i)
      output_shape.dim(i) = first_shape.dim(i);

    for (uint32_t i = 1; i < node->numValues(); ++i)
    {
      auto input_shape = loco::shape_get(node->values(i)).as<loco::TensorShape>();

      for (uint32_t j = 0; j < output_shape.rank(); ++j)
      {
        if (j == static_cast<uint32_t>(axis))
          output_shape.dim(j) = output_shape.dim(j).value() + input_shape.dim(j).value();
        else
          assert(output_shape.dim(j) == input_shape.dim(j));
      }
    }

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleConst *node) final
  {
    loco::TensorShape shape;

    shape.rank(node->rank());
    for (uint32_t axis = 0; axis < node->rank(); axis++)
      shape.dim(axis) = node->dim(axis);

    return loco::NodeShape{shape};
  }

  loco::NodeShape visit(const luci::CircleConv2D *node) final
  {
    LOGGER(l);

    auto ifm_shape = loco::shape_get(node->input()).as<loco::TensorShape>();  // in NHWC
    auto ker_shape = loco::shape_get(node->filter()).as<loco::TensorShape>(); // in OHWI

    INFO(l) << "[luci] CircleConv2D ShapeInf ifm(" << ifm_shape.rank() << ") ker("
            << ker_shape.rank() << ")" << std::endl;

    assert(ifm_shape.rank() == 4);
    assert(ker_shape.rank() == 4);
    assert(ifm_shape.dim(3) == ker_shape.dim(3));

    uint32_t input_height = ifm_shape.dim(1).value();
    uint32_t input_width = ifm_shape.dim(2).value();
    uint32_t stride_height = node->stride()->h();
    uint32_t stride_width = node->stride()->w();
    uint32_t ker_height = ker_shape.dim(1).value();
    uint32_t ker_width = ker_shape.dim(2).value();
    uint32_t dilation_height = 1;
    uint32_t dilation_width = 1;
    uint32_t effective_ker_height = dilation_height * (ker_height - 1) + 1;
    uint32_t effective_ker_width = dilation_width * (ker_width - 1) + 1;

    uint32_t output_height = 0;
    uint32_t output_width = 0;

    if (node->padding() == luci::Padding::VALID)
    {
      output_height = (input_height + stride_height - effective_ker_height) / stride_height;
      output_width = (input_width + stride_width - effective_ker_width) / stride_width;
    }
    else if (node->padding() == luci::Padding::SAME)
    {
      output_height = (input_height + stride_height - 1) / stride_height;
      output_width = (input_width + stride_width - 1) / stride_width;
    }
    else
      LUCI_ASSERT(false, "Wrong padding type");

    loco::TensorShape ofm_shape;
    ofm_shape.rank(4);
    ofm_shape.dim(0) = ifm_shape.dim(0);
    ofm_shape.dim(1) = output_height;
    ofm_shape.dim(2) = output_width;
    ofm_shape.dim(3) = ker_shape.dim(0);

    return loco::NodeShape{ofm_shape};
  }

  loco::NodeShape visit(const luci::CircleCos *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();

    return loco::NodeShape{x_shape};
  }

  loco::NodeShape visit(const luci::CircleDepthwiseConv2D *node) final
  {
    auto ifm_shape = loco::shape_get(node->input()).as<loco::TensorShape>();  // in NHWC
    auto ker_shape = loco::shape_get(node->filter()).as<loco::TensorShape>(); // in 1 H W CM

    assert(ifm_shape.rank() == 4);
    assert(ker_shape.rank() == 4);
    assert(ker_shape.dim(0).value() == 1);

    uint32_t input_height = ifm_shape.dim(1).value();
    uint32_t input_width = ifm_shape.dim(2).value();
    uint32_t stride_height = node->stride()->h();
    uint32_t stride_width = node->stride()->w();
    uint32_t ker_height = ker_shape.dim(1).value();
    uint32_t ker_width = ker_shape.dim(2).value();
    uint32_t dilation_height = 1;
    uint32_t dilation_width = 1;
    uint32_t effective_ker_height = dilation_height * (ker_height - 1) + 1;
    uint32_t effective_ker_width = dilation_width * (ker_width - 1) + 1;

    uint32_t output_height = 0;
    uint32_t output_width = 0;

    if (node->padding() == luci::Padding::VALID)
    {
      output_height = (input_height + stride_height - effective_ker_height) / stride_height;
      output_width = (input_width + stride_width - effective_ker_width) / stride_width;
    }
    else if (node->padding() == luci::Padding::SAME)
    {
      output_height = (input_height + stride_height - 1) / stride_height;
      output_width = (input_width + stride_width - 1) / stride_width;
    }
    else
      LUCI_ASSERT(false, "Wrong padding type");

    loco::TensorShape ofm_shape;
    ofm_shape.rank(4);
    ofm_shape.dim(0) = ifm_shape.dim(0);
    ofm_shape.dim(1) = output_height;
    ofm_shape.dim(2) = output_width;
    ofm_shape.dim(3) = ker_shape.dim(3);

    return loco::NodeShape{ofm_shape};
  }

  loco::NodeShape visit(const luci::CircleDiv *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleEqual *node) final
  {
    const auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    const auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();
    loco::TensorShape output_shape = broadcast_shape(x_shape, y_shape);
    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleFullyConnected *node) final
  {
    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();
    auto weights_shape = loco::shape_get(node->weights()).as<loco::TensorShape>();

    // Checking shape capability for fully connected layer
    // Input: a tensor of at least rank 2 [D1, D2, ... Dn]
    // Weight: [# of units, K]
    // Output: [D1 * D2 * ... * Dn / K, # of units]
    LUCI_ASSERT(input_shape.rank() >= 2, "Input rank should be at least 2");
    LUCI_ASSERT(weights_shape.rank() == 2, "Incompatible weights rank for fully connected");

    uint32_t input_size = 1;
    for (uint32_t i = 0; i < input_shape.rank(); i++)
    {
      input_size = input_size * input_shape.dim(i).value();
    }
    const uint32_t batch_size = input_size / weights_shape.dim(1).value();
    loco::TensorShape out_shape;
    out_shape.rank(2);
    out_shape.dim(0) = batch_size;
    out_shape.dim(1) = weights_shape.dim(0);

    return loco::NodeShape{out_shape};
  }

  loco::NodeShape visit(const luci::CircleLogicalNot *node) final
  {
    const auto input_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleLogicalOr *node) final
  {
    const auto input_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleMaximum *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleMaxPool2D *node) final
  {
    return infer_pool_2d_shape(node);
  }

  loco::NodeShape visit(const luci::CircleMean *node) final
  {
    const loco::DataType S32 = loco::DataType::S32;

    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();
    auto reduction_indices = dynamic_cast<luci::CircleConst *>(node->reduction_indices());

    { // Exceptions
      // TODO support non-const case
      LUCI_ASSERT(reduction_indices, "Only support constant reduction_indices");
      // TODO support other data type
      LUCI_ASSERT(reduction_indices->dtype() == S32, "Only support int 32");
    }

    std::vector<int32_t> reduction_values;

    for (uint32_t i = 0; i < reduction_indices->size<S32>(); ++i)
    {
      int32_t axis = reduction_indices->at<S32>(i);
      if (axis < 0)
        axis += input_shape.rank();
      if (not(0 <= axis and axis < static_cast<int32_t>(input_shape.rank())))
        INTERNAL_EXN_V("Invalid reduction axis for MEAN", oops::to_uint32(axis));
      reduction_values.push_back(axis);
    }

    loco::TensorShape output_shape;

    if (node->keep_dims())
    {
      output_shape.rank(input_shape.rank());
      for (uint32_t i = 0; i < input_shape.rank(); ++i)
        output_shape.dim(i) = input_shape.dim(i);
      for (uint32_t i = 0; i < reduction_values.size(); ++i)
        output_shape.dim(reduction_values.at(i)) = 1;
    }
    else
    {
      std::vector<bool> check_reduce(input_shape.rank(), false);
      for (uint32_t i = 0; i < reduction_values.size(); ++i)
        check_reduce.at(reduction_values.at(i)) = true;

      uint32_t reduce_cnt = 0;
      for (uint32_t i = 0; i < check_reduce.size(); ++i)
        if (check_reduce.at(i))
          ++reduce_cnt;

      output_shape.rank(input_shape.rank() - reduce_cnt);
      for (uint32_t i = 0, j = 0; i < check_reduce.size(); ++i)
        if (check_reduce.at(i) == false)
          output_shape.dim(j++) = i;
    }

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleMul *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CirclePack *node) final
  {
    LUCI_ASSERT(node->values_count() > 0, "Only support one or more inputs");

    auto first_shape = loco::shape_get(node->values(0)).as<loco::TensorShape>();
    // Make sure all inputs have the same shape.
    for (uint32_t i = 1; i < node->values_count(); ++i)
    {
      auto in_shape = loco::shape_get(node->values(i)).as<loco::TensorShape>();
      LUCI_ASSERT(loco::NodeShape{first_shape} == loco::NodeShape{in_shape},
                  "All inputs must have the same shape");
    }

    // Checking shape capability for pack layer
    // Input: tensors [D1, D2, ... Dn]
    // Axis: K
    // Output: [D1, D2, ... , D_K-1, n, D_K+1, ... Dn]
    auto axis = node->axis();
    if (axis < 0)
      axis += first_shape.rank() + 1;

    LUCI_ASSERT(0 <= axis, "Axis is out of range");
    LUCI_ASSERT(static_cast<uint32_t>(axis) <= first_shape.rank(), "Axis is out of range");

    loco::TensorShape output_shape;
    output_shape.rank(first_shape.rank() + 1);

    uint32_t j = 0;
    for (uint32_t i = 0; i < output_shape.rank(); ++i)
    {
      if (i == static_cast<uint32_t>(axis))
      {
        output_shape.dim(i) = node->values_count();
      }
      else
      {
        output_shape.dim(i) = first_shape.dim(j++);
      }
    }

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CirclePad *node) final
  {
    const loco::DataType S32 = loco::DataType::S32;

    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();
    auto paddings = dynamic_cast<luci::CircleConst *>(node->paddings());

    // TODO support non-const case
    LUCI_ASSERT(paddings, "Only support constant reduction_indices");
    // TODO support other data type
    LUCI_ASSERT(paddings->dtype() == S32, "Only support int 32 for now");
    LUCI_ASSERT(paddings->rank() == 2, "paddings should be rank 2")

    int32_t n = paddings->dim(0).value();
    int32_t v = paddings->dim(1).value();

    LUCI_ASSERT(v == 2, "paddings should be [n, 2]");
    LUCI_ASSERT(n == int32_t(input_shape.rank()),
                "paddings [n, 2] should have same value of input rank");

    loco::TensorShape output_shape;

    output_shape.rank(input_shape.rank());
    for (int32_t ni = 0; ni < n; ++ni)
    {
      int32_t idx = ni * 2;
      int value = input_shape.dim(ni).value();
      value += paddings->at<S32>(idx + 0); // left
      value += paddings->at<S32>(idx + 1); // right
      output_shape.dim(ni) = value;
    }

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleRelu *node) final
  {
    auto input_shape = loco::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleRelu6 *node) final
  {
    auto input_shape = loco::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  /**
   * @note  CircleReshape has new shape info in two places: 2nd input and attribute.
   *        This shape inference forces both to exist, and match each other.
   *        When this condition satisfied, it return the inferred shape
   *
   * TODO Change this policy when not appropriate
   */
  loco::NodeShape visit(const luci::CircleReshape *node) final
  {
    const loco::DataType S32 = loco::DataType::S32;

    loco::TensorShape shape_by_input;
    {
      LUCI_ASSERT(node->shape(), "2nd input shape() should not be nullptr");

      // Only support node's shape() is CircleConst with S32
      // TODO support other node with other types
      auto const_shape_node = dynamic_cast<luci::CircleConst *>(node->shape());
      LUCI_ASSERT(const_shape_node, "Only support CircleConst for shape of CircleReshape");
      LUCI_ASSERT(const_shape_node->dtype() == S32, "Only support int32 CircleConst");

      if (const_shape_node->rank() != 1)
        INTERNAL_EXN_V("Only support rank 1 CircleConst",
                       oops::to_uint32(const_shape_node->rank()));

      shape_by_input.rank(const_shape_node->dim(0).value());

      for (uint32_t axis = 0; axis < shape_by_input.rank(); ++axis)
      {
        shape_by_input.dim(axis) = const_shape_node->at<S32>(axis);
      }
    }

    loco::TensorShape shape_by_attr;
    {
      shape_by_attr.rank(node->newShape()->rank());

      for (uint32_t axis = 0; axis < shape_by_attr.rank(); ++axis)
      {
        shape_by_attr.dim(axis) = node->newShape()->dim(axis);
      }
    }

    LUCI_ASSERT(shape_by_input == shape_by_attr,
                "Warning: Two new shape information mismatched for CircleReshape");

    loco::TensorShape output_shape = shape_by_input;

    // One of the dimensions can have special value -1, meaning its actual value should be inferred.
    const auto input_shape = loco::shape_get(node->tensor()).as<loco::TensorShape>();
    const uint32_t input_element_count = loco::element_count(&input_shape);
    uint32_t output_element_count = 1;
    uint32_t unknown_dim_index = UINT32_MAX;
    for (uint32_t dim_index = 0; dim_index < output_shape.rank(); ++dim_index)
    {
      const uint32_t dim_value = output_shape.dim(dim_index).value();
      if (static_cast<int>(dim_value) == -1)
      {
        LUCI_ASSERT(unknown_dim_index == UINT32_MAX, "More than one unknown dimension");
        unknown_dim_index = dim_index;
      }
      else
      {
        output_element_count *= dim_value;
      }
    }
    if (unknown_dim_index != UINT32_MAX)
    {
      output_shape.dim(unknown_dim_index) = input_element_count / output_element_count;
    }

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleRsqrt *node) final
  {
    auto input_shape = loco::shape_get(node->x()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleSoftmax *node) final
  {
    auto input_shape = loco::shape_get(node->logits()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleSqrt *node) final
  {
    auto input_shape = loco::shape_get(node->x()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleSquaredDifference *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleSub *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  // TODO CircleTanh

  /// @brief Returns output shape of transpose. Use loco::ConstGen and luci::CircleConst for ConstT.
  template <class ConstT>
  loco::TensorShape output_shape_of_transpose(loco::TensorShape input_shape,
                                              const ConstT *perm_node)
  {
    loco::TensorShape output_shape;
    output_shape.rank(input_shape.rank());

    assert(perm_node->dtype() == loco::DataType::S32);
    assert(input_shape.rank() == perm_node->template size<loco::DataType::S32>());

    for (uint32_t out_axis = 0; out_axis < output_shape.rank(); out_axis++)
    {
      auto in_axis = perm_node->template at<loco::DataType::S32>(out_axis);
      output_shape.dim(out_axis) = input_shape.dim(in_axis);
    }

    return output_shape;
  }

  loco::NodeShape visit(const luci::CircleTranspose *node) final
  {
    auto input_shape = loco::shape_get(node->a()).as<loco::TensorShape>();

    auto canon_perm = dynamic_cast<loco::ConstGen *>(node->perm());
    auto circle_perm = dynamic_cast<luci::CircleConst *>(node->perm());

    if (canon_perm)
    {
      return loco::NodeShape{output_shape_of_transpose(input_shape, canon_perm)};
    }
    else if (circle_perm)
    {
      return loco::NodeShape{output_shape_of_transpose(input_shape, circle_perm)};
    }
    else
      INTERNAL_EXN("perm of CircleTranspose should be either ConstGen or CircleConst");
  }

  loco::NodeShape visit(const luci::CircleTransposeConv *node) final
  {
    // TransposeConv's output shape is written in its 'inputSizes' argument
    auto input_sizes_const = dynamic_cast<luci::CircleConst *>(node->inputSizes());
    LUCI_ASSERT(input_sizes_const,
                "Only support when CircleTransposeConv's inputSizes is CircleConst")
    LUCI_ASSERT(input_sizes_const->dtype() == loco::DataType::S32, "Only support S32 dtype")
    LUCI_ASSERT(input_sizes_const->rank() == 1 && input_sizes_const->dim(0).value() == 4,
                "Only support rank 1 with 4 entries")

    loco::TensorShape shape;

    shape.rank(4);
    for (uint32_t axis = 0; axis < 4; ++axis)
      shape.dim(axis) = input_sizes_const->at<loco::DataType::S32>(axis);

    return loco::NodeShape{shape};
  }

  // Circle Only
  loco::NodeShape visit(const luci::CircleInstanceNorm *node) final
  {
    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  // Virtual
  loco::NodeShape visit(const luci::CircleInput *node) final
  {
    loco::TensorShape shape;

    shape.rank(node->rank());
    for (uint32_t axis = 0; axis < node->rank(); axis++)
      shape.dim(axis) = node->dim(axis);

    return loco::NodeShape{shape};
  }

  loco::NodeShape visit(const luci::CircleOutput *node) final
  {
    auto from_shape = loco::shape_get(node->from()).as<loco::TensorShape>();

    return loco::NodeShape{from_shape};
  }
};

} // namespace

namespace luci
{

bool CircleShapeInferenceRule::recognize(const loco::Dialect *d) const
{
  return CircleDialect::get() == d;
}

bool CircleShapeInferenceRule::infer(const loco::Node *node, loco::NodeShape &shape) const
{
  assert(node->dialect() == CircleDialect::get());
  assert(dynamic_cast<const CircleNode *>(node) != nullptr);

  ShapeInferenceAlgorithm alg;
  shape = dynamic_cast<const CircleNode *>(node)->accept(&alg);

  return true;
}

} // namespace luci

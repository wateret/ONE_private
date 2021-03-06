/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_IR_SHAPE_H__
#define __ONERT_IR_SHAPE_H__

#include "ir/Layout.h"
#include "misc/feature/Shape.h"

#include <cstdint>
#include <vector>

namespace onert
{
namespace ir
{

// TODO Remove this dependency.
using FeatureShape = nnfw::misc::feature::Shape;

struct Shape
{
public:
  Shape() = default;

  explicit Shape(int rank) : _dimensions(rank) {}

  Shape(std::initializer_list<int32_t> dimensions) : _dimensions(dimensions) {}

  int rank() const { return _dimensions.size(); }

  const std::vector<int32_t> &dims() const { return _dimensions; }

  int32_t dim(int i) const { return _dimensions.at(i); }

  int32_t &dim(int i) { return _dimensions.at(i); }

  uint64_t num_elements() const;

public:
  FeatureShape asFeature(Layout layout) const;

  /**
   * @brief Add dimension to the beginning
   * @param[in] d dimension to add to the beginning
   */
  void prepend(int32_t d) { _dimensions.insert(_dimensions.cbegin(), d); }

  /**
   * @brief Add dimension to the end
   * @param[in] d dimension to add to the end
   */
  void append(int32_t d) { _dimensions.emplace_back(d); }

  /**
   * @brief Extend rank of Shape object for operand with param.
   * @param[in] to_rank The rank value to be extended to
   */
  void extendRank(int to_rank);

private:
  std::vector<int32_t> _dimensions;
};

inline bool operator==(const Shape &lhs, const Shape &rhs) { return lhs.dims() == rhs.dims(); }
inline bool operator!=(const Shape &lhs, const Shape &rhs) { return lhs.dims() != rhs.dims(); }

Shape permuteShape(const Shape &shape, Layout frontend_layout, Layout backend_layout);

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_SHAPE_H__

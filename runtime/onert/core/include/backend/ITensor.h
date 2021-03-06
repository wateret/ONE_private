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

#ifndef __ONERT_BACKEND_OPERAND_I_TENSOR_H__
#define __ONERT_BACKEND_OPERAND_I_TENSOR_H__

#include <cstring>
#include <cstdint>
#include <functional>

#include "ir/Layout.h"
#include "ir/Coordinates.h"

namespace onert
{
namespace backend
{

class ITensor
{
public:
  virtual ~ITensor() = default;

public:
  virtual uint8_t *buffer() const = 0;
  virtual size_t total_size() const = 0;
  virtual size_t dimension(size_t index) const = 0;
  virtual size_t num_dimensions() const = 0;
  virtual size_t calcOffset(const ir::Coordinates &coords) const = 0;
  virtual ir::Layout layout() const = 0;
  virtual bool has_padding() const = 0;
  virtual void access(const std::function<void(ITensor &tensor)> &fn) = 0;

  /**
   * @brief Return true if the tensor needs dynamic allocation, meaning that during compile-time
   *        the outpus shape cannot be known and the output shape is calculated during
   *        kernel execution-time.
   */
  virtual bool is_dynamic() const { return false; /* default */ }
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_OPERAND_I_TENSOR_H__

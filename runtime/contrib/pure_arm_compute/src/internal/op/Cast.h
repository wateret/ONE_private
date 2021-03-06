/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file Cast.h
 * @brief This file contains accept function and params for Cast operation
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __INTERNAL_OP_CAST_H__
#define __INTERNAL_OP_CAST_H__

#include "internal/op/Node.h"

#include <cstdint>

namespace internal
{
namespace tflite
{
namespace op
{
namespace Cast
{

/**
 * @brief Struct of Cast operation's param
 */
struct Param
{
  int32_t output_index; /**< Output index */

  int32_t input_index; /**< Input index */

  /**
   * @brief Construct a new Param object for Cast as default
   */
  Param() = default;

  /**
   * @brief Construct a new Param object for Cast with params
   * @param [in] inputCount The number of input
   * @param [in] inputs Array containing inputs
   * @param [in] outputCount The number of output
   * @param [in] outputs Array containing outputs
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define operation node for Cast
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new Node object for Cast with param
   * @param [in] param Parameters for Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Destroy the Node object for Cast
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameters for Cast
   * @return Parameters of Cast
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Function for accepting node for Cast
   * @param [in] v Node visitor for invoking visit function of Cast
   * @return N/A
   */
  void accept(NodeVisitor &&) const override;

private:
  const Param _param;
};

} // namespace Cast
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_Cast_H__

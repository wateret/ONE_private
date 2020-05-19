/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SubLayer.h"

#include <cker/operation/BinaryArithmeticOps.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

void SubLayer::subFloat32()
{
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(_activation, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam op_params;
  op_params.type = nnfw::cker::BinaryArithmeticOpType::SUB;
  op_params.float_activation_max = output_activation_max;
  op_params.float_activation_min = output_activation_min;

  if (!HaveSameShapes(_lhs, _rhs))
  {
    nnfw::cker::BroadcastBinaryArithmeticOpSlow(
        op_params, convertToExtendedCkerShape(_lhs),
        reinterpret_cast<const float *>(_lhs->buffer()), convertToExtendedCkerShape(_rhs),
        reinterpret_cast<const float *>(_rhs->buffer()), convertToExtendedCkerShape(_output),
        reinterpret_cast<float *>(_output->buffer()));
    return;
  }

  nnfw::cker::BinaryArithmeticOp(
      op_params, convertTensorToCkerShape(_lhs), reinterpret_cast<const float *>(_lhs->buffer()),
      convertTensorToCkerShape(_rhs), reinterpret_cast<const float *>(_rhs->buffer()),
      convertTensorToCkerShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void SubLayer::subQuant8()
{
  int32_t output_activation_min, output_activation_max;
  CalculateActivationRangeUint8(_activation, _output, &output_activation_min,
                                &output_activation_max);
  // nnfw::cker::SubParam op_params;
  // op_params.quantized_activation_max = output_activation_max;
  // op_params.quantized_activation_min = output_activation_min;

  // cker quant8 sub is not implemented yet
  throw std::runtime_error{"NYI"};
}

void SubLayer::configure(const ITensor *lhs, const ITensor *rhs,
                         const ir::Activation activation, ITensor *output)
{
  _lhs = lhs;
  _rhs = rhs;
  _activation = activation;
  _output = output;
}

void SubLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    subFloat32();
  }
  else if (_output->data_type() == OperandType::QUANT8_ASYMM)
  {
    subQuant8();
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

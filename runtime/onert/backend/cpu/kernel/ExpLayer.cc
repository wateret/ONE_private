/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExpLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Exp.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

ExpLayer::ExpLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void ExpLayer::expFloat32()
{
  nnfw::cker::Exp(convertTensorToCkerShape(_input),
                  reinterpret_cast<const float *>(_input->buffer()),
                  convertTensorToCkerShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void ExpLayer::expQuant8()
{
  // cker quant8 exp is not implemented yet
  throw std::runtime_error{"NYI"};
}

void ExpLayer::configure(const ITensor *input, ITensor *output)
{
  _input = input;
  _output = output;
}

void ExpLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    expFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT8_ASYMM)
  {
    expQuant8();
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

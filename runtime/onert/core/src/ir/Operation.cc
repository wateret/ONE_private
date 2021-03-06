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

#include "ir/Operation.h"

#include <cassert>

namespace onert
{
namespace ir
{

Operation::Operation(OperandConstraint input_constr, const OperandIndexSequence &inputs,
                     const OperandIndexSequence &outputs)
    : _input_constr{input_constr}, _inputs{inputs}, _outputs{outputs}
{
}

Operation::Operation(OperandConstraint input_constr) : _input_constr{input_constr} {}

Operation::~Operation() = default;

void Operation::setInputs(const OperandIndexSequence &indexes)
{
  assert(_input_constr.check(indexes.size()));
  _inputs = indexes;
}

void Operation::setOutputs(const OperandIndexSequence &indexes) { _outputs = indexes; }

void Operation::replaceInput(const OperandIndex &from, const OperandIndex &to)
{
  _inputs.replace(from, to);
}

void Operation::replaceOutput(const OperandIndex &from, const OperandIndex &to)
{
  _outputs.replace(from, to);
}

} // namespace ir
} // namespace onert

/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exec/Execution.h"

#include "util/logging.h"

namespace onert
{
namespace exec
{

Execution::Execution(const std::shared_ptr<IExecutor> &executor) : _executor{executor}
{
  _io_desc.inputs.resize(_executor->graph().getInputs().size());
  _io_desc.outputs.resize(_executor->graph().getOutputs().size());
}

// TODO Remove default parameter
void Execution::setInput(const ir::IOIndex &index, const void *buffer, size_t length,
                         ir::Layout layout)
{
  const auto input_index = graph().getInputs().at(index);
  const auto info = graph().operands().at(input_index).info();

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _io_desc.inputs.at(index.value()) = std::make_unique<InputDesc>(info, buffer, length, layout);
}

// TODO Remove default parameter
void Execution::setInput(const ir::IOIndex &index, const ir::TypeInfo &type, const ir::Shape &shape,
                         const void *buffer, size_t length, ir::Layout layout)
{
  const ir::OperandInfo info{shape, type};

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _io_desc.inputs.at(index.value()) = std::make_unique<InputDesc>(info, buffer, length, layout);
}

// TODO Remove default parameter
void Execution::setOutput(const ir::IOIndex &index, void *buffer, size_t length, ir::Layout layout)
{
  const auto output_index = graph().getOutputs().at(index);
  const auto info = graph().operands().at(output_index).info();

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _io_desc.outputs.at(index.value()) = std::make_unique<OutputDesc>(info, buffer, length, layout);
}

// TODO Remove default parameter
void Execution::setOutput(const ir::IOIndex &index, const ir::TypeInfo &type,
                          const ir::Shape &shape, void *buffer, size_t length, ir::Layout layout)
{
  const ir::OperandInfo info{shape, type};

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _io_desc.outputs.at(index.value()) = std::make_unique<OutputDesc>(info, buffer, length, layout);
}

void Execution::setInputLayout(const ir::IOIndex &index, ir::Layout layout)
{
  const auto &input_desc = _io_desc.inputs.at(index.value());
  _io_desc.inputs.at(index.value()) =
      std::make_unique<InputDesc>(input_desc->info, input_desc->buffer, input_desc->size, layout);
}

void Execution::setOutputLayout(const ir::IOIndex &index, ir::Layout layout)
{
  const auto &output_desc = _io_desc.outputs.at(index.value());
  _io_desc.outputs.at(index.value()) = std::make_unique<OutputDesc>(
      output_desc->info, output_desc->buffer, output_desc->size, layout);
}

void Execution::execute()
{
  VERBOSE(Execution) << "Start execution" << std::endl;

  _executor->execute(_io_desc);
  finished = true;

  VERBOSE(Execution) << "Execution finished" << std::endl;
}

void Execution::startExecute()
{
  VERBOSE(Execution) << "Create asynchronous execution thread" << std::endl;

  _exec_thread = std::make_unique<std::thread>(&Execution::execute, this);
}

void Execution::waitFinish()
{
  VERBOSE(Execution) << "Wait to finish execution" << std::endl;

  _exec_thread->join();
  finished = true;
}

bool Execution::isFinished(void) const { return finished; }

} // namespace exec
} // namespace onert

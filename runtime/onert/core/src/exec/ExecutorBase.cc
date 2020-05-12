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

#include "ExecutorBase.h"
#include "util/logging.h"

namespace onert
{
namespace exec
{

ExecutorBase::ExecutorBase(std::unique_ptr<ir::LoweredGraph> &&lowered_graph,
                           const backend::TensorBuilderSet &tensor_builders)
    : _lowered_graph{std::move(lowered_graph)}, _graph{_lowered_graph->graph()}, _mutex()
{
  auto build_input_tensor_list = [&](const onert::ir::OperandIndexSequence &ind_seq) {
    std::vector<std::shared_ptr<backend::ITensor>> list;
    for (auto ind : ind_seq)
    {
      std::shared_ptr<backend::ITensor> tensor;
      for (auto &tensor_builder : tensor_builders)
      {
        tensor = tensor_builder->tensorAt(ind);
        if (tensor != nullptr)
        {
          if (tensor_builder->supportDynamicTensor())
          {
            DynAllocInfo dyn_alloc_info{ind, tensor_builder->dynamicTensorManager()};
            _input_to_dyn_alloc_info.emplace(tensor, dyn_alloc_info);
          }
          break;
        }
      }
      assert(tensor != nullptr);
      list.push_back(tensor);
    }
    return list;
  };

  auto build_output_tensor_list = [&](const onert::ir::OperandIndexSequence &ind_seq) {
    std::vector<std::shared_ptr<backend::ITensor>> list;
    for (auto ind : ind_seq)
    {
      std::shared_ptr<backend::ITensor> tensor;
      for (auto &tensor_builder : tensor_builders)
      {
        tensor = tensor_builder->tensorAt(ind);
        if (tensor != nullptr)
          break;
      }
      assert(tensor != nullptr);
      list.push_back(tensor);
    }
    return list;
  };

  _input_tensors = build_input_tensor_list(_graph.getInputs());
  _output_tensors = build_output_tensor_list(_graph.getOutputs());

  // Prepare each TensorManager on each backend
  for (auto &tensor_builder : tensor_builders)
  {
    auto s_tensor_manager = tensor_builder->releaseStaticTensorManager();
    if (s_tensor_manager != nullptr)
      _tensor_mgrs.insert(std::move(s_tensor_manager));

    if (tensor_builder->supportDynamicTensor())
    {
      auto d_tensor_manager = tensor_builder->releaseDynamicTensorManager();
      if (d_tensor_manager != nullptr)
        _tensor_mgrs.insert(std::move(d_tensor_manager));
    }
  }
}

std::unique_ptr<ISource> ExecutorBase::source(const ir::IOIndex &index, const ir::TypeInfo &type,
                                              const void *buffer, size_t length,
                                              ir::Layout io_layout)
{
  using ir::DataType;
  switch (type.type())
  {
    case DataType::FLOAT32:
      return source<float>(index, buffer, length, io_layout);
    case DataType::INT32:
      return source<int32_t>(index, buffer, length, io_layout);
    case DataType::UINT32:
      return source<uint32_t>(index, buffer, length, io_layout);
    case DataType::BOOL8:
    case DataType::QUANT8_ASYMM:
    case DataType::UINT8:
      return source<uint8_t>(index, buffer, length, io_layout);
    case DataType::QUANT8_SYMM:
      return source<int8_t>(index, buffer, length, io_layout);
    default:
      throw std::runtime_error("Not supported yet");
  }
}

std::unique_ptr<ISink> ExecutorBase::sink(const ir::IOIndex &index, const ir::TypeInfo &type,
                                          void *buffer, size_t length, ir::Layout io_layout)
{
  using ir::DataType;
  switch (type.type())
  {
    case DataType::FLOAT32:
      return sink<float>(index, buffer, length, io_layout);
    case DataType::INT32:
      return sink<int32_t>(index, buffer, length, io_layout);
    case DataType::UINT32:
      return sink<uint32_t>(index, buffer, length, io_layout);
    case DataType::BOOL8:
    case DataType::QUANT8_ASYMM:
    case DataType::UINT8:
      return sink<uint8_t>(index, buffer, length, io_layout);
    case DataType::QUANT8_SYMM:
      return sink<int8_t>(index, buffer, length, io_layout);
    default:
      throw std::runtime_error("Not supported yet");
  }
}

void ExecutorBase::execute(const IODescription &desc)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  std::lock_guard<std::mutex> lock(_mutex);

  assert(_input_tensors.size() == desc.inputs.size());
  for (uint32_t i = 0; i < _input_tensors.size(); ++i)
  {
    _input_tensors[i]->buffer(static_cast<uint8_t *>(const_cast<void *>(desc.inputs[i]->buffer)),
                              desc.inputs[i]->size);
  }

  assert(_output_tensors.size() == desc.outputs.size());
  for (uint32_t i = 0; i < _output_tensors.size(); ++i)
  {
    _output_tensors[i]->buffer(static_cast<uint8_t *>(const_cast<void *>(desc.outputs[i]->buffer)),
                               desc.outputs[i]->size);
    // user_tensor->setBufferSize(setdesc.inputs[i]->size); // Introduce setBufferSize and do this?
  }

  executeImpl();
}

} // namespace exec
} // namespace onert

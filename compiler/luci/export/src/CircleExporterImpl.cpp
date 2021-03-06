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

#include "CircleExporterImpl.h"
#include "Optimize.h"
#include "CircleTensorExporter.h"
#include "CircleOperationExporter.h"
#include "CircleExporterUtils.h"

#include <oops/InternalExn.h>
#include <mio/circle/schema_generated.h>
#include <flatbuffers/flatbuffers.h>

#include <cassert>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace
{

luci::CircleInput *input_node(loco::Graph *g, const loco::GraphInputIndex &index)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto pull = dynamic_cast<luci::CircleInput *>(g->nodes()->at(n)))
    {
      if (pull->indexed() && pull->index() == index)
      {
        return pull;
      }
    }
  }
  return nullptr;
}

luci::CircleOutput *output_node(loco::Graph *g, const loco::GraphOutputIndex &index)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto push = dynamic_cast<luci::CircleOutput *>(g->nodes()->at(n)))
    {
      if (push->indexed() && push->index() == index)
      {
        return push;
      }
    }
  }
  return nullptr;
}

void registerGraphInputTensors(loco::Graph *graph, luci::SubGraphContext &ctx)
{
  for (uint32_t n = 0; n < graph->inputs()->size(); ++n)
  {
    auto node = input_node(graph, n);
    assert(node != nullptr);
    ctx._inputs.push_back(luci::get_tensor_index(node));
  }
}

void registerGraphOutputTensors(loco::Graph *graph, luci::SubGraphContext &ctx)
{
  for (uint32_t n = 0; n < graph->outputs()->size(); ++n)
  {
    auto push = output_node(graph, n);
    assert(push != nullptr);
    auto node = push->from();
    assert(node != nullptr);
    ctx._outputs.push_back(luci::get_tensor_index(node));
  }
}

} // namespace

namespace
{

using namespace circle;
using namespace flatbuffers;

Offset<Vector<Offset<OperatorCode>>>
encodeOperatorCodes(FlatBufferBuilder &builder, std::unordered_map<luci::OpCode, uint32_t> &opcodes,
                    std::unordered_map<luci::OpCode, std::string> &custom_opcodes)
{
  std::vector<Offset<OperatorCode>> operator_codes_vec(opcodes.size());
  for (auto it : opcodes)
  {
    uint32_t idx = it.second;
    if (it.first.opcode != BuiltinOperator_CUSTOM)
    {
      operator_codes_vec[idx] = CreateOperatorCode(builder, it.first.opcode);
    }
    else // custom op
    {
      auto opCode = it.first;
      auto custom_code = custom_opcodes.find(opCode);
      if (custom_code == custom_opcodes.end())
        INTERNAL_EXN("Cannot find code for customop even though opcode is BuiltinOperator_CUSTOM");

      operator_codes_vec[idx] =
          CreateOperatorCode(builder, it.first.opcode, builder.CreateString(custom_code->second));
    }
  }
  return builder.CreateVector(operator_codes_vec);
}

} // namespace

namespace luci
{

using namespace circle;
using namespace flatbuffers;

CircleExporterImpl::CircleExporterImpl(loco::Graph *graph) { exportGraph(graph); }

::flatbuffers::Offset<::circle::SubGraph>
CircleExporterImpl::exportSubgraph(SerializedModelData &gd)
{
  auto tensors = _builder.CreateVector(gd._tensors);
  auto inputs = _builder.CreateVector(gd._inputs);
  auto outputs = _builder.CreateVector(gd._outputs);
  auto operators = _builder.CreateVector(gd._operators);
  auto df = gd._data_format;
  auto subgraph = CreateSubGraph(_builder, tensors, inputs, outputs, operators, df);
  return subgraph;
}

void CircleExporterImpl::exportGraph(loco::Graph *graph)
{
  // do graph optimization
  optimize(graph);

  _builder.Clear();

  SerializedModelData gd;

  // This version is taken from comment in fbs
  constexpr uint32_t version = 0;

  registerGraphIOName(graph, gd);

  // parse graph into SerializedModelData structure
  exportOpDefinedTensors(graph, _builder, gd);

  // NOTE Invoke these register functions only after each node is annotated with its tensor_index
  registerGraphInputTensors(graph, gd);
  registerGraphOutputTensors(graph, gd);

  exportNodes(graph, _builder, gd);

  // encode operator codes
  auto operator_codes =
      encodeOperatorCodes(_builder, gd._operator_codes, gd._custom_operator_codes);

  // Subgraphs
  Offset<SubGraph> subgraph = exportSubgraph(gd);
  auto subgraphs = _builder.CreateVector(std::vector<Offset<SubGraph>>{subgraph});

  // Description
  std::string description_str = "nnpackage";
  auto description = _builder.CreateString(description_str);

  // create array of buffers
  auto buffers = _builder.CreateVector(gd._buffers);

  // empty metadata
  std::vector<int> metadata_buffer_vec;
  auto metadata_buffer = _builder.CreateVector(metadata_buffer_vec);

  // Model
  auto model_offset = CreateModel(_builder, version, operator_codes, subgraphs, description,
                                  buffers, metadata_buffer);
  FinishModelBuffer(_builder, model_offset);
}

const char *CircleExporterImpl::getBufferPointer() const
{
  return reinterpret_cast<const char *>(_builder.GetBufferPointer());
}

size_t CircleExporterImpl::getBufferSize() const { return _builder.GetSize(); }

} // namespace luci

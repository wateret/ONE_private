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

#include "ir/LoweredGraph.h"

#include <assert.h>
#include <sstream>
#include "util/logging.h"
#include "pass/ConstantInsertionPass.h"
#include "pass/PermutationOperationPass.h"
#include "pass/PermutationInsertionPass.h"
#include "ir/GraphIterator.h"
#include "verifier/Verifier.h"
#include "backend/Backend.h"
#include "backend/IConfig.h"
#include "compiler/BackendResolver.h"
#include "compiler/ManualScheduler.h"
#include "compiler/HEScheduler.h"

namespace onert
{
namespace ir
{

LoweredGraph::LoweredGraph(const Graph &graph, const compiler::CompilerOptions &options)
    : _graph{graph}
{
  // Build backend contexts
  for (auto backend_str : options.backend_list)
  {
    auto &backend_manager = compiler::BackendManager::get();
    backend_manager.loadBackend(backend_str);
    auto backend = backend_manager.get(backend_str);

    if (!backend)
      throw std::runtime_error("Cannot get required backend");

    _backend_contexts.emplace(backend, backend->newContext(_graph, _graph.getKernelBuilder(),
                                                           options.executor == "Linear"));
  }

  // TODO Move "schedule" phase out of here
  // Schedule
  if (options.he_scheduler)
  {
    auto scheduler = compiler::HEScheduler(_backend_contexts, options);
    _backend_resolver = scheduler.schedule(_graph);
    _indexed_ranks = scheduler.getIndexedRanks();
  }
  else
  {
    auto scheduler = compiler::ManualScheduler(options.manual_scheduler_options);
    _backend_resolver = scheduler.schedule(_graph);
  }

  {
    // operand::LowerInfo holder
    OperandIndexMap<std::unique_ptr<operand::LowerInfo>> operands_lower_info;

    _graph.operands().iterate([&](const OperandIndex &index, const Operand &) {
      operands_lower_info[index] = std::make_unique<operand::LowerInfo>();
    });

    // Make op_seqs while checking whether a node can be merged into a op_seq.
    makeOpSequences(operands_lower_info, options);

    _op_seqs.iterate([&](const OpSequenceIndex &, OpSequence &op_seq) {
      assert(op_seq.operations().size() > 0);
      std::reverse(std::begin(op_seq.operations()), std::end(op_seq.operations()));
    });

    _op_seqs.dump("merged and sorted operations without permutation");

    pass::ConstantInsertionPass ci_pass(*this);
    ci_pass.run();

    // Set LowerInfo for each operand from the operand::LowerInfo holder
    manipulateLowerInfo(operands_lower_info);

    dumpLowerInfo();
  }

  // Run Permutation Passes
  {
    pass::PermutationOperationPass po_pass(*this);
    po_pass.run();

    pass::PermutationInsertionPass pi_pass(*this);
    pi_pass.run();
    // Implemented code no longer works.
    // pass::PermutationEliminationPass pe_pass(*this);
    // pe_pass.run();

    // TODO merge perm op_seqs if possible
    _op_seqs.dump("merged and sorted operations with permutation");
  }

  // Graph verifications
  {
    assert(verifier::DAGChecker().verify(_graph));
    assert(verifier::EdgeConsistencyChecker().verify(_graph));
  }
}

const operation::LowerInfo *LoweredGraph::getLowerInfo(const OpSequenceIndex &op_seq_index) const
{
  auto itr = _lower_info_map.operation.find(op_seq_index);
  if (itr == _lower_info_map.operation.end())
    return nullptr;
  return itr->second.get();
}

void LoweredGraph::setLowerInfo(const OpSequenceIndex &op_seq_index,
                                std::unique_ptr<operation::LowerInfo> &&lower_info)
{
  _lower_info_map.operation.insert(std::make_pair(op_seq_index, std::move(lower_info)));
}

void LoweredGraph::removeLowerInfo(const OpSequenceIndex &op_seq_index)
{
  auto &op_seq_lower_info = _lower_info_map.operation;
  assert(op_seq_lower_info.find(op_seq_index) != op_seq_lower_info.end());
  for (auto it = op_seq_lower_info.begin(); it != op_seq_lower_info.end(); ++it)
  {
    if (it->first == op_seq_index)
    {
      op_seq_lower_info.erase(it);
      break;
    }
  }
}

const operand::LowerInfo *LoweredGraph::getLowerInfo(const OperandIndex &index) const
{
  auto itr = _lower_info_map.operand.find(index);
  if (itr == _lower_info_map.operand.end())
    return nullptr;
  return itr->second.get();
}

operand::LowerInfo *LoweredGraph::getLowerInfo(const OperandIndex &index)
{
  auto itr = _lower_info_map.operand.find(index);
  if (itr == _lower_info_map.operand.end())
    return nullptr;
  return itr->second.get();
}

void LoweredGraph::setLowerInfo(const OperandIndex &index,
                                std::unique_ptr<operand::LowerInfo> &&lower_info)
{
  _lower_info_map.operand.insert(std::make_pair(index, std::move(lower_info)));
}

void LoweredGraph::removeLowerInfo(const OperandIndex &index)
{
  _lower_info_map.operand.erase(index);
}

OpSequenceIndex LoweredGraph::appendFreshSingleOpSequence(const OperationIndex &node_index,
                                                          const Operation &node, Layout layout)
{
  // Create a fresh op_seq with one operation, and append it to op_seqs
  // Create a fresh op_seq
  auto op_seq = std::make_unique<OpSequence>(layout);

  // Add an operation
  op_seq->appendOperation(node_index, node);

  // Update input/output
  op_seq->setOutputs(node.getOutputs());
  op_seq->setInputs(node.getInputs());

  return _op_seqs.emplace(std::move(op_seq));
}

void LoweredGraph::makeOpSequences(
    OperandIndexMap<std::unique_ptr<operand::LowerInfo>> &operands_lower_info,
    const compiler::CompilerOptions &options)
{
  // if SUBG_MAX_NODE == 0, no limit on nodes of a op_seq
  const int op_seq_max_node = options.op_seq_max_node;
  assert(op_seq_max_node >= 0);

  bool is_profiling = options.he_profiling_mode;
  OpSequence *op_seq = nullptr;
  OpSequenceIndex op_seq_index;

  // NOTE: The below method appends nodes while making one op_seq if needed. If something better
  // ways, happy to update this code.
  PostDfsConstIterator{}.iterate(
      _graph, [&](const OperationIndex &node_index, const Operation &node) {
        // LowerInfo for in/output operands
        auto backend = _backend_resolver->getBackend(node_index);

        // TODO How to get frontend layout of this node from IR
        auto frontend_layout = Layout::NHWC;
        auto backend_layout = frontend_layout;

        // The layout of each backend should be set at another place
        // TODO Change setting layout of each backend at another place
        // TODO Remove getting id of backend
        if (backend->config()->id() == "acl_cl" || backend->config()->id() == "acl_neon")
        {
          const std::string acl_layout_str = util::getConfigString(util::config::ACL_LAYOUT);
          if (acl_layout_str == "NHWC")
          {
            backend_layout = Layout::NHWC;
          }
          else if (acl_layout_str == "NCHW")
          {
            backend_layout = Layout::NCHW;
          }
        }
        else if (backend->config()->id() == "srcn")
        {
          const std::string ncnn_layout_str = util::getConfigString(util::config::NCNN_LAYOUT);
          if (ncnn_layout_str == "NHWC")
          {
            backend_layout = Layout::NHWC;
          }
          else if (ncnn_layout_str == "NCHW")
          {
            backend_layout = Layout::NCHW;
          }
        }
        else if (backend->config()->id() == "cpu")
        {
          backend_layout = Layout::NHWC;
        }

        for (auto operand : node.getInputs())
        {
          auto &&lower_info = operands_lower_info.at(operand);
          lower_info->addUsePermuteFactor(operand::PermuteFactor{backend, backend_layout});
        }
        for (auto operand : node.getOutputs())
        {
          auto &&lower_info = operands_lower_info.at(operand);
          lower_info->addDefPermuteFactor(operand::PermuteFactor{backend, backend_layout});
        }

        bool new_op_seq = (op_seq == nullptr ||
                           (op_seq_max_node != 0 &&
                            op_seq->operations().size() >= static_cast<size_t>(op_seq_max_node)));

        // for profiling each op_seq must contain just one node,
        // so that we can measure a node separately
        if (new_op_seq || is_profiling || !mergeable(op_seq_index, node_index, backend_layout))
        {
          auto new_op_seq_index = appendFreshSingleOpSequence(node_index, node, frontend_layout);

          // OpSequence LowerInfo
          setLowerInfo(new_op_seq_index,
                       std::make_unique<operation::LowerInfo>(backend, backend_layout));

          op_seq_index = new_op_seq_index;
          op_seq = &(_op_seqs.at(new_op_seq_index));

          VERBOSE(Lower) << "SUBG#" << op_seq_index.value() << " is created for "
                         << "NODE#" << node_index.value() << "(" << node.name() << ")" << std::endl;
        }
        else
        {
          op_seq->appendOperation(node_index, node);
          op_seq->setInputs(node.getInputs());

          VERBOSE(Lower) << "SUBG#" << op_seq_index.value() << " merges "
                         << "NODE#" << node_index.value() << "(" << node.name() << ")" << std::endl;
        }
      });
}

void LoweredGraph::manipulateLowerInfo(
    OperandIndexMap<std::unique_ptr<operand::LowerInfo>> &operands_lower_info)
{
  const auto default_backend = compiler::BackendManager::get().getDefault();
  for (auto index : _graph.getInputs())
  {
    // Pick just any one from the uses, here the first one is chosen
    // For the other uses, Permute operations will be inserted later
    auto &&lower_info = operands_lower_info.at(index);
    assert(lower_info->use_factors().size() > 0);
    lower_info->addDefPermuteFactor(*lower_info->use_factors().begin());
  }
  for (auto index : _graph.getOutputs())
  {
    auto &&lower_info = operands_lower_info.at(index);
    if (_graph.operands().at(index).isConstant())
    {
      lower_info->addDefPermuteFactor(operand::PermuteFactor{
          default_backend,
          Layout::NHWC // TODO Get frontend layout of this node from IR
      });
    }
  }

  // Set LowerInfo for each operand from the operand::LowerInfo holder
  _graph.operands().iterate([&](const OperandIndex &index, Operand &) {
    setLowerInfo(index, std::move(operands_lower_info[index]));
  });
}

void LoweredGraph::dumpLowerInfo()
{
  if (::onert::util::logging::ctx.enabled() == false)
    return;

  std::map<uint32_t, std::string> dumps;

  _graph.operands().iterate([&](const OperandIndex &index, Operand &object) {
    std::stringstream sstream;
    if (!getLowerInfo(index)->def_factors().empty() || !getLowerInfo(index)->use_factors().empty())
    {
      auto factors_to_string = [](const operand::PermuteFactorSet &factors) {
        std::string str;
        for (auto factor : factors)
        {
          str += factor.backend()->config()->id();
          str += "(" + to_string(factor.layout()) + ")";
          str += " ";
        }
        return "{ " + str + "}";
      };

      auto operation_index_to_string = [](const OperationIndexList &operations) {
        std::string str;
        for (auto op : operations.list())
        {
          str += std::to_string(op.value());
          str += " ";
        }
        return "{ " + str + "}";
      };

      const auto lower_info = getLowerInfo(index);
      const auto &shape = object.shape();
      std::string def_ops = operation_index_to_string(object.getDef());
      std::string use_ops = operation_index_to_string(object.getUses());
      std::string def_layouts = factors_to_string(lower_info->def_factors());
      std::string use_layouts = factors_to_string(lower_info->use_factors());
      sstream << "Operand #" << index.value() << " LowerInfo" << std::endl;
      sstream << "  - Shape           : { " << (shape.rank() > 0 ? shape.dim(0) : 0) << " "
              << (shape.rank() > 1 ? shape.dim(1) : 0) << " "
              << (shape.rank() > 2 ? shape.dim(2) : 0) << " "
              << (shape.rank() > 3 ? shape.dim(3) : 0) << " "
              << "}" << std::endl;
      sstream << "  - Def Operations  : " << def_ops << std::endl;
      sstream << "  - Use Operations  : " << use_ops << std::endl;
      sstream << "  - Lower Info" << std::endl;
      sstream << "    - Def Backends    : " << def_layouts << std::endl;
      sstream << "    - Use Backends    : " << use_layouts << std::endl;
    }
    dumps.emplace(index.value(), sstream.str());
  });

  for (const auto &e : dumps)
  {
    if (!e.second.empty())
    {
      VERBOSE(Lower) << e.second;
    }
  }
}

bool LoweredGraph::mergeable(const OpSequenceIndex &op_seq_index, const OperationIndex &node_index,
                             Layout layout)
{
  // Are they mergeable?
  // 1. the same backend id and layout?
  // 2. Is op_seq or node branched?
  // 3. if 1 is true, the op_seq and a node are connected?
  const auto &op_seq = _op_seqs.at(op_seq_index);
  const auto &node = _graph.operations().at(node_index);

  // The same backend id and layout?
  {
    const auto op_seq_backend_layout = getLowerInfo(op_seq_index)->layout();
    const auto &op_seq_backend_id = getLowerInfo(op_seq_index)->backend()->config()->id();
    const auto &node_backend_id = _backend_resolver->getBackend(node_index)->config()->id();
    VERBOSE(Lower) << "SUBG#" << op_seq_index.value() << " { " << op_seq_backend_id << "("
                   << to_string(op_seq_backend_layout) << ") } "
                   << " NODE#" << node_index.value() << " (" << node.name() << ") { "
                   << node_backend_id << "(" << to_string(layout) << ") } " << std::endl;
    if (op_seq_backend_id != node_backend_id || op_seq_backend_layout != layout)
      return false;
  }

  // Branched?
  {
    std::unordered_set<OperationIndex> branched_set;

    // Check for branching up
    const auto &inputs = op_seq.getInputs();
    for (const auto &input : inputs)
    {
      const auto &input_obj = _graph.operands().at(input);
      for (const auto &def : input_obj.getDef().list())
      {
        branched_set.insert(def);
        if (branched_set.size() > 1)
        {
          return false;
        }
      }
    }
    branched_set.clear();

    // Check for branching down
    const auto &outputs = node.getOutputs();
    for (const auto &output : outputs)
    {
      const auto &output_obj = _graph.operands().at(output);
      for (const auto &use : output_obj.getUses().list())
      {
        branched_set.insert(use);
        if (branched_set.size() > 1)
        {
          return false;
        }
      }
    }
  }

  // Connected?
  // an input of one node is an output of the other node? or vice-versa?
  {
    const auto &node_inputs = node.getInputs();
    const auto &node_outputs = node.getOutputs();

    // op_seq's operations are in order so that we just check the first and the last
    std::vector<Element> op_seq_ops{op_seq.operations()[0]};
    if (op_seq.operations().size() > 1)
      op_seq_ops.emplace_back(op_seq.operations()[op_seq.operations().size() - 1]);

    for (const auto &elem : op_seq_ops)
    {
      const auto &n_index = elem.index;
      const auto &n = *elem.node;

      // node's output == op_seq's input?
      const auto &n_inputs = n.getInputs();
      for (auto input : n_inputs)
      {
        if (node_outputs.contains(input))
        {
          VERBOSE(Lower) << "SUBG#" << op_seq_index.value() << " 's NODE#" << n_index.value() << "("
                         << n.name() << ") is connected to NODE#" << node_index.value() << "("
                         << node.name() << ")" << std::endl;
          return true;
        }
      }

      // node's input == op_seq's output?
      const auto &n_outputs = n.getOutputs();
      for (auto output : n_outputs)
      {
        if (node_inputs.contains(output))
        {
          VERBOSE(Lower) << "SUBG#" << op_seq_index.value() << " 's NODE#" << n_index.value()
                         << " (" << n.name() << ") is connected to NODE#" << node_index.value()
                         << std::endl;
          return true;
        }
      }
    }

    VERBOSE(Lower) << "SUBG#" << op_seq_index.value() << " is not connected to NODE#"
                   << node_index.value() << "(" << node.name() << ")" << std::endl;
  }

  return false;
}

} // namespace ir
} // namespace onert

/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "OpPrinter.h"
#include "Read.h"

#include <stdex/Memory.h>

#include <flatbuffers/flexbuffers.h>

using stdex::make_unique;

namespace circledump
{

// TODO move to some header
std::ostream &operator<<(std::ostream &os, const std::vector<int32_t> &vect);

// TODO Re-arrange in alphabetical order

class AddPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_AddOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class ArgMaxPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_ArgMaxOptions())
    {
      os << "    ";
      os << "OutputType(" << EnumNameTensorType(params->output_type()) << ") ";
      os << std::endl;
    }
  }
};

class Conv2DPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto conv_params = op->builtin_options_as_Conv2DOptions())
    {
      os << "    ";
      os << "Padding(" << conv_params->padding() << ") ";
      os << "Stride.W(" << conv_params->stride_w() << ") ";
      os << "Stride.H(" << conv_params->stride_h() << ") ";
      os << "Activation("
         << EnumNameActivationFunctionType(conv_params->fused_activation_function()) << ")";
      os << std::endl;
    }
  }
};

class DivPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_DivOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class Pool2DPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto pool_params = op->builtin_options_as_Pool2DOptions())
    {
      os << "    ";
      os << "Padding(" << pool_params->padding() << ") ";
      os << "Stride.W(" << pool_params->stride_w() << ") ";
      os << "Stride.H(" << pool_params->stride_h() << ") ";
      os << "Filter.W(" << pool_params->filter_width() << ") ";
      os << "Filter.H(" << pool_params->filter_height() << ") ";
      os << "Activation("
         << EnumNameActivationFunctionType(pool_params->fused_activation_function()) << ")";
      os << std::endl;
    }
  }
};

class ConcatenationPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *concatenation_params = op->builtin_options_as_ConcatenationOptions())
    {
      os << "    ";
      os << "Activation("
         << EnumNameActivationFunctionType(concatenation_params->fused_activation_function())
         << ") ";
      os << "Axis(" << concatenation_params->axis() << ")";
      os << std::endl;
    }
  }
};

class ReshapePrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *reshape_params = op->builtin_options_as_ReshapeOptions())
    {
      auto new_shape = circleread::as_index_vector(reshape_params->new_shape());
      os << "    ";
      os << "NewShape(" << new_shape << ")";
      os << std::endl;
    }
  }
};

class DepthwiseConv2DPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto conv_params = op->builtin_options_as_DepthwiseConv2DOptions())
    {
      os << "    ";
      os << "Padding(" << conv_params->padding() << ") ";
      os << "Stride.W(" << conv_params->stride_w() << ") ";
      os << "Stride.H(" << conv_params->stride_h() << ") ";
      os << "DepthMultiplier(" << conv_params->depth_multiplier() << ") ";
      os << "Dilation.W(" << conv_params->dilation_w_factor() << ") ";
      os << "Dilation.H(" << conv_params->dilation_h_factor() << ")";
      os << "Activation("
         << EnumNameActivationFunctionType(conv_params->fused_activation_function()) << ") ";
      os << std::endl;
    }
  }
};

class FullyConnectedPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_FullyConnectedOptions())
    {
      os << "    ";
      os << "WeightFormat(" << EnumNameFullyConnectedOptionsWeightsFormat(params->weights_format())
         << ") ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";

      os << std::endl;
    }
  }
};

class MulPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_MulOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class PackPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_PackOptions())
    {
      os << "    ";
      os << "ValuesCount(" << params->values_count() << ") ";
      os << "Axis(" << params->axis() << ") ";
      os << std::endl;
    }
  }
};

class SoftmaxPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *softmax_params = op->builtin_options_as_SoftmaxOptions())
    {
      os << "    ";
      os << "Beta(" << softmax_params->beta() << ")";
      os << std::endl;
    }
  }
};

class SubPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_SubOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class CustomOpPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (op->custom_options_format() != circle::CustomOptionsFormat::CustomOptionsFormat_FLEXBUFFERS)
    {
      os << "    ";
      os << "Unknown custom option format";
      return;
    }

    const flatbuffers::Vector<uint8_t> *option_buf = op->custom_options();

    if (option_buf == nullptr || option_buf->size() == 0)
    {
      os << "No attrs found." << std::endl;
      return;
    }

    // printing attrs
    // attrs of custom ops are encoded in flexbuffer format
    auto attr_map = flexbuffers::GetRoot(option_buf->data(), option_buf->size()).AsMap();

    os << "    ";
    auto keys = attr_map.Keys();
    for (int i = 0; i < keys.size(); i++)
    {
      auto key = keys[i].ToString();
      os << key << "(" << attr_map[key].ToString() << ") ";
    }

    // Note: attr in "Shape" type does not seem to be converted by circle_convert.
    // When the converted circle file (with custom op) is opened with hexa editory,
    // attrs names can be found but attr name in "Shape" type is not found.

    os << std::endl;
  }
};

OpPrinterRegistry::OpPrinterRegistry()
{
  _op_map[circle::BuiltinOperator_ADD] = make_unique<AddPrinter>();
  _op_map[circle::BuiltinOperator_ARG_MAX] = make_unique<ArgMaxPrinter>();
  _op_map[circle::BuiltinOperator_AVERAGE_POOL_2D] = make_unique<Pool2DPrinter>();
  _op_map[circle::BuiltinOperator_CONCATENATION] = make_unique<ConcatenationPrinter>();
  _op_map[circle::BuiltinOperator_CONV_2D] = make_unique<Conv2DPrinter>();
  _op_map[circle::BuiltinOperator_DEPTHWISE_CONV_2D] = make_unique<DepthwiseConv2DPrinter>();
  _op_map[circle::BuiltinOperator_DIV] = make_unique<DivPrinter>();
  _op_map[circle::BuiltinOperator_FULLY_CONNECTED] = make_unique<FullyConnectedPrinter>();
  _op_map[circle::BuiltinOperator_MAX_POOL_2D] = make_unique<Pool2DPrinter>();
  _op_map[circle::BuiltinOperator_MUL] = make_unique<MulPrinter>();
  _op_map[circle::BuiltinOperator_PACK] = make_unique<PackPrinter>();
  // There is no Option for Pad
  // There is no Option for ReLU and ReLU6
  _op_map[circle::BuiltinOperator_RESHAPE] = make_unique<ReshapePrinter>();
  _op_map[circle::BuiltinOperator_SOFTMAX] = make_unique<SoftmaxPrinter>();
  _op_map[circle::BuiltinOperator_SUB] = make_unique<SubPrinter>();
  _op_map[circle::BuiltinOperator_CUSTOM] = make_unique<CustomOpPrinter>();
}

} // namespace circledump

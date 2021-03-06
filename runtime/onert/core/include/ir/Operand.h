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

#ifndef __ONERT_IR_OPERAND_H__
#define __ONERT_IR_OPERAND_H__

#include <cassert>
#include <cstdint>
#include <memory>
#include <algorithm>

#include "ir/Data.h"
#include "ir/DataType.h"
#include "ir/OperandInfo.h"
#include "ir/OperationIndexList.h"

namespace onert
{
namespace ir
{

class Operand
{
public:
  explicit Operand(const Shape &shape, const TypeInfo &type) : _info{shape, type}, _const{false}
  {
    // DO NOTHING
  }

public:
  const Shape &shape(void) const { return _info.shape(); }
  const TypeInfo &typeInfo(void) const { return _info.typeInfo(); }
  const OperandInfo &info(void) const { return _info; }
  OperandInfo &info(void) { return _info; }
  size_t operandSize(void) const;

  const OperationIndexList &getUses() const { return _uses; }
  const OperationIndexList &getDef() const { return _def; }
  void appendUse(const OperationIndex &idx);
  void removeUse(const OperationIndex &idx);
  void appendDef(const OperationIndex &idx);
  void removeDef(const OperationIndex &idx);

public:
  void type(const DataType type) { _info.type(type); };

public:
  void data(std::shared_ptr<Data> &&data)
  {
    _data = std::move(data);
    _const = true;
  }
  const Data *data(void) const { return _data.get(); }

  void releaseData(void) { _data.reset(); }

  /**
   * @brief Get true if Operand is const, otherwise @c false
   a @return @c true if Operand is const, otherwise @c false
   */
  bool isConstant(void) const { return _const; }

public:
  template <typename T, typename... Args> void data(Args &&... args)
  {
    data(std::make_unique<T>(std::forward<Args>(args)...));
  }

public:
  template <typename T> T asScalar(void) const
  {
    assert((shape().rank() == 0) || ((shape().rank() == 1) && (shape().dim(0) == 1)));
    assert(_data != nullptr);
    assert((_data->base() != nullptr) && (_data->size() == sizeof(T)));

    return *(reinterpret_cast<const T *>(_data->base()));
  }

  template <typename T> std::vector<T> asVector() const
  {
    assert(_data != nullptr);
    assert(_data->size() % sizeof(T) == 0);

    const auto *base = reinterpret_cast<const T *>(_data->base());
    const std::size_t size = _data->size() / sizeof(T);
    return std::vector<T>(base, base + size);
  }

private:
  OperandInfo _info;
  std::shared_ptr<Data> _data;
  bool _const;

  OperationIndexList _uses;
  OperationIndexList _def; // size is 0 (constant) or 1 (from def operation)
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERAND_H__

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

#ifndef __ONERT_BACKEND_HI_PERF_CPU_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_HI_PERF_CPU_KERNEL_GENERATOR_H__

#include <backend/IKernelGenerator.h>

#include "ir/Operands.h"
#include "TensorBuilder.h"

namespace onert
{
namespace backend
{
namespace hi_perf_cpu
{

class KernelGenerator : public IKernelGenerator
{
public:
  KernelGenerator(const ir::Operands &ctx, const std::shared_ptr<TensorBuilder> &tensor_builder);
  // TODO add more ops

private:
  const ir::Operands &_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
};

} // namespace hi_perf_cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_HI_PERF_CPU_KERNEL_GENERATOR_H__

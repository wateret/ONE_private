/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nncc/core/ADT/kernel/NCHWLayout.h"

#include <gtest/gtest.h>

using namespace nncc::core::ADT::kernel;

TEST(ADT_KERNEL_KERNEL_NCHW_LAYOUT, col_increment)
{
  const Shape shape{4, 3, 6, 5};
  const NCHWLayout l;

  ASSERT_EQ(l.offset(shape, 1, 1, 1, 1) + 1, l.offset(shape, 1, 1, 1, 2));
}

TEST(ADT_KERNEL_KERNEL_NCHW_LAYOUT, row_increment)
{
  const Shape shape{4, 3, 6, 5};
  const NCHWLayout l;

  ASSERT_EQ(l.offset(shape, 1, 1, 1, 1) + 5, l.offset(shape, 1, 1, 2, 1));
}

TEST(ADT_KERNEL_KERNEL_NCHW_LAYOUT, ch_increment)
{
  const Shape shape{4, 3, 6, 5};
  const NCHWLayout l;

  ASSERT_EQ(l.offset(shape, 1, 1, 1, 1) + 6 * 5, l.offset(shape, 1, 2, 1, 1));
}

TEST(ADT_KERNEL_KERNEL_NCHW_LAYOUT, n_increment)
{
  const Shape shape{4, 3, 6, 5};
  const NCHWLayout l;

  ASSERT_EQ(l.offset(shape, 1, 1, 1, 1) + 3 * 6 * 5, l.offset(shape, 2, 1, 1, 1));
}

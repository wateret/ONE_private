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

#include "luci/IR/Nodes/CircleArgMax.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleArgMaxTest, constructor_P)
{
  luci::CircleArgMax add_node;

  ASSERT_EQ(add_node.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(add_node.opcode(), luci::CircleOpcode::ARG_MAX);

  ASSERT_EQ(add_node.input(), nullptr);
  ASSERT_EQ(add_node.dimension(), nullptr);
}

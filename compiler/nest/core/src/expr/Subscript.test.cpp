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

#include "nest/expr/Subscript.h"
#include "nest/expr/VarNode.h"

#include <memory>

#include <gtest/gtest.h>

TEST(SUBSCRIPT, ctor)
{
  nest::VarID id_0{0};
  nest::VarID id_1{1};

  auto expr_0 = std::make_shared<nest::expr::VarNode>(id_0);
  auto expr_1 = std::make_shared<nest::expr::VarNode>(id_1);

  nest::expr::Subscript sub{expr_0, expr_1};

  ASSERT_EQ(sub.rank(), 2);
  ASSERT_EQ(sub.at(0), expr_0);
  ASSERT_EQ(sub.at(1), expr_1);
}

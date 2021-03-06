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

#ifndef __COCO_IR_STRIDE_2D_H__
#define __COCO_IR_STRIDE_2D_H__

#include <cstdint>

namespace coco
{

class Stride2D
{
public:
  Stride2D() : _vertical{1}, _horizontal{1}
  {
    // DO NOTHING
  }

public:
  Stride2D(uint32_t vertical, uint32_t horizontal) : _vertical{vertical}, _horizontal{horizontal}
  {
    // DO NOTHING
  }

public:
  uint32_t vertical(void) const { return _vertical; }
  Stride2D &vertical(uint32_t value);

public:
  uint32_t horizontal(void) const { return _horizontal; }
  Stride2D &horizontal(uint32_t value);

private:
  uint32_t _vertical;
  uint32_t _horizontal;
};

} // namespace coco

#endif // __COCO_IR_STRIDE_2D_H__

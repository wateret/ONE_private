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

#include "Model.h"
#include "CircleExpContract.h"

#include <luci/Importer.h>
#include <luci/CircleOptimizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>

#include <stdex/Memory.h>
#include <oops/InternalExn.h>

#include <functional>
#include <iostream>
#include <map>
#include <string>

using OptionHook = std::function<int(const char **)>;

using Algorithms = luci::CircleOptimizer::Options::Algorithm;

void print_help(const char *progname)
{
  std::cerr << "USAGE: " << progname << " [options] input output" << std::endl;
  std::cerr << "   --fuse_instnorm : Enable FuseInstanceNormalization Pass" << std::endl;
  std::cerr << std::endl;
}

int entry(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "ERROR: Failed to parse arguments" << std::endl;
    std::cerr << std::endl;
    print_help(argv[0]);
    return 255;
  }

  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;
  luci::CircleOptimizer optimizer;

  auto options = optimizer.options();

  // TODO merge this with help message
  argparse["--fuse_instnorm"] = [&options](const char **) {
    options->enable(Algorithms::FuseInstanceNorm);
    return 0;
  };

  for (int n = 1; n < argc - 2; ++n)
  {
    const std::string tag{argv[n]};
    auto it = argparse.find(tag);
    if (it == argparse.end())
    {
      std::cerr << "Option '" << tag << "' is not supported" << std::endl;
      std::cerr << std::endl;
      print_help(argv[0]);
      return 255;
    }

    n += it->second((const char **)&argv[n + 1]);
  }

  std::string input_path = argv[argc - 2];
  std::string output_path = argv[argc - 1];

  // Load model from the file
  std::unique_ptr<luci::Model> model = luci::load_model(input_path);
  if (model == nullptr)
  {
    std::cerr << "ERROR: Failed to load '" << input_path << "'" << std::endl;
    return 255;
  }

  const circle::Model *input_model = model->model();
  if (input_model == nullptr)
  {
    std::cerr << "ERROR: Failed to read '" << input_path << "'" << std::endl;
    return 255;
  }

  // Import from input Circle file
  luci::Importer importer;
  auto graph = importer.import(input_model);

  // call luci optimizations
  optimizer.optimize(graph.get());

  if (!luci::validate(graph.get()))
    return 255;

  // Export to output Circle file
  luci::CircleExporter exporter;

  CircleExpContract contract(graph.get(), output_path);

  return exporter.invoke(&contract) ? 0 : 255;
}

/*
 * Complete ROS-free GRIL frontend executable.
 * Copyright (C) 2026 whl-cal contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../LICENSE.
 */

#include <Gril_Calib/FullFrontend.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char *program) {
  std::cerr << "Usage: " << program
            << " --input DATASET --config CONFIG --output RESULT"
               " --trace TRACE --batch-trace BATCH_TRACE"
               " --batch-executable EXECUTABLE --batch-config CONFIG"
               " [--gap-policy golden|reset --forward-gap-s SECONDS]\n";
}

double parse_double(const std::string &text, const std::string &name) {
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value))
    throw std::runtime_error("invalid " + name);
  return value;
}

} // namespace

int main(int argc, char **argv) {
  try {
    FullFrontendRunConfig config;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(argv[index]);
      if (index + 1 >= argc) {
        usage(argv[0]);
        return 2;
      }
      const std::string value(argv[++index]);
      if (argument == "--input")
        config.input_path = value;
      else if (argument == "--config")
        config.config_path = value;
      else if (argument == "--output")
        config.result_path = value;
      else if (argument == "--trace")
        config.trace_path = value;
      else if (argument == "--batch-trace")
        config.batch_trace_path = value;
      else if (argument == "--batch-executable")
        config.batch_executable_path = value;
      else if (argument == "--batch-config")
        config.batch_config_path = value;
      else if (argument == "--gap-policy") {
        if (value == "golden")
          config.forward_gap_policy = ForwardGapPolicy::GoldenEquivalence;
        else if (value == "reset")
          config.forward_gap_policy = ForwardGapPolicy::Reset;
        else
          throw std::runtime_error("gap policy must be golden or reset");
      } else if (argument == "--forward-gap-s") {
        config.forward_gap_s = parse_double(value, "forward gap");
      } else {
        usage(argv[0]);
        return 2;
      }
    }
    run_full_frontend(config);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gril_native_full_frontend: " << error.what() << "\n";
    return 1;
  }
}

// Copyright (c) 2023 Graphcore Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef   __NORM_CE_IMPLEMENT_HPP__
#define   __NORM_CE_IMPLEMENT_HPP__

#include <poplar/Engine.hpp>
#include <poplar/DeviceManager.hpp>

#include <poputil/cyclesTables.hpp>

namespace celib{

using NormCEOutput = std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor> ;

NormCEOutput normCE(poplar::Graph&              graph, 
                    poplar::program::Sequence&  prog,
                    poplar::Tensor &            input,
                    poplar::Tensor const&       scale,
                    poplar::Tensor const&       bias,
                    float                       epsilon,
                    int                         num_groups,
                    bool                        stable_algo,
                    std::string const&          debug_str);

poplar::Tensor normCEInf(poplar::Graph&              graph, 
                         poplar::program::Sequence&  prog,
                         poplar::Tensor &            input,
                         poplar::Tensor const&       scale,
                         poplar::Tensor const&       bias,
                         float                       epsilon,
                         int                         num_groups,
                         bool                        stable_algo,
                         std::string const&          debug_str);

poputil::internal::PerfEstimatorTable makePerfFunctionTable();
  
}

#endif
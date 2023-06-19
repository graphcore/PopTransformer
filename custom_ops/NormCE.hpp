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

#ifndef   __NORM_CE_HPP__
#define   __NORM_CE_HPP__

#include <iostream>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/devicex.hpp>

using namespace popart;
using namespace popart::popx;

namespace popart {

  namespace CustomOperators {
    const OperatorIdentifier normCEId = {"ai.graphcore", "NormCE", 1};
  } // namespace CustomOperators

  namespace CustomGradOperators {
    const OperatorIdentifier normCEGradId = {"ai.graphcore", "NormCEGrad", 1};
  }	// namespace CustomGradOperators

} // namespace popart

class NormCEOp : public Op {
public:
  NormCEOp(OperatorIdentifier const&   opid, 
           Op::Settings const&         settings_, 
           bool                        fwd_after_matmul,
           bool                        bwd_after_matmul,
           int64_t                     fwd_grain_size, 
           int64_t                     bwd_grain_size, 
           float                       epsilon,
           int64_t                     num_groups,
           bool                        stable_algo,
           std::string const&          debug_str);

  NormCEOp(const NormCEOp &)            = default;
  NormCEOp &operator=(const NormCEOp &) = delete;
  ~NormCEOp() override                   = default;

  std::vector<std::unique_ptr<Op>>  getGradOps() final;
  std::unique_ptr<Op>               clone() const override;
  virtual void                      setup() final;

  bool canShard() const override { return false; };
  virtual bool isIdentity() const { return canBeReplacedByIdentity(); };

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  virtual void growAliasModel(AliasModel &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  float                      getSubgraphValue() const final { return getLowSubgraphValue(); }
  int64_t                    isFwdAfterMatmul() const       { return fwd_after_matmul_; };
  int64_t                    isBwdAfterMatmul() const       { return bwd_after_matmul_; };
  int64_t                    getFwdGrainSize() const        { return fwd_grain_size_; }; 
  int64_t                    getBwdGrainSize() const        { return bwd_grain_size_; }; 
  float                      getEpsilon() const             { return epsilon_; }; 
  int64_t                    getNumGroups() const           { return num_groups_; };
  int64_t                    isStableAlgo() const           { return stable_algo_; };
  std::string const&         getDebugStr() const            { return debug_str_; };

private:
  int64_t                    fwd_after_matmul_;
  int64_t                    bwd_after_matmul_;
  int64_t                    fwd_grain_size_;
  int64_t                    bwd_grain_size_;
  float                      epsilon_;
  int64_t                    num_groups_;
  bool                       stable_algo_;
  std::string                debug_str_;
};

class NormCEBaseOpx : public Opx {

public:
  NormCEBaseOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

class NormCEOutplaceOpx : public NormCEBaseOpx {

public:
  NormCEOutplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &prog) const final;
};

class NormCEGradOp : public Op {
public:
  NormCEGradOp(const NormCEOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  virtual void        setup() {

    outInfo(0) = {inInfo(0).dataType(), inInfo(0).shape()};
  }

  /* Describes the relationship of the inputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {

    static const std::vector<popart::GradInOutMapper> in_info = {
      // The input of grad op at index 0 is the gradient of the input at
      // index 0 of the non-grad op
      {0, 0, popart::GradOpInType::GradOut}, // gradient of output
      //{1, 0, popart::GradOpInType::Out}, // output
    };

    return in_info;
  }

  /* Describes the relationship of the outputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> out_info = {
      // The output at index 0 is dLhs, i.e the gradient of the input at index 0
      // of non grad op
      {0, 0},
    };
    return out_info;
  }

  float getSubgraphValue() const final   { return getLowSubgraphValue();} ;
  const std::string& getDebugStr() const { return debug_str_; } ;
  int64_t getGrainSize() const           { return grain_size_; };
  int64_t isAfterMatmul() const          { return after_matmul_; };

private:
  int64_t                    after_matmul_;
  int64_t                    grain_size_;
  std::string                debug_str_;
};


class NormCEGradOpx : public Opx {
public:
  NormCEGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<NormCEGradOp>(op, CustomGradOperators::normCEGradId);
  }

  void grow(poplar::program::Sequence &prog) const final;
};

#endif
// Copyright (c) 2023 Graphcore Ltd.
#pragma once
#include <popart/attributes.hpp>
#include <popart/datatype.hpp>
#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/operatoridentifier.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensorinfo.hpp>
#include <memory>
#include <string>
#include <vector>

namespace CustomOperators {
const popart::OperatorIdentifier KVCachePipelineId = {"ai.graphcore",
                                                      "KVCachePipeline",
                                                      1};
} // namespace CustomOperators

class KVCachePipelineOp : public popart::Op {
public:
  KVCachePipelineOp(const popart::OperatorIdentifier &_opid,
                    int max_len,
                    int max_bps,
                    int step_len,
                    int sequence_axis,
                    bool fp8,
                    bool transpose,
                    std::string const debug_str,
                    const popart::Op::Settings &settings_);

  std::unique_ptr<popart::Op> clone() const override;
  void setup() final;

  static popart::InIndex stepInIndex() { return 0; }
  static popart::InIndex inputInIndex() { return 1; }
  static popart::InIndex sequenceAxisInIndex() { return 2; }
  static popart::OutIndex outIndex() { return 0; }

  int getMaxLen() const { return max_len; }
  int getMaxBps() const { return max_bps; }
  int getStepLen() const { return step_len; }
  int getSequenceAxis() const { return sequence_axis; }
  bool saveFP8() const { return save_fp8; }
  bool getTranspose() const { return transpose; }
  std::string const getDebugStr() const { return debug_str_; };
  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  bool isOutlineable() const override { return false; }

private:
  int max_len;
  int max_bps;
  int step_len;
  int sequence_axis;
  bool save_fp8;
  bool transpose;
  std::string debug_str_;
};

KVCachePipelineOp::KVCachePipelineOp(const popart::OperatorIdentifier &_opid,
                                     int max_len,
                                     int max_bps,
                                     int step_len,
                                     int sequence_axis,
                                     bool fp8,
                                     bool transpose,
                                     std::string const debug_str,
                                     const popart::Op::Settings &settings_)
    : popart::Op(_opid, settings_), max_len(max_len), max_bps(max_bps),
      step_len(step_len), sequence_axis(sequence_axis), save_fp8(fp8),
      transpose(transpose), debug_str_(debug_str) {}

std::unique_ptr<popart::Op> KVCachePipelineOp::clone() const {
  return std::make_unique<KVCachePipelineOp>(*this);
}

void KVCachePipelineOp::setup() {
  auto rank     = inRank(inputInIndex());
  sequence_axis = sequence_axis < 0 ? sequence_axis + rank : sequence_axis;
  if (sequence_axis != rank - 2) {
    throw popart::error("[KVCachePipeline] Only support axis = {}(or -2),"
                        "but now it is {}.",
                        rank - 2,
                        sequence_axis);
  }

  auto oShape           = inShape(inputInIndex());
  oShape[sequence_axis] = max_len;

  outInfo(outIndex()) =
      popart::TensorInfo(inInfo(inputInIndex()).dataType(), oShape);
}

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T1 = {DataType::INT32};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    KVCachePipelineOpDef({OpDefinition::Inputs({{"step", T1}, {"input", T2}}),
                          OpDefinition::Outputs({{"output", T2}}),
                          OpDefinition::Attributes({{"max_len", {"*"}},
                                                    {"max_bps", {"*"}},
                                                    {"step_len", {"*"}},
                                                    {"sequence_axis", {"*"}},
                                                    {"fp8", {"*"}},
                                                    {"transpose", {"*"}},
                                                    {"debug_str", {"*"}}})});

static popart::OpCreator<KVCachePipelineOp> KVCachePipelineOpCreator(
    popart::OpDefinitions({{CustomOperators::KVCachePipelineId,
                            KVCachePipelineOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      int max_len =
          info.attributes.getAttribute<popart::Attributes::Int>("max_len", 0);
      int max_bps =
          info.attributes.getAttribute<popart::Attributes::Int>("max_bps", 0);
      int step_len =
          info.attributes.getAttribute<popart::Attributes::Int>("step_len", 1);
      int sequence_axis = info.attributes.getAttribute<popart::Attributes::Int>(
          "sequence_axis", -2);
      int fp8 = info.attributes.getAttribute<popart::Attributes::Int>("fp8", 0);
      int transpose =
          info.attributes.getAttribute<popart::Attributes::Int>("transpose", 0);
      std::string debug_str =
          info.attributes.getAttribute<popart::Attributes::String>("debug_str",
                                                                   "kv_cache");
      return std::make_unique<KVCachePipelineOp>(info.opid,
                                                 max_len,
                                                 max_bps,
                                                 step_len,
                                                 sequence_axis,
                                                 fp8,
                                                 transpose,
                                                 debug_str,
                                                 info.settings);
    },
    true);
} // namespace

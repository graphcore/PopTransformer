// Copyright (c) 2022 Graphcore Ltd.
#pragma once

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/operatoridentifier.hpp>
#include <popart/popx/devicex.hpp>
#include <poplar/Program.hpp>
#include <memory>

namespace poprt {
namespace compiler {
namespace custom_ops {

class CastToUint8Op : public popart::Op {
public:
  enum class Fp8Format { F143, F152 };
  CastToUint8Op(const popart::OperatorIdentifier &_opid,
                const popart::Op::Settings &settings,
                const Fp8Format fp8Format_);
  std::unique_ptr<Op> clone() const override;
  void setup() override;

  static popart::InIndex getInIndex() { return 0; }
  static popart::InIndex getScaleIndex() { return 1; }
  static popart::OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

  Fp8Format getFP8Format() const { return fp8Format; }

  popart::ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override {
    return {{{CastToUint8Op::getInIndex()}, {CastToUint8Op::getOutIndex()}}};
  }

  bool canBeReplacedByIdentity() const override;
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

private:
  Fp8Format fp8Format;
};

class CastToUint8Opx : public popart::popx::Opx {
public:
  CastToUint8Opx(popart::Op *, popart::popx::Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace custom_ops
} // namespace compiler
} // namespace poprt
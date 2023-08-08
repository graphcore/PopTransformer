// Copyright (c) 2022 Graphcore Ltd.
#pragma once

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/op/matmul.hpp>
#include <popart/popx/opx.hpp>
#include <popart/op.hpp>
#include <popart/operatoridentifier.hpp>
#include <popart/vendored/optional.hpp>
#include <poplar/ArrayRef.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <set>
#include <utility>
#include <vector>
#include <memory>
#include <iosfwd>

namespace poprt {
namespace compiler {
namespace custom_ops {

using namespace popart;
using namespace popart::popx;
using namespace poplar;

class FP8MatMulOp : public MatMulBaseOp {
public:
  enum class Fp8Format { F143, F152 };
  FP8MatMulOp(
      const OperatorIdentifier &_opid,
      const Op::Settings &settings_,
      const nonstd::optional<float> &availableMemoryProportion,
      const SerialiseSettings &serialization_,
      const Fp8Format fp8FormatLHS_,
      const Fp8Format fp8FormatRHS_,
      const MatMulPartialsType &partialsType_ = MatMulPartialsType::FLOAT);
  FP8MatMulOp(const FP8MatMulOp &)            = default;
  FP8MatMulOp &operator=(const FP8MatMulOp &) = delete;
  ~FP8MatMulOp() override                     = default;

  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getLhsInIndex() { return 0; }
  static InIndex getRhsInIndex() { return 1; }
  static InIndex getLhsScaleIndex() { return 2; }
  static InIndex getRhsScaleIndex() { return 3; }
  static OutIndex getOutIndex() { return 0; }

  const popart::Tensor *lhsIn() const;
  const popart::Tensor *rhsIn() const;
  const popart::Tensor *out() const;

  // Return the expanded shape of the inputs & output to matmul
  Shape getExpandedLhsShape() const override { return lhsShape; }
  Shape getExpandedRhsShape() const override { return rhsShape; }
  Shape getExpandedOutShape() const { return outShape; }

  // set/get the option for matmul to create it's inputs
  void setCanCreateInputs(bool value) { canCreateInputs = value; }
  bool getCanCreateInputs() const { return canCreateInputs; }
  void appendOutlineAttributes(OpSerialiserBase &os) const override;

  Fp8Format getLHSFP8Format() const { return fp8FormatLHS; }
  Fp8Format getRHSFP8Format() const { return fp8FormatRHS; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  // Follow the numpy matmul broadcasting rules for the output shape
  Shape npMatMulOut(Shape lhs, Shape rhs);

private:
  // Verifies the input shapes are valid and throws and exception if not
  void verifyInputShapes(const Shape &lhs, const Shape &rhs) const;

  // Flag to indicate if mat mul can create it's inputs.
  // MatMulGradXXOps converted to MatMulOps don't create their inputs
  // Since the data type of ir is uint8 and the matmul is quater,
  // set to false to avoid type checking here, it's different from MatMulOp.
  bool canCreateInputs = true;

  // The expanded shapes of inputs & outputs. They will
  // be a minium of a 3D shapes
  Shape lhsShape;
  Shape rhsShape;
  Shape outShape;

  Fp8Format fp8FormatLHS;
  Fp8Format fp8FormatRHS;
};

class FP8MatMulOpx : public Opx {
public:
  FP8MatMulOpx(Op *, Devicex *);
  ~FP8MatMulOpx() override = default;

  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex index0) const final;

  FP8MatMulOp *getMatMulOp() const;
  void grow(poplar::program::Sequence &) const final;

  static std::vector<std::size_t> onnxShapeToPoplar(const Shape &shape);
  static void appendPoplarOptionsForOp(const MatMulBaseOp &op,
                                       poplar::OptionFlags &opts);
  static void addPartialsType(const MatMulPartialsType &partialsType,
                              poplar::OptionFlags &opts);

  static std::pair<poplar::Tensor, poplar::Tensor>
  groupedMatMulInputsFromOpxInputs(MatMulBaseOp &matmul,
                                   poplar::Tensor lhs,
                                   poplar::Tensor rhs);

  // Check that mamtul pre-planning has worked, and that growing the matmul
  // operation has not added unexpected entries to the planning cache. Note:
  // poplibs matmul creates a 'joint plan' - a plan for the corresponding
  // fwd, bwd and wu matmuls - all at once if the 'fullyConnectedPass' option
  // is 'TRAINING_*'.
  // But if only a subset of these ops exist in our graph, then only a subset
  // of their plans will exist in the cache. In this case we can expect to see
  // up to 2 more plans generated than expected.
  /* void verifyCacheSizeUnchanged(size_t beforeCacheSize) const; */

private:
  // The ONNX tensor shape
  std::vector<std::size_t> getOutputShape() const;
};

} // namespace custom_ops
} // namespace compiler
} // namespace poprt
// Copyright (c) 2022 Graphcore Ltd.
#include "matmul_fp8.hpp"

#include <boost/range/algorithm/find.hpp>
#include <boost/range/algorithm/mismatch.hpp>
#include <boost/range/algorithm/stable_partition.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <boost/range/const_iterator.hpp>
#include <ctype.h>
#include <popart/attributes.hpp>
#include <popart/datatype.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordebuginfo.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/util.hpp>
#include <poplar/Graph.hpp>
#include <poplar/MetadataCreation.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/TensorCloneMethod.hpp>
#include <poplar/Type.hpp>
#include <poplin/MatMul.hpp>
#include <popops/Cast.hpp>
#include <stdint.h>
#include <algorithm>
#include <functional>
#include <iosfwd>
#include <iterator>
#include <numeric>
#include <string>

namespace poprt {
namespace compiler {
namespace custom_ops {

using namespace popart;
using namespace popart::popx;
using namespace poplar;

const popart::OperatorIdentifier FP8MatMulOpId = {"ai.graphcore",
                                                  "FP8MatMulWithoutShapeInfer",
                                                  1};

FP8MatMulOp::FP8MatMulOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_,
    const nonstd::optional<float> &availableMemoryProportion_,
    const SerialiseSettings &serialization_,
    const Fp8Format fp8FormatLHS_,
    const Fp8Format fp8FormatRHS_,
    const MatMulPartialsType &partialsType_)
    : MatMulBaseOp(_opid,
                   settings_,
                   Phase::Fwd,
                   availableMemoryProportion_,
                   serialization_,
                   DataType::FLOAT16,
                   partialsType_),
      fp8FormatLHS(fp8FormatLHS_), fp8FormatRHS(fp8FormatRHS_) {}

std::unique_ptr<Op> FP8MatMulOp::clone() const {
  return std::make_unique<FP8MatMulOp>(*this);
}

const popart::Tensor *FP8MatMulOp::lhsIn() const {
  return inTensor(getLhsInIndex());
}
const popart::Tensor *FP8MatMulOp::rhsIn() const {
  return inTensor(getRhsInIndex());
}
const popart::Tensor *FP8MatMulOp::out() const {
  return outTensor(getOutIndex());
}

Shape FP8MatMulOp::npMatMulOut(Shape lhs, Shape rhs) {
  verifyInputShapes(lhs, rhs);

  auto originalLhsShape = lhs;
  auto originalRhsShape = rhs;

  const bool lhs_prepend = lhs.size() == 1;
  const bool rhs_append  = rhs.size() == 1;

  // If the first argument is 1-D, it is promoted to a matrix by prepending a 1
  // to its dimensions.
  if (lhs_prepend) {
    lhs.insert(lhs.begin(), 1);
  }

  // If the second argument is 1-D, it is promoted to a matrix by appending a 1
  // to its dimensions
  if (rhs_append) {
    rhs.push_back(1);
  }

  // Add a 1 group dim
  bool lhsGroupDimAppend = false;
  if (lhs.size() == 2) {
    lhs.insert(lhs.begin(), 1);
    lhsGroupDimAppend = true;
  }

  // Add a 1 group dim
  bool rhsGroupDimAppend = false;
  if (rhs.size() == 2) {
    rhs.insert(rhs.begin(), 1);
    rhsGroupDimAppend = true;
  }

  // Save the expanded input shapes - minium of 3D
  lhsShape = lhs;
  rhsShape = rhs;

  Shape result =
      prettyNpOut({lhs.begin(), lhs.end() - 2}, {rhs.begin(), rhs.end() - 2});

  // Save the expanded output shape - minium of 3D
  outShape = result;
  outShape.push_back(lhs[lhs.size() - 2]);
  outShape.push_back(rhs[rhs.size() - 1]);

  // After matrix multiplication the prepended 1 is removed.
  // We implement this by not adding it.
  if (!lhs_prepend) {
    result.push_back(lhs[lhs.size() - 2]);
  }

  // After matrix multiplication the appended 1 is removed.
  // We implement this by not adding it.
  if (!rhs_append) {
    result.push_back(rhs[rhs.size() - 1]);
  }

  // Squeeze off any prepended 1's if both
  // lhs & rhs had prepended a group dimension
  if (lhsGroupDimAppend && rhsGroupDimAppend && result[0] == 1) {
    result.erase(result.begin());
  }

  // Special case of 2 1d inputs
  if (originalLhsShape.size() == 1 && originalRhsShape.size() == 1 &&
      result.size() == 1 && result[0] == 1)
    result.erase(result.begin());

  if (lhs[lhs.size() - 1] != rhs[rhs.size() - 2]) {

    // Remove the group dimension if added to return to the user defined
    // shape
    if (lhsGroupDimAppend)
      lhs.erase(lhs.begin());

    if (rhsGroupDimAppend)
      rhs.erase(rhs.begin());

    throw error("{} contracting dimensions unequal: lhs '{}' {}, rhs '{}' {}",
                debugName(),
                lhsIn()->str(),
                lhs,
                rhsIn()->str(),
                rhs);
  }

  return result;
}

void FP8MatMulOp::setup() {
  if (phase == Phase::Fwd) {
    if (getSerialiseSettings().mode !=
        MatMulBaseOp::SerialiseSettings::Mode::None) {

      if (getSerialiseSettings().factor == 0) {
        throw error("Invalid serialisation factor {}. Serialisation factor "
                    "must be a non-zero positive integer.",
                    getSerialiseSettings().factor);
      }

      // assuming
      // lhs = [group_dims, input_channels, reduce_dim]
      // rhs = [group_dims, reduce_dim, output_channels]

      if (getSerialiseSettings().mode ==
          MatMulBaseOp::SerialiseSettings::Mode::InputChannels) {

        // Get the input channels of the left hand size
        auto inputChannelsDim =
            lhsIn()->info.shape()[lhsIn()->info.shape().size() - 2];

        if (inputChannelsDim % getSerialiseSettings().factor != 0) {
          throw error("Invalid serialisation factor {} for input channels dim "
                      "{}. input_channels dim should be a multple of the "
                      "serialisation factor ",
                      getSerialiseSettings().factor,
                      inputChannelsDim);
        }
      } else if (getSerialiseSettings().mode ==
                 MatMulBaseOp::SerialiseSettings::Mode::ReducingDim) {
        // Get the reducing dim of the left hand tensor
        auto reducingChannelsDim =
            lhsIn()->info.shape()[lhsIn()->info.shape().size() - 1];

        if (reducingChannelsDim % getSerialiseSettings().factor != 0) {
          throw error("Invalid serialisation factor {} for reducing dimension "
                      "{}. reducing_dim dim should be a multple of the "
                      "serialisation factor ",
                      getSerialiseSettings().factor,
                      reducingChannelsDim);
        }
      } else {

        // Get the output channels of the right hand size
        auto outputChannelsDim =
            rhsIn()->info.shape()[rhsIn()->info.shape().size() - 1];

        logging::op::info("{}", rhsIn()->info.shape());
        if (outputChannelsDim % getSerialiseSettings().factor != 0) {
          throw error("Invalid serialisation factor {} for output channels dim "
                      "{}. output_channels dim should be a multple of the "
                      "serialisation factor ",
                      getSerialiseSettings().factor,
                      outputChannelsDim);
        }
      }
    }
  }

  auto typeLHS = lhsIn()->info.dataType();
  auto typeRHS = rhsIn()->info.dataType();
  if (typeLHS != DataType::FLOAT16 && typeLHS != DataType::UINT8) {
    throw error("Invalid input data type of {}, it must be FLOAT16 or UINT8",
                typeLHS);
  }
  if (typeRHS != DataType::UINT8) {
    throw error("Invalid weight data type of {}, it must be UINT8", typeRHS);
  }

  // Define the shape of the output tensor
  outInfo(0) = {DataType::FLOAT16,
                npMatMulOut(lhsIn()->info.shape(), rhsIn()->info.shape())};
}

// Verifies the input shapes are valid and throws and exception if not
void FP8MatMulOp::verifyInputShapes(const Shape &lhs, const Shape &rhs) const {
  if (lhs.empty()) {
    throw error("{} doesn't support scalar tensor {} as the lhs input",
                debugName(),
                lhsIn()->str());
  }

  if (rhs.empty()) {
    throw error("{} doesn't support scalar tensor {} as the rhs input",
                debugName(),
                rhsIn()->str());
  }
}

void FP8MatMulOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  MatMulBaseOp::appendOutlineAttributes(os);
  os.appendAttribute("fp8FormatLHS", int(fp8FormatLHS));
  os.appendAttribute("fp8FormatRHS", int(fp8FormatRHS));
}

namespace {
static OpDefinition::DataTypes T1 = {DataType::UINT8};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16};
static OpDefinition::DataTypes T3 = {DataType::INT32};

// Accepts the strings "half", "float" in any kind of letter case.
MatMulPartialsType fromString(const std::string &user_pt) {
  std::string lowered_pt;
  lowered_pt.resize(user_pt.length());

  std::transform(user_pt.begin(), user_pt.end(), lowered_pt.begin(), ::tolower);

  if (lowered_pt == "half") {
    return MatMulPartialsType::HALF;
  } else if (lowered_pt == "float") {
    return MatMulPartialsType::FLOAT;
  } else {
    const auto err_str_tmpl =
        "Unable to get option 'partialsTypeMatMul' from "
        "string '{}'. Possible values are 'float' and 'half' in any letter "
        "case.";
    throw error(err_str_tmpl, user_pt);
  }
}

static OpDefinition
    FP8MatMulOpDef({OpDefinition::Inputs(
                        {{"A", T1}, {"B", T1}, {"scaleA", T3}, {"scaleB", T3}}),
                    OpDefinition::Outputs({{"output", T2}}),
                    OpDefinition::Attributes({
                        {sSerializeMatMulModeAttribute, {"*"}},
                        {sSerializeMatMulFactorAttribute, {"*"}},
                        {sSerializeMatMulPrecisionAttribute, {"*"}},
                        {"fp8FormatLHS", {"*"}},
                        {"fp8FormatRHS", {"*"}},
                    })});

static OpCreator<FP8MatMulOp> FP8MatMulOpCreator(
    OpDefinitions({
        {FP8MatMulOpId, FP8MatMulOpDef},
    }),
    [](const popart::OpCreatorInfo &info) -> std::unique_ptr<popart::Op> {
      // try set the availMemAttribute from an attribute

      nonstd::optional<float> availableMemoryProportion;
      nonstd::optional<int64_t> serialize;

      MatMulBaseOp::SerialiseSettings serialisation;

      FP8MatMulOp::Fp8Format fp8FormatLHS = FP8MatMulOp::Fp8Format::F143;
      if (info.attributes.hasAttribute("fp8FormatLHS")) {
        std::string format =
            info.attributes.getAttribute<Attributes::String>("fp8FormatLHS");
        if (format == "F152") {
          fp8FormatLHS = FP8MatMulOp::Fp8Format::F152;
        } else if (format == "F143") {
          fp8FormatLHS = FP8MatMulOp::Fp8Format::F143;
        } else {
          throw error(
              "Unsupport fp8 lhs meata data {}, it must be F143 or F152",
              format);
        }
      }

      FP8MatMulOp::Fp8Format fp8FormatRHS = FP8MatMulOp::Fp8Format::F143;
      if (info.attributes.hasAttribute("fp8FormatRHS")) {
        std::string format =
            info.attributes.getAttribute<Attributes::String>("fp8FormatRHS");
        if (format == "F152") {
          fp8FormatRHS = FP8MatMulOp::Fp8Format::F152;
        } else if (format == "F143") {
          fp8FormatRHS = FP8MatMulOp::Fp8Format::F143;
        } else {
          throw error(
              "Unsupport fp8 rhs meata data {}, it must be F143 or F152",
              format);
        }
      }

      if (info.attributes.hasAttribute(sSerializeMatMulModeAttribute)) {

        std::string mode = info.attributes.getAttribute<Attributes::String>(
            sSerializeMatMulModeAttribute, sSerializeMatMulMode_None);
        if (mode == sSerializeMatMulMode_InputChannels) {
          serialisation.mode =
              MatMulBaseOp::SerialiseSettings::Mode::InputChannels;
        } else if (mode == sSerializeMatMulMode_ReducingDim) {
          serialisation.mode =
              MatMulBaseOp::SerialiseSettings::Mode::ReducingDim;
        } else if (mode == sSerializeMatMulMode_OutputChannels) {
          serialisation.mode =
              MatMulBaseOp::SerialiseSettings::Mode::OutputChannels;
        } else if (mode == sSerializeMatMulMode_None) {
          serialisation.mode = MatMulBaseOp::SerialiseSettings::Mode::None;
        } else {
          throw error("Unsupport matmul serialisation mode {}", mode);
        }

        serialisation.factor = info.attributes.getAttribute<Attributes::Int>(
            sSerializeMatMulFactorAttribute);

        serialisation.keep_precision =
            info.attributes.getAttribute<Attributes::Int>(
                sSerializeMatMulPrecisionAttribute);
      }

      if (info.attributes.hasAttribute(sAvailMemAttribute)) {
        availableMemoryProportion =
            info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      auto partialsType = MatMulPartialsType::FLOAT;
      // try set the partials from an attribute
      if (info.attributes.hasAttribute(sPartialsTypeAttribute)) {
        std::string partialsTypeAttr =
            info.attributes.getAttribute<Attributes::String>(
                sPartialsTypeAttribute);
        partialsType = fromString(partialsTypeAttr);
      }
      // otherwise see if partials type was set in the session options
      else {
        const auto &opts = info.settings.getIr().getSessionOptions();
        const std::string globalPartialsTypeStr = opts.partialsTypeMatMuls;
        if (!globalPartialsTypeStr.empty()) {
          partialsType = fromString(globalPartialsTypeStr);
        }
      }

      return std::unique_ptr<FP8MatMulOp>(
          new FP8MatMulOp(info.opid,
                          info.settings,
                          availableMemoryProportion,
                          serialisation,
                          fp8FormatLHS,
                          fp8FormatRHS,
                          partialsType));
    },
    true);
} // namespace

static std::pair<poplar::Tensor, poplar::Tensor>
matInitReshape(MatMulBaseOp &matmul, poplar::Tensor lhs, poplar::Tensor rhs) {
  if (lhs.rank() < matmul.getExpandedLhsShape().size()) {
    lhs =
        lhs.reshape(vXtoY<int64_t, std::size_t>(matmul.getExpandedLhsShape()));
  }

  if (rhs.rank() < matmul.getExpandedRhsShape().size()) {
    rhs =
        rhs.reshape(vXtoY<int64_t, std::size_t>(matmul.getExpandedRhsShape()));
  }

  return {lhs, rhs};
}

static std::vector<std::size_t> matchRank(std::vector<std::size_t> shape,
                                          unsigned rank) {
  std::vector<std::size_t> newShape(rank, 1);
  std::copy(shape.rbegin(), shape.rend(), newShape.rbegin());
  return newShape;
}

static std::pair<poplar::Tensor, poplar::Tensor>
matMatchRank(poplar::Tensor lhs, poplar::Tensor rhs) {
  auto rank = std::max(lhs.rank(), rhs.rank());
  return {lhs.reshape(matchRank(lhs.shape(), rank)),
          rhs.reshape(matchRank(rhs.shape(), rank))};
}

static std::vector<unsigned> matDimshuffle(std::vector<std::size_t> lhsShape,
                                           std::vector<std::size_t> rhsShape) {
  std::vector<unsigned> permutation(lhsShape.size() - 2);
  boost::iota(permutation, 0);

  const auto compareDimensions = [&](unsigned dim) {
    return lhsShape[dim] == rhsShape[dim];
  };

  boost::stable_partition(permutation, compareDimensions);

  permutation.push_back(static_cast<unsigned>(lhsShape.size() - 2));
  permutation.push_back(static_cast<unsigned>(lhsShape.size() - 1));

  return permutation;
}

static std::pair<poplar::Tensor, poplar::Tensor>
matDimshuffle(poplar::Tensor lhs, poplar::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();

  return {lhs.dimShuffle(matDimshuffle(lhsShape, rhsShape)),
          rhs.dimShuffle(matDimshuffle(lhsShape, rhsShape))};
}

static std::vector<std::size_t>
lhsReshapeGroups(std::vector<std::size_t> lhsShape,
                 std::vector<std::size_t> rhsShape) {
  auto begin = lhsShape.begin();
  auto groupEnd =
      std::mismatch(lhsShape.begin(), lhsShape.end() - 2, rhsShape.begin())
          .first;
  auto broadcastEnd = lhsShape.end() - 2;

  unsigned groupSize =
      std::accumulate(begin, groupEnd, 1, std::multiplies<std::size_t>());

  unsigned broadcastSize = std::accumulate(
      groupEnd, broadcastEnd, 1, std::multiplies<std::size_t>());

  std::vector<std::size_t> result = {groupSize, broadcastSize, 1, 1};
  std::copy(lhsShape.rbegin(), lhsShape.rbegin() + 2, result.rbegin());

  return result;
}

static std::vector<std::size_t>
rhsReshapeGroups(const std::vector<std::size_t> &lhsShape,
                 const std::vector<std::size_t> &rhsShape) {
  return lhsReshapeGroups(rhsShape, lhsShape);
}

static std::pair<poplar::Tensor, poplar::Tensor>
matReshapeGroups(poplar::Tensor lhs, poplar::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();

  return {lhs.reshape(lhsReshapeGroups(lhsShape, rhsShape)),
          rhs.reshape(rhsReshapeGroups(lhsShape, rhsShape))};
}

static std::vector<std::size_t>
matCombineBroadcastDims(std::vector<std::size_t> shape) {
  return {shape[0], shape[1] * shape[2], shape[3]};
}

static std::pair<poplar::Tensor, poplar::Tensor>
matCombineBroadcastDims(poplar::Tensor lhs, poplar::Tensor rhs) {
  rhs = rhs.dimShuffle({0, 1, 3, 2});
  lhs = lhs.reshape(matCombineBroadcastDims(lhs.shape()));
  rhs = rhs.reshape(matCombineBroadcastDims(rhs.shape()));
  return {lhs, rhs.dimShuffle({0, 2, 1})};
}

static poplar::Tensor matSplitBroadcastDims(poplar::Tensor result,
                                            poplar::Tensor lhs,
                                            poplar::Tensor rhs) {
  return result.reshape(
      {result.dim(0), lhs.dim(1), lhs.dim(2), rhs.dim(1), rhs.dim(3)});
}

static poplar::Tensor matUnDimShuffle(poplar::Tensor result) {
  return result.dimShuffle({0, 1, 3, 2, 4});
}

static poplar::Tensor matExpandBroadcastDims(poplar::Tensor result,
                                             poplar::Tensor lhs,
                                             poplar::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();
  const auto outShape = result.shape();

  const auto itrs =
      std::mismatch(lhsShape.begin(), lhsShape.end() - 2, rhsShape.begin());

  std::vector<std::size_t> newShape;
  newShape.reserve(lhs.rank() + rhs.rank());

  std::copy(lhsShape.begin(), lhsShape.end() - 2, std::back_inserter(newShape));
  std::copy(itrs.second, rhsShape.end() - 2, std::back_inserter(newShape));
  std::copy(outShape.end() - 2, outShape.end(), std::back_inserter(newShape));

  return result.reshape(newShape);
}

static poplar::Tensor matExpandGroupDims(poplar::Tensor result,
                                         poplar::Tensor lhs,
                                         poplar::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();
  const auto outShape = result.shape();

  const auto offset = std::distance(
      lhsShape.begin(), boost::mismatch(lhsShape, rhs.shape()).first);

  std::vector<std::size_t> newShape;
  newShape.reserve(lhs.rank());

  std::copy(lhsShape.begin(),
            lhsShape.begin() + offset,
            std::back_inserter(newShape));
  std::copy(
      outShape.begin() + offset, outShape.end(), std::back_inserter(newShape));

  return result.reshape(newShape);
}

static poplar::Tensor matInterleaveBroadcastDims(poplar::Tensor result,
                                                 poplar::Tensor lhs,
                                                 poplar::Tensor rhs) {
  const auto lhsShape = lhs.shape();

  const auto offset = std::distance(
      lhsShape.begin(), boost::mismatch(lhsShape, rhs.shape()).first);

  const auto length = lhs.rank() - offset - 2;

  std::vector<unsigned> permutation(result.rank());
  boost::iota(permutation, 0);

  for (int i = 0; i < length; ++i) {
    for (int k = 0; k < 2; ++k) {
      permutation[offset + i * 2 + k] =
          static_cast<unsigned>(offset + k * length + i);
    }
  }

  return result.dimShuffle(permutation);
}

static poplar::Tensor matSqueezeBroadcastDims(poplar::Tensor result,
                                              poplar::Tensor lhs,
                                              poplar::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto offset   = std::distance(
      lhsShape.begin(), boost::mismatch(lhsShape, rhs.shape()).first);

  std::vector<std::size_t> squeezeDims;
  for (auto i = offset; i < result.rank() - 2; ++i) {
    if (result.dim(static_cast<unsigned>(i)) == 1) {
      squeezeDims.push_back(i);
    }
  }
  return result.squeeze(squeezeDims);
}

template <typename T1, typename T2>
static std::vector<T1> permute(std::vector<T1> input,
                               std::vector<T2> permutation) {
  auto output = input;

  for (int i = 0; i < output.size(); ++i) {
    output[i] = input[permutation[i]];
  }

  return output;
}

template <typename T>
static std::vector<T> invertPermutation(std::vector<T> permutation) {
  auto output = permutation;

  for (int i = 0; i < output.size(); ++i) {
    output[permutation[i]] = i;
  }

  return output;
}

static std::vector<unsigned>
matShuffleGroupDims(std::vector<std::size_t> rShape,
                    std::vector<std::size_t> lhsShape,
                    std::vector<std::size_t> rhsShape) {
  std::vector<unsigned> mapping;

  mapping.reserve(rShape.size());
  for (int i = 0; i < lhsShape.size() - 2; ++i) {
    if (lhsShape[i] == rhsShape[i]) {
      mapping.push_back(i);
    }
  }

  for (int i = 0; i < rShape.size(); ++i) {
    if (mapping.end() == boost::find(mapping, i)) {
      mapping.push_back(i);
    }
  }

  return invertPermutation(mapping);
}

static poplar::Tensor matShuffleGroupDims(poplar::Tensor result,
                                          poplar::Tensor lhs,
                                          poplar::Tensor rhs) {
  const auto permutation =
      matShuffleGroupDims(result.shape(), lhs.shape(), rhs.shape());

  return result.dimShuffle(permutation);
}

FP8MatMulOpx::FP8MatMulOpx(Op *_op, Devicex *_devicex) : Opx(_op, _devicex) {
  verifyOp<FP8MatMulOp>(_op);
}

FP8MatMulOp *FP8MatMulOpx::getMatMulOp() const {
  return dynamic_cast<FP8MatMulOp *>(op_p);
}

void FP8MatMulOpx::appendPoplarOptionsForOp(const MatMulBaseOp &op,
                                            poplar::OptionFlags &opts) {
  auto &ir = op.getIr();

  if (op.useFullyConnectedPass()) {
    opts.set("fullyConnectedPass", "INFERENCE_FWD");
  }

  if (auto prop = op.getAvailableMemoryProportion()) {
    opts.set("availableMemoryProportion", std::to_string(*prop));
  }

  {
    const auto partialsType = op.getPartialsType();
    addPartialsType(partialsType, opts);
  }
}

// Add the partials type to the poplar::OptionFlags that were computed from the
// poplar::popx::PoplarOptions.
void FP8MatMulOpx::addPartialsType(const MatMulPartialsType &partialsType,
                                   poplar::OptionFlags &opts) {
  switch (partialsType) {
  case MatMulPartialsType::HALF: {
    opts.set("partialsType", "half");
    break;
  }
  case MatMulPartialsType::FLOAT: {
    opts.set("partialsType", "float");
    break;
  }
  default: {
    throw error("Bad MatMulPartialsType {}", static_cast<int>(partialsType));
  }
  }
}

std::vector<std::size_t> FP8MatMulOpx::onnxShapeToPoplar(const Shape &shape) {
  std::size_t m      = shape[shape.size() - 2];
  std::size_t n      = shape[shape.size() - 1];
  std::size_t stacks = std::accumulate(
      shape.begin(), shape.end() - 2, 1, std::multiplies<int64_t>());
  return {stacks, m, n};
}

poplar::Tensor
FP8MatMulOpx::createInput(InIndex index,
                          const poplar::DebugNameAndId &dnai) const {
  auto &matmul = getOp<FP8MatMulOp>();
  std::vector<std::size_t> lhsShape =
      vXtoY<int64_t, std::size_t>(matmul.getExpandedLhsShape());
  std::vector<std::size_t> rhsShape =
      vXtoY<int64_t, std::size_t>(matmul.getExpandedRhsShape());

  lhsShape = matchRank(
      lhsShape,
      static_cast<unsigned>(std::max(lhsShape.size(), rhsShape.size())));
  rhsShape = matchRank(
      rhsShape,
      static_cast<unsigned>(std::max(lhsShape.size(), rhsShape.size())));

  const auto permutation = matDimshuffle(lhsShape, rhsShape);
  const auto lhsShapeP   = permute(lhsShape, permutation);
  const auto rhsShapeP   = permute(rhsShape, permutation);

  const auto lhsReshapeGroupsL = [rhsShapeP](std::vector<std::size_t> shape) {
    return lhsReshapeGroups(shape, rhsShapeP);
  };

  const auto rhsReshapeGroupsL = [lhsShapeP](std::vector<std::size_t> shape) {
    return rhsReshapeGroups(lhsShapeP, shape);
  };

  lhsShape = lhsReshapeGroupsL(lhsShapeP);
  rhsShape = rhsReshapeGroupsL(rhsShapeP);

  lhsShape = matCombineBroadcastDims(lhsShape);

  std::swap(rhsShape[3], rhsShape[2]);
  rhsShape = matCombineBroadcastDims(rhsShape);
  std::swap(rhsShape[2], rhsShape[1]);

  auto opts = dv_p->lowering().matmulOptions;
  appendPoplarOptionsForOp(matmul, opts);

  auto typeLHS = matmul.lhsIn()->info.dataType();
  poplar::Tensor result;
  if (index == FP8MatMulOp::getLhsInIndex()) {
    if (typeLHS == DataType::UINT8) {
      result = poplin::createMatMulGroupedInputLHS(graph(),
                                                   QUARTER,
                                                   HALF,
                                                   lhsShape,
                                                   rhsShape,
                                                   dnai,
                                                   opts,
                                                   &dv_p->matmulCache);
      result = result.reinterpret(UNSIGNED_CHAR);
    } else {
      result = poplin::createMatMulGroupedInputLHS(graph(),
                                                   HALF,
                                                   HALF,
                                                   lhsShape,
                                                   rhsShape,
                                                   dnai,
                                                   opts,
                                                   &dv_p->matmulCache);
    }
    result = result.reshape(lhsShapeP);
    result = result.dimShuffle(invertPermutation(permutation));

    return result.reshape(matmul.lhsIn()->info.shape_szt());
  } else if (index == FP8MatMulOp::getRhsInIndex()) {
    if (typeLHS == DataType::UINT8) {
      result = poplin::createMatMulGroupedInputRHS(graph(),
                                                   QUARTER,
                                                   HALF,
                                                   lhsShape,
                                                   rhsShape,
                                                   dnai,
                                                   opts,
                                                   &dv_p->matmulCache);
      result = result.reinterpret(UNSIGNED_CHAR);
    } else {
      auto result_fp16 =
          poplin::createMatMulGroupedInputRHS(graph(),
                                              HALF,
                                              HALF,
                                              lhsShape,
                                              rhsShape,
                                              dnai,
                                              opts,
                                              &dv_p->matmulCache);
      result =
          graph().clone(poplar::UNSIGNED_CHAR,
                        result_fp16,
                        debugContext("result"),
                        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    }
    result = result.reshape(rhsShapeP);
    result = result.dimShuffle(invertPermutation(permutation));

    return result.reshape(matmul.rhsIn()->info.shape_szt());
  } else {
    throw error("FP8MatMulOpx::createInput invalid input index {}", index);
  }
}

InputCreatorType FP8MatMulOpx::getInputCreatorType(InIndex index) const {
  const FP8MatMulOp *op = dynamic_cast<const FP8MatMulOp *>(op_p);
  bool isScale          = false;
  if ((index == op->getLhsScaleIndex()) || (index == op->getRhsScaleIndex())) {
    isScale = true;
  }
  if (op->getCanCreateInputs() && !isScale) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

std::set<TensorId> FP8MatMulOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

std::pair<poplar::Tensor, poplar::Tensor>
FP8MatMulOpx::groupedMatMulInputsFromOpxInputs(MatMulBaseOp &matmul,
                                               poplar::Tensor lhs,
                                               poplar::Tensor rhs) {
  auto initReshapedTs = matInitReshape(matmul, lhs, rhs);

  auto matchedRankTs =
      matMatchRank(initReshapedTs.first, initReshapedTs.second);

  auto dimShuffledTs = matDimshuffle(matchedRankTs.first, matchedRankTs.second);

  auto reshapedGroupsTs =
      matReshapeGroups(dimShuffledTs.first, dimShuffledTs.second);

  auto combinedBroadcastTs =
      matCombineBroadcastDims(reshapedGroupsTs.first, reshapedGroupsTs.second);

  return combinedBroadcastTs;
}

// TODO, this function will cause the error about Pre-planning failed
/*
void FP8MatMulOpx::verifyCacheSizeUnchanged(size_t beforeCacheSize) const {
  bool expectedCacheSize;
  auto opts = dv_p->lowering().matmulOptions;
  appendPoplarOptionsForOp(getOp<FP8MatMulOp>(), opts);
  auto hasFlag = [](const auto &opts, auto flag) {
    for (auto &x : opts) {
      if (x.first == "fullyConnectedPass") {
        return true;
      }
    }
    return false;
  };
  if (hasFlag(opts, "fullyConnectedPass") &&
      opts.at("fullyConnectedPass") != "INFERENCE_FWD") {
    expectedCacheSize = dv_p->matmulCache.size() <= beforeCacheSize + 2;
  } else {
    expectedCacheSize = beforeCacheSize == dv_p->matmulCache.size();
  }
  if (!expectedCacheSize) {
    throw internal_error(
        "Pre-planning failed for {}. Its plan was not found in the cache",
        op_p->str());
  }
}
*/

void FP8MatMulOpx::grow(poplar::program::Sequence &prog) const {
  auto &matmul = getOp<FP8MatMulOp>();

  auto a_input        = getInTensor(FP8MatMulOp::getLhsInIndex());
  auto b_input        = getInTensor(FP8MatMulOp::getRhsInIndex());
  auto a_scale        = getInTensor(FP8MatMulOp::getLhsScaleIndex())[0];
  auto b_scale        = getInTensor(FP8MatMulOp::getRhsScaleIndex())[0];
  auto fp8_format_lhs = matmul.getLHSFP8Format();
  auto fp8_format_rhs = matmul.getRHSFP8Format();

  QuarterMetadata::Format lhs_quarter_format = QuarterMetadata::Format::F143;
  QuarterMetadata::Format rhs_quarter_format = QuarterMetadata::Format::F143;
  if (fp8_format_lhs == FP8MatMulOp::Fp8Format::F152) {
    lhs_quarter_format = QuarterMetadata::Format::F152;
  }
  if (fp8_format_rhs == FP8MatMulOp::Fp8Format::F152) {
    rhs_quarter_format = QuarterMetadata::Format::F152;
  }

  auto qm_rhs = poplar::createVariableMetadataTensor(
      graph(), rhs_quarter_format, b_scale, prog);

  poplar::Tensor a, b;
  b = graph().clone(poplar::QUARTER,
                    qm_rhs,
                    b_input,
                    debugContext("b"),
                    poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
  prog.add(poplar::program::Copy(b_input, b.reinterpret(UNSIGNED_CHAR)));

  if (a_input.elementType() == poplar::UNSIGNED_CHAR) {
    // When the input is of uint8 dtype, do matmul with fp8
    auto qm_lhs = poplar::createVariableMetadataTensor(
        graph(), lhs_quarter_format, a_scale, prog);
    a = graph().clone(poplar::QUARTER,
                      qm_lhs,
                      a_input,
                      debugContext("a"),
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    prog.add(poplar::program::Copy(a_input, a.reinterpret(UNSIGNED_CHAR)));
  } else {
    // When the input is of fp16 dtype, the weights needs to be converted to
    // fp16, and do matmul with fp16.
    a = a_input;
    b = popops::cast(graph(), b, poplar::HALF, prog, debugContext("b_cast"));
  }

  // Makes both input tensors at least rank 3
  //
  // This doesn't change the example inputs because the
  // rank is already more than 3.
  // a' := a = [2, 1, 4, 5, 1, 7, 8]
  // b' := b = [2, 3, 1, 5, 6, 8, 9]
  auto initReshapedTs = matInitReshape(matmul, a, b);

  // Match the ranks of both tensors by prefixing their shape with 1s
  //
  // This doesn't change the example inputs because the
  // inputs already have equal rank.
  // a' := a = [2, 1, 4, 5, 1, 7, 8]
  // b' := b = [2, 3, 1, 5, 6, 8, 9]
  auto matchedRankTs =
      matMatchRank(initReshapedTs.first, initReshapedTs.second);

  // Partition the group dimensions from the broadcast dimensions
  //
  // The shapes in the given example
  // let a = [2, 1, 4, 5, 1, 7, 8],
  //     b = [2, 3, 1, 5, 6, 8, 9]
  //                                G  |    B    |
  // a' := matDimshuffle(a, b) = [2, 5 | 1, 4, 1 | 7, 8]
  // b' := matDimshuffle(a, b) = [2, 5 | 3, 1, 6 | 8, 9]
  auto dimShuffledTs = matDimshuffle(matchedRankTs.first, matchedRankTs.second);

  // Reduce the group and broadcast dimensions down to a single dimension each
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //                                  G |  B |
  // a' := matReshapeGroups(a, b) = [10 |  4 | 7, 8]
  // b' := matReshapeGroups(a, b) = [10 | 18 | 8, 9]
  auto reshapedGroupsTs =
      matReshapeGroups(dimShuffledTs.first, dimShuffledTs.second);

  // Combine the broadcast dimension into the matrix row or column dimension as
  // appropriate
  //
  // The shapes in the given example
  // let a = [10,  4, 7, 8],
  //     b = [10, 18, 8, 9]
  //                                  G
  // a' := matReshapeGroups(a, b) = [10 | 28,   8]
  // b' := matReshapeGroups(a, b) = [10 |  8, 162]
  auto combinedBroadcastTs =
      matCombineBroadcastDims(reshapedGroupsTs.first, reshapedGroupsTs.second);

  // Perform the grouped matmul
  //
  // The shapes in the given example
  // let a = [10, 28,   8],
  //     b = [10,  8, 162]
  //                        G |  M   N
  // o' := matmul(a, b) = [10 | 28, 162]

  auto opts = dv_p->lowering().matmulOptions;
  appendPoplarOptionsForOp(matmul, opts);

  auto cacheSize = dv_p->matmulCache.size();

  auto outTensor =
      poplin::matMulGrouped(graph(),                    // graph
                            combinedBroadcastTs.first,  // A
                            combinedBroadcastTs.second, // B
                            prog,                       // prog
                            HALF,
                            debugContext("matmulGrouped"), // debugContext
                            opts,                          // options
                            &dv_p->matmulCache);           // cache
  // verifyCacheSizeUnchanged(cacheSize);
  // Log the report plan
  std::stringstream ss;
  poplin::matMulGroupedReportPlan(ss,
                                  graph(),
                                  combinedBroadcastTs.first.elementType(),
                                  outTensor.elementType(),
                                  combinedBroadcastTs.first.shape(),
                                  combinedBroadcastTs.second.shape(),
                                  opts,
                                  &dv_p->matmulCache);
  logging::opx::debug("Grouped Matmul {} plan", op_p->str());
  logging::log(logging::Module::opx, logging::Level::Debug, ss.str());

  // Split the broadcast dimensions from the rows and columns
  //
  // The shapes in the given example
  // let a = [10,  4, 7, 8],
  //     b = [10, 18, 8, 9]
  //     o = [10, 28, 162]
  //                                          G | B1 | M | B2 | N
  // o' := matSplitBroadcastDims(o, a, b) = [10 |  4 | 7 | 18 | 9]
  outTensor = matSplitBroadcastDims(
      outTensor, reshapedGroupsTs.first, reshapedGroupsTs.second);
  // Shuffle the column broadcast dim forward
  //
  // The shapes in the given example
  //     o = [10, 4, 7, 18, 9]
  //                                    G | B1 B2 | M  N
  // o' := matUnDimShuffle(o, a, b) = [10 | 4, 18 | 7, 9]
  outTensor = matUnDimShuffle(outTensor);

  // Expand the broadcast dimensions back to their original shape
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 4, 18, 7, 9]
  //                                           G |    B1   |    B2   | M  N
  // o' := matExpandBroadcastDims(o, a, b) = [10 | 1, 4, 1 | 3, 1, 6 | 7, 9]
  outTensor = matExpandBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);
  // Interleave the broadcast dimensions that should be squeezed
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 1, 4, 1, 3, 1, 6, 7, 9]
  //                                               G |         B        | M  N
  // o' := matInterleaveBroadcastDims(o, a, b) = [10 | 1, 3, 4, 1, 1, 6 | 7, 9]
  outTensor = matInterleaveBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  // Squeeze the broadcast dimensions
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 1, 3, 4, 1, 1, 6, 7, 9]
  //                                            G |    B    | M  N
  // o' := matSqueezeBroadcastDims(o, a, b) = [10 | 3, 4, 6 | 7, 9]
  outTensor = matSqueezeBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  // Expand the group dimensions
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 3, 4, 6, 7, 9]
  //                                        G  |    B    | M  N
  // o' := matExpandGroupDims(o, a, b) = [2, 5 | 3, 4, 6 | 7, 9]
  outTensor =
      matExpandGroupDims(outTensor, dimShuffledTs.first, dimShuffledTs.second);

  // Shuffle the group dimensions back into place
  //
  // The shapes in the given example
  // let a = [2, 1, 4, 5, 1, 7, 8],
  //     b = [2, 3, 1, 5, 6, 8, 9]
  //     o = [2, 5, 3, 4, 6, 7, 9]
  //                                                     | M  N
  // o' := matShuffleGroupDims(o, a, b) = [2, 3, 4, 5, 6 | 7, 9]
  outTensor =
      matShuffleGroupDims(outTensor, matchedRankTs.first, matchedRankTs.second);

  setOutTensor(0, outTensor.reshape(matmul.outInfo(0).shape_szt()));
}

std::vector<std::size_t> FP8MatMulOpx::getOutputShape() const {
  auto matmul = getMatMulOp();
  return FP8MatMulOpx::onnxShapeToPoplar(matmul->outInfo(0).shape());
}

// static popart::RegisterShapeInferenceFunction
//     FP8MatMulOpShapeInference(FP8MatMulOpId, [](auto &ctx) {
//       auto shape_LHS                     = ctx.inInfo(0).shape();
//       auto shape_RHS                     = ctx.inInfo(1).shape();
//       shape_LHS.at(shape_LHS.size() - 1) = shape_RHS.at(shape_RHS.size() - 1);
//       ctx.outInfo(0)                     = {DataType::FLOAT16, shape_LHS};
//     });

namespace {
OpxCreator<FP8MatMulOpx> FP8MatMulOpxCreator({FP8MatMulOpId});
} // namespace

} // namespace custom_ops
} // namespace compiler
} // namespace poprt
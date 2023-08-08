// Copyright (c) 2022 Graphcore Ltd.
#include <poplar/MetadataCreation.hpp>
#include <poplar/Program.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Tensor.hpp>
#include <popops/Cast.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/shapeinference.hpp>
#include <popart/attributes.hpp>
#include <popart/datatype.hpp>
#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/tensorinfo.hpp>
#include <poplar/Type.hpp>
#include <string>

#include "cast_to_uint8.hpp"

namespace poprt {
namespace compiler {
namespace custom_ops {

const popart::OperatorIdentifier castToUint8OpId = {"ai.graphcore",
                                                    "CastToUint8WithoutShapeInfer",
                                                    1};

CastToUint8Op::CastToUint8Op(const popart::OperatorIdentifier &_opid,
                             const popart::Op::Settings &settings_,
                             const Fp8Format fp8Format_)
    : Op(_opid, settings_), fp8Format(fp8Format_) {}

std::unique_ptr<popart::Op> CastToUint8Op::clone() const {
  return std::make_unique<CastToUint8Op>(*this);
}

void CastToUint8Op::setup() {
  auto info = inInfo(getInIndex());
  // Change data type
  info.set(popart::DataType::UINT8);
  outInfo(getOutIndex()) = info;
}

bool CastToUint8Op::canBeReplacedByIdentity() const {
  if (!hasInput(getInIndex())) {
    // Cannot determine whether Op can be replaced by identity, as its input
    // is not connected. Return default false.
    return false;
  }

  return inInfo(getInIndex()).dataType() == popart::DataType::UINT8;
}

void CastToUint8Op::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  popart::Op::appendOutlineAttributes(os);
  os.appendAttribute("fp8Format", int(fp8Format));
}

namespace {

static popart::OpDefinition::DataTypes T1 = {popart::DataType::FLOAT16};
static popart::OpDefinition::DataTypes T2 = {popart::DataType::UINT8};
static popart::OpDefinition::DataTypes T3 = {popart::DataType::INT32};

static popart::OpDefinition castToUint8OpDef(
    {popart::OpDefinition::Inputs({{"input", T1}, {"scale", T3}}),
     popart::OpDefinition::Outputs({{"output", T2}}),
     popart::OpDefinition::Attributes({
         {"fp8Format", {"*"}},
     })});

static popart::OpCreator<CastToUint8Op> castOpCreator(
    popart::OpDefinitions({
        {castToUint8OpId, castToUint8OpDef},
    }),
    [](const popart::OpCreatorInfo &info) {
      CastToUint8Op::Fp8Format fp8Format = CastToUint8Op::Fp8Format::F143;

      if (info.attributes.hasAttribute("fp8Format")) {
        std::string format =
            info.attributes.getAttribute<popart::Attributes::String>(
                "fp8Format");
        if (format == "F152") {
          fp8Format = CastToUint8Op::Fp8Format::F152;
        } else if (format == "F143") {
          fp8Format = CastToUint8Op::Fp8Format::F143;
        } else {
          throw popart::error(
              "Unsupport fp8 meata data {}, it must be F143 or F152", format);
        }
      }

      return std::make_unique<CastToUint8Op>(
          info.opid, info.settings, fp8Format);
    },
    true);
} // namespace

CastToUint8Opx::CastToUint8Opx(popart::Op *op, popart::popx::Devicex *devicex)
    : popart::popx::Opx(op, devicex) {
  verifyOp<CastToUint8Op>(op);
}

void CastToUint8Opx::grow(poplar::program::Sequence &prog) const {
  auto &cast = getOp<CastToUint8Op>();
  auto input = getInTensor(CastToUint8Op::getInIndex());
  auto scale = getInTensor(CastToUint8Op::getScaleIndex())[0];

  auto meta_data_format = cast.getFP8Format();
  poplar::QuarterMetadata::Format fp8_format =
      poplar::QuarterMetadata::Format::F143;
  if (meta_data_format == CastToUint8Op::Fp8Format::F152) {
    fp8_format = poplar::QuarterMetadata::Format::F152;
  }

  auto qm =
      poplar::createVariableMetadataTensor(graph(), fp8_format, scale, prog);

  auto res_cast_fp8 =
      popops::cast(graph(), input, poplar::QUARTER, qm, prog, debugContext());

  auto out = res_cast_fp8.reinterpret(poplar::UNSIGNED_CHAR);
  setOutTensor(CastToUint8Op::getOutIndex(), out);
}

// static popart::RegisterShapeInferenceFunction
//     CastToUint8OpShapeInference(castToUint8OpId, [](auto &ctx) {
//       ctx.outInfo(0) = {popart::DataType::UINT8, ctx.inInfo(0).shape()};
//     });

namespace {
popart::popx::OpxCreator<CastToUint8Opx>
    CastToUint8OpxCreator({castToUint8OpId});
} // namespace

} // namespace custom_ops
} // namespace compiler
} // namespace poprt
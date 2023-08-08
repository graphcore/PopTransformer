// Copyright (c) 2023, Graphcore Ltd.
#ifdef __IPU__
#include <ipu_intrinsics>
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#else
#error Not supported on IPU Model
#endif
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template <class R, class T> R as(T x) { return *reinterpret_cast<R *>(&x); }

class GetInt4ToHalf : public Vertex {
public:
  GetInt4ToHalf();

  Input<int> size;
  Input<Vector<int8_t, ONE_PTR, 4>> input;
  Output<Vector<half, ONE_PTR, 8>> highData;
  Output<Vector<half, ONE_PTR, 8>> lowData;
  Input<Vector<half, ONE_PTR, 8>> scale;
  // static const bool isExternalCodelet = true;

  bool compute() {
    for (int j = 0; j < size; j++) {
      int8_t in     = input[j];
      half scaleVal = scale[j];
      lowData[j]    = half(char(in << 4) >> 4) * scaleVal;
      highData[j]   = half(in >> 4) * scaleVal;
    }
    return true;
  }
};

class GetInt4ToHalf1 : public Vertex {
public:
  GetInt4ToHalf1();

  Input<int> size;
  Input<Vector<half, ONE_PTR, 8>> table; // shape=(512)
  Input<Vector<int8_t, ONE_PTR, 8>> input;
  Output<Vector<half, ONE_PTR, 8>> output;
  // static const bool isExternalCodelet = true;

  bool compute() {
    auto sizeR  = size >> 1;
    auto sizeRL = sizeR << 1;

    half2 *tableV  = (half2 *)(&(table[0]));
    char2 *inputV  = (char2 *)(&(input[0]));
    half4 *outputV = (half4 *)(&(output[0]));

    for (int i = 0; i < sizeR; i++) {
      auto in      = inputV[i];
      int in1      = uint8_t(in[0]);
      half2 value1 = tableV[in1];
      int in2      = uint8_t(in[1]);
      half2 value2 = tableV[in2];
      outputV[i]   = as<half4>(float2{as<float>(value1), as<float>(value2)});
    }

    half2 *output2V = (half2 *)(&(output[0]));
    for (int j = sizeRL; j < size; j++) {
      uint8_t idx = uint8_t(input[j]);
      output2V[j] = tableV[idx];
    }
    return true;
  }
};
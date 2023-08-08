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

class GetStepBps : public Vertex {
public:
  GetStepBps();

  Input<int> maxBps;
  Input<Vector<int>> step;
  Output<Vector<int>> stepTrue;
  Output<Vector<int>> bpsIdx;

  bool compute() {
    int stepVal = step[0];
    if (stepVal > maxBps) {
      stepTrue[0] = int{1};
    }
    bpsIdx[0] = int(stepVal % maxBps);

    return true;
  }
};

class GetkvCachesInit_half : public Vertex {
public:
  GetkvCachesInit_half();

  Input<int> size;
  Input<Vector<int>> step;
  InOut<Vector<half, ONE_PTR, 8>> dataSlice;

  bool compute() {
    auto sizeR  = size >> 2;
    auto sizeRL = sizeR << 2;

    half4 *dataPtrV = (half4 *)(&(dataSlice[0]));
    half *dataPtr   = (half *)dataPtrV;

    if (step[0] == 0) {
      auto zeroV = half4{0, 0, 0, 0};
      auto zero  = half{0};
      for (int i = 0; i < sizeR; ++i) {
        ipu::store_postinc(&dataPtrV, zeroV, 1);
      }
      for (int j = sizeRL; j < size; j++) {
        dataPtr[j] = zero;
      }
    }
    return true;
  }
};

class GetkvCachesInit_char : public Vertex {
public:
  GetkvCachesInit_char();

  Input<int> size;
  Input<Vector<int>> step;
  InOut<Vector<char, ONE_PTR, 8>> dataSlice;

  bool compute() {
    auto sizeR  = size >> 3;
    auto sizeRL = sizeR << 3;

    half4 *dataPtrV = (half4 *)(&(dataSlice[0]));
    char *dataPtr   = (char *)dataPtrV;

    if (step[0] == 0) {
      auto zeroV = half4{0, 0, 0, 0};
      auto zero  = char{0};
      for (int i = 0; i < sizeR; ++i) {
        ipu::store_postinc(&dataPtrV, zeroV, 1);
      }
      for (int j = sizeRL; j < size; j++) {
        dataPtr[j] = zero;
      }
    }
    return true;
  }
};

class GetBpsKvCache_half : public Vertex {
public:
  GetBpsKvCache_half();

  Input<int> size;
  Input<int> idx;
  Input<Vector<int>> bpsIdx;
  InOut<Vector<half, ONE_PTR, 8>> kvCache;
  InOut<Vector<half, ONE_PTR, 8>> kvCacheClone;
  InOut<Vector<half, ONE_PTR, 8>> output;

  bool compute() {
    auto sizeR  = size >> 2;
    auto sizeRL = sizeR << 2;

    half4 *kvCacheV      = (half4 *)(&(kvCache[0]));
    half4 *kvCacheCloneV = (half4 *)(&(kvCacheClone[0]));
    half4 *outputV       = (half4 *)(&(output[0]));

    auto negBpsIdx = bpsIdx[0] - idx;
    if (negBpsIdx != 0) {
      return true;
    }
    for (int i = 0; i < sizeR; ++i) {
      auto val = kvCacheCloneV[i];
      ipu::store_postinc(&kvCacheV, val, 1);
      ipu::store_postinc(&outputV, val, 1);
    }
    for (int j = sizeRL; j < size; j++) {
      auto val   = kvCacheClone[j];
      kvCache[j] = val;
      output[j]  = val;
    }
    return true;
  }
};

class GetBpsKvCache_char : public Vertex {
public:
  GetBpsKvCache_char();

  Input<int> size;
  Input<int> idx;
  Input<Vector<int>> bpsIdx;
  InOut<Vector<char, ONE_PTR, 8>> kvCache;
  InOut<Vector<char, ONE_PTR, 8>> kvCacheClone;
  InOut<Vector<char, ONE_PTR, 8>> output;

  bool compute() {
    auto sizeR  = size >> 3;
    auto sizeRL = sizeR << 3;

    half4 *kvCacheV      = (half4 *)(&(kvCache[0]));
    half4 *kvCacheCloneV = (half4 *)(&(kvCacheClone[0]));
    half4 *outputV       = (half4 *)(&(output[0]));

    auto negBpsIdx = bpsIdx[0] - idx;
    if (negBpsIdx != 0) {
      return true;
    }
    for (int i = 0; i < sizeR; ++i) {
      auto val = kvCacheCloneV[i];
      ipu::store_postinc(&kvCacheV, val, 1);
      ipu::store_postinc(&outputV, val, 1);
    }
    for (int j = sizeRL; j < size; j++) {
      auto val   = kvCacheClone[j];
      kvCache[j] = val;
      output[j]  = val;
    }
    return true;
  }
};

class GetBpsKvCacheLastDim_half : public Vertex {
public:
  GetBpsKvCacheLastDim_half();

  Input<int> size;
  Input<int> idx;
  Input<Vector<int>> bpsIdx;
  InOut<Vector<half, ONE_PTR, 8>> kvCache;
  InOut<Vector<half, ONE_PTR, 8>> output;

  bool compute() {
    auto negBpsIdx = bpsIdx[0] - idx;
    if (negBpsIdx != 0) {
      return true;
    }

    auto sizeR  = size >> 2;
    auto sizeRL = sizeR << 2;

    half4 *kvCacheV = (half4 *)(&(kvCache[0]));
    half4 *outputV  = (half4 *)(&(output[0]));
    for (int j = size - 1; j >= sizeRL; j--) {
      auto val   = kvCache[j - 1];
      kvCache[j] = val;
      output[j]  = val;
    }
    for (int i = sizeR - 1; i > 0; --i) {
      auto lastVal = kvCacheV[i - 1];
      auto val     = kvCacheV[i];
      val          = half4{lastVal[3], val[0], val[1], val[2]};
      kvCacheV[i]  = val;
      outputV[i]   = val;
    }

    auto inputVal = outputV[0];
    auto val      = kvCacheV[0];
    auto firstVal = half4{inputVal[0], val[0], val[1], val[2]};
    outputV[0]    = firstVal;
    kvCacheV[0]   = firstVal;

    return true;
  }
};

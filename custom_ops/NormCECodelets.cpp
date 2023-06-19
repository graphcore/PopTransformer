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

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#include <print.h>

using namespace poplar;

static constexpr auto SPAN        = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR     = poplar::VectorLayout::ONE_PTR;
static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

template<typename T>
struct FloatDef{
};

template<>
struct FloatDef<float>{
  static inline const int         kSftBits        = 1;
  static inline const int         kStep           = (1 << kSftBits);
  static inline const int         kPlus           = kStep - 1;
  static inline const float       kZero           = 0.0f;
  static inline const float       kHalf           = 0.5f;
  static inline const float       kOne            = 1.0f;
  static inline constexpr float2  kZeroV          = { 0.0f, 0.0f };
  typedef   float2                FVType;
  static inline FVType            setV(float const& a)
  {
    FVType x = { a, a };
    return x;
  }; 
};

template<>
struct FloatDef<half>{
  static inline const int         kSftBits        = 2;
  static inline const int         kStep           = (1 << kSftBits);
  static inline const int         kPlus           = kStep - 1;
  static inline const half        kZero           = 0.0f;
  static inline const half        kHalf           = 0.5f;
  static inline const half        kOne            = 1.0f;
  static inline constexpr half4   kZeroV          = { 0.0f, 0.0f, 0.0f, 0.0f };
  typedef   half4                 FVType;
  static inline FVType            setV(half const& a)
  {
    FVType x = { a, a, a, a };
    return x;
  };
};

class GetInPlaceNorm : public Vertex {
public:
  GetInPlaceNorm();

  Input<int> size;
  Input<float> epsilon;
  Input<Vector<half, ONE_PTR, 8>> input;
  Input<Vector<half, ONE_PTR, 8>> bias;
  Input<Vector<half, ONE_PTR, 8>> scale;
  Output<Vector<half, ONE_PTR, 8>> output;
  Input<Vector<float>> mean;
  Input<Vector<float>> power;

  bool compute() {
    auto sizeR      = size >> 2;
    auto sizeRL     = sizeR << 2;
    float meanFVal  = mean[0];
    float powerFVal = power[0];
    powerFVal       = powerFVal - meanFVal * meanFVal;
    powerFVal       = powerFVal + float(epsilon);
    half meanVal    = half(meanFVal);
    half invStdDev  = half(ipu::rsqrt(powerFVal));

    half4 *inputPtrV  = (half4 *)(&(input[0]));
    half4 *biasPtrV   = (half4 *)(&(bias[0]));
    half4 *scalePtrV  = (half4 *)(&(scale[0]));
    half4 *outputPtrV = (half4 *)(&(output[0]));

    half *inputPtr  = (half *)inputPtrV;
    half *biasPtr   = (half *)biasPtrV;
    half *scalePtr  = (half *)scalePtrV;
    half *outputPtr = (half *)outputPtrV;

    for (int i = 0; i < sizeR; ++i) {
      half4 curVal   = inputPtrV[i];
      half4 biasVal  = biasPtrV[i];
      half4 scaleVal = scalePtrV[i];
      outputPtrV[i]  = (curVal - meanVal) * invStdDev * scaleVal + biasVal;
    }
    for (int j = sizeRL; j < size; j++) {
      half curVal   = inputPtr[j];
      half biasVal  = biasPtr[j];
      half scaleVal = scalePtr[j];
      outputPtr[j]  = (curVal - meanVal) * invStdDev * scaleVal + biasVal;
    }
    return true;
  }
};

#if 0
void normKernel(half const*  src, 
                half*        dst, 
                float*       meanDevPair, 
                float const* invScaleEps, 
                int          inner_size,
                half const*  scale,
                half const*  bias){
  float  sum      = 0.0f;
  float  sqrSum   = 0.0f;
  float  invScale = invScaleEps[0];
  float  eps      = invScaleEps[1];
  for(int j = 0 ; j < inner_size ; j ++){
    sum    += (float)src[j];
    sqrSum += (float)(src[j] * src[j]);
  }
  float avg         = sum * invScale;
  half  curMean     = avg;
  float avgT        = float(curMean);
  float avgSqr      = sqrSum * invScale;
  float varianceEst = avgSqr - avgT * avgT;
  if (varianceEst < 0.0f)
    varianceEst = 0.0f;
  varianceEst += eps;
  float invStdDev = 1.0f / sqrt(varianceEst);
  
  half curInvStdDev = invStdDev;
  for(int j = 0 ; j < inner_size ; j ++){
    dst[j] = (src[j] - curMean) * curInvStdDev;
  }
  meanDevPair[0] = avg;
  meanDevPair[1] = invStdDev;
};
#else
void normKernel(float const*  src, 
                float*        dst, 
                float*        meanDevPair, 
                float const*  invScaleEps, 
                int           inner_size,
                float const*  scale,
                float const*  bias);
void normKernel(half const*  src, 
                half*        dst, 
                float*       meanDevPair, 
                float const* invScaleEps, 
                int          inner_size,
                half const*  scale,
                half const*  bias);
#endif

#if 0
void arrangeKernel(float const* src, float* mean, float* invStdDev, int data_size){
  
}

void arrangeKernel(float const* src, half* mean, half* invStdDev, int data_size){
  for(int i = 0 ; i < data_size ; i ++){
    mean[i]      = src[2 * i];
    invStdDev[i] = src[2 * i + 1];
  }
}
#else
void arrangeKernel(float const* src, float* mean, float* invStdDev, int data_size);
void arrangeKernel(float const* src, half* mean, half* invStdDev, int data_size);
#endif

#if 0
void normInfKernel(half const*  src, 
                   half*        dst, 
                   float const* invScaleEps, 
                   int          inner_size,
                   half const*  scale,
                   half const*  bias){
  float  sum      = 0.0f;
  float  sqrSum   = 0.0f;
  float  invScale = invScaleEps[0];
  float  eps      = invScaleEps[1];
  half   gamma    = scale[0];
  half   beta     = bias[0];
  for(int j = 0 ; j < inner_size ; j ++){
    sum    += (float)src[j];
    sqrSum += (float)(src[j] * src[j]);
  }
  float avg         = sum * invScale;
  half  curMean     = avg;
  float avgT        = float(curMean);
  float avgSqr      = sqrSum * invScale;
  float varianceEst = avgSqr - avgT * avgT;
  if (varianceEst < 0.0f)
    varianceEst = 0.0f;
  varianceEst += eps;
  float invStdDev = 1.0f / sqrt(varianceEst);
  
  half curInvStdDev = invStdDev;
  for(int j = 0 ; j < inner_size ; j ++){
    dst[j] = ((src[j] - curMean) * curInvStdDev) * gamma + beta;
  }
};
void normInfReduceKernel(half const* src,
                         float*      dst,
                         int         data_size){
  float  sum      = 0.0f;
  float  sqrSum   = 0.0f;
  for(int j = 0 ; j < data_size ; j ++){
    sum    += (float)src[j];
    sqrSum += (float)(src[j] * src[j]);
  }
  dst[0] = sum;
  dst[1] = sqrSum;
}
void normInfNormalizeKernel(half const* src,
                            half const* scale,
                            half const* bias,
                            half*       dst,
                            int         data_size,
                            float       mean,
                            float       invStdDev){
  for(int j = 0 ; j < data_size ; j ++){
    dst[j] = ((src[j] - (half)mean) * (half)invStdDev) * scale[j] + bias[j];
  }
}
#else
void normInfKernel(half const*  src, 
                   half*        dst, 
                   float const* invScaleEps, 
                   int          inner_size,
                   half const*  scale,
                   half const*  bias);
void normInfKernelAligned16(half const*  src, 
                            half*        dst, 
                            float const* invScaleEps, 
                            int          inner_size,
                            half const*  scale,
                            half const*  bias);
#endif

#define ASM_EXTENRAL_CODE
#if defined(__IPU__) && defined(ASM_EXTENRAL_CODE)
#define ASM_EXTENRAL_CODE_ENABLE  true
#else
#define ASM_EXTENRAL_CODE_ENABLE  false
#endif

namespace celib{

template <typename T>
class NormCEVertex : public MultiVertex {
public:
  NormCEVertex();
  Input<Vector<T, ONE_PTR, 8>>         in_; 
  Input<Vector<T, ONE_PTR, 8>>         scale_;
  Input<Vector<T, ONE_PTR, 8>>         bias_;
  Output<Vector<T, ONE_PTR, 8>>        out_; 
  Output<Vector<float, ONE_PTR, 8>>    meanDevPair_; 
  int                                  inner_size_;
  const Vector<int, SPAN, 4>           work_size_;
  const Vector<int, SPAN, 4>           work_ofs_;
  float                                invScale_;
  float                                eps_;

  template<typename FType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static void proc(FType const*  src, 
                   FType*        dst, 
                   float*        meanDevPair, 
                   float const*  invScaleEps, 
                   int           inner_size,
                   FType const*  scale,
                   FType const*  bias){

  }

  template<typename FType, typename std::enable_if<std::is_same<FType, half>::value, void>::type* = nullptr>
  static void proc(FType const*  src, 
                   FType*        dst, 
                   float*        meanDevPair, 
                   float const*  invScaleEps, 
                   int           inner_size,
                   FType const*  scale,
                   FType const*  bias){
    normKernel(src, dst, meanDevPair, invScaleEps, inner_size, scale, bias);
  }

  bool compute(unsigned workerId) {
    if(work_size_[workerId] <= 0){
      return true;
    }

    T const*   src_ptr        = reinterpret_cast<T const*>(&(in_[0]));
    T const*   scale_ptr      = reinterpret_cast<T const*>(&(scale_[0]));
    T const*   bias_ptr       = reinterpret_cast<T const*>(&(bias_[0]));
    T*         dst_ptr        = reinterpret_cast<T*>(&(out_[0]));
    float*     mean_ptr       = reinterpret_cast<float*>(&(meanDevPair_[0]));
    int        cur_work_ofs   = work_ofs_[workerId];
    int        cur_work_size  = work_size_[workerId];
    float      invScaleEps[2] = { invScale_, eps_ };
    src_ptr   = src_ptr  + cur_work_ofs * inner_size_;
    dst_ptr   = dst_ptr  + cur_work_ofs * inner_size_;
    mean_ptr  = mean_ptr + (cur_work_ofs << 1);
    for(int i = 0 ; i < cur_work_size ; i ++){
      proc<T>(src_ptr, 
              dst_ptr, 
              mean_ptr,
              invScaleEps,
              inner_size_,
              scale_ptr,
              bias_ptr);
      src_ptr  += inner_size_;
      dst_ptr  += inner_size_;
      mean_ptr += 2;
    }
    return true;
  };
};

template class NormCEVertex<float>;
template class NormCEVertex<half>;

template <typename T>
class Float2RealVertex : public Vertex {
public:
  Float2RealVertex();
  Input<Vector<float, ONE_PTR, 8>>  meanDevPair_; 
  Output<Vector<T, ONE_PTR, 8>>     meanData_; 
  Output<Vector<T, ONE_PTR, 8>>     invStdDevData_;
  int                               data_size_;

  template<typename FType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static void proc(float const* src, FType* mean, FType* invStdDev, int data_size){
    arrangeKernel(src, mean, invStdDev, data_size);
  }

  template<typename FType, typename std::enable_if<std::is_same<FType, half>::value, void>::type* = nullptr>
  static void proc(float const* src, FType* mean, FType* invStdDev, int data_size){
    arrangeKernel(src, mean, invStdDev, data_size);
  }

  bool compute() {
    float const*  pair_ptr           = reinterpret_cast<float const*>(&(meanDevPair_[0]));
    T*            mean_data_ptr      = reinterpret_cast<T*>(&(meanData_[0]));
    T*            InvStdDev_data_ptr = reinterpret_cast<T*>(&(invStdDevData_[0]));
    proc<T>(pair_ptr, mean_data_ptr, InvStdDev_data_ptr, data_size_);
    return true;
  };
};

template class Float2RealVertex<float>;
template class Float2RealVertex<half>;

template <typename T, bool interleave_mem>
class NormCEInfVertex : public MultiVertex {
public:
  NormCEInfVertex();
  static constexpr unsigned inAligned = interleave_mem ? 16 : 8;
  Input<Vector<T, ONE_PTR, inAligned, interleave_mem>>  in_; 
  Input<Vector<T, ONE_PTR, 8>>    scale_;
  Input<Vector<T, ONE_PTR, 8>>    bias_;
  Output<Vector<T, ONE_PTR, 8>>   out_; 
  int                             inner_size_;
  const Vector<int, SPAN, 4>      work_size_;
  const Vector<int, SPAN, 4>      work_ofs_;
  float                           invScale_;
  float                           eps_;

  template<typename FType, bool Aligned16, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static void proc(FType const*  src, 
                   FType*        dst, 
                   float const*  invScaleEps, 
                   int           inner_size,
                   FType const*  scale,
                   FType const*  bias){

  }

  template<typename FType, bool Aligned16, typename std::enable_if<std::is_same<FType, half>::value&(Aligned16 == false), void>::type* = nullptr>
  static void proc(FType const*  src, 
                   FType*        dst, 
                   float const*  invScaleEps, 
                   int           inner_size,
                   FType const*  scale,
                   FType const*  bias){
    normInfKernel(src, dst, invScaleEps, inner_size, scale, bias);
  }

  template<typename FType, bool Aligned16, typename std::enable_if<std::is_same<FType, half>::value&(Aligned16 == true), void>::type* = nullptr>
  static void proc(FType const*  src, 
                   FType*        dst, 
                   float const*  invScaleEps, 
                   int           inner_size,
                   FType const*  scale,
                   FType const*  bias){
    normInfKernelAligned16(src, dst, invScaleEps, inner_size, scale, bias);
  }


  bool compute(unsigned workerId) {
    if(work_size_[workerId] <= 0){
      return true;
    }

    T const*   src_ptr        = reinterpret_cast<T const*>(&(in_[0]));
    T const*   scale_ptr      = reinterpret_cast<T const*>(&(scale_[0]));
    T const*   bias_ptr       = reinterpret_cast<T const*>(&(bias_[0]));
    T*         dst_ptr        = reinterpret_cast<T*>(&(out_[0]));
    int        cur_work_ofs   = work_ofs_[workerId];
    int        cur_work_size  = work_size_[workerId];
    float      invScaleEps[2] = { invScale_, eps_ };
    src_ptr   = src_ptr  + cur_work_ofs * inner_size_;
    dst_ptr   = dst_ptr  + cur_work_ofs * inner_size_;
    for(int i = 0 ; i < cur_work_size ; i ++){
      proc<T, interleave_mem>(src_ptr, 
                         dst_ptr, 
                         invScaleEps,
                         inner_size_,
                         scale_ptr,
                         bias_ptr);
      src_ptr += inner_size_;
      dst_ptr += inner_size_;
    }
    return true;
  };
};

template class NormCEInfVertex<float, true>;
template class NormCEInfVertex<float, false>;
template class NormCEInfVertex<half, true>;
template class NormCEInfVertex<half, false>;

template <typename T, int task_size, bool interleave>
#ifdef  ASM_EXTENRAL_CODE
class NormCEInfSplitVertex : public SupervisorVertexIf<true> {
#else
class NormCEInfSplitVertex : public Vertex {
#endif  
  NormCEInfSplitVertex();

  static constexpr unsigned inAligned = interleave ? 16 : 8;
  Input<Vector<T, COMPACT_PTR, inAligned, interleave>>  in_; 
  Input<Vector<T, COMPACT_PTR, 8>>                      scale_;
  Input<Vector<T, COMPACT_PTR, 8>>                      bias_;
  Output<Vector<T, COMPACT_PTR, 8>>                     out_; 
  const Vector<int, COMPACT_PTR, 4>                     work_size_;
  const Vector<int, COMPACT_PTR, 4>                     work_ofs_;
  int                                                   inner_size_;
  float                                                 invScale_;
  float                                                 eps_;

  static const bool isExternalCodelet = ASM_EXTENRAL_CODE_ENABLE;

  bool compute() 
  {
#ifndef ASM_EXTENRAL_CODE
    float  sum      = 0.0f;
    float  sqrSum   = 0.0f;
    float  invScale = invScaleEps[0];
    float  eps      = invScaleEps[1];
    for(int j = 0 ; j < inner_size ; j ++){
      sum    += (float)src[j];
      sqrSum += (float)(src[j] * src[j]);
    }
    float avg         = sum * invScale;
    half  curMean     = avg;
    float avgT        = float(curMean);
    float avgSqr      = sqrSum * invScale;
    float varianceEst = avgSqr - avgT * avgT;
    if (varianceEst < 0.0f)
      varianceEst = 0.0f;
    varianceEst += eps;
    float invStdDev = 1.0f / sqrt(varianceEst);
    
    half curInvStdDev = invStdDev;
    for(int j = 0 ; j < inner_size ; j ++){
      dst[j] = (src[j] - curMean) * curInvStdDev;
    }
#endif
    return true;
  };
};

template class NormCEInfSplitVertex<float, 1, false>;
template class NormCEInfSplitVertex<float, 2, false>;
template class NormCEInfSplitVertex<float, 3, false>;
template class NormCEInfSplitVertex<float, 1, true>;
template class NormCEInfSplitVertex<float, 2, true>;
template class NormCEInfSplitVertex<float, 3, true>;
template class NormCEInfSplitVertex<half, 1, false>;
template class NormCEInfSplitVertex<half, 2, false>;
template class NormCEInfSplitVertex<half, 3, false>;
template class NormCEInfSplitVertex<half, 1, true>;
template class NormCEInfSplitVertex<half, 2, true>;
template class NormCEInfSplitVertex<half, 3, true>;

}

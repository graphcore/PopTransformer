// Copyright (c) 2023, Graphcore Ltd,
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

#ifdef __IPU__
#include <ipu_vector_math>
#else
#error Not supported on IPU Model
#endif
#include <poplar/Vertex.hpp>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

class KVCache : public Vertex {
public:
  KVCache();

  Input<int> size;
  Input<Vector<unsigned char>> input;
  Input<Vector<unsigned char, ONE_PTR, 8>> kvCacheClone;
  Output<Vector<unsigned char, ONE_PTR, 8>> kvCache;

  bool compute() {
    auto sizeR = size >> 2;
    auto sizeRL = sizeR << 2;

    uchar4* dataPtrV = (uchar4*)(&(kvCache[0]));
    unsigned char* dataPtr = (unsigned char*)dataPtrV;
    uchar4* clonePtrV = (uchar4*)(&(kvCacheClone[0]));
    unsigned char* clonePtr = (unsigned char*)clonePtrV;

    uchar4 curVal = clonePtrV[0];
    dataPtrV[0] = {input[0],
                   curVal[0],
                   curVal[1],
                   curVal[2]};

    for (int i = 1; i < sizeR; ++i) {
      unsigned char lastVal = clonePtr[i * 4 - 1];
      uchar4 curVal = clonePtrV[i];
      curVal = {lastVal,
                curVal[0],
                curVal[1],
                curVal[2]};
      dataPtrV[i] = curVal;
    }
    for (int j = sizeRL; j < size; j++) {
      unsigned char curVal = dataPtr[j];
      dataPtr[j] = clonePtr[j - 1];
    }
    return true;
  }
};

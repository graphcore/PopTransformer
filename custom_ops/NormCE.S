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


#include "poplar/TileConstants.hpp"

#define FLOAT_1_0 0x3F800000
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

  .text
	.allow_optimizations
	.file	"4"

	.section	.text._Z10normKernelPKDhPDhPfPKfiS0_S0_,"ax",@progbits
	.globl	_Z10normKernelPKDhPDhPfPKfiS0_S0_              # -- Begin function _Z10normKernelPKDhPDhPfPKfiS0_S0_
	.p2align	3
	.type	_Z10normKernelPKDhPDhPfPKfiS0_S0_,@function
_Z10normKernelPKDhPDhPfPKfiS0_S0_:                       # @_Z10normKernelPKDhPDhPfPKfiS0_S0_
.LnormKernelPKDhPDhPfPKfiS0_S0_begin:
  //m0 -> src_ptr
  //m1 -> dst_ptr
  //m2 -> mean_ptr
  //m3 -> const_ptr
  //m4 -> inner_size
  //   -> scale_
  //   -> bias_
  ld32                $m4,              $m11,         0
  ld32                $m5,              $m11,         1
  ld32                $m6,              $m11,         2
  sub                 $m11,             $m11,         28
  st32                $m0,              $m11,         4
  st32                $m7,              $m11,         5
  st32                $m8,              $m11,         6
  mov                 $m7,              $m5
  mov                 $m8,              $m6
  mov                 $a2:3,            $a14:15
  mov                 $a4:5,            $a14:15
  shr                 $m6,              $m4,          2
  shl                 $m6,              $m6,          2
  sub                 $m5,              $m4,          $m6
  {
    brz               $m6,              normKernelPKDhPDhPfPKfiS0_S0_statistic_proc_tail
    setzi             $a6,              ZAACC_BITMASK
  }
  {
    shr               $m6,              $m6,          2
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  sub                 $m6,              $m6,          1
  ld64step            $a0:1,            $m15,         $m0+=,            1
.align 8
  { 
    rpt               $m6,             ((__normKernelPKDhPDhPfPKfiS0_S0_statistic_sqsum_loop_end - __normKernelPKDhPDhPfPKfiS0_S0_statistic_sqsum_loop_start) / 8) - 1;
    fnop
  }
__normKernelPKDhPDhPfPKfiS0_S0_statistic_sqsum_loop_start:
  {
    ld64step          $a0:1,            $m15,             $m0+=,       1
    f16v8sqacc        $a0:3
  }
__normKernelPKDhPDhPfPKfiS0_S0_statistic_sqsum_loop_end:
  {
    nop
    f16v8sqacc        $a0:3
  }
  {
    nop
    f32v2gina         $a0:1,            $a14:15,        0
  }
  {
    nop
    f32v2gina         $a2:3,            $a14:15,        0
  }
  {
    ld32              $m0,              $m11,           4
    f32v2add          $a0:1,            $a0:1,          $a2:3
  }
  {
    ld64step          $a4:5,            $m15,           $m0+=,            1
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  f32add              $a0,              $a0,            $a1     //sqrSum
  mov                 $a6:7,            $a14:15
.align 8
  { 
    rpt               $m6,             ((__normKernelPKDhPDhPfPKfiS0_S0_statistic_sum_loop_end - __normKernelPKDhPDhPfPKfiS0_S0_statistic_sum_loop_start) / 8) - 1;
    fnop
  }
__normKernelPKDhPDhPfPKfiS0_S0_statistic_sum_loop_start:
  {
    ld64step          $a4:5,            $m15,             $m0+=,       1
    f16v8acc          $a4:7
  }
__normKernelPKDhPDhPfPKfiS0_S0_statistic_sum_loop_end:
  {
    nop
    f16v8acc          $a4:7
  }
  {
    nop
    f32v2gina         $a4:5,            $a14:15,        0
  }
  {
    nop
    f32v2gina         $a6:7,            $a14:15,        0
  }
  f32v2add            $a4:5,            $a4:5,          $a6:7
  f32add              $a1,              $a4,            $a5     //Sum
  brz                 $m5,              normKernelPKDhPDhPfPKfiS0_S0_proc_data
normKernelPKDhPDhPfPKfiS0_S0_statistic_proc_tail:
  sub                 $m5,              $m5,            1
normKernelPKDhPDhPfPKfiS0_S0_statistic_proc_tail_loop:
  ldb16step           $a2,              $m15,           $m0+=,       1
  f16tof32            $a4,              $a2
  f32mul              $a5,              $a4,            $a4
  f32add              $a0,              $a0,            $a5          //sqrsum
  f32add              $a1,              $a1,            $a4          //sum
  brnzdec             $m5,              normKernelPKDhPDhPfPKfiS0_S0_statistic_proc_tail_loop
normKernelPKDhPDhPfPKfiS0_S0_proc_data:
  ld64                $a2:3,            $m3,            $m15,        0
  f32mul              $a0,              $a0,            $a2          //avgSqr
  f32mul              $a1,              $a1,            $a2          //avg
  {
    st32              $a1,              $m2,            0            //a1 -> mean
    f32tof16          $a4,              $a1                          //
  }
  f16tof32            $a4,              $a4
  f32mul              $a4,              $a4,            $a4          //avg * avg

  f32sub              $a4,              $a0,            $a4          //avgSqr - avg * avg
  f32max              $a4,              $a4,            $a15         //max(varianceEst, 0.0f)
  f32add              $a4,              $a4,            $a3          // += eps
  ldconst             $a0,              FLOAT_1_0
  f32sqrt             $a4,              $a4
  f32div              $a0,              $a0,            $a4
  {
    st32              $a0,              $m2,            1            //a0 -> invStdDev
    f32tof16          $a2,              $a1
  }
  mov                 $a3,              $a2
  f32tof16            $a4,              $a0            //stdDev
  mov                 $a5,              $a4

  ld32                $m0,              $m11,         4
  shr                 $m6,              $m4,          2
  shl                 $m6,              $m6,          2
  sub                 $m5,              $m4,          $m6
  brz                 $m6,              normKernelPKDhPDhPfPKfiS0_S0_calc_tail
  shr                 $m6,              $m6,          2
  ld64step            $a0:1,            $m15,         $m0+=,       1
  {
    sub               $m6,              $m6,          1
    f16v4sub          $a0:1,            $a0:1,        $a2:3
  }
.align 8
  { 
    rpt               $m6,             ((__normKernelPKDhPDhPfPKfiS0_S0_calc_inner_loop_end - __normKernelPKDhPDhPfPKfiS0_S0_calc_inner_loop_start) / 8) - 1;
    fnop
  }
__normKernelPKDhPDhPfPKfiS0_S0_calc_inner_loop_start:
  {
    ld64step          $a6:7,           $m15,          $m7+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a4:5
  }
  {
    ld64step          $a6:7,           $m15,          $m8+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a6:7
  }
  {
    ld64step          $a6:7,           $m15,          $m0+=,      1
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m1+=,      1
    f16v4sub          $a0:1,           $a6:7,         $a2:3
  }
__normKernelPKDhPDhPfPKfiS0_S0_calc_inner_loop_end:
  {
    ld64step          $a6:7,           $m15,          $m7+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a4:5
  }
  {
    ld64step          $a6:7,           $m15,          $m8+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a6:7
  }
  {
    nop
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m1+=,      1
    fnop
  }
  brz                 $m5,             normKernelPKDhPDhPfPKfiS0_S0_proc_end
normKernelPKDhPDhPfPKfiS0_S0_calc_tail:
  sub                 $m5,             $m5,           1
  mov                 $m4,             $m10
  mov                 $m2,             $m0
  mov                 $m0,             $m1
  mov                 $m6,             $m1
normKernelPKDhPDhPfPKfiS0_S0_calc_tail_loop:
  ldb16step           $a0,             $m15,          $m2+=,      1
  {
    ldb16step         $a5,             $m15,          $m7+=,      1
    f16v2sub          $a0,             $a0,           $a2
  }
  {
    ldb16step         $a6,             $m15,          $m8+=,      1
    f16v2mul          $a0,             $a0,           $a4
  }
  f16v2mul            $a0,             $a0,           $a5
  f16v2add            $a0,             $a0,           $a6
  call                $m10,            __st16f
  add                 $m6,             $m6,           2
  mov                 $m0,             $m6
  brnzdec             $m5,             normKernelPKDhPDhPfPKfiS0_S0_calc_tail_loop
  mov                 $m10,            $m4
normKernelPKDhPDhPfPKfiS0_S0_proc_end:
  ld32                $m7,             $m11,          5
  ld32                $m8,             $m11,          6
  add                 $m11,            $m11,          28
	br                  $m10
.LnormKernelPKDhPDhPfPKfiS0_S0_end:
	.size	_Z10normKernelPKDhPDhPfPKfiS0_S0_, .LnormKernelPKDhPDhPfPKfiS0_S0_end-_Z10normKernelPKDhPDhPfPKfiS0_S0_
	.section	.stack_sizes,"o",@progbits,.text._Z10normKernelPKDhPDhPfPKfiS0_S0_
	.long	.LnormKernelPKDhPDhPfPKfiS0_S0_begin
	.byte	28
	.section	.text._Z10normKernelPKDhPDhPfPKfiS0_S0_,"ax",@progbits
                                        # -- End function

	.section	.text._Z13normInfKernelPKDhPDhPKfiS0_S0_,"ax",@progbits
	.globl	_Z13normInfKernelPKDhPDhPKfiS0_S0_              # -- Begin function _Z13normInfKernelPKDhPDhPKfiS0_S0_
	.p2align	3
	.type	_Z13normInfKernelPKDhPDhPKfiS0_S0_,@function
_Z13normInfKernelPKDhPDhPKfiS0_S0_:                       # @_Z13normInfKernelPKDhPDhPKfiS0_S0_
.LnormInfKernelPKDhPDhPKfiS0_S0_begin:
  //m0 -> src_ptr
  //m1 -> dst_ptr
  //m2 -> const_ptr
  //m3 -> inner_size
  //   -> scale_
  //   -> bias_
  ld32                $m4,              $m11,         0    //scale
  ld32                $m5,              $m11,         1    //bias
  sub                 $m11,             $m11,         24
  st32                $m0,              $m11,         3
  st32                $m7,              $m11,         4
  st32                $m8,              $m11,         5
  mov                 $m7,              $m4
  mov                 $m8,              $m5
  mov                 $a2:3,            $a14:15
  mov                 $a4:5,            $a14:15
  shr                 $m6,              $m3,          2
  shl                 $m6,              $m6,          2
  sub                 $m5,              $m3,          $m6
  {
    brz               $m6,              normInfKernelPKDhPDhPKfiS0_S0_statistic_proc_tail
    setzi             $a6,              ZAACC_BITMASK
  }
  {
    shr               $m6,              $m6,          2
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  sub                 $m6,              $m6,          1
  ld64step            $a0:1,            $m15,         $m0+=,            1
.align 8
  { 
    rpt               $m6,             ((__normInfKernelPKDhPDhPKfiS0_S0_statistic_sqsum_loop_end - __normInfKernelPKDhPDhPKfiS0_S0_statistic_sqsum_loop_start) / 8) - 1;
    fnop
  }
__normInfKernelPKDhPDhPKfiS0_S0_statistic_sqsum_loop_start:
  {
    ld64step          $a0:1,            $m15,             $m0+=,       1
    f16v8sqacc        $a0:3
  }
__normInfKernelPKDhPDhPKfiS0_S0_statistic_sqsum_loop_end:
  {
    nop
    f16v8sqacc        $a0:3
  }
  {
    nop
    f32v2gina         $a0:1,            $a14:15,        0
  }
  {
    nop
    f32v2gina         $a2:3,            $a14:15,        0
  }
  {
    ld32              $m0,              $m11,           3
    f32v2add          $a0:1,            $a0:1,          $a2:3
  }
  {
    ld64step          $a4:5,            $m15,           $m0+=,            1
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  f32add              $a0,              $a0,            $a1     //sqrSum
  mov                 $a6:7,            $a14:15
.align 8
  { 
    rpt               $m6,             ((__normInfKernelPKDhPDhPKfiS0_S0_statistic_sum_loop_end - __normInfKernelPKDhPDhPKfiS0_S0_statistic_sum_loop_start) / 8) - 1;
    fnop
  }
__normInfKernelPKDhPDhPKfiS0_S0_statistic_sum_loop_start:
  {
    ld64step          $a4:5,            $m15,             $m0+=,       1
    f16v8acc          $a4:7
  }
__normInfKernelPKDhPDhPKfiS0_S0_statistic_sum_loop_end:
  {
    nop
    f16v8acc          $a4:7
  }
  {
    nop
    f32v2gina         $a4:5,            $a14:15,        0
  }
  {
    nop
    f32v2gina         $a6:7,            $a14:15,        0
  }
  f32v2add            $a4:5,            $a4:5,          $a6:7
  f32add              $a1,              $a4,            $a5     //Sum
  brz                 $m5,              normInfKernelPKDhPDhPKfiS0_S0_proc_data
normInfKernelPKDhPDhPKfiS0_S0_statistic_proc_tail:
.align 8
  { 
    rpt               $m5,             ((normInfKernelPKDhPDhPKfiS0_S0_statistic_proc_tail_loop_end - normInfKernelPKDhPDhPKfiS0_S0_statistic_proc_tail_loop_start) / 8) - 1;
    fnop
  }
normInfKernelPKDhPDhPKfiS0_S0_statistic_proc_tail_loop_start:
  {
    ldb16step         $a2,              $m15,           $m0+=,       1
    fnop;
  }
  {
    nop;
    f16tof32          $a5,              $a2
  }
  {
    nop;
    f32mul            $a4,              $a5,            $a5
  }
  {
    nop;
    f32v2add          $a0:1,            $a0:1,          $a4:5
  }
normInfKernelPKDhPDhPKfiS0_S0_statistic_proc_tail_loop_end:
normInfKernelPKDhPDhPKfiS0_S0_proc_data:
  ld64                $a2:3,            $m2,            $m15,        0
  f32mul              $a0,              $a0,            $a2          //avgSqr
  f32mul              $a1,              $a1,            $a2          //avg
  f32tof16            $a4,              $a1
  f16tof32            $a4,              $a4
  f32mul              $a4,              $a4,            $a4          //avg * avg

  f32sub              $a4,              $a0,            $a4          //avgSqr - avg * avg
  f32max              $a4,              $a4,            $a15         //max(varianceEst, 0.0f)
  f32add              $a4,              $a4,            $a3          // += eps
  ldconst             $a0,              FLOAT_1_0
  f32sqrt             $a4,              $a4
  f32div              $a0,              $a0,            $a4
  f32tof16            $a2,              $a1            //avg
  mov                 $a3,              $a2
  f32tof16            $a4,              $a0            //stdDev
  mov                 $a5,              $a4

  ld32                $m0,              $m11,         3
  shr                 $m6,              $m3,          2
  shl                 $m6,              $m6,          2
  sub                 $m5,              $m3,          $m6
  brz                 $m6,              normInfKernelPKDhPDhPKfiS0_S0_calc_tail
  shr                 $m6,              $m6,          2
  ld64step            $a0:1,            $m15,         $m0+=,       1
  {
    sub               $m6,              $m6,          1
    f16v4sub          $a0:1,            $a0:1,        $a2:3
  }
.align 8
  { 
    rpt               $m6,             ((__normInfKernelPKDhPDhPKfiS0_S0_calc_inner_loop_end - __normInfKernelPKDhPDhPKfiS0_S0_calc_inner_loop_start) / 8) - 1;
    fnop
  }
__normInfKernelPKDhPDhPKfiS0_S0_calc_inner_loop_start:
  {
    ld64step          $a6:7,           $m15,          $m7+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a4:5
  }
  {
    ld64step          $a6:7,           $m15,          $m8+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a6:7
  }
  {
    ld64step          $a6:7,           $m15,          $m0+=,      1
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m1+=,      1
    f16v4sub          $a0:1,           $a6:7,         $a2:3
  }
__normInfKernelPKDhPDhPKfiS0_S0_calc_inner_loop_end:
  {
    ld64step          $a6:7,           $m15,          $m7+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a4:5
  }
  {
    ld64step          $a6:7,           $m15,          $m8+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a6:7
  }
  {
    nop
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m1+=,      1
    fnop
  }
  brz                 $m5,             normInfKernelPKDhPDhPKfiS0_S0_proc_end
normInfKernelPKDhPDhPKfiS0_S0_calc_tail:
  sub                 $m5,             $m5,           1
  mov                 $m4,             $m10
  mov                 $m2,             $m0
  mov                 $m0,             $m1
  mov                 $m6,             $m1
normInfKernelPKDhPDhPKfiS0_S0_calc_tail_loop:
  ldb16step           $a0,             $m15,          $m2+=,      1
  {
    ldb16step         $a5,             $m15,          $m7+=,      1
    f16v2sub          $a0,             $a0,           $a2
  }
  {
    ldb16step         $a6,             $m15,          $m8+=,      1
    f16v2mul          $a0,             $a0,           $a4
  }
  f16v2mul            $a0,             $a0,           $a5
  f16v2add            $a0,             $a0,           $a6
  call                $m10,            __st16f
  add                 $m6,             $m6,           2
  mov                 $m0,             $m6
  brnzdec             $m5,             normInfKernelPKDhPDhPKfiS0_S0_calc_tail_loop
  mov                 $m10,            $m4
normInfKernelPKDhPDhPKfiS0_S0_proc_end:
  ld32                $m7,             $m11,          4
  ld32                $m8,             $m11,          5
  add                 $m11,            $m11,          24
	br                  $m10
.LnormInfKernelPKDhPDhPKfiS0_S0_end:
	.size	_Z13normInfKernelPKDhPDhPKfiS0_S0_, .LnormInfKernelPKDhPDhPKfiS0_S0_end-_Z13normInfKernelPKDhPDhPKfiS0_S0_
	.section	.stack_sizes,"o",@progbits,.text._Z13normInfKernelPKDhPDhPKfiS0_S0_
	.long	.LnormInfKernelPKDhPDhPKfiS0_S0_begin
	.byte	24
	.section	.text._Z13normInfKernelPKDhPDhPKfiS0_S0_,"ax",@progbits
                                        # -- End function


	.section	.text._Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_,"ax",@progbits
	.globl	_Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_              # -- Begin function _Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_
	.p2align	3
	.type	_Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_,@function
_Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_:                       # @_Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_
.LnormInfKernelAligned16PKDhPDhPKfiS0_S0_begin:
  //m0 -> src_ptr
  //m1 -> dst_ptr
  //m2 -> const_ptr
  //m3 -> inner_size
  //   -> scale_
  //   -> bias_
  ld32                $m4,              $m11,         0    //scale
  ld32                $m5,              $m11,         1    //bias
  sub                 $m11,             $m11,         24
  st32                $m0,              $m11,         3
  st32                $m7,              $m11,         4
  st32                $m8,              $m11,         5
  mov                 $m7,              $m4
  mov                 $m8,              $m5
  shr                 $m6,              $m3,          3
  shl                 $m6,              $m6,          3
  sub                 $m5,              $m3,          $m6
  {
    brz               $m6,              normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_proc_tail
    setzi             $a6,              ZAACC_BITMASK
  }
  {
    shr               $m6,              $m6,          3
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  sub                 $m6,              $m6,          1
  ld128step           $a0:3,            $m15,         $m0+=,            1
.align 8
  { 
    rpt               $m6,             ((__normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sqsum_loop_end - __normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sqsum_loop_start) / 8) - 1;
    fnop
  }
__normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sqsum_loop_start:
  {
    ld128step         $a0:3,            $m15,             $m0+=,       1
    f16v8sqacc        $a0:3
  }
__normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sqsum_loop_end:
  {
    nop
    f16v8sqacc        $a0:3
  }
  {
    nop
    f32v2gina         $a0:1,            $a14:15,      0
  }
  {
    nop
    f32v2gina         $a2:3,            $a14:15,      0
  }
  f32v2add            $a0:1,            $a0:1,        $a2:3
  {
    nop
    f32v2gina         $a4:5,            $a14:15,      0
  }
  {
    nop
    f32v2gina         $a2:3,            $a14:15,      0
  }
  f32v2add            $a4:5,            $a4:5,        $a2:3
  f32v2add            $a0:1,            $a0:1,        $a4:5
  {
    ld32              $m0,              $m11,         3
    f32add            $a0,              $a0,          $a1     //sqrSum
  }
  {
    ld128step         $a4:7,            $m15,             $m0+=,       1
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
.align 8
  { 
    rpt               $m6,             ((__normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sum_loop_end - __normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sum_loop_start) / 8) - 1;
    fnop
  }
__normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sum_loop_start:
  {
    ld128step         $a4:7,            $m15,             $m0+=,       1
    f16v8acc          $a4:7
  }
__normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_sum_loop_end:
  {
    nop
    f16v8acc          $a4:7
  }
  {
    nop
    f32v2gina         $a2:3,            $a14:15,      0
  }
  {
    nop
    f32v2gina         $a4:5,            $a14:15,      0
  }
  f32v2add            $a2:3,            $a2:3,        $a4:5
  {
    nop
    f32v2gina         $a4:5,            $a14:15,      0
  }  
  {
    nop
    f32v2gina         $a6:7,            $a14:15,      0
  }
  f32v2add            $a4:5,            $a4:5,        $a6:7
  f32v2add            $a4:5,            $a2:3,        $a4:5
  f32add              $a1,              $a4,          $a5     //Sum
  brz                 $m5,              normInfKernelAligned16PKDhPDhPKfiS0_S0_proc_data
normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_proc_tail:
.align 8
  { 
    rpt               $m5,             ((normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_proc_tail_loop_end - normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_proc_tail_loop_start) / 8) - 1;
    fnop
  }
normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_proc_tail_loop_start:
  {
    ldb16step         $a2,              $m15,           $m0+=,       1
    fnop;
  }
  {
    nop;
    f16tof32          $a5,              $a2
  }
  {
    nop;
    f32mul            $a4,              $a5,            $a5
  }
  {
    nop;
    f32v2add          $a0:1,            $a0:1,          $a4:5    //sqrsum, sum
  }
normInfKernelAligned16PKDhPDhPKfiS0_S0_statistic_proc_tail_loop_end:
normInfKernelAligned16PKDhPDhPKfiS0_S0_proc_data:
  ld64                $a2:3,            $m2,            $m15,        0
  f32mul              $a0,              $a0,            $a2          //avgSqr
  f32mul              $a1,              $a1,            $a2          //avg
  f32tof16            $a4,              $a1
  f16tof32            $a4,              $a4
  f32mul              $a4,              $a4,            $a4          //avg * avg

  f32sub              $a4,              $a0,            $a4          //avgSqr - avg * avg
  f32max              $a4,              $a4,            $a15         //max(varianceEst, 0.0f)
  f32add              $a4,              $a4,            $a3          // += eps
  ldconst             $a0,              FLOAT_1_0
  f32sqrt             $a4,              $a4
  f32div              $a0,              $a0,            $a4
  f32tof16            $a2,              $a1            //avg
  mov                 $a3,              $a2
  f32tof16            $a4,              $a0            //stdDev
  mov                 $a5,              $a4

  ld32                $m0,              $m11,         3
  shr                 $m6,              $m3,          2
  shl                 $m6,              $m6,          2
  sub                 $m5,              $m3,          $m6
  brz                 $m6,              normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_tail
  shr                 $m6,              $m6,          2
  ld64step            $a0:1,            $m15,         $m0+=,       1
  {
    sub               $m6,              $m6,          1
    f16v4sub          $a0:1,            $a0:1,        $a2:3
  }
.align 8
  { 
    rpt               $m6,             ((__normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_inner_loop_end - __normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_inner_loop_start) / 8) - 1;
    fnop
  }
__normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_inner_loop_start:
  {
    ld64step          $a6:7,           $m15,          $m7+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a4:5
  }
  {
    ld64step          $a6:7,           $m15,          $m8+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a6:7
  }
  {
    ld64step          $a6:7,           $m15,          $m0+=,      1
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m1+=,      1
    f16v4sub          $a0:1,           $a6:7,         $a2:3
  }
__normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_inner_loop_end:
  {
    ld64step          $a6:7,           $m15,          $m7+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a4:5
  }
  {
    ld64step          $a6:7,           $m15,          $m8+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a6:7
  }
  {
    nop
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m1+=,      1
    fnop
  }
  brz                 $m5,             normInfKernelAligned16PKDhPDhPKfiS0_S0_proc_end
normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_tail:
  sub                 $m5,             $m5,           1
  mov                 $m4,             $m10
  mov                 $m2,             $m0
  mov                 $m0,             $m1
  mov                 $m6,             $m1
normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_tail_loop:
  ldb16step           $a0,             $m15,          $m2+=,      1
  {
    ldb16step         $a5,             $m15,          $m7+=,      1
    f16v2sub          $a0,             $a0,           $a2
  }
  {
    ldb16step         $a6,             $m15,          $m8+=,      1
    f16v2mul          $a0,             $a0,           $a4
  }
  f16v2mul            $a0,             $a0,           $a5
  f16v2add            $a0,             $a0,           $a6
  call                $m10,            __st16f
  add                 $m6,             $m6,           2
  mov                 $m0,             $m6
  brnzdec             $m5,             normInfKernelAligned16PKDhPDhPKfiS0_S0_calc_tail_loop
  mov                 $m10,            $m4
normInfKernelAligned16PKDhPDhPKfiS0_S0_proc_end:
  ld32                $m7,             $m11,          4
  ld32                $m8,             $m11,          5
  add                 $m11,            $m11,          24
	br                  $m10
.LnormInfKernelAligned16PKDhPDhPKfiS0_S0_end:
	.size	_Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_, .LnormInfKernelAligned16PKDhPDhPKfiS0_S0_end-_Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_
	.section	.stack_sizes,"o",@progbits,.text._Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_
	.long	.LnormInfKernelAligned16PKDhPDhPKfiS0_S0_begin
	.byte	24
	.section	.text._Z22normInfKernelAligned16PKDhPDhPKfiS0_S0_,"ax",@progbits
                                        # -- End function


	.section	.text._Z13arrangeKernelPKfPfS1_i,"ax",@progbits
	.globl	_Z13arrangeKernelPKfPfS1_i      # -- Begin function _Z13arrangeKernelPKfPfS1_i
	.p2align	3
	.type	_Z13arrangeKernelPKfPfS1_i,@function
_Z13arrangeKernelPKfPfS1_i:             # @_Z13arrangeKernelPKfPfS1_i
.LarrangeKernelPKfPfS1_i_begin:
	br $m10
.LarrangeKernelPKfPfS1_i_end:
	.size	_Z13arrangeKernelPKfPfS1_i, .LarrangeKernelPKfPfS1_i_end-_Z13arrangeKernelPKfPfS1_i
	.section	.stack_sizes,"o",@progbits,.text._Z13arrangeKernelPKfPfS1_i
	.long	.LarrangeKernelPKfPfS1_i_begin
	.byte	0
	.section	.text._Z13arrangeKernelPKfPfS1_i,"ax",@progbits
                                        # -- End function

	.section	.text._Z13arrangeKernelPKfPDhS1_i,"ax",@progbits
	.globl	_Z13arrangeKernelPKfPDhS1_i     # -- Begin function _Z13arrangeKernelPKfPDhS1_i
	.p2align	3
	.type	_Z13arrangeKernelPKfPDhS1_i,@function
_Z13arrangeKernelPKfPDhS1_i:            # @_Z13arrangeKernelPKfPDhS1_i
.LarrangeKernelPKfPDhS1_i_begin:
  shr         $m4,         $m3,            1
  shl         $m5,         $m4,            1
  sub         $m6,         $m3,            $m5
  brz         $m4,         arrangeKernelPKfPDhS1_i_proc_end
  sub         $m4,         $m4,            1
  ld64step    $a0:1,       $m15,           $m0+=,       1
  ld64step    $a2:3,       $m15,           $m0+=,       1
.align 8
  { 
    rpt               $m4,             ((__arrangeKernelPKfPDhS1_i_loop_end - __arrangeKernelPKfPDhS1_i_loop_start) / 8) - 1;
    fnop
  }
__arrangeKernelPKfPDhS1_i_loop_start:
  {
    ld64step          $a4:5,           $m15,          $m0+=,      1
    f32v2tof16        $a0,             $a0:1
  }
  {
    ld64step          $a6:7,           $m15,          $m0+=,      1
    f32v2tof16        $a1,             $a2:3
  }
  {
    nop;
    sort4x16lo        $a2,             $a0,            $a1
  }
  {
    nop;
    sort4x16hi        $a3,             $a0,            $a1
  }
  {
    st32step          $a2,             $m15,           $m1+=,     1
    or64              $a0:1,           $a4:5,          $a14:15
  }
  {
    st32step          $a3,             $m15,           $m2+=,     1
    or64              $a2:3,           $a6:7,          $a14:15
  }
__arrangeKernelPKfPDhS1_i_loop_end:
  {
    nop;
    f32v2tof16        $a0,             $a0:1
  }
  {
    nop;
    f32v2tof16        $a1,             $a2:3
  }
  {
    nop;
    sort4x16lo        $a2,             $a0,            $a1
  }
  {
    nop;
    sort4x16hi        $a3,             $a0,            $a1
  }
  {
    st32step          $a2,             $m15,           $m1+=,     1
    fnop;
  }
  {
    st32step          $a3,             $m15,           $m2+=,     1
    fnop;
  }
arrangeKernelPKfPDhS1_i_tail_proc:
  brz               $m6,         arrangeKernelPKfPDhS1_i_proc_end
  ld64step          $a4:5,       $m15,          $m0+=,      1
  f32tof16          $a0,         $a4
  mov               $m3,         $m1
  mov               $m4,         $m2
  mov               $m5,         $m10
  mov               $m0,         $m3
  call              $m10,        __st16f
  f32tof16          $a0,         $a5
  mov               $m0,         $m4
  call              $m10,        __st16f
  mov               $m10,        $m4
arrangeKernelPKfPDhS1_i_proc_end:
	br $m10
.LarrangeKernelPKfPDhS1_i_end:
	.size	_Z13arrangeKernelPKfPDhS1_i, .LarrangeKernelPKfPDhS1_i_end-_Z13arrangeKernelPKfPDhS1_i
	.section	.stack_sizes,"o",@progbits,.text._Z13arrangeKernelPKfPDhS1_i
	.long	.LarrangeKernelPKfPDhS1_i_begin
	.byte	0
	.section	.text._Z13arrangeKernelPKfPDhS1_i,"ax",@progbits
                                        # -- End function

	.ident	"clang version 15.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git 3a41ea640254cc484b335ffaa42ea065ed1ff137)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

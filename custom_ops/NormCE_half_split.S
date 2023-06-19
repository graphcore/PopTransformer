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

#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#ifndef  __NORM_SPLIT_S__
#define  __NORM_SPLIT_S__

#define FLOAT_1_0 0x3F800000
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

#if defined(VECTOR_AVAIL_SCALED_PTR64)
#define   SUP_SRC_PTR               0
#define   SUP_SCALE_PTR             2
#define   SUP_BIAS_PTR              4
#define   SUP_DST_PTR               6
#define   SUP_WRK_SIZE_PTR          8
#define   SUP_WRK_OFS_PTR           10
#define   SUP_INNER_SIZE            12
#define   SUP_INV_SCALE             16
#define   SUP_EPS                   20
#else
#define   SUP_SRC_PTR               0
#define   SUP_SCALE_PTR             4
#define   SUP_BIAS_PTR              8
#define   SUP_DST_PTR               12
#define   SUP_WRK_SIZE_PTR          16
#define   SUP_WRK_OFS_PTR           20
#define   SUP_INNER_SIZE            24
#define   SUP_INV_SCALE             28
#define   SUP_EPS                   32
#endif
#define   STACK_SRC_PTR             0
#define   STACK_SCALE_PTR           4
#define   STACK_BIAS_PTR            8
#define   STACK_DST_PTR             12
#define   STACK_WRK_SIZE_PTR        16
#define   STACK_WRK_OFS_PTR         20
#define   STACK_INNER_SIZE          24
#define   STACK_INV_SCALE           28
#define   STACK_EPS                 32
#define   STACK_MEAN_DEV_PTR        36
#define   STACK_MEAN_DEV_SIZE       52
#define   WKR_VERTEX_SIZE           88
#define   TOT_STACK_SIZE            WKR_VERTEX_SIZE

#define   mWorkerEntry              m6
#define   wId                       m4

	.section	.text._Z19normInfReduceKernelPKDhPfi,"ax",@progbits
	.globl	_Z19normInfReduceKernelPKDhPfi # -- Begin function _Z19normInfReduceKernelPKDhPfi
	.p2align	3
	.type	_Z19normInfReduceKernelPKDhPfi,@function
_Z19normInfReduceKernelPKDhPfi:     # @_Z19normInfReduceKernelPKDhPfi
.LnormInfReduceKernelPKDhPfi_begin:
  //$m0 is src_ptr
  //$m1 is dst_ptr(mean_stddev_ptr)
  //$m2 is work_size
  mov                 $a2:3,            $a14:15
  mov                 $a6:7,            $a14:15
  shr                 $m3,              $m2,          2
  shl                 $m3,              $m3,          2
  sub                 $m4,              $m2,          $m3     //data_size_tail
  mov                 $m2,              $m0
  {
    brz               $m3,              normInfReduceKernelPKDhPfi_proc_tail
    setzi             $a6,              ZAACC_BITMASK
  }
  {
    shr               $m3,              $m3,          2
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  sub                 $m3,              $m3,          1
  ld64step            $a0:1,            $m15,         $m0+=,            1
.align 8
  { 
    rpt               $m3,             ((__normInfReduceKernelPKDhPfi_sqsum_loop_end - __normInfReduceKernelPKDhPfi_sqsum_loop_start) / 8) - 1;
    fnop
  }
__normInfReduceKernelPKDhPfi_sqsum_loop_start:
  {
    ld64step          $a0:1,            $m15,         $m0+=,       1
    f16v8sqacc        $a0:3
  }
__normInfReduceKernelPKDhPfi_sqsum_loop_end:
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
  f32add              $a0,              $a0,          $a1     //sqrSum
  {
    ld64step          $a4:5,            $m15,         $m2+=,       1
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  mov                 $a6:7,            $a14:15
.align 8
  { 
    rpt               $m3,             ((__normInfReduceKernelPKDhPfi_sum_loop_end - __normInfReduceKernelPKDhPfi_sum_loop_start) / 8) - 1;
    fnop
  }
__normInfReduceKernelPKDhPfi_sum_loop_start:
  {
    ld64step          $a4:5,            $m15,         $m2+=,       1
    f16v8acc          $a4:7
  }
__normInfReduceKernelPKDhPfi_sum_loop_end:
  {
    nop
    f16v8acc          $a4:7
  }
  {
    nop
    f32v2gina         $a4:5,            $a14:15,      0
  }
  {
    nop
    f32v2gina         $a6:7,            $a14:15,      0
  }
  f32v2add            $a4:5,            $a4:5,        $a6:7
  f32add              $a1,              $a4,          $a5     //Sum
  brz                 $m4,              __normInfReduceKernelPKDhPfi_leave
normInfReduceKernelPKDhPfi_proc_tail:
.align 8
  { 
    rpt               $m4,             ((normInfReduceKernelPKDhPfi_proc_tail_loop_end - normInfReduceKernelPKDhPfi_proc_tail_loop_start) / 8) - 1;
    fnop
  }
normInfReduceKernelPKDhPfi_proc_tail_loop_start:
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
normInfReduceKernelPKDhPfi_proc_tail_loop_end:
__normInfReduceKernelPKDhPfi_leave:
  st32                $a0,              $m1,            0            //sqrsum
  st32                $a1,              $m1,            1            //sum
  br                  $m10
.LnormInfReduceKernelPKDhPfi_end:
  .size	_Z19normInfReduceKernelPKDhPfi, .LnormInfReduceKernelPKDhPfi_end-_Z19normInfReduceKernelPKDhPfi
  .section	.stack_sizes,"o",@progbits,.text._Z19normInfReduceKernelPKDhPfi
  .long	.LnormInfReduceKernelPKDhPfi_begin
  .byte	0
  .section	.text._Z19normInfReduceKernelPKDhPfi,"ax",@progbits
                                        # -- End function

	.section	.text._Z28normInfReduceKernelAligned16PKDhPfi,"ax",@progbits
	.globl	_Z28normInfReduceKernelAligned16PKDhPfi # -- Begin function _Z28normInfReduceKernelAligned16PKDhPfi
	.p2align	3
	.type	_Z28normInfReduceKernelAligned16PKDhPfi,@function
_Z28normInfReduceKernelAligned16PKDhPfi:     # @_Z28normInfReduceKernelAligned16PKDhPfi
.LnormInfReduceKernelAligned16PKDhPfi_begin:
  //$m0 is src_ptr
  //$m1 is dst_ptr(mean_stddev_ptr)
  //$m2 is work_size
  shr                 $m3,              $m2,          3
  shl                 $m3,              $m3,          3
  sub                 $m4,              $m2,          $m3     //data_size_tail
  mov                 $m2,              $m0
  {
    brz               $m3,              normInfReduceKernelAligned16PKDhPfi_proc_tail
    setzi             $a6,              ZAACC_BITMASK
  }
  {
    shr               $m3,              $m3,          3
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
  sub                 $m3,              $m3,          1
  ld128step           $a0:3,            $m15,         $m0+=,        1
.align 8
  { 
    rpt               $m3,             ((__normInfReduceKernelAligned16PKDhPfi_sqsum_loop_end - __normInfReduceKernelAligned16PKDhPfi_sqsum_loop_start) / 8) - 1;
    fnop
  }
__normInfReduceKernelAligned16PKDhPfi_sqsum_loop_start:
  {
    ld128step         $a0:3,            $m15,         $m0+=,       1
    f16v8sqacc        $a0:3
  }
__normInfReduceKernelAligned16PKDhPfi_sqsum_loop_end:
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
  f32add              $a0,              $a0,          $a1     //sqrSum
  {
    ld128step         $a4:7,            $m15,         $m2+=,       1
    uput              $FP_CLR,          $a6           //add zero mask for $ACC
  }
.align 8
  { 
    rpt               $m3,             ((__normInfReduceKernelAligned16PKDhPfi_sum_loop_end - __normInfReduceKernelAligned16PKDhPfi_sum_loop_start) / 8) - 1;
    fnop
  }
__normInfReduceKernelAligned16PKDhPfi_sum_loop_start:
  {
    ld128step         $a4:7,            $m15,         $m2+=,       1
    f16v8acc          $a4:7
  }
__normInfReduceKernelAligned16PKDhPfi_sum_loop_end:
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
  brz                 $m4,              __normInfReduceKernelAligned16PKDhPfi_leave
normInfReduceKernelAligned16PKDhPfi_proc_tail:
.align 8
  { 
    rpt               $m4,             ((normInfReduceKernelAligned16PKDhPfi_proc_tail_loop_end - normInfReduceKernelAligned16PKDhPfi_proc_tail_loop_start) / 8) - 1;
    fnop
  }
normInfReduceKernelAligned16PKDhPfi_proc_tail_loop_start:
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
normInfReduceKernelAligned16PKDhPfi_proc_tail_loop_end:
__normInfReduceKernelAligned16PKDhPfi_leave:
  st32                $a0,              $m1,            0            //sqrsum
  st32                $a1,              $m1,            1            //sum
  br                  $m10
.LnormInfReduceKernelAligned16PKDhPfi_end:
  .size	_Z28normInfReduceKernelAligned16PKDhPfi, .LnormInfReduceKernelAligned16PKDhPfi_end-_Z28normInfReduceKernelAligned16PKDhPfi
  .section	.stack_sizes,"o",@progbits,.text._Z28normInfReduceKernelAligned16PKDhPfi
  .long	.LnormInfReduceKernelAligned16PKDhPfi_begin
  .byte	0
  .section	.text._Z28normInfReduceKernelAligned16PKDhPfi,"ax",@progbits
                                        # -- End function

	.section	.text._Z22normInfNormalizeKernelPKDhS0_S0_PDhiff,"ax",@progbits
	.globl	_Z22normInfNormalizeKernelPKDhS0_S0_PDhiff # -- Begin function _Z22normInfNormalizeKernelPKDhS0_S0_PDhiff
	.p2align	3
	.type	_Z22normInfNormalizeKernelPKDhS0_S0_PDhiff,@function
_Z22normInfNormalizeKernelPKDhS0_S0_PDhiff:     # @_Z22normInfNormalizeKernelPKDhS0_S0_PDhiff
.LnormInfNormalizeKernelPKDhS0_S0_PDhiff_begin:
  //$m0 is src
  //$m1 is scale
  //$m2 is bias
  //$m3 is dst
  //$m4 is size
  f32tof16            $a2,              $a0            //avg
  mov                 $a3,              $a2
  f32tof16            $a4,              $a1            //stdDev
  mov                 $a5,              $a4

  shr                 $m6,              $m4,          2
  shl                 $m6,              $m6,          2
  sub                 $m5,              $m4,          $m6
  brz                 $m6,              normInfNormalizeKernelPKDhS0_S0_PDhiff_tail
  shr                 $m6,              $m6,          2
  ld64step            $a0:1,            $m15,         $m0+=,       1    //src_ptr
  {
    sub               $m6,              $m6,          1
    f16v4sub          $a0:1,            $a0:1,        $a2:3            //- mean
  }
.align 8
  { 
    rpt               $m6,             ((__normInfNormalizeKernelPKDhS0_S0_PDhiff_inner_loop_end - __normInfNormalizeKernelPKDhS0_S0_PDhiff_inner_loop_start) / 8) - 1;
    fnop
  }
__normInfNormalizeKernelPKDhS0_S0_PDhiff_inner_loop_start:
  {
    ld64step          $a6:7,           $m15,          $m1+=,      1    //ld scale
    f16v4mul          $a0:1,           $a0:1,         $a4:5            //* invStdDev
  }
  {
    ld64step          $a6:7,           $m15,          $m2+=,      1    //ld bias
    f16v4mul          $a0:1,           $a0:1,         $a6:7            //* scale
  }
  {
    ld64step          $a6:7,           $m15,          $m0+=,      1    //ld next src
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m3+=,      1    //st dst
    f16v4sub          $a0:1,           $a6:7,         $a2:3            //- mean
  }
__normInfNormalizeKernelPKDhS0_S0_PDhiff_inner_loop_end:
  {
    ld64step          $a6:7,           $m15,          $m1+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a4:5
  }
  {
    ld64step          $a6:7,           $m15,          $m2+=,      1
    f16v4mul          $a0:1,           $a0:1,         $a6:7
  }
  {
    nop
    f16v4add          $a0:1,           $a0:1,         $a6:7
  }
  {
    st64step          $a0:1,           $m15,          $m3+=,      1
    fnop
  }
normInfNormalizeKernelPKDhS0_S0_PDhiff_tail:
  brz                 $m5,             normInfNormalizeKernelPKDhS0_S0_PDhiff_proc_end
  sub                 $m5,             $m5,           1
  mov                 $m4,             $m10
  mov                 $m6,             $m0
  mov                 $m7,             $m1
  mov                 $m0,             $m3
normInfNormalizeKernelPKDhS0_S0_PDhiff_tail_loop:
  ldb16step           $a0,             $m15,          $m6+=,      1
  {
    ldb16step         $a5,             $m15,          $m7+=,      1
    f16v2sub          $a0,             $a0,           $a2
  }
  {
    ldb16step         $a6,             $m15,          $m2+=,      1
    f16v2mul          $a0,             $a0,           $a4
  }
  f16v2mul            $a0,             $a0,           $a5
  f16v2add            $a0,             $a0,           $a6
  call                $m10,            __st16f
  add                 $m3,             $m3,           2
  mov                 $m0,             $m3
  brnzdec             $m5,             normInfNormalizeKernelPKDhS0_S0_PDhiff_tail_loop
  mov                 $m10,            $m4
normInfNormalizeKernelPKDhS0_S0_PDhiff_proc_end:
	br                  $m10
.LnormInfNormalizeKernelPKDhS0_S0_PDhiff_end:
  .size	_Z22normInfNormalizeKernelPKDhS0_S0_PDhiff, .LnormInfNormalizeKernelPKDhS0_S0_PDhiff_end-_Z22normInfNormalizeKernelPKDhS0_S0_PDhiff
  .section	.stack_sizes,"o",@progbits,.text._Z22normInfNormalizeKernelPKDhS0_S0_PDhiff
  .long	.LnormInfNormalizeKernelPKDhS0_S0_PDhiff_begin
  .byte	0
  .section	.text._Z22normInfNormalizeKernelPKDhS0_S0_PDhiff,"ax",@progbits
                                        # -- End function

#define   NORM_SPLIT_CODELET_NAME    __runCodelet_celib__NormCEInfSplitVertex___half_\TSK_SIZE\()_\INTER_LEAVE\()

.macro NORM_SPLIT_SUPERVISOR TSK_SIZE INTER_LEAVE
DEF_STACK_USAGE TOT_STACK_SIZE NORM_SPLIT_CODELET_NAME

.section .text.NORM_SPLIT_CODELET_NAME
.p2align 3
.globl NORM_SPLIT_CODELET_NAME
.type NORM_SPLIT_CODELET_NAME, @function
NORM_SPLIT_CODELET_NAME:
.supervisor
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16             $m1,              $m0,                SUP_SRC_PTR/2
  ldz16             $m2,              $m0,                SUP_SCALE_PTR/2
  ldz16             $m3,              $m0,                SUP_BIAS_PTR/2
  ldz16             $m4,              $m0,                SUP_DST_PTR/2
  ldz16             $m5,              $m0,                SUP_WRK_SIZE_PTR/2
  ldz16             $m6,              $m0,                SUP_WRK_OFS_PTR/2
  shl               $m1,              $m1,                3
  shl               $m2,              $m2,                3
  shl               $m3,              $m3,                3
  shl               $m4,              $m4,                3
  shl               $m5,              $m5,                3
  shl               $m6,              $m6,                3
#else
  ld32              $m1,              $m0,                SUP_SRC_PTR/4
  ld32              $m2,              $m0,                SUP_SCALE_PTR/4
  ld32              $m3,              $m0,                SUP_BIAS_PTR/4
  ld32              $m4,              $m0,                SUP_DST_PTR/4
  ld32              $m5,              $m0,                SUP_WRK_SIZE_PTR/4
  ld32              $m6,              $m0,                SUP_WRK_OFS_PTR/4
#endif
  add               $sp,              $sp,                -TOT_STACK_SIZE
  st32              $m1,              $sp,                STACK_SRC_PTR/4
  st32              $m2,              $sp,                SUP_SCALE_PTR/4
  st32              $m3,              $sp,                SUP_BIAS_PTR/4
  st32              $m4,              $sp,                SUP_DST_PTR/4
  st32              $m5,              $sp,                SUP_WRK_SIZE_PTR/4
  st32              $m6,              $sp,                SUP_WRK_OFS_PTR/4

  ld32              $m1,              $m0,                SUP_INNER_SIZE/4
  ld32              $m2,              $m0,                SUP_INV_SCALE/4
  ld32              $m3,              $m0,                SUP_EPS/4
  st32              $m1,              $sp,                STACK_INNER_SIZE/4
  st32              $m2,              $sp,                STACK_INV_SCALE/4
  st32              $m3,              $sp,                STACK_EPS/4

  setzi             $mWorkerEntry,    worker_reduce_stage_\TSK_SIZE\()_\INTER_LEAVE\()
  runall            $mWorkerEntry,    $sp,                0
  sync              TEXCH_SYNCZONE_LOCAL
  setzi             $mWorkerEntry,    worker_reduce_merge_stage_\TSK_SIZE\()_\INTER_LEAVE\()
  runall            $mWorkerEntry,    $sp,                0
  sync              TEXCH_SYNCZONE_LOCAL
  setzi             $mWorkerEntry,    worker_normalize_stage_\TSK_SIZE\()_\INTER_LEAVE\()
  runall            $mWorkerEntry,    $sp,                0
  sync              TEXCH_SYNCZONE_LOCAL
  add               $sp,              $sp,                TOT_STACK_SIZE
  br                $lr

.worker
worker_reduce_stage_\TSK_SIZE\()_\INTER_LEAVE\():
  get               $wId,             $WSR
  and               $wId,             $wId,               CSR_W_WSR__CTXTID_M1__MASK

  add               $m1,              $mvertex_base,      STACK_MEAN_DEV_PTR
  add               $m1,              $m1,                7
  shr               $m1,              $m1,                3
  shl               $m1,              $m1,                3
  shl               $m2,              $wId,               3
  add               $m1,              $m1,                $m2      //mean_stddev_ptr
  st64              $a14:15,          $mzero,             $m1,     0

  ld32              $m3,              $mvertex_base,      STACK_WRK_SIZE_PTR/4
  ld32              $m3,              $m3,                $wId
  brz               $m3,              __worker_reduce_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit


  ld32              $m0,              $mvertex_base,      STACK_SRC_PTR/4
  ld32              $m2,              $mvertex_base,      STACK_WRK_SIZE_PTR/4
  ld32              $m5,              $mvertex_base,      STACK_WRK_OFS_PTR/4
  ld32              $m2,              $m2,                $wId     //work_size
  ld32              $m5,              $m5,                $wId     //work_ofs

  shl               $m5,              $m5,                1
  add               $m0,              $m0,                $m5      //src_ptr

.ifc \INTER_LEAVE, true
  call              $lr,              _Z28normInfReduceKernelAligned16PKDhPfi
.else
  call              $lr,              _Z19normInfReduceKernelPKDhPfi
.endif
__worker_reduce_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit:
  exitz             $mzero
 
.worker
worker_reduce_merge_stage_\TSK_SIZE\()_\INTER_LEAVE\():
  get               $wId,             $WSR
  and               $wId,             $wId,               CSR_W_WSR__CTXTID_M1__MASK

.if \TSK_SIZE == 1
  brnz              $wId,             worker_reduce_merge_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit
  add               $m0,              $mvertex_base,      STACK_MEAN_DEV_PTR
  add               $m0,              $m0,                7
  shr               $m0,              $m0,                3
  shl               $m0,              $m0,                3
  mov               $m1,              $m0

  mov               $a0:1,            $a14:15
  ld64step          $a2:3,            $mzero,             $m0+=,                1
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld32              $a2,            $mvertex_base,      STACK_INV_SCALE/4
    f32v2add          $a0:1,          $a0:1,              $a2:3
  }
  {
    ld32              $a3,            $mvertex_base,      STACK_EPS/4
    f32mul            $a0,            $a0,                $a2             //avgSqr
  }

  f32mul            $a1,              $a1,            $a2             //avg
  {
    st32              $a1,              $m1,            0            //
    f32tof16          $a4,              $a1                          //f32mul            $a4,              $a1,            $a1          //avg * avg
  }
  {
    nop;
    f16tof32          $a4,              $a4
  }
  {
    nop;
    f32mul            $a4,              $a4,            $a4          //avg * avg
  }

  f32sub              $a4,              $a0,            $a4          //avgSqr - avg * avg
  f32max              $a4,              $a4,            $a15         //max(varianceEst, 0.0f)
  f32add              $a4,              $a4,            $a3          // += eps
  f32sqrt             $a4,              $a4
  ldconst             $a0,              FLOAT_1_0
  f32div              $a0,              $a0,            $a4
  st32                $a0,              $m1,            1
  ld64                $a0:1,            $m1,            0
  st64                $a0:1,            $m1,            1
  st64                $a0:1,            $m1,            2
  st64                $a0:1,            $m1,            3
  st64                $a0:1,            $m1,            4
  st64                $a0:1,            $m1,            5
.else
.if \TSK_SIZE == 2
//TSK_SIZE is 2
  cmpslt            $m6,              $wId,             2    
  brz               $m6,              worker_reduce_merge_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit
  add               $m0,              $mvertex_base,      STACK_MEAN_DEV_PTR
  add               $m0,              $m0,                7
  shr               $m0,              $m0,                3
  shl               $m0,              $m0,                3
  mul               $m6,              $wId,               24        
  add               $m0,              $m0,                $m6
  mov               $m1,              $m0

  mov               $a0:1,            $a14:15
  ld64step          $a2:3,            $mzero,             $m0+=,                1
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld32            $a2,              $mvertex_base,      STACK_INV_SCALE/4
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld32            $a3,              $mvertex_base,      STACK_EPS/4
    f32mul          $a0,              $a0,                $a2             //avgSqr
  }
  f32mul            $a1,              $a1,            $a2             //avg
  {
    st32            $a1,              $m1,            0            //
    f32tof16        $a4,              $a1                          //f32mul          $a4,              $a1,            $a1          //avg * avg
  }

  {
    nop;
    f16tof32          $a4,              $a4
  }
  {
    nop;
    f32mul            $a4,              $a4,            $a4          //avg * avg
  }

  f32sub              $a4,              $a0,            $a4          //avgSqr - avg * avg
  f32max              $a4,              $a4,            $a15         //max(varianceEst, 0.0f)
  f32add              $a4,              $a4,            $a3          // += eps
  f32sqrt             $a4,              $a4
  ldconst             $a0,              FLOAT_1_0
  f32div              $a0,              $a0,            $a4
  st32                $a0,              $m1,            1
  ld64                $a0:1,            $m1,            0
  st64                $a0:1,            $m1,            1
  st64                $a0:1,            $m1,            2
.else
//TSK_SIZE is 3
  cmpslt            $m6,              $wId,               3   
  brz               $m6,              worker_reduce_merge_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit
  add               $m0,              $mvertex_base,      STACK_MEAN_DEV_PTR
  add               $m0,              $m0,                7
  shr               $m0,              $m0,                3
  shl               $m0,              $m0,                3
  mul               $m6,              $wId,               16       
  add               $m0,              $m0,                $m6
  mov               $m1,              $m0

  mov               $a0:1,            $a14:15
  ld64step          $a2:3,            $mzero,             $m0+=,                1
  {
    ld64step        $a2:3,            $mzero,             $m0+=,                1
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld32            $a2,              $mvertex_base,      STACK_INV_SCALE/4
    f32v2add        $a0:1,            $a0:1,              $a2:3
  }
  {
    ld32            $a3,              $mvertex_base,      STACK_EPS/4
    f32mul          $a0,              $a0,                $a2             //avgSqr
  }

  f32mul            $a1,              $a1,            $a2             //avg
  {
    st32            $a1,              $m1,            0            //
    f32tof16        $a4,              $a1
  }

  {
    nop;
    f16tof32          $a4,              $a4
  }
  {
    nop;
    f32mul            $a4,              $a4,            $a4          //avg * avg
  }

  f32sub              $a4,              $a0,            $a4          //avgSqr - avg * avg
  f32max              $a4,              $a4,            $a15         //max(varianceEst, 0.0f)
  f32add              $a4,              $a4,            $a3          // += eps
  f32sqrt             $a4,              $a4
  ldconst             $a0,              FLOAT_1_0
  f32div              $a0,              $a0,            $a4
  st32                $a0,              $m1,            1
  ld64                $a0:1,            $m1,            0
  st64                $a0:1,            $m1,            1
.endif
.endif
worker_reduce_merge_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit:
  exitz               $mzero

.worker
worker_normalize_stage_\TSK_SIZE\()_\INTER_LEAVE\():
  get               $wId,             $WSR
  and               $wId,             $wId,               CSR_W_WSR__CTXTID_M1__MASK
  ld32              $m5,              $mvertex_base,      STACK_WRK_SIZE_PTR/4
  ld32              $m5,              $m5,                $wId
  brz               $m5,              __worker_normalize_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit

  add               $m0,              $mvertex_base,      STACK_MEAN_DEV_PTR
  add               $m0,              $m0,                7
  shr               $m0,              $m0,                3
  shl               $m0,              $m0,                3
  ld64              $a0:1,            $mzero,             $m0,                $wId

  ld32              $m0,              $mvertex_base,      STACK_SRC_PTR/4
  ld32              $m6,              $mvertex_base,      STACK_WRK_OFS_PTR/4
  ld32              $m6,              $m6,                $wId
  shl               $m6,              $m6,                1
  add               $m0,              $m0,                $m6
  ld32              $m1,              $mvertex_base,      STACK_SCALE_PTR/4
  ld32              $m2,              $mvertex_base,      STACK_BIAS_PTR/4
  add               $m1,              $m1,                $m6
  add               $m2,              $m2,                $m6
.if \TSK_SIZE == 1
.else
  ld32              $m8,              $mvertex_base,      STACK_INNER_SIZE/4
  shl               $m8,              $m8,                1
.if \TSK_SIZE == 2
  cmpult            $m7,              $wId,               3
  brnz              $m7,              __worker_normalize_stage_tsksize_2_\INTER_LEAVE\()_working
  sub               $m1,              $m1,                $m8
  sub               $m2,              $m2,                $m8
__worker_normalize_stage_tsksize_2_\INTER_LEAVE\()_working:
.else
//TSK_SIZE == 3
  shr               $m7,              $wId,               1
  mul               $m8,              $m8,                $m7
  sub               $m1,              $m1,                $m8
  sub               $m2,              $m2,                $m8
.endif
.endif
  ld32              $m3,              $mvertex_base,      STACK_DST_PTR/4
  add               $m3,              $m3,                $m6
  mov               $m4,              $m5                 //split_data_size
  call              $lr,              _Z22normInfNormalizeKernelPKDhS0_S0_PDhiff
__worker_normalize_stage_\TSK_SIZE\()_\INTER_LEAVE\()_exit:
  exitz             $mzero

.size NORM_SPLIT_CODELET_NAME, . - NORM_SPLIT_CODELET_NAME

.endm

NORM_SPLIT_SUPERVISOR 1 false
NORM_SPLIT_SUPERVISOR 2 false
NORM_SPLIT_SUPERVISOR 3 false
NORM_SPLIT_SUPERVISOR 1 true
NORM_SPLIT_SUPERVISOR 2 true
NORM_SPLIT_SUPERVISOR 3 true

#endif  //__NORM_SPLIT_S__
#endif  //__IPU__

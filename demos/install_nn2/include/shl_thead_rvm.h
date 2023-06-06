/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* SHL version 2.1.x */

#ifndef INCLUDE_SHL_THEAD_RVM_H_
#define INCLUDE_SHL_THEAD_RVM_H_

#include "shl_thead_rvv.h"

#ifdef __riscv_xtheadmatrix
#include <riscv_matrix.h>
#define MATRIX_PW_I32  // requantize inst: enable
#endif                 // __riscv_xtheadmatrix

int shl_rvm_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvm_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);

int shl_rvm_depthwise_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_rvm_depthwise_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);

int shl_rvm_avgpool2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvm_avgpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvm_global_avgpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);
int shl_rvm_maxpool2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvm_maxpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

int shl_rvm_global_maxpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

void shl_rvm_conv1x1s1_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);
int shl_rvm_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_rvm_group_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);

void shl_rvm_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);
int shl_rvm_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_rvm_group_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);

void shl_rvm_conv1x1s1_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);
int shl_rvm_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_rvm_group_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);

void shl_rvm_conv_im2col_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);
int shl_rvm_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_rvm_group_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);

void shl_rvm_nhwc_gemm_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, const int32_t *bias,
                            int m, int k, int n, int32_t out_zp, int32_t *mult, int32_t *shift);
void shl_rvm_nhwc_gemm_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, const __fp16 *bias,
                            int m, int k, int n);

void shl_rvm_wg_b4f3s1_trans_kernel_nhwc_fp16(struct csinn_tensor *src_kernel,
                                              struct csinn_tensor *dst_kernel);
int shl_rvm_wg_b4f3s1_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
void shl_rvm_wg_b6f3s1_trans_kernel_nhwc_fp16(struct csinn_tensor *src_kernel,
                                              struct csinn_tensor *dst_kernel);
int shl_rvm_wg_b6f3s1_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

int csrr_xrlenb();

#endif  // INCLUDE_SHL_THEAD_RVM_H_

/*
 * Copyright © 2023 Hangzhou C-SKY MicroSystems Co., Ltd. All rights reserved.
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

#ifndef INCLUDE_SHL_RVV_H_
#define INCLUDE_SHL_RVV_H_

#if __riscv_vector
#include <riscv_vector.h>

#if (__riscv_v == 1000000)
#define RVV_1_0_0
#elif (__riscv_v == 7000)
#define RVV_0_7_1
#endif

#ifdef __riscv_xtheadvdot
#define XTHEADVDOT
#define SHL_USE_DOT_INT8  // default: support int8 dot
// #define SHL_USE_DOT_INT4     // easter eggs
#endif  // __riscv_xtheadvdot

#endif  // __riscv_vector

#include "csi_nn.h"
#include "shl_gref.h"
#include "shl_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

/********************************** initialization ******************************/
int shl_rvv_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);

int shl_rvv_depthwise_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_rvv_depthwise_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_rvv_depthwise_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_rvv_depthwise_conv2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);

int shl_rvv_avgpool2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvv_avgpool2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvv_avgpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

int shl_rvv_maxpool2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvv_maxpool2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvv_maxpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);
int shl_rvv_global_maxpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

int shl_rvv_fullyconnected_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weights, struct csinn_tensor *bias,
                                struct csinn_fc_params *params);

int shl_rvv_data_convert_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_siso_params *params);

/************************************ convolution *********************************/
/*********************************** im2col + gemm ********************************/
void shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);

int shl_rvv_conv_im2col_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);

void shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp32(struct csinn_tensor *kernel,
                                                        struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16(struct csinn_tensor *kernel,
                                                        struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int8(struct csinn_tensor *kernel,
                                                        struct csinn_conv2d_params *params);

int shl_rvv_conv_im2col_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);

void shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp32(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_int8(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params);

int shl_rvv_conv_im2col_gemm_pack1ton_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_pack1ton_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);

void shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp32(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_int8(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params);

int shl_rvv_conv_im2col_gemm_packnto1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_packnto1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);

/******************************** conv2d1x1s1 + gemm ******************************/
void shl_rvv_conv1x1s1_gemm_reorder_kernel_fp32(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);

int shl_rvv_conv1x1s1_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

void shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp32(struct csinn_tensor *kernel,
                                                      struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16(struct csinn_tensor *kernel,
                                                      struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_int8(struct csinn_tensor *kernel,
                                                      struct csinn_conv2d_params *params);

int shl_rvv_conv1x1s1_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);

void shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp32(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_int8(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);

int shl_rvv_conv1x1s1_gemm_pack1ton_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_pack1ton_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

void shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp32(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_int8(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);

int shl_rvv_conv1x1s1_gemm_packnto1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_packnto1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

/************************************* winograd ***********************************/
void shl_rvv_wg_b6f3s1_trans_kernel_packn_fp32(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel);
void shl_rvv_wg_b6f3s1_trans_kernel_packn_fp16(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel);

int shl_rvv_wg_b6f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);
int shl_rvv_wg_b6f3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);

void shl_rvv_wg_b4f3s1_trans_kernel_packn_fp32(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel);
void shl_rvv_wg_b4f3s1_trans_kernel_packn_fp16(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel);
void shl_rvv_wg_b4f3s1_trans_kernel_packn_int8(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel);

int shl_rvv_wg_b4f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);
int shl_rvv_wg_b4f3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);
int shl_rvv_wg_b4f3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);

/******************************* depthwise convolution ****************************/
int shl_rvv_dwconv3x3s1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s2_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s2_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s2_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s1_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s2_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);

void shl_rvv_dwconv_reorder_kernel_packn_fp32(struct csinn_tensor *kernel,
                                              struct csinn_conv2d_params *params);
void shl_rvv_dwconv_reorder_kernel_packn_fp16(struct csinn_tensor *kernel,
                                              struct csinn_conv2d_params *params);
void shl_rvv_dwconv_reorder_kernel_packn_int8(struct csinn_tensor *kernel,
                                              struct csinn_conv2d_params *params);

int shl_rvv_dwconv3x3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s2_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s2_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_rvv_dwconv3x3s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);

int shl_rvv_dwconv_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);
int shl_rvv_dwconv_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);
int shl_rvv_dwconv_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);

int shl_rvv_dwconv_nhwc_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvv_dwconv_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);

/*************************************** gemm *************************************/
void shl_rvv_reorder_kernel_n8_fp32(float *a, float *sa, int m, int k, int ldx);
void shl_rvv_reorder_input_z8_fp32(float *b, float *sb, int k, int n, int ldx);
void shl_rvv_gemm_8x8_fp32(float *dst, const float *sa, const float *sb, float *bias, int m, int k,
                           int n, int ldc);

void shl_rvv256_reorder_input_z16_fp32(float *b, float *sb, int k, int n, int ldx);
void shl_rvv256_gemm_8x16_fp32(float *dst, const float *sa, const float *sb, float *bias, int m,
                               int k, int n, int ldc);

void shl_rvv_reorder_kernel_n8_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
void shl_rvv_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);
void shl_rvv_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias, int m,
                            int k, int n, int ldc);

void shl_rvv256_reorder_kernel_n16_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
void shl_rvv256_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);
void shl_rvv256_gemm_16x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                int m, int k, int n, int ldc);

void shl_rvv_reorder_kernel_n8_int8_dot(int8_t *a, int8_t *sa, int m, int k, int ldx);
void shl_rvv_reorder_input_z8_int8_dot(int8_t *b, int8_t *sb, int k, int n, int ldx);
void shl_rvv_gemm_8x8_int32(int32_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias, int m,
                            int k, int n, int ldc);
void shl_rvv_gemm_8x8_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias,
                               int m, int k, int n, int ldc, int32_t out_zp, int32_t *mult,
                               int32_t *shift);

void shl_rvv256_reorder_input_z16_int8(int8_t *b, int8_t *sb, int k, int n, int ldx);
void shl_rvv256_gemm_8x16_int32(int32_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias,
                                int m, int k, int n, int ldc);

void shl_rvv_reorder_input_n8_int4_dot(int8_t *a, int8_t *sa, int m, int k, int ldx);
void shl_rvv_reorder_kernel_n8_int4(int8_t *b, int8_t *sb, int n, int k, int ldx);
void shl_rvv_gemm_8x8_int4_dot(int8_t *dst, const int8_t *sa, const int8_t *sb, int m, int k, int n,
                               int ldc, int32_t *bias, int32_t out_zp, int32_t *mult,
                               int32_t *shift);

/************************************ gemm ncxhwx *********************************/
void shl_rvv_reorder_kernel_packn_fp32(float *a, float *sa, int m, int k, int ldx);
void shl_rvv_reorder_input_z8_packn_fp32(float *b, float *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_8xpack2n_fp32(float *dst, const float *sa, const float *sb, float *bias,
                                       int m, int k, int n, int ldc);
void shl_rvv_reorder_input_z12_packn_fp32(float *b, float *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_12xpack2n_fp32(float *dst, const float *sa, const float *sb, float *bias,
                                        int m, int k, int n, int ldc);

void shl_rvv_reorder_kernel_packn_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
void shl_rvv_reorder_input_z8_packn_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_8xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                       __fp16 *bias, int m, int k, int n, int ldc);
void shl_rvv_reorder_input_z12_packn_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_12xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                        __fp16 *bias, int m, int k, int n, int ldc);

void shl_rvv_reorder_input_z8_packn_int8_dot(int8_t *b, int8_t *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_8xpackn_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                          int32_t *bias, int m, int k, int n, int ldc,
                                          int32_t out_zp, int32_t *mult, int32_t *shift);
void shl_rvv_reorder_input_z12_packn_int8_dot(int8_t *b, int8_t *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_12xpackn_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                           int32_t *bias, int m, int k, int n, int ldc,
                                           int32_t out_zp, int32_t *mult, int32_t *shift);

void shl_rvv_reorder_input_z8_packn_int4(int8_t *b, int8_t *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_8xpackn_int4(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                      int32_t *bias, int m, int k, int n, int ldc, int32_t out_zp,
                                      int32_t *mult, int32_t *shift);

void shl_rvv_reorder_input_z12_packn_int4(int8_t *b, int8_t *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_12xpackn_int4(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                       int32_t *bias, int m, int k, int n, int ldc, int32_t out_zp,
                                       int32_t *mult, int32_t *shift);

void shl_rvv_reorder_input_z12_pack1ton_fp32(float *b, float *sb, int inc, int maxk, int n,
                                             int ldx);
void shl_rvv_reorder_input_z12_pack1ton_fp16(__fp16 *b, __fp16 *sb, int inc, int maxk, int n,
                                             int ldx);
void shl_rvv_reorder_input_z4_pack1ton_int8(int8_t *b, int8_t *sb, int inc, int maxk, int n,
                                            int ldx);
void shl_rvv_reorder_input_z12_pack1ton_int8_dot(int8_t *b, int8_t *sb, int inc, int maxk, int n,
                                                 int ldx);

void shl_rvv_reorder_input_z4_packn_int8(int8_t *b, int8_t *sb, int k, int n, int ldx);
void shl_rvv_ncxhwx_gemm_4xpack2n_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                       int32_t *bias, int m, int k, int n, int ldc, int32_t out_zp,
                                       int32_t *mult, int32_t *shift);

/************************************ pooling *********************************/
int shl_rvv_avgpool2x2s2_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_avgpool2x2s2_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_avgpool2x2s2_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_avgpool2x2s2_p1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s2_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s2_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s2_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s2_p1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s1_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s1_p1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_rvv_maxpool2x2s2_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool2x2s2_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool2x2s2_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool2x2s2_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool2x2s2_p1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool2x2s2_p1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_p1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_p1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s1_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s1_p1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s1_p1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_rvv_global_avgpool2d_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

int shl_rvv_global_maxpool2d_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);
int shl_rvv_global_maxpool2d_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

int shl_rvv_maxpool2x2s2_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool2x2s2_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool2x2s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_maxpool3x3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);

int shl_rvv_avgpool2x2s2_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool2x2s2_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool2x2s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s2_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s2_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);
int shl_rvv_avgpool3x3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params);

int shl_rvv_global_maxpool2d_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params);
int shl_rvv_global_maxpool2d_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params);
int shl_rvv_global_maxpool2d_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params);

int shl_rvv_maxpool_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);
int shl_rvv_maxpool_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);
int shl_rvv_maxpool_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);

int shl_rvv_avgpool_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);
int shl_rvv_avgpool_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);
int shl_rvv_avgpool_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);

int shl_rvv_maxpool_nhwc_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_maxpool_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_rvv_avgpool_nhwc_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_avgpool_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);
int shl_rvv_avgpool_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_rvv_global_maxpool2d_nhwc_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params);
int shl_rvv_global_maxpool2d_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params);
int shl_rvv_global_maxpool2d_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_nhwc_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params);
int shl_rvv_global_avgpool2d_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params);

/************************************ fullyconnected *********************************/
void shl_rvv_fc_gemv_transform_weight_fp32(struct csinn_tensor *weights);
void shl_rvv_fc_gemv_transform_weight_fp16(struct csinn_tensor *weights);
void shl_rvv_fc_gemv_transform_weight_int8(struct csinn_tensor *weights);

int shl_rvv_fullyconnected_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params);
int shl_rvv_fullyconnected_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params);
int shl_rvv_fullyconnected_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params);

/************************************ activation *********************************/
int shl_rvv_relu_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);
int shl_rvv_relu_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);
int shl_rvv_relu_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_rvv_relu6_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params);
int shl_rvv_relu6_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params);
int shl_rvv_relu6_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params);

int shl_rvv_leaky_relu_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params);
int shl_rvv_leaky_relu_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params);
int shl_rvv_leaky_relu_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params);

int shl_rvv_sigmoid_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_sigmoid_params *params);

int shl_rvv_softmax_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params);
int shl_rvv_softmax_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params);
int shl_rvv_softmax_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params);

int shl_rvv_prelu_fp32(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params);
int shl_rvv_prelu_fp16(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params);
int shl_rvv_prelu_int8(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params);

/************************************ layout/memory transform *********************************/
int shl_rvv_concat_fp32(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_concat_params *params);
int shl_rvv_concat_fp16(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_concat_params *params);
int shl_rvv_concat_int8(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_concat_params *params);

int shl_rvv_reshape_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params);
int shl_rvv_reshape_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params);
int shl_rvv_reshape_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params);

/************************************ basic math *********************************/
int shl_rvv_add_fp32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);
int shl_rvv_add_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);
int shl_rvv_add_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_rvv_mul_fp32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);
int shl_rvv_mul_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);
int shl_rvv_mul_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_rvv_sum_stride_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params);

/************************************ utils *********************************/
void shl_rvv_pad_input_fp32(const float *input, float *input_padded, int inc, int inh, int inw,
                            int padded_h, int padded_w, int pad_top, int pad_left);
void shl_rvv_pad_input_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh, int inw,
                            int padded_h, int padded_w, int pad_top, int pad_left);
void shl_rvv_pad_input_int8(const int8_t *input, int8_t *input_padded, int inc, int inh, int inw,
                            int padded_h, int padded_w, int pad_top, int pad_left,
                            int8_t pad_value);

void shl_rvv_pad_input_packn_fp32(const float *input, float *input_padded, int inc, int inh,
                                  int inw, int padded_h, int padded_w, int pad_top, int pad_left);
void shl_rvv_pad_input_packn_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh,
                                  int inw, int padded_h, int padded_w, int pad_top, int pad_left);
void shl_rvv_pad_input_packn_int8(const int8_t *input, int8_t *input_padded, int inc, int inh,
                                  int inw, int padded_h, int padded_w, int pad_top, int pad_left,
                                  int8_t pad_value);

void shl_rvv_pad_input_pack1ton_fp32(const float *input, float *input_padded, int inc, int inh,
                                     int inw, int padded_h, int padded_w, int pad_top,
                                     int pad_left);
void shl_rvv_pad_input_pack1ton_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh,
                                     int inw, int padded_h, int padded_w, int pad_top,
                                     int pad_left);
void shl_rvv_pad_input_pack1ton_int8(const int8_t *input, int8_t *input_padded, int inc, int inh,
                                     int inw, int padded_h, int padded_w, int pad_top, int pad_left,
                                     int8_t pad_value);

void shl_rvv_pad_input_nhwc_fp32(const float *input, float *input_padded, int inh, int inw, int inc,
                                 int padded_h, int padded_w, int pad_top, int pad_left);
void shl_rvv_pad_input_nhwc_fp16(const __fp16 *input, __fp16 *input_padded, int inh, int inw,
                                 int inc, int padded_h, int padded_w, int pad_top, int pad_left);
void shl_rvv_pad_input_nhwc_int8(const int8_t *input, int8_t *input_padded, int inh, int inw,
                                 int inc, int padded_h, int padded_w, int pad_top, int pad_left,
                                 int8_t pad_value);

void shl_rvv_reorder_input_pack1ton_fp32(const float *src, float *dst, int inc, int inh, int inw);
void shl_rvv_reorder_input_pack1ton_fp16(const __fp16 *src, __fp16 *dst, int inc, int inh, int inw);
void shl_rvv_reorder_input_pack1ton_int8(const int8_t *src, int8_t *dst, int inc, int inh, int inw);
void shl_rvv_reorder_input_packnto1_fp32(const float *src, float *dst, int inc, int inh, int inw);
void shl_rvv_reorder_input_packnto1_fp16(const __fp16 *src, __fp16 *dst, int inc, int inh, int inw);
void shl_rvv_reorder_input_packnto1_int8(const int8_t *src, int8_t *dst, int inc, int inh, int inw);

void shl_rvv_saturated_int8(int32_t *src, int8_t *dst, int32_t out_zp, int size);

void shl_rvv_requantize(int32_t *src, int32_t multiplier, int32_t shift, int channel_size);

void shl_rvv_pad_input_int4_trans_int8(const int8_t *input, int8_t *input_padded, int inc, int inh,
                                       int inw, int padded_h, int padded_w, int pad_top,
                                       int pad_left, int8_t pad_value);
void shl_rvv_int4_to_int8(int8_t *src, int8_t *dst, int size);
void shl_rvv_int8_to_int4(int8_t *src, int8_t *dst, int size);
void shl_rvv_int4_trans_int8(int8_t *src, int8_t *dst, int size);
void shl_rvv_saturated_int4(int32_t *src, int8_t *dst, int32_t out_zp, int size);

int shl_rvv_data_convert_int8_to_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_siso_params *params);
int shl_rvv_data_convert_int4_to_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_siso_params *params);

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
void shl_rvv_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec,
                    void *est);

int csrr_vl();
int csrr_vlenb();

enum avgpool_loc_enum {
    AVGPOOL_LEFT_TOP = 0,
    AVGPOOL_RIGHT_TOP,
    AVGPOOL_LEFT_BOTTOM,
    AVGPOOL_RIGHT_BOTTOM,
    AVGPOOL_LEFT,
    AVGPOOL_RIGHT,
    AVGPOOL_TOP,
    AVGPOOL_BOTTOM,
    AVGPOOL_CENTER,
};

int shl_rvv_avgpool_get_window_size(struct csinn_pool_params *params, int idx_h_start,
                                    int idx_h_end, int idx_w_start, int idx_w_end,
                                    enum avgpool_loc_enum loc);

#ifdef SHL_USE_DOT_INT4
int shl_rvv_conv2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_int4(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
void shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int4(struct csinn_tensor *kernel,
                                                        struct csinn_conv2d_params *params);
int shl_rvv_conv_im2col_gemm_packn_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_int4(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
void shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_int4(struct csinn_tensor *kernel,
                                                      struct csinn_conv2d_params *params);
int shl_rvv_conv1x1s1_gemm_packn_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);
void shl_rvv_fc_gemv_transform_weight_int4_dot(struct csinn_tensor *weights);
int shl_rvv_fullyconnected_packn_int4_dot(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *weights, struct csinn_tensor *bias,
                                          struct csinn_fc_params *params);
#endif

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_RVV_H_

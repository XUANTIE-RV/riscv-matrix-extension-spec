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

#ifndef INCLUDE_SHL_REF_H_
#define INCLUDE_SHL_REF_H_

#include "csi_nn.h"
#include "shl_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

int shl_ref_abs_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_abs_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_acos_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_acos_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_acosh_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_acosh_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_add_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_add_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_add_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_add_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_and_u32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_and_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_and_i8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_arange_f32(struct csinn_tensor *output, struct csinn_arange_params *params);

int shl_ref_arange_quant(struct csinn_tensor *output, struct csinn_arange_params *params);

int shl_ref_argmax_stride_i32_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_reduce_params *params);

int shl_ref_argmax_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_reduce_params *params);

int shl_ref_argmin_stride_i32_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_reduce_params *params);

int shl_ref_argmin_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_reduce_params *params);

int shl_ref_asin_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_asin_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_asinh_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_asinh_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_atan_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_atan_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_atanh_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_atanh_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_avgpool2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int shl_ref_avgpool2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_pool_params *params);

int shl_ref_avgpool3d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int shl_ref_avgpool3d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_pool_params *params);

int shl_ref_batch_normalization_f32(struct csinn_tensor *input, struct csinn_tensor *mean,
                                    struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                    struct csinn_tensor *beta, struct csinn_tensor *output,
                                    struct csinn_bn_params *params);

int shl_ref_batch_normalization_quant(struct csinn_tensor *input, struct csinn_tensor *mean,
                                      struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                      struct csinn_tensor *beta, struct csinn_tensor *output,
                                      struct csinn_bn_params *params);

int shl_ref_batch_to_space_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_batch_to_space_params *params);

int shl_ref_batch_to_space_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_batch_to_space_params *params);

int shl_ref_broadcast_to_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_broadcast_to_params *params);

int shl_ref_broadcast_to_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_broadcast_to_params *params);

int shl_ref_ceil_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_ceil_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_clip_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_clip_params *params);

int shl_ref_clip_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_clip_params *params);

int shl_ref_col2im_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_col2im_params *params);

int shl_ref_concat_f32(struct csinn_tensor **input, struct csinn_tensor *output,
                       struct csinn_concat_params *params);

int shl_ref_concat_quant(struct csinn_tensor **input, struct csinn_tensor *output,
                         struct csinn_concat_params *params);

int shl_ref_conv1d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv1d_params *params);

int shl_ref_conv1d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv1d_params *params);

int shl_ref_conv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params);

int shl_ref_conv2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params);

int shl_ref_conv2d_channel_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);

int shl_ref_conv2d_relu_f32(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                            struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                            struct csinn_conv2d_params *params);

int shl_ref_conv2d_relu_quant(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                              struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                              struct csinn_conv2d_params *params);

int shl_ref_cache_matmul_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weight, struct csinn_tensor *bias,
                              struct csinn_cache_matmul_params *params);

int shl_ref_cache_matmul_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *weight, struct csinn_tensor *bias,
                             struct csinn_cache_matmul_params *params);

int shl_ref_cache_matmul_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weight, struct csinn_tensor *bias,
                               struct csinn_cache_matmul_params *params);

int shl_ref_cache_conv1d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weight, struct csinn_tensor *bias,
                              struct csinn_cache_conv1d_params *params);

int shl_ref_cache_conv1d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *weight, struct csinn_tensor *bias,
                             struct csinn_cache_conv1d_params *params);

int shl_ref_cache_conv1d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weight, struct csinn_tensor *bias,
                               struct csinn_cache_conv1d_params *params);

int shl_ref_conv2d_channel_relu_quant(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                                      struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                                      struct csinn_conv2d_params *params);

int shl_ref_conv2d_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *kernel, struct csinn_tensor *bias,
                               struct csinn_conv2d_params *params);

int shl_ref_conv2d_channel_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_channel_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_relu_f32(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                                      struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                                      struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_relu_quant(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                                        struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                                        struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_channel_relu_quant(struct csinn_tensor *o_input,
                                                struct csinn_tensor *o_output,
                                                struct csinn_tensor *o_kernel,
                                                struct csinn_tensor *o_bias,
                                                struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

int shl_ref_depthwise_conv2d_channel_relu6_quant(struct csinn_tensor *input,
                                                 struct csinn_tensor *output,
                                                 struct csinn_tensor *kernel,
                                                 struct csinn_tensor *bias,
                                                 struct csinn_conv2d_params *params);

int shl_ref_group_conv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);

int shl_ref_group_conv2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *kernel, struct csinn_tensor *bias,
                               struct csinn_conv2d_params *params);

int shl_ref_group_conv2d_channel_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);

int shl_ref_group_conv2d_relu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params);

int shl_ref_group_conv2d_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params);

int shl_ref_group_conv2d_channel_relu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);

int shl_ref_conv3d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv3d_params *params);

int shl_ref_conv3d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv3d_params *params);

int shl_ref_cos_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_cos_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_cosh_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_cosh_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_cumprod_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_cumprod_params *params);

int shl_ref_cumprod_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_cumprod_params *params);

int shl_ref_cumsum_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_cumsum_params *params);

int shl_ref_cumsum_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_cumsum_params *params);

int shl_ref_data_convert_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_siso_params *params);
int shl_ref_data_convert_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_siso_params *params);

int shl_ref_deconv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params);

int shl_ref_deconv2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv2d_params *params);

int shl_ref_depthwise_deconv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);

int shl_ref_depthwise_deconv2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params);

int shl_ref_deconv3d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv3d_params *params);

int shl_ref_deconv3d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv3d_params *params);

int shl_ref_depth_to_space_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_depth_to_space_params *params);

int shl_ref_depth_to_space_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_depth_to_space_params *params);

int shl_ref_div_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_div_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_elu_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_relu_params *params);

int shl_ref_elu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_ref_fsmn_f32(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                     struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                     struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                     struct csinn_fsmn_params *params);

int shl_ref_fsmn_quant(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                       struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                       struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                       struct csinn_fsmn_params *params);

int shl_ref_equal_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_equal_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_erf_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_erf_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_exp_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_exp_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_expand_dims_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_expand_dims_params *params);

int shl_ref_expand_dims_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_expand_dims_params *params);

int shl_ref_expm1_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_expm1_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_flatten_params *params);

int shl_ref_flatten_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_flatten_params *params);

int shl_ref_floor_divide_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_floor_divide_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                               struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_floor_mod_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_floor_mod_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_floor_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_floor_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_fullyconnected_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weights, struct csinn_tensor *bias,
                               struct csinn_fc_params *params);

int shl_ref_fullyconnected_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params);

int shl_ref_gather_nd_f32(struct csinn_tensor *input, struct csinn_tensor *indices,
                          struct csinn_tensor *output, struct csinn_gather_nd_params *params);

int shl_ref_gather_nd_quant(struct csinn_tensor *input, struct csinn_tensor *indices,
                            struct csinn_tensor *output, struct csinn_gather_nd_params *params);

int shl_ref_gather_f32(struct csinn_tensor *input, struct csinn_tensor *indices,
                       struct csinn_tensor *output, struct csinn_gather_params *params);

int shl_ref_gather_quant(struct csinn_tensor *input, struct csinn_tensor *indices,
                         struct csinn_tensor *output, struct csinn_gather_params *params);

int shl_ref_global_avgpool2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_ref_global_avgpool2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params);

int shl_ref_global_maxpool2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_ref_global_maxpool2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params);

int shl_ref_greater_equal_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_greater_equal_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_greater_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_greater_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_hard_sigmoid_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_sigmoid_params *params);

int shl_ref_hard_sigmoid_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_sigmoid_params *params);

int shl_ref_im2col_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_im2col_params *params);

int shl_ref_im2col_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_im2col_params *params);

int shl_ref_isnan_bool_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_siso_params *params);

int shl_ref_l2_normalization_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_l2n_params *params);

int shl_ref_l2_normalization_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_l2n_params *params);

int shl_ref_l2pool_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params);

int shl_ref_layer_norm_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *gamma, struct csinn_tensor *beta,
                           struct csinn_layer_norm_params *params);

int shl_ref_layer_norm_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *gamma, struct csinn_tensor *beta,
                             struct csinn_layer_norm_params *params);

int shl_ref_leaky_relu_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_relu_params *params);

int shl_ref_leaky_relu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_relu_params *params);

int shl_ref_less_equal_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_less_equal_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_less_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_less_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_log_softmax_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_softmax_params *params);

int shl_ref_log_softmax_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_softmax_params *params);

int shl_ref_log_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_log_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_log1p_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_log1p_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_logical_and_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_logical_and_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_logical_not_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_siso_params *params);

int shl_ref_logical_not_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_siso_params *params);

int shl_ref_logical_or_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_logical_or_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_logical_xor_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_logical_xor_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_lrn_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_lrn_params *params);

int shl_ref_lrn_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_lrn_params *params);

int shl_ref_matmul_f32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                       struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_ref_matmul_quant(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_ref_max_stride_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int shl_ref_max_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params);

int shl_ref_maximum_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_maximum_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_maxpool2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int shl_ref_maxpool2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_pool_params *params);

int shl_ref_maxpool2d_locat_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

int shl_ref_maxpool2d_locat_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

int shl_ref_maxpool3d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int shl_ref_maxpool3d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_pool_params *params);

int shl_ref_mean_stride_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params);

int shl_ref_mean_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_reduce_params *params);

int shl_ref_mean_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_reduce_params *params);

int shl_ref_min_stride_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int shl_ref_min_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params);

int shl_ref_minimum_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_minimum_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_mod_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_mod_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_mul_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_mul_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_ndarray_size_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_ndarray_size_params *params);

int shl_ref_ndarray_size_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_ndarray_size_params *params);

int shl_ref_ndarray_size_i8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_ndarray_size_params *params);

int shl_ref_ndarray_size_i32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_ndarray_size_params *params);

int shl_ref_negative_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_siso_params *params);

int shl_ref_negative_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_siso_params *params);

int shl_ref_non_max_suppression_std(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                    struct csinn_tensor *output,
                                    struct csinn_non_max_suppression_params *params);

int shl_ref_not_equal_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_not_equal_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_not_u32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_not_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_ref_not_i8(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_ref_or_u32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_or_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_or_i8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_pad_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pad_params *params);

int shl_ref_pad_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_pad_params *params);

int shl_ref_power_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_power_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_prelu_f32(struct csinn_tensor *input, struct csinn_tensor *alpha,
                      struct csinn_tensor *output, struct csinn_prelu_params *params);

int shl_ref_prelu_quant(struct csinn_tensor *input, struct csinn_tensor *alpha,
                        struct csinn_tensor *output, struct csinn_prelu_params *params);

int shl_ref_prod_stride_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params);

int shl_ref_prod_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_reduce_params *params);

int shl_ref_proposal_f32(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                         struct csinn_tensor *im_info, struct csinn_tensor *output,
                         struct csinn_proposal_params *params);

int shl_ref_proposal_quant(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                           struct csinn_tensor *im_info, struct csinn_tensor *output,
                           struct csinn_proposal_params *params);

int shl_ref_psroipooling_f32(struct csinn_tensor *data, struct csinn_tensor *rois,
                             struct csinn_tensor *output, struct csinn_psroipooling_params *params);

int shl_ref_psroipooling_quant(struct csinn_tensor *data, struct csinn_tensor *rois,
                               struct csinn_tensor *output,
                               struct csinn_psroipooling_params *params);

int shl_ref_reduce_logsumexp_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_reduce_params *params);

int shl_ref_reduce_logsumexp_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_reduce_params *params);

int shl_ref_reduce_max_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int shl_ref_reduce_max_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params);

int shl_ref_reduce_mean_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params);

int shl_ref_reduce_mean_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_reduce_params *params);

int shl_ref_reduce_min_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int shl_ref_reduce_min_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params);

int shl_ref_reduce_prod_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params);

int shl_ref_reduce_prod_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_reduce_params *params);

int shl_ref_reduce_sum_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int shl_ref_reduce_sum_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params);

int shl_ref_relu_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int shl_ref_relu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params);

int shl_ref_relu1_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_ref_relu1_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params);

int shl_ref_relu6_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_ref_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params);

int shl_ref_relun_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_ref_relun_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params);

int shl_ref_reshape(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reshape_params *params);

int shl_ref_reshape_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reshape_params *params);

int shl_ref_resize_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_resize_params *params);

int shl_ref_resize_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_resize_params *params);

int shl_ref_reverse_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reverse_params *params);

int shl_ref_reverse_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reverse_params *params);

int shl_ref_roi_align_f32(struct csinn_tensor *data, struct csinn_tensor *rois,
                          struct csinn_tensor *output, struct csinn_roi_align_params *params);

int shl_ref_roipool_f32(struct csinn_tensor *data, struct csinn_tensor *rois,
                        struct csinn_tensor *output, struct csinn_roi_pool_params *params);

int shl_ref_roipool_quant(struct csinn_tensor *data, struct csinn_tensor *rois,
                          struct csinn_tensor *output, struct csinn_roi_pool_params *params);

int shl_ref_round_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_round_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_rsqrt_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_rsqrt_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_scatter_nd_f32(struct csinn_tensor *input, struct csinn_tensor *indices,
                           struct csinn_tensor *updates, struct csinn_tensor *output,
                           struct csinn_scatter_nd_params *params);

int shl_ref_scatter_nd_quant(struct csinn_tensor *input, struct csinn_tensor *indices,
                             struct csinn_tensor *updates, struct csinn_tensor *output,
                             struct csinn_scatter_nd_params *params);

int shl_ref_unsorted_segment_max_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                     struct csinn_tensor *output,
                                     struct csinn_segment_params *params);

int shl_ref_segment_max_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                            struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_max_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                       struct csinn_tensor *output,
                                       struct csinn_segment_params *params);

int shl_ref_segment_max_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                              struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_mean_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                      struct csinn_tensor *output,
                                      struct csinn_segment_params *params);

int shl_ref_segment_mean_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                             struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_mean_quant(struct csinn_tensor *input,
                                        struct csinn_tensor *segment_ids,
                                        struct csinn_tensor *output,
                                        struct csinn_segment_params *params);

int shl_ref_segment_mean_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                               struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_min_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                     struct csinn_tensor *output,
                                     struct csinn_segment_params *params);

int shl_ref_segment_min_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                            struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_min_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                       struct csinn_tensor *output,
                                       struct csinn_segment_params *params);

int shl_ref_segment_min_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                              struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_prod_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                      struct csinn_tensor *output,
                                      struct csinn_segment_params *params);

int shl_ref_segment_prod_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                             struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_prod_quant(struct csinn_tensor *input,
                                        struct csinn_tensor *segment_ids,
                                        struct csinn_tensor *output,
                                        struct csinn_segment_params *params);

int shl_ref_segment_prod_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                               struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_sum_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                     struct csinn_tensor *output,
                                     struct csinn_segment_params *params);

int shl_ref_segment_sum_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                            struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_unsorted_segment_sum_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                       struct csinn_tensor *output,
                                       struct csinn_segment_params *params);

int shl_ref_segment_sum_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                              struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_ref_select_f32(struct csinn_tensor *condition, struct csinn_tensor *input0,
                       struct csinn_tensor *input1, struct csinn_tensor *output,
                       struct csinn_select_params *params);

int shl_ref_select_u8(struct csinn_tensor *condition, struct csinn_tensor *input0,
                      struct csinn_tensor *input1, struct csinn_tensor *output,
                      struct csinn_select_params *params);

int shl_ref_select_i8(struct csinn_tensor *condition, struct csinn_tensor *input0,
                      struct csinn_tensor *input1, struct csinn_tensor *output,
                      struct csinn_select_params *params);

int shl_ref_shape_i32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_shape_params *params);

int shl_ref_shape_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_shape_params *params);

int shl_ref_shape_i8(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_shape_params *params);

int shl_ref_shuffle_channel_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_shuffle_channel_params *params);

int shl_ref_shuffle_channel_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_shuffle_channel_params *params);

int shl_ref_sigmoid_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_sigmoid_params *params);

int shl_ref_sigmoid_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_sigmoid_params *params);

int shl_ref_sign_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_sign_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_sin_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_sin_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_sinh_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_sinh_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_slice_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_slice_params *params);

int shl_ref_slice_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_slice_params *params);

int shl_ref_softmax_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_softmax_params *params);

int shl_ref_softmax_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_softmax_params *params);

int shl_ref_softplus_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_siso_params *params);

int shl_ref_softplus_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_siso_params *params);

int shl_ref_softrelu_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_relu_params *params);

int shl_ref_softrelu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_relu_params *params);

int shl_ref_softsign_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_siso_params *params);

int shl_ref_softsign_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_siso_params *params);

int shl_ref_space_to_batch_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_space_to_batch_params *params);

int shl_ref_space_to_batch_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_space_to_batch_params *params);

int shl_ref_space_to_depth_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_space_to_depth_params *params);

int shl_ref_space_to_depth_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_space_to_depth_params *params);

int shl_ref_split_f32(struct csinn_tensor *input, struct csinn_tensor **output,
                      struct csinn_split_params *params);

int shl_ref_split_quant(struct csinn_tensor *input, struct csinn_tensor **output,
                        struct csinn_split_params *params);

int shl_ref_sqrt_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_sqrt_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_square_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_squeeze(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_squeeze_params *params);

int shl_ref_stack_f32(struct csinn_tensor **input, struct csinn_tensor *output,
                      struct csinn_stack_params *params);

int shl_ref_stack_quant(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_stack_params *params);

int shl_ref_strided_slice_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_strided_slice_params *params);

int shl_ref_strided_slice_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_strided_slice_params *params);

int shl_ref_sub_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_sub_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_sum_stride_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int shl_ref_sum_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params);

int shl_ref_tan_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_ref_tan_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_tanh_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_tanh_f64(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_ref_tanh_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

int shl_ref_threshold_relu_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_relu_params *params);

int shl_ref_threshold_relu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_relu_params *params);

int shl_ref_tile_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_tile_params *params);

int shl_ref_tile_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tile_params *params);

int shl_ref_topk_f32(struct csinn_tensor *input, struct csinn_tensor *output1,
                     struct csinn_tensor *output2, struct csinn_topk_params *params);

int shl_ref_topk_quant(struct csinn_tensor *input, struct csinn_tensor *output1,
                       struct csinn_tensor *output2, struct csinn_topk_params *params);

int shl_ref_transpose(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_transpose_params *params);

int shl_ref_transpose_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_transpose_params *params);

int shl_ref_trunc_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_ref_trunc_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_ref_unpooling_f32(struct csinn_tensor *input, struct csinn_tensor *mask,
                          struct csinn_tensor *output, struct csinn_unpooling_params *params);

int shl_ref_unpooling_quant(struct csinn_tensor *input, struct csinn_tensor *mask,
                            struct csinn_tensor *output, struct csinn_unpooling_params *params);

int shl_ref_unstack_f32(struct csinn_tensor *input, struct csinn_tensor **output,
                        struct csinn_unstack_params *params);

int shl_ref_unstack_qunat(struct csinn_tensor *input, struct csinn_tensor **output,
                          struct csinn_unstack_params *params);

int shl_ref_xor_u32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_xor_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_xor_i8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_ref_yuv_rgb_scale_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_siso_params *params);

int shl_ref_yuv_rgb_scale_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_siso_params *params);

int shl_ref_one_hot_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_one_hot_params *params);

int shl_ref_one_hot_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_one_hot_params *params);

int shl_ref_where_f32(struct csinn_tensor *condition, struct csinn_tensor *x,
                      struct csinn_tensor *y, struct csinn_tensor *output,
                      struct csinn_where_params *params);

int shl_ref_where_quant(struct csinn_tensor *condition, struct csinn_tensor *x,
                        struct csinn_tensor *y, struct csinn_tensor *output,
                        struct csinn_where_params *params);

int32_t shl_ref_max_internal_s32(int32_t a, int32_t b);
int32_t shl_ref_min_internal_s32(int32_t a, int32_t b);
int32_t shl_ref_get_index(int32_t *dim, int32_t index0, int32_t index1, int32_t index2,
                          int32_t index3);
int32_t shl_ref_get_index_5(int32_t *dim, int32_t index0, int32_t index1, int32_t index2,
                            int32_t index3, int32_t index4);
int32_t shl_ref_get_index_iter(int32_t *dim, int dim_count, int32_t *index);
float shl_ref_get_scale(int32_t multiplier, int32_t shift);
float shl_ref_dequantize_u8_to_f32(uint8_t input, struct csinn_quant_info *qinfo);
float shl_ref_dequantize_i8_to_f32(int8_t input, struct csinn_quant_info *qinfo);
uint8_t shl_ref_quantize_f32_to_u8(float input, struct csinn_quant_info *qinfo);
int8_t shl_ref_quantize_f32_to_i8(float input, struct csinn_quant_info *qinfo);
uint8_t shl_ref_quantize_channel_u8(int32_t data, struct csinn_tensor *input,
                                    struct csinn_tensor *output, float wscale);
int8_t shl_ref_quantize_channel_i8(int32_t data, struct csinn_tensor *input,
                                   struct csinn_tensor *output, float wscale);
float shl_ref_uint8_to_float(uint8_t i, struct csinn_tensor *t);
float shl_ref_int8_to_float(int8_t i, struct csinn_tensor *t);
int16_t shl_ref_float32_to_float16(float value);
float shl_ref_float16_to_float32(int16_t value);
int16_t shl_ref_float32_to_bfloat16(float value);
float shl_ref_bfloat16_to_float32(int16_t value);
struct csinn_tensor *shl_ref_nchw_to_nhwc_8(struct csinn_tensor *t);
void shl_ref_nhwc_to_nchw_8(struct csinn_tensor *nt, struct csinn_tensor *t);
struct csinn_tensor *shl_ref_deconv_kernel_nchw_to_nhwc_f32(struct csinn_tensor *t,
                                                            int32_t permute[4]);
struct csinn_tensor *shl_ref_nchw_to_nhwc_f32(struct csinn_tensor *t);
void shl_ref_nhwc_to_nchw_f32(struct csinn_tensor *nt, struct csinn_tensor *t);
int32_t shl_ref_get_reduction_index(int32_t k, const int32_t *strides, const int32_t *extents,
                                    int32_t n);
struct csinn_tensor *shl_ref_alloc_float_tensor(struct csinn_tensor *src);
void shl_ref_free_float_tensor(struct csinn_tensor *src);
struct csinn_tensor *shl_ref_convert_float_tensor(struct csinn_tensor *src);
void shl_ref_conv_free_float_tensor(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias);
struct csinn_tensor *shl_ref_tensor_transform_f32(struct csinn_tensor *input);
int shl_ref_tensor_transform_free_f32(struct csinn_tensor *input);
uint8_t *shl_ref_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess);

struct shl_ref_diso_callback {
    void (*bc)();
    struct csinn_tensor *input0;
    struct csinn_tensor *input1;
    struct csinn_tensor *output;
    int32_t *input_dim;
};

int shl_ref_diso_broadcast_base(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                struct csinn_tensor *output, struct csinn_diso_params *params,
                                struct shl_ref_diso_callback *cb);
int shl_ref_broadcast_to_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                               int32_t *shape, int32_t shape_count);
int shl_ref_broadcast_to_shape_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                   int32_t *shape, int32_t shape_count);
int shl_ref_broadcast_to_shape_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                     int32_t *shape, int32_t shape_count);

int shl_ref_siso_callback_base(struct csinn_tensor *input, struct csinn_tensor *output,
                               void *params, void *cb);
int shl_ref_diso_callback_base(struct csinn_tensor *input0, struct csinn_tensor *input1,
                               struct csinn_tensor *output, void *params, void *cb);
int shl_ref_conv_callback_base(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *kernel, struct csinn_tensor *bias, void *params,
                               void *cb);

void shl_ref_nn_init(struct csinn_tensor *input, struct csinn_tensor *output);

void shl_ref_nn_deinit(struct csinn_tensor *input, struct csinn_tensor *output);

int shl_ref_flatten_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params);

int shl_ref_reshape_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params);

int shl_ref_transpose_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_transpose_params *params);

void asr_buffer_init(struct csinn_asr_buffer_t *buffer, size_t buffer_size, size_t data_lenth);

void *asr_buffer_insert_front(struct csinn_asr_buffer_t *buffer, void *input, size_t len);

void *asr_buffer_insert_back(struct csinn_asr_buffer_t *buffer, void *input, size_t len);

void *asr_buffer_get_buffer(struct csinn_asr_buffer_t *buffer);

void asr_buffer_reset(struct csinn_asr_buffer_t *buffer);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_REF_H_

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

#ifndef INCLUDE_CSI_NN_H_
#define INCLUDE_CSI_NN_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csinn_data_structure.h"
#include "csinn_runtime.h"
#include "shl_debug.h"
#include "shl_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

int csinn_conv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv2d_params *params);

int csinn_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                 struct csinn_conv2d_params *params);

int csinn_depthwise_conv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

int csinn_depthwise_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv2d_params *params);

int csinn_group_conv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                            struct csinn_conv2d_params *params);

int csinn_group_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params);

int csinn_conv2d_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv2d_params *params);

int csinn_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv2d_params *params);

int csinn_depthwise_conv2d_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params);

int csinn_depthwise_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

int csinn_conv2d_relu6_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                            struct csinn_conv2d_params *params);

int csinn_conv2d_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params);

int csinn_deconv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv2d_params *params);

int csinn_deconv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                   struct csinn_conv2d_params *params);

int csinn_conv3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv3d_params *params);

int csinn_conv3d(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                 struct csinn_conv3d_params *params);

int csinn_deconv3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv3d_params *params);

int csinn_deconv3d(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                   struct csinn_conv3d_params *params);

int csinn_fsmn_init(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                    struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                    struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                    struct csinn_fsmn_params *params);

int csinn_fsmn(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
               struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
               struct csinn_tensor *frame_counter, struct csinn_tensor *output,
               struct csinn_fsmn_params *params);

int csinn_fullyconnected_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weights, struct csinn_tensor *bias,
                              struct csinn_fc_params *params);

int csinn_fullyconnected(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *weights, struct csinn_tensor *bias,
                         struct csinn_fc_params *params);

int csinn_fullyconnected_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *weights, struct csinn_tensor *bias,
                                   struct csinn_fc_params *params);

int csinn_fullyconnected_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weights, struct csinn_tensor *bias,
                              struct csinn_fc_params *params);

int csinn_maxpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

int csinn_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

int csinn_maxpool3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

int csinn_maxpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

int csinn_global_maxpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

int csinn_global_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

int csinn_avgpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

int csinn_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

int csinn_avgpool3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

int csinn_avgpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

int csinn_global_avgpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

int csinn_global_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

int csinn_l2pool_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_pool_params *params);

int csinn_l2pool(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_pool_params *params);

int csinn_pool_with_argmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

int csinn_pool_with_argmax(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

int csinn_maxpool2d_locat_init(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);

int csinn_maxpool2d_locat(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int csinn_unpooling_init(struct csinn_tensor *input, struct csinn_tensor *mask,
                         struct csinn_tensor *output, struct csinn_unpooling_params *params);

int csinn_unpooling(struct csinn_tensor *input, struct csinn_tensor *mask,
                    struct csinn_tensor *output, struct csinn_unpooling_params *params);

int csinn_roi_align_init(struct csinn_tensor *data, struct csinn_tensor *rois,
                         struct csinn_tensor *output, struct csinn_roi_align_params *params);

int csinn_roi_align(struct csinn_tensor *data, struct csinn_tensor *rois,
                    struct csinn_tensor *output, struct csinn_roi_align_params *params);

int csinn_negative_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int csinn_negative(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_floor_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_floor(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_ceil_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_ceil(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_sign_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_sign(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_trunc_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_trunc(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_round_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_round(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_abs_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_abs(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_isnan_bool_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_siso_params *params);

int csinn_isnan_bool(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_exp_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_exp(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_expm1_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_expm1(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_sin_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_sin(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_cos_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_cos(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_tanh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_tanh(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_log_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_log(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_sqrt_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_sqrt(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_rsqrt_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_rsqrt(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_square_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int csinn_square(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int csinn_sigmoid_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_sigmoid_params *params);

int csinn_sigmoid(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_sigmoid_params *params);

int csinn_hard_sigmoid_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_sigmoid_params *params);

int csinn_hard_sigmoid(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_sigmoid_params *params);

int csinn_elu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

int csinn_elu(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_relu_params *params);

int csinn_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_relu_params *params);

int csinn_relu(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_relu_params *params);

int csinn_relu1_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int csinn_relu1(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_relu_params *params);

int csinn_relu6_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int csinn_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_relu_params *params);

int csinn_relun_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int csinn_relun(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_relu_params *params);

int csinn_leaky_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_relu_params *params);

int csinn_leaky_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int csinn_softrelu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params);

int csinn_softrelu(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

int csinn_prelu_init(struct csinn_tensor *input, struct csinn_tensor *alpha,
                     struct csinn_tensor *output, struct csinn_prelu_params *params);

int csinn_prelu(struct csinn_tensor *input, struct csinn_tensor *alpha, struct csinn_tensor *output,
                struct csinn_prelu_params *params);

int csinn_softplus_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int csinn_softplus(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_softmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_softmax_params *params);

int csinn_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_softmax_params *params);

int csinn_log_softmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_softmax_params *params);

int csinn_log_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_softmax_params *params);

int csinn_batch_normalization_init(struct csinn_tensor *input, struct csinn_tensor *mean,
                                   struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                   struct csinn_tensor *beta, struct csinn_tensor *output,
                                   struct csinn_bn_params *params);

int csinn_batch_normalization(struct csinn_tensor *input, struct csinn_tensor *mean,
                              struct csinn_tensor *variance, struct csinn_tensor *gamma,
                              struct csinn_tensor *beta, struct csinn_tensor *output,
                              struct csinn_bn_params *params);

int csinn_l2_normalization_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_l2n_params *params);

int csinn_l2_normalization(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_l2n_params *params);

int csinn_lrn_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_lrn_params *params);

int csinn_lrn(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_lrn_params *params);

int csinn_matmul_init(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                      struct csinn_tensor *output, struct csinn_matmul_params *params);

int csinn_matmul(struct csinn_tensor *mat0, struct csinn_tensor *mat1, struct csinn_tensor *output,
                 struct csinn_matmul_params *params);

int csinn_add_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_add(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

int csinn_sub_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_sub(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

int csinn_mul_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_mul(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

int csinn_div_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_div(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

int csinn_floor_divide_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_floor_divide(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_floor_mod_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_floor_mod(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_mod_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_mod(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

int csinn_maximum_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_maximum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_minimum_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_minimum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_power_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_power(struct csinn_tensor *input0, struct csinn_tensor *input1,
                struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_greater_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_greater(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_less_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_less(struct csinn_tensor *input0, struct csinn_tensor *input1,
               struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_logical_and_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_logical_and(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_logical_or_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_logical_or(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_logical_not_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_siso_params *params);

int csinn_logical_not(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int csinn_logical_xor_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_logical_xor(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_not_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_not_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_greater_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_greater_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_less_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_less_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_select_init(struct csinn_tensor *condition, struct csinn_tensor *input0,
                      struct csinn_tensor *input1, struct csinn_tensor *output,
                      struct csinn_select_params *params);

int csinn_select(struct csinn_tensor *condition, struct csinn_tensor *input0,
                 struct csinn_tensor *input1, struct csinn_tensor *output,
                 struct csinn_select_params *params);

int csinn_and_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_and(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

int csinn_or_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_or(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
             struct csinn_diso_params *params);

int csinn_xor_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int csinn_xor(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

int csinn_not_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_not(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_pad_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_pad_params *params);

int csinn_pad(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_pad_params *params);

int csinn_resize_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_resize_params *params);

int csinn_resize(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_resize_params *params);

int csinn_concat_init(struct csinn_tensor **input, struct csinn_tensor *output,
                      struct csinn_concat_params *params);

int csinn_concat(struct csinn_tensor **input, struct csinn_tensor *output,
                 struct csinn_concat_params *params);

int csinn_proposal_init(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                        struct csinn_tensor *im_info, struct csinn_tensor *output,
                        struct csinn_proposal_params *params);

int csinn_proposal(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                   struct csinn_tensor *im_info, struct csinn_tensor *output,
                   struct csinn_proposal_params *params);

int csinn_psroipooling_init(struct csinn_tensor *data, struct csinn_tensor *rois,
                            struct csinn_tensor *output, struct csinn_psroipooling_params *params);

int csinn_psroipooling(struct csinn_tensor *data, struct csinn_tensor *rois,
                       struct csinn_tensor *output, struct csinn_psroipooling_params *params);

int csinn_transpose_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_transpose_params *params);

int csinn_transpose(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_transpose_params *params);

int csinn_reshape_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_reshape_params *params);

int csinn_reshape(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reshape_params *params);

int csinn_shape_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_shape_params *params);

int csinn_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_shape_params *params);

int csinn_expand_dims_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_expand_dims_params *params);

int csinn_expand_dims(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_expand_dims_params *params);

int csinn_reverse_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_reverse_params *params);

int csinn_reverse(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reverse_params *params);

int csinn_flatten_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_flatten_params *params);

int csinn_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_flatten_params *params);

int csinn_crop_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_crop_params *params);

int csinn_crop(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_crop_params *params);

int csinn_slice_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_slice_params *params);

int csinn_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_slice_params *params);

int csinn_split_init(struct csinn_tensor *input, struct csinn_tensor **output,
                     struct csinn_split_params *params);

int csinn_split(struct csinn_tensor *input, struct csinn_tensor **output,
                struct csinn_split_params *params);

int csinn_stack_init(struct csinn_tensor **inputs, struct csinn_tensor *output,
                     struct csinn_stack_params *params);

int csinn_stack(struct csinn_tensor **inputs, struct csinn_tensor *output,
                struct csinn_stack_params *params);

int csinn_unstack_init(struct csinn_tensor *input, struct csinn_tensor **output,
                       struct csinn_unstack_params *params);

int csinn_unstack(struct csinn_tensor *input, struct csinn_tensor **output,
                  struct csinn_unstack_params *params);

int csinn_tile_init(struct csinn_tensor *inputs, struct csinn_tensor *output,
                    struct csinn_tile_params *params);

int csinn_tile(struct csinn_tensor *inputs, struct csinn_tensor *output,
               struct csinn_tile_params *params);

int csinn_arange_init(struct csinn_tensor *output, struct csinn_arange_params *params);

int csinn_arange(struct csinn_tensor *output, struct csinn_arange_params *params);

int csinn_where_init(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                     struct csinn_tensor *output, struct csinn_where_params *params);

int csinn_where(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                struct csinn_tensor *output, struct csinn_where_params *params);

int csinn_gather_init(struct csinn_tensor *input, struct csinn_tensor *indices,
                      struct csinn_tensor *output, struct csinn_gather_params *params);

int csinn_gather(struct csinn_tensor *input, struct csinn_tensor *indices,
                 struct csinn_tensor *output, struct csinn_gather_params *params);

int csinn_gather_nd_init(struct csinn_tensor *input, struct csinn_tensor *indices,
                         struct csinn_tensor *output, struct csinn_gather_nd_params *params);

int csinn_gather_nd(struct csinn_tensor *input, struct csinn_tensor *indices,
                    struct csinn_tensor *output, struct csinn_gather_nd_params *params);

int csinn_squeeze_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_squeeze_params *params);

int csinn_squeeze(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_squeeze_params *params);

int csinn_ndarray_size_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_ndarray_size_params *params);

int csinn_ndarray_size(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_ndarray_size_params *params);

int csinn_space_to_batch_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_space_to_batch_params *params);

int csinn_space_to_batch(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_space_to_batch_params *params);

int csinn_space_to_batch_nd_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_space_to_batch_nd_params *params);

int csinn_space_to_batch_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_space_to_batch_nd_params *params);

int csinn_batch_to_space_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_batch_to_space_params *params);

int csinn_batch_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_batch_to_space_params *params);

int csinn_batch_to_space_nd_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_batch_to_space_nd_params *params);

int csinn_batch_to_space_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_batch_to_space_nd_params *params);

int csinn_space_to_depth_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_space_to_depth_params *params);

int csinn_space_to_depth(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_space_to_depth_params *params);

int csinn_depth_to_space_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_depth_to_space_params *params);

int csinn_depth_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_depth_to_space_params *params);

int csinn_one_hot_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_one_hot_params *params);

int csinn_one_hot(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_one_hot_params *params);

int csinn_sequence_mask_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output,
                             struct csinn_sequence_mask_params *params);

int csinn_sequence_mask(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_sequence_mask_params *params);

int csinn_im2col_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_im2col_params *params);

int csinn_im2col(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_im2col_params *params);

int csinn_col2im_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_col2im_params *params);

int csinn_col2im(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_col2im_params *params);

int csinn_sum_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

int csinn_sum(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

int csinn_mean_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

int csinn_mean(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_reduce_params *params);

int csinn_max_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

int csinn_max(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

int csinn_min_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

int csinn_min(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

int csinn_prod_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

int csinn_prod(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_reduce_params *params);

int csinn_argmin_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

int csinn_argmin(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int csinn_argmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

int csinn_argmax(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int csinn_all_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

int csinn_all(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

int csinn_any_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

int csinn_any(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

int csinn_reorg_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reorg_params *params);

int csinn_reorg(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_reorg_params *params);

int csinn_yuv_rgb_scale_init(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_siso_params *params);

int csinn_yuv_rgb_scale(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int csinn_segment_max_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_max(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_min_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_min(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_sum_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_sum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_mean_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_mean(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_prod_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_segment_prod(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_segment_params *params);

int csinn_threshold_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_relu_params *params);

int csinn_threshold_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_relu_params *params);

int csinn_acos_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);
int csinn_acos(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_acosh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_acosh(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_asin_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_asin(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_asinh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_asinh(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_atan_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_atan(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_atanh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_atanh(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_cosh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_cosh(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_sinh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int csinn_sinh(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

int csinn_tan_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_tan(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_log1p_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int csinn_log1p(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

int csinn_softsign_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int csinn_softsign(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_erf_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int csinn_erf(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

int csinn_cumsum_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_cumsum_params *params);

int csinn_cumsum(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_cumsum_params *params);

int csinn_cumprod_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_cumprod_params *params);

int csinn_cumprod(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_cumprod_params *params);

int csinn_reduce_max_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params);

int csinn_reduce_max(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reduce_params *params);

int csinn_reduce_min_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params);

int csinn_reduce_min(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reduce_params *params);

int csinn_reduce_mean_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int csinn_reduce_mean(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

int csinn_reduce_sum_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params);

int csinn_reduce_sum(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reduce_params *params);

int csinn_reduce_prod_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int csinn_reduce_prod(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

int csinn_reduce_logsumexp_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_reduce_params *params);

int csinn_reduce_logsumexp(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int csinn_broadcast_to_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_broadcast_to_params *params);

int csinn_broadcast_to(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_broadcast_to_params *params);

int csinn_scatter_nd_init(struct csinn_tensor *input, struct csinn_tensor *indices,
                          struct csinn_tensor *updates, struct csinn_tensor *output,
                          struct csinn_scatter_nd_params *params);

int csinn_scatter_nd(struct csinn_tensor *input, struct csinn_tensor *indices,
                     struct csinn_tensor *updates, struct csinn_tensor *output,
                     struct csinn_scatter_nd_params *params);

int csinn_clip_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_clip_params *params);

int csinn_clip(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_clip_params *params);

int csinn_strided_slice_init(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_strided_slice_params *params);

int csinn_strided_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_strided_slice_params *params);

int csinn_topk_init(struct csinn_tensor *input, struct csinn_tensor *output1,
                    struct csinn_tensor *output2, struct csinn_topk_params *params);

int csinn_topk(struct csinn_tensor *input, struct csinn_tensor *output1,
               struct csinn_tensor *output2, struct csinn_topk_params *params);

int csinn_non_max_suppression_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                   struct csinn_tensor *output,
                                   struct csinn_non_max_suppression_params *params);

int csinn_non_max_suppression(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output,
                              struct csinn_non_max_suppression_params *params);

int csinn_shuffle_channel_init(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_shuffle_channel_params *params);

int csinn_shuffle_channel(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_shuffle_channel_params *params);

int csinn_roipool_init(struct csinn_tensor *data, struct csinn_tensor *rois,
                       struct csinn_tensor *output, struct csinn_roi_pool_params *params);

int csinn_roipool(struct csinn_tensor *data, struct csinn_tensor *rois, struct csinn_tensor *output,
                  struct csinn_roi_pool_params *params);

int csinn_layer_norm_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *gamma, struct csinn_tensor *beta,
                          struct csinn_layer_norm_params *params);

int csinn_layer_norm(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_tensor *gamma, struct csinn_tensor *beta,
                     struct csinn_layer_norm_params *params);

int csinn_cache_matmul_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *weight, struct csinn_tensor *bias,
                            struct csinn_cache_matmul_params *params);

int csinn_cache_matmul(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *weight, struct csinn_tensor *bias,
                       struct csinn_cache_matmul_params *params);

int csinn_cache_conv1d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *weight, struct csinn_tensor *bias,
                            struct csinn_cache_conv1d_params *params);

int csinn_cache_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *weight, struct csinn_tensor *bias,
                       struct csinn_cache_conv1d_params *params);

int csinn_conv1d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv1d_params *params);

int csinn_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                 struct csinn_conv1d_params *params);

int csinn_data_convert_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_siso_params *params);
int csinn_data_convert(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_CSI_NN_H_

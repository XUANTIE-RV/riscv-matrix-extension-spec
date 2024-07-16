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

#ifndef INCLUDE_SHL_GREF_H_
#define INCLUDE_SHL_GREF_H_
#include "csi_nn.h"
#include "shl_node.h"
#include "shl_utils.h"

int shl_gref_acos(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_acosh(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_cos(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_cosh(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_asin(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_asinh(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_tan(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_atan(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_atanh(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_threshold_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params);

int shl_gref_trunc(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_topk(struct csinn_tensor *input, struct csinn_tensor *output1,
                  struct csinn_tensor *output2, struct csinn_topk_params *params);

int shl_gref_cumprod(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_cumprod_params *params);

int shl_gref_cumsum(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_cumsum_params *params);

int shl_gref_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                    struct csinn_conv2d_params *params);

int shl_gref_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                    struct csinn_conv2d_params *params);

int shl_gref_depthwise_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);

int shl_gref_group_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params);

int shl_gref_group_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *kernel, struct csinn_tensor *bias,
                               struct csinn_conv2d_params *params);

int shl_gref_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params);

int shl_gref_conv2d_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params);

int shl_gref_conv3d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                    struct csinn_conv3d_params *params);

int shl_gref_deconv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv2d_params *params);

int shl_gref_deconv3d(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv3d_params *params);

int shl_gref_depthwise_deconv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

int shl_gref_depthwise_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);

int shl_gref_depthwise_conv2d_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params);

int shl_gref_fsmn(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                  struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                  struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                  struct csinn_fsmn_params *params);

int shl_gref_fullyconnected(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *weights, struct csinn_tensor *bias,
                            struct csinn_fc_params *params);

int shl_gref_fullyconnected_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params);

int shl_gref_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params);

int shl_gref_maxpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params);

int shl_gref_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params);

int shl_gref_avgpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params);

int shl_gref_global_avgpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_gref_global_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_gref_global_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_gref_l2pool(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

int shl_gref_pool_with_argmax(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_gref_maxpool2d_locat(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_pool_params *params);

int shl_gref_mod(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_non_max_suppression(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output,
                                 struct csinn_non_max_suppression_params *params);

int shl_gref_unpooling(struct csinn_tensor *input, struct csinn_tensor *mask,
                       struct csinn_tensor *output, struct csinn_unpooling_params *params);

int shl_gref_negative(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_gref_floor(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_ceil(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_clip(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_abs(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_exp(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_sin(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_sinh(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_tanh(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_sqrt(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_rsqrt(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_square(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

int shl_gref_sigmoid(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_sigmoid_params *params);

int shl_gref_softsign(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_gref_space_to_batch_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_space_to_batch_nd_params *params);

int shl_gref_elu(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_relu_params *params);

int shl_gref_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_relu_params *params);

int shl_gref_relu1(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

int shl_gref_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

int shl_gref_relun(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

int shl_gref_roi_align(struct csinn_tensor *data, struct csinn_tensor *rois,
                       struct csinn_tensor *output, struct csinn_roi_align_params *params);

int shl_gref_roipool(struct csinn_tensor *data, struct csinn_tensor *rois,
                     struct csinn_tensor *output, struct csinn_roi_pool_params *params);

int shl_gref_round(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_leaky_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params);

int shl_gref_softrelu(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_gref_prelu(struct csinn_tensor *input, struct csinn_tensor *alpha,
                   struct csinn_tensor *output, struct csinn_prelu_params *params);

int shl_gref_softplus(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_gref_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_softmax_params *params);

int shl_gref_batch_normalization(struct csinn_tensor *input, struct csinn_tensor *mean,
                                 struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                 struct csinn_tensor *beta, struct csinn_tensor *output,
                                 struct csinn_bn_params *params);

int shl_gref_l2_normalization(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_l2n_params *params);

int shl_gref_lrn(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_lrn_params *params);

int shl_gref_matmul(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                    struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_gref_add(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_sub(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_mul(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_div(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_floor_divide(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_floor_mod(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_maximum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_minimum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_power(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_greater(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_less(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_log_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params);

int shl_gref_log(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_log1p(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_not_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_not(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_reduce_logsumexp(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_reduce_params *params);

int shl_gref_reduce_max(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reduce_params *params);

int shl_gref_reduce_mean(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reduce_params *params);

int shl_gref_reduce_min(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reduce_params *params);

int shl_gref_reduce_prod(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reduce_params *params);

int shl_gref_reduce_sum(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reduce_params *params);

int shl_gref_greater_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_less_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_select(struct csinn_tensor *condition, struct csinn_tensor *input0,
                    struct csinn_tensor *input1, struct csinn_tensor *output,
                    struct csinn_select_params *params);

int shl_gref_and(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_or(struct csinn_tensor *input0, struct csinn_tensor *input1,
                struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_pad(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_pad_params *params);

int shl_gref_resize(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_resize_params *params);

int shl_gref_concat(struct csinn_tensor **input, struct csinn_tensor *output,
                    struct csinn_concat_params *params);

int shl_gref_proposal(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                      struct csinn_tensor *im_info, struct csinn_tensor *output,
                      struct csinn_proposal_params *params);

int shl_gref_psroipooling(struct csinn_tensor *data, struct csinn_tensor *rois,
                          struct csinn_tensor *output, struct csinn_psroipooling_params *params);

int shl_gref_transpose(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_transpose_params *params);

int shl_gref_reshape(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reshape_params *params);

int shl_gref_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_shape_params *params);

int shl_gref_strided_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_strided_slice_params *params);

int shl_gref_expand_dims(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_expand_dims_params *params);

int shl_gref_expm1(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

int shl_gref_reverse(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reverse_params *params);

int shl_gref_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_flatten_params *params);

int shl_gref_crop(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_crop_params *params);

int shl_gref_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_slice_params *params);

int shl_gref_split(struct csinn_tensor *input, struct csinn_tensor **output,
                   struct csinn_split_params *params);

int shl_gref_stack(struct csinn_tensor **input, struct csinn_tensor *output,
                   struct csinn_stack_params *params);

int shl_gref_tile(struct csinn_tensor *inputs, struct csinn_tensor *output,
                  struct csinn_tile_params *params);

int shl_gref_arange(struct csinn_tensor *output, struct csinn_arange_params *params);

int shl_gref_where(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                   struct csinn_tensor *output, struct csinn_where_params *params);

int shl_gref_unstack(struct csinn_tensor *input, struct csinn_tensor **output,
                     struct csinn_unstack_params *params);

int shl_gref_gather(struct csinn_tensor *input, struct csinn_tensor *indices,
                    struct csinn_tensor *output, struct csinn_gather_params *params);

int shl_gref_gather_nd(struct csinn_tensor *input, struct csinn_tensor *indices,
                       struct csinn_tensor *output, struct csinn_gather_nd_params *params);

int shl_gref_hard_sigmoid(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_sigmoid_params *params);

int shl_gref_isnan_bool(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

int shl_gref_logical_and(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_logical_not(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_siso_params *params);

int shl_gref_logical_or(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_logical_xor(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_squeeze(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_squeeze_params *params);

int shl_gref_segment_max(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_gref_segment_mean(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_gref_segment_min(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_gref_segment_prod(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_gref_segment_sum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_segment_params *params);

int shl_gref_scatter_nd(struct csinn_tensor *input, struct csinn_tensor *indices,
                        struct csinn_tensor *updates, struct csinn_tensor *output,
                        struct csinn_scatter_nd_params *params);

int shl_gref_shuffle_channel(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_shuffle_channel_params *params);

int shl_gref_sign(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_gref_ndarray_size(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_ndarray_size_params *params);

int shl_gref_space_to_batch(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_space_to_batch_params *params);

int shl_gref_batch_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_batch_to_space_params *params);

int shl_gref_batch_to_space_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_batch_to_space_nd_params *params);

int shl_gref_space_to_depth(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_space_to_depth_params *params);

int shl_gref_depth_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_depth_to_space_params *params);

int shl_gref_broadcast_to(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_broadcast_to_params *params);

int shl_gref_one_hot(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_one_hot_params *params);

int shl_gref_sequence_mask(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_sequence_mask_params *params);

int shl_gref_im2col(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_im2col_params *params);

int shl_gref_col2im(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_col2im_params *params);

int shl_gref_sum(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_gref_mean(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reduce_params *params);

int shl_gref_max(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_gref_min(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_gref_prod(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reduce_params *params);

int shl_gref_argmin(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

int shl_gref_argmax(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

int shl_gref_all(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_gref_any(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_gref_reorg(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reorg_params *params);

int shl_gref_erf(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

int shl_gref_xor(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_gref_yuv_rgb_scale(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_siso_params *params);

int shl_gref_layer_norm(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *gamma, struct csinn_tensor *beta,
                        struct csinn_layer_norm_params *params);

int shl_gref_cache_matmul(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *weight, struct csinn_tensor *bias,
                          struct csinn_cache_matmul_params *params);

int shl_gref_cache_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *weight, struct csinn_tensor *bias,
                          struct csinn_cache_conv1d_params *params);

int shl_gref_data_convert(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_siso_params *params);

int shl_gref_one_hot(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_one_hot_params *params);

struct shl_ref_graph {
    struct shl_node **input;
    struct shl_node **output;
    int input_num;
    int output_num;
    struct shl_node **layer;
    int layer_size;
    int layer_index;
};

struct shl_gref_target_data {
    struct shl_ref_graph *graph;
    int is_hybrid_quantization_type;
};

struct shl_ref_graph *shl_subgraph_establish(struct shl_ref_graph *ograph);
struct shl_ref_graph *shl_gref_get_graph(struct csinn_session *sess);
int shl_gref_graph_insert(struct shl_node *node, struct shl_ref_graph *graph);
void shl_gref_post_dfs(struct shl_ref_graph *graph,
                       void (*fvisit)(struct shl_ref_graph *, struct shl_node *));
int shl_gref_is_root_node(struct shl_ref_graph *graph, struct shl_node *node);
struct shl_node *shl_gref_get_input_subgraph(struct shl_ref_graph *graph, struct shl_node *node,
                                             int index);
void shl_gref_reset_graph_visit(struct shl_ref_graph *graph);
void shl_gref_update_input_output(struct shl_ref_graph *graph, int index);
int shl_gref_siso_op(struct csinn_tensor *input, struct csinn_tensor *output, int op, void *params);
int shl_gref_diso_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, int op, void *params);
int shl_gref_sidcso_op(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *const0, struct csinn_tensor *const1, int op,
                       void *params);
void shl_gref_set_tensor(struct csinn_tensor *tensor, struct csinn_session *sess);
void shl_gref_set_const_tensor(struct csinn_tensor *tensor, struct csinn_session *sess);
int shl_gref_get_tensor(int index, struct csinn_tensor *ret, struct csinn_session *sess);
void shl_gref_nbg(struct csinn_tensor **input, struct csinn_tensor **output, uint32_t inputs_count,
                  uint32_t outputs_count, const char *url);

void shl_subgraph_alloc(struct shl_node *node, struct shl_ref_graph *ograph,
                        struct shl_ref_graph *ggraph);
int shl_subgraph_setup(struct shl_node *n);
int shl_subgraph_deinit(struct shl_node *n);
int shl_subgraph_run_init(struct shl_node *n);
int shl_subgraph_run(struct shl_node *n);
int shl_subgraph_run_deinit(struct shl_node *n, struct shl_ref_graph *graph);

struct shl_ref_graph *shl_subgraph_generate(struct shl_ref_graph *ograph);
struct shl_ref_graph *shl_subgraph_rebuild(struct shl_ref_graph *subgraph);
struct shl_ref_graph *shl_subgraph_topology_sort(struct shl_ref_graph *graph);
void shl_subgraph_fvisit_fuse(struct shl_ref_graph *graph, struct shl_node *node);
void shl_subgraph_fvisit_print(struct shl_ref_graph *graph, struct shl_node *node);
int shl_subgraph_get_device(struct shl_node *node);
void *shl_gref_runtime_callback(int api);
#endif  // INCLUDE_SHL_GREF_H_

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
#ifndef INCLUDE_SHL_DEBUG_H_
#define INCLUDE_SHL_DEBUG_H_
#include "csi_nn.h"
#include "shl_node.h"

enum shl_debug_enum {
    SHL_DEBUG_LEVEL_DEBUG = -2,
    SHL_DEBUG_LEVEL_INFO,
    SHL_DEBUG_LEVEL_WARNING,
    SHL_DEBUG_LEVEL_ERROR,
    SHL_DEBUG_LEVEL_FATAL,
};

#ifdef SHL_DEBUG
#define SHL_DEBUG_CALL(func) func
void shl_debug_debug(const char *format, ...);
void shl_debug_info(const char *format, ...);
void shl_debug_warning(const char *format, ...);
void shl_debug_error(const char *format, ...);
void shl_debug_fatal(const char *format, ...);
int shl_debug_callback_unset();
#else
#define SHL_DEBUG_CALL(func)
inline void shl_debug_debug(const char *format, ...) {}
inline void shl_debug_info(const char *format, ...) {}
inline void shl_debug_warning(const char *format, ...) {}
inline void shl_debug_error(const char *format, ...) {}
inline void shl_debug_fatal(const char *format, ...) {}
inline int shl_debug_callback_unset() { return CSINN_CALLBACK_UNSET; }
#endif

int shl_debug_get_level();
void shl_debug_set_level(int level);
int shl_benchmark_layer(struct shl_node *node, uint64_t start_time, uint64_t end_time,
                        int layer_idx);

int shl_conv2d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params, const char *name);

int shl_conv1d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv1d_params *params, const char *name);

int shl_conv3d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv3d_params *params, const char *name);

int shl_fsmn_debug_info(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                        struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                        struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                        struct csinn_fsmn_params *params, const char *name);

int shl_siso_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params, const char *name);

int shl_diso_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params,
                        const char *name);

int shl_relu_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params, const char *name);

int shl_arange_debug_info(struct csinn_tensor *output, struct csinn_arange_params *params,
                          const char *name);

int shl_pool_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_pool_params *params, const char *name);

int shl_pad_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pad_params *params, const char *name);

int shl_crop_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_crop_params *params, const char *name);

int shl_roi_pool_debug_info(struct csinn_tensor *data, struct csinn_tensor *rois,
                            struct csinn_tensor *output, struct csinn_roi_pool_params *params,
                            const char *name);

int shl_bn_debug_info(struct csinn_tensor *input, struct csinn_tensor *mean,
                      struct csinn_tensor *variance, struct csinn_tensor *gamma,
                      struct csinn_tensor *beta, struct csinn_tensor *output,
                      struct csinn_bn_params *params, const char *name);

int shl_batch_to_space_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_batch_to_space_params *params, const char *name);

int shl_batch_to_space_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_batch_to_space_nd_params *params,
                                     const char *name);

int shl_cache_matmul_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weight, struct csinn_tensor *bias,
                                struct csinn_cache_matmul_params *params, const char *name);

int shl_cache_conv1d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weight, struct csinn_tensor *bias,
                                struct csinn_cache_conv1d_params *params, const char *name);

int shl_space_to_depth_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_space_to_depth_params *params, const char *name);

int shl_depth_to_space_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_depth_to_space_params *params, const char *name);

int shl_space_to_batch_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_space_to_batch_params *params, const char *name);

int shl_space_to_batch_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_space_to_batch_nd_params *params,
                                     const char *name);

int shl_broadcast_to_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_broadcast_to_params *params, const char *name);

int shl_reduce_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params, const char *name);

int shl_clip_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_clip_params *params, const char *name);

int shl_col2im_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_col2im_params *params, const char *name);

int shl_concat_debug_info(struct csinn_tensor **input, struct csinn_tensor *output,
                          struct csinn_concat_params *params, const char *name);

int shl_cumprod_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_cumprod_params *params, const char *name);

int shl_cumsum_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_cumsum_params *params, const char *name);

int shl_expand_dims_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_expand_dims_params *params, const char *name);

int shl_flatten_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_flatten_params *params, const char *name);

int shl_fullyconnected_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *weights, struct csinn_tensor *bias,
                                  struct csinn_fc_params *params, const char *name);

int shl_gather_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *indices,
                             struct csinn_tensor *output, struct csinn_gather_nd_params *params,
                             const char *name);

int shl_gather_debug_info(struct csinn_tensor *input, struct csinn_tensor *indices,
                          struct csinn_tensor *output, struct csinn_gather_params *params,
                          const char *name);

int shl_hard_sigmoid_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_sigmoid_params *params, const char *name);

int shl_im2col_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_im2col_params *params, const char *name);

int shl_l2n_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_l2n_params *params, const char *name);

int shl_layer_norm_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *gamma, struct csinn_tensor *beta,
                              struct csinn_layer_norm_params *params, const char *name);

int shl_softmax_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_softmax_params *params, const char *name);

int shl_lrn_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_lrn_params *params, const char *name);

int shl_matmul_debug_info(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                          struct csinn_tensor *output, struct csinn_matmul_params *params,
                          const char *name);

int shl_ndarray_size_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_ndarray_size_params *params, const char *name);

int shl_nms_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_non_max_suppression_params *params,
                       const char *name);

int shl_one_hot_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_one_hot_params *params, const char *name);

int shl_prelu_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_prelu_params *params,
                         const char *name);

int shl_proposal_debug_info(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                            struct csinn_tensor *im_info, struct csinn_tensor *output,
                            struct csinn_proposal_params *params, const char *name);

int shl_psroipooling_debug_info(struct csinn_tensor *data, struct csinn_tensor *rois,
                                struct csinn_tensor *output,
                                struct csinn_psroipooling_params *params, const char *name);

int shl_reorg_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reorg_params *params, const char *name);

int shl_reshape_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reshape_params *params, const char *name);

int shl_resize_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_resize_params *params, const char *name);

int shl_reverse_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reverse_params *params, const char *name);

int shl_roi_align_debug_info(struct csinn_tensor *data, struct csinn_tensor *rois,
                             struct csinn_tensor *output, struct csinn_roi_align_params *params,
                             const char *name);

int shl_scatter_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *indices,
                              struct csinn_tensor *updates, struct csinn_tensor *output,
                              struct csinn_scatter_nd_params *params, const char *name);

int shl_segment_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params,
                           const char *name);

int shl_select_debug_info(struct csinn_tensor *condition, struct csinn_tensor *input0,
                          struct csinn_tensor *input1, struct csinn_tensor *output,
                          struct csinn_select_params *params, const char *name);

int shl_sequence_mask_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output,
                                 struct csinn_sequence_mask_params *params, const char *name);

int shl_shape_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_shape_params *params, const char *name);

int shl_shuffle_channel_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_shuffle_channel_params *params, const char *name);

int shl_sigmoid_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_sigmoid_params *params, const char *name);

int shl_slice_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_slice_params *params, const char *name);

int shl_split_debug_info(struct csinn_tensor *input, struct csinn_tensor **output,
                         struct csinn_split_params *params, const char *name);

int shl_squeeze_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_squeeze_params *params, const char *name);

int shl_stack_debug_info(struct csinn_tensor **input, struct csinn_tensor *output,
                         struct csinn_stack_params *params, const char *name);

int shl_strided_slice_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_strided_slice_params *params, const char *name);

int shl_tile_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tile_params *params, const char *name);

int shl_topk_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_topk_params *params,
                        const char *name);

int shl_transpose_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_transpose_params *params, const char *name);

int shl_unpooling_debug_info(struct csinn_tensor *input, struct csinn_tensor *mask,
                             struct csinn_tensor *output, struct csinn_unpooling_params *params,
                             const char *name);

int shl_unstack_debug_info(struct csinn_tensor *input, struct csinn_tensor **output,
                           struct csinn_unstack_params *params, const char *name);

int shl_where_debug_info(struct csinn_tensor *condition, struct csinn_tensor *x,
                         struct csinn_tensor *y, struct csinn_tensor *output,
                         struct csinn_where_params *params, const char *name);

#endif  // INCLUDE_SHL_DEBUG_H_

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

#ifndef INCLUDE_SHL_UTILS_H_
#define INCLUDE_SHL_UTILS_H_

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (!defined SHL_BUILD_RTOS)
#include <omp.h>
#endif
#include "csinn_data_structure.h"
#ifdef SHL_MCONF_CONFIG
#include "mconf_config.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void shl_get_top5(float *buf, uint32_t size, float *prob, uint32_t *cls);
void shl_show_top5(struct csinn_tensor *output, struct csinn_session *sess);
uint64_t shl_get_timespec();
void shl_print_time_interval(uint64_t start, uint64_t end, const char *msg);
void shl_statistical_mean_std(float *data, int sz);
void shl_quantize_multiplier(double double_multiplier, int32_t *quantized_multiplier,
                             int32_t *shift);

void shl_register_runtime_callback(int api, void *cb);
void shl_register_op_callback(int api, void *cb);
int shl_op_callback_map(struct csinn_params_base *base, int op, int dtype);

void *shl_get_p0_cb(struct csinn_params_base *base);
void *shl_get_init_cb(struct csinn_params_base *base);

enum csinn_rmode_enum shl_get_run_mode(struct csinn_params_base *base);

struct shl_cb_op_list {
    struct shl_cb_op_list *next;
    enum csinn_dtype_enum dtype;
    enum csinn_op_enum op_name;
    struct csinn_callback *cb;
};

struct shl_cb_op_list *shl_cb_list_end(struct shl_cb_op_list *list);
struct csinn_callback *shl_cb_list_match(struct shl_cb_op_list *list, enum csinn_dtype_enum dtype,
                                         enum csinn_op_enum op_name);

struct shl_bm_sections {
    int32_t graph_offset;
    int32_t graph_size;
    int32_t params_offset;
    int32_t params_size;
    int32_t info_offset;
    int32_t info_size;
    int32_t debug_offset;
    int32_t debug_size;
};

struct shl_binary_model_section_info {
    int32_t section_num;
    int32_t section_info_size;
    int32_t reserve[6];
    struct shl_bm_sections sections[127];
};

char *shl_bm_header_str();

void shl_dump_bm_header(FILE *f);
void shl_dump_bm_section_info(FILE *f, struct shl_binary_model_section_info *info);
void shl_dump_bm_graph_info_section(FILE *f, struct csinn_session *sess);
void shl_bm_session_load(struct csinn_session *dest, struct csinn_session *src);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_UTILS_H_

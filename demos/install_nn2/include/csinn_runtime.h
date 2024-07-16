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

#ifndef INCLUDE_CSINN_RUNTIME_H_
#define INCLUDE_CSINN_RUNTIME_H_

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

#ifdef __cplusplus
extern "C" {
#endif

#define VERSION_MAJOR 2
#define VERSION_MINOR 0
#define VERSION_PATCH 0
#define VERSION_SHIFT 8
int csinn_version(char *vstr);

/* tensor */
int csinn_tensor_size(struct csinn_tensor *tensor);
int csinn_tensor_byte_size(struct csinn_tensor *tensor);
struct csinn_tensor *csinn_alloc_tensor(struct csinn_session *session);
void csinn_free_tensor(struct csinn_tensor *tensor);
void csinn_realloc_quant_info(struct csinn_tensor *tensor, int quant_info_num);
void csinn_tensor_copy(struct csinn_tensor *dest, struct csinn_tensor *src);
int csinn_tensor_data_convert(struct csinn_tensor *dest, struct csinn_tensor *src);
int csinn_tensor_layout_convert(struct csinn_tensor *dest, struct csinn_tensor *src);

/* op parameters */
void *csinn_alloc_params(int params_size, struct csinn_session *session);
void csinn_free_params(void *params);

/* session */
struct csinn_session *csinn_alloc_session();
void csinn_free_session(struct csinn_session *session);
void csinn_session_init(struct csinn_session *session);
void csinn_session_deinit(struct csinn_session *session);
int csinn_session_setup(struct csinn_session *session);
int csinn_session_run(struct csinn_session *session);
int csinn_load_binary_model(struct csinn_session *session);
struct csinn_session *__attribute__((weak)) csinn_import_binary_model(char *bm_addr);

/* input/output */
void csinn_set_input_number(int number, struct csinn_session *sess);
void csinn_set_output_number(int number, struct csinn_session *sess);
int csinn_get_input_number(struct csinn_session *sess);
int csinn_get_output_number(struct csinn_session *sess);
int csinn_set_input(int index, struct csinn_tensor *input, struct csinn_session *sess);
int csinn_set_output(int index, struct csinn_tensor *output, struct csinn_session *sess);
int csinn_get_input(int index, struct csinn_tensor *input, struct csinn_session *sess);
int csinn_get_output(int index, struct csinn_tensor *output, struct csinn_session *sess);
int csinn_update_input(int index, struct csinn_tensor *input, struct csinn_session *sess);
int csinn_update_output(int index, struct csinn_tensor *output, struct csinn_session *sess);
int csinn_set_tensor_entry(struct csinn_tensor *tensor, struct csinn_session *sess);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_CSINN_RUNTIME_H_

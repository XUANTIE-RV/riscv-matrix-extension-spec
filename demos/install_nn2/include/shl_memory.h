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
#ifndef INCLUDE_SHL_MEMORY_H_
#define INCLUDE_SHL_MEMORY_H_

#include <stdint.h>
#include <stdlib.h>

void shl_mem_print_map();
void *shl_mem_alloc(int64_t size);
void *shl_mem_alloc_aligned(int64_t size, int aligned_bytes);
void *shl_mem_calloc(size_t nmemb, size_t size);
void *shl_mem_realloc(void *ptr, size_t size);
void shl_mem_free(void *ptr);

#endif  // INCLUDE_SHL_MEMORY_H_

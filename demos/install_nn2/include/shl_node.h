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

#ifndef INCLUDE_SHL_NODE_H_
#define INCLUDE_SHL_NODE_H_

struct shl_node {
    int type;
    struct shl_node **in;
    struct shl_node **out;
    int subgraph_idx;
    int in_num;
    int out_num;
    char *name;
    void *data;
    int ref_count;
    int ref_count_init;
    int visited;
    int *restricted_map;
    int restricted_map_num;
};

/* node */
struct shl_node *shl_node_alloc(int node_type, char *name, int in_num, int out_num, void *data);
struct shl_node *shl_node_var_alloc(char *name, void *data);
struct shl_node *shl_node_const_var_alloc(char *name, void *data);
int shl_node_free(struct shl_node *node);
int shl_node_add_in(struct shl_node *node, struct shl_node *in, int index);
int shl_node_add_out(struct shl_node *node, struct shl_node *out, int index);
int shl_node_get_in_number(struct shl_node *node);
int shl_node_get_out_number(struct shl_node *node);
int shl_node_get_non_const_in_number(struct shl_node *node);
struct shl_node *shl_node_get_in(struct shl_node *node, int index);
struct shl_node *shl_node_get_out(struct shl_node *node, int index);
int shl_node_restrict_map_insert(int value, struct shl_node *node);
int shl_node_find(struct shl_node **list, int len, struct shl_node *node);

#endif  // INCLUDE_SHL_NODE_H_

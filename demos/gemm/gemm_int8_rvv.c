#include <stdio.h>
#include <string.h>
#include "shl_ref.h"
#include "shl_thead_rvv.h"

int main(int argc, char **argv)
{
    int m = 160;
    int k = 160;
    int n = 160;
    printf("gemm_int8 rvv:[%dx%dx%d]\n", m, n, k);
    int8_t *out_ptr = (int8_t *)malloc(m * n * sizeof(int8_t));
    int8_t *in_ptr = (int8_t *)malloc(m * k * sizeof(int8_t));
    int8_t *ker_ptr = (int8_t *)malloc(k * n * sizeof(int8_t));
    int32_t *bias_ptr = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *mult = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *shift = (int32_t *)malloc(n * sizeof(int32_t));

    shl_rvv_ncxhwx_gemm_4xpack2n_int8(out_ptr, ker_ptr, in_ptr, bias_ptr, m, k, n, n, 100, mult, shift);

    return 0;
}

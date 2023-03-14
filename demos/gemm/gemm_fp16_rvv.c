#include <stdio.h>
#include <string.h>
#include "shl_ref.h"
#include "shl_thead_rvv.h"

int main(int argc, char **argv)
{
    int m = 160;
    int k = 160;
    int n = 160;
    printf("gemm_fp16 rvv:[%dx%dx%d]\n", m, n, k);
    __fp16 *out_ptr = (__fp16 *)malloc(m * n * sizeof(__fp16));
    __fp16 *in_ptr = (__fp16 *)malloc(m * k * sizeof(__fp16));
    __fp16 *ker_ptr = (__fp16 *)malloc(k * n * sizeof(__fp16));
    __fp16 *bias_ptr = (__fp16 *)malloc(n * sizeof(__fp16));

    shl_rvv_ncxhwx_gemm_8xpack2n_fp16(out_ptr, ker_ptr, in_ptr, bias_ptr, m, k, n, n);

    return 0;
}

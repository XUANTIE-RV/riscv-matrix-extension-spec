#include <stdio.h>
#include <thead_matrix.h>
#define N 16

void __attribute__((noinline))
print_data(const char *fmt, mint32_t ma, mint32_t mb, mint32_t ans, mrow_t mrow, mcol_t mcol)
{
  unsigned int row, col;
  int32_t tmp_ma[N];
  int32_t tmp_mb[N];
  int32_t tmp_ans[N];

  printf("%s:\n", fmt);

  __riscv_th_mst(tmp_ma, 8, ma, mrow, mcol);
  __riscv_th_mst(tmp_mb, 8, mb, mrow, mcol);
  __riscv_th_mst(tmp_ans, 8, ans, mrow, mcol);

  printf("ma:\t\tmb:\t\tans:\n");
  for (row = 0; row < mrow; row++)
  {
    for (col = 0; col < mcol; col++)
    {
      printf("%-3d ", tmp_ma[row * mrow + col]);
    }
    printf("\t");
    for (col = 0; col < mcol; col++)
    {
      printf("%-3d ", tmp_mb[row * mrow + col]);
    }
    printf("\t");
    for (col = 0; col < mcol; col++)
    {
      if (tmp_ans[0] == 0)
        printf("%-2d ", tmp_ans[row * mrow + col]);
      else
        printf("%-2d = %-2d * %-2d  ", tmp_ans[row * mrow + col], tmp_ma[row * mrow + col], tmp_mb[row * mrow + col]);
    }
    printf("\n");
  }
}

int main()
{
  printf("===== demo: matmul-intrinsic =====\n");
  /* init data */
  int32_t x[N] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int32_t y[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  int32_t z[N] = {0};

  uint8_t msize_m = 2;
  uint8_t msize_k = 2;
  long stride = 2 * sizeof(int32_t); // sizeof(int32_t) * 2;

  /* init matrix value*/
  mint32_t ma = __riscv_th_mld(x, stride, msize_m, msize_k);
  mint32_t mb = __riscv_th_mld(y, stride, msize_m, msize_k);
  mint32_t ans = __riscv_th_mld(z, stride, msize_m, msize_k);

  print_data("Initial value of matrix", ma, mb, ans, msize_m, msize_k);

  ans = __riscv_th_mmul_mm(ma, mb, msize_m, msize_k);

  print_data("Results of multiplication", ma, mb, ans, msize_m, msize_k);
  return 0;
}
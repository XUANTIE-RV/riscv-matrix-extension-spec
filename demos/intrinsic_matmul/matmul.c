#include <stdio.h>
#include <riscv_matrix.h>
#define N 16

void __attribute__((inline))
print_data(const char *fmt, mint32_t ma, mint32_t mb, mint32_t ans)
{
  unsigned int row, col;
  int32_t tmp_ma[N];
  int32_t tmp_mb[N];
  int32_t tmp_ans[N];

  printf("%s:\n", fmt);

  mst_i32_mi32(tmp_ma, 8, ma);
  mst_i32_mi32(tmp_mb, 8, mb);
  mst_i32_mi32(tmp_ans, 8, ans);

  printf("ma:\t\tmb:\t\tans:\n");
  for (row = 0; row < 2; row++)
  {
    for (col = 0; col < 2; col++)
    {
      printf("%-3d ", tmp_ma[row + col]);
    }
    printf("\t");
    for (col = 0; col < 2; col++)
    {
      printf("%-3d ", tmp_mb[row + col]);
    }
    printf("\t");
    for (col = 0; col < 2; col++)
    {
      if (tmp_ans[0] == 0)
        printf("%-2d ", tmp_ans[row + col]);
      else
        printf("%-2d = %-2d * %-2d  ", tmp_ans[row + col], tmp_ma[row + col], tmp_mb[row + col]);
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
  uint8_t msize_n = 2;
  uint16_t msize_k = 8; // sizeof(int32_t) * 2;
  long stride = 8;      // sizeof(int32_t) * 2;

  /* Configuration matrix size */
  mcfgm(msize_m);
  mcfgn(msize_n);
  mcfgk(msize_k);

  /* init matrix value*/
  mint32_t ma = mld_i32(x, stride);
  mint32_t mb = mld_i32(y, stride);
  mint32_t ans = mld_i32(z, stride);

  print_data("Initial value of matrix", ma, mb, ans);

  ans = mmul_mi32(ma, mb);
  print_data("Results of multiplication", ma, mb, ans);

  return 0;
}

NN2_DIR=install_nn2

CC = gcc/bin/riscv64-unknown-linux-gnu-gcc
CFLAGS += -march=rv64gcv_zfh_xtheadc_xtheadmatrix -mabi=lp64d -O2
INCLUDE += -I ${NN2_DIR}/include
LDFLAGS_RVM = -L ${NN2_DIR}/lib -lshl_rvm -lm -static -Wl,--no-warn-mismatch
LDFLAGS_RVV = -L ${NN2_DIR}/lib -lshl_rvv -lm -static -Wl,--no-warn-mismatch

all: gemm_int8 gemm_fp16 resnet50_int8 resnet50_fp16 intrinsic gemm_int8_rvv gemm_fp16_rvv resnet50_int8_rvv resnet50_fp16_rvv

gemm_int8:
	$(CC) $(CFLAGS) gemm/gemm_int8.c -o gemm/gemm_int8.elf ${INCLUDE} ${LDFLAGS_RVM}

gemm_fp16:
	$(CC) $(CFLAGS) gemm/gemm_fp16.c -o gemm/gemm_fp16.elf ${INCLUDE} ${LDFLAGS_RVM}

resnet50_int8:
	$(CC) $(CFLAGS) resnet50/resnet50_int8.c -o resnet50/resnet50_int8.elf ${INCLUDE} ${LDFLAGS_RVM}

resnet50_fp16:
	$(CC) $(CFLAGS) resnet50/resnet50_fp16.c -o resnet50/resnet50_fp16.elf ${INCLUDE} ${LDFLAGS_RVM}

intrinsic:
	$(CC) $(CFLAGS) intrinsic_matmul/matmul.c -o intrinsic_matmul/matmul.elf  -O2 -lm -static

gemm_int8_rvv:
	$(CC) $(CFLAGS) gemm/gemm_int8_rvv.c -o gemm/gemm_int8_rvv.elf ${INCLUDE} ${LDFLAGS_RVV}

gemm_fp16_rvv:
	$(CC) $(CFLAGS) gemm/gemm_fp16_rvv.c -o gemm/gemm_fp16_rvv.elf ${INCLUDE} ${LDFLAGS_RVV}

resnet50_int8_rvv:
	$(CC) $(CFLAGS) resnet50/resnet50_int8_rvv.c -o resnet50/resnet50_int8_rvv.elf ${INCLUDE} ${LDFLAGS_RVV}

resnet50_fp16_rvv:
	$(CC) $(CFLAGS) resnet50/resnet50_fp16_rvv.c -o resnet50/resnet50_fp16_rvv.elf ${INCLUDE} ${LDFLAGS_RVV}

clean:
	rm -rf cpfdata
	rm -rf log
	rm -rf */*.elf

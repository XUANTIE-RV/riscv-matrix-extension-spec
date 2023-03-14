#!/bin/bash

qemu=qemu/bin/qemu-riscv64
cpf64=qemu/bin/cpf64
if [ ! -d "perf_data" ];then
  mkdir perf_data
else
  echo ""
fi

run() {
    dir_name=$1
    module=$2

    echo ${module}" is running"
    ${qemu} -CPF -cpu rv64,x-v=true,vext_spec=v1.0,vlen=128,x-matrix=on,mlen=128 ./${dir_name}/${module}.elf | tee -a log & sleep 5
    ${cpf64} record -e ${dir_name}/${module}.elf
    ${cpf64} stat   -e ${dir_name}/${module}.elf --dump-inst >perf_data/${module}.log
    ${cpf64} report -g >/dev/null
    mv cpfdata/inst-info.txt perf_data/${module}-inst-info
    mv cpfdata/callgraph.json perf_data/${module}-callgraph.json
}

run gemm gemm_int8
run gemm gemm_fp16
run resnet50 resnet50_int8
run resnet50 resnet50_fp16
run intrinsic_matmul matmul
# run gemm gemm_int8_rvv
# run gemm gemm_fp16_rvv
# run resnet50 resnet50_int8_rvv
# run resnet50 resnet50_fp16_rvv

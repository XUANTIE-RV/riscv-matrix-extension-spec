#!/bin/bash
#set parameter

echo "--- make qemu ---"
mkdir qemu
tar -zxvf ./toolchain/xuantie-qemu-x86_64-Ubuntu-18.04.tar.gz -C qemu

echo "--- download compipler ---"
wget -O ./toolchain/Xuantie-gcc-linux-glibc-x86_64-matrix.tar.gz https://github.com/T-head-Semi/riscv-matrix-extension-spec/releases/download/v0.3.0/Xuantie-gcc-linux-glibc-x86_64-matrix.tar.gz

echo "--- make compiler ---"
mkdir gcc
tar -zxvf ./toolchain/Xuantie-gcc-linux-glibc-x86_64-matrix.tar.gz -C gcc

#make
echo "--- The environment is ready ---"

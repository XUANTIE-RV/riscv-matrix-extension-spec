#!/bin/bash
#set parameter

echo "--- make qemu ---"
mkdir qemu
tar -zxvf ./toolchain/xuantie-qemu-x86_64-Ubuntu-18.04.tar.gz -C qemu

echo "--- make compiler ---"
mkdir gcc
tar -zxvf ./toolchain/Xuantie-gcc-linux-glibc-x86_64-V2.6.1-matrix-v0.2-part1.tar.gz -C gcc
tar -zxvf ./toolchain/Xuantie-gcc-linux-glibc-x86_64-V2.6.1-matrix-v0.2-part2.tar.gz -C gcc

make
echo "--- The environment is ready ---"

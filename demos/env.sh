#!/bin/bash

#######set parameter

compiler_file="./toolchain/Xuantie-gcc-linux-glibc-x86_64-matrix.tar.gz"
compiler_md5="15e6fe82825664494310a1a671b73d6a"
compiler_address="https://github.com/T-head-Semi/riscv-matrix-extension-spec/releases/download/v0.3.0/Xuantie-gcc-linux-glibc-x86_64-matrix.tar.gz"

######make qemu
echo "--- Make qemu ---"
mkdir qemu
tar -zxvf ./toolchain/xuantie-qemu-x86_64-Ubuntu-18.04.tar.gz -C qemu


######make gcc
if [ -f "$compiler_file" ]; then
    if [ "$(md5sum $compiler_file | awk '{ print $1 }')" == "$compiler_md5" ]; then
        echo "--- Already have compiler tar ---"
    else
        echo "--- Wrong compiler tar ---"
        rm $compiler_file
    fi
fi

if [ ! -f "$compiler_file" ]; then
    echo "--- Download compipler ---"
    wget -O $compiler_file $compiler_address
    if [ $? != 0 ]; then
        echo "Download fail. Take a rest and retry it"
        echo "Or download compiler from github and put it in toolchain dir"
        echo "--- Fatal Error ---"
        exit 1
    fi
fi


echo "--- Make compiler ---"
mkdir gcc
tar -zxvf ./toolchain/Xuantie-gcc-linux-glibc-x86_64-matrix.tar.gz -C gcc

make
echo "--- The environment is ready ---"

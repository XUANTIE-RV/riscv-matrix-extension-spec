# RISC-V Matrix Extension Specification

## Introduction
This is a matrix extension proposal for AI applications under RISC-V architecture. The extension has the following features.

* Scalability
    - Register size can be scaled from 64 bytes to 2048 bytes 
    - Peak performance of the extension varies from 0.125 Tops/Ghz to 32 Tops/Ghz
    - Binary portability
* Multiple data types
    - Support int8/int16/fp16/fp32
* Independence
    - Strongly inspired by the RISC-V Vector extension
    - Decoupled architecture from Vector extension
* Extensibility for future
    - Support extensions for bf16/int4 and other future extensions

The extension is still under construction, and this is a preview demo project.
Some key directories are shown below.
```
|--spec                     ## The RISC-V Matrix Extension specification
|--doc/                     ## The user guide for tools  
    |--shl                  ## The SHL 2.0 user guide
    |--abi                  ## The Matrix Extension ABI Manual
    |--intrinsic            ## The Matrix Extension intrinsic API Reference Manual
|--shl/                     ## A neural networks library using RISC-V Matrix Extension
|--hhb/                     ## A toolkit used for deploying neural network models
|--xuantie-gnu-toolchain/   ## GNU toolchain
    |--riscv-gcc/           ## Compiler
    |--riscv-binutils-gdb/  ## Assembler
|--qemu/                    ## Emulator
|--demos/               
    |--resnet50             ## A resnet50 evaluation demo using nn library
    |--GEMM                 ## A GEMM evaluation demo using intrinsic
```

## Quick Start
### Prepare and clone repos.
```
git clone https://github.com/T-head-Semi/riscv-matrix-extension-spec.git
cd riscv-matrix-extension-spec
git submodule update --init
```
### Demo quick start
Several demos in binaries are provided to evaluate RISC-V Matrix Extension's performance. You can quickly start it with the following instructions.
```
cd demos/
./env.sh
make 
./run.sh
```
Please refer to the [demos/README](https://github.com/T-head-Semi/riscv-matrix-extension-spec/blob/master/demos/README.md) for details.

## Matrix Extension Documents
This project is built using AsciiDoctor (Ruby). The repository has been setup to build the PDF on checkin using GitHub actions. Workflow dependencies are located in the dependencies directory.

For more information on AsciiDoctor, specification guidelines, or building locally, see the
[RISC-V Documentation Developer Guide](https://github.com/riscv/docs-dev-guide).

RISC-V Matrix Extension Specification is kept in ./spec.
User guide and reference manual for RISC-V Matrix Extension tools are kept in ./doc.

We will also release our latest documentation in the Releases.

The final documents form of PDF can be generated using the `make` command under corresponding folder. The generation method of each document is as follows.

| Folder | Command     |     Documents |
| ----   | ----        | ----      |
| spec | make        | RISC-V Matrix Extension specification |
| doc  | make shl    | SHL 2.0 USER GUIDE |
| doc  | make its    | Matrix Extension intrinsic API Reference Manual |
| doc  | make abi    | Matrix Extension ABI Manual |


## Building and Installing RISC-V Matrix Extension project
Prepare the toolchain
```
cd xuantie-gnu-toolchain/
git submodule update --init
```
Compile and install qemu. Please refer to the [Xuantie qemu project](https://github.com/riscv/docs-dev-guide) for details.
```
mkdir qemu/build
cd qemu/build
../configure --target-list="riscv32-linux-user,riscv64-linux-user"
make
```

Get your own case and compile into matrix.elf. Both intrinsic and nn libraries can be used to perform this step.
Please refer to [T-HEAD GNU Compiler Toolchain](https://github.com/T-head-Semi/xuantie-gnu-toolchain) or [HHB](https://www.yuque.com/za4k4z/kvkcoh/sxltga) and [SHL](https://github.com/T-head-Semi/csi-nn2) for details.

Evaluation matrix performance on qemu with RISC-V Matrix Extension(with vector length set to VLEN and matrix length set to MLEN)
```
qemu-riscv64 -cpu rv64,x-v=true,vext_spec=v1.0,vlen=VLEN,x-matrix=on,mlen=MLEN ./matrix.elf
```


 

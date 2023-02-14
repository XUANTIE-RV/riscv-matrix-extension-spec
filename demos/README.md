# RISC-V Matrix Extension Demo Description

We provide complete C code models, gemm and matmul-intrinsic demos to make it easy for you to get started with RISC-V Matrix Extension. You can quickly start running the demos after downloading the project from git by the steps below.

```
cd demos/
./env.sh
./run.sh
```

If you want to learn how to deploy a neural network model using RISC-V Matrix Extension, you can refer to: [RISC-V matrix extension example](https://csi-nn2.opensource.alibaba.com/blog/RVM%20example)

All demos are tested on Ubuntu 18.04.5.

## demo list

| Case Name |     Description |
| ----        | ----      |
| gemm_int8           | General Matrix Multiply implemented by convolution layer (data type 8 bit integer) |
| gemm_fp16           | General Matrix Multiply implemented by convolution layer (data type 8 bit integer) |
| resnet50_int8       | ResNet demo implemented using NN library (data type 8 bit integer) |
| resnet50_fp16       | ResNet demo implemented using NN library (data type 16 bit floating-point) |
| matmul              | Matrix multiplication demo implemented using intrinsic |

Case name can be used to make and run a specified demo.

## evaluation results

We use qemu and cpf to count the number of instructions of the program. Compared with vector extension 1.0, RISC-V Matrix Extension has an improvement of 2.82x - 5.14x on resnet50, speed up 3.81x - 8.93x on gemm (160 x 160 x 160)

| Case Name | vector instruction | Matrix Extension instruction | speed up |
| ----                | ----      | ---       | ---      |
| gemm_int8           | 291851    | 76583     | 3.81     |
| gemm_fp16           | 917560    | 102777    | 8.93     |
| resnet50_int8       | 316367132 | 113795401 | 2.82     |
| resnet50_fp16       | 633397401 | 125276864 | 5.14     |

The complete instruction distribution and function hotspots will be generated in perf_data.
```
case-name.log records demo info statistics.
case-name-inst-info records detailed instruction statistics.
case-name-callgraph.json records program function calling relationship and instructions statistics.
```

During the running of the model, the complete information of each layer of the model will be printed, including feature map size, kernel size, convolution/pooling stride, padding, theoretical calculation amount and measured calculation power. The data in the table below is the running time of qemu on the server. If executed in the actual hardware environment, the time-consuming is accurate

```
[  0]: conv2d_relu       946.26ms  ^*^:[   1, 224, 224,   3] ==> [   1, 112, 112,  64] | k: 7x7 | s: 2x2 | p: 3 3 3 3 |  MOPS:236.03 ( 0.2494GOPS)
[  1]: maxpool2d          31.78ms  ^*^:[   1, 112, 112,  64] ==> [   1,  56,  56,  64] | k: 3x3 | s: 2x2 | p: 0 0 0 0 |
[  2]: conv2d            314.25ms  ^*^:[   1,  56,  56,  64] ==> [   1,  56,  56, 256] | k: 1x1 | s: 1x1 | p: 0 0 0 0 |  MOPS:102.76 ( 0.3270GOPS)
[  3]: conv2d_relu        78.78ms  ^*^:[   1,  56,  56,  64] ==> [   1,  56,  56,  64] | k: 1x1 | s: 1x1 | p: 0 0 0 0 |  MOPS: 25.69 ( 0.3261GOPS)
[  4]: conv2d_relu       719.58ms  ^*^:[   1,  56,  56,  64] ==> [   1,  56,  56,  64] | k: 3x3 | s: 1x1 | p: 1 1 1 1 |  MOPS:231.21 ( 0.3213GOPS)
[  5]: conv2d            313.04ms  ^*^:[   1,  56,  56,  64] ==> [   1,  56,  56, 256] | k: 1x1 | s: 1x1 | p: 0 0 0 0 |  MOPS:102.76 ( 0.3283GOPS)
[  6]: add                85.66ms  ^*^:[   1,  56,  56, 256] ==> [   1,  56,  56, 256]
[  7]: relu               69.14ms  ^*^:[   1,  56,  56, 256] ==> [   1,  56,  56, 256]
...
...
[ 88]: fullyconnected     36.28ms  ^*^:[   1,2048] ==> [   1,1000] MOPS:  4.10 ( 0.1129GOPS)
[ 89]: softmax             3.77ms  ^*^:[   1,1000] ==> [   1,1000]
[layer-benchmark]: network exec time = 23572.835938
```

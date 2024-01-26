记录一下看源码的笔记

## pretrain_gpt.py 入口文件，多个进程同时执行，这里以其中1个进程为例

### 调用`megatron/training.py`中的`pretrain`函数

#### 调用`megatron/initialize.py`中的`initialize_megatron`函数

##### 调用`megatron/arguments.py`中的`parse_args`参数解析。

##### 调用`megatron/arguments.py`中的`validate_args`函数：检查并计算3D并行参数、global_batch_size、参数fp16、bf16的设置，检查其它参数的合理性

##### 调用`megatron/global_vars.py`中的`set_global_variables`函数，

1. 将args Namespace 赋值给全局变量 _GLOBAL_ARGS

2. 调用`_build_num_microbatches_calculator`函数，具体是位于`megatron/microbatches.py`中`ConstantNumMicroBatches`类，计算 num_micro_batches

3. 调用`_build_tokenizer`函数，具体是位于`megatron/tokenizer/tokenizer.py`中的`build_tokenizer`函数，加载tokenizer

4. 剩下的逻辑为初始化`tensorboard`、`timers`

##### 调用`megatron/initialize.py`文件的`finish_mpu_init`函数。

1. 调用`megatron/global_vars.py`中的`get_args`函数，返回`_GLOBAL_ARGS`。

2. 调用`megatron/initialize.py`中的`_initialize_distributed`函数，该函数主要是初始化了各个并行进程组。是非常关键的函数。

（1）调用`/mnt/private_zhaodali_cq/llm_install/DeepSpeed/deepspeed/accelerator/real_accelerator.py`中的`get_accelerator`函数。

1）首先通过`DS_ACCELERATOR`环境变量得到当前环境为`cuda`。

2）然后调用`/mnt/private_zhaodali_cq/llm_install/DeepSpeed/deepspeed/accelerator/cuda_accelerator.py`中的`CUDA_Accelerator._init_pynvml`函数。NVML库提供了一种直接和显卡交互的方式，可以用来获取显卡的状态，如温度、风扇速度、电压、显存使用情况等。另外，注意这里设置了通信方式为`nccl`。

> 其实这个步骤在最开始调用deepspeed就运行了，但是这里为了了解代码整体逻辑，所以不计较那么多了。

3）调用`deepspeed.init_distributed`函数，该函数位于`/mnt/private_zhaodali_cq/llm_install/DeepSpeed/deepspeed/comm/comm.py`中。

- 调用同文件的`init_deepspeed_backend`函数，该函数只能初始化`ccl`的通信方式，其余的比如说nccl的通信方式都只是简单的打印日志说不支持。

- 调用同文件的`set_backend`函数，对我的场景，该函数没有用途，`cdb`仍然为`None`

- 最终调用的是`TorchBackend`初始化`cdb`变量，该函数位于`/mnt/private_zhaodali_cq/llm_install/DeepSpeed/deepspeed/comm/torch.py`文件中。

-- 根据torch的版本以及模块信息判断是否存在一些功能

-- 调用`init_process_group`函数，该函数内部调用了`torch.distributed.init_process_group`函数，设置了通信方式为`nccl`。

> torch.distributed.init_process_group是PyTorch分布式包中的一个函数，它用于初始化分布式环境。这个函数主要用于设置分布式训练中的通信后端（backend）和进程组（process group）。   在分布式训练中，需要在多个机器或者同一台机器的多个进程之间进行通信，比如同步模型参数或者梯度等。PyTorch提供了几种通信后端，如"Gloo", "NCCL", "MPI"等，不同的后端有不同的特性和优势，可以根据实际需求选择。    进程组是一组进程的集合，可以进行集体通信操作。init_process_group函数可以设置进程组的大小，以及每个进程在进程组中的排名（rank）。    此外，init_process_group函数还可以设置其他一些参数，如超时时间，启动方法等。    总的来说，torch.distributed.init_process_group函数是进行PyTorch分布式训练的关键步骤，它负责初始化分布式环境，设置通信后端和进程组，以便进行后续的分布式训练。

4）调用`mpu.initialize_model_parallel`初始化进程组。该函数位于`megatron/core/parallel_state.py`文件的`initialize_model_parallel`函数。这个函数比较复杂且关键，建议直接看该函数官方注释中的例子。注意该函数也执行了`_set_global_memory_buffer`函数申请全局缓存。

```
来自 https://www.cnblogs.com/rossiXYZ/p/15868988.html

因为调用了 mpu.initialize_model_parallel 来设置模型并行，数据并行等各种进程组，所以我们假定目前进程组都已经设置成功，所以每个 rank 对应的进程都有自己的全局变量。假定目前有16个GPU，属于两个node，rank 0 ～7 属于第一个节点，rank 8 ～ 15 属于第二个节点。下面的 gi 指的是第 i 个 GPU。

_TENSOR_MODEL_PARALLEL_GROUP ：当前 rank 所属于的Intra-layer model parallel group，就是tensor 并行进程组。
假如每一层分为两个tensor，则 _TENSOR_MODEL_PARALLEL_GROUP 例子为：[g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]。
_PIPELINE_MODEL_PARALLEL_GROUP ：当前 rank 所属于的Intra-layer model parallel group，就是流水线进程组。
假如流水线深度为4，则例子为 [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]。
_MODEL_PARALLEL_GROUP ：当前 rank 所属于的模型并行进程组，包括了以上两组。
针对我们例子，就是完整模型被复制了两份，两份分别对应的 GPU 具体是[0, 1, 4, 5, 8, 9, 12, 13]，[2, 3, 6, 7, 10, 11, 14, 15]
_EMBEDDING_GROUP ： 嵌入对应的进程组。
_DATA_PARALLEL_GROUP ：当前 rank 所属于的Data parallel group。
假如数据并行度数为2，则例子为[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]。

上面这几个全局变量都是`torch.distributed.distributed_c10d.ProcessGroup`类型，因为均是通过`torch.distributed.new_group`创建的。torch.distributed.new_group函数在PyTorch分布式包中用于创建一个新的进程组。在分布式训练中，进程组是一组可以进行集体通信操作的进程的集合
```
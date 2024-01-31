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

上面这几个全局变量都是`torch.distributed.distributed_c10d.ProcessGroup`类型，因为均是通过`torch.distributed.new_group`创建的。torch.distributed.new_group函数在PyTorch分布式包中用于创建一个新的进程组。在分布式训练中，进程组是一组可以进行集体通信操作的进程的集合。可以通过`torch.distributed.get_process_group_ranks(_TENSOR_MODEL_PARALLEL_GROUP)`得到对应的进程组。
```

3. 调用`megatron/initialize.py`中的`_set_random_seed`函数，设置随机初始化

##### 调用`megatron/initialize.py`文件的`_compile_dependencies`函数。

1. 调用`megatron/data/dataset_utils.py`中的`compile_helper`函数。执行`make -C megatron/data`，编译该文件夹下面的`Makefile`文件。

2. 调用`megatron/fused_kernels/__init__.py`中的`load`函数，注意这里保证了使用先编译rank 0。另外，这里编译的结果应该放在/tmp目录，

#### 调用`megatron/initialize.py`中的`set_jit_fusion_options`函数，设置一些jit fusion选项

##### 调用`megatron/training.py`中的`_create_ds_config_dict`函数。加载类似于`examples_deepspeed/rebase/ds_config_gbs256_mbs2_log10_zero1.json`文件，赋值给ds_config_dict。

##### 调用`megatron/training.py`中的`setup_model_and_optimizer`函数。

1. 调用`megatron/training.py`中的`get_model`函数。

（1）根据rank的编号得到是否是pipeline的第一个或者最后一个stage。

（2）调用`pretrain_gpt.py`中的`model_provider`函数。

1）调用`megatron/arguments.py`中的`core_transformer_config_from_args`函数，从args中读取transformer的核心参数，比如说`num_layers`等。

2）调用`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py`中的`Init`类，该类很复杂，等待后续分析。

3）调用`megatron/model/gpt_model.py`中的`GPTModelPipe`类。

- 添加_to_float16函数、EmbeddingPipe层、ParallelTransformerLayerPipe层、LayerNorm层、EmbeddingPipe层、float16_to_fp32函数

- 调用`PipeModelDataParallelTopology`类，该类供了一种方式来描述这种并行结构，包括哪些层在哪些设备上，以及如何在设备之间传递数据。这可以帮助DeepSpeed更有效地管理资源和调度计算任务。该类很复杂，等待后续分析。

- 定义`checkpoint_activations`方法中的`interval`属性。

- 定义损失函数`CrossEntropy`。

4）将`pretrain_gpt.py`中的`get_batch_pipe`函数赋给`model._megatron_batch_fn`。

5）创建上三角矩阵(对角线为False)，维度为(1, 1, seq_length, seq_length)，将其赋值给`args.attn_mask`。

6）返回model

（3）将model放到list中，重新赋值给model

（4）对于model中的每一个参数，均设置 tensor model parallel 属性。注意这里的属性均设置了默认值

（5）如果使用了deepspeed，返回model。注意他是一个list

2. 调用`megatron/utils.py`中的`unwrap_model`函数。若model是`(torchDDP, LocalDDP, Float16Module)`中的任何一个属性，则取` model_module.module`。对于我们调试的场景，该函数你没有任何作用。

3. 调用`megatron/optimizer/__init__.py`中的`get_megatron_optimizer`函数。

（1）调用`megatron/optimizer/__init__.py`中的`get_param_groups`函数。创建`param groups`，为模型中的不同参数，创建不同的优化参数。

（2）调用`Adam`创建优化器。

（3）如果使用了deepspeed，返回optimizer。

4. 调用`megatron/training.py`中的`get_optimizer_param_scheduler`函数，创建learning rate scheduler。

（1）调用`megatron/training.py`中的`update_train_iters`函数，使用`args.train_samples // args.global_batch_size`将`train_samples`转为`train_iters`。

（2）将`train_samples`赋值给`lr_decay_samples`。

（3）实例化`megatron/optimizer_param_scheduler.py`中的`OptimizerParamScheduler`类，创建Anneals learning rate and weight decay，赋值给opt_param_scheduler。

（4）返回opt_param_scheduler。

5. 调用`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/__init__.py`中的`initialize`方法，初始化DeepSpeed引擎。传参为model、optimizer、opt_param_scheduler、args、deepspeed配置。出参为`model, optimizer, _, opt_param_scheduler`。该类很复杂，等待后续分析。

6. 调用`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/pipe/engine.py`中的`PipelineEngine`类的`set_batch_fn`函数，将`get_batch_pipe`函数赋值给`PipelineEngine`类的`batch_fn`属性。

7. 将model放到list中，重新赋值给model

8. 调用`megatron/checkpointing.py`中的`load_checkpoint`函数。

（1）从args中得到`load`的值，得到权重保存的目录。

（2）调用`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/engine.py`中的`load_checkpoint`函数。

1）尝试读取`latest`文件，该文件保存了最新权重的step数。

2）我这里因为没有权重，所以先略过，等待后续分析。

9. 返回`model, optimizer, opt_param_scheduler`。

##### 调用`megatron/training.py`中的`build_train_valid_test_data_iterators`函数。返回`train_data_iterator, valid_data_iterator, test_data_iterator`。

1. 调用`megatron/training.py`中的`build_train_valid_test_data_loaders`函数。

（1）判断当前进程在数据进程组中的rank，对于每个model parallel group，只有rank=0的才会加载data loader。等待后续分析。

（2）调用`megatron/training.py`中的`build_train_valid_test_datasets`函数。

1）根据迭代次数与`global_batch_size`，计算 整个训练过程中 的训练、验证、测试集的样本数(一条样本应该是seq_length token)，赋值给变量`train_val_test_num_samples`。

2）回调函数`build_train_valid_test_datasets_provider`，该函数位于`pretrain_gpt.py`的`train_valid_test_datasets_provider`方法。

2. 解析一下位于`pretrain_gpt.py`的`train_valid_test_datasets_provider`方法。

（1）调用`megatron/data/gpt_dataset.py`中的`build_train_valid_test_datasets`方法。该方法进一步调用了`megatron/data/gpt_dataset.py`中的`_build_train_valid_test_datasets`函数。

1）调用`megatron/data/gpt_dataset.py`中的`get_indexed_dataset_`函数。该函数进一步调用了`make_indexed_dataset`函数，即`megatron/data/indexed_dataset.py`中的`make_dataset`函数。

- 该函数内部初始化了`megatron/data/indexed_dataset.py`文件中的`MMapIndexedDataset`类。该类初始化函数主要加载了`examples_deepspeed/rebase/data/oscar-en-10k_text_document.idx`文件。并从中解析出一些信息，包括文件的版本号、数据类型、长度、文档数量等。

- 返回`indexed_dataset`，在该场景中为`megatron.data.indexed_dataset.MMapIndexedDataset`类型，

2）调用`megatron/data/dataset_utils.py`中的`get_train_valid_test_split_`函数。将数据按照`splits_string`进行划分，划分的整体为`total_num_of_documents`。当`total_num_of_documents=10000,splits=[0.949,0.05,0.001]`，返回值为`splits_index=[0, 9490,9990,10000]`。所以这个返回是一个累加结果。并且看来数据并没有打乱。

3）对于train、valid、test，分别调用子函数`build_dataset`。这里以train为例进行分析。由`np.arange`生成对应的document索引。然后初始化了`megatron/data/gpt_dataset.py`中的`GPTDataset`类。该类继承自`torch.utils.data.Dataset`，是数据读取的关键类。

- 该类调用了`megatron/data/gpt_dataset.py`中的`_build_index_mappings`函数。

-- 计算得到`tokens_per_epoch`以及`num_epochs`。由`num_epochs`约等于`num_samples*seq_length/(tokens_per_epoch)`，可以得知一个sample应该是一个seq_length token的样本。在数据预处理时，若token数不足seq_length，则使用pad token补足，若超过了，则可以截断。

-- 在当前数据路径下创建`index-cache`文件夹，用于存放缓存文件。

-- 调用`_build_doc_idx`函数，该函数即`megatron/data/gpt_dataset.py`中的`_build_doc_idx`函数。该函数的主要用途是将train对应的documents(document的下标)复制num_epochs遍，然后随机打乱，返回`doc_idx`。本场景中shape为202004140

-- 将`doc_idx`存储到`index-cache`文件夹中。

-- 调用CPP代码，得到每个sample在document的二维索引，尺寸为`(292978031, 2)`，比如说
```python
array([[   0,    0],
       [   0, 2048],
       [   2,  532],
       [   2, 2580],
       ...,
       [202004139,       353]])
```

-- 将`sample_idx`存储到`index-cache`文件夹中。

-- 根据是否分离最后一个epoch，得到`num_samples_`的值。

-- 调用`megatron/data/gpt_dataset.py`中的`_build_shuffle_idx`函数。得到shape为total_size(292978030)的随机序列`shuffle_idx`，范围为[0,292978029]

-- 返回`doc_idx, sample_idx, shuffle_idx, desc, desc_hash`

4）对于train、vaild、test调用`build_dataset`完毕后，返回`train_dataset, valid_dataset, test_dataset`

（2）返回`train_ds, valid_ds, test_ds`。

3. 将`train_ds, valid_ds, test_ds`分别传递给`build_pretraining_data_loader`创建dataloader。该函数位于`megatron/data/data_samplers.py`中。

（1）该函数首先通过`MegatronPretrainingSampler`创建了`batch_sampler`。

（2）将`batch_sampler`传递给`torch.utils.data.DataLoader`创建dataloader。等待后续分析

4. 返回创建好的`train_dataloader, valid_dataloader, test_dataloader`。

##### 调用`megatron/training.py`中的`train`函数。

1. `model`变量是list，对于其中的每个值，执行它的`train`函数，在我们场景下为`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/engine.py`中`PipelineEngine`类的`train`函数。将模型设置为`train`模式。

2. 调用`megatron/arguments.py`中的`core_transformer_config_from_args`函数，从args中读取transformer的核心参数，比如说`num_layers`等。

3. 进入训练循环，循环条件为`iteration < args.train_iters and (args.train_tokens is None or args.consumed_train_tokens < args.train_tokens)`。

（1）由`mpu.get_data_parallel_world_size()*args.micro_batch_size*num_microbatches`计算得到`global_batch_size`。

（2）调用`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/engine.py`中的`set_train_batch_size`函数，设置`train_batch_size`以及`gradient_accumulation_steps`，后者的值等于`train_batch_size // (self.train_micro_batch_size_per_gpu() * self.dp_world_size)`。

（3）调用`megatron/training.py`中的`train_step`方法。

1）调用`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/pipe/engine.py`中的`PipelineEngine`类的`train_batch`方法。

- 初始化`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/pipe/schedule.py`中的`TrainSchedule`类，赋值给`sched`。

- 执行`self._exec_schedule(sched)`，该函数位于`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/pipe/engine.py`中`PipelineEngine`类的`_exec_schedule`函数。注意该函数会重复以下过程，`pipe_schedule`中总共有256组`step_cmds`。如下所示，这里应该是为了实现DP的前向传播。

```python
[LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)]
[BackwardPass(buffer_id=0)]
[LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)]
[BackwardPass(buffer_id=1)]
...
[LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)]
[BackwardPass(buffer_id=0)]
[LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)]
[BackwardPass(buffer_id=1), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
```

-- 该函数，首先依次执行`LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)`，加载MicroBatch，并前向传播。

--- `ForwardPass`实际调用的是`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/pipe/engine.py`的`PipelineEngine`类的`_exec_forward_pass`函数。

---- 调用`self.module.loss_fn`函数计算损失，该函数实际位于`megatron/model/gpt_model.py`中的`CrossEntropy`函数。

---- 累加`self.total_loss`。

-- 该函数，执行`BackwardPass(buffer_id=0)`。反向传播。

--- `BackwardPass`实际调用的是`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/pipe/engine.py`的`PipelineEngine`类的`_exec_backward_pass`函数。

- 执行`self._aggregate_total_loss`函数。

2）调用`/data6/zhaodali/llm_install/DeepSpeed/deepspeed/runtime/engine.py`中的`get_global_grad_norm`方法，得到`grad_norm`。

（4）更新`args.consumed_train_tokens`的值。

（5）更新`training_log`。

（6）判断是否需要执行valid，若需要则执行。

（7）判断是否要保存checkpoint，若需要保存，则执行`megatron/training.py`中的`save_checkpoint_and_time`函数。

1）调用`megatron/checkpointing.py`中的`save_checkpoint`函数，这里应该会继续调用deepspeed的`save_checkpoint`函数。

2）调用`megatron/utils.py`中的`checkpoint_throughput_calculator`函数，计算checkpoint存储的吞吐量。

4. 返回`iteration`。

##### 调用`megatron/checkpointing.py`中的`save_checkpoint`函数，这里应该会继续调用deepspeed的`save_checkpoint`函数。

## 附录知识

### torch.distributed.barrier()

来自于[通俗理解torch.distributed.barrier()工作原理](https://blog.csdn.net/weixin_41041772/article/details/109820870)

在pytorch的多卡训练中，通常有两种方式，一种是单机多卡模式（存在一个节点，通过torch.nn.DataParallel(model)实现），一种是多机多卡模式（存在一个节点或者多个节点，通过torch.nn.parallel.DistributedDataParallel(model)，在单机多卡环境下使用第二种分布式训练模式具有更快的速度。

pytorch在分布式训练过程中，对于数据的读取是采用主进程预读取并缓存，然后其它进程从缓存中读取，不同进程之间的数据同步具体通过torch.distributed.barrier()实现。

看下面这段代码：

```python
def create_dataloader():
    #使用上下文管理器中实现的barrier函数确保分布式中的主进程首先处理数据，然后其它进程直接从缓存中读取
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels()
 
from contextlib import contextmanager
 
#定义的用于同步不同进程对数据读取的上下文管理器
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield   #中断后执行上下文代码（即读取数据并处理），然后返回到此处继续往下执行
    if local_rank == 0:
        torch.distributed.barrier()
```

（1）进程号rank理解
在多进程上下文中，我们通常假定rank 0是第一个进程或者主进程，其它进程分别具有0，1，2不同rank号，这样总共具有4个进程。

（2）单一进程数据处理
通常有一些操作是没有必要以并行的方式进行处理的，如数据读取与处理操作，只需要一个进程进行处理并缓存，然后与其它进程共享缓存处理数据，但是由于不同进程是同步执行的，单一进程处理数据必然会导致进程之间出现不同步的现象，为此，torch中采用了barrier()函数对其它非主进程进行阻塞，来达到同步的目的。

（3）barrier()具体原理
在上面的代码示例中，如果执行create_dataloader()函数的进程不是主进程，即rank不等于0或者-1，上下文管理器会执行相应的torch.distributed.barrier()，设置一个阻塞栅栏，让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；如果执行create_dataloader()函数的进程是主进程，其会直接去读取数据并处理，然后其处理结束之后会接着遇到torch.distributed.barrier()，此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。

## vLLM 异构推理测试流程

## 环境搭建

### 从源代码构建 vLLM Docker 镜像

[GPU - vLLM 文档](https://docs.vllm.com.cn/en/latest/getting_started/installation/gpu.html#build-an-image-with-vllm)

### vllm-het

**运行vllm-het**

**执行构建脚本**

```bash
./build.sh
```

vllm-het应该在Ray集群的创建前执行，一旦对vllm-het进行修改，则应停止当前Ray集群，并重新创建。



**拉取指定版本的vLLM（可选）**

在环境变量指定vLLM版本，带参数执行构建脚本。

```bash
export VLLM_VERSION_PROVIDER=nvidia_0.9.1 (nvidia_0.9.1, amd_0.9.1)
./build.sh start
```



## 启动推理框架

### 构建ray集群

**主节点**

```bash
ray start --head --port=端口号 --num-cpus=n --num-gpus=n #指定使用的cpu和GPU数量
```

**其余节点**

```bash
ray start --address=主节点ip:端口号 --node-ip-address=当前从节点ip --num-cpus=n --num-gpus=n
```

Ray集群创建要求，所有节点的python版本和ray版本保持一致。



### vLLM 启动命令

```bash
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server   \
        --model /root/.cache/huggingface/modelscope/hub/models/Qwen/Qwen2-7b   \ #替换为具体的模型路径（各节点模型路径保持一致）
        --port 8000   \
        --dtype half   \
        --max_model_len 4096   \
        --pipeline_parallel_size 2   \     # 设置pp并行
        --distributed_executor_backend ray   \
        --api-key mysecretkey123 \
        --enforce-eager
```




## 执行测试程序

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mysecretkey123" \
  -d '{
    "model": "/root/.cache/huggingface/modelscope/hub/models/Qwen/Qwen2-7b",
    "messages":[
      {"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":"介绍大模型推理流程？"}
    ],
    "temperature":0.7,
    "top_p":0.8,
    "max_tokens":200
  }'
```

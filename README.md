# vllm-het

Support inference with heterogeneous GPUs using VLLM

## Init submodule on demand

prepare version first

```
export VLLM_VERSION_PROVIDER=nvidia_0.9.1 (nvidia_0.9.1, amd_0.9.1)
```

then init

```
bash build.sh
```

## Note

This system should be executed before the creation of the ray cluster. When changing system, the ray cluster needs to be stopped, and create again.

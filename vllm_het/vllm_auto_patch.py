import importlib
import functools
import logging
import torch
import torch.distributed
from typing import Any, Callable, Optional, Union
from torch.distributed import Backend, ProcessGroup
from types import ModuleType
import sys, time, traceback, importlib.util, importlib.abc
# from .p2p_backend import init_p2p_comm, TCPComm, _HET_COMM

logger = logging.getLogger("sitecustomize")
logger.info("sitecustomize loaded, preparing vllm patch")

LOG_PATH = "/tmp/vllm_auto_patch.log"

TARGET = "vllm.distributed.parallel_state"

def log(msg):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass

def apply_patch(mod: ModuleType):
    try:
        GC = getattr(mod, "GroupCoordinator", None)
        if GC is None:
            log("parallel_state.GroupCoordinator not found")
            return
        orig_init = getattr(GC, "__init__", None)
        if getattr(orig_init, "_patched_by_vllm_auto", False):
            log("GroupCoordinator.__init__ already patched; skip")
            return

        # ======= 示例：仅演示如何替换；把你的 vllm_het_init/send/recv 放进来 =======
        orig_init = getattr(GC, "__init__", None)
        orig_send_tensor_dict = getattr(GC, "send_tensor_dict", None)
        orig_recv_tensor_dict = getattr(GC, "recv_tensor_dict", None)

        # p2p = importlib.import_module("vllm.distributed.p2p_backend")

        def vllm_het_init(
            self,
            group_ranks: list[list[int]],
            local_rank: int,
            torch_distributed_backend: Union[str, Backend],
            use_device_communicator: bool,
            use_message_queue_broadcaster: bool = False,
            group_name: Optional[str] = None,
        ):
            orig_init(
                self,
                group_ranks,
                local_rank,
                torch_distributed_backend,
                use_device_communicator,
                use_message_queue_broadcaster,
                group_name,
            )
            self.use_hmc = True
            # for ranks in group_ranks:
            #     if self.rank in ranks:
            #         init_p2p_comm(self.cpu_group, p2p_backend=default_backend)

        def vllm_het_send(
            self,
            tensor_dict: dict[str, Union[torch.Tensor, Any]],
            dst: Optional[int] = None,
            all_gather_group: Optional["GroupCoordinator"] = None,
        ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
            """Send the input tensor dictionary.
            NOTE: `dst` is the local rank of the source rank.
            """
            print("\t\ncustomize send\t\n")
            # Bypass the function if we are using only 1 GPU.
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return tensor_dict

            all_gather_size = (1 if all_gather_group is None else
                            all_gather_group.world_size)
            all_gather_rank = (0 if all_gather_group is None else
                            all_gather_group.rank_in_group)

            group = self.device_group
            metadata_group = self.cpu_group

            if dst is None:
                dst = (self.rank_in_group + 1) % self.world_size
            assert dst < self.world_size, f"Invalid dst rank ({dst})"

            metadata_list: list[tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict,
                dict), f"Expecting a dictionary, got {type(tensor_dict)}"
            metadata_list, tensor_list = mod._split_tensor_dict(tensor_dict)
            self.send_object(metadata_list, dst=dst)
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip sending empty tensors.
                    continue

                # send-allgather: send only a slice, then do allgather.
                if (all_gather_group is not None
                        and tensor.numel() % all_gather_size == 0):
                    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

                if self.use_hmc:
                    if tensor.is_cpu:
                        # self.clients[self.ranks[dst]].send_tensor(tensor)
                        torch.distributed.send(tensor,
                                            dst=self.ranks[dst],
                                            group=metadata_group)
                    else:
                        cpu_tensor = tensor.cpu()
                        torch.distributed.send(cpu_tensor,
                                            dst=self.ranks[dst],
                                            group=metadata_group)
                else:
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        torch.distributed.send(tensor,
                                            dst=self.ranks[dst],
                                            group=metadata_group)
                    else:
                        # use group for GPU tensors
                        torch.distributed.send(tensor,
                                            dst=self.ranks[dst],
                                            group=group)
            return None
        
        def vllm_het_recv(
            self,
            src: Optional[int] = None,
            all_gather_group: Optional["GroupCoordinator"] = None,
        ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
            
            print("\t\ncustomize recv\t\n")
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return None

            all_gather_size = (1 if all_gather_group is None else
                            all_gather_group.world_size)
            all_gather_rank = (0 if all_gather_group is None else
                            all_gather_group.rank_in_group)

            group = self.device_group
            metadata_group = self.cpu_group

            if src is None:
                src = (self.rank_in_group - 1) % self.world_size
            assert src < self.world_size, f"Invalid src rank ({src})"

            recv_metadata_list = self.recv_object(src=src)
            tensor_dict: dict[str, Any] = {}
            for key, value in recv_metadata_list:
                if isinstance(value, mod.TensorMetadata):
                    tensor = torch.empty(value.size,
                                        dtype=value.dtype,
                                        device=value.device)
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue

                    # send-allgather: send only a slice, then do allgather.
                    use_all_gather = (all_gather_group is not None
                                    and tensor.numel() % all_gather_size == 0)

                    if use_all_gather:
                        orig_shape = tensor.shape
                        tensor = tensor.reshape(all_gather_size,
                                                -1)[all_gather_rank]
                    
                    if self.use_hmc:
                        if tensor.is_cpu:
                            torch.distributed.recv(tensor,
                                                src=self.ranks[src],
                                                group=metadata_group)
                        else:
                            cpu_tensor = torch.empty_like(tensor, device='cpu')
                            torch.distributed.recv(cpu_tensor,
                                                src=self.ranks[src],
                                                group=metadata_group)
                            device = torch.device(f'cuda:{self.local_rank}')
                            tensor = cpu_tensor.to(device)
                    else:
                        if tensor.is_cpu:
                            # use metadata_group for CPU tensors
                            torch.distributed.recv(tensor,
                                                src=self.ranks[src],
                                                group=metadata_group)
                        else:
                            # use group for GPU tensors
                            torch.distributed.recv(tensor,
                                                src=self.ranks[src],
                                                group=group)
                    if use_all_gather:
                        # do the allgather
                        tensor = all_gather_group.all_gather(  # type: ignore
                            tensor, dim=0)
                        tensor = tensor.reshape(orig_shape)

                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            return tensor_dict
        
        
        GC.__init__ = vllm_het_init
        GC.send_tensor_dict = vllm_het_send
        GC.recv_tensor_dict = vllm_het_recv

        log("patch applied OK: " + str(getattr(mod, "__file__", None)))
    except Exception:
        log("apply_patch failed:\n" + traceback.format_exc())



if TARGET in sys.modules:
    apply_patch(sys.modules[TARGET])

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None

        # 使用 PathFinder 做路径查找，避免再次走 sys.meta_path（防止递归）
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception:
            log("PathFinder.find_spec raised:\n" + traceback.format_exc())
            return None

        if spec is None or spec.loader is None:
            return None

        orig_loader = spec.loader

        # 如果已经被我们或其它逻辑包了一次，就别包两次
        if getattr(orig_loader, "_wrapped_by_vllm_auto", False):
            return spec

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                if hasattr(orig_loader, "create_module"):
                    return orig_loader.create_module(spec)
                return None

            def exec_module(self, module):
                # 先让原始 loader 加载模块
                orig_loader.exec_module(module)
                # 然后在模块加载完成后应用补丁
                try:
                    apply_patch(module)
                except Exception:
                    log("apply_patch failed in Loader.exec_module:\n" + traceback.format_exc())

        # 标记以避免重复包装（对 orig_loader 做标记）
        try:
            setattr(orig_loader, "_wrapped_by_vllm_auto", True)
        except Exception:
            pass

        spec.loader = _Loader()
        log("install loader wrapper for " + TARGET)
        return spec

if not any(isinstance(x, _Finder) for x in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
    log("vllm_auto_patch: meta_path finder installed")
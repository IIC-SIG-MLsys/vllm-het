import importlib
import functools
import logging
import torch
import torch.distributed
from typing import Any, Callable, Optional, Union
from torch.distributed import Backend, ProcessGroup

logger = logging.getLogger("sitecustomize")
logger.info("sitecustomize loaded, preparing vllm patch")

def patch_parallel_state():
    try:
        mod = importlib.import_module("vllm.distributed.parallel_state")
    except Exception as e:
        logger.exception("直接导入 vllm.distributed.parallel_state 失败，跳过 patch：%s", e)
        return

    orig_init = mod.GroupCoordinator.__init__
    orig_send_tensor_dict = mod.GroupCoordinator.send_tensor_dict
    orig_recv_tensor_dict = mod.GroupCoordinator.recv_tensor_dict

    p2p = importlib.import_module("vllm.distributed.p2p_backend")

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

            if self.use_uccl_p2p:
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
                
                if self.use_uccl_p2p:
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
    
    

    mod.GroupCoordinator.__init__ = functools.wraps(orig_init)(vllm_het_init)
    mod.GroupCoordinator.send_tensor_dict = functools.wraps(orig_send_tensor_dict)(vllm_het_send)
    mod.GroupCoordinator.recv_tensor_dict = functools.wraps(orig_recv_tensor_dict)(vllm_het_recv)


patch_parallel_state()
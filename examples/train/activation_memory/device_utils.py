# SPDX-License-Identifier: Apache-2.0
"""Device selection and allocator counters used by the activation-memory examples."""
import os
from functools import lru_cache
from importlib import import_module, util
from importlib.metadata import version

import torch


@lru_cache(maxsize=1)
def device_type() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if util.find_spec('torch_npu') is not None:
        import_module('torch_npu')
        return 'npu'
    return 'cpu'


def accelerator():
    kind = device_type()
    if kind == 'cpu':
        raise RuntimeError('Activation-memory measurements require CUDA or Ascend NPU')
    return getattr(torch, kind)


def set_device(local_rank: int) -> None:
    if device_type() != 'cpu':
        accelerator().set_device(local_rank)


def synchronize() -> None:
    accelerator().synchronize()


def reset_peak_memory_stats() -> None:
    accelerator().reset_peak_memory_stats()


def max_memory_allocated() -> int:
    return accelerator().max_memory_allocated()


def max_memory_reserved() -> int:
    return accelerator().max_memory_reserved()


def runtime_metadata():
    if device_type() == 'npu':
        return {'device_type': 'npu', 'torch_npu': version('torch-npu')}
    return {'device_type': device_type(), 'cuda_version': torch.version.cuda}


def allocator_config() -> str:
    if device_type() == 'npu':
        return os.environ.get('PYTORCH_NPU_ALLOC_CONF', '')
    return os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''))

# SPDX-License-Identifier: Apache-2.0
"""Compare both DMD2 execution schedules from identical Wan weights and RNG states."""
import random
from copy import deepcopy

import numpy as np
import torch
from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.methods.distribution_matching.dmd2 import SequentialDMD2Method
from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed

import device_utils


def local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def cpu_copy(tensor):
    return local(tensor.detach()).to('cpu', copy=True)


def check_schedules(method, batch):
    params = {
        f'{role}.{name}': parameter
        for role in ('student', 'critic')
        for name, parameter in getattr(method, role).transformer.named_parameters()
        if parameter.requires_grad
    }
    initial = {name: cpu_copy(p) for name, p in params.items()}
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    device = device_utils.accelerator()
    device_rng = device.get_rng_state()
    sampler_rng = method.cuda_generator.get_state()
    optimizers = method.get_optimizers(0)

    def tensors_after_update():
        for name, p in params.items():
            yield 'parameters', name, p
            for opt in optimizers:
                for key, value in opt.state.get(p, {}).items():
                    if torch.is_tensor(value):
                        yield key, name, value

    def execute(schedule, collect):
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        device.set_rng_state(device_rng)
        method.cuda_generator.set_state(sampler_rng)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses, _, _ = schedule(method, deepcopy(batch), 0, grad_accum_rounds=1)
        for name, p in params.items():
            if p.grad is not None:
                collect('gradients', name, p.grad)
        norms = {r: clip_grad_norm_if_needed(m, 1.0) for r, m in method.get_grad_clip_targets(0).items()}
        for optimizer in optimizers:
            optimizer.step()
        for group, name, value in tensors_after_update():
            collect(group, name, value)
        return {'losses': {k: float(v.detach()) for k, v in losses.items()}, 'gradient_norms': norms}

    reference = {}

    def save(group, name, tensor):
        reference[(group, name)] = cpu_copy(tensor)

    original = execute(TrainingMethod.train_microbatch, save)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
        opt.state.clear()
    with torch.no_grad():
        for name, p in params.items():
            local(p).copy_(initial[name])
    del initial
    groups = {}

    def compare(group, name, tensor):
        target = reference.pop((group, name))
        value = cpu_copy(tensor)
        stats = groups.setdefault(group, dict(tensors=0, elements=0, unequal=0, max_abs=0.0))
        stats['tensors'] += 1
        stats['elements'] += value.numel()
        if not torch.equal(value, target):
            stats['unequal'] += 1
            stats['max_abs'] = max(stats['max_abs'], float((target - value).abs().max()))

    sequential = execute(SequentialDMD2Method.train_microbatch, compare)
    if reference:
        raise RuntimeError('Sequential update did not produce all reference tensors')
    return {'original': original, 'sequential': sequential, 'tensors': groups,
            'exact': original == sequential and all(s['unequal'] == 0 for s in groups.values())}

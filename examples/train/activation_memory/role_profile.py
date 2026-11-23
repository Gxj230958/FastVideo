# SPDX-License-Identifier: Apache-2.0
"""Measure role lifetimes in FastVideo's production data-latent DMD2 method."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from time import perf_counter_ns

import torch
from fastvideo.train.utils.activation_memory import BlockCosts, MemoryProfile, PolicyCost

import device_utils


def calibrate_dmd2(method: Any, step: Callable[..., Any], template: MemoryProfile, *,
                   warmup: int, repetitions: int, training_sample: str) -> MemoryProfile:
    costs = template.metadata['block_measurements']
    count = len(method.student.transformer.blocks)
    events = tuple([f'student.forward.{index}' for index in range(count)] +
                   ['score.critic', 'score.teacher.conditional', 'score.teacher.unconditional', 'rollout.student'] +
                   [f'critic.forward.{index}'
                    for index in range(count)] + [f'student.backward.{index}' for index in reversed(range(count))] +
                   [f'critic.backward.{index}'
                    for index in reversed(range(count))] + ['clipping', 'optimizer.critic', 'optimizer.student'])
    indices = {event: index for index, event in enumerate(events)}
    blocks = []
    for role in ('student', 'critic'):
        for index in range(count):
            start, end = indices[f'{role}.forward.{index}'], indices[f'{role}.backward.{index}']
            choices = []
            for policy, cost in costs.items():
                device, host = [0] * len(events), [0] * len(events)
                for event in range(start, end + 1):
                    device[event] = cost['saved_bytes']
                    if policy == 'offload':
                        host[event] = cost['host_bytes']
                device[start] = max(device[start], cost['forward_peak'])
                device[end] = max(device[end], cost['backward_peak'])
                choices.append(PolicyCost(policy, cost['time_ms'], tuple(device), tuple(host)))
            blocks.append(BlockCosts(role, f'blocks.{index}', tuple(choices)))
    for _ in range(warmup):
        step()
    observed = [0] * len(events)
    phase: str | None = None

    def transition(name: str) -> None:
        nonlocal phase
        device_utils.synchronize()
        if phase is not None and phase in indices:
            observed[indices[phase]] = max(observed[indices[phase]], device_utils.max_memory_allocated())
        phase = name
        device_utils.reset_peak_memory_stats()

    with observe_role_events(method, transition):
        for _ in range(repetitions):
            step(transition)
        transition('complete')
    reference = [
        sum(next(choice for choice in block.choices if choice.policy == 'recompute').device_bytes[event]
            for block in blocks) for event in range(len(events))
    ]
    fixed = tuple(max(0, actual - selected) for actual, selected in zip(observed, reference, strict=True))
    return MemoryProfile(
        events, fixed, tuple(blocks), {
            **template.metadata,
            'executor': 'FastVideo DMD2Method / WanModel / FSDP2',
            'method': 'DMD2 data_latent',
            'training_sample': training_sample,
            'optimizer': 'AdamW with FastVideo optimizer configuration',
            'calibration_peaks': observed,
            'event_accounting': 'continuous pre-hook intervals including FSDP transitions',
            'teacher_trainable': False,
        })


@contextmanager
def observe_role_events(method, transition):
    count = len(method.student.transformer.blocks)
    mode = None
    originals, handles = [], []

    def instrument_predict(role: str, original: Callable[..., Any]) -> Callable[..., Any]:

        def predict(*args: Any, **kwargs: Any) -> Any:
            nonlocal mode
            mode = f'{role}.forward' if torch.is_grad_enabled() else 'score'
            if torch.is_grad_enabled():
                transition(f'{role}.forward.0')
            elif role == 'student':
                transition('rollout.student')
            elif role == 'critic':
                transition('score.critic')
            else:
                condition = 'conditional' if kwargs['conditional'] else 'unconditional'
                transition(f'score.teacher.{condition}')
            return original(*args, **kwargs)

        return predict

    def instrument_backward(role: str, original: Callable[..., Any]) -> Callable[..., Any]:

        def backward(*args: Any, **kwargs: Any) -> Any:
            nonlocal mode
            mode = f'{role}.backward'
            transition(f'{role}.backward.{count - 1}')
            return original(*args, **kwargs)

        return backward

    def before(role: str, direction: str, index: int) -> Callable[..., None]:

        def begin(module: Any, *unused: Any) -> None:
            if mode == f'{role}.{direction}':
                transition(f'{role}.{direction}.{index}')

        return begin

    for role in ('student', 'critic', 'teacher'):
        model = getattr(method, role)
        originals.append((model, model.predict_noise, model.backward))
        model.predict_noise = instrument_predict(role, model.predict_noise)
        if role != 'teacher':
            model.backward = instrument_backward(role, model.backward)
            for index, block in enumerate(model.transformer.blocks):
                handles.extend([
                    block.register_forward_pre_hook(before(role, 'forward', index)),
                    block.register_full_backward_pre_hook(before(role, 'backward', index))
                ])
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for model, predict, backward in originals:
            model.predict_noise, model.backward = predict, backward


STALL_EVENTS = ('critic.forward.18', 'critic.backward.20', 'critic.backward.21')


def allocator_step(method, step, *, map_retries=False):
    """Read allocator counters on either side of the three critic intervals."""
    device = device_utils.accelerator()
    keys = ('num_alloc_retries', 'num_ooms', 'reserved_bytes.all.allocated', 'reserved_bytes.all.freed',
            'allocated_bytes.all.current', 'reserved_bytes.all.current', 'inactive_split_bytes.all.current')
    intervals = {}
    active = None
    started = 0

    def snapshot():
        stats = device.memory_stats()
        return {key: stats[key] for key in keys}

    transitions = []

    def transition(name):
        nonlocal active, started
        if map_retries or active is not None or name in STALL_EVENTS:
            now = perf_counter_ns()
            counters = snapshot()
            if map_retries:
                transitions.append((name, counters))
            if active is not None:
                intervals[active].update(exit=counters, host_ms=(now - started) / 1e6)
            if name in STALL_EVENTS:
                intervals[name] = {'entry': counters}
                started = perf_counter_ns()
        active = name if name in STALL_EVENTS else None

    with observe_role_events(method, transition):
        entry = snapshot()
        record = step(transition)
        device_utils.synchronize()
        record['allocator_whole_step'] = {'entry': entry, 'exit': snapshot()}
    if map_retries:
        boundaries = [('setup', entry), *transitions, ('complete', record['allocator_whole_step']['exit'])]
        record['retry_locations'] = []
        for (name, before), (_, after) in zip(boundaries, boundaries[1:]):
            retries = after['num_alloc_retries'] - before['num_alloc_retries']
            if retries:
                record['retry_locations'].append({'event': name, 'retries': retries,
                                                  'freed_bytes': after['reserved_bytes.all.freed'] -
                                                  before['reserved_bytes.all.freed']})
    record['allocator_intervals'] = intervals
    return record


def trace_step(method, step, sync_event=None):
    """Measure stream intervals and group CPU operator self times by submission interval."""
    from collections import Counter

    markers = []
    scope = None
    snapshots = {}
    barrier_ms = []
    device = device_utils.accelerator()

    def mark(name):
        marker = device.Event(enable_timing=True)
        marker.record()
        markers.append((name, marker, perf_counter_ns()))

    def memory():
        return {'allocated_bytes': device.memory_allocated(), 'reserved_bytes': device.memory_reserved()}

    def transition(name):
        nonlocal scope
        if scope is not None:
            scope.__exit__(None, None, None)
        if name in STALL_EVENTS or name == sync_event:
            snapshots[name] = {'entry': memory()}
        if name == sync_event:
            mark('barrier.' + name)
            started = perf_counter_ns()
            device_utils.synchronize()
            barrier_ms.append((perf_counter_ns() - started) / 1e6)
            snapshots[name]['after_sync'] = memory()
        mark(name)
        scope = torch.profiler.record_function('role_event/' + name)
        scope.__enter__()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with observe_role_events(method, transition):
            transition('setup')
            record = step(transition)
            transition('complete')
            scope.__exit__(None, None, None)
            device_utils.synchronize()

    elapsed = Counter()
    if sync_event and len(barrier_ms) != 1:
        raise ValueError(f'Expected one visit to {sync_event}, observed {len(barrier_ms)}')
    host = Counter()
    for (name, first, start), (_, second, end) in zip(markers, markers[1:]):
        elapsed[name] += first.elapsed_time(second)
        host[name] += (end - start) / 1e6
    record['barrier_host_ms'] = sum(barrier_ms)
    record['host_interval_ms'] = dict(host)
    record['allocator_at_entry'] = snapshots
    cpu_ops = {}
    for op in prof.events():
        parent = op.cpu_parent
        while parent is not None and not parent.name.startswith('role_event/'):
            parent = parent.cpu_parent
        if parent is not None:
            name = parent.name.removeprefix('role_event/')
            cpu_ops.setdefault(name, Counter())[op.name] += op.self_cpu_time_total / 1000
    record['stream_ms'] = dict(elapsed)
    record['cpu_self_ms'] = cpu_ops
    return record

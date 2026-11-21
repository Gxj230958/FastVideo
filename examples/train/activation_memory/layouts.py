# SPDX-License-Identifier: Apache-2.0
"""Manual activation layouts for training and candidate generation."""


def uniform_indices(count, save_count):
    return {int((index + 0.5) * count / save_count) for index in range(save_count)}


def policy_for(model, policy, checkpoint_every=2, save_count=None):
    count = len(model.blocks)
    if policy in {'prefix', 'suffix', 'uniform'}:
        if save_count is None or not 0 <= save_count <= count:
            raise ValueError('Manual layouts require --save-count between zero and the number of blocks')
        if policy == 'prefix':
            saved = set(range(save_count))
        elif policy == 'suffix':
            saved = set(range(count - save_count, count))
        else:
            saved = uniform_indices(count, save_count)
        return {f'blocks.{index}': 'save' if index in saved else 'recompute' for index in range(count)}
    if policy == 'block_skip':
        return {f'blocks.{index}': 'recompute' if index % checkpoint_every == 0 else 'save'
                for index in range(len(model.blocks))}
    if policy == 'mixed':
        return {f'blocks.{index}': ('save', 'recompute', 'offload')[index % 3]
                for index in range(len(model.blocks))}
    return {f'blocks.{index}': policy for index in range(len(model.blocks))}


# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method, SequentialDMD2Method


def run_schedule(method_class):
    method = object.__new__(method_class)
    torch.nn.Module.__init__(method)
    student = torch.nn.Parameter(torch.tensor(1.25))
    critic = torch.nn.Parameter(torch.tensor(0.75))
    events = []

    def backward(role):
        def accumulate(loss, ctx, *, grad_accum_rounds):
            events.append(role)
            (loss / grad_accum_rounds).backward()
        return accumulate

    def critic_loss(batch):
        events.append("critic_forward")
        return (critic * batch.latents - student.detach()).square(), (None, None), {}

    method.student = SimpleNamespace(prepare_batch=lambda batch, **kwargs: SimpleNamespace(**batch),
                                     backward=backward("student_backward"))
    method.critic = SimpleNamespace(backward=backward("critic_backward"))
    method.cuda_generator = None
    method._rollout_mode = "data_latent"
    method.method_config = {"generator_update_interval": 2}
    method._student_rollout = lambda batch, **kwargs: student * batch.latents
    method._dmd_loss = lambda pred, batch: (pred - critic.detach()).square()
    method._critic_flow_matching_loss = critic_loss
    opt = torch.optim.AdamW([student, critic], lr=0.01)
    records = []
    for iteration in (0, 1):
        opt.zero_grad(set_to_none=True)
        for x in (1.0, 3.0):
            batch = dict(latents=torch.tensor(x), timesteps=None, attn_metadata_vsa=None)
            losses, _, _ = method.train_microbatch(batch, iteration, grad_accum_rounds=2)
            records.append(tuple(float(v.detach()) for v in losses.values()))
            records.append(tuple(float(p.grad) if p.grad is not None else None for p in (student, critic)))
        opt.step()
        records.append(tuple(float(p.detach()) for p in (student, critic)))
    return records, events


def test_interleaved_role_backward_preserves_accumulation_and_critic_only_steps():
    original, joint_events = run_schedule(DMD2Method)
    sequential, early_events = run_schedule(SequentialDMD2Method)
    assert joint_events == ["critic_forward", "student_backward", "critic_backward"] * 2 + [
        "critic_forward", "critic_backward",
    ] * 2
    assert early_events == ["student_backward", "critic_forward", "critic_backward"] * 2 + [
        "critic_forward", "critic_backward",
    ] * 2
    assert original == sequential

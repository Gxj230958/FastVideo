# SPDX-License-Identifier: Apache-2.0
"""Select uniform student and critic policies by measuring memory-feasible layouts."""
import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

from layouts import uniform_indices

STUDENT_COUNTS = (0, 5, 10, 15, 20, 24, 30)
CRITIC_COUNTS = (0, 4, 8, 12, 16, 20, 24, 30)


def candidates(profile, budget):
    rows = []
    for student, critic in itertools.product(STUDENT_COUNTS, CRITIC_COUNTS):
        events = list(profile['fixed_device_bytes'])
        saved = {role: uniform_indices(30, n)
                 for role, n in [('student', student), ('critic', critic)]}
        for block in profile['blocks']:
            index = int(block['block'].split('.')[-1])
            policy = 'save' if index in saved[block['role']] else 'recompute'
            choice = next(item for item in block['choices'] if item['policy'] == policy)
            events = [a + b for a, b in zip(events, choice['device_bytes'], strict=True)]
        rows.append(dict(student=student, critic=critic, predicted_peak_bytes=max(events), feasible=max(events) <= budget))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'profile', 'training-sample', 'data', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--launcher', help='Shell launcher that accepts a Python script and arguments')
    parser.add_argument('--budget-gib', type=float, default=54)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    for role in ('student', 'critic'):
        blocks = {b['block'] for b in profile['blocks'] if b['role'] == role}
        if blocks != {f'blocks.{i}' for i in range(30)}:
            raise ValueError('The 30-block grid requires a 30-block profile for each role')
    budget = int(args.budget_gib * 2**30)
    matrix = candidates(profile, budget)
    shrink = sum(row['feasible'] for row in matrix) > 40
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['bash', args.launcher] if args.launcher else [sys.executable]
    for row in matrix:
        s, c = row['student'], row['critic']
        if not row['feasible'] or (shrink and (s not in (0, 15, 24, 30) or c not in (0, 8, 16, 24, 30))):
            continue
        target = output.parent / f'candidate_{s}_{c}.json'
        with target.with_suffix('.log').open('w') as log:
            subprocess.run([*command, str(Path(__file__).with_name('profile_dmd2.py')), '--model', args.model,
                            '--training-sample', args.training_sample, '--data', args.data, '--policy', 'uniform',
                            '--save-count', str(s), '--critic-save-count', str(c), '--warmup', '2', '--steps', '5',
                            '--output', str(target)], check=True, stdout=log, stderr=subprocess.STDOUT)
        run = json.loads(target.read_text())
        if any(sum(p == 'save' for p in run['manual_policies'][role].values()) != n
               for role, n in [('student', s), ('critic', c)]):
            raise RuntimeError('Measured layout differs from candidate')
        times = [step['step_ms'] for step in run['steps']]
        row.update(mean_ms=mean(times), min_ms=min(times), max_ms=max(times), peak_bytes=run['peak_allocated_bytes'],
                   source=target.name)
        output.write_text(json.dumps({'matrix': matrix}, indent=2) + '\n')
    winner = min((r for r in matrix if r.get('peak_bytes', budget + 1) <= budget), key=lambda r: r['mean_ms'])
    output.write_text(json.dumps({'budget_bytes': budget, 'winner': winner, 'matrix': matrix}, indent=2) + '\n')


if __name__ == '__main__':
    main()

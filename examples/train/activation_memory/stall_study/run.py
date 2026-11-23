# SPDX-License-Identifier: Apache-2.0
"""Run one matched block of the DMD2 synchronization experiment."""
import argparse
import gzip
import json
import itertools
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EVENTS = ('none', 'critic.forward.18', 'critic.backward.20', 'critic.backward.21')


def predicted_gib(layout, profile):
    policies = json.loads((HERE / 'layouts' / f'{layout}.json').read_text())
    peaks = profile['fixed_device_bytes'].copy()
    for block in profile['blocks']:
        policy = policies[block['role']][block['block']]
        selected = next(option for option in block['choices'] if option['policy'] == policy)
        peaks = [a + b for a, b in zip(peaks, selected['device_bytes'], strict=True)]
    return max(peaks) / 2 ** 30


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'data', 'output', 'sample'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--launcher', help='Shell script accepting a Python program and arguments')
    p.add_argument('--phase', required=True, choices=('baseline', 'trace', 'confirm-baseline', 'confirm-trace', 'allocator', 'serial'))
    p.add_argument('--block', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--confirm-event', choices=EVENTS[1:])
    args = p.parse_args()
    if args.phase == 'baseline':
        cases = [(x, 'off') for x in ('solver', 'manual', 'selected', 'odd', 'even')]
    elif args.phase == 'serial':
        cases = [(name, 'serial') for name in ('solver', 'manual', 'selected')]
    elif args.phase == 'allocator':
        cases = [('solver', 'allocator'), ('manual', 'allocator')]
    elif args.phase == 'confirm-baseline':
        cases = [('solver', 'off'), ('manual', 'off')]
    else:
        if args.phase == 'confirm-trace' and not args.confirm_event:
            p.error('confirm-trace requires --confirm-event')
        events = EVENTS if args.phase == 'trace' else ('none', args.confirm_event)
        cases = [(layout, event) for event, layout in itertools.product(events, ('solver', 'manual'))]
    offset = args.block * 3 % len(cases)
    cases = cases[offset:] + cases[:offset]
    if args.block == 1 and args.phase != 'confirm-baseline':
        cases.reverse()
    if args.phase == 'allocator' and args.sample == 'hiker':
        cases.append(('selected', 'allocator'))
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    data = Path(args.data)
    with gzip.open(HERE / 'ascend_profile.json.gz', 'rt') as src:
        profile = json.load(src)
    for layout, event in cases:
        predicted = predicted_gib(layout, profile)
        if predicted > 54:
            raise ValueError(f'{layout} requires {predicted:.3f} GiB under the Ascend profile')
        target = out / f'{args.sample}_{args.block}_{layout}_{event}.json'
        if target.exists():
            continue
        partial = target.with_suffix('.partial')
        cmd = ['bash', args.launcher] if args.launcher else [sys.executable]
        cmd += [str(HERE.parent / 'profile_dmd2.py'), '--model', args.model, '--training-sample',
                str(data / f'{args.sample}_17_256_448.pt'), '--data', str(data / f'{args.sample}_parquet'),
                '--layout-file', str(HERE / 'layouts' / f'{layout}.json'), '--warmup', '5',
                '--steps', '5' if event in ('off', 'serial') else '2', '--output', str(partial)]
        if event == 'allocator':
            cmd += ['--allocator-events']
        elif event not in ('off', 'serial'):
            cmd += ['--trace-events']
        if event not in ('off', 'serial', 'none', 'allocator'):
            cmd += ['--sync-event', event]
        print(target.stem, flush=True)
        with target.with_suffix('.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        partial.replace(target)


if __name__ == '__main__':
    main()

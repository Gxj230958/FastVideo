# SPDX-License-Identifier: Apache-2.0
"""Replay matched critic edits against the exact DMD2 solver layout."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / 'stall_study'
EDITS = {'solver': (), 'A1': (12,), 'B1': (24,), 'A2': (11, 12), 'B2': (24, 25),
         'Ap1': (10,), 'Bp1': (26,), 'Ap2': (9, 10), 'Bp2': (26, 27)}


def materialize(name, profile):
    source = 'selected' if name == 'selected' else 'solver'
    layout = json.loads((STUDY / 'layouts' / f'{source}.json').read_text())
    for index in EDITS.get(name, ()):
        layout['critic'][f'blocks.{index}'] = 'recompute'
    peaks = profile['fixed_device_bytes'].copy()
    for block in profile['blocks']:
        policy = layout[block['role']][block['block']]
        cost = next(x for x in block['choices'] if x['policy'] == policy)
        peaks = [a + b for a, b in zip(peaks, cost['device_bytes'], strict=True)]
    return layout, peaks[profile['events'].index('critic.forward.18')] / 2**30, max(peaks) / 2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'data', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--sample', choices=('hiker', 'cosmos'), required=True)
    parser.add_argument('--stage', choices=('single', 'double', 'validation', 'migration'), required=True)
    parser.add_argument('--mode', choices=('timing', 'counters'), required=True)
    parser.add_argument('--block', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--launcher')
    args = parser.parse_args()
    root = Path(args.output)
    if args.stage == 'migration':
        if args.mode != 'counters' or args.sample != 'hiker':
            parser.error('Migration mapping uses counters on Hiker')
        cases = ['solver', 'A1', 'B1']
    elif args.stage == 'validation':
        decision = json.loads((root / 'prediction.json').read_text())
        k = decision['edit_count']
        cases = ['solver', f'Ap{k}', f'Bp{k}']
    elif args.stage == 'single':
        cases = ['solver', 'A1', 'B1', 'selected']
    else:
        cases = ['A2', 'B2']
    offset = args.block % len(cases)
    cases = cases[offset:] + cases[:offset]
    directory = root / args.stage / args.sample / str(args.block) / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    with gzip.open(STUDY / 'ascend_profile.json.gz', 'rt') as fp:
        profile = json.load(fp)
    for name in cases:
        out = directory / f'{name}.json'
        if out.exists():
            continue
        layout, hotspot, peak = materialize(name, profile)
        if peak > 54:
            raise ValueError(f'{name} exceeds budget: {peak} GiB')
        policy = directory / f'{name}.layout.json'
        policy.write_text(json.dumps(layout, indent=2) + '\n')
        pending = out.with_suffix('.partial')
        cmd = ['bash', args.launcher] if args.launcher else [sys.executable]
        cmd += [str(HERE.parent / 'profile_dmd2.py'), '--model', args.model,
                '--data', str(data / f'{args.sample}_parquet'), '--training-sample',
                str(data / f'{args.sample}_17_256_448.pt'), '--layout-file', str(policy),
                '--warmup', '5', '--steps', '5', '--output', str(pending)]
        if args.mode == 'counters':
            cmd.append('--map-retries' if args.stage == 'migration' else '--allocator-events')
        start = datetime.now(timezone.utc).isoformat()
        print(args.stage, args.sample, args.block, args.mode, name, flush=True)
        with out.with_suffix('.log').open('w') as fp:
            subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, check=True)
        run = json.loads(pending.read_text())
        if run['manual_policies'] != layout or run['world_size'] != 1:
            raise ValueError('Observed layout or rank count differs from the matched design')
        run.update(started_utc=start, predicted_forward18_gib=hotspot, predicted_peak_gib=peak)
        pending.write_text(json.dumps(run, indent=2) + '\n')
        pending.replace(out)


if __name__ == '__main__':
    main()

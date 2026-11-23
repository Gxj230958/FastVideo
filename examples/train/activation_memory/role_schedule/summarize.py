# SPDX-License-Identifier: Apache-2.0
"""Summarize each independent process measuring the DMD2 role schedules."""
import csv
import gzip
import json
import sys
from pathlib import Path
from statistics import mean

GIB = 2**30


def process(path):
    with gzip.open(path, 'rt') as stream:
        run = json.load(stream)
    steps = run['steps']
    sample, card, mode, schedule = path.name.split('.')[0].split('_')
    row = dict(sample=sample, card=card, mode=mode, schedule=schedule, steps=len(steps),
               step_ms=mean(s['step_ms'] for s in steps),
               peak_allocated_gib=max(s['peak_allocated_bytes'] for s in steps) / GIB,
               peak_reserved_gib=max(s['peak_reserved_bytes'] for s in steps) / GIB)
    if mode == 'counters':
        def delta(interval, key):
            return interval['exit'][key] - interval['entry'][key]

        for metric in ('num_alloc_retries', 'num_ooms'):
            row[metric] = sum(delta(s['allocator_whole_step'], metric) for s in steps)
        for label, event in [('f18', 'critic.forward.18'), ('b20', 'critic.backward.20'), ('b21', 'critic.backward.21')]:
            windows = [s['allocator_intervals'][event] for s in steps]
            row[label + '_host_ms'] = mean(w['host_ms'] for w in windows)
            row[label + '_entry_allocated_gib'] = mean(w['entry']['allocated_bytes.all.current'] for w in windows) / GIB
            row[label + '_released_gib'] = mean(delta(w, 'reserved_bytes.all.freed') for w in windows) / GIB
            row[label + '_retries'] = sum(delta(w, 'num_alloc_retries') for w in windows)
    if mode == 'trace':
        row['barrier_ms'] = mean(s['barrier_host_ms'] for s in steps)
        row['b20_host_ms'] = mean(s['host_interval_ms']['critic.backward.20'] for s in steps)
        row['b20_empty_tensor_ms'] = mean(s['cpu_self_ms']['critic.backward.20'].get('empty_tensor', 0.0)
                                          for s in steps)
    return row


def main():
    rows = [process(p) for p in sorted(Path(sys.argv[1]).glob('*.json.gz'))]
    writer = csv.DictWriter(sys.stdout, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()

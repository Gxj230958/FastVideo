# SPDX-License-Identifier: Apache-2.0
"""Extract process means, trace intervals and figures from archived single-rank runs."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EVENTS = ('critic.forward.18', 'critic.backward.20', 'critic.backward.21')
LABELS = {'solver': 'Solver 24+30', 'selected': 'Selected 24+16', 'manual': 'Manual 15+15',
          'odd': 'Odd 15+30', 'even': 'Even 15+30'}
GIB = 2 ** 30


def read_runs(source, archive=None):
    for path in sorted(source.iterdir()):
        if not path.name.endswith(('.json', '.json.gz')):
            continue
        sample, block, layout, condition = path.name.split('.json')[0].split('_')
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as src:
            run = json.load(src)
        policies = json.loads((HERE / 'layouts' / f'{layout}.json').read_text())
        if run['world_size'] != 1 or run['manual_policies'] != policies or run['sync_event'] != (
                None if condition in ('off', 'serial', 'none', 'allocator') else condition):
            raise ValueError(f'Layout or synchronization mismatch in {path}')
        steps = run['steps']
        if [s['iteration'] for s in steps] != list(range(5, 10 if condition in ('off', 'serial') else 7)):
            raise ValueError(f'Incomplete measurement in {path}')
        if archive is not None:
            archive.mkdir(exist_ok=True, parents=True)
            run.pop('rank_measurements', None)
            run['layout_file'] = f'layouts/{layout}.json'
            packed = json.dumps(run, separators=(',', ':'), allow_nan=False).encode()
            (archive / (path.name.removesuffix('.gz') + '.gz')).write_bytes(gzip.compress(packed, mtime=0))
        yield dict(sample=sample, block=int(block), card=run['visible_devices'], layout=layout,
                   condition=condition), steps


def write_rows(path, rows):
    with path.open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def extract(source, archive=None):
    processes, events, counters = [], [], []
    for key, steps in read_runs(source, archive):
        processes.append(dict(**key, step_ms=mean(s['step_ms'] for s in steps),
                              peak_alloc_gib=max(s['peak_allocated_bytes'] for s in steps) / GIB,
                              peak_reserved_gib=max(s['peak_reserved_bytes'] for s in steps) / GIB))
        for s in steps:
            for event, interval in s.get('allocator_intervals', {}).items():
                start, end = interval['entry'], interval['exit']
                counters.append(dict(**key, iteration=s['iteration'], event=event, host_ms=interval['host_ms'],
                                     retries=end['num_alloc_retries'] - start['num_alloc_retries'],
                                     ooms=end['num_ooms'] - start['num_ooms'],
                                     released_gib=(end['reserved_bytes.all.freed'] - start['reserved_bytes.all.freed']) / GIB,
                                     obtained_gib=(end['reserved_bytes.all.allocated'] - start['reserved_bytes.all.allocated']) / GIB))
            if 'stream_ms' not in s:
                continue
            for name in EVENTS:
                ops = s['cpu_self_ms'][name]
                top = max(ops, key=ops.get)
                entry = s['allocator_at_entry'][name]['entry']
                events.append(dict(**key, iteration=s['iteration'], event=name, host_ms=s['host_interval_ms'][name],
                                   stream_ms=s['stream_ms'][name], empty_ms=ops.get('empty_tensor', 0),
                                   stack_ms=ops.get('aten::stack', 0), top_op=top, top_ms=ops[top],
                                   barrier_ms=s['barrier_host_ms'] if key['condition'] == name else 0,
                                   allocated_gib=entry['allocated_bytes'] / GIB, reserved_gib=entry['reserved_bytes'] / GIB))
    return processes, events, counters


def export(fig, path):
    for ext in ('svg', 'pdf'):
        fig.savefig(path.with_suffix('.' + ext), bbox_inches='tight')
    plt.close(fig)


def plot_steps(rows, folder):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True, width_ratios=(5, 2), layout='constrained')
    for ax, sample, layouts in zip(axes, ('hiker', 'cosmos'), (tuple(LABELS), ('solver', 'manual'))):
        subset = [r for r in rows if r['sample'] == sample and r['condition'] == 'off']
        for i, name in enumerate(layouts):
            values = sorted((r for r in subset if r['layout'] == name), key=lambda r: r['block'])
            for r in values:
                block = r['block']
                ax.scatter(i + (block - 1) * 0.11, r['step_ms'] / 1000, color='#356887', marker=('o', 's', '^')[block],
                           zorder=3, label=f"Repeat {block + 1}" if i == 0 else None)
        ax.set(xticks=range(len(layouts)), xticklabels=[LABELS[l].replace(' ', '\n') for l in layouts],
               title=sample.capitalize(), xlim=(-0.5, len(layouts) - 0.5), ylim=(0, None))
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('Full update (s), process mean')
    axes[1].legend(loc='lower right', frameon=False)
    export(fig, folder / 'layout_time')


def plot_trace(events, folder):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True, layout='constrained')
    for ax, event, title in zip(axes, EVENTS, ('Critic forward 18', 'Critic backward 20')):
        conditions = [('solver', 'none'), ('solver', event), ('solver', EVENTS[2]), ('manual', event)]
        for i, (layout, cond) in enumerate(conditions):
            group = [r for r in events if (r['sample'], r['layout'], r['condition'], r['event']) ==
                     ('hiker', layout, cond, event)]
            barrier, host = (mean(r[k] for r in group) for k in ('barrier_ms', 'host_ms'))
            ax.bar(i, barrier, color='#b56341', label='Sync at target' if i == 0 else None)
            ax.bar(i, host, bottom=barrier, color='#48839c', label='Target interval' if i == 0 else None)
            if i == 1:
                ax.annotate(f'Sync {barrier:.2f} ms', (i, (barrier + host) * .58), ha='center',
                            color='white', fontsize=8, rotation=90)
            for block in range(3):
                values = [r['host_ms'] + r['barrier_ms'] for r in group if r['block'] == block]
                ax.plot(i + (block - 1) * .1, mean(values), 'k.', ms=4)
        ax.set(title=title, xticks=range(4), xticklabels=('Solver\nbase', 'Solver\ntarget sync',
                                                        'Solver\nB21 sync', 'Manual\ntarget sync'))
        ax.grid(axis='y', alpha=.17)
    axes[0].set_ylabel('Host wall time (ms)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, frameon=False, fontsize=9)
    export(fig, folder / 'barrier_time')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--archive', type=Path, help='Export compressed, single-copy measurements')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    processes, events, counters = extract(args.runs, args.archive)
    write_rows(args.output / 'processes.csv', processes)
    write_rows(args.output / 'events.csv', events)
    write_rows(args.output / 'allocator.csv', counters)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    plot_steps(processes, args.output)
    plot_trace(events, args.output)


if __name__ == '__main__':
    main()

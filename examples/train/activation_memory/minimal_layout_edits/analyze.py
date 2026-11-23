# SPDX-License-Identifier: Apache-2.0
"""Summarize five-step process means and match comparisons by physical card."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

GIB = 2 ** 30
WINDOWS = {'f18': 'critic.forward.18', 'b20': 'critic.backward.20', 'b21': 'critic.backward.21'}
ORDER = ['solver', 'A1', 'B1', 'A2', 'B2', 'selected', 'Ap1', 'Bp1', 'Ap2', 'Bp2']


def read_records(source):
    for path in sorted(source.glob('*/*/*/*/*')):
        if not path.name.endswith(('.json', '.json.gz')) or path.name.endswith('.layout.json'):
            continue
        stage, sample, block, mode, filename = path.relative_to(source).parts
        loader = gzip.open if filename.endswith('.gz') else open
        with loader(path, 'rt') as fp:
            record = json.load(fp)
        yield (stage, sample, int(block), mode, filename.split('.')[0]), record


def delta(interval, key):
    return interval['exit'][key] - interval['entry'][key]


def rows(source, archive=None):
    steps, windows, locations = [], [], []
    processes = {}
    for (stage, sample, block, mode, name), run in read_records(source):
        card = int(run['visible_devices'])
        key = dict(stage=stage, sample=sample, block=block, card=card, layout=name, mode=mode)
        if run['warmup_steps'] != 5 or [s['iteration'] for s in run['steps']] != list(range(5, 10)):
            raise ValueError(f'Incomplete process: {key}')
        if archive is not None:
            folder = archive / stage / sample / str(block) / mode
            folder.mkdir(exist_ok=True, parents=True)
            record = {k: v for k, v in run.items() if k != 'rank_measurements'}
            record['layout_file'] = f'layouts/{name}.json'
            data = json.dumps(record, separators=(',', ':'), allow_nan=False).encode()
            (folder / f'{name}.json.gz').write_bytes(gzip.compress(data, mtime=0))
        current = []
        for s in run['steps']:
            row = dict(**key, iteration=s['iteration'], step_ms=s['step_ms'],
                       allocated_gib=s['peak_allocated_bytes'] / GIB, reserved_gib=s['peak_reserved_bytes'] / GIB,
                       loss=s['losses']['total_loss'])
            if mode == 'counters':
                for scope, interval in [('whole', s['allocator_whole_step'])] + [
                        (short, s['allocator_intervals'][event]) for short, event in WINDOWS.items()]:
                    retry = delta(interval, 'num_alloc_retries')
                    row[scope] = retry
                    windows.append(dict(**key, iteration=s['iteration'], scope=scope, retries=retry,
                                        ooms=delta(interval, 'num_ooms'),
                                        freed_gib=delta(interval, 'reserved_bytes.all.freed') / GIB,
                                        reserve_change_gib=delta(interval, 'reserved_bytes.all.current') / GIB,
                                        host_ms=interval.get('host_ms', '')))
                if 'retry_locations' in s:
                    if sum(loc['retries'] for loc in s['retry_locations']) != row['whole']:
                        raise ValueError(f'Event map omits whole-step retries: {key}')
                    for loc in s['retry_locations']:
                        locations.append(dict(**key, iteration=s['iteration'], event=loc['event'],
                                              retries=loc['retries'], freed_gib=loc['freed_bytes'] / GIB))
                row['elsewhere'] = row['whole'] - sum(row[k] for k in WINDOWS)
                if row['elsewhere'] < 0:
                    raise ValueError(f'Overlapping counter windows: {key}')
            current.append(row)
        steps.extend(current)
        group = processes.setdefault((stage, sample, block, card, name), {})
        if mode == 'timing':
            group.update(mean_ms=mean(r['step_ms'] for r in current),
                         allocated_gib=max(r['allocated_gib'] for r in current),
                         reserved_gib=max(r['reserved_gib'] for r in current))
        else:
            group.update({k: mean(r[k] for r in current) for k in ('whole', *WINDOWS, 'elsewhere')})
    means = [dict(stage=t, sample=s, block=b, card=c, layout=l, **v) for (t, s, b, c, l), v in processes.items()]
    return steps, windows, means, locations


def write_csv(path, values):
    with path.open('w') as fp:
        fields = list(dict.fromkeys(k for r in values for k in r))
        writer = csv.DictWriter(fp, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def comparisons(means):
    result = []
    for r in means:
        if r['stage'] == 'migration' or not r['layout'].startswith('A'):
            continue
        sample, stage = r['sample'], r['stage']
        peers = {m['layout']: m for m in means if m['sample'] == sample and m['card'] == r['card']
                 and (m['stage'] == stage or (stage == 'double' and m['layout'] == 'solver' and m['stage'] == 'single'))}
        b, solver = peers['B' + r['layout'][1:]], peers['solver']
        result.append(dict(stage=stage, sample=sample, card=r['card'], pair=r['layout'] + '/' + b['layout'],
                           A_minus_B_ms=r['mean_ms'] - b['mean_ms'], A_over_solver=r['mean_ms'] / solver['mean_ms'],
                           B_over_solver=b['mean_ms'] / solver['mean_ms'], A_minus_B_f18=r['f18'] - b['f18'],
                           A_minus_B_whole=r['whole'] - b['whole']))
    return result


def figure(means, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 9})
    groups = [('Discovery · Hiker', [r for r in means if r['stage'] in ('single', 'double')])]
    for sample in ('hiker', 'cosmos'):
        group = [r for r in means if r['sample'] == sample and r['stage'] == 'validation']
        if group:
            groups.append(('Validation · ' + sample.capitalize(), group))
    fig, axes = plt.subplots(len(groups), 2, squeeze=False, figsize=(10, 3.0 * len(groups) + .7), layout='constrained')
    for (title, group), (timing, retries) in zip(groups, axes):
        names = [l for l in ORDER if any(r['layout'] == l for r in group)]
        for block, marker in enumerate(('o', 's', '^')):
            values = [next(r for r in group if r['block'] == block and r['layout'] == n)['mean_ms'] for n in names]
            timing.scatter([i + (block - 1) * .12 for i in range(len(names))], [v / 1000 for v in values],
                           marker=marker, color='#275775', s=24, label=f'Card {block}')
        timing.set(ylabel='Complete update (s)', title=title, ylim=(0, None), xticks=range(len(names)), xticklabels=names)
        bottom = [0.] * len(names)
        for k, label, color in [('f18', 'Forward 18', '#b45c42'), ('b20', 'Backward 20', '#417c96'),
                                ('b21', 'Backward 21', '#8eb4bf'), ('elsewhere', 'Outside all three', '#bbb6ad')]:
            values = [mean(r[k] for r in group if r['layout'] == n) for n in names]
            retries.bar(names, values, bottom=bottom, label=label, color=color)
            bottom = [x + y for x, y in zip(bottom, values)]
        for i,n in enumerate(names):
            v = max(r['allocated_gib'] for r in group if r['layout'] == n)
            retries.text(i, bottom[i] + .04, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        retries.set(ylabel='Retries per update', title='Whole-step retries by location', ylim=(0, max(bottom) + .5))
        for ax in (timing, retries):
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
    axes[0,0].legend(frameon=False, ncol=3, loc='lower left', fontsize=8)
    handles, labels = axes[0,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    for ext in ('svg', 'pdf'):
        fig.savefig(output / f'paired_layouts.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    steps, windows, means, locations = rows(args.records, args.archive)
    for filename, data in [('steps', steps), ('windows', windows), ('processes', means), ('paired', comparisons(means))]:
        write_csv(args.output / f'{filename}.csv', data)
    if locations:
        write_csv(args.output / 'migration.csv', locations)
    figure(means, args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""汇总 h200_stall_results/raw 下的实验 JSON，输出 CSV 与汇总表。"""
import json
import sys
from pathlib import Path

RESULTS = Path('/inspire/hdd/global_user/s26045/fastvideo/h200_stall_results')
RAW = RESULTS / 'raw'
GiB = 2**30


def load(p):
    return json.loads(Path(p).read_text())


def step_ms_mean(rec):
    ms = [s['step_ms'] for s in rec['steps']]
    return sum(ms) / len(ms), ms


def alloc_retry_delta(step):
    w = step.get('allocator_whole_step')
    if not w:
        return None
    return w['exit']['num_alloc_retries'] - w['entry']['num_alloc_retries']


def summarize_normal():
    rows = []
    for p in sorted((RAW / 'normal').glob('*.json')):
        # 命名: L1_r1_A.json
        tag, rnd, sched = p.stem.split('_')
        d = load(p)
        mean_ms, _ = step_ms_mean(d)
        rows.append(dict(layout=tag, round=int(rnd[1:]), sched=sched, mean_step_ms=round(mean_ms, 2),
                         n_steps=len(d['steps']),
                         peak_alloc_gib=round(d['peak_allocated_bytes'] / GiB, 3),
                         peak_reserved_gib=round(max(s['peak_reserved_bytes'] for s in d['steps']) / GiB, 3),
                         first_loss=round(d['steps'][0]['losses'].get('generator', d['steps'][0]['losses'].get(list(d['steps'][0]['losses'])[0])), 6),
                         file=str(p.relative_to(RESULTS))))
    if not rows:
        return
    import csv
    with open(RESULTS / 'csv' / 'normal.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for layout in ('L1', 'L2'):
        print(f'--- {layout} 配对（A=原调度 B=提前反向，值为10步均值 ms）')
        for rnd in (1, 2, 3, 4):
            a = next((r for r in rows if r['layout'] == layout and r['round'] == rnd and r['sched'] == 'A'), None)
            b = next((r for r in rows if r['layout'] == layout and r['round'] == rnd and r['sched'] == 'B'), None)
            if a and b:
                print(f"r{rnd}: A={a['mean_step_ms']} B={b['mean_step_ms']} A-B={round(a['mean_step_ms'] - b['mean_step_ms'], 2)} A/B={round(a['mean_step_ms'] / b['mean_step_ms'], 4)}")


def summarize_diag():
    rows = []
    for p in sorted((RAW / 'diag').glob('*_alloc.json')):
        d = load(p)
        n = len(d['steps'])
        retry_steps = 0
        whole_total = 0
        win_totals = {}
        win_ms = {}
        win_freed = {}
        for s in d['steps']:
            delta = alloc_retry_delta(s) or 0
            whole_total += delta
            win_sum = 0
            for name, iv in s.get('allocator_intervals', {}).items():
                if 'exit' not in iv:
                    continue
                r = iv['exit']['num_alloc_retries'] - iv['entry']['num_alloc_retries']
                win_totals[name] = win_totals.get(name, 0) + r
                win_ms.setdefault(name, []).append(iv.get('host_ms', 0.0))
                win_freed[name] = win_freed.get(name, 0) + (iv['exit']['reserved_bytes.all.freed'] - iv['entry']['reserved_bytes.all.freed'])
                win_sum += r
            ooms = s['allocator_whole_step']['exit']['num_ooms'] - s['allocator_whole_step']['entry']['num_ooms']
            if delta or ooms:
                retry_steps += 1
        rows.append(dict(run=p.stem, steps=n, steps_with_retry=retry_steps, whole_step_retries=whole_total,
                         window_retries=win_totals,
                         window_host_ms_mean={k: round(sum(v) / len(v), 2) for k, v in win_ms.items()},
                         window_freed_gib={k: round(v / GiB, 3) for k, v in win_freed.items()}))
    if not rows:
        return
    with open(RESULTS / 'csv' / 'diag.json', 'w') as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


def summarize_trace():
    rows = []
    for p in sorted((RAW / 'trace').glob('*.json')):
        d = load(p)
        for s in d['steps']:
            host = s.get('host_interval_ms', {})
            top = {}
            for name, ops in s.get('cpu_self_ms', {}).items():
                if ops:
                    op, ms = max(ops.items(), key=lambda kv: kv[1])
                    top[name] = (op, round(ms, 2))
            rows.append(dict(run=p.stem, iter=s['iteration'],
                             barrier_host_ms=round(s.get('barrier_host_ms', 0.0), 3),
                             sync20_host_ms=round(host.get('critic.backward.20', 0.0), 3),
                             host_ms={k: round(v, 2) for k, v in sorted(host.items(), key=lambda kv: -kv[1])[:6]},
                             top_cpu_op=top, step_ms=round(s['step_ms'], 1)))
    if not rows:
        return
    with open(RESULTS / 'csv' / 'trace.json', 'w') as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'normal'):
        summarize_normal()
    if which in ('all', 'diag'):
        summarize_diag()
    if which in ('all', 'trace'):
        summarize_trace()

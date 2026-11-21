# SPDX-License-Identifier: Apache-2.0
"""Compare a bilinear joint cost and a fitted activation-pressure threshold on held-out layouts."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('matrix', 'profile', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    grid = json.loads(args.matrix.read_text())['matrix']
    profile = json.loads(args.profile.read_text())
    meta = profile['metadata']
    rows = [dict(r) for r in grid if 'mean_ms' in r]
    # Reserve the 15-block student sweep and the dense 30+24 combination for validation.
    def held_out(row):
        return row['student'] == 15 or (row['student'], row['critic']) == (30, 24)
    for r in rows:
        r['additive_ms'] = meta['time_model_reference_ms'] + sum(
            r[role] / 30 * meta['role_step_deltas_ms'][role]['save'] for role in ('student', 'critic'))
        r['saved_blocks'] = r['student'] + r['critic']
    train = [r for r in rows if not held_out(r)]
    anchors = {(r['student'], r['critic']): r['mean_ms'] for r in train}
    a, b, c, d = (anchors[k] for k in ((0, 0), (24, 0), (0, 24), (24, 24)))
    def residual(r):
        return r['mean_ms'] - r['additive_ms']
    fits = []
    for threshold in sorted({r['saved_blocks'] for r in train})[1:]:
        offsets = [mean(residual(r) for r in train if (r['saved_blocks'] >= threshold) == flag)
                   for flag in (False, True)]
        mse = mean((residual(r) - offsets[r['saved_blocks'] >= threshold])**2 for r in train)
        fits.append((mse, threshold, offsets))
    _, threshold, offsets = min(fits)
    scores = {}
    for r in rows:
        x, y = r['student'] / 24, r['critic'] / 24
        r['bilinear_ms'] = a + (b-a)*x + (c-a)*y + (d-b-c+a)*x*y
        r['threshold_ms'] = r['additive_ms'] + offsets[r['saved_blocks'] >= threshold]
        r['held_out'] = held_out(r)
    for mode in ('additive', 'bilinear', 'threshold'):
        errors = [abs(r[mode+'_ms']/r['mean_ms']-1) for r in rows if r['held_out']]
        scores[mode] = dict(median_abs_relative_error=median(errors), max_abs_relative_error=max(errors))
    args.output.write_text(json.dumps(dict(threshold=threshold, offsets_ms=offsets, anchors_ms=[a,b,c,d],
                                           scores=scores, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()

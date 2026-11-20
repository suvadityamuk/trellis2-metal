"""Offline: bench/*.json -> docs/timing_<type>.png (per-image stacked stages) and docs/speedup.png (sums per pipeline type).
  ref/venv/bin/python tools/plot_bench.py"""
import json, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np

V = [('official', 'Official PyTorch-MPS'), ('parity', 'Metal parity'), ('fast', 'Metal fast')]
S = [('cond', 'DINOv3 cond', '#937860'), ('sparse_structure', 'sparse structure', '#4C72B0'), ('shape_slat', 'shape SLat', '#DD8452'),
     ('tex_slat', 'texture SLat', '#55A868'), ('decode_shape', 'shape decoder', '#C44E52'), ('decode_tex', 'texture decoder', '#8172B3')]
T = [(t, n) for t, n in (('512', '512'), ('1024c', '1024_cascade')) if all(os.path.exists(f'bench/{k}_{t}.json') for k, _ in V)]
D = {t: {k: json.load(open(f'bench/{k}_{t}.json')) for k, _ in V} for t, _ in T}
short = lambda i: 'T' if i == 'T.png' else i[:4]

for t, tn in T:
    d = D[t]; imgs = list(d['official'])
    fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.27
    for j, (k, lab) in enumerate(V):
        x = np.arange(len(imgs)) + (j - 1) * w; b = np.zeros(len(imgs))
        for s, sl, c in S:
            h = np.array([d[k][i][s] for i in imgs])
            ax.bar(x, h, w * 0.92, bottom=b, color=c, alpha=[0.5, 0.75, 1][j], label=sl if j == 2 else None); b += h
        tot = [d[k][i]['total'] for i in imgs]
        for xi, tt in zip(x, tot): ax.text(xi, tt + max(tot) * 0.01, f'{tt:.0f}', ha='center', fontsize=8)
    ax.set_xticks(np.arange(len(imgs)), [short(i) for i in imgs]); ax.set_ylabel('seconds (M5 Pro, 20-core GPU)')
    ax.set_title(f'{tn}: per-image time; bars left→right: official PyTorch-MPS, Metal parity, Metal fast')
    ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(f'docs/timing_{t}.png', dpi=130)

fig, axs = plt.subplots(1, len(T), figsize=(6.5 * len(T), 2.8), squeeze=False)
for ax, (t, tn) in zip(axs[0], T):
    d = D[t]; imgs = list(d['official'])
    tot = [sum(d[k][i]['total'] for i in imgs) for k, _ in V]
    bars = ax.barh([l for _, l in V][::-1], tot[::-1], color=['#2a9d8f', '#e9c46a', '#e76f51'])
    for b, tt in zip(bars, tot[::-1]): ax.text(tt + max(tot) * 0.01, b.get_y() + b.get_height() / 2, f'{tt:.0f} s  ({tot[0] / tt:.2f}×)', va='center')
    ax.set_xlabel(f'{tn}: total seconds, 5 example images'); ax.set_xlim(0, max(tot) * 1.3); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig('docs/speedup.png', dpi=130)

# markdown tables for the README
for t, tn in T:
    d = D[t]; imgs = list(d['official'])
    print(f'\n### {tn}\n\n| Image | tokens | Official PyTorch-MPS | Metal parity | Metal fast |\n|---|---|---|---|---|')
    for i in imgs:
        print(f"| {short(i)} | {d['official'][i].get('tokens', d['parity'][i]['tokens'])} | " + ' | '.join(
            ('**%.1f**' if k == 'fast' else '%.1f') % d[k][i]['total'] for k, _ in V) + ' |')
    tot = [sum(d[k][i]['total'] for i in imgs) for k, _ in V]
    print(f'| **Sum** | | {tot[0]:.1f} | {tot[1]:.1f} ({tot[0] / tot[1]:.2f}×) | **{tot[2]:.1f} ({tot[0] / tot[2]:.2f}×)** |')
    print('\n| Stage (sum of 5) | Official | Metal parity | Metal fast |\n|---|---|---|---|')
    for s, sl, _ in S:
        v = [sum(d[k][i][s] for i in imgs) for k, _ in V]
        print(f'| {sl} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} ({v[0] / max(v[2], 1e-9):.1f}×) |')

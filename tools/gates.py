"""Tolerance gates, end to end: mesh distance of {official with switched attention backend, Metal parity, Metal fast}
to the official output, for every benchmark image.   ref/venv/bin/python tools/gates.py [512|1024c ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_mesh import compare

for t in sys.argv[1:] or ['512', '1024c']:
    ty = {'1024c': '1024_cascade'}.get(t, t)
    off = json.load(open(f'bench/official_{t}.json')); res = {}
    print(f'\n### {ty}\n\n| Image | official (alt attention) vs official | Metal parity vs official | Metal fast vs official |\n|---|---|---|---|')
    for img in off:
        stem = os.path.splitext(img)[0]; ref = f'bench/official_out/{stem}_{ty}.npz'; row = []; res[img] = {}
        for k, p in (('alt', f'bench/official_alt_out/{stem}_{ty}.npz'), ('parity', f'bench/parity_out/{stem}_{ty}.safetensors'),
                     ('fast', f'bench/fast_out/{stem}_{ty}.safetensors')):
            if not os.path.exists(p): row.append('n/a'); continue
            r = compare(ref, p); res[img][k] = r
            row.append('%.1fe-3 / %.1f%s' % (r['chamfer'] * 1e3, r['color_l1'], ' (identical mesh)' if k == 'parity' and r['verts_a'] == r['verts_b'] else ''))
        print(f"| {'T' if img == 'T.png' else img[:4]} | " + ' | '.join(row) + ' |', flush=True)
    json.dump(res, open(f'bench/gates_{t}.json', 'w'), indent=1)

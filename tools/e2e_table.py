"""Offline: tables from bench/e2e/*.json (tools: old fast = master build, new fast = this build with --glb).
  python3 tools/e2e_table.py 512 1024c"""
import json, sys
def load(p): return json.load(open(p))
for t in sys.argv[1:]:
    o = [load(f'bench/e2e/oldfast_{t}_r{r}.json') for r in (1, 2)]; n = [load(f'bench/e2e/newfast_glb_{t}_r{r}.json') for r in (1, 2)]
    keys = ['total', 'to_glb', 'export', 'total_glb']; st = ['glb.remesh', 'glb.simplify', 'glb.uv_unwrap', 'glb.bake', 'glb.textures', 'export']
    print(f'== {t}: image | old fast r1 r2 mean | new fast (image->mesh) r1 r2 mean | to_glb | export | image->GLB | GLB faces / MB')
    S = {}
    def add(k, v): S[k] = S.get(k, 0) + v
    for im in o[0]:
        om = [x[im]['total'] for x in o]; nm = [x[im]['total'] for x in n]
        m = {k: sum(x[im][k] for x in n) / 2 for k in keys + st}
        print(f"{im[:4]:5s} | {om[0]:7.2f} {om[1]:7.2f} {sum(om)/2:7.2f} | {nm[0]:7.2f} {nm[1]:7.2f} {sum(nm)/2:7.2f} | {m['to_glb']:5.2f} | {m['export']:4.2f} | {m['total_glb']:7.2f} | {n[1][im]['glb_faces']} / {n[1][im]['glb_bytes']/1e6:.1f}")
        for k, v in (('o1', om[0]), ('o2', om[1]), ('n1', nm[0]), ('n2', nm[1])): add(k, v)
        for k in keys + st: add(k, m[k])
    print(f"sum   | {S['o1']:7.2f} {S['o2']:7.2f} {(S['o1']+S['o2'])/2:7.2f} | {S['n1']:7.2f} {S['n2']:7.2f} {(S['n1']+S['n2'])/2:7.2f} | {S['to_glb']:5.2f} | {S['export']:4.2f} | {S['total_glb']:7.2f}")
    print('   new/old image->mesh: %+.2f%%' % (100 * ((S['n1'] + S['n2']) / (S['o1'] + S['o2']) - 1)))
    print('   to_glb stage sums:', ', '.join(f"{k.replace('glb.', '')} {S[k]:.2f}" for k in st))
    print('   peak footprint GB: old', max(x[im]['peak_footprint_gb'] for x in o for im in x), 'new', max(x[im]['peak_footprint_gb'] for x in n for im in x))
    stages = ['sparse_structure', 'shape_slat', 'tex_slat', 'decode_shape', 'decode_tex', 'cond']
    for s in stages: print(f"   {s}: old {sum(x[im][s] for x in o for im in x)/2:.2f} new {sum(x[im][s] for x in n for im in x)/2:.2f}")

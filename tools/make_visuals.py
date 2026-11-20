"""Offline: render benchmark outputs with build/render (Metal rasterizer; vertex base color, simple shading, same camera)
-> docs/renders_<type>.png (input | official | Metal parity | Metal fast) and docs/T_fast.gif (turntable).
  ref/venv/bin/python tools/make_visuals.py [512|1024c ...]"""
import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.numpy import save_file

tmp = tempfile.mkdtemp()
def render(mesh, tag, views=1, size=640, az0=-35, mode='color'):
    if mesh.endswith('.npz'):
        d = np.load(mesh); st = os.path.join(tmp, tag + '.safetensors')
        save_file({k: np.ascontiguousarray(d[k]) for k in ('vertices', 'faces', 'attrs')}, st); mesh = st
    pre = os.path.join(tmp, tag)
    subprocess.run(['build/render', '--mesh', mesh, '--out', pre, '--views', str(views), '--size', str(size), '--az0', str(az0), '--mode', mode],
                   check=True, stdout=subprocess.DEVNULL)
    return [Image.open(f'{pre}_{i:03d}.png').convert('RGB') for i in range(views)]

def font(sz):
    for p in ('/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNS.ttf'):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

D = 'ref/TRELLIS.2/assets/example_image'
for t in sys.argv[1:] or ['512', '1024c']:
    ty = {'1024c': '1024_cascade'}.get(t, t)
    B = {k: json.load(open(f'bench/{k}_{t}.json')) for k in ('official', 'parity', 'fast')}
    imgs = list(B['official']); S = 480; cols = ['input', 'Official PyTorch-MPS', 'Metal parity', 'Metal fast', 'Metal fast (geometry)']
    W = Image.new('RGB', (S * len(cols), 40 + S * len(imgs)), 'white'); dr = ImageDraw.Draw(W); f = font(22); fs = font(18)
    for j, c in enumerate(cols): dr.text((j * S + S // 2, 20), c, fill='black', font=f, anchor='mm')
    for i, img in enumerate(imgs):
        stem = os.path.splitext(img)[0]; y = 40 + i * S
        src = Image.open(f'{D}/{img}').convert('RGBA'); bg = Image.new('RGBA', src.size, 'white'); bg.alpha_composite(src)
        bg.thumbnail((S - 60, S - 60)); W.paste(bg.convert('RGB'), ((S - bg.width) // 2, y + (S - bg.height) // 2))
        cells = [(f'bench/official_out/{stem}_{ty}.npz', 'official', 'color'), (f'bench/parity_out/{stem}_{ty}.safetensors', 'parity', 'color'),
                 (f'bench/fast_out/{stem}_{ty}.safetensors', 'fast', 'color'), (f'bench/fast_out/{stem}_{ty}.safetensors', 'fast', 'clay')]
        for j, (p, k, mode) in enumerate(cells):
            W.paste(render(p, f'{k}{i}{mode}', size=S, mode=mode)[0], ((j + 1) * S, y))
            if mode == 'color': dr.text(((j + 1) * S + S // 2, y + S - 16), f"{B[k][img]['total']:.1f} s", fill='#444', font=fs, anchor='mm')
    W.save(f'docs/renders_{t}.png'); print('wrote', f'docs/renders_{t}.png')

t = '1024c' if os.path.exists('bench/fast_out/T_1024_cascade.safetensors') else '512'
fr = render(f"bench/fast_out/T_{ {'1024c': '1024_cascade'}.get(t, t)}.safetensors", 'tt', views=36, size=480, az0=0)
fr = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in fr]
fr[0].save('docs/T_fast.gif', save_all=True, append_images=fr[1:], duration=80, loop=0); print('wrote docs/T_fast.gif')

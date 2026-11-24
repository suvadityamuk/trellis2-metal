"""Offline: side-by-side renders of the GLBs written by build/t2 --glb, next to the pre-to_glb mesh and a reference GLB.
All cells are drawn by build/render (same camera and shading; GLBs with their textures).
  ref/venv/bin/python tools/make_glb_visuals.py TYPE MESH_DIR GLB_DIR REF_GLB_DIR OUT.png [REF_LABEL]
MESH_DIR holds <stem>_<type>.safetensors (build/t2 --out), GLB_DIR <stem>_<type>.glb (build/t2 --glb), REF_GLB_DIR
<stem[:8]>.glb or <stem>.glb (e.g. the ROCm port's out/eager), or - for none."""
import glob, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

ty, mesh_dir, glb_dir, ref_dir, out = sys.argv[1:6]
ref_label = sys.argv[6] if len(sys.argv) > 6 else 'reference GLB'
tmp = tempfile.mkdtemp()
def render(args, tag, size, mode='color'):
    pre = os.path.join(tmp, tag)
    subprocess.run(['build/render', *args, '--out', pre, '--views', '1', '--size', str(size), '--az0', '-35', '--mode', mode], check=True, stdout=subprocess.DEVNULL)
    return Image.open(f'{pre}_000.png').convert('RGB')
def font(sz):
    for p in ('/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNS.ttf'):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

D = 'ref/TRELLIS.2/assets/example_image'
imgs = sorted(os.listdir(D), key=lambda s: (s[0] == 'T', s)); imgs = [i for i in imgs if os.path.exists(f'{glb_dir}/{os.path.splitext(i)[0]}_{ty}.glb')]
S = 480; cols = ['input', 'Metal fast mesh (vertex colour)', 'Metal GLB (textured)', 'Metal GLB (geometry)', ref_label, ref_label + ' (geometry)'][:6 if ref_dir != '-' else 4]
W = Image.new('RGB', (S * len(cols), 40 + S * len(imgs)), 'white'); dr = ImageDraw.Draw(W); f = font(20)
for j, c in enumerate(cols): dr.text((j * S + S // 2, 20), c, fill='black', font=f, anchor='mm')
for i, img in enumerate(imgs):
    stem = os.path.splitext(img)[0]; y = 40 + i * S
    src = Image.open(f'{D}/{img}').convert('RGBA'); bg = Image.new('RGBA', src.size, 'white'); bg.alpha_composite(src)
    bg.thumbnail((S - 60, S - 60)); W.paste(bg.convert('RGB'), ((S - bg.width) // 2, y + (S - bg.height) // 2))
    g = f'{glb_dir}/{stem}_{ty}.glb'
    ref = next((p for p in (f'{ref_dir}/{stem[:8]}.glb', f'{ref_dir}/{stem}.glb') if os.path.exists(p)), None)
    cells = [(['--mesh', f'{mesh_dir}/{stem}_{ty}.safetensors'], 'color'), (['--glb', g], 'color'), (['--glb', g], 'clay')]
    if ref: cells += [(['--glb', ref], 'color'), (['--glb', ref], 'clay')]
    for j, (a, mode) in enumerate(cells): W.paste(render(a, f'c{i}_{j}', S, mode), ((j + 1) * S, y))
W.save(out); print('wrote', out)

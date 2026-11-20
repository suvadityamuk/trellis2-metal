"""Extracts the precompiled Metal shader libraries that PyTorch embeds in libtorch_cpu.dylib (one per
aten/src/ATen/native/mps/kernels/*.metal). The parity build loads these exact libraries so its pointwise /
attention / norm kernels are the same machine code torch-MPS runs. Offline tool; needs only the reference venv."""
import sys, struct, re, os, json
src, out = sys.argv[1], sys.argv[2]
d = open(src, 'rb').read()
os.makedirs(out, exist_ok=True)
idx = {}
for m in re.finditer(b'MTLB', d):
    o = m.start()
    size = struct.unpack_from('<Q', d, o + 0x10)[0]
    if not (1024 < size < 512 << 20) or o + size > len(d): continue
    blob = d[o:o + size]
    if blob[-4:] != b'ENDT' and b'ENDT' not in blob[-64:]: pass
    names = sorted(set(n.decode() for n in re.findall(rb'NAME..([A-Za-z_][A-Za-z0-9_]*)\x00', blob)))
    fn = f'lib_{len(idx):02d}.metallib'
    open(os.path.join(out, fn), 'wb').write(blob)
    idx[fn] = dict(offset=o, size=size, functions=names)
json.dump(idx, open(os.path.join(out, 'index.json'), 'w'), indent=0)
for k, v in idx.items(): print(k, v['size'], len(v['functions']), v['functions'][:4])

"""Writes refdump/manifest.txt: runtime dtype/shape of every parameter/buffer of every model as the official pipeline
holds them after loading (the native loader converts checkpoint dtypes to these)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, mps_ref
pipe = mps_ref.load_pipeline()
mods = dict(pipe.models); mods["dino"] = pipe.image_cond_model.model
with open(os.path.join(mps_ref.ROOT, "refdump", "manifest.txt"), "w") as f:
    for mn, m in mods.items():
        for k, v in m.state_dict().items():
            f.write(f"{mn}\t{k}\t{str(v.dtype).replace('torch.', '')}\t{','.join(map(str, v.shape))}\n")
print(pipe.image_cond_model.model.config)
print(type(pipe.image_cond_model.model))

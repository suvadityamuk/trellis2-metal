// o_voxel.postprocess.to_glb on Metal + CPU threads: mesh with sparse voxel attributes -> textured GLB.
// Same sequence and settings as the official example.py call (remesh=True, remesh_band=1, remesh_project=0):
//   fill small holes -> BVH of the input mesh -> narrow-band dual-contouring remesh -> simplify to decimation_target
//   -> chart clustering + xatlas UV atlas -> rasterize in UV space, snap each texel to the input surface and sample
//   the voxel attributes trilinearly -> TELEA inpaint of the empty texels -> GLB (y-up, WebP textures).
// example.py first calls mesh.simplify(16777216) because of an nvdiffrast limit; that step is not needed here.
#pragma once
#include "bake.h"
#include "gltf.h"
#include <map>

namespace glb {

struct GlbOpts {
  int64_t faces = 1000000;   // decimation_target
  int texture_size = 4096;
  float band = 1.f;          // remesh_band
  float hole_perimeter = 3e-2f;
  int xatlas_groups = 64; int64_t chart_max_faces = 1024; int xatlas_threads = 0;
  TexOpts tex;
  bool verbose = false;
};

struct GlbOut {
  std::map<std::string, double> t;  // remesh, simplify, uv_unwrap, bake, textures, export, total (+ sub-stage detail with a dot)
  HoleStats holes; RemeshStats remesh; SimplifyStats simp; UnwrapStats unwrap; TeleaStats telea; GlbStats file;
  int64_t in_faces = 0, remesh_faces = 0, simp_faces = 0, out_verts = 0, out_faces = 0, baked_texels = 0;
  // kept for tests / tolerance checks
  Mesh simplified; Arr<int32_t> vmap, faces; Arr<float> uv, nrm; Tensor base, mr, ids, lv;
  Arr<uint8_t> img0;  // planar [6, P] bake before inpainting (keep only)
};

// V [nv, 3], F [nf, 3]: mesh in [-0.5, 0.5]^3 (z up). coords [n, 3] int32 (x, y, z), attrs [n, 6] float32 (base rgb,
// metallic, roughness, alpha), on a res^3 grid over the same cube.
inline void to_glb(const float* V, const int32_t* F, int64_t nv, int64_t nf, const int32_t* coords, const float* attrs, int64_t nvox, int res,
                   const std::string& path, const GlbOpts& o, GlbOut& r, bool keep = false) {
  double T0 = now(), t0 = T0;
  auto lap = [&](const char* k) { double t = now(); r.t[k] = t - t0; t0 = t; if (o.verbose) fprintf(stderr, "[to_glb] %-10s %.3fs\n", k, r.t[k]); };
  r.in_faces = nf;
  tk::sync();

  // ---- remesh
  Mesh hf; r.holes = fill_holes(V, F, nv, nf, o.hole_perimeter, hf);
  r.t["remesh.fill_holes"] = now() - t0;
  LBVH bvh; bvh.build(hf.V.p, hf.F.p, hf.nf);
  r.t["remesh.bvh"] = bvh.t_build;
  Mesh m; r.remesh = remesh_narrow_band_dc(bvh, res, (float)(res + 3 * o.band) / (float)res, o.band, m);
  r.remesh_faces = m.nf;
  hf.V.clear(); hf.F.clear();
  lap("remesh");

  // ---- simplify
  simplify(m, o.faces, r.simp, 1e-2f, 1e-3f, o.verbose);
  lap("simplify");

  // ---- uv unwrap
  clean_mesh(m);
  r.simp_faces = m.nf;
  Arr<float> vn; vertex_normals(m.V.p, m.F.p, m.nv, m.nf, vn);
  uv_unwrap(m.V.p, m.F.p, m.nv, m.nf, r.vmap, r.faces, r.uv, r.unwrap, o.xatlas_groups, o.chart_max_faces, o.xatlas_threads);
  const int64_t NV = r.vmap.n, NF = r.faces.n / 3;
  r.out_verts = NV; r.out_faces = NF;
  lap("uv_unwrap");

  // ---- bake
  const int W = o.texture_size, H = o.texture_size; const int64_t P = (int64_t)W * H;
  Tensor tpos = gbuf<float>(NV * 3), tuv = gbuf<float>(NV * 2), tf = gbuf<int32_t>(NF * 3);
  {
    float* p = tpos.ptr<float>();
    par_for(NV, 1 << 16, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) memcpy(p + 3 * i, m.V.p + 3 * (int64_t)r.vmap[i], 12); });
    memcpy(tuv.ptr<float>(), r.uv.p, NV * 8); memcpy(tf.ptr<int32_t>(), r.faces.p, NF * 12);
  }
  Tensor img, lv;
  {
    VoxGrid vg; vg.build(coords, attrs, nvox, res);
    r.t["bake.voxgrid"] = now() - t0;
    Tensor ids = rasterize_uv(tuv, tf, NF, W, H);
    bake_texels(bvh, ids, tf, tpos, tuv, vg, W, H, img, lv);
    tk::sync();
    if (keep) r.ids = ids;
  }
  bvh = LBVH();
  lap("bake");

  // ---- textures: inpaint + pack
  { const int32_t* l = lv.ptr<int32_t>(); const int C = 64; int64_t cnt[C] = {0};
    par_chunks(P, C, [&](int c, int64_t a, int64_t b) { int64_t s = 0; for (int64_t i = a; i < b; i++) s += l[i] == 0; cnt[c] = s; });
    for (int c = 0; c < C; c++) r.baked_texels += cnt[c]; }
  if (keep) { r.img0.alloc(P * 6); memcpy(r.img0.p, img.ptr<uint8_t>(), (size_t)P * 6); }
  telea(lv, img, W, H, r.telea);
  if (keep) r.lv = lv;
  r.t["textures.levels"] = r.telea.t_levels; r.t["textures.fill"] = r.telea.t_fill;
  Tensor base, mr; pack_textures(img, P, base, mr);
  lap("textures");

  // ---- export: glTF axes (y up): (x, y, z) -> (x, z, -y)
  Arr<float> pos(NV * 3); r.nrm.alloc(NV * 3);
  {
    const float* p = tpos.ptr<float>();
    par_for(NV, 1 << 16, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; i++) {
        pos[3 * i] = p[3 * i]; pos[3 * i + 1] = p[3 * i + 2]; pos[3 * i + 2] = -p[3 * i + 1];
        const float* n = vn.p + 3 * (int64_t)r.vmap[i];
        r.nrm[3 * i] = n[0]; r.nrm[3 * i + 1] = n[2]; r.nrm[3 * i + 2] = -n[1];
      }
    });
  }
  if (!path.empty()) write_glb(path, pos.p, r.nrm.p, r.uv.p, NV, r.faces.p, NF, base.ptr<uint8_t>(), mr.ptr<uint8_t>(), W, H, o.tex, r.file);
  r.t["export.encode"] = r.file.t_encode;
  lap("export");
  r.t["total"] = now() - T0;
  if (keep) { r.simplified = std::move(m); r.base = base; r.mr = mr; }
}

}  // namespace glb

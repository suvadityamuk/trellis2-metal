// FlexiDualGridVaeDecoder (ref/TRELLIS.2/trellis2/models/sc_vaes/fdg_vae.py, eval path) + flexible_dual_grid_to_mesh
// (ref/TRELLIS.2/o-voxel/o_voxel/convert/flexible_dual_grid.py, train=False, split_weight given) — parity build.
// The mesh extraction runs on the CPU in float32, exactly like the reference (tools/mps_ref.py fdg_cpu).
#pragma once
#include "sparse_ops.h"

namespace t2 {


struct MeshOut {
  Tensor vertices;  // F32 [N,3]
  Tensor faces;     // I32 [2L,3]
};

// coords I32 [N,3] (or [N,4] with a leading batch column), dual_vertices F32 [N,3], intersected BOOL/U8 [N,3],
// split_weight F32 [N,1]. aabb = [[-0.5]*3, [0.5]*3], grid_size = resolution.
inline MeshOut flexible_dual_grid_to_mesh(const Tensor& coords_, const Tensor& dual_vertices_, const Tensor& intersected_,
                                          const Tensor& split_weight_, int grid_size) {
  tk::sync();
  Tensor coords = contiguous_cpu(coords_), dv = contiguous_cpu(dual_vertices_), inter = contiguous_cpu(intersected_),
         sw = contiguous_cpu(split_weight_);
  const int64_t N = coords.shape[0]; const int cw = (int)coords.shape[1], c0 = cw - 3;
  if (dv.dt != tk::F32 || sw.dt != tk::F32 || coords.dt != tk::I32 || inter.esize() != 1) { fprintf(stderr, "flexible_dual_grid_to_mesh: bad dtypes\n"); abort(); }
  const int32_t* c = coords.ptr<int32_t>(); const float* d = dv.ptr<float>(); const uint8_t* fl = inter.raw(); const float* w = sw.ptr<float>();
  // voxel_size = (aabb[1] - aabb[0]) / grid_size  (float32 / int32 -> float32)
  const float aabb0 = -0.5f, aabb1 = 0.5f;
  const float voxel = (aabb1 - aabb0) / (float)grid_size;
  MeshOut r; r.vertices = Tensor(tk::F32, {N, 3});
  float* mv = r.vertices.ptr<float>();
  // mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]
  parallel_for(N, 1 << 15, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++)
      for (int k = 0; k < 3; k++) mv[i * 3 + k] = ((float)c[i * cw + c0 + k] + d[i * 3 + k]) * voxel + aabb0;
  });
  // hashmap: voxel -> row
  const int64_t G = grid_size;
  int lg = 4; while (((int64_t)1 << lg) < 2 * N) lg++;
  const uint64_t cap = (uint64_t)1 << lg, mask = cap - 1; const int sh = 64 - lg;
  std::vector<int64_t> hk(cap, -1); std::vector<int32_t> hv(cap);
  for (int64_t i = 0; i < N; i++) {
    const int64_t key = ((int64_t)c[i * cw + c0] * G + c[i * cw + c0 + 1]) * G + c[i * cw + c0 + 2];
    uint64_t h = ((uint64_t)key * 0x9E3779B97F4A7C15ull) >> sh;
    while (hk[h] >= 0) h = (h + 1) & mask;
    hk[h] = key; hv[h] = (int32_t)i;
  }
  static const int off[3][4][3] = {
      {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},  // x-axis
      {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},  // y-axis
      {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},  // z-axis
  };
  // pass 1: per (voxel, axis) with intersected flag, look up the 4 edge-neighbor voxels; valid iff all 4 exist
  std::vector<int32_t> quad((size_t)N * 12); std::vector<uint8_t> valid((size_t)N * 3);
  parallel_for(N, 4096, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++)
      for (int ax = 0; ax < 3; ax++) {
        valid[i * 3 + ax] = 0;
        if (!fl[i * 3 + ax]) continue;
        bool ok = true;
        for (int q = 0; q < 4 && ok; q++) {
          const int64_t x = c[i * cw + c0] + off[ax][q][0], y = c[i * cw + c0 + 1] + off[ax][q][1], z = c[i * cw + c0 + 2] + off[ax][q][2];
          if (x < 0 || y < 0 || z < 0 || x >= G || y >= G || z >= G) { ok = false; break; }
          const int64_t key = (x * G + y) * G + z;
          uint64_t h = ((uint64_t)key * 0x9E3779B97F4A7C15ull) >> sh;
          int32_t f = -1;
          while (hk[h] >= 0) { if (hk[h] == key) { f = hv[h]; break; } h = (h + 1) & mask; }
          if (f < 0) ok = false; else quad[(i * 3 + ax) * 4 + q] = f;
        }
        valid[i * 3 + ax] = ok;
      }
  });
  std::vector<int64_t> pos((size_t)N + 1); pos[0] = 0;
  for (int64_t i = 0; i < N; i++) pos[i + 1] = pos[i] + valid[i * 3] + valid[i * 3 + 1] + valid[i * 3 + 2];
  const int64_t L = pos[N];
  r.faces = Tensor(tk::I32, {2 * L, 3});
  int32_t* tf = r.faces.ptr<int32_t>();
  parallel_for(N, 4096, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      int64_t o = pos[i];
      for (int ax = 0; ax < 3; ax++) {
        if (!valid[i * 3 + ax]) continue;
        const int32_t* q = &quad[(i * 3 + ax) * 4];
        const float w02 = w[q[0]] * w[q[2]], w13 = w[q[1]] * w[q[3]];
        static const int s1[6] = {0, 1, 2, 0, 2, 3}, s2[6] = {0, 1, 3, 3, 1, 2};
        const int* s = (w02 > w13) ? s1 : s2;
        for (int k = 0; k < 6; k++) tf[o * 6 + k] = q[s[k]];
        o++;
      }
    }
  });
  return r;
}

struct ShapeDecoder {
  SparseUnetVaeDecoder dec;
  float voxel_margin = 0.5f;
  int resolution = 256;
  void load(const std::string& json, const std::string& st) {
    dec.load(json, st, 7);
    Json cfg = loadJson(json); const Json& a = cfg["args"];
    if (a.has("resolution")) resolution = (int)a["resolution"].i();
    if (a.has("voxel_margin")) voxel_margin = (float)a["voxel_margin"].d();
  }
  void set_resolution(int r) { resolution = r; }

  struct Out {
    MeshOut mesh;
    SparseT h;                  // raw output_layer result [N,7] f32 + coords [N,4]
    std::vector<SparseT> subs;  // subdivision logits per level (guide for the texture decoder)
  };
  // decode_shape_slat: decoder(slat, return_subs=True). Call set_resolution() first (pipeline does).
  Out run(const SparseT& slat, SparseCache* cache = nullptr) const {
    Out r;
    {
      SparseUnetVaeDecoder::Out d = dec.forward(slat, nullptr, cache);
      r.h = d.h; r.subs = std::move(d.subs);
    }
    const Tensor& f = r.h.feats;
    // vertices = (1 + 2 * voxel_margin) * sigmoid(h[..., 0:3]) - voxel_margin
    Tensor v = un("sigmoid", f.slice(1, 0, 3));
    v = mul(v, (double)(1 + 2 * voxel_margin));
    v = sub(v, (double)voxel_margin);
    // quad_lerp = softplus(h[..., 6:7])
    Tensor q = softplus(f.slice(1, 6, 7));
    tk::sync();
    // intersected = h[..., 3:6] > 0   (comparison only; evaluated on the CPU)
    const int64_t N = f.shape[0];
    Tensor inter(tk::U8, {N, 3});
    {
      Tensor fc = contiguous_cpu(f); const float* p = fc.ptr<float>(); uint8_t* o = inter.raw();
      const int64_t C = f.shape[1];
      parallel_for(N, 1 << 15, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; i++) for (int k = 0; k < 3; k++) o[i * 3 + k] = p[i * C + 3 + k] > 0.0f;
      });
    }
    r.mesh = flexible_dual_grid_to_mesh(r.h.coords, v, inter, q, resolution);
    return r;
  }
};

}  // namespace t2

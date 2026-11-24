// Texture SLat decoder: SparseUnetVaeDecoder (pred_subdiv = false, guided by the shape decoder's subdivisions) and
// the decode_latent assembly (ref/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py decode_tex_slat / decode_latent).
#pragma once
#include "shape_decoder.h"

namespace t2 {

struct TexDecoder {
  SparseUnetVaeDecoder dec;
  void load(const std::string& json, const std::string& st) { dec.load(json, st, 6); }
  // decode_tex_slat: decoder(slat, guide_subs=subs) * 0.5 + 0.5  -> feats f32 [N,6], coords [N,4]
  SparseT run(const SparseT& slat, const std::vector<SparseT>& subs, SparseCache* cache = nullptr) const {
    SparseT h = dec.forward(slat, &subs, cache).h;
    Tensor f = mul(h.feats, 0.5);
    f = add(f, 0.5);
    return SparseT{f, h.coords};
  }
};

// decode_latent: MeshWithVoxel(vertices, faces, coords = texvox.coords[:, 1:], attrs = texvox.feats)
struct MeshWithVoxel {
  Tensor vertices, faces;  // F32 [V,3], I32 [F,3]
  Tensor coords, attrs;    // I32 [N,3], F32 [N,6]
};
inline MeshWithVoxel make_mesh_with_voxel(const MeshOut& m, const SparseT& texvox) {
  tk::sync();
  MeshWithVoxel r; r.vertices = m.vertices; r.faces = m.faces; r.attrs = texvox.feats;
  r.coords = contiguous_cpu(texvox.coords.slice(1, 1, 4));
  return r;
}

}  // namespace t2

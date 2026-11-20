// Bitwise parity of the SLat decoders (shape -> mesh + subs, texture -> voxel attrs) against the reference dump.
#include "../src/models/tex_decoder.h"
#include <chrono>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string root = ".", dump = "refdump/T_512.safetensors", stage = "all", taps;
    int resolution = 512;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--root") root = argv[++i]; else if (a == "--dump") dump = argv[++i];
      else if (a == "--stage") stage = argv[++i]; else if (a == "--taps") taps = argv[++i];
      else if (a == "--res") resolution = atoi(argv[++i]);
    }
    tk::init((root + "/build/torchlibs/lib_01.metallib").c_str());
    initOps();
    STFile ref(root + "/" + dump);
    std::unique_ptr<STFile> tapf; if (!taps.empty()) tapf.reset(new STFile(taps));
    const std::string ck = root + "/models/trellis2/ckpts/";
    bool ok = true;
    auto gb = [](size_t b) { return (double)b / (1 << 30); };
    auto tapper = [&](const char* model) {
      return [&, model](const std::string& name, const Tensor& h) {
        if (!tapf) return;
        std::string k = std::string("tap.") + model + "." + name + ".0";
        if (tapf->has(k + ".feats")) ok &= cmp(k, h, tapf->get(k + ".feats"));
        else if (tapf->has(k)) ok &= cmp(k, h, tapf->get(k));
      };
    };
    std::vector<SparseT> subs; MeshOut mesh;
    SparseCache cache;  // neighbor maps shared between both decoders (same coords at every level)
    if (stage == "shape" || stage == "all") {
      double t0 = now();
      ShapeDecoder m; m.load(ck + "shape_dec_next_dc_f16c32_fp16.json", ck + "shape_dec_next_dc_f16c32_fp16.safetensors");
      m.set_resolution(resolution);
      if (tapf) m.dec.tap = tapper("shape_slat_decoder");
      tk::sync(); printf("shape decoder loaded in %.2fs\n", now() - t0);
      SparseT x{ref.get("shape.feats"), ref.get("shape.coords")};
      t0 = now();
      ShapeDecoder::Out o = m.run(x, &cache);
      tk::sync();
      printf("decode shape: %.2fs (N=%lld, V=%lld, F=%lld)  pool peak %.2f GB\n", now() - t0, (long long)o.h.feats.shape[0],
             (long long)o.mesh.vertices.shape[0], (long long)o.mesh.faces.shape[0], gb(pool().peak));
      ok &= cmp("shapedec_h.0.feats", o.h.feats, ref.get("shapedec_h.0.feats"));
      ok &= cmp("shapedec_h.0.coords", o.h.coords, ref.get("shapedec_h.0.coords"));
      for (size_t i = 0; i < o.subs.size(); i++) {
        std::string k = "subs." + std::to_string(i);
        ok &= cmp(k + ".feats", o.subs[i].feats, ref.get(k + ".feats"));
        ok &= cmp(k + ".coords", o.subs[i].coords, ref.get(k + ".coords"));
      }
      ok &= cmp("mesh.vertices", o.mesh.vertices, ref.get("mesh.vertices"));
      ok &= cmp("mesh.faces", o.mesh.faces, ref.get("mesh.faces"));
      subs = o.subs; mesh = o.mesh;
    }
    pool().trim();
    if (stage == "tex" || stage == "all") {
      // stage inputs from the dump
      subs.clear();
      for (int i = 0; i < 4; i++) { std::string k = "subs." + std::to_string(i); subs.push_back(SparseT{ref.get(k + ".feats"), ref.get(k + ".coords")}); }
      mesh.vertices = ref.get("mesh.vertices"); mesh.faces = ref.get("mesh.faces");
      double t0 = now();
      TexDecoder m; m.load(ck + "tex_dec_next_dc_f16c32_fp16.json", ck + "tex_dec_next_dc_f16c32_fp16.safetensors");
      if (tapf) m.dec.tap = tapper("tex_slat_decoder");
      tk::sync(); printf("tex decoder loaded in %.2fs\n", now() - t0);
      SparseT x{ref.get("tex.feats"), ref.get("tex.coords")};
      t0 = now();
      SparseT v = m.run(x, subs, &cache);
      tk::sync();
      printf("decode tex: %.2fs (N=%lld)  pool peak %.2f GB\n", now() - t0, (long long)v.feats.shape[0], gb(pool().peak));
      ok &= cmp("texvox.feats", v.feats, ref.get("texvox.feats"));
      ok &= cmp("texvox.coords", v.coords, ref.get("texvox.coords"));
      MeshWithVoxel out = make_mesh_with_voxel(mesh, v);
      ok &= cmp("out.vertices", out.vertices, ref.get("out.vertices"));
      ok &= cmp("out.faces", out.faces, ref.get("out.faces"));
      ok &= cmp("out.attrs", out.attrs, ref.get("out.attrs"));
      ok &= cmp("out.coords", out.coords, ref.get("out.coords"));
    }
    printf("pool: peak %.2f GB, live %.2f GB\n", gb(pool().peak), gb(pool().live));
    printf(ok ? "PARITY OK\n" : "PARITY FAILED\n");
    return ok ? 0 : 1;
  }
}

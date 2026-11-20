// Bitwise parity checks of the "front" stages against the reference dump:
//   image   load + preprocess_image (src/models/image.h)            -> `image`
//   cond    DINOv3 get_cond (src/models/dino.h), fed from the dump   -> `cond.0.cond`, `cond.0.neg_cond`
//           and chained: example image file -> preprocess -> cond
//   ssdec   sparse-structure decoder + coords (src/models/ss_decoder.h), fed from `ss_s.0.x.11` -> `ssdec.0`, `ss`
//   extra   (--extra <dir>) raw reference files written by a python script: <dir>/imgN.pre (preprocess output),
//           imgN.r512 / imgN.r1024 (PIL resize), cond_<res>_N.safetensors (get_cond at other resolutions / images)
// usage: build/parity_front [--root .] [--dump refdump/T_512.safetensors] [--stage all|image|cond|ssdec|extra]
//                           [--image <path>] [--extra <dir>]
#include "../src/models/dino.h"
#include "../src/models/ss_decoder.h"
#include <chrono>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static Tensor imgTensor(const Image8& im) { return from_cpu(tk::U8, {im.h, im.w, im.c}, im.d.data()); }
static Image8 tensorImg(const Tensor& t_) {
  tk::sync(); Tensor t = contiguous_cpu(t_);
  Image8 im; im.h = (int)t.shape[0]; im.w = (int)t.shape[1]; im.c = (int)t.shape[2]; im.d.assign(t.raw(), t.raw() + t.nbytes()); return im;
}
static bool readFile(const std::string& p, std::vector<uint8_t>& v) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) return false;
  fseek(f, 0, SEEK_END); v.resize(ftell(f)); fseek(f, 0, SEEK_SET);
  bool ok = fread(v.data(), 1, v.size(), f) == v.size(); fclose(f); return ok;
}
static bool cmpRaw(const std::string& name, const Image8& im, const std::string& path) {
  std::vector<uint8_t> v;
  if (!readFile(path, v)) { printf("  %-28s (no reference file %s)\n", name.c_str(), path.c_str()); return true; }
  if (v.size() != im.d.size()) { printf("  %-28s SIZE MISMATCH %zu vs %zu (%dx%d)\n", name.c_str(), im.d.size(), v.size(), im.w, im.h); return false; }
  return cmp(name, imgTensor(im), from_cpu(tk::U8, {im.h, im.w, im.c}, v.data()));
}

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string root = ".", dump = "refdump/T_512.safetensors", stage = "all", image = "ref/TRELLIS.2/assets/example_image/T.png", extra;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--root") root = argv[++i]; else if (a == "--dump") dump = argv[++i];
      else if (a == "--stage") stage = argv[++i]; else if (a == "--image") image = argv[++i];
      else if (a == "--extra") extra = argv[++i];
    }
    auto on = [&](const char* s) { return stage == "all" || stage == s; };
    tk::init((root + "/build/torchlibs/lib_01.metallib").c_str());
    initOps();
    STFile ref(root + "/" + dump);
    bool ok = true;
    const char* ex[5] = {"T.png", "0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp",
                         "0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp",
                         "0f168a4b1b6e96c72e9627c97a212c27a4572250ff58e25703b9d0c2bc74191a.webp",
                         "130c2b18f1651a70f8aa15b2c99f8dba29bb943044d92871f9223bd3e989e8b1.webp"};

    if (on("image")) {
      double t0 = now();
      Image8 raw = load_image(root + "/" + image);
      double t1 = now();
      Image8 pre = preprocess_image(raw);
      printf("image: decode %.3fs, preprocess %.3fs (%dx%d -> %dx%d)\n", t1 - t0, now() - t1, raw.w, raw.h, pre.w, pre.h);
      ok &= cmp("image", imgTensor(pre), ref.get("image"));
    }
    if (on("extra") && !extra.empty()) {
      for (int i = 0; i < 5; i++) {
        std::string n = std::to_string(i);
        Image8 raw = load_image(root + "/ref/TRELLIS.2/assets/example_image/" + ex[i]);
        ok &= cmpRaw("img" + n + " decode", raw, extra + "/img" + n + ".rgba");
        Image8 pre = preprocess_image(raw);
        ok &= cmpRaw("img" + n + " preprocess", pre, extra + "/img" + n + ".pre");
        ok &= cmpRaw("img" + n + " resize512", resize_lanczos(pre, 512, 512), extra + "/img" + n + ".r512");
        ok &= cmpRaw("img" + n + " resize1024", resize_lanczos(pre, 1024, 1024), extra + "/img" + n + ".r1024");
      }
    }
    if (on("cond") || (on("extra") && !extra.empty())) {
      double t0 = now();
      Dino dino; dino.load(root + "/models/dino/config.json", root + "/models/dino/model.safetensors");
      tk::sync();
      printf("dino loaded in %.2fs\n", now() - t0);
      if (on("cond")) {
        Image8 img = tensorImg(ref.get("image"));
        for (int it = 0; it < 2; it++) {  // first run warms up pipelines / MPSGraph
          t0 = now();
          Tensor t = Dino::imageToTensor(img, 512);
          double t1 = now();
          Tensor cond = dino.features(t), neg = Dino::negCond(cond); tk::sync();
          printf("get_cond(512) %s: cpu resize+to-float %.3fs, gpu %.3fs, total %.3fs\n", it ? "warm" : "cold", t1 - t0, now() - t1, now() - t0);
          if (it) { ok &= cmp("cond.0.cond", cond, ref.get("cond.0.cond")); ok &= cmp("cond.0.neg_cond", neg, ref.get("cond.0.neg_cond")); }
        }
        // chained: file -> preprocess -> cond
        t0 = now();
        Tensor cond = dino(preprocess_image(load_image(root + "/" + image)), 512); tk::sync();
        printf("chained image->cond: %.3fs\n", now() - t0);
        ok &= cmp("chained cond.0.cond", cond, ref.get("cond.0.cond"));
      }
      if (on("extra") && !extra.empty()) {
        for (int res : {512, 1024}) for (int i = 0; i < 5; i++) {
          std::string p = extra + "/cond_" + std::to_string(res) + "_" + std::to_string(i) + ".safetensors";
          if (access(p.c_str(), R_OK)) continue;
          STFile cf(p);
          t0 = now();
          Tensor cond = dino(preprocess_image(load_image(root + "/ref/TRELLIS.2/assets/example_image/" + ex[i])), res); tk::sync();
          printf("get_cond(%d) img%d: %.3fs\n", res, i, now() - t0);
          ok &= cmp("img" + std::to_string(i) + " cond@" + std::to_string(res), cond, cf.get("cond"));
        }
      }
    }
    if (on("ssdec")) {
      double t0 = now();
      SSDecoder d; d.load(root + "/models/ss_dec/ss_dec_conv3d_16l8_fp16.json", root + "/models/ss_dec/ss_dec_conv3d_16l8_fp16.safetensors");
      tk::sync();
      printf("ss decoder loaded in %.2fs\n", now() - t0);
      for (int it = 0; it < 2; it++) {
        t0 = now();
        Tensor lg; Tensor c = d.run(ref.get("ss_s.0.x.11"), 32, &lg); tk::sync();
        printf("ss decode + coords %s: %.3fs\n", it ? "warm" : "cold", now() - t0);
        if (it) { ok &= cmp("ssdec.0", lg, ref.get("ssdec.0")); ok &= cmp("ss", c, ref.get("ss")); }
      }
    }
    printf(ok ? "PARITY OK\n" : "PARITY FAILED\n");
    return ok ? 0 : 1;
  }
}

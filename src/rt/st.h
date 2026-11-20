// safetensors reader (mmap) + a tiny JSON parser (also used for pipeline.json / model configs).
#pragma once
#include "tensor.h"
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace t2 {

// ---- JSON -----------------------------------------------------------------------------------------------------------
struct Json {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  double num = 0; bool b = false; std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;
  const Json* find(const std::string& k) const { for (auto& kv : obj) if (kv.first == k) return &kv.second; return nullptr; }
  const Json& operator[](const std::string& k) const {
    auto p = find(k); if (!p) { fprintf(stderr, "json: missing key %s\n", k.c_str()); abort(); } return *p;
  }
  const Json& operator[](size_t i) const { return arr[i]; }
  bool has(const std::string& k) const { return find(k) != nullptr; }
  double d() const { return num; }
  int64_t i() const { return (int64_t)num; }
  size_t size() const { return t == Arr ? arr.size() : obj.size(); }
};
struct JsonParser {
  const char* p; const char* e;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++; }
  [[noreturn]] void fail(const char* m) { fprintf(stderr, "json parse error: %s\n", m); abort(); }
  std::string pstr() {
    std::string s; p++;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        p++;
        switch (*p) {
          case 'n': s += '\n'; break; case 't': s += '\t'; break; case 'r': s += '\r'; break;
          case 'b': s += '\b'; break; case 'f': s += '\f'; break;
          case 'u': { unsigned c = (unsigned)strtoul(std::string(p + 1, 4).c_str(), nullptr, 16); p += 4;
            if (c < 0x80) s += (char)c; else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
            else { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); } break; }
          default: s += *p;
        }
        p++;
      } else s += *p++;
    }
    p++; return s;
  }
  Json val() {
    ws(); Json j;
    if (p >= e) fail("eof");
    if (*p == '{') {
      j.t = Json::Obj; p++; ws();
      if (*p == '}') { p++; return j; }
      for (;;) { ws(); if (*p != '"') fail("key"); std::string k = pstr(); ws(); if (*p != ':') fail(":"); p++;
        j.obj.emplace_back(k, val()); ws(); if (*p == ',') { p++; continue; } if (*p == '}') { p++; break; } fail("obj"); }
    } else if (*p == '[') {
      j.t = Json::Arr; p++; ws();
      if (*p == ']') { p++; return j; }
      for (;;) { j.arr.push_back(val()); ws(); if (*p == ',') { p++; continue; } if (*p == ']') { p++; break; } fail("arr"); }
    } else if (*p == '"') { j.t = Json::Str; j.str = pstr(); }
    else if (!strncmp(p, "true", 4)) { j.t = Json::Bool; j.b = true; p += 4; }
    else if (!strncmp(p, "false", 5)) { j.t = Json::Bool; j.b = false; p += 5; }
    else if (!strncmp(p, "null", 4)) { p += 4; }
    else { char* q; j.t = Json::Num; j.num = strtod(p, &q); if (q == p) fail("num"); p = q; }
    return j;
  }
};
inline Json parseJson(const char* s, size_t n) { JsonParser P{s, s + n}; return P.val(); }
inline Json loadJson(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  std::string s; char buf[65536]; size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  fclose(f); return parseJson(s.data(), s.size());
}

// ---- safetensors ------------------------------------------------------------------------------------------------------
struct STEntry { DT dt; Shape shape; const uint8_t* data; size_t nbytes; };

struct STFile {
  std::string path; int fd = -1; uint8_t* map = nullptr; size_t len = 0;
  std::unordered_map<std::string, STEntry> ents;
  std::vector<std::string> order;
  STFile() {}
  STFile(const STFile&) = delete;
  explicit STFile(const std::string& p) { open(p); }
  ~STFile() { close(); }
  void close() { if (map) munmap(map, len); if (fd >= 0) ::close(fd); map = nullptr; fd = -1; ents.clear(); }
  static DT parseDT(const std::string& s) {
    if (s == "F32") return tk::F32; if (s == "F16") return tk::F16; if (s == "BF16") return tk::BF16;
    if (s == "I32") return tk::I32; if (s == "I64") return tk::I64; if (s == "U8") return tk::U8;
    if (s == "BOOL") return tk::BOOL; if (s == "F64") return tk::F64; if (s == "I16") return tk::I16; if (s == "I8") return tk::I8;
    fprintf(stderr, "safetensors: unsupported dtype %s\n", s.c_str()); abort();
  }
  void open(const std::string& p) {
    path = p; fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    struct stat sb; fstat(fd, &sb); len = sb.st_size;
    map = (uint8_t*)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { fprintf(stderr, "mmap failed %s\n", p.c_str()); exit(1); }
    uint64_t hn; memcpy(&hn, map, 8);
    Json h = parseJson((const char*)map + 8, hn);
    const uint8_t* base = map + 8 + hn;
    for (auto& kv : h.obj) {
      if (kv.first == "__metadata__") continue;
      STEntry e; e.dt = parseDT(kv.second["dtype"].str);
      for (auto& d : kv.second["shape"].arr) e.shape.push_back(d.i());
      int64_t a = kv.second["data_offsets"][0].i(), b = kv.second["data_offsets"][1].i();
      e.data = base + a; e.nbytes = b - a;
      ents[kv.first] = e; order.push_back(kv.first);
    }
  }
  bool has(const std::string& k) const { return ents.count(k) != 0; }
  const STEntry& ent(const std::string& k) const {
    auto it = ents.find(k);
    if (it == ents.end()) { fprintf(stderr, "safetensors %s: missing tensor %s\n", path.c_str(), k.c_str()); abort(); }
    return it->second;
  }
  // Loads tensor k as dtype `want` (exact widening F16/BF16 -> F32 only; same dtype = plain copy). The pooled buffer
  // is written on the CPU, so this syncs first.
  Tensor get(const std::string& k, DT want) const {
    const STEntry& e = ent(k);
    tk::sync();
    Tensor r(want, e.shape);
    int64_t n = r.numel();
    if (e.dt == want) { memcpy(r.raw(), e.data, e.nbytes); }
    else if (e.dt == tk::BF16 && want == tk::F32) {
      const uint16_t* s = (const uint16_t*)e.data; uint32_t* d = r.ptr<uint32_t>();
      for (int64_t i = 0; i < n; i++) d[i] = (uint32_t)s[i] << 16;
    } else if (e.dt == tk::F16 && want == tk::F32) {
      const __fp16* s = (const __fp16*)e.data; float* d = r.ptr<float>();
      for (int64_t i = 0; i < n; i++) d[i] = (float)s[i];
    } else { fprintf(stderr, "safetensors: %s: unsupported conversion %s -> %s\n", k.c_str(), tk::dtName(e.dt), tk::dtName(want)); abort(); }
    return r;
  }
  Tensor get(const std::string& k) const { return get(k, ent(k).dt); }
};

// writer (debug dumps / parity artifacts)
inline void saveSafetensors(const std::string& path, const std::vector<std::pair<std::string, Tensor>>& ts) {
  tk::sync();
  std::string h = "{"; size_t off = 0; bool first = true;
  std::vector<Tensor> cs;
  for (auto& kv : ts) {
    Tensor c = contiguous_cpu(kv.second); cs.push_back(c);
    const char* dn = c.dt == tk::F32 ? "F32" : c.dt == tk::F16 ? "F16" : c.dt == tk::BF16 ? "BF16" : c.dt == tk::I32 ? "I32" :
                     c.dt == tk::I64 ? "I64" : c.dt == tk::U8 ? "U8" : c.dt == tk::BOOL ? "BOOL" : "F64";
    if (!first) h += ","; first = false;
    h += "\"" + kv.first + "\":{\"dtype\":\"" + dn + "\",\"shape\":[";
    for (size_t i = 0; i < c.shape.size(); i++) h += (i ? "," : "") + std::to_string(c.shape[i]);
    h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + c.nbytes()) + "]}";
    off += c.nbytes();
  }
  h += "}"; while (h.size() % 8) h += ' ';
  FILE* f = fopen(path.c_str(), "wb"); uint64_t hn = h.size();
  fwrite(&hn, 8, 1, f); fwrite(h.data(), 1, h.size(), f);
  for (auto& c : cs) fwrite(c.raw(), 1, c.nbytes(), f);
  fclose(f);
}

}  // namespace t2

CXX      := clang++
CXXFLAGS := -std=c++20 -O2 -fobjc-arc -ffp-contract=off -Wno-deprecated-declarations -Isrc -Isrc/torchk -Isrc/rt
FRAMEWORKS := -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
              -framework Foundation -framework IOKit -framework CoreGraphics -framework ImageIO -framework Accelerate
HDRS := $(wildcard src/*/*.h)

all: build/parity

build/base.o: src/torchk/base.mm src/torchk/base.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%: apps/%.mm build/base.o $(HDRS)
	$(CXX) $(CXXFLAGS) $< build/base.o $(FRAMEWORKS) -o $@

# to_glb (src/glb): vendored xatlas (UV parameterization) and libwebp encoder (GLB textures), built from source
WEBP_SRCS := $(wildcard third_party/libwebp/src/enc/*.c third_party/libwebp/src/dsp/*.c third_party/libwebp/src/utils/*.c third_party/libwebp/sharpyuv/*.c)
WEBP_OBJS := $(patsubst third_party/libwebp/%.c,build/webp/%.o,$(WEBP_SRCS))
build/webp/%.o: third_party/libwebp/%.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -DNDEBUG -Ithird_party/libwebp -c $< -o $@
build/libwebp.a: $(WEBP_OBJS)
	rm -f $@ && ar rcs $@ $^
build/xatlas.o: third_party/xatlas/xatlas.cpp third_party/xatlas/xatlas.h
	@mkdir -p build
	$(CXX) -std=c++17 -O2 -DNDEBUG -Wno-deprecated-declarations -c $< -o $@
GLB_LIBS := build/xatlas.o build/libwebp.a
build/t2 build/glb build/glb_test: build/%: apps/%.mm build/base.o $(GLB_LIBS) $(HDRS)
	$(CXX) $(CXXFLAGS) $< build/base.o $(GLB_LIBS) $(FRAMEWORKS) -o $@

clean:
	rm -rf build/*.o build/parity build/t2

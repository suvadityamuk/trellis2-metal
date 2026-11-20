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

clean:
	rm -rf build/*.o build/parity build/t2

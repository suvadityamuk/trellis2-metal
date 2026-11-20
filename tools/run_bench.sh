#!/bin/zsh
# Full benchmark: official PyTorch-MPS vs Metal parity vs Metal fast, 5 example images, sequential (idle GPU).
#   tools/run_bench.sh [512|1024_cascade ...]
cd "$(dirname "$0")/.."
D=ref/TRELLIS.2/assets/example_image
IM=($D/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp $D/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp
    $D/0f168a4b1b6e96c72e9627c97a212c27a4572250ff58e25703b9d0c2bc74191a.webp $D/130c2b18f1651a70f8aa15b2c99f8dba29bb943044d92871f9223bd3e989e8b1.webp $D/T.png)
ARGS=(); for i in $IM; do ARGS+=(--image $i); done
mkdir -p bench/official_out bench/official_alt_out bench/parity_out bench/fast_out logs
for T in ${@:-512 1024_cascade}; do
  S=${T/1024_cascade/1024c}
  for M in parity fast; do
    [[ -f bench/${M}_$S.json ]] || ./build/t2 $ARGS --type $T --mode $M --warmup 1 --quiet --bench bench/${M}_$S.json --out bench/${M}_out > logs/bench_${M}_$S.log 2>&1
  done
  [[ -f bench/official_$S.json ]] || ref/venv/bin/python tools/bench_ref.py --type $T --images $IM --out bench/official_$S.json --save bench/official_out > logs/bench_official_$S.log 2>&1
  # tolerance band: the official pipeline with its attention backend switched (MPS prefill kernel -> MPSGraph)
  [[ -f bench/official_alt_$S.json ]] || PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1 ref/venv/bin/python tools/bench_ref.py --type $T --images $IM \
      --out bench/official_alt_$S.json --save bench/official_alt_out > logs/bench_official_alt_$S.log 2>&1
done
echo done

#!/bin/zsh
# Drift-robust A/B of the fast build's matmul backend (MPSNDArray vs own Metal 4 GEMM): one process per (image, backend),
# backend order flipped per image, each process warmed up on the smallest image.
#   tools/ab_bench.sh [512|1024_cascade ...]
cd "$(dirname "$0")/.."
D=ref/TRELLIS.2/assets/example_image
IM=($D/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp $D/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp
    $D/0f168a4b1b6e96c72e9627c97a212c27a4572250ff58e25703b9d0c2bc74191a.webp $D/130c2b18f1651a70f8aa15b2c99f8dba29bb943044d92871f9223bd3e989e8b1.webp $D/T.png)
W=$IM[4]
mkdir -p bench/ab
for T in ${@:-512 1024_cascade}; do
  S=${T/1024_cascade/1024c}; n=0
  for i in $IM; do
    n=$((n+1)); if (( n % 2 )); then O=(mpp mps); else O=(mps mpp); fi
    for G in $O; do
      T2_GEMM=$G ./build/t2 --image $W --image $i --type $T --mode fast --quiet 2>/dev/null | grep "^fast" | tail -1 | sed "s/^/$G /" >> bench/ab/$S.txt
    done
  done
done
echo done

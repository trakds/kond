#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-$(command -v nvcc || true)}"
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
[ -z "$NVCC" ] && { echo "нет nvcc — установите CUDA Toolkit 12.x" >&2; exit 1; }

ARCH="${ARCH:-sm_75 sm_80 sm_86 sm_89}"
PTX="${PTX:-}"                       # напр. PTX="sm_90" для forward-compat
GENCODE=""
for a in $ARCH; do GENCODE="$GENCODE -gencode arch=compute_${a#sm_},code=$a"; done
for p in $PTX; do  GENCODE="$GENCODE -gencode arch=compute_${p#sm_},code=compute_${p#sm_}"; done

mkdir -p build
echo "nvcc: $NVCC"
echo "arch: $ARCH${PTX:+ (+PTX: $PTX)}"

t0=$(date +%s)
"$NVCC" -O3 -std=c++17 $GENCODE ${XDEF:-} -o build/qpow kernel.cu -pthread -ldl
echo "linux: build/qpow  ($(( $(date +%s) - t0 ))s)"

if [ "${WIN:-0}" = "1" ]; then
  CC=x86_64-w64-mingw32-g++
  command -v "$CC" >/dev/null || { echo "нет $CC — sudo apt install g++-mingw-w64-x86-64" >&2; exit 1; }
  "$NVCC" -O3 -std=c++17 $GENCODE ${XDEF:-} -ccbin "$CC" \
          -o build/qpow.exe kernel.cu -lpthread -lws2_32
  echo "win  : build/qpow.exe"
fi

ls -la build/
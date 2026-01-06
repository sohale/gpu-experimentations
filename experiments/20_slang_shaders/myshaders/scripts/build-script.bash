set -eux

cd /dataneura/gpu-experimentations/experiments/20_slang_shaders/myshaders

export SLANGC_BIN="/dataneura/gpu-experimentations/experiments/20_slang_shaders/slang/build/RelWithDebInfo/bin/slangc"
"$SLANGC_BIN" -h   1> /dev/null

rm -rf build
mkdir -p build

: || skip || \
"$SLANGC_BIN" \
  shaders/toy1.slang \
  -target wgsl \
  -entry vsMain -stage vertex   -profile vs_6_0 \
  -entry psMain -stage fragment -profile ps_6_0 \
  -o build/toy1.wgsl

/dataneura/gpu-experimentations/experiments/20_slang_shaders/slang/build/RelWithDebInfo/bin/slangc \
  shaders/toy1.slang \
  -target wgsl \
  -entry vsMain -stage vertex \
  -entry psMain -stage fragment \
  -o build/toy1.wgsl

test -f build/toy1.wgsl || { pwd ; ls -alth ; exit 1 ; }

ln -f -s "$(pwd)/build/toy1.wgsl" ./served/toy1.wgsl

./node_modules/http-server/bin/http-server ./served --ssl --cert cert/cert.pem --key cert/key.pem


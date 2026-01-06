echo "If you found this useful, and saved your time, consider tipping me. I shared these, while I am not working under a permanent employment."
echo "dont run this. See https://github.com/shader-slang/slang/blob/master/docs/building.md "
exit 1

cd /dataneura/gpu-experimentations/experiments/20_slang_shaders
git clone https://github.com/shader-slang/slang --recursive
# don't `cd` into it yet:
git fetch https://github.com/shader-slang/slang.git 'refs/tags/*:refs/tags/*'

cd slang
# cmake --preset default
# cmake --build --preset releaseWithDebugInfo

# didnt work:
#cmake --preset vs2022
# # ? start devenv ./build/slang.sln
#cmake --build --preset releaseWithDebugInfo

#key: dont ignore: if you switch between verions etc:
rm -rf build

CC=clang-18 CXX=clang++-18  cmake --preset default
CC=clang-18 CXX=clang++-18  cmake --build --preset releaseWithDebugInfo
# worked


# Emscripten 💖

# cd /dataneura/gpu-experimentations/experiments/20_slang_shaders/slang/
cd ..
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk/
./emsdk install latest
# worked

# wait! You need to set
# Also it creates ~/.bash_profile, which disbaled by ~/.bashrc AND ~/.profile , both!

./emsdk activate latest
source "/dataneura/gpu-experimentations/experiments/20_slang_shaders/emsdk/emsdk_env.sh"
# NO! echo 'source "/dataneura/gpu-experimentations/experiments/20_slang_shaders/emsdk/emsdk_env.sh"' >> $HOME/.bash_profile
# instead, manually edit, or move to ~/.profile, etc. I added in the bottom of ~/.bashrc
# echo 'source "/dataneura/gpu-experimentations/experiments/20_slang_shaders/emsdk/emsdk_env.sh"' >> $HOME/.bashrc
# You may need to restart the shell, etc.

cd ../slang/
# Build generators. from https://github.com/shader-slang/slang/blob/master/docs/building.md
CC=clang-18 CXX=clang++-18  cmake --workflow --preset generators --fresh
# worked

mkdir generators
# still in slang
CC=clang-18 CXX=clang++-18   cmake --install build --prefix generators --component generators
pushd ../emsdk
source ./emsdk_env.sh
popd
# still in slang?
CC=clang-18 CXX=clang++-18  emcmake cmake -DSLANG_GENERATORS_PATH=generators/bin --preset emscripten -G "Ninja"
# worked

CC=clang-18 CXX=clang++-18   cmake --build --preset emscripten --target slang-wasm
# ...

# Then, find the slang binary and run it
./build/RelWith.../slang -h

# now refer to suild-script.bash


# For server: First enable nvm/npm
nvm -v && npm -v
npm install http-server
# --dev-dependcy?

# A web-GPU needs to be served via https:

# using self-signed cert
mkdir cert
cd cert/
openssl req -x509 -newkey rsa:2048   -keyout key.pem   -out cert.pem   -days 365   -nodes
cd ..

# uses self-signed cert & ssl to serve on https
./node_modules/http-server/bin/http-server ./served --ssl --cert cert/cert.pem --key cert/key.pem

# make sure explitily enter this, note https, note 8080, not 8000
echo "https://46.101.15.163:8080/"

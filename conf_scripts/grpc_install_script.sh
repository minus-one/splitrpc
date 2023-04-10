#! /bin/zsh

GRPC_INSTALL_DIR=/data/azk68/grpc_install_dir/
sudo apt install -y build-essential autoconf libtool pkg-config
git clone --recurse-submodules -b v1.42.0 https://github.com/grpc/grpc
cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_DIR} ../..
make -j
make install
popd

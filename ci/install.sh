set -ex

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-driver-dev-8.0 cuda-cudart-8-0 cuda-cublas-8-0

# Not sure why I have to do this, but it seems to work...
sudo cp /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/* /usr/local/cuda-8.0/targets/x86_64-linux/lib/

if [ "$DOCS" = "yes" ]; then
  pip install 'travis-cargo<0.2' --user;
  export PATH=$HOME/.local/bin:$PATH;
fi

if [ "$RUSTFMT" = "yes" ]; then
    rustup component add rustfmt-preview
fi

if [ "$CLIPPY" = "yes" ]; then
    rustup component add clippy-preview
fi
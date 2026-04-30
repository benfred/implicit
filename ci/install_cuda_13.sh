dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

dnf install -y \
    cuda-nvcc-13-0 \
    cuda-cudart-devel-13-0 \
    libcurand-devel-13-0 \
    libcublas-devel-13-0 \
    ninja-build

ln -s cuda-13.0 /usr/local/cuda

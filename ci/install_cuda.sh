yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-11-2 \
    cuda-cudart-devel-11-2 \
    libcurand-devel-11-2 \
    libcublas-devel-11-2 \
    cuda-nvprof-11-2 \
    ninja-build
ln -s cuda-11.2 /usr/local/cuda

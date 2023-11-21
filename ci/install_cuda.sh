yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-11-4 \
    cuda-cudart-devel-11-4 \
    libcurand-devel-11-4 \
    libcublas-devel-11-4 \
    cuda-nvprof-11-4 \
    ninja-build
ln -s cuda-11.4 /usr/local/cuda

if (UNIX)
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
if (${CUDAToolkit_VERSION} VERSION_LESS "11.0.0")
    message("implicit requires CUDA 11.0 or greater for GPU acceleration - found CUDA ${CUDAToolkit_VERSION}")

elseif(DEFINED ENV{IMPLICIT_DISABLE_CUDA})
    # disable building the CUDA extension if the IMPLICIT_DISABLE_CUDA environment variable is set
    message("Disabling building the GPU extension since IMPLICIT_DISABLE_CUDA env var is set")

else()
    enable_language(CUDA)
    cython_transpile(_cuda.pyx LANGUAGE CXX)

    add_compile_options(-DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)

    # use rapids-cmake to install dependencies
    set(rapids-cmake-version "26.02")
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/refs/heads/release/26.02/RAPIDS.cmake
        ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(rapids-cmake)
    include(rapids-cpm)
    include(rapids-cuda)
    include(rapids-export)
    include(rapids-find)

    rapids_cpm_init()
    rapids_cmake_build_type(Release)

    # We must find CCCL ourselves before raft so that we get the right version.
    include(${rapids-cmake-dir}/cpm/cccl.cmake)
    rapids_cpm_cccl() 

    # get rmm
    include(${rapids-cmake-dir}/cpm/rmm.cmake)
    rapids_cpm_rmm()

    rapids_cpm_find(raft 26.02
        GLOBAL_TARGETS raft::raft
        CPM_ARGS
          GIT_REPOSITORY  https://github.com/rapidsai/raft.git
          GIT_TAG         v26.02.00
          SOURCE_SUBDIR   cpp
          OPTIONS
              "BUILD_TESTS OFF"
              "BUILD_PRIMS_BENCH OFF"
              "RAFT_COMPILE_LIBRARY OFF"
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Wno-deprecated-gpu-targets -Xfatbin=-compress-all --expt-relaxed-constexpr")

    python_add_library(_cuda MODULE _cuda.cxx
        als.cu
        bpr.cu
        matrix.cu
        random.cu
        knn.cu
    )

    if(DEFINED ENV{IMPLICIT_CUDA_ARCH})
        message("using cuda arch $ENV{IMPLICIT_CUDA_ARCH}")
        set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES $ENV{IMPLICIT_CUDA_ARCH})
    else()
        if (${CUDAToolkit_VERSION} VERSION_LESS "11.1.0")
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80")
        elseif (${CUDAToolkit_VERSION} VERSION_LESS "11.8.0")
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80;86")
        elseif (${CUDAToolkit_VERSION} VERSION_LESS "13.0.0")
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80;86;90")
        else()
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "80;86;90;100;120")
        endif()
        get_target_property(CUDA_ARCH _cuda CUDA_ARCHITECTURES)
        message("using cuda architectures ${CUDA_ARCH} for cuda version ${CUDAToolkit_VERSION}")
    endif()
    target_link_libraries(_cuda PRIVATE CUDA::cublas CUDA::curand raft::raft)

    install(TARGETS _cuda LIBRARY DESTINATION implicit/gpu)
endif()
endif()
endif()

FILE(GLOB gpu_python_files *.py)
install(FILES ${gpu_python_files} DESTINATION implicit/gpu)

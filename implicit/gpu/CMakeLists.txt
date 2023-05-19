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
    add_cython_target(_cuda CXX)

    # use rapids-cmake to install dependencies
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.04/RAPIDS.cmake
        ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(rapids-cmake)
    include(rapids-cpm)
    include(rapids-cuda)
    include(rapids-export)
    include(rapids-find)
    rapids_cpm_init()
    rapids_cmake_build_type(Release)

    # get rmm
    include(${rapids-cmake-dir}/cpm/rmm.cmake)
    rapids_cpm_rmm(BUILD_EXPORT_SET implicit-exports INSTALL_EXPORT_SET implicit-exports)

    # get raft
    # note: we're using RAFT in header only mode right now - mainly to reduce binary
    # size of the compiled wheels
    rapids_cpm_find(raft 23.06
        CPM_ARGS
          GIT_REPOSITORY  https://github.com/rapidsai/raft.git
          GIT_TAG         branch-23.06
          DOWNLOAD_ONLY   YES
    )
    include_directories(${raft_SOURCE_DIR}/cpp/include)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Wno-deprecated-gpu-targets -Xfatbin=-compress-all --expt-relaxed-constexpr")

    add_library(_cuda MODULE ${_cuda}
        als.cu
        bpr.cu
        matrix.cu
        random.cu
        knn.cu
    )

    python_extension_module(_cuda)

    if(DEFINED ENV{IMPLICIT_CUDA_ARCH})
        message("using cuda arch $ENV{IMPLICIT_CUDA_ARCH}")
        set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES $ENV{IMPLICIT_CUDA_ARCH})
    else()
        if (${CUDAToolkit_VERSION} VERSION_LESS "11.1.0")
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "37;50;70;60;80")
        else()
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "37;50;70;60;80;86")
        endif()
    endif()
    target_link_libraries(_cuda CUDA::cublas CUDA::curand rmm::rmm)

    install(TARGETS _cuda LIBRARY DESTINATION implicit/gpu)
endif()
endif()
endif()

FILE(GLOB gpu_python_files *.py)
install(FILES ${gpu_python_files} DESTINATION implicit/gpu)

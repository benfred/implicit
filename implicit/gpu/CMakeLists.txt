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
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.12/RAPIDS.cmake
        ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(rapids-cmake)
    include(rapids-cpm)
    include(rapids-cuda)
    include(rapids-export)
    include(rapids-find)
    include(${rapids-cmake-dir}/cpm/package_override.cmake)

    rapids_cpm_init()
    rapids_cmake_build_type(Release)

    # thrust/cub have a cmake issue where the conda build fails
    # to find them, and needs these patches
    # https://github.com/benfred/cub/commit/97934d146b771fd2e8bda75f73349a4b3c9e10a7
    # https://github.com/benfred/thrust/commit/8452c764cc8d772314169e99811535f3a9108cfe
    # (note that cub is pulled in through thrust here - meaning we only need to override
    # the thrust version to pull it in)
    # Issue is tracked in https://github.com/NVIDIA/thrust/issues/1966
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
      [=[
        {
          "packages" : {
            "Thrust" : {
              "version" : "1.17.2",
              "git_url" : "https://github.com/benfred/thrust.git",
              "git_tag" : "no_cmake_find_root_path",
              "git_shallow" : true,
              "always_download" : true,
            }
          }
        }
    ]=])
    rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

    # get rmm
    include(${rapids-cmake-dir}/cpm/rmm.cmake)
    rapids_cpm_rmm(BUILD_EXPORT_SET implicit-exports INSTALL_EXPORT_SET implicit-exports)

    # get raft
    # note: we're using RAFT in header only mode right now - mainly to reduce binary
    # size of the compiled wheels
    rapids_cpm_find(raft 23.12
        CPM_ARGS
          GIT_REPOSITORY  https://github.com/rapidsai/raft.git
          GIT_TAG         branch-23.12
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
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80")
        elseif (${CUDAToolkit_VERSION} VERSION_LESS "11.8.0")
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80;86")
        else()
            set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80;86;90")
        endif()
        get_target_property(CUDA_ARCH _cuda CUDA_ARCHITECTURES)
        message("using cuda architectures ${CUDA_ARCH} for cuda version ${CUDAToolkit_VERSION}")
    endif()
    target_link_libraries(_cuda CUDA::cublas CUDA::curand rmm::rmm)

    install(TARGETS _cuda LIBRARY DESTINATION implicit/gpu)
endif()
endif()
endif()

FILE(GLOB gpu_python_files *.py)
install(FILES ${gpu_python_files} DESTINATION implicit/gpu)

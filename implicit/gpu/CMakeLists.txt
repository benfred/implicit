add_cython_target(_cuda CXX)

find_package(CUDAToolkit)


if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    
    # use rapids-cmake to install dependencies
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.02/RAPIDS.cmake
        ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(rapids-cmake)
    include(rapids-cpm)
    include(rapids-cuda)
    include(rapids-export)
    include(rapids-find)
    rapids_cpm_init()
    rapids_cmake_build_type(Release)

    # get faiss from git
    rapids_cpm_find(FAISS 1.7.2
        CPM_ARGS
          GIT_REPOSITORY  https://github.com/facebookresearch/faiss.git
          VERSION         1.7.2
          DOWNLOAD_ONLY   YES
    )
    FILE(GLOB FAISS_BLOCK_SELECT ${FAISS_SOURCE_DIR}/faiss/gpu/utils/blockselect/*.cu)
    include_directories(${FAISS_SOURCE_DIR})

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Wno-deprecated-gpu-targets")

    add_library(_cuda MODULE ${_cuda}
        als.cu
        bpr.cu
        matrix.cu
        device_buffer.cu
        random.cu
        knn.cu
        # we need a limited subset of faiss for the blockselect code. rather than compile all of
        # faiss, just include the minimal subset here
        ${FAISS_BLOCK_SELECT}
        ${FAISS_SOURCE_DIR}/faiss/gpu/utils/BlockSelectFloat.cu
        ${FAISS_SOURCE_DIR}/faiss/gpu/utils/DeviceUtils.cu
        ${FAISS_SOURCE_DIR}/faiss/gpu/GpuResources.cpp
    )

    python_extension_module(_cuda)

    set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "37;50;70;60;80;86")
    target_link_libraries(_cuda CUDA::cublas CUDA::curand)

    install(TARGETS _cuda LIBRARY DESTINATION implicit/gpu)
endif()

FILE(GLOB gpu_python_files *.py)
install(FILES ${gpu_python_files} DESTINATION implicit/gpu)

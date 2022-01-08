# TODO: use fully qualified include directories
include_directories(.)

add_cython_target(_cuda CXX)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    
    include_directories(../../thirdparty/faiss)

    FILE(GLOB faiss_block_select ../../thirdparty/faiss/faiss/gpu/utils/blockselect/*.cu)

    # TODO: faiss targets?
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Wno-deprecated-gpu-targets")
        
    add_library(_cuda MODULE ${_cuda}
        als.cu
        bpr.cu
        matrix.cu
        device_buffer.cu
        random.cu
        knn.cu
        ${faiss_block_select}
        ../../thirdparty/faiss/faiss/gpu/utils/BlockSelectFloat.cu
        ../../thirdparty/faiss/faiss/gpu/utils/DeviceUtils.cu
        ../../thirdparty/faiss/faiss/gpu/GpuResources.cpp
    )
    
    # TODO: add cudart/cublas/curand variables
    python_extension_module(_cuda)

    set_target_properties(_cuda PROPERTIES CUDA_ARCHITECTURES "37;50;70;60;80;86")
    target_link_libraries(_cuda CUDA::cublas CUDA::curand)

    install(TARGETS _cuda LIBRARY DESTINATION implicit/gpu)
endif()

# TODO: warn if cuda not found

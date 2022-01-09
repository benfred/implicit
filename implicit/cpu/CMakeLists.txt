add_cython_target(_als CXX)
add_library(_als MODULE ${_als})
python_extension_module(_als)
install(TARGETS _als LIBRARY DESTINATION implicit/cpu)

add_cython_target(bpr CXX)
add_library(bpr MODULE ${bpr})
python_extension_module(bpr)
install(TARGETS bpr LIBRARY DESTINATION implicit/cpu)

add_cython_target(topk CXX)
add_library(topk MODULE ${topk})
python_extension_module(topk)
install(TARGETS topk LIBRARY DESTINATION implicit/cpu)

add_cython_target(lmf CXX)
add_library(lmf MODULE ${lmf})
python_extension_module(lmf)
install(TARGETS lmf LIBRARY DESTINATION implicit/cpu)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_als OpenMP::OpenMP_CXX)
    target_link_libraries(bpr OpenMP::OpenMP_CXX)
    target_link_libraries(topk OpenMP::OpenMP_CXX)
    target_link_libraries(lmf OpenMP::OpenMP_CXX)
endif()

FILE(GLOB cpu_python_files *.py)
install(FILES ${cpu_python_files} DESTINATION implicit/cpu)

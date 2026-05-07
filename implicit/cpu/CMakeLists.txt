cython_transpile(_als.pyx LANGUAGE CXX)
python_add_library(_als MODULE _als.cxx)
install(TARGETS _als LIBRARY DESTINATION implicit/cpu)

cython_transpile(bpr.pyx LANGUAGE CXX)
python_add_library(bpr MODULE bpr.cxx)
install(TARGETS bpr LIBRARY DESTINATION implicit/cpu)

cython_transpile(topk.pyx LANGUAGE CXX)
python_add_library(topk MODULE topk.cxx)
install(TARGETS topk LIBRARY DESTINATION implicit/cpu)

cython_transpile(lmf.pyx LANGUAGE CXX)
python_add_library(lmf MODULE lmf.cxx)
install(TARGETS lmf LIBRARY DESTINATION implicit/cpu)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_als PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(bpr PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(topk PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(lmf PUBLIC OpenMP::OpenMP_CXX)
endif()

FILE(GLOB cpu_python_files *.py)
install(FILES ${cpu_python_files} DESTINATION implicit/cpu)

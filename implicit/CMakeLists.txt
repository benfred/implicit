if(NOT CYTHON_FLAGS)
    set(CYTHON_FLAGS
        "--directive always_allow_keywords=True,binding=True,embedsignature=True,language_level=3"
    )
endif()

cython_transpile(_nearest_neighbours.pyx LANGUAGE CXX)
python_add_library(_nearest_neighbours MODULE _nearest_neighbours.cxx)
install(TARGETS _nearest_neighbours LIBRARY DESTINATION implicit)

cython_transpile(evaluation.pyx LANGUAGE CXX)
python_add_library(evaluation MODULE evaluation.cxx)
install(TARGETS evaluation LIBRARY DESTINATION implicit)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_nearest_neighbours PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(evaluation PUBLIC OpenMP::OpenMP_CXX)
endif()

add_subdirectory(cpu)
add_subdirectory(gpu)

FILE(GLOB python_files *.py)
install(FILES ${python_files} DESTINATION implicit)

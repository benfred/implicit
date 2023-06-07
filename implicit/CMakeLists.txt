if(NOT CYTHON_FLAGS)
    set(CYTHON_FLAGS
        "--directive always_allow_keywords=True,binding=True,embedsignature=True,language_level=3"
    )
endif()

add_cython_target(_nearest_neighbours CXX)
add_library(_nearest_neighbours MODULE ${_nearest_neighbours})
python_extension_module(_nearest_neighbours)
install(TARGETS _nearest_neighbours LIBRARY DESTINATION implicit)

add_cython_target(evaluation CXX)
add_library(evaluation MODULE ${evaluation})
python_extension_module(evaluation)
install(TARGETS evaluation LIBRARY DESTINATION implicit)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_nearest_neighbours OpenMP::OpenMP_CXX)
    target_link_libraries(evaluation OpenMP::OpenMP_CXX)
endif()

add_subdirectory(cpu)
add_subdirectory(gpu)

FILE(GLOB python_files *.py)
install(FILES ${python_files} DESTINATION implicit)

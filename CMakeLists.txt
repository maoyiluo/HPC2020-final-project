cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

find_library(OpenMP_LIBRARY
    NAMES omp
)

find_path(OpenMP_INCLUDE_DIR
    omp.h
)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(OpenMP)

find_package_handle_standard_args(OpenMP DEFAULT_MSG 
    OpenMP_LIBRARY OpenMP_INCLUDE_DIR)

if(OPENMP_FOUND)
    set(OpenMP_LIBRARIES ${OpenMP_LIBRARY})
    set(OpenMP_INCLUDE_DIRS ${OpenMP_INCLUDE_DIR})
    set(OpenMP_COMPILE_OPTIONS -Xpreprocessor -fopenmp)

    add_library(OpenMP::OpenMP SHARED IMPORTED)
    set_target_properties(OpenMP::OpenMP PROPERTIES
        IMPORTED_LOCATION ${OpenMP_LIBRARIES}
        INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${OpenMP_COMPILE_OPTIONS}"
    )

endif()

add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
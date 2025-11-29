cmake_minimum_required(VERSION 3.10)
project(Visualization)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio video)

set(CMAKE_CXX_STANDARD 11)

include_directories(
    ${OPENCV_INCLUDE_DIR}
    )

add_executable(Visualization Main.cpp Preprocess.cpp Preprocess.hpp Farneback.cpp Farneback.hpp LIC.cpp LIC.hpp)
set_target_properties(Visualization PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(Visualization ${OpenCV_LIBRARIES})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
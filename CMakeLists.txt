cmake_minimum_required(VERSION 3.16)

project(GaussianSplattingCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PNG REQUIRED)

set(YAML_CPP_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL
    OFF
    CACHE BOOL "" FORCE)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml-cpp/CMakeLists.txt)
  message(
    FATAL_ERROR
      "Missing submodule third_party/yaml-cpp. Run: git submodule add https://github.com/jbeder/yaml-cpp.git third_party/yaml-cpp && git submodule update --init --recursive"
  )
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml-cpp)

add_library(utils STATIC src/Utils.cpp)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC Eigen3::Eigen)

add_library(gaussian STATIC src/Gaussian.cpp)
target_include_directories(gaussian PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
  gaussian
  PUBLIC Eigen3::Eigen
  PRIVATE utils)

add_library(camera STATIC src/Camera.cpp)
target_include_directories(camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
  camera
  PUBLIC Eigen3::Eigen
  PRIVATE utils yaml-cpp)

add_library(rasterizer_utils STATIC src/rasterizerUtils.cpp)
target_include_directories(rasterizer_utils
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
  rasterizer_utils
  PUBLIC Eigen3::Eigen
  PRIVATE gaussian camera)

add_library(rasterizer STATIC src/Rasterizer.cpp)
target_include_directories(rasterizer
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
  rasterizer
  PUBLIC Eigen3::Eigen
  PRIVATE gaussian camera rasterizer_utils)

add_library(visualize STATIC src/Visualize.cpp)
target_include_directories(visualize PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(visualize PRIVATE PNG::PNG)

add_executable(gaussian_demo src/main.cpp)
target_link_libraries(gaussian_demo PRIVATE gaussian camera rasterizer_utils
                                            rasterizer visualize)

if(MSVC)
  target_compile_options(utils PRIVATE /W4 /permissive-)
  target_compile_options(gaussian PRIVATE /W4 /permissive-)
  target_compile_options(camera PRIVATE /W4 /permissive-)
  target_compile_options(rasterizer_utils PRIVATE /W4 /permissive-)
  target_compile_options(rasterizer PRIVATE /W4 /permissive-)
  target_compile_options(visualize PRIVATE /W4 /permissive-)
  target_compile_options(gaussian_demo PRIVATE /W4 /permissive-)
else()
  target_compile_options(utils PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(gaussian PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(camera PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(rasterizer_utils PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(rasterizer PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(visualize PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(gaussian_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

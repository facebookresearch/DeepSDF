project("DeepSDF")
cmake_minimum_required(VERSION 3.0)

find_package(Pangolin REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(third-party/cnpy)

set(SOURCES
    src/ShaderProgram.cpp )

add_executable(PreprocessMesh src/PreprocessMesh.cpp ${SOURCES})
target_include_directories(PreprocessMesh PRIVATE third-party/cnpy)
target_link_libraries(PreprocessMesh PRIVATE CLI11::CLI11 pangolin cnpy Eigen3::Eigen)
target_compile_features(PreprocessMesh PRIVATE cxx_std_14)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

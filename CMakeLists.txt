project("DeepSDF")
cmake_minimum_required(VERSION 3.0)

include_directories(
${Pangolin_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nanoflann REQUIRED)
find_package(Pangolin REQUIRED)

add_subdirectory(third-party/cnpy)

set(SOURCES
    src/ShaderProgram.cpp
    src/Utils.cpp)


LINK_LIBRARIES("/usr/local/lib/libpango_opengl.so")
LINK_LIBRARIES("/usr/local/lib/libpango_image.so")
LINK_LIBRARIES("/usr/lib/x86_64-linux-gnu/libOpenGL.so.0")


add_executable(PreprocessMesh src/PreprocessMesh.cpp ${SOURCES})
target_include_directories(PreprocessMesh PUBLIC third-party/cnpy)
target_include_directories(PreprocessMesh PUBLIC third-party/cnpy )
target_link_libraries(PreprocessMesh PUBLIC CLI11::CLI11 pangolin pango_core pango_display pango_geometry pango_glgeometry GLEW cnpy Eigen3::Eigen nanoflann::nanoflann)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

LINK_LIBRARIES("/usr/local/lib/libpango_opengl.so")
LINK_LIBRARIES("/usr/local/lib/libpango_image.so")
LINK_LIBRARIES("/usr/lib/x86_64-linux-gnu/libOpenGL.so.0")

add_executable(SampleVisibleMeshSurface src/SampleVisibleMeshSurface.cpp ${SOURCES})
target_include_directories(SampleVisibleMeshSurface PUBLIC third-party/cnpy)
target_link_libraries(SampleVisibleMeshSurface PUBLIC CLI11::CLI11 pango_core pango_display pango_geometry pango_glgeometry GLEW cnpy Eigen3::Eigen nanoflann::nanoflann)
target_compile_features(SampleVisibleMeshSurface PUBLIC cxx_std_14)
set_target_properties(SampleVisibleMeshSurface PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

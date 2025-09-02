cmake_minimum_required(VERSION 3.20)

project(RayTracer LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xptxas -O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -Xptxas -O2")

# Disable response files so all compiler options appear in compile_commands.json
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

# Include FetchContent
include(FetchContent)

# Fetch GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)

# Configure GLFW to not include OpenGL headers
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw)

# Fetch GLAD2
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)

# Find required packages
find_package(OpenGL REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${glad_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})



# Link libraries
target_link_libraries(${PROJECT_NAME} 
    OpenGL::GL 
    glfw 
    glad
    ${CMAKE_CUDA_LIBRARIES}
)

# Set CUDA flags
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

# Enable CUDA OpenGL interop and configure GLAD
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    USE_CUDA_OPENGL_INTEROP
    GLAD_GL_IMPLEMENTATION
)
# Copy assets folder to build directory
add_custom_target(copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    ${CMAKE_BINARY_DIR}/assets
)

add_dependencies(${PROJECT_NAME} copy_assets)

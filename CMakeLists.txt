cmake_minimum_required(VERSION 3.16)
project(executor LANGUAGES CXX VERSION 0.1.0)

# C++ 标准设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建选项
option(EXECUTOR_BUILD_TESTS "Build tests" ON)
option(EXECUTOR_BUILD_EXAMPLES "Build examples" OFF)
option(EXECUTOR_BUILD_SHARED "Build shared library" OFF)

# 包含 CMake 模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# 可选：包含 Sanitizers（用于调试构建）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(Sanitizers OPTIONAL)
endif()

# 可选：包含 Coverage（代码覆盖率）
include(Coverage OPTIONAL)

# 安装目录（须在 add_subdirectory(src) 之前，src 中 install(EXPORT) 使用 CMAKE_INSTALL_LIBDIR）
include(GNUInstallDirs)

# 添加子目录
add_subdirectory(src)

if(EXECUTOR_BUILD_TESTS)
    enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()

if(EXECUTOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 安装配置（executor 目标已由 src/CMakeLists 安装并导出）
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# find_package(executor) 支持：生成并安装 executorConfig.cmake、executorConfigVersion.cmake
include(CMakePackageConfigHelpers)
set(EXECUTOR_CMAKE_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/executor")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/executorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    cmake/executorConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/executorConfigVersion.cmake"
    DESTINATION ${EXECUTOR_CMAKE_CONFIG_DIR}
)

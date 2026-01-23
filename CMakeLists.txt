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

# 添加子目录
add_subdirectory(src)

if(EXECUTOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(EXECUTOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 安装配置
include(GNUInstallDirs)

install(TARGETS executor
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# 安装 ExecutorConfig.cmake（如果存在）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExecutorConfig.cmake")
    install(FILES cmake/ExecutorConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/executor
    )
endif()

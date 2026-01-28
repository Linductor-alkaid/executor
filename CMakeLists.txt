cmake_minimum_required(VERSION 3.16)
project(executor LANGUAGES CXX VERSION 0.2.0)

# C++ 标准设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建选项
option(EXECUTOR_BUILD_TESTS "Build tests" ON)
option(EXECUTOR_BUILD_EXAMPLES "Build examples" OFF)
option(EXECUTOR_BUILD_SHARED "Build shared library" OFF)

# GPU 支持选项
option(EXECUTOR_ENABLE_GPU "Enable GPU support" ON)
option(EXECUTOR_ENABLE_CUDA "Enable CUDA support" ON)

# 包含 CMake 模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# 可选：包含 Sanitizers（用于调试构建）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(Sanitizers OPTIONAL)
endif()

# 可选：包含 Coverage（代码覆盖率）
include(Coverage OPTIONAL)

# GPU/CUDA 支持检测
if(EXECUTOR_ENABLE_GPU)
    if(EXECUTOR_ENABLE_CUDA)
        # 查找 CUDA（优先使用现代方法）
        # CMake 3.18+ 支持 CUDAToolkit
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
            find_package(CUDAToolkit QUIET)
            if(CUDAToolkit_FOUND)
                set(CUDA_FOUND TRUE)
                set(CUDA_VERSION ${CUDAToolkit_VERSION})
                set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT_DIR})
                set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
                set(CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
            endif()
        endif()
        
        # 如果现代方法失败，尝试传统方法
        if(NOT CUDA_FOUND)
            find_package(CUDA QUIET)
        endif()
        
        if(CUDA_FOUND)
            message(STATUS "CUDA found: Version ${CUDA_VERSION}")
            if(CUDA_TOOLKIT_ROOT_DIR)
                message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
            endif()
            set(EXECUTOR_CUDA_ENABLED ON)
            # 定义编译宏
            add_definitions(-DEXECUTOR_ENABLE_GPU -DEXECUTOR_ENABLE_CUDA)
        else()
            message(WARNING "CUDA support requested but CUDA not found. Disabling CUDA support.")
            set(EXECUTOR_CUDA_ENABLED OFF)
            set(EXECUTOR_ENABLE_CUDA OFF)
        endif()
    else()
        set(EXECUTOR_CUDA_ENABLED OFF)
    endif()
    
    # 如果启用了 GPU 支持（即使 CUDA 未找到，也定义宏以支持其他后端）
    if(EXECUTOR_ENABLE_GPU)
        add_definitions(-DEXECUTOR_ENABLE_GPU)
    endif()
else()
    set(EXECUTOR_CUDA_ENABLED OFF)
    set(EXECUTOR_ENABLE_CUDA OFF)
endif()

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

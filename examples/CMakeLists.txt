# examples/CMakeLists.txt
# 配置示例可执行文件

# 注意：当前阶段还没有示例文件，后续阶段会添加
# 此文件为示例构建做好准备

# 收集示例源文件
file(GLOB EXAMPLE_SOURCES "*.cpp")

# 从通用列表中排除 GPU 示例（仅在启用 GPU 时单独添加）
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "gpu_.*\\.cpp$")

# 如果没有示例文件，创建一个占位符说明
if(NOT EXAMPLE_SOURCES)
    message(STATUS "No example files found. Examples will be added in later stages.")
    return()
endif()

# 为每个示例文件创建可执行文件
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    
    # 链接 executor 库
    target_link_libraries(${EXAMPLE_NAME} PRIVATE executor)
endforeach()

# GPU 示例：仅在启用 GPU 时构建
if(EXECUTOR_ENABLE_GPU)
    # 设备查询示例（CUDA或OpenCL任一可用即可）
    if(EXECUTOR_CUDA_ENABLED OR EXECUTOR_OPENCL_ENABLED)
        add_executable(gpu_device_query gpu_device_query.cpp)
        target_link_libraries(gpu_device_query PRIVATE executor)
    endif()

    # CUDA 特定示例
    if(EXECUTOR_CUDA_ENABLED)
        set(_cuda_inc_dirs ${CUDA_INCLUDE_DIRS})
        if(NOT _cuda_inc_dirs AND CUDAToolkit_INCLUDE_DIRS)
            set(_cuda_inc_dirs ${CUDAToolkit_INCLUDE_DIRS})
        endif()

        add_executable(gpu_basic gpu_basic.cpp)
        target_link_libraries(gpu_basic PRIVATE executor)
        target_include_directories(gpu_basic PRIVATE ${_cuda_inc_dirs})

        add_executable(gpu_multi_device gpu_multi_device.cpp)
        target_link_libraries(gpu_multi_device PRIVATE executor)
        target_include_directories(gpu_multi_device PRIVATE ${_cuda_inc_dirs})

        # gpu_unified_memory 包含 CUDA kernel 语法，必须用 nvcc 编译
        # Ninja/Makefile 生成器下可用 enable_language(CUDA)；VS 生成器需要 CUDA toolset，跳过
        if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
            enable_language(CUDA)
            add_executable(gpu_unified_memory gpu_unified_memory.cpp)
            target_link_libraries(gpu_unified_memory PRIVATE executor)
            set_source_files_properties(gpu_unified_memory.cpp PROPERTIES LANGUAGE CUDA)
        else()
            message(STATUS "Skipping gpu_unified_memory: Visual Studio generator requires CUDA toolset integration")
        endif()
    endif()

    # OpenCL 特定示例
    if(EXECUTOR_OPENCL_ENABLED)
        add_executable(gpu_opencl gpu_opencl.cpp)
        target_link_libraries(gpu_opencl PRIVATE executor)
    endif()
endif()

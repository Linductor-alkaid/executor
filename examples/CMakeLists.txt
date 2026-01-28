# examples/CMakeLists.txt
# 配置示例可执行文件

# 注意：当前阶段还没有示例文件，后续阶段会添加
# 此文件为示例构建做好准备

# 收集示例源文件
file(GLOB EXAMPLE_SOURCES "*.cpp")

# 从通用列表中排除 GPU 示例（仅在启用 GPU 时单独添加）
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "gpu_basic\\.cpp$")
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "gpu_multi_device\\.cpp$")

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

# GPU 示例：仅在启用 GPU 且找到 CUDA 时构建
if(EXECUTOR_ENABLE_GPU AND EXECUTOR_CUDA_ENABLED)
    add_executable(gpu_basic gpu_basic.cpp)
    target_link_libraries(gpu_basic PRIVATE executor)
    add_executable(gpu_multi_device gpu_multi_device.cpp)
    target_link_libraries(gpu_multi_device PRIVATE executor)
endif()

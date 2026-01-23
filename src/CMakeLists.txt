# src/CMakeLists.txt
# 配置 executor 库的构建

# 收集源文件
# 注意：当前阶段还没有源文件，后续阶段会添加
file(GLOB_RECURSE EXECUTOR_SOURCES
    "executor/*.cpp"
    "executor/*.c"
)

# 收集头文件（用于 IDE 显示）
file(GLOB_RECURSE EXECUTOR_HEADERS
    "executor/*.hpp"
    "executor/*.h"
)

# 确定库类型
if(EXECUTOR_BUILD_SHARED)
    set(LIBRARY_TYPE SHARED)
else()
    set(LIBRARY_TYPE STATIC)
endif()

# 如果没有源文件，创建一个空的源文件占位符
if(NOT EXECUTOR_SOURCES)
    message(STATUS "No source files found. Creating placeholder library.")
    # 创建一个空的源文件
    set(PLACEHOLDER_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/executor_placeholder.cpp")
    file(WRITE ${PLACEHOLDER_SOURCE} "// Placeholder source file\n// This will be replaced in later stages\nnamespace executor {\n    void placeholder() {}\n}\n")
    list(APPEND EXECUTOR_SOURCES ${PLACEHOLDER_SOURCE})
endif()

# 创建库目标
add_library(executor ${LIBRARY_TYPE}
    ${EXECUTOR_SOURCES}
    ${EXECUTOR_HEADERS}
)

# 设置包含目录
# PUBLIC: 使用此库的项目也会获得这些包含目录
# PRIVATE: 仅此库内部使用
target_include_directories(executor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 平台特定的链接库
if(UNIX AND NOT APPLE)
    # Linux: 链接 pthread（用于thread_utils中的pthread函数）
    find_package(Threads REQUIRED)
    target_link_libraries(executor PUBLIC Threads::Threads)
    # 链接rt库（用于某些Linux系统上的实时调度函数）
    target_link_libraries(executor PRIVATE rt)
elseif(WIN32)
    # Windows: 可能需要额外的库
    # 目前仅使用标准库，无需额外链接
endif()

# 设置 C++ 标准（确保与根 CMakeLists.txt 一致）
target_compile_features(executor PUBLIC cxx_std_20)

# MSVC: 设置源文件编码为UTF-8，避免C4819警告
if(MSVC)
    target_compile_options(executor PRIVATE /utf-8)
endif()

# 设置输出名称
set_target_properties(executor PROPERTIES
    OUTPUT_NAME executor
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# 安装目标
install(TARGETS executor
    EXPORT executorTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 安装导出文件（用于 find_package）
install(EXPORT executorTargets
    FILE executorTargets.cmake
    NAMESPACE executor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/executor
)

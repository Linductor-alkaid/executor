# src/CMakeLists.txt
# 配置 executor 库的构建

# 收集源文件
# 注意：当前阶段还没有源文件，后续阶段会添加
file(GLOB_RECURSE EXECUTOR_SOURCES
    "executor/*.cpp"
    "executor/*.c"
)

# 如果未启用CUDA，排除GPU相关源文件
if(NOT EXECUTOR_CUDA_ENABLED)
    list(FILTER EXECUTOR_SOURCES EXCLUDE REGEX ".*gpu.*")
endif()

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

# CUDA 支持（如果启用）
if(EXECUTOR_CUDA_ENABLED)
    # 添加 CUDA 包含目录
    if(CUDA_INCLUDE_DIRS)
        target_include_directories(executor
            PRIVATE
                ${CUDA_INCLUDE_DIRS}
        )
    elseif(CUDAToolkit_INCLUDE_DIRS)
        target_include_directories(executor
            PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS}
        )
    endif()
    
    # 链接 CUDA 库
    if(CUDAToolkit_FOUND)
        # 使用现代 CUDAToolkit 目标
        target_link_libraries(executor PRIVATE CUDA::cudart)
        # Windows: 使用延迟加载，避免DLL加载失败时程序崩溃
        if(WIN32)
            # CUDA运行时DLL名称格式：cudart64_<major>.dll (例如 cudart64_12.dll)
            # 注意：不要包含小版本号
            if(CUDA_VERSION)
                string(REGEX REPLACE "([0-9]+)\\.[0-9]+.*" "\\1" cuda_major ${CUDA_VERSION})
                # 获取当前的链接标志
                get_target_property(current_flags executor LINK_FLAGS)
                if(current_flags)
                    set(new_flags "${current_flags} /DELAYLOAD:cudart64_${cuda_major}.dll")
                else()
                    set(new_flags "/DELAYLOAD:cudart64_${cuda_major}.dll")
                endif()
                set_target_properties(executor PROPERTIES
                    LINK_FLAGS "${new_flags}"
                )
            else()
                # 默认尝试多个可能的版本
                get_target_property(current_flags executor LINK_FLAGS)
                if(current_flags)
                    set(new_flags "${current_flags} /DELAYLOAD:cudart64_11.dll /DELAYLOAD:cudart64_12.dll")
                else()
                    set(new_flags "/DELAYLOAD:cudart64_11.dll /DELAYLOAD:cudart64_12.dll")
                endif()
                set_target_properties(executor PROPERTIES
                    LINK_FLAGS "${new_flags}"
                )
            endif()
            # 需要链接延迟加载辅助库
            target_link_libraries(executor PRIVATE delayimp)
        endif()
    elseif(CUDA_LIBRARIES)
        # 使用传统 CUDA 变量
        target_link_libraries(executor PRIVATE ${CUDA_LIBRARIES})
        # 设置 CUDA 库路径
        if(CUDA_LIBRARY_DIRS)
            target_link_directories(executor PRIVATE ${CUDA_LIBRARY_DIRS})
        endif()
        # Windows: 使用延迟加载
        if(WIN32)
            # 尝试找到cudart DLL名称
            foreach(lib ${CUDA_LIBRARIES})
                if(lib MATCHES "cudart")
                    get_filename_component(lib_name ${lib} NAME_WE)
                    string(REGEX REPLACE "^lib" "" dll_name ${lib_name})
                    set_target_properties(executor PROPERTIES
                        LINK_FLAGS "/DELAYLOAD:${dll_name}.dll"
                    )
                    target_link_libraries(executor PRIVATE delayimp)
                    break()
                endif()
            endforeach()
        endif()
    endif()
endif()

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

# 设置输出名称和版本
if(WIN32)
    # Windows: 动态库需要设置输出名称和导入库名称
    if(EXECUTOR_BUILD_SHARED)
        set_target_properties(executor PROPERTIES
            OUTPUT_NAME executor
            VERSION ${PROJECT_VERSION}
        )
        # 对于多配置生成器（如 Visual Studio），需要为每个配置设置输出目录
        foreach(config Release Debug RelWithDebInfo MinSizeRel)
            string(TOUPPER ${config} config_upper)
            set_target_properties(executor PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/bin/${config}
                LIBRARY_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/lib/${config}
                ARCHIVE_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/lib/${config}
            )
        endforeach()
    else()
        # 静态库
        set_target_properties(executor PROPERTIES
            OUTPUT_NAME executor
        )
        # 对于多配置生成器，为每个配置设置输出目录
        foreach(config Release Debug RelWithDebInfo MinSizeRel)
            string(TOUPPER ${config} config_upper)
            set_target_properties(executor PROPERTIES
                ARCHIVE_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/lib/${config}
            )
        endforeach()
    endif()
else()
    # Linux/Unix
    set_target_properties(executor PROPERTIES
        OUTPUT_NAME executor
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

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

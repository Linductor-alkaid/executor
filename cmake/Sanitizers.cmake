# Sanitizers.cmake
# 提供 AddressSanitizer (ASAN)、UndefinedBehaviorSanitizer (UBSAN) 等支持
# 用于调试构建，帮助发现内存错误和未定义行为

# 选项：启用 sanitizers
option(EXECUTOR_ENABLE_SANITIZERS "Enable sanitizers (ASAN, UBSAN, etc.)" OFF)

if(EXECUTOR_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # GCC/Clang 支持 sanitizers
        set(SANITIZER_FLAGS "")
        
        # AddressSanitizer (检测内存错误)
        option(EXECUTOR_ENABLE_ASAN "Enable AddressSanitizer" ON)
        if(EXECUTOR_ENABLE_ASAN)
            list(APPEND SANITIZER_FLAGS "-fsanitize=address")
            # 检测内存泄漏
            list(APPEND SANITIZER_FLAGS "-fno-omit-frame-pointer")
        endif()
        
        # UndefinedBehaviorSanitizer (检测未定义行为)
        option(EXECUTOR_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" ON)
        if(EXECUTOR_ENABLE_UBSAN)
            list(APPEND SANITIZER_FLAGS "-fsanitize=undefined")
            # 在第一次错误时停止
            list(APPEND SANITIZER_FLAGS "-fno-sanitize-recover=all")
        endif()
        
        # ThreadSanitizer (检测数据竞争)
        option(EXECUTOR_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
        if(EXECUTOR_ENABLE_TSAN)
            list(APPEND SANITIZER_FLAGS "-fsanitize=thread")
        endif()
        
        # 应用 sanitizer 标志
        if(SANITIZER_FLAGS)
            string(REPLACE ";" " " SANITIZER_FLAGS_STR "${SANITIZER_FLAGS}")
            add_compile_options(${SANITIZER_FLAGS})
            add_link_options(${SANITIZER_FLAGS})
            
            message(STATUS "Sanitizers enabled: ${SANITIZER_FLAGS_STR}")
        endif()
        
    elseif(MSVC)
        # MSVC 使用不同的 sanitizer 支持
        message(WARNING "Sanitizers are not fully supported on MSVC. Consider using Clang on Windows.")
    else()
        message(WARNING "Sanitizers are not supported for this compiler.")
    endif()
endif()

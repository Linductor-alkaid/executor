# CompilerWarnings.cmake
# 统一编译器警告配置，支持 GCC、Clang、MSVC

# 选项：是否将警告视为错误
option(EXECUTOR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(MSVC)
    # MSVC 警告配置
    add_compile_options(/W4)
    if(EXECUTOR_WARNINGS_AS_ERRORS)
        add_compile_options(/WX)
    endif()
    
    # 禁用一些常见的 MSVC 警告
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang 警告配置
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    # 额外的有用警告
    add_compile_options(
        -Wcast-align
        -Wcast-qual
        -Wconversion
        -Wctor-dtor-privacy
        -Wdisabled-optimization
        -Wformat=2
        -Winit-self
        -Wlogical-op
        -Wmissing-declarations
        -Wmissing-include-dirs
        -Wnoexcept
        -Wold-style-cast
        -Woverloaded-virtual
        -Wredundant-decls
        -Wshadow
        -Wsign-conversion
        -Wstrict-null-sentinel
        -Wstrict-overflow=5
        -Wswitch-default
        -Wundef
        -Wunused
        -Wzero-as-null-pointer-constant
    )
    
    # Clang 特定警告
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Wdocumentation
            -Wno-documentation-unknown-command
        )
    endif()
    
    # 如果启用警告为错误
    if(EXECUTOR_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()

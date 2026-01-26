# tests/CMakeLists.txt
# 配置测试框架和测试可执行文件

# 注意：当前阶段还没有测试文件，后续阶段会添加
# 此文件为测试框架做好准备

# 收集测试源文件
file(GLOB TEST_SOURCES "*.cpp")

# 如果未启用CUDA，排除CUDA相关测试
if(NOT EXECUTOR_CUDA_ENABLED)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*cuda.*")
endif()

# 如果没有测试文件，创建一个占位符说明
if(NOT TEST_SOURCES)
    message(STATUS "No test files found. Tests will be added in later stages.")
    return()
endif()

# 为每个测试文件创建可执行文件
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # 链接 executor 库
    target_link_libraries(${TEST_NAME} PRIVATE executor)
    
    # 如果是CUDA测试，添加CUDA支持
    if(TEST_NAME MATCHES ".*cuda.*" AND EXECUTOR_CUDA_ENABLED)
        if(CUDA_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
        elseif(CUDAToolkit_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endif()
        
        if(CUDAToolkit_FOUND)
            target_link_libraries(${TEST_NAME} PRIVATE CUDA::cudart)
            # Windows: 使用延迟加载
            if(WIN32)
                if(CUDA_VERSION)
                    string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" cuda_major ${CUDA_VERSION})
                    # 获取当前的链接标志
                    get_target_property(current_flags ${TEST_NAME} LINK_FLAGS)
                    if(current_flags)
                        set(new_flags "${current_flags} /DELAYLOAD:cudart64_${cuda_major}.dll")
                    else()
                        set(new_flags "/DELAYLOAD:cudart64_${cuda_major}.dll")
                    endif()
                    set_target_properties(${TEST_NAME} PROPERTIES
                        LINK_FLAGS "${new_flags}"
                    )
                else()
                    get_target_property(current_flags ${TEST_NAME} LINK_FLAGS)
                    if(current_flags)
                        set(new_flags "${current_flags} /DELAYLOAD:cudart64_11.dll /DELAYLOAD:cudart64_12.dll")
                    else()
                        set(new_flags "/DELAYLOAD:cudart64_11.dll /DELAYLOAD:cudart64_12.dll")
                    endif()
                    set_target_properties(${TEST_NAME} PROPERTIES
                        LINK_FLAGS "${new_flags}"
                    )
                endif()
                target_link_libraries(${TEST_NAME} PRIVATE delayimp)
            endif()
        elseif(CUDA_LIBRARIES)
            target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES})
            if(CUDA_LIBRARY_DIRS)
                target_link_directories(${TEST_NAME} PRIVATE ${CUDA_LIBRARY_DIRS})
            endif()
            # Windows: 使用延迟加载
            if(WIN32)
                foreach(lib ${CUDA_LIBRARIES})
                    if(lib MATCHES "cudart")
                        get_filename_component(lib_name ${lib} NAME_WE)
                        string(REGEX REPLACE "lib" "" dll_name ${lib_name})
                        set_target_properties(${TEST_NAME} PROPERTIES
                            LINK_FLAGS "/DELAYLOAD:${dll_name}.dll"
                        )
                        target_link_libraries(${TEST_NAME} PRIVATE delayimp)
                        break()
                    endif()
                endforeach()
            endif()
        endif()
    endif()
    
    # 添加src目录到包含路径，以便测试可以访问内部头文件
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    
    # MSVC: 设置源文件编码为UTF-8，避免C4819警告
    if(MSVC)
        target_compile_options(${TEST_NAME} PRIVATE /utf-8)
    endif()
    
    # 应用覆盖率选项（如果启用）
    if(EXECUTOR_ENABLE_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${TEST_NAME} PRIVATE
                --coverage
                -fprofile-arcs
                -ftest-coverage
            )
            target_link_options(${TEST_NAME} PRIVATE --coverage)
        endif()
    endif()
    
    # 添加测试
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # 根据测试类型设置 LABELS 和 TIMEOUT
    # 单元测试
    if(TEST_NAME MATCHES "^test_(util|task|thread_pool|statistics_collector|task_monitor|cuda_executor)$")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit"
            TIMEOUT 30
        )
    # 集成/端到端测试
    elseif(TEST_NAME MATCHES "^test_(executor_facade|executor_manager|e2e_workflow|thread_pool_integration|monitor_integration|cycle_manager_integration|realtime_thread_executor|thread_pool_executor)$")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "integration"
            TIMEOUT 60
        )
    # 性能/压力测试
    elseif(TEST_NAME MATCHES "^test_(performance|stress|benchmark)$")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "benchmark;stress"
            TIMEOUT 120
        )
    # 性能基准测试（可配置、JSON 输出，用于优化前后对比）
    elseif(TEST_NAME STREQUAL "benchmark_baseline")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "benchmark;baseline"
            TIMEOUT 120
        )
    # 默认：单元测试
    else()
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit"
            TIMEOUT 30
        )
    endif()
endforeach()

# 如果使用 Google Test 或其他测试框架，可以在这里配置
# 例如：
# find_package(GTest REQUIRED)
# target_link_libraries(test_name PRIVATE executor GTest::GTest GTest::Main)

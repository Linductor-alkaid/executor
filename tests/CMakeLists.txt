# tests/CMakeLists.txt
# 配置测试框架和测试可执行文件

# 注意：当前阶段还没有测试文件，后续阶段会添加
# 此文件为测试框架做好准备

# 收集测试源文件
file(GLOB TEST_SOURCES "*.cpp")

# 如果未启用CUDA，排除CUDA/GPU相关测试（executor 不编译 gpu 源）
if(NOT EXECUTOR_CUDA_ENABLED)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*cuda.*")
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*gpu_memory_manager.*")
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
    
    # 如果启用了 GPU 支持，传递宏定义给测试
    if(EXECUTOR_ENABLE_GPU)
        target_compile_definitions(${TEST_NAME} PRIVATE EXECUTOR_ENABLE_GPU)
    endif()
    
    # 如果是 CUDA/GPU 测试，添加 CUDA 头文件支持（不链接库，使用动态加载）
    if((TEST_NAME MATCHES ".*cuda.*" OR TEST_NAME MATCHES ".*gpu_multi_device.*" OR TEST_NAME MATCHES ".*gpu_memory_manager.*" OR TEST_NAME MATCHES ".*gpu_monitor.*") AND EXECUTOR_CUDA_ENABLED)
        if(CUDA_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
        elseif(CUDAToolkit_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endif()
        # 不链接CUDA库，使用动态加载
        # CudaLoader会在运行时动态搜索和加载CUDA DLL
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
    if(TEST_NAME MATCHES "^test_(util|task|thread_pool|statistics_collector|task_monitor|cuda_executor|gpu_memory_manager)$")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit"
            TIMEOUT 30
        )
    # 集成/端到端测试
    elseif(TEST_NAME MATCHES "^test_(executor_facade|executor_manager|executor_manager_gpu|gpu_multi_device|gpu_monitor|e2e_workflow|thread_pool_integration|monitor_integration|cycle_manager_integration|realtime_thread_executor|thread_pool_executor)$")
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
    # 定时器精度基准测试（多周期 delayed/periodic jitter，JSON 输出）
    elseif(TEST_NAME STREQUAL "benchmark_timer_precision")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "benchmark"
            TIMEOUT 120
        )
    # 实时线程周期精度基准测试（RealtimeThreadExecutor cycle jitter，JSON 输出）
    elseif(TEST_NAME STREQUAL "benchmark_realtime_precision")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "benchmark"
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

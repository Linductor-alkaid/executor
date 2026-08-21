# tests/CMakeLists.txt
# 配置测试框架和测试可执行文件

# 收集测试源文件
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

# 不依赖 GTest 的 standalone 测试：桌面 CI 和 Android smoke 都可直接构建运行。
foreach(STANDALONE_TEST_SOURCE
        android_smoke.cpp
        test_stop_token_compat.cpp
        test_executor_manager_android_defaults.cpp
        test_android_lockfree_queue_core.cpp
        test_multithread_mpsc.cpp
        test_lockfree_worker_queue_concurrent_steal.cpp)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/${STANDALONE_TEST_SOURCE}$")
    get_filename_component(STANDALONE_TEST_NAME ${STANDALONE_TEST_SOURCE} NAME_WE)
    add_executable(${STANDALONE_TEST_NAME} ${STANDALONE_TEST_SOURCE})
    target_link_libraries(${STANDALONE_TEST_NAME} PRIVATE executor)
    target_include_directories(${STANDALONE_TEST_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    if(MSVC)
        target_compile_options(${STANDALONE_TEST_NAME} PRIVATE /utf-8)
    endif()
    add_test(NAME ${STANDALONE_TEST_NAME} COMMAND ${STANDALONE_TEST_NAME})
    if(STANDALONE_TEST_NAME STREQUAL "android_smoke")
        # android_smoke 支持 EXECUTOR_ANDROID_SOAK_SECONDS 长稳模式。
        set(STANDALONE_TEST_TIMEOUT 900)
    else()
        set(STANDALONE_TEST_TIMEOUT 120)
    endif()
    set_tests_properties(${STANDALONE_TEST_NAME} PROPERTIES
        LABELS "unit;standalone"
        TIMEOUT ${STANDALONE_TEST_TIMEOUT}
    )
endforeach()

# 查找 Google Test
find_package(GTest)
if(NOT GTest_FOUND)
    # 可选兜底：无系统 GTest 时（如未装 libgtest-dev 的 runner），用 FetchContent
    # 拉取固定版本的 googletest。默认关闭，离线构建不受影响。
    option(EXECUTOR_FETCH_GTEST
        "Download googletest via FetchContent when system GTest is not found" OFF)
    if(EXECUTOR_FETCH_GTEST)
        include(FetchContent)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        # googletest 提供 gtest / gtest_main 目标与 GTest::gtest 别名，但没有
        # find_package 模块模式的 GTest::GTest / GTest::Main 名称；补齐以保持
        # 测试目标的链接语句不变。
        if(NOT TARGET GTest::GTest)
            add_library(GTest::GTest INTERFACE IMPORTED)
            target_link_libraries(GTest::GTest INTERFACE gtest)
        endif()
        if(NOT TARGET GTest::Main)
            add_library(GTest::Main INTERFACE IMPORTED)
            target_link_libraries(GTest::Main INTERFACE gtest_main)
        endif()
    else()
        # 用 WARNING 而非 STATUS：静默跳过约 95% 的测试会让 CI 看起来是绿的。
        message(WARNING
            "GTest not found: skipping all GTest-based tests; only the standalone "
            "tests will build and run. Install libgtest-dev (Ubuntu) or set "
            "-DEXECUTOR_FETCH_GTEST=ON to get the full suite.")
        return()
    endif()
endif()
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/test_dispatch_task_fallback\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/test_object_pool_double_release_detect\\.cpp$")

if(NOT (EXECUTOR_CUDA_ENABLED OR EXECUTOR_OPENCL_ENABLED))
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/test_gpu_executor_direct_config_validation\\.cpp$")
endif()

# 如果未启用CUDA，排除CUDA/GPU相关测试（executor 不编译 gpu 源）
if(NOT EXECUTOR_CUDA_ENABLED)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*cuda.*")
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*gpu_memory_manager.*")
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*unified_memory.*")
endif()

# 如果未启用OpenCL，排除OpenCL相关测试
if(NOT EXECUTOR_OPENCL_ENABLED)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*opencl.*")
endif()

# 如果没有测试文件，创建一个占位符说明
if(NOT TEST_SOURCES)
    message(STATUS "No test files found. Tests will be added in later stages.")
    return()
endif()

add_executable(test_comm_harness_foundation
    harness/test_comm_harness_foundation.cpp
)
target_link_libraries(test_comm_harness_foundation PRIVATE executor GTest::GTest GTest::Main)
target_include_directories(test_comm_harness_foundation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/harness
)
if(MSVC)
    target_compile_options(test_comm_harness_foundation PRIVATE /utf-8)
endif()
if(EXECUTOR_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_comm_harness_foundation PRIVATE
            --coverage
            -fprofile-arcs
            -ftest-coverage
        )
        target_link_options(test_comm_harness_foundation PRIVATE --coverage)
    endif()
endif()
add_test(NAME test_comm_harness_foundation COMMAND test_comm_harness_foundation)
set_tests_properties(test_comm_harness_foundation PROPERTIES
    LABELS "unit;harness"
    TIMEOUT 30
)

add_executable(test_comm_facade_usage
    harness/test_comm_facade_usage.cpp
)
target_link_libraries(test_comm_facade_usage PRIVATE executor GTest::GTest GTest::Main)
if(MSVC)
    target_compile_options(test_comm_facade_usage PRIVATE /utf-8)
endif()
if(EXECUTOR_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_comm_facade_usage PRIVATE
            --coverage
            -fprofile-arcs
            -ftest-coverage
        )
        target_link_options(test_comm_facade_usage PRIVATE --coverage)
    endif()
endif()
add_test(NAME test_comm_facade_usage COMMAND test_comm_facade_usage)
set_tests_properties(test_comm_facade_usage PROPERTIES
    LABELS "unit;harness"
    TIMEOUT 30
)

# 为每个测试文件创建可执行文件
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # Keep CUDA memory validation runnable on hosts without a CUDA SDK.  The
    # executor source uses its fallback CUDA types there, so this target can
    # exercise the pre-driver validation gate without linking a CUDA runtime.
    if(TEST_NAME STREQUAL "test_gpu_memory_validation" AND NOT EXECUTOR_CUDA_ENABLED)
        target_sources(${TEST_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/executor/gpu/cuda_executor.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/executor/gpu/cuda_loader.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/executor/gpu/gpu_memory_manager.cpp
        )
    endif()

    if(TEST_NAME STREQUAL "test_gpu_loader_failure")
        if(NOT EXECUTOR_CUDA_ENABLED)
            target_sources(${TEST_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/executor/gpu/cuda_loader.cpp
            )
        endif()
        if(NOT EXECUTOR_OPENCL_ENABLED)
            target_sources(${TEST_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/executor/gpu/opencl_loader.cpp
            )
        endif()
    endif()

    # 链接 executor 库和 GTest
    target_link_libraries(${TEST_NAME} PRIVATE executor GTest::GTest GTest::Main)
    
    # 如果启用了 GPU 支持，传递宏定义给测试
    if(EXECUTOR_ENABLE_GPU)
        target_compile_definitions(${TEST_NAME} PRIVATE EXECUTOR_ENABLE_GPU)
    endif()

    if(TEST_NAME STREQUAL "test_thread_pool")
        target_compile_definitions(${TEST_NAME} PRIVATE EXECUTOR_THREAD_POOL_TEST_HOOKS)
    endif()

    if(TEST_NAME STREQUAL "test_multithread_mpsc")
        target_compile_definitions(${TEST_NAME} PRIVATE LOCKFREE_QUEUE_DEBUG_HOOKS)
    endif()
    
    # 如果是 CUDA/GPU 测试，添加 CUDA 头文件支持（不链接库，使用动态加载）
    if((TEST_NAME MATCHES ".*cuda.*" OR TEST_NAME MATCHES ".*gpu_memory_validation.*" OR TEST_NAME MATCHES ".*gpu_executor.*" OR TEST_NAME MATCHES ".*gpu_multi_device.*" OR TEST_NAME MATCHES ".*gpu_memory_manager.*" OR TEST_NAME MATCHES ".*gpu_monitor.*" OR TEST_NAME MATCHES ".*gpu_dep_async.*" OR TEST_NAME MATCHES ".*gpu_wait_for_completion.*" OR TEST_NAME MATCHES ".*unified_memory.*" OR TEST_NAME STREQUAL "test_gpu_loader_failure") AND EXECUTOR_CUDA_ENABLED)
        if(CUDA_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
        elseif(CUDAToolkit_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endif()
        # 不链接CUDA库，使用动态加载
        # CudaLoader会在运行时动态搜索和加载CUDA DLL
    endif()

    # 如果是 OpenCL 测试，添加 OpenCL 头文件支持（不链接库，使用动态加载）
    if((TEST_NAME MATCHES ".*opencl.*" OR TEST_NAME MATCHES ".*gpu_executor.*") AND EXECUTOR_OPENCL_ENABLED)
        if(OpenCL_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} SYSTEM PRIVATE ${OpenCL_INCLUDE_DIRS})
        endif()
        target_compile_definitions(${TEST_NAME} PRIVATE EXECUTOR_ENABLE_OPENCL)
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
    if(TEST_NAME STREQUAL "test_gpu_memory_validation")
        add_test(NAME CudaExecutorMemoryValidation COMMAND ${TEST_NAME})
    else()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endif()
    
    # 根据测试类型设置 LABELS 和 TIMEOUT
    # 单元测试
    if(TEST_NAME STREQUAL "test_lockfree_worker_queue_concurrent_steal")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit;concurrency;lockfree"
            TIMEOUT 60
        )
    elseif(TEST_NAME MATCHES "^test_comm_(channel|double_buffer|mailbox|phase_gate)$")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit;concurrency;lockfree"
            TIMEOUT 60
        )
    elseif(TEST_NAME STREQUAL "test_comm_topic")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit;concurrency"
            TIMEOUT 60
        )
    elseif(TEST_NAME STREQUAL "test_comm_realtime_memory")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit;realtime;lockfree"
            TIMEOUT 60
        )
    elseif(TEST_NAME STREQUAL "test_comm_types")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit;lockfree"
            TIMEOUT 30
        )
    elseif(TEST_NAME STREQUAL "test_self_stop_handoff")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit;concurrency"
            TIMEOUT 30
        )
    elseif(TEST_NAME MATCHES "^test_(dispatcher_resize_under_load|thread_pool_resize_uaf|thread_pool_resize_status|concurrent_stop_submit)$")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "integration;concurrency;thread_pool"
            TIMEOUT 120
        )
    elseif(TEST_NAME MATCHES "^test_(util|task|thread_pool|thread_pool_empty_task_rejection|statistics_collector|task_monitor|cuda_executor|gpu_memory_manager|push_task_ex_default)$")
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
    # 优化基线基准测试（测量优化路线图中各优化目标的当前性能）
    elseif(TEST_NAME STREQUAL "benchmark_optimization_baseline")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "benchmark;optimization"
            TIMEOUT 120
        )
    # 批量提交基准测试（benchmark_batch_* 系列）
    elseif(TEST_NAME MATCHES "^benchmark_batch_.*")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "benchmark;batch"
            TIMEOUT 120
        )
    # 无锁任务执行器基准测试（高吞吐量压测，需要更长超时）
    elseif(TEST_NAME STREQUAL "benchmark_lockfree_task_executor")
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "benchmark"
            TIMEOUT 120
        )
    elseif(TEST_NAME STREQUAL "test_gpu_memory_validation")
        set_tests_properties(CudaExecutorMemoryValidation PROPERTIES
            LABELS "unit"
            TIMEOUT 30
        )
    # 默认：单元测试
    else()
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "unit"
            TIMEOUT 30
        )
    endif()

    if(EXECUTOR_ENABLE_TSAN AND TEST_NAME MATCHES "^test_(comm_(channel|double_buffer|mailbox|phase_gate|topic)|thread_pool|thread_pool_resize_uaf|thread_pool_resize_status|thread_pool_monitor_concurrent_set|dispatcher_resize_under_load|concurrent_stop_submit|load_balancer_concurrent_resize|opencl_lifecycle|self_stop_handoff)$")
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 180
        )
    endif()
endforeach()

add_executable(test_dispatch_task_fallback test_dispatch_task_fallback.cpp)
target_link_libraries(test_dispatch_task_fallback PRIVATE executor GTest::GTest GTest::Main)
target_include_directories(test_dispatch_task_fallback PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
if(MSVC)
    target_compile_options(test_dispatch_task_fallback PRIVATE /utf-8)
endif()
if(EXECUTOR_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_dispatch_task_fallback PRIVATE
            --coverage
            -fprofile-arcs
            -ftest-coverage
        )
        target_link_options(test_dispatch_task_fallback PRIVATE --coverage)
    endif()
endif()
add_test(NAME test_dispatch_task_fallback COMMAND test_dispatch_task_fallback)
set_tests_properties(test_dispatch_task_fallback PROPERTIES
    LABELS "unit"
    TIMEOUT 30
)

add_executable(test_object_pool_double_release_detect test_object_pool_double_release_detect.cpp)
target_link_libraries(test_object_pool_double_release_detect PRIVATE executor GTest::GTest GTest::Main)
target_include_directories(test_object_pool_double_release_detect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
if(MSVC)
    target_compile_options(test_object_pool_double_release_detect PRIVATE /utf-8)
endif()
if(EXECUTOR_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_object_pool_double_release_detect PRIVATE
            --coverage
            -fprofile-arcs
            -ftest-coverage
        )
        target_link_options(test_object_pool_double_release_detect PRIVATE --coverage)
    endif()
endif()
add_test(NAME test_object_pool_double_release_detect COMMAND test_object_pool_double_release_detect)
set_tests_properties(test_object_pool_double_release_detect PROPERTIES
    LABELS "unit"
    TIMEOUT 30
)

add_executable(test_object_pool_double_release_detect_ndebug test_object_pool_double_release_detect.cpp)
target_link_libraries(test_object_pool_double_release_detect_ndebug PRIVATE executor GTest::GTest GTest::Main)
target_include_directories(test_object_pool_double_release_detect_ndebug PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
target_compile_definitions(test_object_pool_double_release_detect_ndebug PRIVATE NDEBUG)
if(MSVC)
    target_compile_options(test_object_pool_double_release_detect_ndebug PRIVATE /utf-8)
endif()
if(EXECUTOR_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_object_pool_double_release_detect_ndebug PRIVATE
            --coverage
            -fprofile-arcs
            -ftest-coverage
        )
        target_link_options(test_object_pool_double_release_detect_ndebug PRIVATE --coverage)
    endif()
endif()
add_test(NAME test_object_pool_double_release_detect_ndebug COMMAND test_object_pool_double_release_detect_ndebug)
set_tests_properties(test_object_pool_double_release_detect_ndebug PROPERTIES
    LABELS "unit"
    TIMEOUT 30
)

# 如果使用 Google Test 或其他测试框架，可以在这里配置
# 例如：
# find_package(GTest REQUIRED)
# target_link_libraries(test_name PRIVATE executor GTest::GTest GTest::Main)

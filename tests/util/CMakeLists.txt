add_executable(test_lockfree_mpsc_arm_stress test_lockfree_mpsc_arm_stress.cpp)

target_link_libraries(test_lockfree_mpsc_arm_stress PRIVATE
    executor
    GTest::GTest
    GTest::Main
)

target_include_directories(test_lockfree_mpsc_arm_stress PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

if(MSVC)
    target_compile_options(test_lockfree_mpsc_arm_stress PRIVATE /utf-8)
endif()

if(EXECUTOR_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_lockfree_mpsc_arm_stress PRIVATE
            --coverage
            -fprofile-arcs
            -ftest-coverage
        )
        target_link_options(test_lockfree_mpsc_arm_stress PRIVATE --coverage)
    endif()
endif()

add_test(NAME test_lockfree_mpsc_arm_stress COMMAND test_lockfree_mpsc_arm_stress)

set_tests_properties(test_lockfree_mpsc_arm_stress PROPERTIES
    LABELS "unit;stress;lockfree"
    TIMEOUT 120
)

# tests/CMakeLists.txt
# 配置测试框架和测试可执行文件

# 注意：当前阶段还没有测试文件，后续阶段会添加
# 此文件为测试框架做好准备

# 收集测试源文件
file(GLOB TEST_SOURCES "*.cpp")

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
    
    # 添加测试
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# 如果使用 Google Test 或其他测试框架，可以在这里配置
# 例如：
# find_package(GTest REQUIRED)
# target_link_libraries(test_name PRIVATE executor GTest::GTest GTest::Main)

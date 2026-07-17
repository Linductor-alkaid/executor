add_executable(tutorial_01_first_task 01_first_task.cpp)
target_link_libraries(tutorial_01_first_task PRIVATE executor)

if(BUILD_TESTING)
    add_test(NAME tutorial_01_first_task COMMAND tutorial_01_first_task)
    set_tests_properties(tutorial_01_first_task PROPERTIES
        LABELS "tutorial;smoke"
        TIMEOUT 15
    )
endif()

# ExecutorConfig.cmake
# 用于 find_package(executor) 支持
# 此文件在安装时会被复制到 ${CMAKE_INSTALL_LIBDIR}/cmake/executor/

include(CMakeFindDependencyMacro)

# 获取当前文件所在目录
get_filename_component(EXECUTOR_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(EXECUTOR_ROOT_DIR "${EXECUTOR_CMAKE_DIR}/../.." ABSOLUTE)

# 设置变量供 find_package 使用
set(EXECUTOR_FOUND TRUE)
set(EXECUTOR_VERSION 0.1.0)
set(EXECUTOR_INCLUDE_DIRS "${EXECUTOR_ROOT_DIR}/include")
set(EXECUTOR_LIBRARY_DIRS "${EXECUTOR_ROOT_DIR}/lib")

# 查找库文件
find_library(EXECUTOR_LIBRARY
    NAMES executor
    PATHS ${EXECUTOR_LIBRARY_DIRS}
    NO_DEFAULT_PATH
)

# 如果找不到，尝试默认路径
if(NOT EXECUTOR_LIBRARY)
    find_library(EXECUTOR_LIBRARY
        NAMES executor
    )
endif()

# 创建导入目标
if(EXECUTOR_LIBRARY AND NOT TARGET executor::executor)
    add_library(executor::executor UNKNOWN IMPORTED)
    set_target_properties(executor::executor PROPERTIES
        IMPORTED_LOCATION "${EXECUTOR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${EXECUTOR_INCLUDE_DIRS}"
    )
endif()

# 检查是否成功找到
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(executor
    FOUND_VAR EXECUTOR_FOUND
    REQUIRED_VARS EXECUTOR_LIBRARY EXECUTOR_INCLUDE_DIRS
    VERSION_VAR EXECUTOR_VERSION
)

# 设置兼容性变量
set(EXECUTOR_LIBRARIES ${EXECUTOR_LIBRARY})
set(EXECUTOR_INCLUDE_DIR ${EXECUTOR_INCLUDE_DIRS})

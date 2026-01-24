# executorConfig.cmake
# 供 find_package(executor) 使用，与 executorTargets.cmake、executorConfigVersion.cmake 一同安装到
# ${CMAKE_INSTALL_LIBDIR}/cmake/executor/

include(CMakeFindDependencyMacro)
find_dependency(Threads REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/executorTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/executorConfigVersion.cmake")

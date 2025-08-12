cmake_minimum_required(VERSION 3.15)
project(sprk_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 默认 Release 构建
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 找 Conda 的 Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 加入本地 pybind11
add_subdirectory(pybind11)

# 找 Eigen3（MSYS2 的 clang64 安装 eigen3 即可）
find_package(Eigen3 REQUIRED NO_MODULE)

# 构建 Python 模块
pybind11_add_module(sprk src/sprk.cpp)

target_include_directories(sprk PRIVATE ${EIGEN3_INCLUDE_DIR})

# Windows 需要 .pyd 后缀
if(WIN32)
    set_target_properties(sprk PROPERTIES SUFFIX ".pyd")
endif()

target_link_libraries(sprk PRIVATE Python3::Python)

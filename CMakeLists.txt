cmake_minimum_required(VERSION 3.0)
project(transformer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 LibTorch 路径
set(LIBTORCH_PATH "D:/浏览器下载/libtorch-win-shared-with-deps-debug-2.6.0+cpu/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})

find_package(Torch REQUIRED)
message(STATUS "Torch library status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

# 主程序
add_executable(transformer main.cpp)
target_link_libraries(transformer PRIVATE "${TORCH_LIBRARIES}")

# 测试程序
add_executable(test_torch test.cpp)
target_link_libraries(test_torch PRIVATE "${TORCH_LIBRARIES}")

# 复制 DLL 文件到输出目录
file(GLOB TORCH_DLLS "${LIBTORCH_PATH}/lib/*.dll")
foreach(DLL ${TORCH_DLLS})
    add_custom_command(TARGET transformer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLL}
        $<TARGET_FILE_DIR:transformer>)
    add_custom_command(TARGET test_torch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLL}
        $<TARGET_FILE_DIR:test_torch>)
endforeach(DLL)
cmake_minimum_required(VERSION 3.15)

project(vulkan-test CXX)

set(CMAKE_CXX_STANDARD 20)

find_package( Vulkan REQUIRED )
find_package( glfw3 REQUIRED )
find_package( glm REQUIRED )

find_program(glslc_executable NAMES glslc)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

set(SHADERS shader.vert shader.frag)

foreach(SHADER ${SHADERS})
    add_custom_command(
            OUTPUT ${SHADER_BINARY_DIR}/${SHADER}.spv
            COMMAND ${glslc_executable} ${SHADER_SOURCE_DIR}/${SHADER} -o ${SHADER_BINARY_DIR}/${SHADER}.spv
            DEPENDS ${SHADER_SOURCE_DIR}/${SHADER}
    )
    list(APPEND SHADER_BINARIES ${SHADER_BINARY_DIR}/${SHADER}.spv)
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_BINARIES})

add_executable(vulkan-test src/main.cpp)
target_link_libraries( vulkan-test PRIVATE Vulkan::Vulkan glfw glm::glm )

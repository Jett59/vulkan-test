cmake_minimum_required(VERSION 3.15)

project(vulkan-test CXX)

set(CMAKE_CXX_STANDARD 20)

find_package( Vulkan REQUIRED )
find_package( glfw3)
find_package( glm )

IF(NOT glfw3_FOUND)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)  
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)  
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)  
    add_subdirectory(third-party/glfw)
ENDIF()

IF(NOT glfw3_FOUND)
    add_subdirectory(third-party/glm)
ENDIF()

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

if(MSVC)
  target_compile_options(vulkan-test PRIVATE /W4 /WX)
else()
  target_compile_options(vulkan-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

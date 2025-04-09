cmake_minimum_required(VERSION 3.20)

project(generative-ai-inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(lib)

file(GLOB_RECURSE SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(generative-ai-inference ${SOURCE_FILES})

target_include_directories(generative-ai-inference PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(generative-ai-inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_include_directories(generative-ai-inference PRIVATE ${PROJECT_SOURCE_DIR}/lib/sentencepiece/src)

target_link_libraries(generative-ai-inference PRIVATE sentencepiece ${ONNXRUNTIME_LIB})

add_custom_command(TARGET generative-ai-inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        $<TARGET_FILE_DIR:generative-ai-inference>
)


add_custom_command(TARGET generative-ai-inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/ressources"
        $<TARGET_FILE_DIR:generative-ai-inference>/ressources
)
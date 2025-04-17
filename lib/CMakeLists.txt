cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

FetchContent_Declare(
    onnxruntime_nupkg
        URL https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.21.0
        DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_GetProperties(onnxruntime_nupkg)
if(NOT onnxruntime_nupkg_POPULATED)
    FetchContent_Populate(onnxruntime_nupkg)
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_nupkg_SOURCE_DIR}/build/native/include" PARENT_SCOPE)

if(WIN32)
    set(ONNXRUNTIME_LIB_DIR "${onnxruntime_nupkg_SOURCE_DIR}/runtimes/win-x64/native")
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll" PARENT_SCOPE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib" PARENT_SCOPE)
elseif(APPLE)
    set(ONNXRUNTIME_LIB_DIR "${onnxruntime_nupkg_SOURCE_DIR}/runtimes/osx-x64/native")
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib" PARENT_SCOPE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib" PARENT_SCOPE)
elseif(UNIX)
    set(ONNXRUNTIME_LIB_DIR "${onnxruntime_nupkg_SOURCE_DIR}/runtimes/linux-x64/native")
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so" PARENT_SCOPE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so" PARENT_SCOPE)
endif()

add_subdirectory(sentencepiece)

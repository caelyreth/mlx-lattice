cmake_minimum_required(VERSION 3.27)

if(NOT SKBUILD)
  message(
    FATAL_ERROR
      "This project is built with scikit-build-core.")
endif()

# MARK: - project

project(
  mlx_lattice
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# MARK: - language

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MARK: - python

if(DEFINED ENV{VIRTUAL_ENV}
   AND DEFINED Python_EXECUTABLE
   AND NOT EXISTS "${Python_EXECUTABLE}")
  set(Python_EXECUTABLE
      "$ENV{VIRTUAL_ENV}/bin/python"
      CACHE FILEPATH "Active virtual environment Python executable" FORCE)
endif()

find_package(
  Python 3.12
  REQUIRED
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

# MARK: - nanobind

if(NOT DEFINED nanobind_DIR OR NOT EXISTS "${nanobind_DIR}")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_DIR
    COMMAND_ERROR_IS_FATAL ANY)
  set(nanobind_DIR
      "${nanobind_DIR}"
      CACHE PATH "nanobind CMake package directory" FORCE)
endif()
find_package(nanobind CONFIG REQUIRED)

# MARK: - native

add_subdirectory(native)

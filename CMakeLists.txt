cmake_minimum_required(VERSION 2.8)
project(evaluator CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

set(CMAKE_BUILD_TYPE "Release")

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  message("Using Intel compiler flags")
  set(CMAKE_CXX_FLAGS "-std=c++14 -xhost -qopenmp")
  set(CMAKE_EXE_LINKER_FLAGS "-qopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
else()
  message("Using gcc/clang compiler flags")
  set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -fopenmp")
  set(CMAKE_EXE_LINKER_FLAGS "-fopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(${MPI_INCLUDE_PATH})
include_directories(include)
message("${MPI_INCLUDE_PATH}")
message("${MPI_LIBRARIES}")

add_executable(test_program src/test_program.cc)
target_link_libraries(test_program ${MPI_LIBRARIES})

add_executable(test_dg_precomputed src/test_dg_precomputed.cc)
target_link_libraries(test_dg_precomputed ${MPI_LIBRARIES})

add_executable(test_dg_integrate src/test_dg_integrate.cc)
target_link_libraries(test_dg_integrate ${MPI_LIBRARIES})

add_executable(test_program_vect src/test_program_vect.cc)
target_link_libraries(test_program_vect ${MPI_LIBRARIES})

add_executable(test_gauss src/test_gauss.cc)
add_executable(test_polynomial src/test_polynomial.cc)
add_executable(test_multiplication src/test_multiplication.cc)

find_package(Likwid)

if (LIKWID_FOUND)
  message("${LIKWID_INCLUDE_PATH}")
  include_directories(${LIKWID_INCLUDE_PATH})

  add_executable(test_program_likwid src/test_program.cc)
  target_link_libraries(test_program_likwid ${MPI_LIBRARIES})
  target_link_libraries(test_program_likwid ${LIKWID_LIBRARIES})
  target_compile_definitions(test_program_likwid PRIVATE LIKWID_PERFMON)

  add_executable(test_dg_precomputed_likwid src/test_dg_precomputed.cc)
  target_link_libraries(test_dg_precomputed_likwid ${MPI_LIBRARIES})
  target_link_libraries(test_dg_precomputed_likwid ${LIKWID_LIBRARIES})
  target_compile_definitions(test_dg_precomputed_likwid PRIVATE LIKWID_PERFMON)

  add_executable(test_dg_integrate_likwid src/test_dg_integrate.cc)
  target_link_libraries(test_dg_integrate_likwid ${MPI_LIBRARIES})
  target_link_libraries(test_dg_integrate_likwid ${LIKWID_LIBRARIES})
  target_compile_definitions(test_dg_integrate_likwid PRIVATE LIKWID_PERFMON)

  add_executable(test_program_vect_likwid src/test_program_vect.cc)
  target_link_libraries(test_program_vect_likwid ${MPI_LIBRARIES})
  target_link_libraries(test_program_vect_likwid ${LIKWID_LIBRARIES})
  target_compile_definitions(test_program_vect_likwid PRIVATE LIKWID_PERFMON)
endif()

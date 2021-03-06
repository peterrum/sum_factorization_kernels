set(LIKWID_INSTALL_ROOT "" CACHE STRING "Location of LIKWID installation")

FIND_PATH(LIKWID_INCLUDE_PATH likwid.h HINTS ${LIKWID_INSTALL_ROOT}/include)
FIND_LIBRARY(LIKWID_LIBRARIES likwid HINTS ${LIKWID_INSTALL_ROOT}/lib)

IF (LIKWID_INCLUDE_PATH AND LIKWID_LIBRARIES)
  SET(LIKWID_FOUND TRUE)
ENDIF (LIKWID_INCLUDE_PATH AND LIKWID_LIBRARIES)


IF (LIKWID_FOUND)
  IF (NOT LIKWID_FIND_QUIETLY)
    MESSAGE(STATUS "Found Likwid: ${LIKWID_LIBRARIES}")
  ENDIF (NOT LIKWID_FIND_QUIETLY)
ELSE (LIKWID_FOUND)
  IF (LIKWID_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find Likwid")
  ENDIF (LIKWID_FIND_REQUIRED)
ENDIF (LIKWID_FOUND)

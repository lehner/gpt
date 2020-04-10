cmake_minimum_required(VERSION 3.14)
project(cgpt)
if(SKBUILD)
	message(STATUS "we are building with scikit-build")
  	find_package(PythonExtensions REQUIRED)
endif()


macro(gridconfig var flag)
	execute_process(COMMAND "grid-config" "${flag}"
		RESULT_VARIABLE gridresult
		OUTPUT_VARIABLE ${var}
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_STRIP_TRAILING_WHITESPACE)
	if(${gridresult} EQUAL 0)
		message(STATUS "grid-config ${flag} ok")
		message(STATUS "     yields ${${var}}")
	else()
		message(FATAL_ERROR "grid-config ${flag} failed")
	endif()
endmacro()

macro(pythonconfig var flag)
	execute_process(COMMAND "python3-config" "${flag}"
		RESULT_VARIABLE pyresult
		OUTPUT_VARIABLE ${var}
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_STRIP_TRAILING_WHITESPACE)
	if(${pyresult} EQUAL 0)
		message(STATUS "python3-config ${flag} ok")
		message(STATUS "        yields ${${var}}")
	else()
		message(FATAL_ERROR "python3-config ${flag} failed")
	endif()
endmacro()

add_library(cgpt MODULE
lib/cgpt/lib/block.cc
lib/cgpt/lib/random.cc
lib/cgpt/lib/expression_gammamul.cc
lib/cgpt/lib/expression_linear_combination.cc
lib/cgpt/lib/expression_mul.cc
lib/cgpt/lib/operators.cc
lib/cgpt/lib/basis.cc
lib/cgpt/lib/eval.cc
lib/cgpt/lib/init.cc
lib/cgpt/lib/util.cc
lib/cgpt/lib/coordinates.cc
lib/cgpt/lib/distribute.cc
lib/cgpt/lib/grid.cc
lib/cgpt/lib/io.cc
lib/cgpt/lib/lattice.cc
lib/cgpt/lib/lib.cc
lib/cgpt/lib/mpi.cc
lib/cgpt/lib/time.cc
lib/cgpt/lib/transform.cc)


gridconfig(gsummary   --summary)
gridconfig(gprefix    --prefix)
gridconfig(ggit       --git)
gridconfig(gcxxflags  --cxxflags)
gridconfig(glibs      --libs)
gridconfig(gldflags   --ldflags)
gridconfig(gcxx       --cxx)

set(CMAKE_CXX_COMPILER ${gcxx})

message(STATUS "configuring with this grid:\n${gsummary}")
message(STATUS "grid git revision is: ${ggit}")

separate_arguments(gcxxflags)
separate_arguments(glibs)
separate_arguments(gldflags)



if(SKBUILD)
	target_compile_options(cgpt PRIVATE "-I${gprefix}/include;${gcxxflags}")
	target_link_options(cgpt PRIVATE "-L${gprefix}/lib;-lGrid;${gldflags};${glibs}")
	target_link_libraries(cgpt "Grid;${glibs}")
	python_extension_module(cgpt)
        message(STATUS "installing cgpt to  ${CMAKE_INSTALL_PREFIX}/${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
	install(TARGETS cgpt LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR})
else()
	pythonconfig(pycflags   --includes)
	pythonconfig(pyldflags  --ldflags)
	pythonconfig(pylibs     --libs)
	separate_arguments(pycflags)
	separate_arguments(pyldflags)
	separate_arguments(pylibs)
	set_target_properties(cgpt PROPERTIES PREFIX "")
	target_include_directories(cgpt PRIVATE ${PYTHON_INCLUDE_DIRS})
	target_compile_options(cgpt PRIVATE "-I${gprefix}/include;${gcxxflags};${pycflags}")
	target_link_options(cgpt PRIVATE "-L${gprefix}/lib;-lGrid;${gldflags};${glibs};${pyldflags};${pylibs}")
	target_link_libraries(cgpt "Grid;${glibs};${pyldflags};${pylibs}")
	install(TARGETS cgpt LIBRARY DESTINATION ".")
endif()

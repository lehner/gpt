cmake_minimum_required(VERSION 3.13.4)
project(cgpt)
if(SKBUILD)
    message(STATUS "we are building with scikit-build")
    find_package(PythonExtensions REQUIRED)
    find_package(NumPy REQUIRED)
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

macro(pythoninter var cmd)
    execute_process(COMMAND "python3" "-c" "${cmd}"
        RESULT_VARIABLE pyresult
        OUTPUT_VARIABLE ${var}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
    if(${pyresult} EQUAL 0)
        message(STATUS "python3 ${cmd} ok")
        message(STATUS " yields ${${var}}")
    else()
        message(FATAL_ERROR "python3 ${cmd} failed")
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
lib/cgpt/lib/transform.cc
)


gridconfig(gsummary   --summary)
gridconfig(gprefix    --prefix)
gridconfig(ggit       --git)
gridconfig(gcxxflags  --cxxflags)
gridconfig(glibs      --libs)
gridconfig(gldflags   --ldflags)
gridconfig(gcxx       --cxx)

message(STATUS "configuring with this grid:\n${gsummary}")
message(STATUS "grid git revision is: ${ggit}")

separate_arguments(gcxxflags)
separate_arguments(glibs)
separate_arguments(gldflags)

# Super ugly hack to work around problems with Cmake, Grid and nvcc:
# Cmake appends -fPIC for libaries, but nvcc does not understand -fPIC
# python appends -fopenmp to flags, but nvcc does not unterstand -fopenmp
# Cmake uses compiler for linking, but grids adds "-x cu" to compiler flags,
# that prevents nvcc from using object files.  Solution:
# Write a custom compiler script, that filters command line options.
# Beware: breaks easily
set(compname "${CMAKE_BINARY_DIR}/customcc")
file(WRITE  ${compname} "#!/bin/bash\n")
file(APPEND ${compname} "x=\"\$@\"\n")
file(APPEND ${compname} "subs=\"\"\n")
file(APPEND ${compname} "skipnext=0\n")
file(APPEND ${compname} "for name in \"\$@\"; do\n")
file(APPEND ${compname} "    if test \$skipnext -eq 1\n")
file(APPEND ${compname} "    then subs=\"\$subs \$name\"\n")
file(APPEND ${compname} "    else\n")
file(APPEND ${compname} "        if test \"\$name\" = '-fPIC'\n")
file(APPEND ${compname} "        then subs=\"$subs -Xcompiler -fPIC\"\n")
file(APPEND ${compname} "        else\n")
file(APPEND ${compname} "            if test \"\$name\" = '-fopenmp'\n")
file(APPEND ${compname} "            then\n")
file(APPEND ${compname} "                subs=\"\$subs -Xcompiler -fopenmp\"\n")
file(APPEND ${compname} "            else\n")
file(APPEND ${compname} "                subs=\"\$subs \$name\"\n")
file(APPEND ${compname} "            fi\n")
file(APPEND ${compname} "        fi\n")
file(APPEND ${compname} "    fi\n")
file(APPEND ${compname} "    if test \"\$name\" = '-Xcompiler'\n")
file(APPEND ${compname} "    then skipnext=1\n")
file(APPEND ${compname} "    else skipnext=0\n")
file(APPEND ${compname} "    fi\n")
file(APPEND ${compname} "    \n")
file(APPEND ${compname} "    \n")
file(APPEND ${compname} "done\n")
file(APPEND ${compname} "if expr \"\$subs\" : \".* -c \"; then\n")
file(APPEND ${compname} "    xpr=\"${gcxx} \$subs\"\n")
file(APPEND ${compname} "else\n")
file(APPEND ${compname} "    xpr=\$( echo \"${gcxx} \$subs\" | sed -e 's/-x .*cu//'g)\n")
file(APPEND ${compname} "fi\n")
file(APPEND ${compname} "echo \"CUSTOM COMPILING \$subs\" 1>&2\n")
file(APPEND ${compname} "eval \"\${xpr}\"\n")
execute_process(COMMAND chmod u+x ${compname})
if(${gcxx} MATCH "^nvcc")
    set(CMAKE_CXX_COMPILER ${compname})
endif()

if(SKBUILD)
    target_compile_options(cgpt PRIVATE
        "-I${gprefix}/include"
        ${gcxxflags}
        "-I${NumPy_INCLUDE_DIRS}")
    target_link_options(cgpt PRIVATE
        "-L${gprefix}/lib;-lGrid;${gldflags};${glibs}")
    target_link_libraries(cgpt "Grid;${glibs}")
    python_extension_module(cgpt)
    install(TARGETS cgpt LIBRARY DESTINATION "lib/wrap")
else()
    pythonconfig(pycflags   --includes)
    pythonconfig(pyldflags  --ldflags)
    pythonconfig(pylibs     --libs)
    pythoninter(numpyinclude
        "import numpy; print('-I'+numpy.get_include())")
    separate_arguments(pycflags)
    separate_arguments(pyldflags)
    separate_arguments(pylibs)
    set_target_properties(cgpt PROPERTIES PREFIX "")
    target_compile_options(cgpt PRIVATE
        "-I${gprefix}/include;${gcxxflags}"
        ${pycflags}
        ${numpyinclude})
    target_link_options(cgpt PRIVATE
        "-L${gprefix}/lib;-lGrid;${gldflags};${glibs}"
        ${pyldflags}
        ${pylibs})
    target_link_libraries(cgpt "Grid;${glibs};${pyldflags};${pylibs}")
    install(TARGETS cgpt LIBRARY DESTINATION ".")
endif()
# vim:expandtab:sts=4:shiftwidth=4

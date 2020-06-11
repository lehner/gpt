#                                               -*- Autoconf -*-
#
# Authors: Stefan Solbrig 2020
#
# You can specifiy a minimal autoconf version here, but you don't have to
AC_PREREQ([2.69])

# Specifiy a minimal python version here
AC_DEFUN([PYTHON_MIN_VERSION],[3.6])

# project name, version and maintainer's address
AC_INIT([gpt],[0.1],[christoph.lehner@ur.de])

# Use automake
# the next macro takes a space separated list of arguments.
# First argument is the minimal automake version. foreign means that
# you don't follow GNU standards (ChangeLog COPYING files etc.)
# and -Wall turns on all automake warnings.
AM_INIT_AUTOMAKE([1.15.1 foreign subdir-objects -Wall])

# separate directory for m4 stuff (libtools prefers this)
AC_CONFIG_MACRO_DIR([m4])

# C++ -Compiler
AC_PROG_CXX

# use configure header
AC_CONFIG_HEADERS([config.h])

# Trick to set empty default value for CXXFLAGS
: ${CXXFLAGS=""}

# archiver (Automake needs this for libraries)
AM_PROG_AR

# need for dynamic libraries (no need for ranlib)
LT_INIT([disable-static])

AC_ARG_ENABLE([cgpt],
              AS_HELP_STRING([--disable-cgpt], [Disable building cgpt]))
AC_ARG_ENABLE([gpt],
              AS_HELP_STRING([--disable-gpt], [Disable instaling gpt]))

AC_ARG_WITH(python,
            AS_HELP_STRING([--with-python=PATH],
                           [Path to Python interpreter. Searches $PATH if only a program name is given]),
            [PYTHON="$withval"], [])

AC_MSG_CHECKING(for Python interpreter)
AS_IF([test x"$PYTHON" = xno],
      [AC_MSG_ERROR([Python is non-optional])],
      [test x"$PYTHON" = x || test x"$PYTHON" = x"yes"],
      [PYTHON="python3"],
      [which "$PYTHON"],
      [],
      [AC_MSG_ERROR([Python interpreter $PYTHON does not exist])])

AM_PATH_PYTHON([PYTHON_MIN_VERSION])

AC_MSG_CHECKING([for the sysconfig Python module])
AS_IF(["$PYTHON" -c "import sysconfig"],
      [AC_MSG_RESULT([yes])],
      [AC_MSG_ERROR([no])])

AC_MSG_CHECKING([for Python include paths])
PYTHON_INCLUDES=`$PYTHON -c "import sysconfig; \
    print(*{'-I'+sysconfig.get_path('include'), '-I'+sysconfig.get_path('platinclude')});"`
AC_MSG_RESULT([$PYTHON_INCLUDES])
AC_SUBST([PYTHON_INCLUDES])

ac_save_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="$ac_save_CPPFLAGS $PYTHON_INCLUDES"
AC_CHECK_HEADER(Python.h,
                [],
                [AS_IF([test x$enable_cgpt != xno],
                       [AC_MSG_ERROR([Missing Python.h])],
                       [])
                ])
CPPFLAGS="$ac_save_CPPFLAGS"

AC_MSG_CHECKING([for the numpy Python module])
AS_IF(["$PYTHON" -c "import numpy"],
      [AC_MSG_RESULT([yes])],
      [AC_MSG_ERROR([no])])

AC_MSG_CHECKING([for numpy include paths])
NUMPY_INCLUDE=`$PYTHON -c "import numpy; \
    print(numpy.get_include());"`
AC_MSG_RESULT([$NUMPY_INCLUDE])
AC_SUBST([NUMPY_INCLUDE])

AC_ARG_WITH(grid,
            AS_HELP_STRING([--with-grid=DIR],
                           [Specify grid installation directory DIR]),
            [GRID_PREFIX="$withval"], [])

AC_ARG_ENABLE([fake-grid],
              [AS_HELP_STRING([--enable-fake-grid],
                              [Fake grid-config. Useful if you just want to do a 'make dist'])],
              [ac_fake_grid=yes],
              [ac_fake_grid=no])

AS_IF([test x"${ac_fake_grid}" = x"yes"],
      [
       AC_SUBST([GRID_CXXFLAGS],[-DFAKEGRID])
       AC_SUBST([GRID_LDFLAGS],[-LFAKEGRID])
       AC_SUBST([GRID_CXX],[gcc])
       AC_SUBST([GRID_CXXLD],[gcc])
       AC_SUBST([GRID_LIBS],[-lFAKEGRID])
       AC_SUBST([GRID_PREFIX],[FAKEGRID])
      ],
      [
       AS_IF([test x"$GRID_PREFIX" = x"no"],
             [],
             [test x"$GRID_PREFIX" = x || test x"$GRID_PREFIX" = x"yes"],
             [AC_PATH_PROG(GRID_CONFIG, [grid-config], [])],
             [AC_PATH_PROG(GRID_CONFIG, [grid-config], [], [$GRID_PREFIX/bin:$PATH])])
       AS_IF([test x"$GRID_CONFIG" = x],
             [
              AS_IF([test x"enable_cgpt" != x"no"],
                    [AC_MSG_ERROR([cannot find grid-config])],
                    [])
             ],
             [
              AC_SUBST([GRID_CXXFLAGS],[`${GRID_CONFIG} --cxxflags`])
              AC_SUBST([GRID_LDFLAGS],[`${GRID_CONFIG} --ldflags`])
              AC_SUBST([GRID_CXX],[`${GRID_CONFIG} --cxx`])
              AC_SUBST([GRID_CXXLD],[`${GRID_CONFIG} --cxxld`])
              AC_SUBST([GRID_LIBS],  [`${GRID_CONFIG} --libs`])
              AS_IF([test x"$GRID_PREFIX" = x || test x"$GRID_PREFIX" = x"yes"],
                    [AC_SUBST([GRID_PREFIX],[`${GRID_CONFIG} --prefix`])],
                    [])
             ])
       ])

# Unfortunately, not all versions of Grid have --cxxld and it does
# not print to stderr in case of an error. So detect error by
# matching the output to Usage:
AS_IF([expr "${GRID_CXXLD}" : ".*Usage:" > /dev/null],
      [GRID_CXXLD=])

# If GRID_CXXLD is unset or null, use GRID_CXX.
# Set CXX to GRID_CXX if CXX is unset or null. This means, by default
# use Grid's CXX, but user can override it.  Same for GRID_CXXLD.
: ${GRID_CXXLD:=${GRID_CXX}}
: ${CXX:=${GRID_CXX}}
: ${CXXLD:=${GRID_CXXLD}}

# The following code tries to fix nvcc (and possibly other weird
# compilers) that take an -Xcompiler or -Xlinker argument flag.
# These flags are also used by GNU libtool, hence a conflict arises.
# Fix the conflict by 'escaping' the problematic flags.
AC_PROG_SED
AC_DEFUN([fixxer],[# The FiXXer
AS_IF([expr "${$1}" : ".*$2 " > /dev/null],
      [$1=`echo "${$1}" | ${SED} -e "s/$2 /$2 $2 $2 /g"`])])

for token in -Xlinker -Xcompiler
do
    fixxer([GRID_CXXFLAGS],${token})
done

# Makefiles to be generated. These are all *.in  files.
# We have to use subdirs, because for the python files I need to
# use nobase_XYZ.
AC_CONFIG_FILES([Makefile lib/cgpt/Makefile lib/gpt/Makefile])

AM_CONDITIONAL([ENABLE_CGPT], [test x$enable_cgpt != xno])
AM_CONDITIONAL([ENABLE_GPT], [test x$enable_gpt != xno])

# Generate output. This has to be the last command!
AC_OUTPUT

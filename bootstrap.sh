#!/bin/sh

# Generate list of *.cc
echo "Generate list of cc files..."
find lib/cgpt/lib -type f -name '*.cc' | \
    sed                   \
    -e '1i                \
    CGPT_CCFILES =       \\
    '                     \
    -e 's,lib/cgpt/,,'    \
    -e 's,$, \\,'         \
    -e '$ s/ \\$//'       \
    > lib/cgpt/ccfiles.inc

# Generate list of *.h
echo "Generate list of header files..."
find lib/cgpt/lib -type f -name '*.h' | \
    sed                   \
    -e '1i                \
    CGPT_HFILES =        \\
    '                     \
    -e 's,lib/cgpt/,,'    \
    -e 's,$, \\,'         \
    -e '$ s/ \\$//'       \
    > lib/cgpt/hfiles.inc

# Generate list of python files in lib/gpt
echo "Generate list of py files..."
find lib/gpt -type f -name '*.py' | \
    sed                   \
    -e '1i                \
    GPT_PYFILES =        \\
    '                     \
    -e 's,lib/gpt/,,'     \
    -e 's,$, \\,'         \
    -e '$ s/ \\$//'       \
    > lib/gpt/pyfiles.inc

# Generate all autotools files
autoreconf -fvi -Wall

#!/bin/sh

# Generate list of *.cc
echo "Generate list of cc files..."
find lib/cgpt/lib -type f -name '*.cc' -printf 'lib/%P \\\n' | \
    sed '1s/^/CGPT_CCFILES = \\\n/' | \
    sed -e '$ s/ \\$//' \
    > lib/cgpt/ccfiles.inc

# Generate list of *.h
echo "Generate list of header files..."
find lib/cgpt/lib -type f -name '*.h' -printf 'lib/%P \\\n' | \
    sed '1s/^/CGPT_HFILES = \\\n/' | \
    sed -e '$ s/ \\$//' \
    > lib/cgpt/hfiles.inc

# Generate list of python files in lib/gpt
echo "Generate list of py files..."
find lib/gpt -type f -name '*.py' -printf '%P \\\n' | \
    sed '1s/^/GPT_PYFILES = \\\n/' | \
    sed -e '$ s/ \\$//' \
    > lib/gpt/pyfiles.inc

# Generate all autotools files
autoreconf -fvi -Wall

#!/bin/sh

# Generate list of *.cc
echo "Generate list of cc files..."
find lib/cgpt/lib -type f -name '*.cc' | \
sed -e '1i\
CGPT_CCFILES = \\' \
-e 's,$, \\,' | \
sed -e '$s, \\$,,' \
> cgpt_ccfiles.inc

# Generate list of *.cc
echo "Generate list of header files..."
find lib/cgpt/lib -type f -name '*.h' | \
sed -e '1i\
CGPT_HFILES = \\' \
-e 's,$, \\,' | \
sed -e '$s, \\$,,' \
> cgpt_hfiles.inc

# Generate list of python files in lib/gpt
echo "Generate list of py files..."
find lib/gpt -type f -name '*.py' | \
sed -e '1i\
GPT_PYFILES = \\' \
-e 's,$, \\,'  \
-e 's,^lib/,,' | \
sed -e '$s, \\$,,' \
> gpt_pyfiles.inc

# Generate all autotools files
autoreconf -fvi -Wall

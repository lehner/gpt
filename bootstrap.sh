#!/bin/sh

# Generate list of *.cc
echo "Generate list of cc files..."
find lib/cgpt/lib -type f -name '*.cc' | \
sed -e '1i\
CGPT_CCFILES = \\' \
-e 's,$, \\,'  \
-e '$a\
# do not remove this line' \
> cgpt_ccfiles.inc

# Generate list of python files in lib/gpt
echo "Generate list of py files..."
find lib/gpt -type f -name '*.py' | \
sed -e '1i\
GPT_PYFILES = \\' \
-e 's,$, \\,'  \
-e 's,^lib/,,' \
-e '$a\
# do not remove this line' \
> gpt_pyfiles.inc

# Generate all autotools files
autoreconf -fvi -Wall

#
# Add lib to python path
#
lib="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../lib" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=${lib}:$PYTHONPATH

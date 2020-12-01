#
# Add lib to python path (without ":" if empty beforehand)
#
lib="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../lib/build" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=${lib}${PYTHONPATH:+:${PYTHONPATH}}

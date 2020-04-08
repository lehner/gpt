# How to use gpt with a pythonic setup or via cmake

## Prerequisites

All the methods listed below rely on:

* a `grid-config` in your path, from a Grid that was
  built with the `-fPIC` flag,
* a `python3-config` or a python3 installation that
  cmake can find

## Fully automated with pip

You can use pip in the top level directory.
It will read the pyproject.toml file and pulls all
necessary dependencies (like skbuild, ninja, numpy)

    pip install --user --verbose `pwd`


## Python default with setuptools


Just run

    python3 setup.py --install [--user|--prefix=...]

and it will build gpt and cgpt and install it in
python-typical directories.

In this case, you can influence the unterlying
cmake and ninja (or make) like this:

    python3 setup.py --build  -- [cmake flags] -- [ninja/make flags]
    python3 setup.py --install [--user|--prefix=...]

The `--` separates the flags for python, cmake and ninja (or make).
For example, to get very verbose output:

    python3 setup.py --build -- -DCMAKE_VERBOSE_MAKEFILE=TRUE -GNinja -- -v -d stats -j 20

This will set the cmake variable CMAKE_VERBOSE_MAKEFILE to TRUE,
force cmake to use the Ninja backend,
and start ninja in verbose mode with 20 threads and statistics output.

In this case, you should have the prerequistes installed:

* numpy
* scikit-build
* cmake
* ninja, if you want parallel builds


## Just build cgpt

If you just want to build cgpt (e.g., because you like to run gpt
from the source directory for development) you can just build
with with cmake.

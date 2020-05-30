# How to use gpt with a pythonic setup or via cmake

## Prerequisites

All the methods listed below rely on:

* a `python3-config` or a python3 installation that
  autotools can find

## Standard build with GNU autotools

The build follows the default procedure:

    cd /path/to/source
    ./bootstrap.sh
    mkdir /path/to/builddir
    cd    /path/to/builddir
    /path/to/source/configure --with-grid=/path/to/grid/installdir --prefix=/path/to/installdir
    make -j7
    make install

This will take care of the Python specific paths as well,
e.g., using `--prefix=$HOME/.local`  will install it in
`$HOME/.local/~/.local/lib/python3.7/site-packages`
(Paths might differ slightly, depending on your Linux distro and
your Python distro.)

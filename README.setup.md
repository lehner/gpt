# GPT Setup Instructions

## Prerequisites

All the methods listed below rely on:

* Specified Grid installation (--with-grid) has to be built with the `-fPIC` flag.  GPT is developed with the feature/gpt branch of https://github.com/lehner/Grid.
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

You may, e.g., decide to install GPT in your home directory
using `--prefix=$HOME/.local`.

## Alternative make system
You may also decide to use the `make` script in lib/cgpt for rapid development.  In this case
use
```bash
source gpt/scripts/source.sh
```
to properly set the Python environment.

# GPT Setup Instructions

## Prerequisites

All the methods listed below rely on:

* Specified Grid installation (--with-grid) has to be built with the `-fPIC` flag.  GPT is developed with the feature/gpt branch of https://github.com/lehner/Grid.
* a `python3-config` or a python3 installation that
  autotools can find

## Bootstrap

GPT includes scripts to automatically download and setup GPT and its dependencies
for common architectures.  These scripts are listed in gpt/scripts/bootstrap
and should be invoked from the gpt directory, e.g., as
```bash
git clone https://github.com/lehner/gpt
cd gpt
scripts/bootstrap/debian10.clang.avx2.no-mpi
```

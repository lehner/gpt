# GPT Setup Instructions

## Prerequisites

* Grid based on the feature/gpt branch of https://github.com/lehner/Grid
* Python 3.6 or newer

## Bootstrap

GPT includes scripts to automatically download and setup GPT and its dependencies
for common architectures.  These scripts are listed in [gpt/scripts/bootstrap](https://github.com/lehner/gpt/tree/master/scripts/bootstrap)
and should be invoked from the gpt directory, e.g., as
```bash
git clone https://github.com/lehner/gpt
cd gpt
scripts/bootstrap/debian10.clang.avx2.no-mpi
```

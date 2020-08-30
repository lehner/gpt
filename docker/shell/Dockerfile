ARG BASE_IMAGE=gptdev/base

FROM ${BASE_IMAGE} AS build

ARG COMPILER=gcc
ARG MPI=none

# FIXME: actually we currently do not need the libopenmpi3 dependencies, but
#        the cgpt.so compiled with clang-9 is linking against libmpi{,_cxx}.so
#        therefore for now we still need this dependency
RUN \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        libfftw3-double3 libfftw3-single3 libomp5-9 libmpfr6 libopenmpi3 && \
    rm -rf /var/lib/apt/lists/*

COPY gpt-packages/python-gpt-Linux-python-3.8-${COMPILER}-${MPI}.deb /app/

COPY gpt-packages/tutorials/ /app/tutorials/
COPY gpt-packages/tests/ /app/tests/
COPY gpt-packages/benchmarks/ /app/benchmarks/
COPY gpt-packages/applications/ /app/applications/

RUN \
    dpkg -i python-gpt-Linux-python-3.8-${COMPILER}-${MPI}.deb && \
    rm python-gpt-Linux-python-3.8-${COMPILER}-${MPI}.deb

RUN groupadd -r gpt && \
    useradd --no-log-init -m -r -g gpt gpt && \
    adduser gpt sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

FROM scratch
COPY --from=build / /

WORKDIR /app

USER gpt
VOLUME /gpt-code

ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:${PYTHONPATH}"
ENV PATH="/app/bin:/home/gpt/.local/bin:${PATH}"

CMD ["/bin/bash"]

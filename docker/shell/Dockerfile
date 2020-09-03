ARG BASE_IMAGE=gptdev/base

FROM ${BASE_IMAGE} AS build

ARG COMPILER=gcc
ARG MPI=none

RUN \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        libfftw3-double3 libfftw3-single3 libomp5-9 libmpfr6 && \
    rm -rf /var/lib/apt/lists/*

COPY gpt-packages/python-gpt-Linux-python-3.8-${COMPILER}-${MPI}.deb /app/

RUN \
    dpkg -i python-gpt-Linux-python-3.8-${COMPILER}-${MPI}.deb && \
    rm python-gpt-Linux-python-3.8-${COMPILER}-${MPI}.deb

RUN groupadd -r gpt && \
    useradd --no-log-init -m -r -g gpt gpt && \
    adduser gpt sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

FROM scratch
COPY --from=build / /

RUN sudo mkdir /gpt-code && sudo chown gpt:gpt /gpt-code

WORKDIR /gpt-code

VOLUME /gpt-code

COPY tests/ /gpt-code/tests/
COPY benchmarks/ /gpt-code/benchmarks/
COPY applications/ /gpt-code/applications/

USER gpt

ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:${PYTHONPATH}"
ENV PATH="/gpt-code/bin:/home/gpt/.local/bin:${PATH}"

CMD ["/bin/bash"]

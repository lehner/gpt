ARG COMPILER=gcc
ARG MPI=none
ARG BASE_IMAGE=gptdev/shell

FROM ${BASE_IMAGE}:${COMPILER}-${MPI}

RUN sudo pip3 install --no-cache-dir notebook matplotlib && \
    sudo mkdir /notebooks && sudo chown gpt:gpt /notebooks

COPY tutorials/ /notebooks/tutorials/

VOLUME /notebooks

ENV GPT_QUIET=YES

CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip='*'", "--no-browser", "--notebook-dir=/notebooks"]

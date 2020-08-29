ARG COMPILER=gcc
ARG MPI=none
ARG BASE_IMAGE=gptdev/shell

FROM ${BASE_IMAGE}:${COMPILER}-${MPI}

RUN sudo pip3 install --no-cache-dir notebook && \
    sudo mkdir /app/notebooks && sudo chown gpt:gpt /app/notebooks

VOLUME /app/notebooks

CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip='*'", "--no-browser", "--notebook-dir=/app"]

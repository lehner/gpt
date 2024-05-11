FROM ubuntu:22.04

WORKDIR /app

RUN \
    apt-get update && \
    apt-get install --no-install-recommends -y python3.10 python3-pip sudo && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir numpy

ENV PATH="/app/bin:${PATH}"
ENV TZ="Europe/Berlin"

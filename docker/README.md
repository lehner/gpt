# GPT - Docker images
In addition to the notebook docker image described in the main [README](../README.md),
we provide the gptdev/shell Docker image which is further described below.
For more information on how to use Docker, please see the [Docker documentation](https://docs.docker.com/get-started/).

### Shell

This image includes a user preinstalled Python 3.8 and GPT setup.  Start the session with
```
docker run --rm -it -v$(pwd):/gpt-code gptdev/shell
```
which mounts the current working directory in the image's working
directory.  Alternatively, you can use a non-persistent version
of the GPT applications, benchmarks, and tests folders by
running
```
docker run --rm -it gptdev/shell
```

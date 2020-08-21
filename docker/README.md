# GPT - Docker images

## Different image types

### Base

This is the base for all gpt Docker images, everything which is shared between all other images, should be included here. This image will probably not useful otherwise.

This image includes the basic tools, these are currently very basic and require some more work, to be really useful, for more information run `gpt` inside the Docker container.

### Play

This image includes a user preinstalled Python 3.8 and gpt setup. You can use the folder `/gpt-code` to include your code you want to run with gpt.

If you are on the host system in the directory where the gpt code is located run:
```
docker run -it -v "$(pwd):/gpt-code" gpt/play
```

### Notebook

If you want to use `gpt` in a Jupyter notebook you should use this image. You can use the folder `/notebooks` to include your current notebooks.

If you are on the host system in the directory where your notebooks are located run:
```
docker run -p 8888:8888 -v "$(pwd):/notebooks" gpt/notebook
```

Connect in the browser to `localhost:8888` and enter the secret key, which should be visible from the console output.
It is also possible to run the command with `-d` in the backround and view the key with docker `docker logs <container id>`. Where the container can be found either from the output when starting the container or with `docker container ls`.

## Tags

For the `gpt/play` and `gpt/notebook` in addition to the `latest` tag, which points to `clang-none` the following tags are available:
```
gcc-none
clang-none
```

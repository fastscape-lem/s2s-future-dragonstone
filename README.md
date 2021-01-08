[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fastscape-lem/s2s-future-dragonstone/master?urlpath=lab)
[![Test notebooks](https://github.com/fastscape-lem/s2s-future-dragonstone/workflows/Test%20notebooks/badge.svg)](https://github.com/fastscape-lem/s2s-future-dragonstone/actions)

# S2S-Future DragonStone

This repository contains a collection of [Jupyter](http://jupyter.org/)
notebooks with examples of using the Fastscape software stack for landscape
evolution modeling and topographic analysis.

These notebooks were developed by Jean Braun in the framework of the first
training program (called DragonStone) of the S2S-Future ITN (Innovative
Training Network) in January 2021.

More information concerning FastScape on this website: https://fastscape.org

- [How to run the notebooks?](#how-to-run-the-notebooks)
    - [Run in the cloud (Binder)](#run-in-the-cloud-binder)
    - [Install and run locally (Docker)](#install-and-run-locally-docker)
    - [Install and run locally (Conda)](#install-and-run-locally-conda)
- [How to contribute?](#how-to-contribute)

## How to run the notebooks?

### Run in the cloud (Binder)

You can run the notebooks in your browser without installing anything thanks to
[binder](https://mybinder.org/). Just follow the link below or click on the
"launch binder" badge above and it will launch remotely a new notebook server
for you:

- [Run on binder](https://mybinder.org/v2/gh/fastscape-lem/s2s-future-dragonstone/master?urlpath=lab)

This service is for demo purpose only, do not rely on it for doing more serious
work.

### Install and run locally (Docker)

[Docker](https://www.docker.com/) images are built automatically for this
repository. Those images provide the whole computing environment, pre-installed
and pre-configured for running the notebooks. The only requirement is to
have Docker installed on your machine. It is available on all platforms
Linux/Windows/Mac and it can be installed from the Docker website or using one
of your platform's package managers.

Run the command below to first pull the latest image:

```bash
$ docker pull fastscape/s2s-future-dragonstone:latest
```

Then run the command below to start the Jupyterlab application from the Docker
container. Replace `dragonstone` by any other name you want to give to your
local container (optional). Also Replace `/path/to/local-notebook-folder` by the
full path to the directory on your machine where you want to create/copy, edit and
permanently store notebooks for this training course.

```bash
$ docker run \
    -it \
    --name dragonstone \
    -p 8888:8888 \
    -v /path/to/local-notebook-folder:/home/jovyan/my-local-folder \
    fastscape/s2s-future-dragonstone \
    jupyter lab --ip 0.0.0.0
```

You can then enter in your browser the url and token provided to start using the
Jupyterlab application.

You may want to copy the `notebooks` folder in your local working folder mounted
in the docker container as `my-local-folder`. Open a terminal in Jupyterlab and
run the following command:

```bash
$ cp -R notebooks my-local-folder/
```

When you are done you can stop and remove the container:

``` bash
$ docker stop dragonstone
$ docker rm dragonstone
```

#### Troubleshooting

*The url I entered in my browser doesn't point to Jupyterlab*

You may already have another application running on localhost using the port
`8888`. Try another port when running the `docker run` command above, e.g.,
using `-p 8889:8888`. You also need to change the port in the entered url
accordingly (e.g., `localhost:8889`).

*The url I entered in my browser gives a page asking for a token*

Copy and paste the token given in the url. If the token is invalid, you may have
another Jupyterlab application already running on your machine. Try using
another port as described above.

Check [Docker's documentation](https://docs.docker.com/) for additional command
line help and options.

### Install and run locally (Conda)

Assuming that you have `git` and [conda](https://conda.io/docs/index.html)
installed, you can install all the packages required to run the notebooks in a
new conda environment using the following commands:

```bash
$ git clone https://github.com/fastscape-lem/s2s-future-dragonstone
$ cd s2s-future-dragonstone
$ conda env create -f environment.yml
$ conda activate s2s-future-dragonstone
```

You also need to install a few Jupyterlab extensions with the following command
(this step won't be required anymore with Jupyterlab >= 3.x):

```bash
$ jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @pyviz/jupyterlab_pyviz \
    dask-labextension \
    ipygany
```

Finally run the command below to start the Jupyterlab application. It should
open a new tab in your browser.

```bash
$ jupyter lab
```

## How to contribute?

Your contribution is welcome! Your can do so by reporting issues, suggesting new
notebook examples or improvements to the current examples.

A few extra steps are required to prepare your contributions. You can first
update the conda environment using the following command:

```bash
$ conda env update -n s2s-future-dragonstone --file environment-dev.yml 
```

This installs a few additional packages like
[pre-commit](https://pre-commit.com/), which is used to ensure that all notebook
cell outputs are cleared before adding or updating notebooks in this git
repository. Run the command below to enable pre-commit (you only need to do this
once):

```bash
$ pre-commit install
```

The script below is useful if you want to ensure that all notebooks are running
without error:

```bash
$ python execute_all_notebooks.py
```

This script (as well as a script to build the Docker image) is run each time you
open or update a pull-request on GitHub.

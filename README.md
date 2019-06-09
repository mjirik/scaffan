# scaffan
Scaffold Analyser

[![Build Status](https://travis-ci.org/mjirik/scaffan.svg?branch=master)](https://travis-ci.org/mjirik/scaffan)
[![Coverage Status](https://coveralls.io/repos/github/mjirik/scaffan/badge.svg?branch=master)](https://coveralls.io/github/mjirik/scaffan?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Application for scaffold analysis.


You may want to use it with [Hamamatsu NDP Viewer](https://www.hamamatsu.com/eu/en/product/type/U12388-01/index.html)

# Install



# Linux

```commandline
conda install -c mjirik -c bioconda -c conda-forge openslide-python lxml imma io3d
```

# Windows

[Install Conda](https://conda.io/miniconda.html) and check "Add Anaconda to my PATH environment variable" 

Run in terminal:
```commandline

conda create -n scaffan -c mjirik -c bioconda -c conda-forge pip scaffan
activate scaffan
pip install openslide-python keras tensorflow
```

### Create icon

```bash
activate scaffan
python -m scaffan install
```

## Update

```commandline
activate scaffan
conda install -c mjirik -c bioconda -c conda-forge -y scaffan 
```

## Run

```commandline
activate scaffan
python -m scaffan gui
```


## Known issues

There are two problems with `openslide` (not with `openslide-python`) package on windows. 
* The package is not in conda channel. This is solved by automatic download of the dll binaries.
* Dll binaries cannot be used together with libxml. There is workaround in scaffan. 
It uses subprocess call to process separately image data and image annotations.

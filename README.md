# scaffan
Scaffold Analyser

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Install



# Linux

```commandline
conda install -c mjirik -c bioconda -c conda-forge openslides-python lxml imma io3d
```

# Windows

[Install Conda](https://conda.io/miniconda.html) and check "Add Anaconda to my PATH environment variable" 

Run in terminal:
```commandline

conda create -n scaffan -c mjirik -c bioconda -c conda-forge python=3.6 pip scaffan
activate scaffan
pip install openslide-python keras tensorflow
```

## Update

```commandline
activate scaffan
conda install -c mjirik -c bioconda -c conda-forge -y scaffan 
```

## Run

```commandline
activate scaffan
python -m scaffan
```


## Known issues

There are two problems with `openslide` (not with `openslide-python`) package on windows. 
* The package is not in conda channel. This is solved by automatic download of the dll binaries.
* Dll binaries cannot be used together with libxml. There is workaround in scaffan. 
It uses subprocess call to process separately image data and image annotations.

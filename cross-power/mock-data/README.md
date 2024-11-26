# Cross-Power Spectrum Estimation

This directory contains information and scripts pertaining to the cross-correlation power spectrum estimation analysis using mock data.

For a detailed description of the MeerKAT cross-power spectrum pipeline see
- [Cunnington+2023a](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.6262C/abstract): cross-power spectrum detection using MeerKAT and the [WiggleZ](https://wigglez.swin.edu.au/site/forward.html) Dark Energy Survey
- [Cunnington+2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.2453C/abstract): "transfer function" validation using simulated data used to estimate power spectrum uncertainties and signal loss
- [MeerKLASS Collaboration 2024](https://arxiv.org/abs/2407.21626): MeerKLASS collaboration paper detailing the 2021 x GAMA cross-correlation power spectrum work which this repo replicates

The files contained in this repo are as follows
| File | Description |
| ---- | ----------- |
| `config.yaml` | [jsonargparse](https://jsonargparse.readthedocs.io/en/latest/) yaml configuration file which contains all the required command line arguments to run the power spectrum analysis |
| `env.yaml` | `conda` environment yaml used to create the Python environment (more details below) |
| `galaxy_cross_mocks.py` | Forked and modified version of the corresponding MeerKLASS [script](https://github.com/meerklass/meerpower/blob/main/allLband/galaxy_cross.py) |
| `mock_data-fits-creation.ipynb` | Jupyter notebook documenting the (minimal) steps used to create the mock datasets |

Please note that the code in this repo currently only supports an analysis of mock L-band [MeerKLASS](https://github.com/meerklass) and [GAMA](https://www.gama-survey.org/) galaxy catalog data.


## Software dependencies

There are three software dependencies required to run the code in this repo:

1. Python
2. MeerKLASS cross-correlation power spectrum code (`meerpower`)
3. Python environment

The following two subsections contain all relevant information on installing these dependencies.

### Python

Running this code requires Python and the ability to build a Python environment.  The suggested approach to building this environment is to install [`miniforge`](https://github.com/conda-forge/miniforge) which, in turn, installs `mamba`.  `mamba` is a package manager which is a drop-in replacement for `conda` but with a much faster and more robust dependency solver.  To install `miniforge`, please see the README on the [`miniforge` GitHub](https://github.com/conda-forge/miniforge) and the section on [installation](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).  As of the time of this writing, the installation is quite light weight and only requires copy/pasting two lines for execution on the command line.  There are instructions for both Unix-like and Windows based systems.  Please note that you may need to restart your terminal for the installation to complete and `mamba` to be initialized.

If you wish to use an existing package manager such as [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), you can do so by replacing all instances of `mamba` with `conda` in subsequent commands.

### meerpower

The MeerKLASS cross-correlation power spectrum code is stored on GitHub at [meerpower](https://github.com/meerklass/meerpower).  Please clone the `meerpower` repository via e.g.
```
git clone https://github.com/meerklass/meerpower
```
It is assumed that this repo will be cloned into the current working directory.  If you clone this repo somewhere else, the configuration yaml, `config.yaml`, will need to be modified accordingly.  Please see the section on _Running the analysis_ below for more details.

### Python environment

To build the required python environment, we will use the Python environment yaml provided in this repo, `env.yaml`.  The Python environment can be created using this yaml file and `mamba` via
```
mamba env create -f env.yaml
```
Recall that, if using `conda`, you can directly replace `mamba` with `conda` in the above example.  Building the environment with `conda` might take longer, however.

Once the Python environment is built, it can be activated via
```
mamba activate gridimp
```


## Input data

Running the MeerKLASS cross-correlation power spectrum code requires a radio image, the pixel counts for each pixel in the image, a galaxy catalog to cross-correlate with, mock radio images, and mock galaxy catalog data.  For this analysis, we will be analyzing mock radio image data and the Galaxy and Mass Assembly ([GAMA](https://www.gama-survey.org/)) galaxy catalog.  These data are 622 MB in size and currently hosted on [google drive](https://drive.google.com/drive/folders/17Y_Crphch_l3Q9kkUibmKa1CLIyWgcmW?usp=sharing).  Please download these data to your desired location on your machine.  It is assumed that these data will be downloaded to the current working directory in a subdirectory called `data/`.  If the data have been downloaded elsewhere, the configuration yaml, `config.yaml`, will need to be updated accordingly.  Please see the section on _Running the analysis_ below for more details.

The descriptions for each of the data files are as follows:

| Item | Size | File name(s) |
| ---- | ---- | ------------ |
| Radio image | 304 MB | Tsky.fits |
| Image pixel counts | 304 MB | Npix_count.fits |
| Mock radio images | 9.2 GB total<br>19 MB each | Tsky_mock*[0-9].npy (500 files) |
| GAMA data | 14 MB | GAMA.fits |

In total, the data are ~10 GB in size.

## Running the analysis

Running the cross-correlation power spectrum analysis can be done via the command line.  The command line interface for the code uses `jsonargparse`, a python package which allows for command line arguments to be passed via the command line directly or parsed via a yaml file.  A pre-configured yaml file has been provided `config.yaml` which assumes that the data have been downloaded to a folder called `data/` in the current working directory, i.e. the data are in `./data/`.

The only command line argument which can be freely modified is `--Nmocks`, or `Nmocks` in `config.yaml`.  The value of `Nmocks` determines the number of mock radio images to use in the analysis.  Setting `Nmocks` to 1 (default), will run the analysis using a single mock radio image and generate a plot and output files for the cross-correlation power spectrum from this single mock radio image.  The analysis should run in approximately one minute with `Nmocks = 1`.  Setting `Nmocks` to a value greater than 1 will run the power spectrum calculation multiple times using `Nmocks` mock radio images and generate a plot of the sample mean and 95% confidence interval of the recovered cross-correlation power spectra.  The maximum value of `Nmocks` is 500, the number of mock radio images in the google drive.

Before running the analysis on the command line, please make sure that the paths to the downloaded data are correct in `config.yaml`.  For the purposes of this demonstration, it is assumed that the data have been downloaded and stored in the current working directory inside a directory called `data/`, i.e. in `./data/`.  It is also assumed that the `meerpower` code has been cloned to the current working directory.  Please also modify the `config.yaml` accordingly if that is not the case.  If both data and `meerpower` have been placed in the current working directory, then no modification of `config.yaml` is required.

Please also ensure that the desired python environment has been created and is active.  To activate the python environment provided with this demonstration, we need only run
```
mamba activate gridimp
```

With the configurarion yaml and the Python environment in place, running the analysis only requires one command
```
python galaxy_cross_mocks.py --config config.yaml
```

If you wish to run the analysis with `Nmocks` > 1 (the default is `Nmocks = 1`), you can either modify `config.yaml` or append the desired value as a command line argument e.g.
```
python galaxy_cross_mocks.py --config config.yaml --Nmocks 500
```
Using `jsonargparse`, arguments following the specification of a configuration yaml take precedent over the value in the yaml file.


## Outputs

The analysis will produce a directory `output/` in the current working directory with the outputs from the analysis.  The location of these outputs can be changed via the `--out-dir` command line argument or via `out_dir` in `config.yaml`. The outputs files are as follows

| File | Shape | Description |
| ---- | ----- | ----------- |
| `k.npy` | (Nkbins,) | Fourier-space, spherically-averaged k bin centers |
| `Pk_gHI_all.npy` | (Nmocks, Nkbins) | Cross-correlation power spectra for each mock radio image |
| `Pk_gHI_all.pdf` | N/a | Plot of the cross-correlation power spectrum if `Nmocks == 1` or the sample mean and 95% confidence interval if `Nmocks > 1` |

### A note regarding the outputs

Because the mock radio images are random realizations of a log-normal density field (please see section 3 of [MeerKLASS Collaboration 2024](https://arxiv.org/abs/2407.21626) for more details), we do not expect any correlation with the galaxy positions in the GAMA catalog.  **We therefore expect the cross-correlation power spectrum to be, on average, zero.**  This analysis is thus a null test which verifies that no spurious correlation is introduced by steps in the analysis.  Note, however, that for 1 or a small number of `Nmocks`, it might be the case that the sample means and/or the confidence intervals are inconsistent with zero because we are dealing with a small sample size.

## UKSRC related links and information

### Confluence pages

- [Cross-Power spectrum estimation using L-band MeerKLASS data](https://confluence.skatelescope.org/display/SRCSC/Cross-Power+Spectrum+Estimation+Using+L-Band+MeerKLASS+Data): page detailing the work done recreating the MeerKLASS results in [MeerKLASS Collaboration 2024](https://arxiv.org/abs/2407.21626).  This analysis requires access to proprietary data, so the steps performed there are not identical.  However, more details on the pipeline and links to other relevant papers are provided.

### Jira tickets

- [TEAL-771](https://jira.skatelescope.org/browse/TEAL-771): prepare mock data for the power spectrum pipeline
- [TEAL-772](https://jira.skatelescope.org/browse/TEAL-772): run mock data through power spectrum pipeline

### Developers

- [Burba, Jacob](https://github.com/jburba) (Manchester)

## Acknowledgements

Thank you to [Cunnington, Steve](https://github.com/stevecunnington) (Manchester, MeerKAT) for all of your help getting this code up and running.

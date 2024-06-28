# Cross-Power Spectrum Estimation

This directory contains information and scripts pertaining to the cross-power spectrum estimation analysis.

For a detailed description of the MeerKAT cross-power spectrum pipeline see
- [Cunnington+2023a](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.6262C/abstract): cross-power spectrum detection using MeerKAT and the [WiggleZ](https://wigglez.swin.edu.au/site/forward.html) Dark Energy Survey
- [Cunnington+2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.2453C/abstract): "transfer function" validation using simulated data used to estimate power spectrum uncertainties and signal loss

The files contained in this repo are as follows
| File | Description |
| ---- | ----------- |
| `meerklass-pspec-mambaorg.def` | Apptainer definition file |
| `config.yaml` | [jsonargparse](https://jsonargparse.readthedocs.io/en/latest/) yaml configuration file which contains all the required command line arguments to run the power spectrum analysis |
| `micromamba.sh` | Shell script to instantiate the micromamba executable for activate python environments |
| `run-slurm.sh` | Slurm sbatch script |

Please note that the code in this repo currently only supports an analysis of 2021 L-band [MeerKLASS](https://github.com/meerklass) and [GAMA](https://www.gama-survey.org/) galaxy catalog data.  In the future, it will be updated with support for the newer UHF-band MeerKLASS data when those data are readily available.

## Software dependencies

The MeerKLASS software is primarily hosted in private GitHub repos. Access to these repos is obtained by emailing the organizer for the [MeerKLASS](https://github.com/meerklass) GitHub organization (Mario Santos, mgrsantos[at]uwc.ac.za). The power spectrum code is stored in [meerklass/meerpower](https://github.com/meerklass/meerpower). This repo contains a `conda` environment yaml containing all of the required python dependencies (more on this below).

To run the code on Azimuth, [@jburba](https://github.com/jburba) needed to make some changes to the meerpower codebase. To do this, [@jburba](https://github.com/jburba) made a fork of the meerpower repo ([meerpower-uksrc](https://github.com/jburba/meerpower-uksrc)). This forked repo is also private to respect the visibility permissions of the original meerpower repo. For access to this fork, please contact [@jburba](https://github.com/jburba) (jacob.burba[at]manchester.ac.uk). If/when the meerpower repo becomes publicly available, the permissions will be changed accordingly in the forked repo.

## Building the container

The file `meerklass-pspec-mambaorg.def` is an Apptainer definition file which can be used to build a container.  This container is based on the [mambaorg/micromamba](https://hub.docker.com/r/mambaorg/micromamba) Docker image which comes with `micromamba` preconfigured.  `micromamba` is used to build the required python environment in the container.

Before building the container from this def file, you must first clone the forked meerpower repo [meerpower-uksrc](https://github.com/jburba/meerpower-uksrc) (please see the section above "Software dependencies") into your current working directory.  The container can then be built via
```
sudo apptainer build meerpower-crossps.sif meerklass-pspec-mambaorg.def
```

## Input data

The MeerKAT data are stored on a compute cluster in South Africa called [ilifu](https://www.ilifu.ac.za/).  Access to ilifu must be requested and the right group permissions must be configured to access the data (please see the [confluence page](https://confluence.skatelescope.org/display/SRCSC/Cross-Power+Spectrum+Estimation+Using+L-Band+MeerKLASS+Data) for more details).  That confluence page also contains a table with all of the required data files for the MeerKLASS and GAMA analysis.  These data have been transferred to
```
/project/jburba/meerkat/lband/2021/
```
and are accessible by any Azimuth platform.  Please note that these data are not public.  Approval was granted by MeerKAT to transfer these data off ilifu and are provided on Azimuth for the use of the UKSRC team.

## Command line arguments

If you would like more information on the command line arguments used in the analysis prior to running the container, you can run
```
apptainer run meerpower-crossps.sif --help
```

## Running the container

Once the container is built, the analysis can be run with Slurm via
```
sbatch run-slurm.sh --bind /path/to/input/data/:/data meerpower-crossps.sif --config config.yaml
```
Note that you will need to change `/path/to/input/data` to the location of the data.  If you are running on Azimuth, for example, you can instead run
```
sbatch run-slurm.sh --bind /project/jburba/meerkat/lband/2021/:/data meerpower-crossps.sif --config config.yaml
```

## Outputs

The container will produce a directory `output/` in the current working directory with the outputs from the analysis. Nested subdirectories will be created within `output/` based on the chosen MeerKAT data and galaxy catalog. For the parameters in the yaml configuration file, running the container will result in the following directory structure
```
output/
└── 2021Lband
    └── gama
        └── TFdata
```
In this case, `output/2021Lband/gama` contains a set of `numpy`-compatible files
| File | Shape | Description |
| ---- | ----- | ----------- |
| `k_{suffix}.npy` | (Nkbins,) | Fourier-space, spherically-averaged k bin centers |
| `Pk_rec_i_{suffix}.npy` | (Nkbins,) | Cross-power spectrum sample mean per k bin |
| `Pk_rec_i_errorbars_{suffix}.npy` | (2, Nkbins) | 68% confidence bounds where the 0th and 1st indices of the first axis contain the lower and upper bounds, respectively |
| `TFData/T_{suffix}.npy` | (2, Nkbins) | Transfer functions where the 0th and 1st indices of the first axis contain the transfer functions for the residuals of foreground cleaned data + mock minus the foreground cleaned data and the foreground cleaned data + mock with no subtraction of the foreground cleaned data, respectively (see section 3.1 of [Cunnington+2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.2453C/abstract) for more details) |
where `suffix` is a string containing the input parameters used in the analysis and is defined automatically by the code.

The `output/2021Lband/gama/` directory also contains a set of plots in PDF format
| File | Description |
| ---- | ----------- |
| `tf_variance_{suffix}.pdf` | Scatter in the transfer-function realizations for all mock signals |
| `pspec_{suffix}.pdf` | Cross-power spectrum with 68% confidence intervals, a fiducial model power spectrum, and the variance from the transfer functions plotted in the background |
| `pspec_distributions_{suffix}.pdf` | Histograms of each cross-power spectrum bandpower |

## UKSRC related links and information

### Confluence pages

- [Cross-Power spectrum estimation using L-band MeerKLASS data](https://confluence.skatelescope.org/display/SRCSC/Cross-Power+Spectrum+Estimation+Using+L-Band+MeerKLASS+Data)

### Jira tickets

- [TEAL-517](https://jira.skatelescope.org/browse/TEAL-517): gather software/pipelines
- [TEAL-518](https://jira.skatelescope.org/browse/TEAL-518): gather data and transfer to azimuth
- [TEAL-550](https://jira.skatelescope.org/browse/TEAL-550): run cross-power spectrum analysis on azimuth

### Developers

- [Burba, Jacob](https://github.com/jburba) (Manchester)

## Acknowledgements

Thank you to [Cunnington, Steve](https://github.com/stevecunnington) (Manchester, MeerKAT) for all of your help getting this code up and running.

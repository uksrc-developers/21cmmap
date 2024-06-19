# Cross-Power Spectrum Estimation

This directory contains information and scripts pertaining to the cross-power spectrum estimation analysis.

For a detailed description of the MeerKAT cross-power spectrum pipeline see
- [Cunnington+2023a](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.6262C/abstract): cross-power spectrum detection using MeerKAT and the [WiggleZ](https://wigglez.swin.edu.au/site/forward.html) Dark Energy Survey
- [Cunnington+2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.2453C/abstract): "transfer function" validation using simulated data used to estimate power spectrum uncertainties and signal loss

## Software dependencies

The MeerKLASS software is primarily hosted in private GitHub repos but some of the repos are public. Access to these repos is obtained by emailing the organizer for the [MeerKLASS](https://github.com/meerklass) GitHub organization (Mario Santos, mgrsantos[at]uwc.ac.za). The power spectrum code is stored in [meerklass/meerpower](https://github.com/meerklass/meerpower). This repo contains a `conda` environment yaml
```
meerklass/meerpower/blob/main/environment_gridimp.yml
```
containing all of the required python dependencies.

To run the code on Azimuth, [@jburba](https://github.com/jburba) needed to make some changes to the meerpower codebase. To do this, [@jburba](https://github.com/jburba) made a fork of the meerpower repo ([meerpower-uksrc](https://github.com/jburba/meerpower-uksrc)). This forked repo is also private to respect the visibility permissions of the original meerpower repo. For access to this fork, please contact [@jburba](https://github.com/jburba) (jacob.burba[at]manchester.ac.uk). If/when the meerpower repo becomes publicly available, the permissions will be changed accordingly in the forked repo.

## Developers and contributors

**Developers**

- Burba, Jacob (UoM)

**Contributors**

- Cunnington, Steven (UoM)

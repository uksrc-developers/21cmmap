# 21cmmap

**21CMMAP: Late-time 21cm Intensity Mapping in Autocorrelation Mode**

This repository contains the code used in the 21CMMAP demonstrator case which contains the following [MeerKAT](https://www.sarao.ac.za/science/meerkat/about-meerkat/) analyses:

1. `cross-power/`: Cross-power spectrum estimation of [MeerKLASS](https://ui.adsabs.harvard.edu/abs/2016mks..confE..32S/abstract) image data with overlapping galaxy surveys (see e.g. [Cunnington+2023a](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.6262C/abstract)).  

2. `otf/`: On The Fly (OTF) observing mode allows for cross-correlations data to be used as part of the MeerKAT imaging pipeline (as opposed to the traditional approach which only uses autocorrelations).

Each directory contains a README with more information about the corresponding analysis.

## Confluence pages

- [General information about the 21CMMAP demonstrator case](https://confluence.skatelescope.org/display/SRCSC/21CMMAP%3A+Late-time+21cm+Intensity+Mapping+in+Autocorrelation+Mode)
- [Cross-Power spectrum estimation using L-band MeerKLASS data](https://confluence.skatelescope.org/display/SRCSC/Cross-Power+Spectrum+Estimation+Using+L-Band+MeerKLASS+Data)
- [On The Fly (OTF) Observing Mode](https://confluence.skatelescope.org/display/SRCSC/21CMMAP%3A+Late-time+21cm+Intensity+Mapping+in+Autocorrelation+Mode)

## Jira tickets

**Cross-power spectrum estimation:**

- [TEAL-517](https://jira.skatelescope.org/browse/TEAL-517): gather software/pipelines
- [TEAL-518](https://jira.skatelescope.org/browse/TEAL-518): gather data and transfer to azimuth
- [TEAL-550](https://jira.skatelescope.org/browse/TEAL-550): run cross-power spectrum analysis on azimuth

**OTF observing mode:**

- [TEAL-599](https://jira.skatelescope.org/browse/TEAL-599): gather software/pipelines
- [TEAL-600](https://jira.skatelescope.org/browse/TEAL-600): gather data and transfer to azimuth

## Developers

- Burba, Jacob (UoM)

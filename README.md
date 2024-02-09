Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

This Zenodo dataset contains the data processing pipeline, as well as the data products, corresponding to the scientific journal paper "NANOGrav 12.5-Year Data Set: Dispersion Measure Mis-Estimation with Varying Bandwidths". The processing pipeline is structured as follows:

- `1_fit_dispersion_3terms.py`  reads the .tim files (containing the times-of-arrival), creates the broadband and narrowband datasets, and fits a dispersion model to both datasets using three parameters.
- `2_plot_fits_differences.py` creates the residual plots showing the differences in the fitted values of the parameters.
- `3_autocovariance.py` calculates the autocovariance function of the differences in the fitted values, and creates the corresponding plots.
- All the files starting with `plot` are convenience scripts for creating the plots presented in the paper.
- All the files starting with `sophia` are utility functions that are used in the main pipeline.
- The folder `NANOGrav_12yv4` contains the dataset analyzed in this work.
- The folder `NG_timing_analysis` contains utility functions created by the NANOGrav collaboration that are used in the main pipeline.

Please do not hesitate to send all your questions, concerns, or commentaries to sophia.sosa@nanograv.org

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
-

## [0.34.2] - 2022-12-10
### Added
- README.md instructions

### Fixed
- Jupyter Notebook and .gitignore

## [0.34.0] - 2022-11-13
### Added
- Dockerfile
- Jupyter-lab out-of-the-box from Docker image
- Command line tool for running detections
- Versioning control (tags, setup.py, version.py)
- Black formatting

### Fixed
- README.md and CHANGELOG -> .md
- Python package structure best practices

## [0.33.0]
### Added
- Changed parameters named to correlate with written paper

## [0.32.0]
### Added
- Added support for shapefile cropping

## [0.31.0]
### Fixed
- Fixed erros in results saving process

## [0.30.0]
### Added
- Fixed errors in LSTM modelling process

## [0.29.0]
### Added
- Changed MNDWI threshold to 0.0

## [0.28.0]
### Added
- Added reference validation (Earth Earth Engine)
- Added reduction parameter (meadian or min, if using reference validation)
- Major fixes in the statistics functions

## [0.27.0]
### Added
- New batch script (added Lake Taihu area)

### Fixed
- Fixes in the script.py attributes
- Fixes in the reduction plot

## [0.26.0]
### Fixed
- Fixes in the grid calculation process
- Fixes in the median calculation results

## [0.25.0]
### Added
- Changed accuracy calculation method in grid-wise
- Added median, mode and mean final results comparison

### Fixed
- Fixes in indice_threshold selection

## [0.24.0]
### Fixed
- Fixes in the interpolation process of classification/prediction dataframe

## [0.23.0]
### Fixed
- Fixes in scene plot results

## [0.22.0]
### Fixed
- Fixes in MLP and LSTM modelling and prediction process
- Changes in grid and scene results

## [0.21.0]
### Added
- Selection of best parameters and change in scene results format

## [0.20.0]
### Fixed
- Fix error when using attributes disable_attribute_lat_lon and disable_attribute_doy

## [0.19.0]
### Fixed
- Fix SVM slow modelling process

## [0.18.0]
### Added
- Added ROI validation from Google Earth Engine Console

### Fixed
- Fixes in the modelling process and label selection

## [0.17.0]
### Added
- Added parameters for new tests and modification of the LSTM modelling process
- Changed training labels selection, using only edges values (0 and total indexes)

## [0.16.0]
### Added
- Changed some default parameters based on previous tests

## [0.15.0]
### Added
- Changed accuracy calculation process (removed indetermined bloom pixels)

## [0.14.0]
### Added
- Added parameter 'normalized': when activated, it will normalized indices values to range -1 and 1 (ndwi, ndvi and sabi)
- Added 'class_mode' (regression modelling process) and 'class_weight' (it will assign weights to classes before the training process) parameters
- Added parameter 'propagate': when activated, it will propagate predictions from one day to another in prediction date range
- Added parameter 'rs_train_size', 'rs_iter': it allows increase the randomized search training size and iterations
- Added parameter 'pca_size'

## [0.13.0]
### Added
- Added new indices: SABI and NDWI
- Added RandomizedSearchCV to MLP

## [0.12.0]
### Added
- LSTM to encoding and decoding version, Random Forest and SVM RandomizedSearchCV version, removed mlp_relu and grid fixes
- New MLP and LSTM parametrization
- Module GEE and Misc updateds

## [0.11.0]
### Added
- Multilayer Perceptron and LSTM fixes

## [0.10.0]
### Added
- Algorithm modified to use regression instead of classification (it will predict ndvi, fai and slope)

## [0.9.0]
### Fixed
- Few algorithm fixes

## [0.8.0]
### Fixed
- Trying to fix de accuracy problem in training process

## [0.7.0]
### Fixed
- Fix in the evaluation process

## [0.6.0]
### Fixed
- Model parameters fixes

## [0.5.0]
### Fixed
- Pre-training and results plots fixes

## [0.4.0]
### Added
- Changed the metrics to evaluate the training process

## [0.3.0]
### Added
- Created spare function to assess training parameters

## [0.2.0]
### Added
- Accuracy metrics insertion and spreadsheet creation with final results

## [0.1.0]
### Added
- Repository creation
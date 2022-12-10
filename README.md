# Anomalous Behaviour Forecast

Module responsible for forecasting anomalies occurrences based on images from Google Earth Engine API and machine learning



### Dependencies

- Python >= 3.7.7 64-bit, < 3.10 64-bit
- Google Earth Engine enabled account: see https://earthengine.google.com/



## Instalation

To install this script and all its dependencies, execute the follow commands:

1) Create a virtual environment: `python3 -m venv venv`
2) Enable it: `source venv/bin/activate`
2) Install the script dependencies: `pip install -e .`



## Attention, before running this script:

Before running the script and after installing the libraries, you must authenticate with the Google Earth Engine API. However, the command line tool will automatically require you to do it so:

```bash
# from local command line
earthengine authenticate

# from inside Docker container
earthengine authenticate --auth_mode=notebook

# from inside Jupyter Notebook
import ee
ee.Authenticate()
```

In some versions of macOS, it might be necessary to run this command using `sudo`.

Additionally, make sure that folder `results` (or whatever path you've defined) has writing permissions:

```bash
chmod 777 /path/to/abf/results
```


## Docker image

There is also a Docker image which provides this script with all necessary dependencies easy and ready. To use it, run:

```bash
docker run -p 8888:8888 --name abf phmananias/abf:latest
```

or you can build it locally and then run it:
```bash
docker build -t abf:latest .
docker run -p 8888:8888 --name abf abf:latest
```


## Command line tool

This module brings a default command line `adf` for you. To see available parameters, please run:

```bash
> abf --help
Usage: abf [OPTIONS]

Options:
  --lat_lon TEXT                  Two diagonal points (Latitude 1, Longitude
                                  1, Latitude 2, Longitude 2) of the study
                                  area  [default: -83.48811946836814,41.857760
                                  95627803,-83.18290554014548,41.6776173953378
                                  26]
  --instant TEXT                  Date to end time series (it will forecast 5
                                  days starting from this date)  [default:
                                  2013-10-11]
  --name TEXT                     Place where to save generated files
                                  [default: erie]
  --sensor TEXT                   Define the selected sensor where images will
                                  be downloaded from: landsat, sentinel, modis [default: modis]
  --spanning_period INTEGER       Spanning period used to build the timeseries
                                  and training set  [default: 180]
  --past_steps INTEGER            Past steps to be used as input forecast
                                  [default: 4]
  --forecasting_period INTEGER    Forecasting period to be used as output
                                  forecast  [default: 5]
  --spatial_context_size INTEGER  Spatial context size in pixels that will be
                                  used in continuous/occurrences results
                                  [default: 7]
  --model TEXT                    Select the desired module: mlp, lstm, rf,
                                  svm or all (None)  [default: rf]
  --fill_missing TEXT             Defines algorithm to be used to fill empty
                                  dates and values: dummy, ffill, bfill, time,
                                  linear  [default: time]
  --remove_dummies                Defines if the dummies will be removed
                                  before training (only works with
                                  fill_missing=dummy)
  --scaler TEXT                   Defines algorithm to be used in the data
                                  scalling process: robust, minmax or standard
                                  [default: minmax]
  --disable_reducer               Defines if reducer will not be applied to
                                  remove unnecessary features
  --disable_normalization         Defines if normalization (-1,1) will not be
                                  applied to indices mndwi, ndvi and sabi
  --regression_mode               Defines whether will use raw values or
                                  classes in the regression models
  --class_weight                  Defines whether classes will have defined
                                  weights for each
  --propagate                     Defines whether predictions will be
                                  propagated ahead
  --rs_train_size FLOAT           It allow increase the randomized search
                                  dataset training size (it can be a floater
                                  or integer)  [default: 500.0]
  --rs_iter INTEGER               It allow increase the randomized search
                                  iteration size  [default: 500]
  --pca_size FLOAT                Define PCA reducer variance size  [default:
                                  0.9]
  --reduction TEXT                Define which reduction (median or min) will
                                  be used in the reduction stats  [default:
                                  median]
  --convolve                      Define if a convolution box-car low-pass
                                  filter will be applied to images before
                                  training
  --convolve_radius INTEGER       Define the amont of radius will be used in
                                  the convolution box-car low-pass filter
                                  [default: 1]
  --enable_attribute_lat_lon      Enable attributes lat and lon in the
                                  training process
  --disable_attribute_doy         Disable attribute doy from training modeling
  --disable_shuffle               Disable data shutffle before splitting into
                                  train and test matrix
  --save_pairplots                Save pairplots from attributes and indices
  --save_train                    Enable saving the training dataset (csv)
  --cloud_threshold FLOAT         Define which cloud threshold will be used in
                                  the timeseries modelling process  [default:
                                  0.5]
  --shapefile TEXT                Use a shapefile to clip a region of interest
  --output_folder TEXT            Specify desired results output folder
  --help                          Show this message and exit.
```


## Installing using PIP

You can directly install the latest version of this script using the following command:

```bash
pip install git+https://github.com/pedroananias/abf.git@latest
```


### How to execute the default script?

```bash
abf --lat_lon=-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826 --name=erie --instant=2019-07-11
```


### What are the results?

The script will generate a 5-days forecast of occurrence of algae blooming in the inserted study area starting from input date. Therefore, a folder located in 'data' is created and named based on the date and version of the script executed. Example: /path/to/abf/data/20201026_090433[v=V20-erie,d=2019-07-11,dt=180,din=5,dout=5,m=rf,g=12]. 

ATTENTION: big date range tends to lead to memory leak, stopping the script execution. It is always a good pratice to split the dates in two or three parts, unless you have a big amount of memory in your computer.

The following results are generated:

- timeseries.csv (Time series from dates ranging current day and days threshold parameter)
- results.csv (Accuracy results from the forecasted period validated using indices thresholding approach)
- prediction/csv (CSVs from generated outputs and predictions separated by day)
- prediction/geojson (GeoJSONs from generated outputs and predictions separated by day)
- prediction/image  (Plot from generated outputs and predictions separated by day)



## Sandbox example

This script comes with a Jupyter Notebook sandbox example file. To open it, please run the command below inside the script's root directory:

```bash
jupyter-lab
```


## Apple Arm64 CPU installation instructions

In order to run this script with an Apple Arm64 CPU with Tensorflow support, please follow these steps:

1) brew install --cask anaconda
2) cd /path/to/abf
3) conda create -n abf python=3.9
3) conda activate abf
4) conda install -c apple tensorflow-deps
5) pip install tensorflow-macos tensorflow-metal
6) pip install -e .
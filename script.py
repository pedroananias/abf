#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################################################################################################################
# ### ABF - Anomalous Behaviour Forecast
# ### Script responsible for executing the anomalous behaviour forecast using machine learning and deep learning
# ### Python 3.7 64Bits is required!
#
# ### Change History
# - Version 1: 
# - Repository creation
#
# - Version 2: 
# - Accuracy metrics insertion and spreadsheet creation with final results
#
# - Version 3: 
# - Created spare function to assess training parameters
#
# - Version 4: 
# - Changed the metrics to evaluate the training process
#
# - Version 5: 
# - pre-training and results plots fixes
#
# - Version 6: 
# - model parameters fixes
#
# - Version 7: 
# - fix in the evaluation process
#
# - Version 8: 
# - trying to fix de accuracy problem in training process
#
# - Version 9: 
# - few algorithm fixes
#
# - Version 10: 
# - algorithm modified to use regression instead of classification (it will predict ndvi, fai and slope)
#
# - Version 11: 
# - Multilayer Perceptron and LSTM fixes
#
# - Version 12: 
# - LSTM to encoding and decoding version, Random Forest and SVM RandomizedSearchCV version, removed mlp_relu and grid fixes
# - New MLP and LSTM parametrization
# - Module GEE and Misc updateds
#
# - Version 13: 
# - Added new indices: SABI and NDWI
# - Added RandomizedSearchCV to MLP
#
# - Version 14: 
# - Added parameter 'normalized': when activated, it will normalized indices values to range -1 and 1 (ndwi, ndvi and sabi)
# - Added 'class_mode' (regression modelling process) and 'class_weight' (it will assign weights to classes before the training process) parameters
# - Added parameter 'propagate': when activated, it will propagate predictions from one day to another in prediction date range
# - Added parameter 'rs_train_size', 'rs_iter': it allows increase the randomized search training size and iterations
# - Added parameter 'pca_size'
#
# - Version 15: 
# - Changed accuracy calculation process (removed indetermined bloom pixels)
#
# - Version 16:
# - Changed some default parameters based on previous tests
#
# - Version 17:
# - Added parameters for new tests and modification of the LSTM modelling process
# - Changed training labels selection, using only edges values (0 and total indexes)
#
# - Version 18:
# - Fixes in the modelling process and label selection
# - Added ROI validation from Google Earth Engine Console
#####################################################################################################################################

# ### Version
version = "V18"



# ### Module imports

# Main
import ee
import pandas as pd
import math
import requests
import time
import warnings
import os
import sys
import argparse
import traceback
import gc

# Sub
from datetime import datetime as dt
from datetime import timedelta

# Extras modules
from modules import misc, gee, abf


# ### Script args parsing

# starting arg parser
parser = argparse.ArgumentParser(description=version)

# create arguments
parser.add_argument('--lat_lon', dest='lat_lon', action='store', default="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826",
                   help="Two diagnal points (Latitude 1, Longitude 1, Latitude 2, Longitude 2) of the study area")
parser.add_argument('--from_date', dest='from_date', action='store', default="2019-07-11",
                   help="Date to end time series (it will forecast 5 days starting from this date)")
parser.add_argument('--name', dest='name', action='store', default="erie",
                   help="Place where to save generated files")
parser.add_argument('--days_threshold', dest='days_threshold', action='store', type=int, default=1825,
                   help="Days threshold used to build the timeseries and training set")
parser.add_argument('--days_in', dest='days_in', action='store', type=int, default=1,
                   help="Day threshold to be used as input forecast")
parser.add_argument('--days_out', dest='days_out', action='store', type=int, default=5,
                   help="Day threshold to be used as output forecast")
parser.add_argument('--model', dest='model', action='store', default=None,
                   help="Select the desired module: mlp, lstm, rf, svm or all (None)")
parser.add_argument('--fill_missing', dest='fill_missing', action='store', default="time",
                   help="Defines algorithm to be used to fill empty dates and values: dummy, ffill, bfill, time, linear")
parser.add_argument('--grid_size', dest='grid_size', action='store', type=int, default=3,
                   help="Grid size that will be used in prediction")
parser.add_argument('--remove_dummies', dest='remove_dummies', action='store_true',
                   help="Defines if the dummies will be removed before training (only works with fill_missing=dummy)")
parser.add_argument('--reducer', dest='reducer', action='store_true',
                   help="Defines if reducer will be applied to remove unnecessary features")
parser.add_argument('--non_normalized', dest='non_normalized', action='store_false',
                   help="Defines if normalization (-1,1) will not be applied to indices ndwi, ndvi and sabi")
parser.add_argument('--class_mode', dest='class_mode', action='store_true',
                   help="Defines whether will use raw values or classes in the regression models")
parser.add_argument('--class_weight', dest='class_weight', action='store_true',
                   help="Defines whether classes will have defined weights for each")
parser.add_argument('--propagate', dest='propagate', action='store_true',
                   help="Defines whether predictions will be propagated ahead")
parser.add_argument('--rs_train_size', dest='rs_train_size', action='store', type=float, default=0.01,
                   help="It allow increase th randomized search dataset training size")
parser.add_argument('--rs_iter', dest='rs_iter', action='store', type=int, default=500,
                   help="It allow increase th randomized search iteration size")
parser.add_argument('--pca_size', dest='pca_size', action='store', type=float, default=0.900,
                   help="Define PCA reducer variance size")
parser.add_argument('--convolve', dest='convolve', action='store_true',
                   help="Define if a convolution box-car low-pass filter will be applied to images before training")
parser.add_argument('--convolve_radius', dest='convolve_radius', action='store', type=int, default=1,
                   help="Define the amont of radius will be used in the convolution box-car low-pass filter")
parser.add_argument('--disable_attribute_lat_lon', dest='disable_attribute_lat_lon', action='store_false',
                   help="Disable attributes lat and lons from training modeling")
parser.add_argument('--disable_attribute_doy', dest='disable_attribute_doy', action='store_false',
                   help="Disable attribute doy from training modeling")
parser.add_argument('--save_pairplots', dest='save_pairplots', action='store_true',
                   help="Save pairplots from attributes and indices")
parser.add_argument('--save_grid', dest='save_grid', action='store_true',
                   help="Save study area images grids")
parser.add_argument('--save_train', dest='save_train', action='store_true',
                   help="Enable saving the training dataset (csv)")

# parsing arguments
args = parser.parse_args()




# ### Start

try:

  # Start script time counter
  start_time = time.time()

  # Google Earth Engine API initialization
  ee.Initialize()



  # ### Working directory

  # Data path
  folderRoot = os.path.dirname(os.path.realpath(__file__))+'/data'
  if not os.path.exists(folderRoot):
    os.mkdir(folderRoot)

  # Images path
  folderCache = os.path.dirname(os.path.realpath(__file__))+'/cache'
  if not os.path.exists(folderCache):
    os.mkdir(folderCache)



  # ### Selection of coordinates (colon, lat and lon separated by comma, all together) and dates by the user (two dates, beginning and end, separated by commas)
  x1,y1,x2,y2 = args.lat_lon.split(",")

  # Assemble Geometry on Google Earth Engine
  geometry = ee.Geometry.Polygon(
          [[[float(x1),float(y2)],
            [float(x2),float(y2)],
            [float(x2),float(y1)],
            [float(x1),float(y1)],
            [float(x1),float(y2)]]])


  
  # ### ABF execution

  # folder to save results from algorithm at
  folder = folderRoot+'/'+dt.now().strftime("%Y%m%d_%H%M%S")+'[v='+str(version)+'-'+str(args.name)+',d='+str(args.from_date)+',dt='+str(args.days_threshold)+',din='+str(args.days_in)+',dout='+str(args.days_out)+',m='+str(args.model)+']'
  if not os.path.exists(folder):
    os.mkdir(folder)

  # create algorithm
  algorithm = abf.Abf(days_threshold=args.days_threshold,
                      grid_size=args.grid_size,
                      sensor="modis", 
                      geometry=geometry,
                      lat_lon=args.lat_lon,
                      path=folder,
                      cache_path=folderCache,
                      force_cache=False,
                      morph_op=None, 
                      morph_op_iters=1,
                      convolve=args.convolve,
                      convolve_radius=args.convolve_radius,
                      scaler='robust',
                      days_in=args.days_in,
                      days_out=args.days_out,
                      from_date=args.from_date,
                      model=args.model,
                      fill_missing=args.fill_missing,
                      remove_dummies=args.remove_dummies,
                      reducer=args.reducer,
                      normalized=args.non_normalized,
                      class_mode=args.class_mode,
                      class_weight=args.class_weight,
                      propagate=args.propagate,
                      rs_train_size=args.rs_train_size,
                      rs_iter=args.rs_iter,
                      pca_size=args.pca_size,
                      attribute_lat_lon=args.disable_attribute_lat_lon,
                      attribute_doy=args.disable_attribute_doy,
                      shuffle=True,
                      test_mode=False)

  # preprocessing
  algorithm.process_timeseries_data()
  algorithm.process_training_data(df=algorithm.df_timeseries)

  # save pairplots
  if args.save_pairplots == True:
    if not os.path.exists(folder+"/pairplots"):
      os.mkdir(folder+"/pairplots")
    algorithm.save_attributes_pairplot(df=algorithm.df_timeseries,path=folder+"/pairplots/attributes.png")
    algorithm.save_indices_pairplot(df=algorithm.df_timeseries,path=folder+"/pairplots/indices.png")

  # save grids
  if args.save_grid == True:
    algorithm.save_grid_plot(folder=folder+"/grids")
    
  # train/predict
  else:
    algorithm.train(batch_size=2048, disable_gpu=True)
    algorithm.predict(folder=folder+"/prediction")
    algorithm.validate_using_roi(path='users/pedroananias/'+str(args.name), rois=['date_sensor_regular', 'date_sensor_anomaly'], labels=[0, 1])

    # prediction results
    algorithm.save_dataset(df=algorithm.df_results, path=folder+'/results.csv')
    #algorithm.save_results_plot(df=algorithm.df_results, path=folder+'/results.png')

    # preprocessing results
    algorithm.save_dataset(df=algorithm.df_timeseries, path=folder+'/timeseries.csv')
    #algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries.png')
    #algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries_join.png', join=True)

    # save training datasets
    if args.save_train == True:
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_train[0]), path=folder+'/df_train_X.csv')
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_train[1]), path=folder+'/df_train_y.csv')
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_test[0]), path=folder+'/df_test_X.csv')
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_test[1]), path=folder+'/df_test_y.csv')

  # results
  # add results and save it on disk
  path_df_results = folderRoot+'/results.csv'
  df_results = pd.read_csv(path_df_results).drop(['Unnamed: 0'], axis=1, errors="ignore").append(algorithm.df_results) if os.path.exists(path_df_results) else algorithm.df_results.copy(deep=True)
  df_results.to_csv(r''+path_df_results)

  # clear memory
  del algorithm
  gc.collect()

  # ### Script termination notice
  script_time_all = time.time() - start_time
  debug = "***** Script execution completed successfully (-- %s seconds --) *****" %(script_time_all)
  print()
  print(debug)

except:

    # ### Script execution error warning

    # Execution
    print()
    print()
    debug = "***** Error on script execution: "+str(traceback.format_exc())
    print(debug)

    # Removes the folder created initially with the result of execution
    script_time_all = time.time() - start_time
    debug = "***** Script execution could not be completed (-- %s seconds --) *****" %(script_time_all)
    print(debug)

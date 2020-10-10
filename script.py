#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################################
# ### ABF - Anomaly and Algal Bloom Forecast
# ### Script responsible for executing the anomaly and algal bloom forecast using machine learning and deep learning
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
#######################################################################################################################

# ### Version
version = "V13"



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
parser.add_argument('--days_threshold', dest='days_threshold', action='store', type=int, default=365,
                   help="Days threshold used to build the timeseries and training set")
parser.add_argument('--days_in', dest='days_in', action='store', type=int, default=5,
                   help="Day threshold to be used as input forecast")
parser.add_argument('--days_out', dest='days_out', action='store', type=int, default=5,
                   help="Day threshold to be used as output forecast")
parser.add_argument('--model', dest='model', action='store', default=None,
                   help="Select the desired module: mlp, lstm, rf, svm or all (None)")
parser.add_argument('--fill_missing', dest='fill_missing', action='store', default="time",
                   help="Define algorithm to be used to fill empty dates and values: dummy, ffill, bfill, time, slinear, pchip, quadratic, cubic, akima")
parser.add_argument('--grid_size', dest='grid_size', action='store', type=int, default=3,
                   help="Grid size that will be used in prediction")
parser.add_argument('--remove_dummies', dest='remove_dummies', action='store_true',
                   help="Define if the dummies will be removed before training (only works with fill_missing=dummy)")
parser.add_argument('--reducer', dest='reducer', action='store_true',
                   help="Define if reducer will be applied to remove unnecessary features")
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

  # results
  path_df_results = folderRoot+'/results.csv'
  if not os.path.exists(path_df_results):
    df_results = pd.DataFrame(columns=['model','date','acc','kappa'])
  else:
    df_results = pd.read_csv(path_df_results).drop(['Unnamed: 0'], axis=1)

  # default configuration
  batch_size      = 4096
  morph_op        = None
  morph_op_iters  = 1
  convolve        = False
  convolve_radius = 1
  shuffle         = True
  
  # folder to save results from algorithm at
  folder = folderRoot+'/'+dt.now().strftime("%Y%m%d_%H%M%S")+'[v='+str(version)+'-'+str(args.name)+',date='+str(args.from_date)+',dt='+str(args.days_threshold)+',gs='+str(args.grid_size)+',din='+str(args.days_in)+',dout='+str(args.days_out)+',mop='+str(morph_op)+',mit='+str(morph_op_iters)+',cv='+str(convolve)+',cvr='+str(convolve_radius)+',sf='+str(shuffle)+',m='+str(args.model)+',fill='+str(args.fill_missing)+',rd='+str(args.remove_dummies)+']'
  if not os.path.exists(folder):
    os.mkdir(folder)

  # create algorithm
  algorithm = abf.Abf(days_threshold=args.days_threshold,
                      grid_size=args.grid_size,
                      sensor="modis", 
                      geometry=geometry,
                      lat_lon=args.lat_lon,
                      cache_path=folderCache,
                      force_cache=False,
                      morph_op=morph_op, 
                      morph_op_iters=morph_op_iters,
                      convolve=convolve,
                      convolve_radius=convolve_radius,
                      scaler='robust',
                      days_in=args.days_in,
                      days_out=args.days_out,
                      from_date=args.from_date,
                      model=args.model,
                      fill_missing=args.fill_missing,
                      remove_dummies=args.remove_dummies,
                      test_mode=False,
                      shuffle=shuffle,
                      reducer=args.reducer)

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
    algorithm.train(batch_size=batch_size, disable_gpu=True)
    algorithm.predict(folder=folder+"/prediction")

    # prediction results
    algorithm.save_dataset(df=algorithm.df_results, path=folder+'/results.csv')
    algorithm.save_results_plot(df=algorithm.df_results, path=folder+'/results.png')

    # preprocessing results
    algorithm.save_dataset(df=algorithm.df_timeseries, path=folder+'/timeseries.csv')
    algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries.png')
    algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries_join.png', join=True)

    # save training datasets
    if args.save_train == True:
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_train[0]), path=folder+'/df_train_X.csv')
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_train[1]), path=folder+'/df_train_y.csv')
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_test[0]), path=folder+'/df_test_X.csv')
      algorithm.save_dataset(df=pd.DataFrame(algorithm.df_test[1]), path=folder+'/df_test_y.csv')


  # add results do dataframe
  description = str(args.name)+"-"+str(args.model)+"-"+str(args.from_date)+"-"+str(args.days_threshold)+"-"+str(args.days_in)+'-'+str(args.days_out)+'-'+str(args.reducer)+'-'+str(args.fill_missing)
  for index, row in algorithm.df_results.iterrows():
    df_results.loc[len(df_results)] = {
      'model':   description+'-'+str(row['type']),
      'date':    row['date_predicted'],
      'acc':     row['acc'],
      'kappa':   row['kappa'],
    }

  # save results
  df_results.drop_duplicates(subset=['model','date'], keep='last').sort_values(by=['acc','kappa'], ascending=False).to_csv(r''+path_df_results)

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

#!/usr/bin/python
# -*- coding: utf-8 -*-

#########################################################################################################################################
# ### ABF - Anomalous Behaviour Forecast
# ### Module responsable for anomalous behaviour forecast algorithm application
#########################################################################################################################################

# Dependencies
# Base
import ee
import numpy as np
import pandas as pd
import hashlib
import PIL
import requests
import os
import joblib
import gc
import sys
import traceback
import math
import scipy
import os
import traceback
import random
import seaborn as sns
import copy
import re
import time
import geojson
import warnings
import multiprocessing
from io import BytesIO
from datetime import datetime as dt
from datetime import timedelta as td

# Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Machine Learning
from sklearn import preprocessing, svm, model_selection, metrics, feature_selection, ensemble, multioutput, decomposition, manifold, utils

# Deep Learning
import tensorflow as tf

# Local
from modules import misc, gee

# Warning correction from Pandas
pd.plotting.register_matplotlib_converters()
warnings.filterwarnings("ignore")

# Anomalous Behaviour Forecast
class Abf:

  # configuration
  name                        = "Anomalous Behaviour Forecast"
  anomaly                     = 1
  dummy                       = -99999
  cloud_threshold             = 0.50 # only images where clear pixels are greater thans 50%
  max_tile_pixels             = 10000000 # if higher, will split the geometry into tiles
  indices_thresholds          = {'ndwi': 0.3, 'ndvi': -0.15, 'sabi': -0.10, 'fai': -0.004}
  n_cores                     = int(multiprocessing.cpu_count()*0.75) # only 75% of available cores
  random_state                = 123 # random state used in numpy and related shuffling problems
  plots_grid                  = 5 # used to divide the plot in grid

  # attributes
  attributes                  = ['cloud', 'ndwi', 'ndvi', 'sabi', 'fai', 'wind', 'temperature', 'drainage_direction', 'precipitation', 'elevation', 'pressure', 'evapotranspiration', 'emissivity']
  attributes_clear            = ['cloud', 'ndwi', 'ndvi', 'sabi', 'fai']
  attributes_inverse          = ['ndwi']
  attributes_selected         = ['lat', 'lon', 'doy', 'ndwi', 'ndvi', 'sabi', 'fai', 'wind', 'temperature', 'drainage_direction', 'precipitation', 'elevation', 'pressure', 'evapotranspiration', 'emissivity']
  
  # supports
  dates_timeseries            = [None, None]
  dates_timeseries_interval   = []
  scaler_str                  = None
  sensor_params               = None
  resolution                  = [0,0]
  limits                      = [0,0]
  pixels_lat_lons             = None

  # masks
  water_mask                  = None
  
  # collections
  collection                  = None
  collection_water            = None

  # clips
  image_clip                  = None

  # sample variables
  sample_total_pixel          = None
  sample_clip                 = None
  sample_lon_lat              = [[0,0],[0,0]]
  splitted_geometry           = []
  randomizedsearch_categorical= ()

  # dataframes
  df_columns                  = ['pixel','index','row','column','date','doy','lat','lon']+attributes
  df_columns_clear            = ['pixel','index','row','column','date','doy','lat','lon']+attributes_clear
  df_columns_results          = ['model', 'type', 'sensor', 'path', 'date_predicted', 'date_execution', 'time_execution', 'runtime', 'days_threshold', 'grid_size', 'size_train', 'size_dates', 'scaler', 'morph_op', 'morph_op_iters', 'convolve', 'convolve_radius', 'days_in', 'days_out', 'fill_missing', 'remove_dummies', 'shuffle', 'reducer', 'normalized', 'class_mode', 'class_weight', 'propagate', 'rs_train_size', 'rs_iter', 'pca_size', 'attribute_lat_lon', 'attribute_doy', 'acc', 'bacc', 'kappa', 'vkappa', 'tau', 'vtau', 'mcc', 'f1score', 'rmse', 'mae', 'r2score', 'tp', 'tn', 'fp', 'fn']
  df_columns_scene            = ['date','model','validation', 'prediction', 'difference']
  df_timeseries               = None
  df_timeseries_scene         = None
  df_timeseries_grid          = None
  df_train                    = [None,None]
  df_test                     = [None,None]
  df_randomizedsearch         = [None,None]
  df_classification           = None
  df_results                  = None
  df_pretraining              = None
  df_scene                    = None

  # hash
  hash_string                 = "abf-20210329"

  # dates
  classification_dates        = None
  predict_dates               = None

  # dicts
  classifiers                 = {}
  classifiers_runtime         = {}
  predictions                 = {}
  merges                      = {}

  # constructor
  def __init__(self,
               geometry:          ee.Geometry,
               days_threshold:    int           = 90,
               grid_size:         int           = 7,
               sensor:            str           = "modis",
               scale:             int           = None,
               path:              str           = None,
               cache_path:        str           = None,
               lat_lon:           str           = None,
               force_cache:       bool          = False,
               morph_op:          str           = None,
               morph_op_iters:    int           = 1,
               convolve:          bool          = False,
               convolve_radius:   int           = 1,
               scaler:            str           = 'minmax',
               days_in:           int           = 4,
               days_out:          int           = 5,
               from_date:         str           = None,
               model:             str           = None,
               fill_missing:      str           = "time",
               remove_dummies:    bool          = False,
               shuffle:           bool          = True,
               reducer:           bool          = True,
               normalized:        bool          = True,
               class_mode:        bool          = True,
               class_weight:      bool          = False,
               propagate:         bool          = False,
               rs_train_size:     float         = 500.0,
               rs_iter:           int           = 500,
               pca_size:          float         = 0.900,
               attribute_lat_lon: bool          = False,
               attribute_doy:     bool          = True,
               test_mode:         bool          = False):
    
    # get sensor parameters
    self.sensor_params  = gee.get_sensor_params(sensor)
    self.scale          = self.sensor_params['scale'] if not scale else scale

    # warning
    print()
    print("Selected sensor: "+self.sensor_params['name'])

    # user defined parameters
    self.geometry                     = geometry
    self.days_threshold               = days_threshold
    self.path                         = path
    self.cache_path                   = cache_path
    self.lat_lon                      = lat_lon
    self.sensor                       = sensor
    self.force_cache                  = force_cache
    self.grid_size                    = int(grid_size)-1 if (int(grid_size) % 2) == 0 else int(grid_size)
    self.morph_op                     = morph_op
    self.morph_op_iters               = morph_op_iters
    self.convolve                     = convolve
    self.convolve_radius              = convolve_radius
    self.days_in                      = days_in
    self.days_out                     = days_out
    self.model                        = model
    self.fill_missing                 = fill_missing
    self.shuffle                      = shuffle
    self.remove_dummies               = remove_dummies
    self.reducer                      = reducer
    self.normalized                   = normalized
    self.class_mode                   = class_mode
    self.class_weight                 = class_weight
    self.propagate                    = propagate
    self.rs_train_size                = int(rs_train_size) if rs_train_size >= 1 else rs_train_size
    self.rs_iter                      = rs_iter
    self.pca_size                     = pca_size
    self.attribute_lat_lon            = attribute_lat_lon
    self.attribute_doy                = attribute_doy

    # fix days_in and days_out (avoid errors)
    self.days_in                      = self.days_in  if self.days_in   >= 1 else 1
    self.days_out                     = self.days_out if self.days_out  >= 1 else 1

    # faster loading
    if not test_mode:

      # create scaler
      if scaler == 'minmax':
        self.scaler_str = 'minmax'
        self.scaler     = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
      elif scaler == 'robust':
        self.scaler_str = 'robust'
        self.scaler     = preprocessing.RobustScaler(quantile_range=(25, 75), copy=True)
      else:
        self.scaler_str = 'standard'
        self.scaler     = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

      # time series expansion
      # based on GDAL last image
      if from_date is None:
        self.dates_timeseries[1]        = dt.fromtimestamp(ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").filterBounds(self.geometry).sort('system:time_start', False).first().get('system:time_start').getInfo()/1000.0)
      else:
        self.dates_timeseries[1]        = dt.strptime(from_date, "%Y-%m-%d")

      # correct time series expansion (sensor start date)
      self.dates_timeseries[0] = self.dates_timeseries[1] - td(days=self.days_threshold)
      if self.dates_timeseries[0] < self.sensor_params['start']:
        self.dates_timeseries[0] = self.sensor_params['start']

      # creating final sensor collection
      collection, collection_water      = gee.get_sensor_collections(geometry=self.geometry, sensor=self.sensor, dates=[dt.strftime(self.dates_timeseries[0], "%Y-%m-%d"), dt.strftime(self.dates_timeseries[1], "%Y-%m-%d")])

      # create useful time series
      self.collection                   = collection
      self.collection_water             = collection_water
      self.dates_timeseries_interval    = misc.remove_duplicated_dates([dt.strptime(d, "%Y-%m-%d") for d in self.collection.map(lambda image: ee.Feature(None, {'date': image.date().format('YYYY-MM-dd')})).distinct('date').aggregate_array("date").getInfo()])

      # preprocessing - water mask extraction
      self.water_mask                   = self.create_water_mask(self.morph_op, self.morph_op_iters)

      # count sample pixels and get sample min max coordinates
      self.sample_clip                  = self.clip_image(ee.Image(abs(self.dummy)))
      self.sample_total_pixel           = gee.get_image_counters(image=self.sample_clip.select("constant"), geometry=self.geometry, scale=self.scale)["constant"]
      coordinates_min, coordinates_max  = gee.get_image_min_max(image=self.sample_clip, geometry=self.geometry, scale=self.scale)
      self.sample_lon_lat               = [[float(coordinates_min['latitude']),float(coordinates_min['longitude'])],[float(coordinates_max['latitude']),float(coordinates_max['longitude'])]]

      # split geometry in tiles
      self.splitted_geometry            = self.split_geometry()

      # warning
      print("Statistics: scale="+str(self.sensor_params['scale'])+" meters, pixels="+str(self.sample_total_pixel)+", initial_date='"+self.dates_timeseries[0].strftime("%Y-%m-%d")+"', end_date='"+self.dates_timeseries[1].strftime("%Y-%m-%d")+"', interval_images='"+str(self.collection.size().getInfo())+"', interval_unique_images='"+str(len(self.dates_timeseries_interval))+"', water_mask_images='"+str(self.collection_water.size().getInfo())+"', grid_size='"+str(self.grid_size)+"', days_in='"+str(self.days_in)+"', days_out='"+str(self.days_out)+"', morph_op='"+str(self.morph_op)+"', morph_op_iters='"+str(self.morph_op_iters)+"', convolve='"+str(self.convolve)+"', convolve_radius='"+str(self.convolve_radius)+"', scaler='"+str(self.scaler_str)+"', model='"+str(self.model)+"', fill_missing='"+str(self.fill_missing)+"', reducer='"+str(self.reducer)+"', normalized='"+str(self.normalized)+"', class_mode='"+str(self.class_mode)+"', class_weight='"+str(self.class_weight)+"', propagate='"+str(self.propagate)+"', rs_train_size='"+str(self.rs_train_size)+"', rs_iter='"+str(self.rs_iter)+"', pca_size='"+str(self.pca_size)+"', attr_lat_lon='"+str(self.attribute_lat_lon)+"', attr_doy='"+str(self.attribute_doy)+"'")

      # gargage collect
      gc.collect()


  # create the water mask
  def create_water_mask(self, morph_op: str = None, morph_op_iters: int = 1):

    # water mask
    if self.sensor == "modis":
      water_mask = self.collection_water.mode().select('water_mask').eq(1)
    elif "landsat" in self.sensor:
      water_mask = self.collection_water.mode().select('water').eq(2)
    else:
      water_mask = self.collection_water.mode().select('water').gt(0)

    # morphological operations
    if not morph_op is None and morph_op != '':
      if morph_op   == 'closing':
        water_mask = water_mask.focal_max(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters).focal_min(kernel=ee.Kernel.circle(radius=1), iterations=morph_op_iters)
      elif morph_op == 'opening':
        water_mask = water_mask.focal_min(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters).focal_max(kernel=ee.Kernel.circle(radius=1), iterations=morph_op_iters)
      elif morph_op == 'dilation':
        water_mask = water_mask.focal_max(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters)
      elif morph_op == 'erosion':
        water_mask = water_mask.focal_min(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters)

    # build image with mask
    return ee.Image(0).blend(ee.Image(abs(self.dummy)).updateMask(water_mask))


  # clipping image
  def clip_image(self, image: ee.Image, scale: int = None, geometry: ee.Geometry = None):
    geometry  = self.geometry if geometry is None else geometry
    scale     = self.sensor_params['scale'] if scale is None else scale
    if image is None:
      return None
    clip      = image.clipToBoundsAndScale(geometry=geometry, scale=scale)
    if self.sensor_params['scale'] > self.scale:
      return clip.resample('bicubic').reproject(crs=image.projection(),scale=self.scale)
    else:
      return clip


  # applying water mask to indices
  def apply_water_mask(self, image: ee.Image, remove_empty_pixels = False, apply_attributes: bool = True):
    attributes = self.attributes if apply_attributes else self.attributes_clear
    for i, indice in enumerate(attributes):
        image = gee.apply_mask(image, self.water_mask, indice,  indice+"_water", remove_empty_pixels)
    return image


  # extract image from collection
  def extract_image_from_collection(self, date, convolve: bool = False, convolve_radius: int = 1, apply_attributes: bool = True, convolve_force_disabled: bool = False):
    try:
      collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=1)).strftime("%Y-%m-%d")))
      if int(collection.size().getInfo()) == 0:
        collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=2)).strftime("%Y-%m-%d")))
        if int(collection.size().getInfo()) == 0:
          return None
      if apply_attributes:
        image = self.apply_attributes(collection.max(), date)
      else:
        image = collection.max()
      if (convolve or self.convolve) and convolve_force_disabled == False:
        convolve_radius = convolve_radius if self.convolve_radius is None or self.convolve_radius == 0 else self.convolve_radius
        image = image.convolve(ee.Kernel.square(radius=convolve_radius, units='pixels', normalize=True))
      return self.apply_water_mask(image, False, apply_attributes=apply_attributes)
    except:
      return None


  # split images into tiles
  def split_geometry(self):

    # check total of pixels
    total = self.sample_total_pixel*(len(self.attributes)+2)
    if total > self.max_tile_pixels:

      # total of tiles needed
      tiles = math.ceil(self.sample_total_pixel/self.max_tile_pixels)

      # lat and lons range
      latitudes       = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=tiles+1)
      longitudes      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=tiles+1)

      # go through all latitudes and longitudes
      geometries = []
      for i, latitude in enumerate(latitudes[:-1]):
        for j, longitude in enumerate(longitudes[:-1]):
          x1 = [i,j]
          x2 = [i+1,j+1]
          geometry = gee.get_geometry_from_lat_lon(str(latitudes[x1[0]])+","+str(longitudes[x1[1]])+","+str(latitudes[x2[0]])+","+str(longitudes[x2[1]]))
          geometries.append(geometry)

      # return all created geometries
      return geometries

    else:
      
      # return single geometry
      return [gee.get_geometry_from_lat_lon(self.lat_lon)]


  # apply external attributes
  def apply_attributes(self, image: ee.Image, date):

    # sensors
    gldas                 = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").filterBounds(self.geometry).filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=1)).strftime("%Y-%m-%d"))).median().resample('bicubic').reproject(crs=image.projection(),scale=self.sensor_params['scale'])
    hydro_sheds           = ee.Image('WWF/HydroSHEDS/03DIR')
    alos                  = ee.Image("JAXA/ALOS/AW3D30/V2_2")
    modis_11A1            = ee.ImageCollection("MODIS/006/MOD11A1").filterBounds(self.geometry).filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=1)).strftime("%Y-%m-%d"))).median().resample('bicubic').reproject(crs=image.projection(),scale=self.sensor_params['scale'])
    
    # atributtes
    wind                  = gldas.select('Wind_f_inst').rename('wind')
    temperature           = gldas.select('SoilTMP0_10cm_inst').rename('temperature')
    drainage_direction    = hydro_sheds.select('b1').rename('drainage_direction')
    precipitation         = gldas.select('Rainf_f_tavg').rename('precipitation')
    elevation             = alos.select('AVE_DSM').rename('elevation')
    pressure              = gldas.select('Psurf_f_inst').rename('pressure')
    evapotranspiration    = gldas.select('Evap_tavg').rename('evapotranspiration')
    emissivity            = modis_11A1.select('Emis_31').rename('emissivity')
    
    # return imagem with the bands added
    return image.addBands([wind, temperature, drainage_direction, precipitation, elevation, pressure, evapotranspiration, emissivity])
    

  # apply label
  def apply_label(self, df: pd.DataFrame):

    # set default label value
    # prevent not found column error
    if 'label' in df.columns:
      df = df.drop(['label'], axis=1)
    df['label'] = 0

    # define labels for two or more positive thresholds
    for indice in self.indices_thresholds:
      if indice in df.columns:
        str_op = '>' if indice not in self.attributes_inverse else '<='
        df_query = df.query(str(indice)+" "+str_op+" "+str(self.indices_thresholds[indice]))
        df.loc[df['index'].isin(df_query['index']), 'label'] = df_query['label']+1

    # return same dataframe with new column
    gc.collect()
    return df


  # apply label in y numpy array
  def apply_label_y(self, y: np.array):

    # check if it is in class_mode
    if self.class_mode:
      gc.collect()
      return y.reshape(1,-1)[0]
    else:

      # attributes
      attributes_clear = [a for a in self.attributes_clear if a != 'cloud']
    
      # create support dataframe
      y = pd.DataFrame(data=y.reshape(-1,4), columns=attributes_clear)
      y['index'] = range(0,len(y))

      # return same dataframe with new column
      gc.collect()
      return self.apply_label(y)['label'].values.reshape(1,-1)[0]


  # apply grid number in a dataframe
  def apply_grids(self, df: pd.DataFrame):

    # set default row and column values
    # prevent not found column error
    if 'row' in df.columns:
      df = df.drop(['row'], axis=1)
    df['row'] = 0
    if 'column' in df.columns:
      df = df.drop(['column'], axis=1)
    df['column'] = 0

    # groupby to get resolution
    width  = len(df.groupby('lat'))
    height = len(df.groupby('lon'))

    # lat and lons range
    latitudes       = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=width+1)
    longitudes      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=height+1)

    # latitudes grid
    row = 1
    for i, lat in enumerate(latitudes):
      query = None
      if i == 0:
        query = "(lat <= "+str(latitudes[i+1])+")"
      elif i == len(latitudes)-2:
        query = "(lat > "+str(lat)+")"
      elif i < len(latitudes)-2:
        query = "(lat > "+str(lat)+" and lat <="+str(latitudes[i+1])+")"
      if query:
        df.loc[df['index'].isin(df.query(query)['index']), 'row'] = row
        row += 1

    # longitudes grid
    column = 1
    for i, lon in enumerate(longitudes):
      query = None
      if i == 0:
        query = "(lon <= "+str(longitudes[i+1])+")"
      elif i == len(longitudes)-2:
        query = "(lon > "+str(lon)+")"
      elif i < len(longitudes)-2:
        query = "(lon > "+str(lon)+" and lon <="+str(longitudes[i+1])+")"
      if query:
        df.loc[df['index'].isin(df.query(query)['index']), 'column'] = column
        column += 1

    # return modified dataframe
    return df


  # apply Reducer analysis
  def apply_reducer(self, X_train, X_test, mode=1):

    # warning
    print()
    print("Starting Feature Reducer...")

    # start
    n_features_start  = X_train.shape[1]

    # apply reducer
    if mode==1:
        pca               = decomposition.PCA(self.pca_size, random_state=self.random_state)
        X_train           = pca.fit_transform(X_train)
        X_test            = pca.transform(X_test)
        self.reducer      = pca
    elif mode==2:
        mlle              = manifold.LocallyLinearEmbedding(n_neighbors=n_features_start,n_components=int(n_features_start*0.75),method='modified',n_jobs=self.n_cores,random_state=self.random_state)
        X_train           = mlle.fit_transform(X_train)
        X_test            = mlle.transform(X_test)
        self.reducer      = mlle

    # finish
    n_features_end    = X_train.shape[1]

    # result
    print("Feature Reducer finished! Features were reduced from "+str(n_features_start)+" -> "+str(n_features_end)+" ...")
    return X_train, X_test


  # normalize indices
  def normalize_indices(self, df: pd.DataFrame):
    if self.normalized:
      print()
      print("Normalizing indices...")
      if 'ndwi' in self.attributes:
        df.loc[df['ndwi']<-1, 'ndwi'] = -1
        df.loc[df['ndwi']>1, 'ndwi'] = 1
      if 'ndvi' in self.attributes:
        df.loc[df['ndvi']<-1, 'ndvi'] = -1
        df.loc[df['ndvi']>1, 'ndvi'] = 1
      if 'sabi' in self.attributes:
        df.loc[df['sabi']<-1, 'sabi'] = -1
        df.loc[df['sabi']>1, 'sabi'] = 1
    return df


  # get cache files for datte
  def get_cache_files(self, date):
    prefix            = self.hash_string.encode()+self.lat_lon.encode()+self.sensor.encode()+str(self.morph_op).encode()+str(self.morph_op_iters).encode()+str(self.convolve).encode()+str(self.convolve_radius).encode()
    hash_image        = hashlib.md5(prefix+(date.strftime("%Y-%m-%d")+'original').encode())
    hash_timeseries   = hashlib.md5(prefix+(self.dates_timeseries[0].strftime("%Y-%m-%d")+self.dates_timeseries[1].strftime("%Y-%m-%d")).encode()+str(self.days_in).encode()+str(self.days_out).encode()+str(self.normalized).encode()+str(self.fill_missing).encode()+str(self.reducer).encode()+str(self.class_mode).encode())
    hash_classifiers  = hashlib.md5(prefix+(self.dates_timeseries[0].strftime("%Y-%m-%d")+self.dates_timeseries[1].strftime("%Y-%m-%d")).encode()+('classifier').encode())
    hash_runtime      = hashlib.md5(prefix+(self.dates_timeseries[0].strftime("%Y-%m-%d")+self.dates_timeseries[1].strftime("%Y-%m-%d")).encode()+('runtime').encode())
    return [self.cache_path+'/'+hash_image.hexdigest(), self.cache_path+'/'+hash_timeseries.hexdigest(), self.cache_path+'/'+hash_classifiers.hexdigest(), self.cache_path+'/'+hash_runtime.hexdigest()]


  # fill missing dates in a dataframe
  def fill_missing_dates(self, df: pd.DataFrame, min_date = None, max_date = None, fill: str = 'dummy'):

    # remove clouds and fix date column
    df                    = df[df['cloud'] == 0.0]
    df['date']            = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='ignore')

    # get min and max dates
    if min_date is None:
      min_date = min(df['date'])
    if max_date is None:
      max_date = max(df['date'])

    # extract pixel information
    if self.pixels_lat_lons is None:
      pixels_lat_lons       = np.unique(df[['pixel','row','column','lat','lon']].values.astype(str), axis=0)
    else:
      pixels_lat_lons       = self.pixels_lat_lons
    pixels_dates          = df[['pixel','row','column','lat','lon','date']]
    pixels_dates          = pixels_dates.values.astype(str)

    # extract missing dates
    range_dates           = list(set([d.strftime('%Y-%m-%d') for d in pd.date_range(start=min_date,end=max_date).to_pydatetime().tolist()]))
    range_dates.sort()
    
    # extract independent arrays
    array_pixels          = np.repeat(pixels_lat_lons[:,0], len(range_dates))
    array_rows            = np.repeat(pixels_lat_lons[:,1], len(range_dates))
    array_columns         = np.repeat(pixels_lat_lons[:,2], len(range_dates))
    array_lats            = np.repeat(pixels_lat_lons[:,3], len(range_dates))
    array_lons            = np.repeat(pixels_lat_lons[:,4], len(range_dates))
    
    # extract independent array of dates (repeating)
    array_dates = []
    for i in range(0,len(pixels_lat_lons)):
      array_dates = np.concatenate((array_dates, range_dates), axis=0)

    # zip it all together
    array         = np.array(list(zip(array_pixels,array_rows,array_columns,array_lats,array_lons,array_dates)))
    
    # creating future merging dataframes
    df1           = pd.DataFrame(array).sort_values(by=[0, 1, 2, 3, 4, 5])
    df2           = pd.DataFrame(pixels_dates).sort_values(by=[0, 1, 2, 3, 4, 5])
    
    # casting df1
    df1[0]        = pd.to_numeric(df1[0], downcast='integer')
    df1[1]        = pd.to_numeric(df1[1], downcast='integer')
    df1[2]        = pd.to_numeric(df1[2], downcast='integer')
    df1[3]        = df1[3].astype(float)
    df1[4]        = df1[4].astype(float)
    df1[5]        = pd.to_datetime(df1[5], format='%Y-%m-%d', errors='ignore')
    
    # casting df2
    df2[0]        = pd.to_numeric(df2[0], downcast='integer')
    df2[1]        = pd.to_numeric(df2[1], downcast='integer')
    df2[2]        = pd.to_numeric(df2[2], downcast='integer')
    df2[3]        = df2[3].astype(float)
    df2[4]        = df2[4].astype(float)
    df2[5]        = pd.to_datetime(df2[5], format='%Y-%m-%d', errors='ignore')
    
    # merging df with the missing dates
    df_diff       = df2.merge(df1, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only'][[0, 1, 2, 3, 4, 5]]
    df_diff       = df_diff.rename(columns={0: "pixel", 1: "row", 2: "column", 3: "lat", 4: "lon", 5: "date"})
    
    # incorporate diff in the selected dataframe
    df            = pd.concat([df,df_diff], sort=False)
    
    # fix values in the new dataframe
    df = df.reset_index(drop=True)
    if fill == 'dummy':
      df = df.fillna(self.dummy).sort_values(by=['pixel','date'])
    elif fill == 'ffill' or fill == 'bfill':
      df = df.fillna(method=fill).sort_values(by=['pixel','date'])
    else:
      df = df.groupby('pixel').apply(lambda group: group.reset_index().set_index('date').interpolate(method=fill)).drop(['pixel','level_0'], axis=1).reset_index().sort_values(by=['pixel','date'])
    
    # fix index and final data
    df['index']   = np.arange(start=0, stop=len(df), step=1, dtype=np.int64)
    df['doy']     = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='ignore').dt.month

    # return result
    gc.collect()
    return df.reset_index(drop=True)


  # merge two or more timeseries
  def merge_timeseries(self, df_list: list):
    df = pd.concat(df_list, ignore_index=True, sort=False)
    gc.collect()
    return self.fix_timeseries(df)


  # fix timeseries values
  def fix_timeseries(self, df: pd.DataFrame):

    # fix index and pixel columns
    df['index'] = np.arange(start=0, stop=len(df), step=1, dtype=np.int64)
    if 'pixel' not in df.columns:
      df['pixel'] = df['index']
    df['pixel'] = df['pixel'].astype(dtype=np.int64, errors='ignore')

    # fix other columns
    df[['lat','lon']]       = df[['lat','lon']].astype(dtype=np.float64, errors='ignore')
    df['date']              = pd.to_datetime(df['date'], errors='ignore')
    df['doy']               = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='ignore').dt.month.astype(dtype=np.int64, errors='ignore')

    # fix column order
    df[self.attributes]     = df[self.attributes].apply(pd.to_numeric, errors='ignore')

    # clear memory
    gc.collect()

    # return fixed datframe
    return df.sort_values(by=['date', 'pixel'])


  # extract image's coordinates and pixels values
  def extract_image_pixels(self, image: ee.Image, date, apply_attributes: bool = True, force_cache: bool = False, disable_threshold: bool = False):

    # warning
    print("Processing date ["+str(date.strftime("%Y-%m-%d"))+"]...")

    # attributes
    lons_lats_attributes     = None
    cache_files              = self.get_cache_files(date)
    df_timeseries            = pd.DataFrame(columns=self.df_columns)

    # trying to find image in cache
    try:

      # warning - user disabled cache
      if self.force_cache or force_cache:
        raise Exception()

      # extract pixel values from cache
      lons_lats_attributes  = joblib.load(cache_files[0])
      if lons_lats_attributes is None or len(lons_lats_attributes) != (len(self.df_columns if apply_attributes else self.df_columns_clear)-6):
        raise Exception()

      # check if image is empty and return empty dataframe
      if len(lons_lats_attributes) == 0:
        
        # clear memory
        del lons_lats_attributes
        gc.collect()

        # return empty dataframe
        return df_timeseries

    # error finding image in cache
    except:

      # image exists
      try:

        # go through each tile
        if apply_attributes:
          lons_lats_attributes  = np.array([]).reshape(0, len(self.attributes)+2)
        else:
          lons_lats_attributes  = np.array([]).reshape(0, len(self.attributes_clear)+2)
        for i, geometry in enumerate(self.splitted_geometry):

          # geometry
          print("Extracting geometry ("+str(len(lons_lats_attributes))+") "+str(i+1)+" of "+str(len(self.splitted_geometry))+"...")

          # counters
          clip = self.clip_image(self.extract_image_from_collection(date=date, apply_attributes=apply_attributes), geometry=geometry, scale=self.scale)
          if not clip is None:
            counters            = clip.select([self.sensor_params['red'],'water','water_nocloud']).reduceRegion(reducer=ee.Reducer.count(), geometry=geometry, scale=self.scale).getInfo()

            # count image pixels
            total_pixel         = int(counters[self.sensor_params['red']])
            total_water_pixel   = int(counters['water'])
            total_water_nocloud = int(counters['water_nocloud'])

            # calculate pixel score
            if (len(self.splitted_geometry) > 1):
              sample_total_pixel = gee.get_image_counters(image=self.clip_image(ee.Image(abs(self.dummy))).select("constant"), geometry=geometry, scale=self.scale)["constant"]
            else:
              sample_total_pixel = self.sample_total_pixel

            # score total pixel and cloud
            pixel_score         = 0.0 if total_pixel == 0 else round(total_pixel/sample_total_pixel,5)
            water_nocloud_score = 0.0 if total_water_nocloud == 0 else round(total_water_nocloud/total_water_pixel, 5)

            # warning
            print("Pixel and cloud score for geometry #"+str(i+1)+": "+str(pixel_score)+" and "+str(water_nocloud_score)+"!")

            # check if image is good for processing
            if (pixel_score >= 0.50 and water_nocloud_score >= self.cloud_threshold) or disable_threshold:
              if apply_attributes:
                geometry_lons_lats_attributes = gee.extract_latitude_longitude_pixel(image=clip, geometry=self.geometry, bands=[a+"_water" for a in self.attributes], scale=self.sensor_params['scale'])
              else:
                geometry_lons_lats_attributes = gee.extract_latitude_longitude_pixel(image=clip, geometry=self.geometry, bands=[a+"_water" for a in self.attributes_clear], scale=self.sensor_params['scale'])
              lons_lats_attributes = np.vstack((lons_lats_attributes, geometry_lons_lats_attributes))
              del geometry_lons_lats_attributes
              gc.collect()

        # empty image
        if len(lons_lats_attributes) == 0:

          # warning
          print("Image is not good for processing and it was discarded!")
          
          # Clear pixels, image is not good
          lons_lats_attributes = None
          joblib.dump(np.array([]), cache_files[0], compress=3)

          # clear memory
          del lons_lats_attributes
          gc.collect()

          # return empty dataframe
          return df_timeseries

        # save in cache
        else:
          joblib.dump(lons_lats_attributes, cache_files[0], compress=3)

      # error in the extraction process
      except:
        
        # warning
        print("Error while extracting pixels from image "+str(date.strftime("%Y-%m-%d"))+": "+str(traceback.format_exc()))

        # reset attributes
        lons_lats_attributes = None

        # remove cache file
        if os.path.exists(cache_files[0]):
          os.remove(cache_files[0])

    # check if has attributes to process
    try:

      # check if they are valid
      if lons_lats_attributes is None:
        raise Exception()

      # build arrays
      extra_attributes        = np.array(list(zip([0]*len(lons_lats_attributes),[0]*len(lons_lats_attributes),[0]*len(lons_lats_attributes),[0]*len(lons_lats_attributes),[date.strftime("%Y-%m-%d")]*len(lons_lats_attributes),[0]*len(lons_lats_attributes))))
      lons_lats_attributes    = np.hstack((extra_attributes, lons_lats_attributes))

      # build dataframe and fix column index values
      df_timeseries           = pd.DataFrame(data=lons_lats_attributes, columns=self.df_columns if apply_attributes else self.df_columns_clear).infer_objects().sort_values(['lat','lon']).reset_index(drop=True)
      df_timeseries['pixel']  = range(0,len(df_timeseries))

      # show example of time series
      print(df_timeseries.head())

      # gabagge collect
      del lons_lats_attributes, extra_attributes
      gc.collect()

      # return all pixels in an three pandas format
      return df_timeseries

    # no data do return
    except:

      # warning
      print("Error while extracting pixels from image "+str(date.strftime("%Y-%m-%d"))+": "+str(traceback.format_exc()))

      # remove cache file
      if os.path.exists(cache_files[0]):
        os.remove(cache_files[0])
      
      # clear memory
      del lons_lats_attributes
      gc.collect()

      # return empty dataframe
      return df_timeseries


  # process a timeseries
  def process_timeseries_data(self, force_cache: bool = False):

    # warning
    print()
    print("Starting time series processing ...")

    # attributes
    df_timeseries  = pd.DataFrame(columns=self.df_columns)
    attributes = [a for a in self.attributes if a != 'cloud']

    # check timeseries is already on cache
    cache_files    = self.get_cache_files(date=dt.now())
    try:

      # warning
      print("Trying to extract it from the cache...")

      # warning 2
      if self.force_cache or force_cache:
        print("User selected option 'force_cache'! Forcing building of time series...")
        raise Exception()

      # extract dataframe from cache
      df_timeseries = self.fix_timeseries(df=joblib.load(cache_files[1]))
   
    # if do not exist, process normally and save it in the end
    except:

      # warning
      print("Error trying to get it from cache: either doesn't exist or is corrupted! Creating it again...")

      # process all dates in time series
      for date in self.dates_timeseries_interval:

        # extract pixels from image
        # check if is good image (with pixels)
        try:
          df_timeseries_ = self.extract_image_pixels(image=self.extract_image_from_collection(date=date), date=date)
          if df_timeseries_.size > 0:
            df_timeseries = self.merge_timeseries(df_list=[df_timeseries, df_timeseries_])
          gc.collect()
        except:
          pass

      # get only good dates
      # fix dataframe index
      if not df_timeseries is None:
        df_timeseries['index'] = range(0,len(df_timeseries))

        # save in cache
        if self.cache_path:
          joblib.dump(df_timeseries, cache_files[1], compress=3)

    # groupping latitude e longitude
    self.resolution[0]  = len(df_timeseries.groupby('lat'))
    self.resolution[1]  = len(df_timeseries.groupby('lon'))

    # remove dummies
    for attribute in attributes:
      df_timeseries = df_timeseries[(df_timeseries[attribute]!=abs(self.dummy))]

    # change cloud values
    df_timeseries.loc[df_timeseries['cloud'] == abs(self.dummy), 'cloud'] = 0.0

    # remove duplicated values
    df_timeseries.drop_duplicates(subset=['pixel','date','lat','lon']+self.attributes, keep='last', inplace=True)

    # generate grid limits
    self.limits = [math.ceil(self.resolution[0]/self.plots_grid),math.ceil(self.resolution[1]/self.plots_grid)]

    # save modified dataframe to its original variable and apply grids
    self.df_timeseries = self.apply_grids(df=df_timeseries[self.df_columns].dropna())

    # normalize indexes
    self.df_timeseries = self.normalize_indices(df=self.df_timeseries)

    # correction of empty dates
    self.dates_timeseries_interval = [dt.strptime(date.astype(str).split("T")[0],"%Y-%m-%d") for date in self.df_timeseries['date'].unique()]

    # garbagge collect
    del df_timeseries
    gc.collect()

    # warning
    print("finished!")


  # process training dataset
  def process_training_data(self, df: pd.DataFrame):

    # warning
    print()
    print("Processing training data ("+str(len(df['date'].unique()))+" images)...")

    # show statistics
    print(df.describe())

    # warning
    print()
    print("Filling empty dates...")

    # attributes
    attributes_clear  = [a for a in self.attributes_clear if a != 'cloud']

    # fill empty dates
    df = self.fill_missing_dates(df=df, fill=self.fill_missing)

    # added labels
    df = self.apply_label(df)

    # fix dataframes
    self.pixels_lat_lons = np.unique(df[['pixel','row','column','lat','lon']].values.astype(str), axis=0)
    self.df_timeseries = df[self.df_columns+['label']]

    # show statistics
    print(df.describe())

    # warning
    print()
    print("Selecting classification dates...")

    # get the classification dates based o the selected days_out
    self.classification_dates = [(self.dates_timeseries_interval[-1] - td(days=int(d))) for d in range(0, self.days_out)]
    self.classification_dates.sort()

    # show dates
    print(self.classification_dates)

    # warning
    print()
    print("Selecting prediction dates...")

    # build base dates that will be used in prediction
    self.predict_dates = [(self.classification_dates[-1] + td(days=int(d))) for d in range(1, len(self.classification_dates)+1)]
    self.predict_dates.sort()

    # show dates
    print(self.predict_dates)

    # warning
    print()
    print("Shifting dates to build training set...")

    # shifting dates em buling training set
    df_pixel = misc.series_to_supervised(df[self.attributes_selected+['label']].values, self.days_in, self.days_out, dropnan=True)

    # remove dummies
    if self.remove_dummies == True and self.fill_missing == 'dummy':
      for column in df_pixel.columns:
        df_pixel = df_pixel[(df_pixel[column]!=self.dummy)]
    
    # get only data that matters
    str_label       = 'var'+str(len(self.attributes_selected)+1)
    str_lat         = 'var'+str(self.attributes_selected.index('lat')+1)
    str_lon         = 'var'+str(self.attributes_selected.index('lon')+1)
    str_doy         = 'var'+str(self.attributes_selected.index('doy')+1)
    if "ndwi" in self.attributes_selected:
      str_label_ndwi  = 'var'+str(self.attributes_selected.index('ndwi')+1)
    else:
      str_label_ndwi = ""
    if "ndvi" in self.attributes_selected:
      str_label_ndvi = 'var'+str(self.attributes_selected.index('ndvi')+1)
    else:
      str_label_ndvi = ""
    if "sabi" in self.attributes_selected:
      str_label_sabi = 'var'+str(self.attributes_selected.index('sabi')+1)
    else:
      str_label_sabi = ""
    if "fai" in self.attributes_selected:
      str_label_fai = 'var'+str(self.attributes_selected.index('fai')+1)
    else:
      str_label_fai = ""
    in_labels       = [s for i, s in enumerate(df_pixel.columns) if 't-' in s and not str_label+'(' in s]
    in_labels       = [s for i, s in enumerate(in_labels) if (self.attribute_lat_lon or not str_lat in s) and (self.attribute_lat_lon or not str_lon in s) and (self.attribute_doy or not str_doy in s)]
    if self.class_mode:
      out_labels      = [s for i, s in enumerate(df_pixel.columns) if str_label+'(' in s  and not 't-' in s]
    else:
      out_labels      = [s for i, s in enumerate(df_pixel.columns) if (str_label_ndwi+'(' in s or str_label_ndvi+'(' in s or str_label_sabi+'(' in s or str_label_fai+'(' in s) and not 't-' in s]
    df_pixel        = df_pixel[in_labels+out_labels]
    new_columns     = list(range(0,len(in_labels+out_labels)))
    for i, c in enumerate(new_columns[:-len(out_labels)]):
      new_columns[new_columns.index(c)] = "in_"+str(i)
    for i, c in enumerate(new_columns[-len(out_labels):]):
      new_columns[new_columns.index(c)] = "out_"+str(i)
    df_pixel.columns = new_columns

    # get only good labels (0 or all indexes)
    if self.class_mode:
      df_queries = []
      for c in df_pixel.columns:
        if 'out_' in c:
          df_queries.append("("+c+" == 0 or "+c+" == "+str(len(attributes_clear))+")")
      df_pixel = df_pixel.query(" and ".join(df_queries))

    # show statistics
    print(df_pixel.describe())

    # data
    X = df_pixel[df_pixel.columns[:-len(out_labels)]].values.reshape((-1, len(df_pixel.columns[:-len(out_labels)])))
    y = df_pixel[df_pixel.columns[-len(out_labels):]].values.reshape((-1, len(df_pixel.columns[-len(out_labels):])))
    X = self.scaler.fit_transform(X, y)
    if self.shuffle:
      print()
      print("Shuffling...")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.90, shuffle=self.shuffle, random_state=self.random_state)

    # check if user defined to apply reducer
    if self.reducer:
      X_train, X_test   = self.apply_reducer(X_train, X_test)

    # building array of final dataframes splitting by train and test sets
    self.df_train       = [X_train, y_train]
    self.df_test        = [X_test, y_test]

    # clear variables
    del df_pixel, X_train, X_test, y_train, y_test
    gc.collect()

    # Final statistics
    print()
    print("Train datasets length: train=(%s, %s), test=(%s, %s)" %(self.df_train[0].shape, self.df_train[1].shape, self.df_test[0].shape, self.df_test[1].shape))
    print("finished!")


  # start training process
  def train(self, disable_gpu: bool = True, batch_size: int = 2048):

    # jump line
    print()
    print("Starting the training process...")

    # check if it should disable GPU in Tensorflow process
    if disable_gpu:
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # total attributes
    in_attributes   = self.df_train[0].shape[1]
    out_attributes  = self.df_train[1].shape[1]

    # fix RandomizedSearchCV traning size based on maximum data size
    self.rs_train_size = int(self.rs_train_size) if self.rs_train_size > 1 else float(self.rs_train_size) if self.rs_train_size <= len(self.df_train) else len(self.df_train)

    # get data
    X_train, y_train                    = self.df_train
    X_test, y_test                      = self.df_test
    X_gridsearch, _, y_gridsearch, _    = model_selection.train_test_split(X_train, y_train, train_size=self.rs_train_size, random_state=self.random_state, shuffle=self.shuffle)
    
    # fill randomized search dataframe
    self.df_randomizedsearch          = [X_gridsearch,y_gridsearch]

    # warning
    print()
    print("RandomizedSearchCV datasets length: rs=(%s, %s)" %(X_gridsearch.shape, y_gridsearch.shape))

    # warning
    if self.class_mode:
      print()
      print("RandomizedSearchCV datasets frequency:")
      print(np.asarray((np.unique(y_gridsearch, return_counts=True)), dtype=int).T)
    
    # compute class weight from gridsearch dataset
    classes = None
    class_weight = None
    class_weight2 = None
    if self.class_mode:
      classes = np.unique(self.apply_label_y(y_gridsearch))
      if self.class_weight:
        class_weight = []
        class_weight2 = dict(zip(classes,utils.class_weight.compute_class_weight("balanced", classes, self.apply_label_y(y_gridsearch))))
        for c in range(0,y_gridsearch.shape[1]):
          class_weight.append(dict(zip(classes,utils.class_weight.compute_class_weight("balanced", np.unique(y_gridsearch[:,c]), y_gridsearch[:,c]))))
        print()
        print("Defining class weights...")
        print(class_weight)
        print(class_weight2)
      else:
        class_weight = None
        class_weight2 = None

    # create df_true dataframe
    y_test_label = self.apply_label_y(y_test)

    # keras parameter
    tf.keras.backend.set_floatx('float64')

    ########################################################################
    # 1) MultiLayer Perceptron (Keras)

    # check if model was selected in options
    if self.model is None or self.model == "mlp":
      try:

        # jump line
        print()
        print("Creating the MultiLayer Perceptron with RandomizedSearchCV parameterization model...")

        # change label size
        if self.class_mode:
          y_gridsearch_categorical = tf.keras.utils.to_categorical(y_gridsearch)
          y_gridsearch_ = y_gridsearch_categorical.reshape(y_gridsearch_categorical.shape[0], y_gridsearch_categorical.shape[1]*y_gridsearch_categorical.shape[2])
          y_train_categorical = tf.keras.utils.to_categorical(y_train)
          y_train_ = y_train_categorical.reshape(y_train_categorical.shape[0], y_train_categorical.shape[1]*y_train_categorical.shape[2])
          self.randomizedsearch_categorical = y_gridsearch_categorical.shape

        #################################
        # Custom MultiLayer Perceptron Model
        # Change KerasRegressor
        class KerasRegressorModified(tf.keras.wrappers.scikit_learn.KerasRegressor):
          def fit(self, x, y, **kwargs):
            return super(tf.keras.wrappers.scikit_learn.KerasRegressor, self).fit(x, y, validation_split=0.3, batch_size=batch_size, class_weight=class_weight2)

        if self.class_mode:
          def mlp_modified(optimizer='adam', dropout=0.2):
            mlp_modified = tf.keras.Sequential(
              [
                tf.keras.layers.Dense(512, activation="tanh", input_shape=(in_attributes,)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(256, activation="tanh"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(128, activation="tanh"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(y_gridsearch_.shape[1], activation='softmax')
              ]
            )
            mlp_modified.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
            return mlp_modified
        else:
          def mlp_modified(optimizer='adam', dropout=0.2):
            mlp_modified = tf.keras.Sequential(
              [
                tf.keras.layers.Dense(512, activation="tanh", input_shape=(in_attributes,)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(256, activation="tanh"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(128, activation="tanh"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(out_attributes)
              ]
            )
            mlp_modified.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            return mlp_modified
        ##################################

        # RandomizedSearchCV
        random_grid = {
          'dropout':       [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
          'epochs':        [10, 50, 100, 200, 300, 400, 500],
          'optimizer':     ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        }

        # apply RandomizedSearchCV and get best estimator and training the model
        start_time = time.time()
        rs = model_selection.RandomizedSearchCV(estimator=KerasRegressorModified(build_fn=mlp_modified, verbose=1), param_distributions=random_grid, scoring="neg_mean_squared_error", n_iter=self.rs_iter, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch, y_gridsearch_ if self.class_mode else y_gridsearch)
        mlp = rs.best_estimator_
        mlp.fit(X_train, y_train_ if self.class_mode else y_train)

        # training the model
        # model name
        str_model = "MLP (dropout="+str(rs.best_params_['dropout'])+",epochs="+str(rs.best_params_['epochs'])+",optimizer="+str(rs.best_params_['optimizer'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = mlp

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_pred = mlp.predict(X_test)
        if self.class_mode:
          y_pred[y_pred > 0.5] = 1
          y_pred[y_pred <= 0.5] = 0
          y_pred = y_pred.reshape(y_pred.shape[0], y_gridsearch_categorical.shape[1], y_gridsearch_categorical.shape[2])
          y_pred = np.argmax(y_pred,axis=1)
          y_pred_label = self.apply_label_y(y_pred.astype("int32"))
        else:
          y_pred_label  = self.apply_label_y(y_pred)
        measures = misc.concordance_measures(metrics.confusion_matrix(y_test_label, y_pred_label), y_test_label, y_pred_label)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])

        # warning
        print("finished!")
      except:
        pass

    ########################################################################



    ########################################################################
    # 2) LSTM (Keras)

    # check if model was selected in options
    if self.model is None or self.model == "lstm":
      try:

        # jump line
        print()
        print("Creating the LSTM (Bidirection w/ Encoder-Decoder) with RandomizedSearchCV parameterization model...")

        # attributes
        days_in = self.days_in
        #days_out = self.days_out
        in_size = len(self.attributes_selected)
        #out_size = 1 if self.class_mode else len(self.indices_thresholds)

        # enabled attributes
        if not self.attribute_lat_lon:
          in_size = in_size - 2
        if not self.attribute_doy:
          in_size = in_size - 1

        #################################
        # Custom LSTM Model
        # Change KerasRegressor
        class KerasRegressorModified(tf.keras.wrappers.scikit_learn.KerasRegressor):
          def fit(self, x, y, **kwargs):
            kwargs = self.filter_sk_params(tf.keras.Sequential.predict_classes, kwargs)
            sample_weight = np.array([class_weight2[y_] for y_ in np.mean(y.reshape(-1,y.shape[2]), axis=1).astype("int32")]) if class_weight2 else None
            return super(tf.keras.wrappers.scikit_learn.KerasRegressor, self).fit(x, y, validation_split=0.3, batch_size=int(batch_size/2), sample_weight=sample_weight, **kwargs)
          def predict(self, x, **kwargs):
            return self.model.predict(x, **kwargs)
            
        def lstm_modified(optimizer='adam', dropout=0.5, size=32, epochs=50):
          tf.keras.backend.set_floatx('float64')
          lstm_modified = tf.keras.Sequential(
            [
              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(size), activation="tanh", input_shape=(1, X_gridsearch.shape[1]))),
              tf.keras.layers.Dropout(dropout),
              tf.keras.layers.RepeatVector(days_in),
              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(size), activation='tanh', return_sequences=True)),
              tf.keras.layers.Dropout(dropout),
              tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(int(size), activation='tanh')),
              tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
            ]
          )
          lstm_modified.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
          return lstm_modified

        def lstm_scorer(y_true, y_pred):
          return -metrics.mean_squared_error(y_true.reshape(1,-1)[0], y_pred.reshape(1,-1)[0])
        ##################################

        # RandomizedSearchCV
        random_grid = {
          'dropout':      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
          'epochs':       [5, 10, 20, 30, 40, 50, 100],
          'size':         [16, 32, 64, 128, 256, 512],
          'optimizer':    ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'], 
        }

        # apply RandomizedSearchCV and get best estimator and training the model
        start_time = time.time()
        rs = model_selection.RandomizedSearchCV(estimator=KerasRegressorModified(build_fn=lstm_modified, verbose=1), param_distributions=random_grid, scoring=metrics.make_scorer(lstm_scorer), n_iter=self.rs_iter, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch.reshape(X_gridsearch.shape[0], 1, X_gridsearch.shape[1]), y_gridsearch.reshape(y_gridsearch.shape[0], 1, y_gridsearch.shape[1]))
        lstm = rs.best_estimator_
        lstm.fit(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), y_train.reshape(y_train.shape[0], 1, y_train.shape[1]))

        # training the model
        # model name
        str_model = "LSTM (dropout="+str(rs.best_params_['dropout'])+",epochs="+str(rs.best_params_['epochs'])+",size="+str(rs.best_params_['size'])+",optimizer="+str(rs.best_params_['optimizer'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = lstm

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_pred = lstm.predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
        if self.class_mode:
          y_pred_label = self.apply_label_y(y_pred.astype("int32"))
        else:
          y_pred_label = self.apply_label_y(y_pred)
        measures = misc.concordance_measures(metrics.confusion_matrix(y_test_label, y_pred_label), y_test_label, y_pred_label)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])

        # warning
        print("finished!")
      except:
        sys.exit()
        pass

    #########################################################################



    ########################################################################
    # 3) Random Forest MultiClassifier (sklearn)

    # check if model was selected in options
    if self.model is None or self.model == "rf":
      try:

        # jump line
        print()
        print("Creating the Random Forest Regressor/Classifier with RandomizedSearchCV parameterization model...")

        # RandomizedSearchCV
        random_grid = {
          'n_estimators':       np.linspace(1, 2000, num=101, dtype=int, endpoint=True),
          'max_depth':          np.linspace(1, 30, num=30, dtype=int, endpoint=True),
          'min_samples_split':  np.linspace(2, 20, num=10, dtype=int, endpoint=True),
          'min_samples_leaf':   np.linspace(2, 20, num=10, dtype=int, endpoint=True),
          'max_features':       ['auto', 'sqrt', 1.0, 0.75, 0.50],
          'bootstrap':          [True]
        }

        # apply RandomizedSearchCV and get best estimator and training the model
        start_time = time.time()
        if self.class_mode:
          rs = model_selection.RandomizedSearchCV(estimator=ensemble.RandomForestClassifier(n_jobs=self.n_cores, verbose=1, random_state=self.random_state, class_weight=class_weight), param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=self.rs_iter, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        else:
          rs = model_selection.RandomizedSearchCV(estimator=ensemble.RandomForestRegressor(n_jobs=self.n_cores, verbose=1, random_state=self.random_state), param_distributions=random_grid, scoring="neg_mean_squared_error", n_iter=self.rs_iter, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch, y_gridsearch)
        rf = rs.best_estimator_
        rf.fit(X_train, y_train)

        # model name
        str_model = "RF (n_estimators="+str(rs.best_params_['n_estimators'])+",max_features="+str(rs.best_params_['max_features'])+",max_depth="+str(rs.best_params_['max_depth'])+",min_samples_leaf="+str(rs.best_params_['min_samples_leaf'])+",min_samples_split="+str(rs.best_params_['min_samples_split'])+",bootstrap="+str(rs.best_params_['bootstrap'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = rf

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_pred_label  = self.apply_label_y(rf.predict(X_test))
        measures      = misc.concordance_measures(metrics.confusion_matrix(y_test_label, y_pred_label), y_test_label, y_pred_label)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])

        # warning
        print("finished!")
      except:
        pass

    ########################################################################



    ########################################################################
    # 4) Support Vector Machines Classifier (sklearn)

    # check if model was selected in options
    if self.model is None or self.model == "svm":
      try:

        # jump line
        print()
        print("Creating the Support Vector Machine Regressor/Classifier with RandomizedSearchCV parameterization model...")

        # RandomizedSearchCV
        random_grid = [
          {
            'estimator__kernel':    ['rbf'],
            'estimator__gamma':     scipy.stats.expon(scale=.100),
            'estimator__C':         scipy.stats.expon(scale=1),
            'estimator__epsilon':   [0.001],
            'estimator__shrinking': [True]
          }
        ]

        # apply RandomizedSearchCV and get best estimator
        start_time = time.time()
        if self.class_mode:
          random_grid_clear = []
          for g in random_grid:
            if g['estimator__epsilon']:
              del g['estimator__epsilon']
              random_grid_clear.append(g)
          rs = model_selection.RandomizedSearchCV(estimator=multioutput.MultiOutputClassifier(svm.SVC(verbose=0, random_state=self.random_state, class_weight=class_weight2), n_jobs=self.n_cores), param_distributions=random_grid_clear, scoring='neg_mean_squared_error', n_iter=self.rs_iter, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        else:
          rs = model_selection.RandomizedSearchCV(estimator=multioutput.MultiOutputRegressor(svm.SVR(verbose=0), n_jobs=self.n_cores), param_distributions=random_grid, scoring="neg_mean_squared_error", n_iter=self.rs_iter, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch, y_gridsearch)
        svm_model = rs.best_estimator_
        svm_model.fit(X_train, y_train)
        
        # model name
        if self.class_mode:
          str_model = "SVM (kernel="+str(rs.best_params_['estimator__kernel'])+",C="+str(rs.best_params_['estimator__C'])+",gamma="+str(rs.best_params_['estimator__gamma'])+",shrinking="+str(rs.best_params_['estimator__shrinking'])+")"
        else:
          str_model = "SVM (kernel="+str(rs.best_params_['estimator__kernel'])+",C="+str(rs.best_params_['estimator__C'])+",gamma="+str(rs.best_params_['estimator__gamma'])+",epsilon="+str(rs.best_params_['estimator__epsilon'])+",shrinking="+str(rs.best_params_['estimator__shrinking'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = svm_model

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_pred_label  = self.apply_label_y(svm_model.predict(X_test))
        measures      = misc.concordance_measures(metrics.confusion_matrix(y_test_label, y_pred_label), y_test_label, y_pred_label)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])

        # warning
        print("finished!")
      except:
        pass

    ########################################################################



  # start the prediction process
  def predict(self, folder: str, path: str = None, chla_threshold: int = 20):

    # jump line
    print()
    print("Starting the predicting process...")

    # attributes
    attributes_clear  = [a for a in self.attributes_clear if a != 'cloud']

    # dictionary of attributes
    dict_attributes = {}
    dict_attributes[0] = 'lat'
    dict_attributes[1] = 'lon'
    for i, attribute in enumerate(attributes_clear):
      dict_attributes[i+2] = attribute

    # check folder exists
    if not os.path.exists(folder):
      os.mkdir(folder)

    # check folder exists - geojson
    if not os.path.exists(folder+'/geojson'):
      os.mkdir(folder+'/geojson')

    # check folder exists - csv
    if not os.path.exists(folder+'/csv'):
      os.mkdir(folder+'/csv')

    # check folder exists - image
    if not os.path.exists(folder+'/image'):
      os.mkdir(folder+'/image')

    # attributes
    df_predict 	          = pd.DataFrame(columns=self.df_columns)
    df_classification 	  = pd.DataFrame(columns=self.df_columns)

    # attributes reference
    df_columns_reference  = ['pixel', 'date', 'lat', 'lon', 'chla']
    df_reference          = pd.DataFrame(columns=df_columns_reference)

    # increase sensor collectino to get new images
    collection, _       = gee.get_sensor_collections(geometry=self.geometry, sensor=self.sensor, dates=[dt.strftime(self.dates_timeseries[0], "%Y-%m-%d"), dt.strftime(self.predict_dates[-1] + td(days=1), "%Y-%m-%d")])
    self.collection     = collection

    # process all dates that will be used in prediction
    print()
    print("Extracting images ("+str(len(self.classification_dates))+") that it will be used in the prediction process")
    for date in self.classification_dates:
      # extract pixels from image
      # check if is good image (with pixels)
      df_classification_ = self.extract_image_pixels(image=self.extract_image_from_collection(date=date), date=date, disable_threshold=True)
      if df_classification_.size > 0:
        df_classification = self.merge_timeseries(df_list=[df_classification, df_classification_])

      # get only good dates
      # fix dataframe index
      if not df_classification is None:
        df_classification['index'] = range(0,len(df_classification))

    # process all dates that will be predicted
    print()
    print("Extracting images ("+str(len(self.predict_dates))+") that it will be used as the validation dataset (to be predicted)")
    for date in self.predict_dates:
      # extract pixels from image
      # check if is good image (with pixels)
      df_predict_ = self.extract_image_pixels(image=self.extract_image_from_collection(date=date, apply_attributes=False, convolve_force_disabled=True), date=date, apply_attributes=False, disable_threshold=True)
      if df_predict_.size > 0:
        df_predict = self.merge_timeseries(df_list=[df_predict, df_predict_])

      # get only good dates
      # fix dataframe index
      if not df_predict is None:
        df_predict['index'] = range(0,len(df_predict))

    # process all dates that will be used as reference
    print()
    print("Extracting images ("+str(len(self.predict_dates))+") that it will be used as the reference dataset (to be predicted)")
    try:
      for date in self.predict_dates:

        # warning
        print("Processing date ["+str(date.strftime("%Y-%m-%d"))+"]...")

        # initialize vars
        lons_lats_attributes  = np.array([]).reshape(0, 3)

        # extract pixels from image
        # check if is good image (with pixels)
        image = ee.Image(abs(self.dummy)).blend(ee.Image.load(path+"/"+date.strftime("%Y-%m-%d")+"_chla")).rename('chla')
        image = gee.apply_mask(image, self.water_mask, "chla",  "chla_water", False)

        # go through each geometry and get pixels
        for geometry in self.splitted_geometry:
          clip                            = self.clip_image(image=image, geometry=geometry, scale=self.scale)
          geometry_lons_lats_attributes   = gee.extract_latitude_longitude_pixel(image=clip, geometry=geometry, bands=['chla_water'], scale=self.scale)
          lons_lats_attributes            = np.vstack((lons_lats_attributes, geometry_lons_lats_attributes))

        # concatenate geometry pixel values
        extra_attributes     = np.array(list(zip(range(0,len(lons_lats_attributes)))))
        extra_attributes2    = np.array(list(zip([date]*len(lons_lats_attributes))))
        lons_lats_attributes = np.hstack((extra_attributes, extra_attributes2, lons_lats_attributes))

        # append new data from roi to dataframe
        df_reference_ = pd.DataFrame(data=lons_lats_attributes, columns=df_columns_reference).infer_objects()
        if df_reference_.size > 0:
          df_reference = df_reference.append(df_reference_, ignore_index = True)

      # remove empty pixels
      df_reference = df_reference[(df_reference["chla"]!=abs(self.dummy))]
        
      # fix label
      df_reference.loc[(df_reference['chla']<=-1), 'label'] = -1
      df_reference.loc[((df_reference['chla']>=0) & (df_reference['chla']<=chla_threshold)), 'label'] = 0
      df_reference.loc[(df_reference['chla']>chla_threshold), 'label'] = len(attributes_clear)

      # fix clouds
      df_reference.loc[(df_reference['chla']!=-1), 'cloud'] = abs(self.dummy)
      df_reference.loc[(df_reference['chla']==-1), 'cloud'] = 1.0

      # fix index
      if not df_reference is None:
        df_reference['index'] = range(0,len(df_reference))

      # fix null values
      df_reference.fillna(0, inplace=True)

    except Exception as e:
      print("Error while extracting image pixels: "+str(e))

    # remove dummies
    for attribute in attributes_clear:
      df_classification = df_classification[(df_classification[attribute]!=abs(self.dummy))]
      df_predict = df_predict[(df_predict[attribute]!=abs(self.dummy))]

    # change cloud values
    df_predict.loc[(df_predict['cloud'] == abs(self.dummy)) , 'cloud']                = 0.0
    df_classification.loc[(df_classification['cloud'] == abs(self.dummy)) , 'cloud']  = 0.0
    if len(df_reference) > 0:
      df_reference.loc[(df_reference['cloud'] == abs(self.dummy)) , 'cloud']            = 0.0

    # apply grids
    df_predict                  = self.apply_grids(df=df_predict)
    df_classification           = self.apply_grids(df=df_classification)
    if len(df_reference) > 0:
      df_reference = self.apply_grids(df=df_reference)

    # fix values in dataframe - predict
    df_predict                  = self.apply_label(self.fill_missing_dates(df=df_predict, min_date=min(self.predict_dates), max_date=max(self.predict_dates))).reset_index(drop=True)

    # fix values in dataframe - classification
    # concat it with timeseries dataframe before doing the concatenation and remove the old dates after that
    df_classification           = self.fill_missing_dates(df=pd.concat([self.df_timeseries,df_classification]), min_date=min(self.classification_dates), max_date=max(self.classification_dates), fill=self.fill_missing).reset_index(drop=True)
    df_classification           = df_classification[df_classification['date']>=min(self.classification_dates)]

    # normalize indexes
    df_classification           = self.normalize_indices(df=df_classification)
    df_predict                  = self.normalize_indices(df=df_predict)

    # shifiting dates em buling training set
    df_classification_proc      = misc.series_to_supervised(df_classification[df_classification['doy']==max(df_classification['doy'])][self.attributes_selected].values, self.days_in, self.days_out,dropnan=True)
    str_lat                     = 'var'+str(self.attributes_selected.index('lat')+1)+'(t-'+str(self.days_in)+')'
    str_lon                     = 'var'+str(self.attributes_selected.index('lon')+1)+'(t-'+str(self.days_in)+')'
    in_labels                   = [s for i, s in enumerate(df_classification_proc.columns) if 't-' in s]
    in_labels                   = [s for i, s in enumerate(in_labels) if (self.attribute_lat_lon or not 'var'+str(self.attributes_selected.index('lat')+1) in s) and (self.attribute_lat_lon or not 'var'+str(self.attributes_selected.index('lon')+1) in s) and (self.attribute_doy or not 'var'+str(self.attributes_selected.index('doy')+1) in s)]
    array_lats_lons             = df_classification_proc[[str_lat,str_lon]].values

    # splitting training and testing sets
    X                           = self.scaler.transform(df_classification_proc[in_labels].values.reshape((-1, len(in_labels))))
    if self.reducer:
      X = self.reducer.transform(X)

    # building array of final classification dataframe
    self.df_classification      = X

    # Final statistics
    print()
    print("Prediction datasets length: classification=(%s), predict=(%s, %s)" %(self.df_classification.shape, len(df_predict), len(df_predict.columns)))

    # fix clouds - final approach
    df_predict.loc[(df_predict['cloud']==self.dummy), 'cloud'] = 1.0
    df_classification.loc[(df_classification['cloud']==self.dummy), 'cloud'] = 1.0
    if len(df_reference) > 0:
      df_reference.loc[(df_reference['cloud']==self.dummy), 'cloud'] = 1.0

    # go through all dates to save geojson
    features = []
    for index, row in df_predict.iterrows():
      features.append(geojson.Feature(geometry=geojson.Point((float(row['lat']), float(row['lon']))), properties={"index": int(index), "date": row['date'].strftime('%Y-%m-%d'), "label": int(row['label']), "cloud": int(row['cloud'])}))
    fc = geojson.FeatureCollection(features)
    f = open(folder+"/geojson/validation.json","w")
    geojson.dump(fc, f)
    f.close()

    # go through all dates to save geojson
    if len(df_reference) > 0:
      features = []
      for index, row in df_reference.iterrows():
        features.append(geojson.Feature(geometry=geojson.Point((float(row['lat']), float(row['lon']))), properties={"index": int(index), "date": row['date'].strftime('%Y-%m-%d'), "chla": float(row['chla']), "label": int(row['label']), "cloud": int(row['cloud'])}))
      fc = geojson.FeatureCollection(features)
      f = open(folder+"/geojson/reference.json","w")
      geojson.dump(fc, f)
      f.close()

    # plot configuration
    image_empty_clip_io         = PIL.Image.open(BytesIO(requests.get(self.clip_image(ee.Image([99999,99999,99999])).select(['constant','constant_1','constant_2']).getThumbUrl({'min':0, 'max':99999}), timeout=60).content))
    markersize_scatter          = (72./300)*((self.scale/self.resolution[0])*4)
    xticks                      = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=self.plots_grid+1)
    yticks                      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=self.plots_grid+1)

    # colormap
    color_map                   = np.empty((5,1), dtype=object)
    color_map[0]                = "cyan"
    for i in range(1,len(attributes_clear)):
      color_map[i] = "black"
    color_map[len(attributes_clear)] = "magenta"

    # colorbar tixks
    colorbar_ticks_max          = 100
    colorbar_ticks              = np.linspace(0, colorbar_ticks_max if colorbar_ticks_max > 1 else 2, num=11, dtype=int)
    colorbar_ticks_labels       = [str(l) for l in colorbar_ticks]
    colorbar_ticks_labels[-1]   = str(colorbar_ticks_labels[-1])

    # legends
    legends_colors = [Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='cyan',fill=True),
                       Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='magenta',fill=True),
                       Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='gray',fill=True),
                       Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='black',fill=True)]

    # legends 2
    legends_colors2 = [Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='gray',fill=True),
                       Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='black',fill=True)]
    
    # legends captions
    legends_colors_captions = ['Regular', 'Anomaly', 'Land/Cloud/Shadow', 'Indetermined']

    # legends captions 2
    legends_colors_captions2 = ['Land/Cloud/Shadow', 'Indetermined']

    # clear scene results
    self.df_scene = pd.DataFrame({}, columns=self.df_columns_scene)

    # go through each classifier
    dict_results = []
    for model in self.classifiers:

      # predict and merging results with lats and lons
      start_time = time.time()

      # clear results by scene
      dict_results_scene = []

      # clear images results by grid
      images_grid = []

      # check if model is MLP
      if 'MLP' in model:
        y_pred = self.classifiers[model].predict(X)
        if self.class_mode:
          y_pred[y_pred > 0.5] = 1
          y_pred[y_pred <= 0.5] = 0
          y_pred = y_pred.reshape(y_pred.shape[0], self.randomizedsearch_categorical[1], self.randomizedsearch_categorical[2])
          y_pred = np.argmax(y_pred,axis=1)

      # OR LSTM
      elif 'LSTM' in model:
        if self.class_mode:
          y_pred = self.classifiers[model].predict(X.reshape(X.shape[0], 1, X.shape[1])).reshape(-1, self.days_in*1)
        else:
          y_pred = self.classifiers[model].predict(X.reshape(X.shape[0], 1, X.shape[1])).reshape(-1, self.days_in*len(attributes_clear))
      
      # regular model
      else:
        y_pred = self.classifiers[model].predict(X)

      # check if it is selected one day to output to avoid concatanate error
      if self.days_out == 1:
        df_data = np.hstack((array_lats_lons,y_pred[:, None]))
      else:
        df_data = np.concatenate((array_lats_lons, y_pred), axis=1)
      df = pd.DataFrame(data=df_data)
        
      # check if propagation model was selected
      if self.propagate and self.days_out > 1:
        for i in range(1,self.days_out):

          # check mode and fix column names
          features = len(attributes_clear)
          if self.class_mode:
            features = 1
            df_new = df.rename(columns={0:'lat', 1:'lon', 2:'label'})[['lat','lon','label']]
          else:
            df_new = df.rename(columns=dict_attributes)[['lat','lon']+attributes_clear]

          # initiate absence columns
          df_new['date'] = df_classification['date'].max() + td(days=1)
          df_new['doy'] = df_new['date'].dt.month

          # format prediction dataset
          df_classification = df_classification[df_classification['date']!=df_classification['date'].min()].append(df_new).fillna(method='ffill')
          df_classification = self.fill_missing_dates(df=df_classification, min_date=df_classification['date'].min(), max_date=df_classification['date'].max(), fill=self.fill_missing).reset_index(drop=True)
          df_classification_proc = misc.series_to_supervised(df_classification[df_classification['doy']==max(df_classification['doy'])][self.attributes_selected].values, self.days_in, self.days_out,dropnan=True)
          array_lats_lons = df_classification_proc[[str_lat,str_lon]].values

          # scale prediction dataset
          X = self.scaler.transform(df_classification_proc[in_labels].values.reshape((-1, len(in_labels))))
          if self.reducer:
            X = self.reducer.transform(X)

          # check if model is MLP
          if 'MLP' in model:
            y_pred = self.classifiers[model].predict(X)
            if self.class_mode:
              y_pred[y_pred > 0.5] = 1
              y_pred[y_pred <= 0.5] = 0
              y_pred = y_pred.reshape(y_pred.shape[0], self.randomizedsearch_categorical[1], self.randomizedsearch_categorical[2])
              y_pred = np.argmax(y_pred,axis=1)
              df = pd.DataFrame(data=np.concatenate((array_lats_lons, y_pred), axis=1))
            else:
              df = pd.DataFrame(data=np.concatenate((array_lats_lons, y_pred), axis=1))

          # check if model is LSTM to format dataset for it
          elif 'LSTM' in model:
            if self.class_mode:
              y_pred = self.classifiers[model].predict(X.reshape(X.shape[0], 1, X.shape[1])).reshape(-1, self.days_in*1)
            else:
              y_pred = self.classifiers[model].predict(X.reshape(X.shape[0], 1, X.shape[1])).reshape(-1, self.days_in*len(attributes_clear))
            df_new = pd.DataFrame(data=np.concatenate((array_lats_lons, y_pred), axis=1))

          # regular dataset format (rf and svm)
          else:
            df_new = pd.DataFrame(data=np.concatenate((array_lats_lons, self.classifiers[model].predict(X)), axis=1))

          # append new predictions to incorporate and propagate it with previous ones
          df_new.columns = df_new.columns.astype(int)
          df_new2 = pd.DataFrame(data=df_new[df_new.columns.values[2:-(features*i)]].values, columns=df_new.columns.values[2:-(features*i)].astype(int)+(features*i))
          df_new2[[0,1]] = df_new[[0,1]]
          for c in range(2,(features*i)+2):
              df_new2[c] = np.nan
          df = df.append(df_new2.sort_index(axis=1)).groupby([0,1]).median().reset_index().fillna(method='ffill')
          del df_new, df_new2

      # final fixes
      df = df.groupby([0,1]).median().reset_index().fillna(method='ffill')
      if self.class_mode:
        df[df.columns[2:]] = df[df.columns[2:]].astype(dtype=np.int64, errors='ignore')

      # save time
      self.classifiers_runtime[model] = self.classifiers_runtime[model] + (time.time() - start_time)

      # save predictions
      c_model = 0
      j = 0
      for i, date in enumerate(self.predict_dates):
        array = [0,1]
        if self.class_mode:
          array = array + [i+len(array)]
          df_pred = df[array].copy(deep=True).reset_index()
          df_pred.rename(columns={0:'lat', 1:'lon', i+2:'label'}, inplace=True)
        else:

          # array to select columns
          j = j + len(array) if j == 0 else j + (len(array)*2)
          array = array + list(range(j, j+len(attributes_clear)))

          # dictionary of attributes
          dict_attributes_ = {}
          dict_attributes_[0] = 'lat'
          dict_attributes_[1] = 'lon'
          for w, attribute in enumerate(attributes_clear):
            dict_attributes_[j+w] = attribute

          # create dataframe
          df_pred = df[array].copy(deep=True).reset_index()
          df_pred.rename(columns=dict_attributes_, inplace=True)
          df_pred = self.apply_label(df_pred).drop(attributes_clear,axis=1)
        df_pred.rename(columns={'label':'label_predicted_'+str(i)}, inplace=True)
        if not model in self.predictions:
          self.predictions[model] = df_pred
        else:
          self.predictions[model] = self.predictions[model].join(df_pred.drop(['lat','lon','index'],axis=1))
      c_model += 1

      # model str 2
      model_short = ''
      if 'MLP' in model:
        model_short = 'mlp'
      elif 'LSTM' in model:
        model_short = 'lstm'
      elif 'RF' in model:
        model_short = 'rf'
      elif 'SVM' in model:
        model_short = 'svm'

      # jump line
      print()
      print("Starting the prediction for "+str(model)+" model...")

      ##################################################################################
      # Pre-calculations and fixes

      # check if it has reference
      # get date pixels
      df_true                             = df_reference if (len(df_reference) > 0) else df_predict
      df_prediction                       = pd.DataFrame()

      # build prediction dataframe
      for i, date in enumerate(self.predict_dates):
        df_prediction_ = self.predictions[model][['lat','lon','label_predicted_'+str(i)]].copy().rename(columns={'label_predicted_'+str(i):'label_predicted'})
        df_prediction_['date'] = date
        df_prediction = df_prediction.append(df_prediction_)
      df_prediction.drop(['index'], inplace=True, axis=1, errors='ignore')
      df_prediction['index'] = range(0,len(df_prediction))
      df_prediction = self.apply_grids(df_prediction)
      df_prediction.reset_index(inplace=True, drop=True)

      # fix dataframes columns
      df_prediction['pixel']            = 0
      df_prediction['cloud']            = 0
      df_prediction['label']            = None
      df_prediction['class']            = 0
      df_prediction['color']            = "black"
      df_prediction['class_predicted']  = 0
      df_prediction['color_predicted']  = "black"
      
      # update prediction dataframe based on true columns
      for index, row in df_true.iterrows():
        df_prediction.loc[(df_prediction['date']==row['date']) & (df_prediction['lat']==row['lat']) & (df_prediction['lon']==row['lon']), 'pixel'] = row['pixel']
        df_prediction.loc[(df_prediction['date']==row['date']) & (df_prediction['lat']==row['lat']) & (df_prediction['lon']==row['lon']), 'cloud'] = row['cloud']
        df_prediction.loc[(df_prediction['date']==row['date']) & (df_prediction['lat']==row['lat']) & (df_prediction['lon']==row['lon']), 'label'] = row['label']

      # fix indexes
      df_prediction.loc[(df_prediction['label'].isnull()), 'label'] = df_prediction[df_prediction['label'].isnull()]['label_predicted']
      df_prediction.reset_index(inplace=True, drop=True)

      # calculation of anomalie occurrence based on pixel windows (without indetermined)
      for i, date in enumerate(self.predict_dates):

        # true and prediction from selected date
        df_prediction_ = df_prediction[(df_prediction['date']==date)]

        # windows
        grid_size_ = int((self.grid_size - 1) / 2)
        for i, row in df_prediction_.iterrows():

          # calculation of cols and rows to query de grid
          col_start   = row['column']-grid_size_
          col_end     = row['column']+grid_size_
          row_start   = row['row']-grid_size_
          row_end     = row['row']+grid_size_

          # get pixels
          df_grid = df_prediction_[((df_prediction_['column']>=col_start) & (df_prediction_['column']<=col_end)) & ((df_prediction_['row']>=row_start) & (df_prediction_['row']<=row_end))]
          if len(df_grid)>0:

            # true
            df_grid.loc[df_grid['label']<len(attributes_clear), 'label'] = 0
            df_grid.loc[df_grid['label']==len(attributes_clear), 'label'] = 1
            pct_occurrence = int((df_grid['label'].sum()/len(df_grid)) * 100)
            pct_occurrence = pct_occurrence if pct_occurrence > 0 else 0
            df_prediction.loc[(df_prediction['index'].isin(df_grid['index'].values)), 'class'] = pct_occurrence

            # prediction
            df_grid.loc[df_grid['label_predicted']<len(attributes_clear), 'label_predicted'] = 0
            df_grid.loc[df_grid['label_predicted']==len(attributes_clear), 'label_predicted'] = 1
            pct_occurrence = int((df_grid['label_predicted'].sum()/len(df_grid)) * 100)
            pct_occurrence = pct_occurrence if pct_occurrence > 0 else 0
            df_prediction.loc[(df_prediction['index'].isin(df_grid['index'].values)), 'class_predicted'] = pct_occurrence

      # apply color
      for index, color in enumerate(color_map):
        df_prediction.loc[(df_prediction['label'] == index), 'color']                       = color[0]
        df_prediction.loc[(df_prediction['label_predicted'] == index), 'color_predicted']   = color[0]

      # final fixes
      df_prediction = df_prediction.infer_objects()
      df_prediction.fillna(0, inplace=True)
    
      # Salvar GeoJSON - Anomaly Prediction
      for i, date in enumerate(self.predict_dates):
        df_prediction_ = df_prediction[(df_prediction['date']==date)]
        features = []
        if len(df_prediction_) > 0:
          for index, row in df_prediction_.iterrows():
            features.append(ee.Feature(ee.Geometry.Point(row['lat'],row['lon']), {"label": int(row['label']), "label_predicted": int(row['label_predicted']), "class": int(row['class']), "class_predicted": int(row['class_predicted'])}))
          fc = ee.FeatureCollection(features)
          f = open(folder+"/geojson/prediction_"+str(model_short)+"_"+str(date.strftime("%Y-%m-%d"))+".json","wb")
          f.write(requests.get(fc.getDownloadURL('GEO_JSON'), allow_redirects=True, timeout=60).content)
          f.close()

      # save dataframes
      df_prediction.to_csv(folder+'/csv/df_prediction_'+str(model_short)+'.csv')

      # save merge in dataframe
      if not model in self.merges:
        self.merges[model] = df_prediction
      else:
        self.merges[model] = self.merges[model].append(df_prediction, ignore_index=True)

      ##################################################################################
      # Plotting
      # plot types
      plot_types = ['pixel','grid','scene']
      for plot_type in plot_types:

        # create plot
        # scene
        if plot_type == 'scene':
          fig = plt.figure(figsize=(14,6), dpi=300)
          plt.tight_layout(pad=10.0)
          plt.rc('xtick',labelsize=10)
          plt.rc('ytick',labelsize=10)
          plt.box(False)

        #others
        else:
          plot_count = 1
          fig = plt.figure(figsize=(14,7), dpi=300)
          plt.tight_layout(pad=10.0)
          plt.rc('xtick',labelsize=3)
          plt.rc('ytick',labelsize=3)
          plt.box(False)
          plt.axis('off')

        # Title
        if plot_type == "pixel":
          plt.title(self.name+": pixel-wise from "+self.predict_dates[0].strftime("%Y-%m-%d")+" to "+self.predict_dates[-1].strftime("%Y-%m-%d")+"\n"+model, fontdict = {'fontsize' : 8}, pad=30)
        elif plot_type == "grid":
          plt.title(self.name+": grid-wise ("+str(self.grid_size)+"x"+str(self.grid_size)+") from "+self.predict_dates[0].strftime("%Y-%m-%d")+" to "+self.predict_dates[-1].strftime("%Y-%m-%d")+"\n"+model, fontdict = {'fontsize' : 8}, pad=30)
        elif plot_type == "scene":
          plt.title(self.name+": scene-wise occurrences from "+self.predict_dates[0].strftime("%Y-%m-%d")+" to "+self.predict_dates[-1].strftime("%Y-%m-%d")+"\n"+model, fontdict = {'fontsize' : 8}, pad=30)

        ##################################################################################
        # RGB and Validation/Reference
        # go through all dates to get rgb images
        for i, date in enumerate(self.predict_dates):
          
          # extract image
          image             = self.extract_image_from_collection(date=date, apply_attributes=False)
          if image:
            image_clip      = self.clip_image(image)
            image_clip_rgb  = image_clip.select(self.sensor_params['red'], self.sensor_params['green'], self.sensor_params['blue']).getThumbUrl({'min':0, 'max':3000})
            image_clip_io   = PIL.Image.open(BytesIO(requests.get(image_clip_rgb, timeout=60).content))
          else:
            image_clip_io   = image_empty_clip_io

          # plot RBG image
          if plot_type != "scene":
            c = fig.add_subplot(3,len(self.predict_dates),plot_count)
            c.set_title(date.strftime("%Y-%m-%d"), fontdict = {'fontsize' : 4.5})
            c.set_xticks(xticks)
            c.set_yticks(yticks)
            c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.imshow(image_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
            c.margins(x=0,y=0)
            plot_count += 1

        # go through all dates to get rgb images
        # pixel
        if plot_type == 'pixel':
          for i, date in enumerate(self.predict_dates):

            # true
            df_prediction_ = df_prediction[(df_prediction['date']==date)]

            # plot validation results
            c = fig.add_subplot(3,len(self.predict_dates),plot_count)
            c.set_title("Validation/Reference", fontdict = {'fontsize' : 4.5})
            c.set_xticks(xticks)
            c.set_yticks(yticks)
            c.grid(color='b', linestyle='dashed', linewidth=0.1)
            c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
            if len(df_prediction_) > 0:
              c.scatter(df_prediction_[(df_prediction_['cloud']==0)]['lat'], df_prediction_[(df_prediction_['cloud']==0)]['lon'], marker='s', s=markersize_scatter, c=df_prediction_[(df_prediction_['cloud']==0)]['color'].values, edgecolors='none')
              c.scatter(df_prediction_[(df_prediction_['cloud']==1)]['lat'], df_prediction_[(df_prediction_['cloud']==1)]['lon'], marker='s', s=markersize_scatter, c="gray", edgecolors='none')
            c.margins(x=0,y=0)
            plot_count += 1

        # grid
        elif plot_type == 'grid':
          for i, date in enumerate(self.predict_dates):

            # true
            df_prediction_ = df_prediction[(df_prediction['date']==date)]

            # plot
            c = fig.add_subplot(3,len(self.predict_dates),plot_count)
            c.set_title("Validation/Reference ("+str(self.grid_size)+"x"+str(self.grid_size)+")", fontdict = {'fontsize' : 4.5})
            c.set_xticks(xticks)
            c.set_yticks(yticks)
            c.grid(color='b', linestyle='dashed', linewidth=0.1)
            c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
            if len(df_prediction_) > 0:
              s = c.scatter(df_prediction_['lat'], df_prediction_['lon'], marker='s', s=markersize_scatter, c=df_prediction_['class'], cmap=plt.get_cmap('jet'), edgecolors='none')
              s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
              c.scatter(df_prediction_[(df_prediction_['cloud']==1)]['lat'], df_prediction_[(df_prediction_['cloud']==1)]['lon'], marker='s', s=markersize_scatter, c="gray", edgecolors='none')
              c.scatter(df_prediction_[(df_prediction_['cloud']==0) & (df_prediction_["label"]==-1)]['lat'], df_prediction_[(df_prediction_['cloud']==0) & (df_prediction_["label"]==-1)]['lon'], marker='s', s=markersize_scatter, c="black", edgecolors='none')
            c.margins(x=0,y=0)
            plot_count += 1

        ##################################################################################
        # Prediction
        # for through each date to evaluate the predictions
        for i, date in enumerate(self.predict_dates):

          # predictions from date
          df_prediction_ = df_prediction[(df_prediction['date']==date)]
          
          # pixel plot
          if plot_type == 'pixel':

            # warning
            print()
            print("[Pixel] Evaluating prediction for date "+str(date.strftime("%Y-%m-%d"))+"...")

            # labels arrays
            df_merge_ = df_prediction_[(df_prediction_['cloud']==0) & ((df_prediction_['label']==0) | (df_prediction_['label']==len(attributes_clear)))]
            if len(df_merge_) > 0:
              y_pred   = df_merge_['label_predicted'].values.reshape((-1, 1))
              y_true   = df_merge_['label'].values.reshape((-1, 1))
            else:
              y_pred = [0]
              y_true = [len(attributes_clear)]

            # report
            measures = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)
            print(measures['string'])

            # add results - pixel
            dict_results.append({
              'model':            str(model),
              'type':             plot_type,
              'sensor':           str(self.sensor),
              'path':             str(self.path),
              'date_predicted':   date.strftime("%Y-%m-%d"),
              'date_execution':   dt.now().strftime("%Y-%m-%d"),
              'time_execution':   dt.now().strftime("%H:%M:%S"),
              'runtime':          str(self.classifiers_runtime[model]),
              'days_threshold':   str(self.days_threshold), 
              'grid_size':        str(self.grid_size),
              'size_train':       str("D="+str(self.df_train[0].shape)+"Dy="+str(self.df_train[0].shape)+";R="+str(self.df_randomizedsearch[0].shape)+";Ry="+str(self.df_randomizedsearch[1].shape)),
              'size_dates':       str(len(self.dates_timeseries_interval)),
              'scaler':           str(self.scaler_str),
              'morph_op':         str(self.morph_op),
              'morph_op_iters':   str(self.morph_op_iters),
              'convolve':         str(self.convolve),
              'convolve_radius':  str(self.convolve_radius),
              'days_in':          str(self.days_in), 
              'days_out':         str(self.days_out), 
              'fill_missing':     str(self.fill_missing), 
              'remove_dummies':   str(self.remove_dummies), 
              'shuffle':          str(self.shuffle), 
              'reducer':          str(self.reducer), 
              'normalized':       str(self.normalized), 
              'class_mode':       str(self.class_mode), 
              'class_weight':     str(self.class_weight), 
              'propagate':        str(self.propagate), 
              'rs_train_size':    str(self.rs_train_size), 
              'rs_iter':          str(self.rs_iter), 
              'pca_size':         str(self.pca_size),
              'attribute_lat_lon':str(self.attribute_lat_lon),
              'attribute_doy':    str(self.attribute_doy),
              'acc':              float(measures["acc"]),
              'bacc':             float(measures["bacc"]),
              'kappa':            float(measures["kappa"]),
              'vkappa':           float(measures["vkappa"]),
              'tau':              float(measures["tau"]),
              'vtau':             float(measures["vtau"]),
              'mcc':              float(measures["mcc"]),
              'f1score':          float(measures["f1score"]),
              'rmse':             float(measures["rmse"]),
              'mae':              float(measures["mae"]),
              'r2score':          float(measures["r2score"]),
              'tp':               int(measures["tp"]),
              'tn':               int(measures["tn"]),
              'fp':               int(measures["fp"]),
              'fn':               int(measures["fn"])
            })

            # plot title
            title_plot = "Pred. (Acc:"+str(round(measures["acc"],2))+",Kpp:"+str(round(measures["kappa"],2))+",MCC:"+str(round(measures["mcc"],2))+",F1:"+str(round(measures["f1score"],2))+")"

            # plot predicted results
            c = fig.add_subplot(3,len(self.predict_dates),plot_count)
            c.set_title(title_plot, fontdict = {'fontsize' : 4.5})
            c.set_xticks(xticks)
            c.set_yticks(yticks)
            c.grid(color='b', linestyle='dashed', linewidth=0.1)
            c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
            if len(df_prediction_) > 0:
              c.scatter(df_prediction_['lat'], df_prediction_['lon'], marker='s', s=markersize_scatter, c=df_prediction_['color_predicted'].values, edgecolors='none')
            if i == len(self.predict_dates)-1:
              c.legend(legends_colors, legends_colors_captions, loc='upper center', bbox_to_anchor=(-0.11, -0.24), ncol=4, fontsize='x-small', fancybox=True, shadow=True)
            c.margins(x=0,y=0)
            plot_count += 1

          # grid
          elif plot_type == 'grid':

            # warning
            print()
            print("[Grid] Evaluating prediction for date "+str(date.strftime("%Y-%m-%d"))+"...")

            # measures (without the indetermined)
            df_merge_ = df_prediction_[(df_prediction_['cloud']==0) & ((df_prediction_['label']==0) | (df_prediction_['label']==len(attributes_clear)))]
            if len(df_merge_) > 0:
              y_pred = df_merge_['class'].values
              y_true = df_merge_['class_predicted'].values
            else:
              y_pred = [0]
              y_true = [100]
              
            # report
            measures = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)
            print(measures['string'])

            # add results - pixel
            dict_results.append({
              'model':            str(model),
              'type':             plot_type,
              'sensor':           str(self.sensor),
              'path':             str(self.path),
              'date_predicted':   date.strftime("%Y-%m-%d"),
              'date_execution':   dt.now().strftime("%Y-%m-%d"),
              'time_execution':   dt.now().strftime("%H:%M:%S"),
              'runtime':          str(self.classifiers_runtime[model]),
              'days_threshold':   str(self.days_threshold), 
              'grid_size':        str(self.grid_size),
              'size_train':       str("D="+str(self.df_train[0].shape)+"Dy="+str(self.df_train[0].shape)+";R="+str(self.df_randomizedsearch[0].shape)+";Ry="+str(self.df_randomizedsearch[1].shape)),
              'size_dates':       str(len(self.dates_timeseries_interval)),
              'scaler':           str(self.scaler_str),
              'morph_op':         str(self.morph_op),
              'morph_op_iters':   str(self.morph_op_iters),
              'convolve':         str(self.convolve),
              'convolve_radius':  str(self.convolve_radius),
              'days_in':          str(self.days_in), 
              'days_out':         str(self.days_out), 
              'fill_missing':     str(self.fill_missing), 
              'remove_dummies':   str(self.remove_dummies), 
              'shuffle':          str(self.shuffle), 
              'reducer':          str(self.reducer), 
              'normalized':       str(self.normalized), 
              'class_mode':       str(self.class_mode), 
              'class_weight':     str(self.class_weight), 
              'propagate':        str(self.propagate), 
              'rs_train_size':    str(self.rs_train_size), 
              'rs_iter':          str(self.rs_iter), 
              'pca_size':         str(self.pca_size),
              'attribute_lat_lon':str(self.attribute_lat_lon),
              'attribute_doy':    str(self.attribute_doy),
              'acc':              float(measures["acc"]),
              'bacc':             float(measures["bacc"]),
              'kappa':            float(measures["kappa"]),
              'vkappa':           float(measures["vkappa"]),
              'tau':              float(measures["tau"]),
              'vtau':             float(measures["vtau"]),
              'mcc':              float(measures["mcc"]),
              'f1score':          float(measures["f1score"]),
              'rmse':             float(measures["rmse"]),
              'mae':              float(measures["mae"]),
              'r2score':          float(measures["r2score"]),
              'tp':               int(measures["tp"]),
              'tn':               int(measures["tn"]),
              'fp':               int(measures["fp"]),
              'fn':               int(measures["fn"])
            })

            # plot title
            title_plot = "Pred. (RMSE:"+str(round(measures["rmse"],2))+",MAE:"+str(round(measures["mae"],2))+",R^2:"+str(round(measures["r2score"],2))+")"

            # plot
            c = fig.add_subplot(3,len(self.predict_dates),plot_count)
            c.set_title(title_plot, fontdict = {'fontsize' : 4.5})
            c.set_xticks(xticks)
            c.set_yticks(yticks)
            c.grid(color='b', linestyle='dashed', linewidth=0.1)
            c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
            if len(df_prediction_) > 0:
              s = c.scatter(df_prediction_['lat'], df_prediction_['lon'], marker='s', s=markersize_scatter, c=df_prediction_['class_predicted'], cmap=plt.get_cmap('jet'), edgecolors='none')
              s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
              # add colorbar last image of grid-wise results
              images_grid.append(s)
            c.margins(x=0,y=0)
            plot_count += 1

          # scene
          elif plot_type == 'scene':

            # warning
            print()
            print("[Scene] Evaluating prediction for date "+str(date.strftime("%Y-%m-%d"))+"...")

            # labels arrays
            pct_true  = 100
            pct_pred  = 0
            y_true    = [pct_true]
            y_pred    = [pct_pred]

            # fix label value
            df_prediction_.loc[(df_prediction_["label"]<len(attributes_clear)), 'label'] = 0
            df_prediction_.loc[(df_prediction_["label"]==len(attributes_clear)), 'label'] = 1
            df_prediction_.loc[(df_prediction_["label_predicted"]<len(attributes_clear)), 'label_predicted'] = 0
            df_prediction_.loc[(df_prediction_["label_predicted"]==len(attributes_clear)), 'label_predicted'] = 1

            # has pixels
            if len(df_prediction_)>0:

              # true
              pct_true  = int((df_prediction_['label'].sum()/len(df_prediction_)) * 100)
              pct_true  = 0 if pct_true < 0 else pct_true
              y_true    = [pct_true]

              # pred
              pct_pred  = int((df_prediction_['label_predicted'].sum()/len(df_prediction_)) * 100)
              pct_pred  = 0 if pct_pred < 0 else pct_pred
              y_pred    = [pct_pred]


            # report
            measures = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)
            print(measures['string'])

            # add results do dataframe that will be used to create final graphic
            dict_results_scene.append({
              'date': date.strftime("%Y-%m-%d"),
              'model': str(model_short),
              'validation': pct_true,
              'prediction': pct_pred,
              'difference': pct_pred-pct_true
            })

            # add results - scene
            dict_results.append({
              'model':            str(model),
              'type':             plot_type,
              'sensor':           str(self.sensor),
              'path':             str(self.path),
              'date_predicted':   date.strftime("%Y-%m-%d"),
              'date_execution':   dt.now().strftime("%Y-%m-%d"),
              'time_execution':   dt.now().strftime("%H:%M:%S"),
              'runtime':          str(self.classifiers_runtime[model]),
              'days_threshold':   str(self.days_threshold), 
              'grid_size':        str(self.grid_size),
              'size_train':       str("D="+str(self.df_train[0].shape)+"Dy="+str(self.df_train[0].shape)+";R="+str(self.df_randomizedsearch[0].shape)+";Ry="+str(self.df_randomizedsearch[1].shape)),
              'size_dates':       str(len(self.dates_timeseries_interval)),
              'scaler':           str(self.scaler_str),
              'morph_op':         str(self.morph_op),
              'morph_op_iters':   str(self.morph_op_iters),
              'convolve':         str(self.convolve),
              'convolve_radius':  str(self.convolve_radius),
              'days_in':          str(self.days_in), 
              'days_out':         str(self.days_out), 
              'fill_missing':     str(self.fill_missing), 
              'remove_dummies':   str(self.remove_dummies), 
              'shuffle':          str(self.shuffle), 
              'reducer':          str(self.reducer), 
              'normalized':       str(self.normalized), 
              'class_mode':       str(self.class_mode), 
              'class_weight':     str(self.class_weight), 
              'propagate':        str(self.propagate), 
              'rs_train_size':    str(self.rs_train_size), 
              'rs_iter':          str(self.rs_iter), 
              'pca_size':         str(self.pca_size),
              'attribute_lat_lon':str(self.attribute_lat_lon),
              'attribute_doy':    str(self.attribute_doy),
              'acc':              float(measures["acc"]),
              'bacc':             float(measures["bacc"]),
              'kappa':            float(measures["kappa"]),
              'vkappa':           float(measures["vkappa"]),
              'tau':              float(measures["tau"]),
              'vtau':             float(measures["vtau"]),
              'mcc':              float(measures["mcc"]),
              'f1score':          float(measures["f1score"]),
              'rmse':             float(measures["rmse"]),
              'mae':              float(measures["mae"]),
              'r2score':          float(measures["r2score"]),
              'tp':               int(measures["tp"]),
              'tn':               int(measures["tn"]),
              'fp':               int(measures["fp"]),
              'fn':               int(measures["fn"])
            })

        # save plot
        #scene
        if plot_type == "scene":
          
          # create scene dataframe
          self.df_scene = self.df_scene.append(dict_results_scene)

          # creat plots
          ax_ = fig.add_subplot(1,1,1)
          ax  = self.df_scene[['validation','prediction']].plot(kind='bar', width=.5, ax=ax_)
          ax2 = self.df_scene['difference'].plot(secondary_y=True, color='red', ax=ax_)

          # line at 0
          plt.plot([-10,100],[0,0],"--",color="black", lw=0.5)

          # adjust ticks
          ax.set_xticklabels((self.df_scene['date'].values))
          ax.set_xlim(-1,len(self.df_scene['date'].values))
          ax.set_ylim(self.df_scene['difference'].min(),self.df_scene['validation'].max()+1 if self.df_scene['validation'].max() > self.df_scene['prediction'].max() else self.df_scene['prediction'].max()+1)
          ax2.set_ylim(ax.get_ylim())
          ax2.set_yticks([])

          # labels
          ax.set_xlabel('Predicted dates')
          ax.set_ylabel('% of occurrence')
          
          # fix legend
          handles, labels = ax.get_legend_handles_labels()
          handles.append(Line2D([0], [0], color="red", lw=0.5))
          ax.get_legend().remove()
          plt.legend(handles=handles, labels=["Validation/Reference","Prediction","Difference"], loc='upper right', fancybox=True, shadow=True)

          # save plot
          if len(self.predict_dates) > 1:
            fig.savefig(folder+'/image/results_'+str(plot_type)+'_'+str(model_short)+'.png', bbox_inches='tight')

        # others
        else:

          # grid colobar and legend - [left, bottom, width, height]
          if plot_type == "grid":
            
            # colobar
            cbar = fig.colorbar(images_grid[-1], cax=fig.add_axes([0.601, 0.055, 0.25, 0.01]), ticks=colorbar_ticks, orientation='horizontal')
            cbar.set_label("% of occurrence", fontdict = {'fontsize' : 4.5})
            cbar.ax.tick_params(labelsize=4.5)

            # legend
            plt.legend(legends_colors2, legends_colors_captions2, loc='upper center', bbox_to_anchor=(-1.30, 0.4), ncol=4, fontsize='x-small', fancybox=True, shadow=True)

          # other
          if len(self.predict_dates) > 1:
            fig.savefig(folder+'/image/results_'+str(plot_type)+'_'+str(model_short)+'.png')

      # clear
      del df

    # save results
    self.df_results = pd.DataFrame(dict_results, columns=self.df_columns_results)

    # warning
    print('finished!')


  # calculate prediction reduction performance (median)
  def predict_reduction(self, folder: str, reduction: str = "median"):

    # jump line
    print()
    print("Starting the prediction reduction performance calculation (median)...")

    # attributes
    attributes_clear  = [a for a in self.attributes_clear if a != 'cloud']
        
    # plot configuration
    image_empty_clip_io         = PIL.Image.open(BytesIO(requests.get(self.clip_image(ee.Image([99999,99999,99999])).select(['constant','constant_1','constant_2']).getThumbUrl({'min':0, 'max':99999}), timeout=60).content))
    markersize_scatter          = (72./300)*((self.scale/self.resolution[0])*8.5)
    xticks                      = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=self.plots_grid+1)
    yticks                      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=self.plots_grid+1)

    # colormap
    color_map                   = np.empty((5,1), dtype=object)
    color_map[0]                = "cyan"
    color_map[1]                = "magenta"

    # colorbar tixks
    colorbar_ticks_max          = 100
    colorbar_ticks              = np.linspace(0, colorbar_ticks_max if colorbar_ticks_max > 1 else 2, num=11, dtype=int)
    colorbar_ticks_labels       = [str(l) for l in colorbar_ticks]
    colorbar_ticks_labels[-1]   = str(colorbar_ticks_labels[-1])

    # legends
    legends_colors = [Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='cyan',fill=True),
                      Rectangle((0, 0),1,1,linewidth=0,edgecolor=None,facecolor='magenta',fill=True)]

    # legends captions
    legends_colors_captions = ['Regular', 'Anomaly']

    # start an empty result dictionary
    dict_results = []

    # check if there are one or predictions
    if not self.merges is None and len(self.merges)>0:

      # go over each ML model
      for model in self.merges:

        # model str 2
        model_short = ''
        if 'MLP' in model:
          model_short = 'mlp'
        elif 'LSTM' in model:
          model_short = 'lstm'
        elif 'RF' in model:
          model_short = 'rf'
        elif 'SVM' in model:
          model_short = 'svm'

        # fix merges
        self.merges[model]  = self.merges[model].apply(pd.to_numeric, errors='ignore')
        self.df_scene       = self.df_scene.apply(pd.to_numeric, errors='ignore')

        # reductions without indetermined
        df_reduction = self.merges[model][self.merges[model]['label']!=-1]
        df_reduction.loc[(df_reduction['label']<len(attributes_clear)), 'label'] = 0
        df_reduction.loc[(df_reduction['label']==len(attributes_clear)), 'label'] = 1
        df_reduction.loc[(df_reduction['label_predicted']<len(attributes_clear)), 'label_predicted'] = 0
        df_reduction.loc[(df_reduction['label_predicted']==len(attributes_clear)), 'label_predicted'] = 1

        # apply reduction
        if reduction == 'min':
          df_reduction = df_reduction.groupby(['lat','lon']).min().reset_index()
        else:
          df_reduction = df_reduction.groupby(['lat','lon']).median().reset_index()
        df_scene_median = self.df_scene[self.df_scene['model']==model_short].groupby(['model']).median().reset_index()

        # adjusts for reduction calculation
        df_reduction.loc[(df_reduction['label'] > 0.5), 'label']                      = 1
        df_reduction.loc[(df_reduction['label'] <= 0.5), 'label']                     = 0
        df_reduction.loc[(df_reduction['label_predicted'] > 0.5), 'label_predicted']  = 1
        df_reduction.loc[(df_reduction['label_predicted'] <= 0.5), 'label_predicted'] = 0

        # apply colors
        df_reduction['color']            = 'black'
        df_reduction['color_predicted']  = 'black'
        for index, color in enumerate(color_map):
          df_reduction.loc[(df_reduction['label'] == index), 'color']                        = color[0]
          df_reduction.loc[(df_reduction['label_predicted'] == index), 'color_predicted']    = color[0]

        # create figure
        fig = plt.figure(figsize=(10,14), dpi=300)
        plt.tight_layout(pad=10.0)
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        plt.box(False)
        plt.axis('off')

        # GridSpec
        gs = GridSpec(3, 2, figure=fig)

        # title
        if len(self.predict_dates) > 1:
          plt.title(self.name+": pixel/grid/scene-wise reductions (median) from "+self.predict_dates[0].strftime("%Y-%m-%d")+" to "+self.predict_dates[-1].strftime("%Y-%m-%d")+"\n"+model, fontdict = {'fontsize' : 10}, pad=30)
        else:
          plt.title(self.name+": pixel/grid/scene-wise reductions (median) from "+self.predict_dates[0].strftime("%Y-%m-%d")+"\n"+model, fontdict = {'fontsize' : 10}, pad=30)

        # Pixel-wise (Validation)
        c = fig.add_subplot(gs[0, 0])
        c.set_title("Validation/Reference ("+str(reduction)+")", fontdict = {'fontsize' : 8})
        c.set_xticks(xticks)
        c.set_yticks(yticks)
        c.grid(color='b', linestyle='dashed', linewidth=0.1)
        c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
        c.scatter(df_reduction['lat'], df_reduction['lon'], marker='s', s=markersize_scatter, c=df_reduction['color'].values, edgecolors='none')
        c.margins(x=0,y=0)
        c.legend(legends_colors, legends_colors_captions, loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=4, fontsize='xx-small', fancybox=True, shadow=True)

        # Grise-wise (Validation)
        c = fig.add_subplot(gs[0, 1])
        c.set_title("Validation/Reference ("+str(reduction)+" - "+str(self.grid_size)+"x"+str(self.grid_size)+")", fontdict = {'fontsize' : 8})
        c.set_xticks(xticks)
        c.set_yticks(yticks)
        c.grid(color='b', linestyle='dashed', linewidth=0.1)
        c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
        s = c.scatter(df_reduction['lat'], df_reduction['lon'], marker='s', s=markersize_scatter, c=df_reduction['class'], cmap=plt.get_cmap('jet'), edgecolors='none')
        s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
        cbar = fig.colorbar(s, cax=fig.add_axes([0.635, 0.64, 0.18, 0.01]), ticks=colorbar_ticks, orientation='horizontal')
        cbar.ax.tick_params(labelsize=5)
        c.margins(x=0,y=0)
            
        # Pixel-wise (Prediction)
        # warning
        print()
        print("[Pixel-"+str(reduction)+"] Evaluating reduction...")

        # labels arrays
        if len(df_reduction) > 0:
          y_pred = df_reduction['label_predicted'].values.reshape((-1, 1))
          y_true = df_reduction['label'].values.reshape((-1, 1))
        else:
          y_pred = [0]
          y_true = [1]

        # report
        measures = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)
        print(measures['string'])

        # add results - pixel
        dict_results.append({
          'model':            str(model),
          'type':             "pixel-"+str(reduction),
          'sensor':           str(self.sensor),
          'path':             str(self.path),
          'date_predicted':   self.predict_dates[0].strftime("%Y-%m-%d"),
          'date_execution':   dt.now().strftime("%Y-%m-%d"),
          'time_execution':   dt.now().strftime("%H:%M:%S"),
          'runtime':          str(self.classifiers_runtime[model]),
          'days_threshold':   str(self.days_threshold), 
          'grid_size':        str(self.grid_size),
          'size_train':       str("D="+str(self.df_train[0].shape)+"Dy="+str(self.df_train[0].shape)+";R="+str(self.df_randomizedsearch[0].shape)+";Ry="+str(self.df_randomizedsearch[1].shape)),
          'size_dates':       str(len(self.dates_timeseries_interval)),
          'scaler':           str(self.scaler_str),
          'morph_op':         str(self.morph_op),
          'morph_op_iters':   str(self.morph_op_iters),
          'convolve':         str(self.convolve),
          'convolve_radius':  str(self.convolve_radius),
          'days_in':          str(self.days_in), 
          'days_out':         str(self.days_out), 
          'fill_missing':     str(self.fill_missing), 
          'remove_dummies':   str(self.remove_dummies), 
          'shuffle':          str(self.shuffle), 
          'reducer':          str(self.reducer), 
          'normalized':       str(self.normalized), 
          'class_mode':       str(self.class_mode), 
          'class_weight':     str(self.class_weight), 
          'propagate':        str(self.propagate), 
          'rs_train_size':    str(self.rs_train_size), 
          'rs_iter':          str(self.rs_iter), 
          'pca_size':         str(self.pca_size),
          'attribute_lat_lon':str(self.attribute_lat_lon),
          'attribute_doy':    str(self.attribute_doy),
          'acc':              float(measures["acc"]),
          'bacc':             float(measures["bacc"]),
          'kappa':            float(measures["kappa"]),
          'vkappa':           float(measures["vkappa"]),
          'tau':              float(measures["tau"]),
          'vtau':             float(measures["vtau"]),
          'mcc':              float(measures["mcc"]),
          'f1score':          float(measures["f1score"]),
          'rmse':             float(measures["rmse"]),
          'mae':              float(measures["mae"]),
          'r2score':          float(measures["r2score"]),
          'tp':               int(measures["tp"]),
          'tn':               int(measures["tn"]),
          'fp':               int(measures["fp"]),
          'fn':               int(measures["fn"])
        })

        # plot
        c = fig.add_subplot(gs[1, 0])
        c.set_title("Prediction ("+str(reduction)+";Acc:"+str(round(measures["acc"],2))+",Kpp:"+str(round(measures["kappa"],2))+",MCC:"+str(round(measures["mcc"],2))+",F1:"+str(round(measures["f1score"],2))+")", fontdict = {'fontsize' : 8})
        c.set_xticks(xticks)
        c.set_yticks(yticks)
        c.grid(color='b', linestyle='dashed', linewidth=0.1)
        c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
        c.scatter(df_reduction['lat'], df_reduction['lon'], marker='s', s=markersize_scatter, c=df_reduction['color_predicted'].values, edgecolors='none')
        c.margins(x=0,y=0)
              
        # Grid-wise (Prediction)
        # measures
        # warning
        print()
        print("[Grid-"+str(reduction)+"] Evaluating reduction...")

        # measures
        if len(df_reduction) > 0:
          y_pred = df_reduction['class'].astype(int).values
          y_true = df_reduction['class_predicted'].astype(int).values
        else:
          y_pred = [0]
          y_true = [100]
          
        # report
        measures = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)
        print(measures['string'])

        # add results - grid
        dict_results.append({
          'model':            str(model),
          'type':             "grid-"+str(reduction),
          'sensor':           str(self.sensor),
          'path':             str(self.path),
          'date_predicted':   self.predict_dates[0].strftime("%Y-%m-%d"),
          'date_execution':   dt.now().strftime("%Y-%m-%d"),
          'time_execution':   dt.now().strftime("%H:%M:%S"),
          'runtime':          str(self.classifiers_runtime[model]),
          'days_threshold':   str(self.days_threshold), 
          'grid_size':        str(self.grid_size),
          'size_train':       str("D="+str(self.df_train[0].shape)+"Dy="+str(self.df_train[0].shape)+";R="+str(self.df_randomizedsearch[0].shape)+";Ry="+str(self.df_randomizedsearch[1].shape)),
          'size_dates':       str(len(self.dates_timeseries_interval)),
          'scaler':           str(self.scaler_str),
          'morph_op':         str(self.morph_op),
          'morph_op_iters':   str(self.morph_op_iters),
          'convolve':         str(self.convolve),
          'convolve_radius':  str(self.convolve_radius),
          'days_in':          str(self.days_in), 
          'days_out':         str(self.days_out), 
          'fill_missing':     str(self.fill_missing), 
          'remove_dummies':   str(self.remove_dummies), 
          'shuffle':          str(self.shuffle), 
          'reducer':          str(self.reducer), 
          'normalized':       str(self.normalized), 
          'class_mode':       str(self.class_mode), 
          'class_weight':     str(self.class_weight), 
          'propagate':        str(self.propagate), 
          'rs_train_size':    str(self.rs_train_size), 
          'rs_iter':          str(self.rs_iter), 
          'pca_size':         str(self.pca_size),
          'attribute_lat_lon':str(self.attribute_lat_lon),
          'attribute_doy':    str(self.attribute_doy),
          'acc':              float(measures["acc"]),
          'bacc':             float(measures["bacc"]),
          'kappa':            float(measures["kappa"]),
          'vkappa':           float(measures["vkappa"]),
          'tau':              float(measures["tau"]),
          'vtau':             float(measures["vtau"]),
          'mcc':              float(measures["mcc"]),
          'f1score':          float(measures["f1score"]),
          'rmse':             float(measures["rmse"]),
          'mae':              float(measures["mae"]),
          'r2score':          float(measures["r2score"]),
          'tp':               int(measures["tp"]),
          'tn':               int(measures["tn"]),
          'fp':               int(measures["fp"]),
          'fn':               int(measures["fn"])
        })

        # plot
        c = fig.add_subplot(gs[1, 1])
        c.set_title("Prediction ("+str(reduction)+";RMSE:"+str(round(measures["rmse"],2))+",MAE:"+str(round(measures["mae"],2))+",R^2:"+str(round(measures["r2score"],2))+")", fontdict = {'fontsize' : 8})
        c.set_xticks(xticks)
        c.set_yticks(yticks)
        c.grid(color='b', linestyle='dashed', linewidth=0.1)
        c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.imshow(image_empty_clip_io, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
        s = c.scatter(df_reduction['lat'], df_reduction['lon'], marker='s', s=markersize_scatter, c=df_reduction['class_predicted'], cmap=plt.get_cmap('jet'), edgecolors='none')
        s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
        c.margins(x=0,y=0)

        # median geojson
        features = []
        if len(df_reduction) > 0:
          for index, row in df_reduction.iterrows():
            features.append(ee.Feature(ee.Geometry.Point(row['lat'],row['lon']), {"label": int(row['label']), "label_predicted": int(row['label_predicted']), "class": int(row['class']), "class_predicted": int(row['class_predicted'])}))
          fc = ee.FeatureCollection(features)
          f = open(folder+"/geojson/prediction_"+str(model_short)+"_median.json","wb")
          f.write(requests.get(fc.getDownloadURL('GEO_JSON'), allow_redirects=True, timeout=60).content)
          f.close()

        # save dataframe
        df_reduction.to_csv(folder+'/csv/df_prediction_'+str(model_short)+'_median.csv')
            
        # Scene-wise (Validation X Prediction)
        # measures
        # warning
        print()
        print("[Scene-median] Evaluating reduction...")

        # measures
        if len(df_scene_median)>0:
          y_true      = [df_scene_median['validation'].astype(int)]
          y_pred      = [df_scene_median['prediction'].astype(int)]
          difference  = df_scene_median['prediction'].astype(int)-df_scene_median['validation'].astype(int)
        else:
          y_true      = [0]
          y_pred      = [100]
          difference  = 0

        # report
        measures = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)
        print(measures['string'])

        # add results - scene
        dict_results.append({
          'model':            str(model),
          'type':             "scene-median",
          'sensor':           str(self.sensor),
          'path':             str(self.path),
          'date_predicted':   self.predict_dates[0].strftime("%Y-%m-%d"),
          'date_execution':   dt.now().strftime("%Y-%m-%d"),
          'time_execution':   dt.now().strftime("%H:%M:%S"),
          'runtime':          str(self.classifiers_runtime[model]),
          'days_threshold':   str(self.days_threshold), 
          'grid_size':        str(self.grid_size),
          'size_train':       str("D="+str(self.df_train[0].shape)+"Dy="+str(self.df_train[0].shape)+";R="+str(self.df_randomizedsearch[0].shape)+";Ry="+str(self.df_randomizedsearch[1].shape)),
          'size_dates':       str(len(self.dates_timeseries_interval)),
          'scaler':           str(self.scaler_str),
          'morph_op':         str(self.morph_op),
          'morph_op_iters':   str(self.morph_op_iters),
          'convolve':         str(self.convolve),
          'convolve_radius':  str(self.convolve_radius),
          'days_in':          str(self.days_in), 
          'days_out':         str(self.days_out), 
          'fill_missing':     str(self.fill_missing), 
          'remove_dummies':   str(self.remove_dummies), 
          'shuffle':          str(self.shuffle), 
          'reducer':          str(self.reducer), 
          'normalized':       str(self.normalized), 
          'class_mode':       str(self.class_mode), 
          'class_weight':     str(self.class_weight), 
          'propagate':        str(self.propagate), 
          'rs_train_size':    str(self.rs_train_size), 
          'rs_iter':          str(self.rs_iter), 
          'pca_size':         str(self.pca_size),
          'attribute_lat_lon':str(self.attribute_lat_lon),
          'attribute_doy':    str(self.attribute_doy),
          'acc':              float(measures["acc"]),
          'bacc':             float(measures["bacc"]),
          'kappa':            float(measures["kappa"]),
          'vkappa':           float(measures["vkappa"]),
          'tau':              float(measures["tau"]),
          'vtau':             float(measures["vtau"]),
          'mcc':              float(measures["mcc"]),
          'f1score':          float(measures["f1score"]),
          'rmse':             float(measures["rmse"]),
          'mae':              float(measures["mae"]),
          'r2score':          float(measures["r2score"]),
          'tp':               int(measures["tp"]),
          'tn':               int(measures["tn"]),
          'fp':               int(measures["fp"]),
          'fn':               int(measures["fn"])
        })

        # plot
        c = fig.add_subplot(gs[2, :])
        c.set_title("Scene (median;RMSE:"+str(round(measures["rmse"],2))+",MAE:"+str(round(measures["mae"],2))+",R^2:"+str(round(measures["r2score"],2))+")", fontdict = {'fontsize' : 8})
        c1 = df_scene_median[['validation','prediction']].plot(kind='bar', width=.5, ax=c)
        c2 = plt.plot([-10,100],[difference,difference],"-",color="red", lw=0.5)
        c3 = plt.plot([-10,100],[0,0],"--",color="black", lw=0.5)
        c.set_xticklabels([])
        c.set_xticks([])
        c.set_ylim(-100, 100)
        c.set_ylabel('% of occurrence', fontdict = {'fontsize' : 8})
        handles, labels = c1.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="red", lw=0.5))
        c1.get_legend().remove()
        plt.legend(handles=handles, labels=["Validation/Reference","Prediction","Difference"], loc='lower center', bbox_to_anchor=(0.50, -0.15), ncol=3, fancybox=True, shadow=True, fontsize='x-small')

        # other
        fig.savefig(folder+'/image/results_median_'+str(model_short)+'.png')

        # save dataframe
        df_scene_median.to_csv(folder+'/csv/df_prediction_'+str(model_short)+'_median_scene.csv')

      # save results
      self.df_results = self.df_results.append(dict_results)

    # warning
    print('finished!')


  # validate imagem using a ROI from GEE
  # GEE Code Console: https://code.earthengine.google.com
  def validate_using_roi(self, path: str,  rois: list, labels: list = []):

    # check if there are one or predictions
    if not self.predictions is None and len(self.predictions)>0:

      # atributes
      df_columns   = ['pixel', 'lat', 'lon', 'label']
      df_true      = pd.DataFrame(columns=df_columns)

      # go over each ML model
      for model in self.predictions:

        # go over each predicted date
        for i, date in enumerate(self.predict_dates):

          # go over each roi
          for j, roi in enumerate(rois):

            # replace text in roi with date and sensor
            roi = roi.replace("date", date.strftime("%Y-%m-%d")).replace("sensor", self.sensor)

            # extract FeatureCollection from roi selected through the GEE Code Console and process image
            print()
            print("[ROI] '"+path+"/"+roi+"' with label '"+str(labels[j])+"'...")

            # get each geometry pixel values
            lons_lats_attributes  = np.array([]).reshape(0, 3)
            has_roi = False
            try:

              # go through each geometry and get pixels
              for geometry in self.splitted_geometry:
                clip                            = self.clip_image(image=ee.Image(2).toByte().paint(ee.FeatureCollection(path+'/'+roi), labels[j]), geometry=geometry, scale=self.scale)
                geometry_lons_lats_attributes   = gee.extract_latitude_longitude_pixel(image=clip, geometry=geometry, bands=['constant'], scale=self.scale)
                lons_lats_attributes            = np.vstack((lons_lats_attributes, geometry_lons_lats_attributes))

              # concatenate geometry pixel values
              extra_attributes     = np.array(list(zip(range(0,len(lons_lats_attributes)))))
              lons_lats_attributes = np.hstack((extra_attributes, lons_lats_attributes))

              # append new data from roi to dataframe
              df_true = df_true.append(pd.DataFrame(data=lons_lats_attributes,columns=df_columns).infer_objects(), ignore_index = True)

              # set that roi was found
              has_roi = True

            # error: no data in ROI or it does not exist
            except Exception as e:
              has_roi = False
              print("Error while extracting roi pixels: "+str(e))

          # roi was found?
          if has_roi:

            # fix columns labels
            df_true = df_true.apply(pd.to_numeric, errors='ignore')
            
            # prepare the datas
            df_true = df_true[df_true['label']!=2.0][['pixel', 'lat', 'lon', 'label']]

            # get rois label list
            array_labels = df_true.groupby('label').count().values[:,0].astype(str)

            # prepare model data
            df_pred         = self.predictions[model][['lat', 'lon', 'label_predicted_'+str(i)]].rename(columns={'label_predicted_'+str(i): 'label'})

            # merge true and pred dataframes, finding matched pair of latitude and longitude
            merged          = pd.merge(df_true, df_pred, on=['lat','lon'], how='inner', suffixes=('_true', '_pred'))

            # get predictions on test set
            y_true, y_pred  = merged['label_true'].values.astype(int), merged['label_pred'].values.astype(int)
            measures        = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)

            # reports
            print(measures['string'])

            # add results - pixel
            self.df_results = self.df_results.append({
              'model':            str(model),
              'type':             'roi',
              'sensor':           str(self.sensor),
              'path':             str(self.path),
              'date_predicted':   date.strftime("%Y-%m-%d"),
              'date_execution':   dt.now().strftime("%Y-%m-%d"),
              'time_execution':   dt.now().strftime("%H:%M:%S"),
              'runtime':          str(self.classifiers_runtime[model]),
              'days_threshold':   str(self.days_threshold), 
              'grid_size':        str(self.grid_size),
              'size_train':       str(array_labels),
              'size_dates':       str(len(self.dates_timeseries_interval)),
              'scaler':           str(self.scaler_str),
              'morph_op':         str(self.morph_op),
              'morph_op_iters':   str(self.morph_op_iters),
              'convolve':         str(self.convolve),
              'convolve_radius':  str(self.convolve_radius),
              'days_in':          str(self.days_in), 
              'days_out':         str(self.days_out), 
              'fill_missing':     str(self.fill_missing), 
              'remove_dummies':   str(self.remove_dummies), 
              'shuffle':          str(self.shuffle), 
              'reducer':          str(self.reducer), 
              'normalized':       str(self.normalized), 
              'class_mode':       str(self.class_mode), 
              'class_weight':     str(self.class_weight), 
              'propagate':        str(self.propagate), 
              'rs_train_size':    str(self.rs_train_size), 
              'rs_iter':          str(self.rs_iter), 
              'pca_size':         str(self.pca_size),
              'attribute_lat_lon':str(self.attribute_lat_lon),
              'attribute_doy':    str(self.attribute_doy),
              'acc':              float(measures["acc"]),
              'bacc':             float(measures["bacc"]),
              'kappa':            float(measures["kappa"]),
              'vkappa':           float(measures["vkappa"]),
              'tau':              float(measures["tau"]),
              'vtau':             float(measures["vtau"]),
              'mcc':              float(measures["mcc"]),
              'f1score':          float(measures["f1score"]),
              'rmse':             float(measures["rmse"]),
              'mae':              float(measures["mae"]),
              'tp':               int(measures["tp"]),
              'tn':               int(measures["tn"]),
              'fp':               int(measures["fp"]),
              'fn':               int(measures["fn"])
            }, ignore_index=True)

    # warning
    print("finished!")


  # save classification plot
  def save_timeseries_plot(self, df: pd.DataFrame, path: str, join: bool = False):

    # warning
    print()
    print("Creating time series (join="+str(join)+") plot to file '"+path+"'...")

    # remove dummies
    df = df[df['cloud']!=self.dummy]
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='ignore')
    df.index = df['date']

    # attributes
    attributes = [a for a in self.attributes if a != 'cloud']

    # build date string
    str_date = min(df['date']).strftime("%Y-%m-%d") + ' to ' + max(df['date']).strftime("%Y-%m-%d")

    # line styles
    linestyle = ['solid', 'dashed', 'dashdot', 'dotted']

    # crate a joined plot?
    if join:

      # create the plot
      fig = plt.figure(figsize=(10,5), dpi=300)
      fig.suptitle('Daily average time series  ('+str_date+')', fontsize=14)
      fig.autofmt_xdate()
      plt.tight_layout()
      plt.rc('xtick',labelsize=6)
      plt.rc('ytick',labelsize=6)

      # normalize data
      for attribute in attributes:
        df[attribute] = df[attribute] / df[attribute].max()

      # groupby month and year
      df = df.groupby(pd.Grouper(freq='D')).median().drop(['cloud'], axis=1).dropna()

      # separe each indice in a plot (vertically)
      ax = fig.add_subplot(1,1,1)
      ax.grid(True, linestyle='dashed', color='#a0a0a0', linewidth=0.1)
      ax.set_xlabel('Days')
      ax.set_ylabel('Normalized values')
      for i, attribute in enumerate(attributes):
        ax.plot(df.index, df[attribute], linewidth=0.5, label=attribute.upper(), linestyle=linestyle[random.randint(0,len(linestyle)-1)])
      ax.legend(borderaxespad=0., fancybox=True, shadow=True, ncol=3, fontsize="xx-small")
      ax.margins(x=0)

    # else, grid plot
    else:

      # create the plot
      fig = plt.figure(figsize=(40,10), dpi=300)
      fig.suptitle('Daily pixels time series ('+str_date+')', fontsize=16)
      plt.tight_layout()
      plt.rc('xtick',labelsize=5)
      plt.rc('ytick',labelsize=5)
      columns = 4

      # separe each indice in a plot (vertically)
      for i, attribute in enumerate(attributes):
        ax = fig.add_subplot(math.ceil(len(attributes)/columns),columns,i+1)
        ax.grid(True, linestyle='dashed', color='#a0a0a0', linewidth=0.1)
        ax.title.set_text(attribute.upper())
        ax.plot(df['index'], df[attribute], linewidth=0.2)
        ax.margins(x=0)

    # save it to file
    fig.savefig(path, bbox_inches='tight')

    # warning
    print("finished!")


  # save attriutes pair plot
  def save_attributes_pairplot(self, df: pd.DataFrame, path: str):

    # warning
    print()
    print("Creating attributes pair plot to file '"+path+"'...")

    # attributes
    attributes = [a for a in self.attributes if a != 'cloud']

    # normalize data
    df = df[df['cloud']!=self.dummy][['date']+attributes]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    df[attributes] = scaler.fit_transform(df[attributes])

    # dates
    g_dates = df.groupby('date').groups.keys()

    # creating the plot
    plot = sns.pairplot(df[df['date']==next(iter(g_dates))], vars=attributes, diag_kind="kde", plot_kws=dict(edgecolor="none", linewidth=0.2, s=1), corner=True)
    plot.savefig(path)

    # warning
    print("finished!")


  # save indices pair plot
  def save_indices_pairplot(self, df: pd.DataFrame, path: str):

    # warning
    print()
    print("Creating indices pair plot to file '"+path+"'...")

    # indices
    indices = [a for a in self.attributes_clear if a != 'cloud']

    # normalize data
    df = df[df['cloud']!=self.dummy][['date']+indices]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    df[indices] = scaler.fit_transform(df[indices])

    # dates
    g_dates = df.groupby('date').groups.keys()

    # creating the plot
    plot = sns.pairplot(df[df['date']==next(iter(g_dates))], vars=indices, diag_kind="kde", plot_kws=dict(edgecolor="none", linewidth=0.2, s=1), corner=True)
    plot.savefig(path)

    # warning
    print("finished!")


  # save results plot
  def save_results_plot(self, df: pd.DataFrame, path: str):

    # warning
    print()
    print("Creating results plot to file '"+path+"'...")

    # get models group
    df = df[['model','type','date_predicted','acc','tau','mcc','rmse']]
    g_models = df.groupby('model').groups.keys()

    # create the plot
    fig = plt.figure(figsize=(10,16), dpi=300)
    plt.tight_layout()
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)

    # ticks
    acc_ticks = np.arange(0, 1.1, 0.1)
    rmse_ticks = np.arange(0, max(df['rmse']), max(df['rmse'])/(len(acc_ticks)/2)).astype(int)

    # accuracy
    for i, model in enumerate(g_models):

      # axis
      ax = fig.add_subplot(len(g_models),1,i+1)
      ax_ = ax.twinx()
      data = df[df["model"]==model].pivot(index='date_predicted', columns='type', values=['acc', 'rmse']).reset_index()
      data.plot(ax=ax, x="date_predicted", y="acc", kind="bar")
      data.plot(ax=ax_, x="date_predicted", y="rmse", kind="line")
      ax.set_title(model, fontsize=10)
      ax.grid(True, linestyle='dashed', color='#a0a0a0', linewidth=0.5)

      # axis x1
      ax.set_ylabel('Accuracy')
      ax.set_xlabel('Predicted Dates')
      ax.legend(title="Accuracy", borderaxespad=0., fancybox=True, shadow=True, ncol=1, fontsize="x-small", loc='upper left', bbox_to_anchor=(1.06, 1))
      ax.set_yticks(acc_ticks)
      ax.set_xlim(None, None)
      ax.autoscale()
      ax.set_ylim(0,1)

      # axis x2
      ax_.set_ylabel('RMSE')
      ax_.set_yticks(rmse_ticks)
      ax_.legend(title="RMSE", borderaxespad=0., fancybox=True, shadow=True, ncol=1, fontsize="x-small", loc='upper left', bbox_to_anchor=(1.06, 0.3))
      ax_.set_ylim(0,max(df['rmse'])*2)

    # save it to file
    fig.savefig(path, bbox_inches='tight')

    # warning
    print("finished!")


  # save a image to file
  def save_image(self, image: ee.Image, path: str, bands: list = None, options: dict = {'min':0, 'max': 3000}):
    
    # warning
    print()
    print("Saving image to file '"+path+"'...")

    # default to RGB bands
    if not bands:
      bands = [self.sensor_params['red'], self.sensor_params['green'], self.sensor_params['blue']]

    # extract imagem from GEE using getThumbUrl function and saving it
    imageIO = PIL.Image.open(BytesIO(requests.get(image.select(bands).getThumbUrl(options), timeout=60).content))
    imageIO.save(path)
    
    # warning
    print("finished!")


  # save a collection to folder (time series)
  def save_image_collection(self, path: str, bands: list = None, options: dict = {'min':0, 'max': 3000}):
    
    # warning
    print()
    print("Saving image collection to folder '"+path+"'...")

    # check if folder exists
    if not os.path.exists(path):
      os.mkdir(path)

    # go through all the collection
    for date in self.dates_timeseries_interval:
      self.save_image(image=self.clip_image(self.extract_image_from_collection(date=date)), path=path+"/"+date.strftime("%Y-%m-%d")+".png", bands=bands, options=options)
    
    # warning
    print("finished!")


  # save a dataset to file
  def save_dataset(self, df: pd.DataFrame, path: str):
    
    # warning
    print()
    print("Saving dataset to file '"+path+"'...")

    # saving dataset to file
    df.to_csv(r''+path, index=False)
    
    # warning
    print("finished!")
#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################################
# ### Google Earth Engine API Support Functions
# ### Module responsible for storing Google Earth Engine's processing functions such as pixel extraction, etc.
##################################################################################################################

# Dependents Modules
import ee
import numpy as np
import traceback
from datetime import datetime as dt
from datetime import timedelta as td

# Configurations
# Defines the type of dummy value used
dummy = -99999


# Return the parameters of each sensor
def get_sensor_params(sensor: str):

  # COPERNICUS/S2_SR
  if sensor == "sentinel":
    return {
      'name': 'Sentinel-2/MSI [GEE Dataset = COPERNICUS/S2_SR]',
      'sensor': 'sentinel',
      'blue': 'B2',
      'red': 'B4',
      'green': 'B3',
      'swir': 'B11',
      'nir': 'B8',
      'c_nir': 834.05,
      'c_red': 664.75,
      'c_swir': 1612.05,
      'scale': 20,
      'start': dt.strptime("2017-03-28", "%Y-%m-%d"),
      'property_id': 'PRODUCT_ID'
    }

  # Modis MOD09GA.006
  elif sensor == "modis":
    return {
      'name': 'Modis MOD09GA.006 [GEE Dataset = MODIS/006/MOD09GA]',
      'sensor': 'modis',
      'blue': 'sur_refl_b03',
      'red': 'sur_refl_b01',
      'green': 'sur_refl_b04',
      'swir': 'sur_refl_b06',
      'nir': 'sur_refl_b02',
      'c_nir': 858.5,
      'c_red': 645.0,
      'c_swir': 1640.0,
      'scale': 500,
      'start': dt.strptime("2000-02-24", "%Y-%m-%d"),
      'property_id': 'system:id'
    }

  # Landsat-5/ETM
  elif sensor == "landsat5":
    return {
      'name': 'Landsat-5/ETM [GEE Dataset = LANDSAT/LT05/C01/T1_SR]',
      'sensor': 'landsat5',
      'blue': 'B1',
      'red': 'B3',
      'green': 'B2',
      'swir': 'B5',
      'nir': 'B4',
      'c_nir': 835.0,
      'c_red': 660.0,
      'c_swir': 1650.0,
      'scale': 30,
      'start': dt.strptime("1984-03-01", "%Y-%m-%d"),
      'end': dt.strptime("2011-10-28", "%Y-%m-%d"),
      'property_id': 'system:id'
    }

  # Landsat-7/ETM+
  elif sensor == "landsat7":
    return {
      'name': 'Landsat-7/ETM+ [GEE Dataset = LANDSAT/LE07/C01/T1_SR]',
      'sensor': 'landsat7',
      'blue': 'B1',
      'red': 'B3',
      'green': 'B2',
      'swir': 'B5',
      'nir': 'B4',
      'c_nir': 835.0,
      'c_red': 660.0,
      'c_swir': 1650.0,
      'scale': 30,
      'start': dt.strptime("1999-04-15", "%Y-%m-%d"),
      'property_id': 'system:id'
    }

  # Landsat-5/7/8 Merge
  elif sensor == "landsat578":
    return {
      'name': 'Landsat-5/7/8 Merge [GEE Dataset = LANDSAT/LT05/C01/T1_SR + LANDSAT/LE07/C01/T1_SR + LANDSAT/LC08/C01/T1_SR]',
      'sensor': 'landsat578',
      'blue': 'B2',
      'red': 'B4',
      'green': 'B3',
      'swir': 'B6',
      'nir': 'B5',
      'c_nir': 865,
      'c_red': 654.5,
      'c_swir': 1608.5,
      'scale': 30,
      'start': dt.strptime("1984-03-01", "%Y-%m-%d"),
      'property_id': 'system:id'
    }

  # Landsat-8/OLI
  else:
    return {
      'name': 'Landsat-8/OLI [GEE Dataset = LANDSAT/LC08/C01/T1_SR]',
      'sensor': 'landsat',
      'blue': 'B2',
      'red': 'B4',
      'green': 'B3',
      'swir': 'B6',
      'nir': 'B5',
      'c_nir': 865,
      'c_red': 654.5,
      'c_swir': 1608.5,
      'scale': 30,
      'start': dt.strptime("2013-02-11", "%Y-%m-%d"),
      'property_id': 'system:id'
    }


# Extract the coordinates of an image, pixel by pixel
def extract_latitude_longitude_pixel(image: ee.Image, geometry: ee.Geometry, bands: list, scale: int = 30, tile_scale: int = 16):

  # extract pixels lat and lons
  coordinates   = image.addBands(ee.Image.pixelLonLat()).select(['longitude', 'latitude']+bands)
  coordinates   = coordinates.reduceRegion(reducer=ee.Reducer.toList(), geometry=geometry, scale=scale, bestEffort=True, tileScale=tile_scale)

  # add bands
  band_values = []
  for band in bands:
    band_values.append(np.array(ee.List(coordinates.get(band)).getInfo(), dtype=np.float64))

  # build results
  band_values = np.array(band_values)
  result      = np.zeros(shape=(2+band_values.shape[0], band_values.shape[1]))
  result[0]   = np.array(ee.List(coordinates.get('longitude')).getInfo(), dtype=np.float64)
  result[1]   = np.array(ee.List(coordinates.get('latitude')).getInfo(), dtype=np.float64)
  for i, band_value in enumerate(band_values):
    result[i+2] = band_value

  # result
  return np.stack(result, axis=1)
  

# Return a collection for a given sensor
def get_sensor_collections(geometry: ee.Geometry, sensor: str = "landsat", dates: list = None):

  # COPERNICUS/S2_SR
  if sensor == "sentinel":
    collection          = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(geometry).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE','less_than', 100)).sort('system:time_start', True).map(apply_masks_sentinel)
    collection_water    = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(geometry).filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than', 10).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE',0)).sort('system:time_start', True).map(apply_masks_sentinel)

  # Modis MOD09GA.006
  elif sensor == "modis":
    collection          = ee.ImageCollection('MODIS/006/MOD09GA').filterBounds(geometry).sort('system:time_start', True).map(apply_masks_modis)
    collection_water    = ee.ImageCollection('MODIS/006/MOD44W').filterBounds(geometry)

  # Landsat-5/ETM
  elif sensor == "landsat5":
    collection          = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(geometry).filterMetadata('CLOUD_COVER','less_than', 100).sort('system:time_start', True).map(apply_masks_landsat5)  
    collection_water    = ee.ImageCollection('GLCF/GLS_WATER').filterBounds(geometry)

  # Landsat-7/ETM+
  elif sensor == "landsat7":
    collection          = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(geometry).filterMetadata('CLOUD_COVER','less_than', 100).sort('system:time_start', True).map(apply_masks_landsat7)  
    collection_water    = ee.ImageCollection('GLCF/GLS_WATER').filterBounds(geometry)

  # Landsat-5/7/8 - Merge
  elif sensor == "landsat578":
    
    # Landsat-5/ETM
    collection5         = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterDate(get_sensor_params("landsat5")["start"],get_sensor_params("landsat5")["end"]).filterBounds(geometry).filterMetadata('CLOUD_COVER','less_than', 100).sort('system:time_start', True).select(["B1","B2","B3","B4","B5","B6","pixel_qa"]).map(apply_masks_landsat5)  
    
    # Landsat-7/ETM+
    collection7         = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterDate(get_sensor_params("landsat5")["end"],get_sensor_params("landsat")["start"]).filterBounds(geometry).filterMetadata('CLOUD_COVER','less_than', 100).sort('system:time_start', True).select(["B1","B2","B3","B4","B5","B6","pixel_qa"]).map(apply_masks_landsat7)  
    
    # Landsat-8/OLI
    collection8         = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometry).filterMetadata('CLOUD_COVER','less_than', 100).sort('system:time_start', True).select(["B1","B2","B3","B4","B5","B6","pixel_qa"]).map(apply_masks_landsat)  
    
    # Landsat Merge
    collection          = collection8.merge(collection7.merge(collection5)).sort('system:time_start', True)
    collection_water    = ee.ImageCollection('GLCF/GLS_WATER').filterBounds(geometry)

  # Landsat-8/OLI
  else:
    collection          = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometry).filterMetadata('CLOUD_COVER','less_than', 100).sort('system:time_start', True).map(apply_masks_landsat)  
    collection_water    = ee.ImageCollection('GLCF/GLS_WATER').filterBounds(geometry)
    
  # Filter dates
  if dates:
    collection = collection.filterDate(dates[0],dates[1])
    if not collection_water:
      collection_water = collection.filterDate(dt.strftime(dt.strptime(dates[0], "%Y-%m-%d")-td(days=180), "%Y-%m-%d"), dt.strftime(dt.strptime(dates[1], "%Y-%m-%d")+td(days=180), "%Y-%m-%d")).sort('system:time_start', True)

  return collection, collection_water


# Cloud Mask and Cloud Shadows
def mask_cloud_shadow(image, sensor: str):

  # COPERNICUS/S2_SR
  # Values: https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
  if sensor == "sentinel":
      qa = image.select('QA60')
      return qa.bitwiseAnd(1 << 10).eq(0) and (qa.bitwiseAnd(1 << 11).eq(0))

  # Modis MOD09GA.006
  # Values: https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD09GA
  elif sensor == "modis":
      qa = image.select('state_1km')
      return qa.bitwiseAnd(1 << 10).eq(0)

  # Landsat-5/ETM
  # Binary Values: https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
  # Absolute Values: https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_SurfaceReflectance-LEDAPS_ProductGuide-v2.pdf
  elif sensor == "landsat5":
      qa = image.select('pixel_qa')
      return (qa.bitwiseAnd(1 << 5).eq(0) and (qa.bitwiseAnd(1 << 7).eq(0))) or (qa.bitwiseAnd(1 << 3).eq(0))

  # Landsat-7/ETM+
  # Binary Values: https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
  # Absolute Values: https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_SurfaceReflectance-LEDAPS_ProductGuide-v2.pdf
  elif sensor == "landsat7":
      qa = image.select('pixel_qa')
      return (qa.bitwiseAnd(1 << 5).eq(0) and (qa.bitwiseAnd(1 << 7).eq(0))) or (qa.bitwiseAnd(1 << 3).eq(0))

  # Landsat-8/OLI
  # Binary Values: https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
  # Absolute Values: https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1368_L8_SurfaceReflectanceCode-LASRC_ProductGuide-v2.pdf
  else:
      qa = image.select('pixel_qa')
      return qa.bitwiseAnd(1 << 3).eq(0) and (qa.bitwiseAnd(1 << 5).eq(0))


# Water Mask
def mask_water(image, sensor: str):

  # Sentinel-2/MSI
  # Values: https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
  if sensor == "sentinel":
      qa = image.select('SCL')
      return qa.eq(6).Not()

  # Modis MOD09GA.006
  # Values: https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD09GA
  elif sensor == "modis":
      qa = image.select('state_1km')
      return ((qa.bitwiseAnd(1 << 3).neq(0)) and (qa.bitwiseAnd(1 << 4).eq(0) or qa.bitwiseAnd(1 << 5).eq(0)))

  # Landsat-5/ETM
  # Binary Values: https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
  # Absolute Values: https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_SurfaceReflectance-LEDAPS_ProductGuide-v2.pdf
  elif sensor == "landsat5":
      qa = image.select('pixel_qa')
      return qa.bitwiseAnd(1 << 2).eq(0)

  # Landsat-7/ETM+
  # Binary Values: https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
  # Absolute Values: https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_SurfaceReflectance-LEDAPS_ProductGuide-v2.pdf
  elif sensor == "landsat7":
      qa = image.select('pixel_qa')
      return qa.bitwiseAnd(1 << 2).eq(0)
      
  # Landsat-8/OLI
  # Binary Values: https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
  # Absolute Values: https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1368_L8_SurfaceReflectanceCode-LASRC_ProductGuide-v2.pdf
  else:
      qa = image.select('pixel_qa')
      return qa.bitwiseAnd(1 << 2).eq(0)


# Sentinel-2/MSI
def apply_masks_sentinel(image):
  return apply_masks(image, get_sensor_params("sentinel"))

# Landsat-5/ETM
def apply_masks_landsat5(image):
  return apply_masks(image, get_sensor_params("landsat5"))

# Landsat-7/ETM+
def apply_masks_landsat7(image):
  return apply_masks(image, get_sensor_params("landsat7"))

# Lantsat-8/OLI
def apply_masks_landsat(image):
  return apply_masks(image, get_sensor_params("landsat"))

# Modis MOD09GA.006
def apply_masks_modis(image):
  return apply_masks(image, get_sensor_params("modis"))


# Apply indexes and apply water masks, cloud
def apply_masks(image, params: dict):

  # Apply the water and cloud masks and cloud shade
  blank           = ee.Image(abs(dummy))
  water           = blank.updateMask(mask_water(image, params['sensor']).Not()).rename('water')
  cloud           = ee.Image(1).updateMask(mask_cloud_shadow(image, params['sensor']).Not()).rename('cloud')
  nocloud         = blank.updateMask(mask_cloud_shadow(image, params['sensor'])).rename('nocloud')
  water_nocloud   = water.updateMask(nocloud).rename('water_nocloud')
      
  # Apply the indexes available in the image
  ndwi            = image.expression('(green - swir) / (green + swir)',{'swir':image.select(params['swir']).multiply(0.0001),'green':image.select(params['green']).multiply(0.0001)}).rename('ndwi').cast({"ndwi": "double"})
  ndvi            = image.expression('(nir - red) / (nir + red)',{'nir':image.select(params['nir']).multiply(0.0001),'red':image.select(params['red']).multiply(0.0001)}).rename('ndvi').cast({"ndvi": "double"})
  sabi            = image.expression('(nir - red) / (blue + green)',{'nir':image.select(params['nir']).multiply(0.0001),'red':image.select(params['red']).multiply(0.0001),'blue':image.select(params['blue']).multiply(0.0001),'green':image.select(params['green']).multiply(0.0001)}).rename('sabi').cast({"sabi": "double"}) # Alawadi (2010)
  fai             = image.expression('nir - (red + (swir - red) * ((c_nir - c_red) / (c_swir - c_red)))',{'swir':image.select(params['swir']).multiply(0.0001),'nir':image.select(params['nir']).multiply(0.0001),'red':image.select(params['red']).multiply(0.0001),'c_nir':params['c_nir'],'c_red':params['c_red'],'c_swir':params['c_swir']}).rename('fai').cast({"fai": "double"}) # Oyama et al (2015)
  label           = image.expression('((cloud == 1) ? -1 : (ndwi < 0.3)+(ndvi > -0.15)+(sabi > -0.10)+(fai > -0.004))', {'ndwi': ndwi.select('ndwi'), 'ndvi': ndvi.select('ndvi'), 'sabi': sabi.select('sabi'), 'fai': fai.select('fai'), 'cloud': cloud.select('cloud')}).rename('label')

  # Create the bands to the image and return it
  return image.addBands([water, water_nocloud, cloud, nocloud, ndwi, ndvi, sabi, fai, label])


# Apply to a mask and return image with the new band
def apply_mask(image, mask, band_from, band_to, remove_empty_pixels = False):
  image_mask = image.select(band_from).updateMask(mask).rename(band_to)
  if remove_empty_pixels:
    return image.addBands([image_mask])
  image_mask = ee.Image(abs(dummy)).blend(image_mask).rename(band_to)
  return image.addBands([image_mask])


# Get image min and max values
def get_image_min_max(image: ee.Image, geometry: ee.Geometry, scale: int = None, tile_scale: int = 16):
  if scale is None:
    scale = image.projection().nominalScale()
  image = image.addBands(ee.Image.pixelLonLat())
  return image.reduceRegion(reducer=ee.Reducer.min(), geometry=geometry, scale=scale, bestEffort=True, tileScale=tile_scale).getInfo(), image.reduceRegion(reducer=ee.Reducer.max(), geometry=geometry, scale=scale, bestEffort=True, tileScale=tile_scale).getInfo()


# Get image counters
def get_image_counters(image: ee.Image, geometry: ee.Geometry, scale: int = None, tile_scale: int = 16):
  if scale is None:
    scale = image.projection().nominalScale()
  return image.reduceRegion(reducer=ee.Reducer.count(), geometry=geometry, scale=scale, bestEffort=True, tileScale=tile_scale).getInfo()


# Get geometry from diagonal points
def get_geometry_from_lat_lon(lat_lon: str):

  # Selection of coordinates (colon, lat and lon separated by comma, all together) and dates by the user (two dates, beginning and end, separated by commas)
  x1,y1,x2,y2 = lat_lon.split(",")

  # Assemble Geometry on Google Earth Engine
  geometry = ee.Geometry.Polygon(
        [[[float(x1),float(y2)],
          [float(x2),float(y2)],
          [float(x2),float(y1)],
          [float(x1),float(y1)],
          [float(x1),float(y2)]]])

  # return the geometry
  return geometry
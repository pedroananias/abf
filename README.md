# Anomalous Behaviour Forecast

Module responsable for forecasting anomalies occurences based on images from Google Earth Engine API and machine learning



### Dependencies

- Python 3.7.7 64-bit ou superior
- Modules: oauth2client earthengine-api matplotlib pandas numpy requests pillow geojson argparse joblib scipy seaborn sklearn tensorflow



### Attention, before running:

Before running the script and after installing the libraries, you must authenticate with the Google Earth Engine API using the following command:

```
earthengine authenticate
```



### How to execute the default script?

python /path/to/abf/script.py --lat_lon=-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826 --name=erie --from_date=2019-07-11 --reducer --class_mode --disable_attribute_lat_lon




### What are the results?

The script will generate a 5-days forecast of occurrence of algae blooming in the inserted study area starting from input date. Therefore, a folder located in 'data' is created and named based on the date and version of the script executed. Example: /path/to/abf/data/20201026_090433[v=V20-erie,d=2019-07-11,dt=180,din=5,dout=5,m=rf,g=12]. 

ATTENTION: big date range tends to lead to memory leak, stopping the script execution. It is always a good pratice to split the dates in two or three parts, unless you have a big amount of memory in your computer.

The following results are generated:

- timeseries.csv (Time series from dates ranging current day and days threshold parameter)
- results.csv (Accuracy results from the forecasted period validated using indices thresholding approach)
- prediction/csv (CSVs from generated outputs and predictions separated by day)
- prediction/geojson (GeoJSONs from generated outputs and predictions separated by day)
- prediction/image  (Plot from generated outputs and predictions separated by day)



### Example

```
# Import
import ee
from modules import abf

# Initialize Google Earth Engine
ee.Initialize()

# folder where to save results
folder = "/path/to/desired/folder"

# create algorithm object
algorithm = abf.Abf(days_threshold=180,
                    grid_size=12,
                    sensor="modis", 
                    geometry=geometry,
                    lat_lon="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826",
                    cache_path=folder,
                    force_cache=False,
                    days_in=5,
                    days_out=5,
                    from_date="2019-07-11",
                    model="rf",
                    reducer=True,
                    class_mode=True,
                    attribute_lat_lon=False)

# preprocessing
algorithm.process_timeseries_data()
algorithm.process_training_data(df=algorithm.df_timeseries)

# train/predict
algorithm.train(batch_size=2048, disable_gpu=True)
algorithm.predict(folder=folder+"/prediction")

# prediction results
algorithm.save_dataset(df=algorithm.df_results, path=folder+'/results.csv')

# preprocessing results
algorithm.save_dataset(df=algorithm.df_timeseries, path=folder+'/timeseries.csv')
```

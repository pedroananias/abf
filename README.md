# Anomaly and Algal Bloom Forecast

Module responsable for forecasting anomalies and algal bloom occurences based on images from Google Earth Engine API and machine learning



### Dependencies

- Python 3.7.7 64-bit ou superior
- Modules: oauth2client earthengine-api matplotlib pandas numpy requests pillow geojson argparse hashlib joblib warnings scipy seaborn sklearn tensorflow



### Attention, before running:

Before running the script and after installing the libraries, you must authenticate with the Google Earth Engine API using the following command:

```
earthengine authenticate
```



### How to execute the default script?

python /path/to/abf/script.py --lat_lon=-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826 --name=erie --from_date=2019-07-11




### What are the results?

The script will generate a 5-days forecast of occurrence of algae blooming in the inserted study area starting from input date. Therefore, a folder located in 'data' is created and named based on the date and version of the script executed. Example: /path/to/abyo/data/20200701_111426[v=V13-erie,date=2019-07-11,dt=1825,gs=3,din=5,dout=5,mop=None,mit=1,cv=False,cvr=1,sf=True,m=rf,fill=time,rd=False]. 

ATTENTION: big date range tends to lead to memory leak, stopping the script execution. It is alwaysa a good pratice to split the dates in two or three parts, unless you have a big amount of memory in your computer.

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
algorithm = abf.Abf(days_threshold=1825,
                    grid_size=3,
                    sensor="modis", 
                    geometry=geometry,
                    lat_lon="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826",
                    cache_path=folder,
                    force_cache=False,
                    scaler='robust',
                    days_in=5,
                    days_out=5,
                    from_date="2019-07-11",
                    model="rf")

# preprocessing
algorithm.process_timeseries_data()
algorithm.process_training_data(df=algorithm.df_timeseries)

# train/predict
algorithm.train(batch_size=4096, disable_gpu=True)
algorithm.predict(folder=folder+"/prediction")

# prediction results
algorithm.save_dataset(df=algorithm.df_results, path=folder+'/results.csv')
algorithm.save_results_plot(df=algorithm.df_results, path=folder+'/results.png')

# preprocessing results
algorithm.save_dataset(df=algorithm.df_timeseries, path=folder+'/timeseries.csv')
algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries.png')
algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries_join.png', join=True)
```

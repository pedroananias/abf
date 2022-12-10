# Main
import ee
import pandas as pd
import time
import warnings
import os
import traceback
import gc
from pathlib import Path
import click

# Sub
from datetime import datetime as dt
from ee import EEException

# Extras src
from abf.abf import Abf

# ignore warnings
warnings.filterwarnings("ignore")

this_directory = Path(__file__).parent
__version__ = ""
exec((this_directory / "version.py").read_text(encoding="utf-8"))


@click.command()
@click.option(
    "--lat_lon",
    show_default=True,
    default="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826",
    help="Two diagonal points (Latitude 1, Longitude 1, Latitude 2, Longitude 2) "
         "of the study area",
)
@click.option(
    "--instant",
    show_default=True,
    default="2013-10-11",
    help="Date to end time series (it will forecast 5 days starting from this date)",
)
@click.option(
    "--name",
    show_default=True,
    default="erie",
    help="Place where to save generated files",
)
@click.option(
    "--sensor",
    default="modis",
    show_default=True,
    help="Define the selected sensor where images will be downloaded from: "
    "landsat, sentinel, modis",
)
@click.option(
    "--spanning_period",
    show_default=True,
    type=int,
    default=180,
    help="Spanning period used to build the timeseries and training set",
)
@click.option(
    "--past_steps",
    show_default=True,
    type=int,
    default=4,
    help="Past steps to be used as input forecast",
)
@click.option(
    "--forecasting_period",
    show_default=True,
    type=int,
    default=5,
    help="Forecasting period to be used as output forecast",
)
@click.option(
    "--spatial_context_size",
    show_default=True,
    type=int,
    default=7,
    help="Spatial context size in pixels that will be used "
         "in continuous/occurrences results",
)
@click.option(
    "--model",
    show_default=True,
    default="rf",
    help="Select the desired module: mlp, lstm, rf, svm or all (None)",
)
@click.option(
    "--fill_missing",
    show_default=True,
    default="time",
    help="Defines algorithm to be used to fill empty dates and values: "
         "dummy, ffill, bfill, time, linear",
)
@click.option(
    "--remove_dummies",
    is_flag=True,
    show_default=True,
    default=False,
    help="Defines if the dummies will be removed before training "
         "(only works with fill_missing=dummy)",
)
@click.option(
    "--scaler",
    show_default=True,
    default="minmax",
    help="Defines algorithm to be used in the data scalling process: "
         "robust, minmax or standard",
)
@click.option(
    "--disable_reducer",
    is_flag=True,
    show_default=True,
    default=False,
    help="Defines if reducer will not be applied to remove unnecessary features",
)
@click.option(
    "--disable_normalization",
    is_flag=True,
    show_default=True,
    default=False,
    help="Defines if normalization (-1,1) will not be applied "
         "to indices mndwi, ndvi and sabi",
)
@click.option(
    "--regression_mode",
    is_flag=True,
    show_default=True,
    default=False,
    help="Defines whether will use raw values or classes in the regression models",
)
@click.option(
    "--class_weight",
    is_flag=True,
    show_default=True,
    default=False,
    help="Defines whether classes will have defined weights for each",
)
@click.option(
    "--propagate",
    is_flag=True,
    show_default=True,
    default=False,
    help="Defines whether predictions will be propagated ahead",
)
@click.option(
    "--rs_train_size",
    show_default=True,
    type=float,
    default=500.0,
    help="It allow increase the randomized search dataset training size "
         "(it can be a floater or integer)",
)
@click.option(
    "--rs_iter",
    show_default=True,
    type=int,
    default=500,
    help="It allow increase the randomized search iteration size",
)
@click.option(
    "--pca_size",
    show_default=True,
    type=float,
    default=0.900,
    help="Define PCA reducer variance size",
)
@click.option(
    "--reduction",
    show_default=True,
    default="median",
    help="Define which reduction (median or min) will be used in the reduction stats",
)
@click.option(
    "--convolve",
    is_flag=True,
    show_default=True,
    default=False,
    help="Define if a convolution box-car low-pass filter will be applied "
         "to images before training",
)
@click.option(
    "--convolve_radius",
    show_default=True,
    type=int,
    default=1,
    help="Define the amont of radius will be used in "
         "the convolution box-car low-pass filter",
)
@click.option(
    "--enable_attribute_lat_lon",
    is_flag=True,
    show_default=True,
    default=False,
    help="Enable attributes lat and lon in the training process",
)
@click.option(
    "--disable_attribute_doy",
    is_flag=True,
    show_default=True,
    default=False,
    help="Disable attribute doy from training modeling",
)
@click.option(
    "--disable_shuffle",
    is_flag=True,
    show_default=True,
    default=False,
    help="Disable data shutffle before splitting into train and test matrix",
)
@click.option(
    "--save_pairplots",
    is_flag=True,
    show_default=True,
    default=False,
    help="Save pairplots from attributes and indices",
)
@click.option(
    "--save_train",
    is_flag=True,
    show_default=True,
    default=False,
    help="Enable saving the training dataset (csv)",
)
@click.option(
    "--cloud_threshold",
    show_default=True,
    type=float,
    default=0.50,
    help="Define which cloud threshold will be used in the timeseries modelling process",
)
@click.option(
    "--shapefile",
    help="Use a shapefile to clip a region of interest",
)
@click.option(
    "--output_folder",
    default=None,
    help="Specify desired results output folder",
)
def forecast(
    lat_lon: str,
    instant: str,
    name: str,
    sensor: str,
    spanning_period: int,
    past_steps: int,
    forecasting_period: int,
    spatial_context_size: int,
    model: str,
    fill_missing: str,
    remove_dummies: bool,
    scaler: str,
    disable_reducer: bool,
    disable_normalization: bool,
    regression_mode: bool,
    class_weight: bool,
    propagate: bool,
    rs_train_size: float,
    rs_iter: int,
    pca_size: float,
    reduction: str,
    convolve: bool,
    convolve_radius: int,
    enable_attribute_lat_lon: bool,
    disable_attribute_doy: bool,
    disable_shuffle: bool,
    save_pairplots: bool,
    save_train: bool,
    cloud_threshold: float,
    shapefile: str,
    output_folder: str,
):
    try:

        # Start script time counter
        start_time = time.time()

        # Google Earth Engine API initialization
        try:
            ee.Initialize()
        except (Exception, EEException) as e:
            print(f"Google Earth Engine authentication/initialization error: {e}. "
                  f"Please, manually log in GEE paltform with `earthengine authenticate`. "
                  f"** See README.md file for the complete instructions **")

        # ### Working directory

        # Data path
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            folderRoot = output_folder + "/results"
            if not os.path.exists(folderRoot):
                os.mkdir(folderRoot)
        else:
            folderRoot = os.path.dirname(os.path.realpath(__file__)) + "/../../results"
            if not os.path.exists(folderRoot):
                os.mkdir(folderRoot)

        # Images path
        folderCache = os.path.dirname(os.path.realpath(__file__)) + "/.cache"
        if not os.path.exists(folderCache):
            os.mkdir(folderCache)

        # ### Selection of coordinates

        # (colon, lat and lon separated by comma, all together)
        # and dates by the user (two dates, beginning and end, separated by commas)
        x1, y1, x2, y2 = lat_lon.split(",")

        # Assemble Geometry on Google Earth Engine
        geometry = ee.Geometry.Polygon(
            [
                [
                    [float(x1), float(y2)],
                    [float(x2), float(y2)],
                    [float(x2), float(y1)],
                    [float(x1), float(y1)],
                    [float(x1), float(y2)],
                ]
            ]
        )

        # ### ABF execution

        # folder to save results from algorithm at
        folder = (
            folderRoot
            + "/"
            + dt.now().strftime("%Y%m%d_%H%M%S")
            + "[v="
            + str(__version__)
            + "-"
            + str(name)
            + ",m="
            + str(model)
            + ",d="
            + str(instant)
            + ",ct="
            + str(cloud_threshold)
            + "]"
        )
        if not os.path.exists(folder):
            os.mkdir(folder)

        # create algorithm
        algorithm = Abf(
            days_threshold=spanning_period,
            grid_size=spatial_context_size,
            sensor=sensor,
            geometry=geometry,
            lat_lon=lat_lon,
            path=folder,
            cache_path=folderCache,
            force_cache=False,
            morph_op=None,
            morph_op_iters=1,
            convolve=convolve,
            convolve_radius=convolve_radius,
            scaler=scaler,
            days_in=past_steps,
            days_out=forecasting_period,
            from_date=instant,
            model=model,
            fill_missing=fill_missing,
            remove_dummies=remove_dummies,
            reducer=not disable_reducer,
            normalized=not disable_normalization,
            class_mode=regression_mode,
            class_weight=class_weight,
            propagate=propagate,
            rs_train_size=rs_train_size,
            rs_iter=rs_iter,
            pca_size=pca_size,
            attribute_lat_lon=enable_attribute_lat_lon,
            attribute_doy=not disable_attribute_doy,
            shuffle=not disable_shuffle,
            test_mode=False,
            shapefile=shapefile,
            cloud_threshold=cloud_threshold,
        )

        # preprocessing
        algorithm.process_timeseries_data()
        algorithm.process_training_data(df=algorithm.df_timeseries)

        # save pairplots output
        if save_pairplots:
            if not os.path.exists(folder + "/pairplots"):
                os.mkdir(folder + "/pairplots")
            algorithm.save_attributes_pairplot(
                df=algorithm.df_timeseries, path=folder + "/pairplots/attributes.png"
            )
            algorithm.save_indices_pairplot(
                df=algorithm.df_timeseries, path=folder + "/pairplots/indices.png"
            )

        # train/predict
        else:
            algorithm.train(batch_size=2048, disable_gpu=True)
            algorithm.predict(
                folder=folder + "/prediction",
                #path="users/pedroananias/" + str(name),
                #chla_threshold=20,
            )
            algorithm.predict_reduction(
                folder=folder + "/prediction", reduction=str(reduction)
            )

            # prediction results
            algorithm.save_dataset(
                df=algorithm.df_results, path=folder + "/results.csv"
            )
            algorithm.save_dataset(
                df=algorithm.df_scene, path=folder + "/results_scene.csv"
            )

            # preprocessing results
            algorithm.save_dataset(
                df=algorithm.df_timeseries, path=folder + "/timeseries.csv"
            )

            # save training datasets
            if save_train:
                algorithm.save_dataset(
                    df=pd.DataFrame(algorithm.df_train[0]),
                    path=folder + "/df_train_X.csv",
                )
                algorithm.save_dataset(
                    df=pd.DataFrame(algorithm.df_train[1]),
                    path=folder + "/df_train_y.csv",
                )
                algorithm.save_dataset(
                    df=pd.DataFrame(algorithm.df_test[0]),
                    path=folder + "/df_test_X.csv",
                )
                algorithm.save_dataset(
                    df=pd.DataFrame(algorithm.df_test[1]),
                    path=folder + "/df_test_y.csv",
                )
                algorithm.save_dataset(
                    df=pd.DataFrame(algorithm.df_classification),
                    path=folder + "/df_classification_X.csv",
                )

        # results
        # add results and save it on disk
        path_df_results = folderRoot + "/results.csv"
        df_results = (
            pd.read_csv(path_df_results)
            .drop(["Unnamed: 0"], axis=1, errors="ignore")
            .append(algorithm.df_results)
            if os.path.exists(path_df_results)
            else algorithm.df_results.copy(deep=True)
        )
        df_results.to_csv(r"" + path_df_results)

        # clear memory
        del algorithm
        gc.collect()

        # ### Script termination notice
        script_time_all = time.time() - start_time
        debug = (
            "***** Script execution completed successfully (-- %s seconds --) *****"
            % script_time_all
        )
        print()
        print(debug)

    except Exception:

        # ### Script execution error warning

        # Execution
        print()
        print()
        debug = "***** Error on script execution: " + str(traceback.format_exc())
        print(debug)

        # Removes the folder created initially with the result of execution
        script_time_all = time.time() - start_time
        debug = (
            "***** Script execution could not be completed (-- %s seconds --) *****"
            % script_time_all
        )
        print(debug)


if __name__ == "__main__":
    forecast()

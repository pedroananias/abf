#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ARRAYS - FULL
declare -a DAYS_THRESHOLD=("90" "180") # "90" "180" "365" "730"
declare -a DAYS_IN_OUT=("--days_in=1 --days_out=5" "--days_in=2 --days_out=5" "--days_in=3 --days_out=5" "--days_in=4 --days_out=5" "--days_in=5 --days_out=5" "--days_in=10 --days_out=5" "--days_in=15 --days_out=5" "--days_in=30 --days_out=5")
declare -a GRID_SIZE=("--grid_size=7")
declare -a MODELS=("rf") # "rf" "svm" "mlp" "lstm"
declare -a FILLS_MISSING=("--fill_missing=time")
declare -a REDUCER=("--reducer --pca_size=0.900")
declare -a CLASS_MODE=("--class_mode" "--class_mode --class_weight")
declare -a NORMALIZE=("")
declare -a RS_TRAIN_SIZES=("--rs_train_size=500.0")
declare -a RS_ITERS=("--rs_iter=500")
declare -a ATTR=("--disable_attribute_lat_lon")
declare -a PROPAGATE=("" "--propagate")
declare -a SCALER=("--scaler=robust" "--scaler=minmax" "--scaler=standard")

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE CHILIKA #1

# ARGUMENTS
NAME="chilika"
LAT_LON="85.08856074928927,19.698732779758075,85.28279700936078,19.546819354913833"
FROM_DATE="2014-11-25"

# EXECUTIONS
for rs_train_size in "${RS_TRAIN_SIZES[@]}"
do
	for rs_iter in "${RS_ITERS[@]}"
	do
		for class_mode in "${CLASS_MODE[@]}"
		do
			for day_threshold in "${DAYS_THRESHOLD[@]}"
			do
				for days_in_out in "${DAYS_IN_OUT[@]}"
				do
					for grid_size in "${GRID_SIZE[@]}"
					do
						for reducer in "${REDUCER[@]}"
						do
							for fill_missing in "${FILLS_MISSING[@]}"
							do
								for model in "${MODELS[@]}"
								do
									for normalized in "${NORMALIZE[@]}"
									do
										for propagate in "${PROPAGATE[@]}"
										do
											for scaler in "${SCALER[@]}"
											do
												eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler"
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

############################################################################################



############################################################################################
## LAKE ERIE #1

# ARGUMENTS
NAME="erie"
LAT_LON="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826"
FROM_DATE="2013-10-11"

# EXECUTIONS
for rs_train_size in "${RS_TRAIN_SIZES[@]}"
do
	for rs_iter in "${RS_ITERS[@]}"
	do
		for class_mode in "${CLASS_MODE[@]}"
		do
			for attr in "${ATTR[@]}"
			do
				for day_threshold in "${DAYS_THRESHOLD[@]}"
				do
					for days_in_out in "${DAYS_IN_OUT[@]}"
					do
						for grid_size in "${GRID_SIZE[@]}"
						do
							for reducer in "${REDUCER[@]}"
							do
								for fill_missing in "${FILLS_MISSING[@]}"
								do
									for model in "${MODELS[@]}"
									do
										for normalized in "${NORMALIZE[@]}"
										do
											for propagate in "${PROPAGATE[@]}"
											do
												for scaler in "${SCALER[@]}"
												do
													eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler"
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

############################################################################################



############################################################################################
## LAKE CHILIKA #2

# ARGUMENTS
NAME="chilika"
LAT_LON="85.08856074928927,19.698732779758075,85.28279700936078,19.546819354913833"
FROM_DATE="2019-11-20"

# EXECUTIONS
for rs_train_size in "${RS_TRAIN_SIZES[@]}"
do
	for rs_iter in "${RS_ITERS[@]}"
	do
		for class_mode in "${CLASS_MODE[@]}"
		do
			for day_threshold in "${DAYS_THRESHOLD[@]}"
			do
				for days_in_out in "${DAYS_IN_OUT[@]}"
				do
					for grid_size in "${GRID_SIZE[@]}"
					do
						for reducer in "${REDUCER[@]}"
						do
							for fill_missing in "${FILLS_MISSING[@]}"
							do
								for model in "${MODELS[@]}"
								do
									for normalized in "${NORMALIZE[@]}"
									do
										for propagate in "${PROPAGATE[@]}"
										do
											for scaler in "${SCALER[@]}"
											do
												eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler"
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

############################################################################################



############################################################################################
## LAKE ERIE #2

# ARGUMENTS
NAME="erie"
LAT_LON="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826"
FROM_DATE="2015-07-27"

# EXECUTIONS
for rs_train_size in "${RS_TRAIN_SIZES[@]}"
do
	for rs_iter in "${RS_ITERS[@]}"
	do
		for class_mode in "${CLASS_MODE[@]}"
		do
			for attr in "${ATTR[@]}"
			do
				for day_threshold in "${DAYS_THRESHOLD[@]}"
				do
					for days_in_out in "${DAYS_IN_OUT[@]}"
					do
						for grid_size in "${GRID_SIZE[@]}"
						do
							for reducer in "${REDUCER[@]}"
							do
								for fill_missing in "${FILLS_MISSING[@]}"
								do
									for model in "${MODELS[@]}"
									do
										for normalized in "${NORMALIZE[@]}"
										do
											for propagate in "${PROPAGATE[@]}"
											do
												for scaler in "${SCALER[@]}"
												do
													eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler"
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

############################################################################################


############################################################################################
## LAKE CHILIKA #3

# ARGUMENTS
NAME="chilika"
LAT_LON="85.08856074928927,19.698732779758075,85.28279700936078,19.546819354913833"
FROM_DATE="2020-03-29"

# EXECUTIONS
for rs_train_size in "${RS_TRAIN_SIZES[@]}"
do
	for rs_iter in "${RS_ITERS[@]}"
	do
		for class_mode in "${CLASS_MODE[@]}"
		do
			for day_threshold in "${DAYS_THRESHOLD[@]}"
			do
				for days_in_out in "${DAYS_IN_OUT[@]}"
				do
					for grid_size in "${GRID_SIZE[@]}"
					do
						for reducer in "${REDUCER[@]}"
						do
							for fill_missing in "${FILLS_MISSING[@]}"
							do
								for model in "${MODELS[@]}"
								do
									for normalized in "${NORMALIZE[@]}"
									do
										for propagate in "${PROPAGATE[@]}"
										do
											for scaler in "${SCALER[@]}"
											do
												eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler"
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

############################################################################################



############################################################################################
## LAKE ERIE #3

# ARGUMENTS
NAME="erie"
LAT_LON="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826"
FROM_DATE="2019-07-11"

# EXECUTIONS
for rs_train_size in "${RS_TRAIN_SIZES[@]}"
do
	for rs_iter in "${RS_ITERS[@]}"
	do
		for class_mode in "${CLASS_MODE[@]}"
		do
			for attr in "${ATTR[@]}"
			do
				for day_threshold in "${DAYS_THRESHOLD[@]}"
				do
					for days_in_out in "${DAYS_IN_OUT[@]}"
					do
						for grid_size in "${GRID_SIZE[@]}"
						do
							for reducer in "${REDUCER[@]}"
							do
								for fill_missing in "${FILLS_MISSING[@]}"
								do
									for model in "${MODELS[@]}"
									do
										for normalized in "${NORMALIZE[@]}"
										do
											for propagate in "${PROPAGATE[@]}"
											do
												for scaler in "${SCALER[@]}"
												do
													eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler"
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

############################################################################################
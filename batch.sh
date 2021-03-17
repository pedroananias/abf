#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ARRAYS - FULL
declare -a DAYS_THRESHOLD=("180" "365")
declare -a DAYS_IN_OUT=("--days_in=5 --days_out=5")
declare -a GRID_SIZE=("--grid_size=9" "--grid_size=12")
declare -a MODELS=("rf" "svm" "mlp" "lstm")
declare -a FILLS_MISSING=("--fill_missing=time")
declare -a REDUCER=("--reducer --pca_size=0.900")
declare -a CLASS_MODE=("--class_mode")
declare -a NORMALIZE=("")
declare -a RS_TRAIN_SIZES=("--rs_train_size=0.01")
declare -a RS_ITERS=("--rs_iter=25")
declare -a ATTR=("--disable_attribute_lat_lon")

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE ERIE

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
											eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr"
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
## LAKE CHILIKA

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
										eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $days_in_out $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter"
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
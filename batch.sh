#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ATTRIBUTES
declare -a GRID_SIZES=(3 5 7)
declare -a MODELS=("rf" "svm") # "rf" "svm" "mlp" "lstm"
declare EXTRA="--days_threshold=90 --days_in=4 --days_out=5"

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE CHILIKA
NAME="chilika"
LAT_LON="85.08856074928927,19.698732779758075,85.28279700936078,19.546819354913833"
declare -a DATES=("2014-11-25" "2019-11-20" "2020-03-29")

# EXECUTIONS
for model in "${MODELS[@]}"
do
	for date in "${DATES[@]}"
	do
		for grid_size in "${GRID_SIZES[@]}"
		do
			eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$date --model=$model --grid_size=$grid_size $EXTRA"
		done
	done
done

############################################################################################



############################################################################################
## LAKE ERIE
NAME="erie"
LAT_LON="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826"
declare -a DATES=("2013-10-11" "2015-07-27" "2019-07-11")

# EXECUTIONS
for model in "${MODELS[@]}"
do
	for date in "${DATES[@]}"
	do
		for grid_size in "${GRID_SIZES[@]}"
		do
			eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$date --model=$model --grid_size=$grid_size $EXTRA"
		done
	done
done

############################################################################################



############################################################################################
## LAKE TAIHU
NAME="taihu"
LAT_LON="119.88067005824256,31.273892900245198,120.12089092261887,31.125992693422525"
declare -a DATES=("2016-08-26" "2017-07-25")

# EXECUTIONS
for model in "${MODELS[@]}"
do
	for date in "${DATES[@]}"
	do
		for grid_size in "${GRID_SIZES[@]}"
		do
			eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$date --model=$model --grid_size=$grid_size $EXTRA"
		done
	done
done

############################################################################################
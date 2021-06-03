#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ATTRIBUTES
declare -a DAYS_THRESHOLD=("180") # "90" "180" "365" "730"
declare -a REDUCTIONS=("median") # "median" "min"
declare -a MODELS=("rf" "svm" "lstm") # "rf" "svm" "mlp" "lstm"
declare EXTRA="--grid_size=7 --days_in=4 --days_out=5"

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE CHILIKA
NAME="chilika"
LAT_LON="85.08856074928927,19.698732779758075,85.28279700936078,19.546819354913833"
declare -a DATES=("2014-11-25")

# EXECUTIONS
for reduction in "${REDUCTIONS[@]}"
do
	for model in "${MODELS[@]}"
	do
		for date in "${DATES[@]}"
		do
			for day_threshold in "${DAYS_THRESHOLD[@]}"
			do
				eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$date --model=$model --days_threshold=$day_threshold --reduction=$reduction $EXTRA"
			done
		done
	done
done

############################################################################################



############################################################################################
## LAKE ERIE
NAME="erie"
LAT_LON="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826"
declare -a DATES=("2013-10-11" "2015-07-27")

# EXECUTIONS
for reduction in "${REDUCTIONS[@]}"
do
	for model in "${MODELS[@]}"
	do
		for date in "${DATES[@]}"
		do
			for day_threshold in "${DAYS_THRESHOLD[@]}"
			do
				eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$date --model=$model --days_threshold=$day_threshold --reduction=$reduction $EXTRA"
			done
		done
	done
done

############################################################################################



############################################################################################
## LAKE TAIHU
NAME="taihu"
LAT_LON="119.85091169141491,31.553377091606887,120.64485195719814,30.91800041328722"
declare -a DATES=("2016-08-26" "2017-07-25")
declare -a REDUCTIONS=("median" "min") # "median" "min"

# EXECUTIONS
for reduction in "${REDUCTIONS[@]}"
do
	for model in "${MODELS[@]}"
	do
		for date in "${DATES[@]}"
		do
			for day_threshold in "${DAYS_THRESHOLD[@]}"
			do
				eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$date --model=$model --days_threshold=$day_threshold --reduction=$reduction $EXTRA"
			done
		done
	done
done

############################################################################################
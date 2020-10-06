#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ARRAYS
declare -a MODELS=("rf" "ocsvm")
declare -a REDUCER=("" "--reducer")

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE ERIE

# ARGUMENTS
NAME="erie"
LAT_LON="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826"
FROM_DATE="2019-07-11"

# EXECUTIONS
for reducer in "${REDUCER[@]}"
do
	for model in "${MODELS[@]}"
	do
		for days_in in 5 10 15 30
		do
			for day_threshold in 1825 730 365
			do
				eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --days_in=$days_in --model=$model --days_threshold=$day_threshold $reducer"
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
FROM_DATE="2020-03-30"

# EXECUTIONS
for reducer in "${REDUCER[@]}"
do
	for model in "${MODELS[@]}"
	do
		for days_in in 5 10 15 30
		do
			for day_threshold in 1825 730 365
			do
				eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --days_in=$days_in --model=$model --days_threshold=$day_threshold $reducer"
			done
		done
	done
done

############################################################################################
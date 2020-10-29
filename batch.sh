#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ARRAYS
declare -a MODELS=("mlp") #"rf" "svm" "lstm" "mlp"
declare -a FILLS_MISSING=("time")
declare -a REDUCER=("" "--reducer")
declare -a CLASS_MODE=("" "--class_mode" "--class_mode --class_weight")
declare -a PROPAGATE=("" "--propagate")
declare -a NORMALIZE=("" "--non_normalized")

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE ERIE

# ARGUMENTS
NAME="erie"
LAT_LON="-83.48811946836814,41.85776095627803,-83.18290554014548,41.677617395337826"
FROM_DATE="2019-07-11"

# EXECUTIONS
for day_threshold in 90
do
	for model in "${MODELS[@]}"
	do
		for days_in in 3 5 7 10
		do
			for days_out in 3 5
			do
				for class_mode in "${CLASS_MODE[@]}"
				do
					for fill_missing in "${FILLS_MISSING[@]}"
					do
						for propagate in "${PROPAGATE[@]}"
						do
							for normalized in "${NORMALIZE[@]}"
							do
								for reducer in "${REDUCER[@]}"
								do
									eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --days_in=$days_in --days_out=$days_out --model=$model --days_threshold=$day_threshold --fill_missing=$fill_missing $reducer $class_mode $propagate $normalized"
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


# ############################################################################################
# ## LAKE CHILIKA

# # ARGUMENTS
# NAME="chilika"
# LAT_LON="85.08856074928927,19.698732779758075,85.28279700936078,19.546819354913833"
# FROM_DATE="2020-03-30"

# # EXECUTIONS
# for model in "${MODELS[@]}"
# do
# 	for fill_missing in "${FILLS_MISSING[@]}"
# 	do
# 		for reducer in "${REDUCER[@]}"
# 		do
# 			for days_in in 5 7 10
# 			do
# 				for day_threshold in 365 180
# 				do
# 					eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --days_in=$days_in --model=$model --days_threshold=$day_threshold --fill_missing=$fill_missing $reducer"
# 				done
# 			done
# 		done
# 	done
# done

# ############################################################################################
#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ARRAYS - FULL
declare -a DAYS_THRESHOLD=("180" "365" "730" "1825")
declare -a DAYS_IN_OUT=("--days_in=5 --days_out=5")
declare -a GRID_SIZE=("--grid_size=6" "--grid_size=9")
declare -a MODELS=("rf" "svm" "mlp" "lstm")
declare -a FILLS_MISSING=("--fill_missing=ffill" "--fill_missing=time" "--fill_missing=linear")
declare -a REDUCER=("--reducer --pca_size=0.950")
declare -a CLASS_MODE=("--class_mode" "--class_mode --class_weight")
declare -a NORMALIZE=("")
declare -a RS_TRAIN_SIZES=("--rs_train_size=0.01" "--rs_train_size=0.025" "--rs_train_size=0.05" "--rs_train_size=0.10")
declare -a RS_ITERS=("--rs_iter=25" "--rs_iter=50" "--rs_iter=100" "--rs_iter=500")

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

# ############################################################################################
# ## LAKE ERIE

# # ARGUMENTS
# NAME="erie"
# LAT_LON="-83.46168361631736,41.74451005963491,-83.39542232481345,41.69992268431667"
# FROM_DATE="2019-07-11"

# # EXECUTIONS
# for rs_train_size in "${RS_TRAIN_SIZES[@]}"
# do
# 	for rs_iter in "${RS_ITERS[@]}"
# 	do
# 		for class_mode in "${CLASS_MODE[@]}"
# 		do
# 			for day_threshold in "${DAYS_THRESHOLD[@]}"
# 			do
# 				for days_in_out in "${DAYS_IN_OUT[@]}"
# 				do
# 					for reducer in "${REDUCER[@]}"
# 					do
# 						for fill_missing in "${FILLS_MISSING[@]}"
# 						do
# 							for model in "${MODELS[@]}"
# 							do
# 								for propagate in "${PROPAGATE[@]}"
# 								do
# 									for normalized in "${NORMALIZE[@]}"
# 									do
# 										eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $days_in_out $fill_missing $reducer $class_mode $propagate $normalized $rs_train_size $rs_iter"
# 									done
# 								done
# 							done
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done

# ############################################################################################


############################################################################################
## LAKE CHILIKA

# ARGUMENTS
NAME="chilika"
LAT_LON="85.15749649545916,19.628963984868907,85.21105484506853,19.590154721044673"
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
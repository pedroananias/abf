#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ARRAYS - FULL
# declare -a MODELS=("rf" "svm" "lstm" "mlp")
# declare -a FILLS_MISSING=("time")
# declare -a REDUCER=("" "--reducer")
# declare -a CLASS_MODE=("" "--class_mode" "--class_mode --class_weight")
# declare -a PROPAGATE=("" "--propagate")
# declare -a NORMALIZE=("" "--non_normalized")
# declare -a DAYS_IN_OUT=("--days_in=3 --days_out=3" "--days_in=3 --days_out=5" "--days_in=5 --days_out=5" "--days_in=10 --days_out=3" "--days_in=10 --days_out=5")
# declare -a GS_TRAIN_SIZES=("--gs_train_size=0.01" "--gs_train_size=0.025" "--gs_train_size=0.05")

# ARRAYS - FILTERED
declare -a DAYS_THRESHOLD=("90" "180")
declare -a MODELS=("svm")
declare -a FILLS_MISSING=("time")
declare -a REDUCER=("--reducer")
declare -a CLASS_MODE=("--class_mode")
declare -a PROPAGATE=("")
declare -a NORMALIZE=("")
declare -a DAYS_IN_OUT=("--days_in=5 --days_out=5")
declare -a RS_TRAIN_SIZES=("--rs_train_size=0.01")
declare -a RS_ITERS=("--rs_iter=25" "--rs_iter=500")
declare -a PCA_SIZES=("--pca_size=0.99" "--pca_size=0.95" "--pca_size=0.90")

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
		for pca_size in "${PCA_SIZES[@]}"
		do
			for day_threshold in "${DAYS_THRESHOLD[@]}"
			do
				for model in "${MODELS[@]}"
				do
					for days_in_out in "${DAYS_IN_OUT[@]}"
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
											eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE $days_in_out --model=$model --days_threshold=$day_threshold --fill_missing=$fill_missing $reducer $class_mode $propagate $normalized $rs_train_size $rs_iter $pca_size"
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
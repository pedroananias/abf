#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"
CLEAR="sudo pkill -f /home/pedro/anaconda3"

# ARRAYS - FULL
declare -a DAYS_THRESHOLD=("60" "90" "180") # "90" "180" "365" "730"
declare -a GRID_SIZE=("--grid_size=7")
declare -a MODELS=("rf" "svm") # "rf" "svm" "mlp" "lstm"
declare -a FILLS_MISSING=("--fill_missing=time")
declare -a REDUCER=("--pca_size=0.900")
declare -a CLASS_MODE=("")
declare -a NORMALIZE=("")
declare -a RS_TRAIN_SIZES=("--rs_train_size=500.0")
declare -a RS_ITERS=("--rs_iter=500")
declare -a ATTR=("")
declare -a PROPAGATE=("")
declare -a SCALER=("--scaler=minmax")
declare -a REDUCTION=("--reduction=median" "--reduction=min")

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE TAIHU #1

# ARGUMENTS
NAME="taihu"
LAT_LON="119.85091169141491,31.553377091606887,120.64485195719814,30.91800041328722"
FROM_DATE="2016-08-25"

# EXECUTIONS
for days_in in {3..4}
do
	for days_out in {5..5}
	do
		for rs_train_size in "${RS_TRAIN_SIZES[@]}"
		do
			for rs_iter in "${RS_ITERS[@]}"
			do
				for class_mode in "${CLASS_MODE[@]}"
				do
					for day_threshold in "${DAYS_THRESHOLD[@]}"
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
													for attr in "${ATTR[@]}"
													do
                                                        for reduction in "${REDUCTION[@]}"
													    do
														    eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold $grid_size $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler $reduction"
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
	done
done
############################################################################################



############################################################################################
## LAKE TAIHU #2

# ARGUMENTS
NAME="taihu"
LAT_LON="119.85091169141491,31.553377091606887,120.64485195719814,30.91800041328722"
FROM_DATE="2017-07-24"

# EXECUTIONS
for days_in in {3..4}
do
	for days_out in {5..5}
	do
		for rs_train_size in "${RS_TRAIN_SIZES[@]}"
		do
			for rs_iter in "${RS_ITERS[@]}"
			do
				for class_mode in "${CLASS_MODE[@]}"
				do
					for day_threshold in "${DAYS_THRESHOLD[@]}"
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
													for attr in "${ATTR[@]}"
													do
                                                        for reduction in "${REDUCTION[@]}"
													    do
														    eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --name=$NAME --from_date=$FROM_DATE --model=$model --days_threshold=$day_threshold --days_in=$days_in --days_out=$days_out $grid_size $fill_missing $reducer $class_mode $normalized $rs_train_size $rs_iter $attr $propagate $scaler $reduction"
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
	done
done
############################################################################################
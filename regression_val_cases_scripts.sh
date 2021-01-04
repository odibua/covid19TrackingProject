#!/usr/bin/env bash
#python managers.py --mode perform_cases_multilinear_regression --state california --county alameda imperial riverside sacramento sanfrancisco sonoma --validate_state_name california --validate_county_names losangeles santaclara --ethnicity_list Black White Asian Hispanic --reg_key discrepancy

# Training regions list of lists
train_state="california"
train_case_counties_array=("None") #("alameda imperial riverside sacramento sanfrancisco sonoma" "None")
train_death_counties_array=("None") #("alameda imperial sanfrancisco santaclara" "None")

# Validation regions list of lists
validation_state="california"
validation_case_counties_array=("losangeles sacramento" "None") # "None")
validation_death_counties_array=("losangeles santaclara" "None") # "None")

# Regression type list
regression_type_array="gp" #"multilinear multilinear_ridge multilinear_lasso" #

# Regression key list
regression_key_str="--reg_key discrepancy" #"--reg_key discrepancy" #"--reg_key mortality_rate"

# Mode list
mode_array="perform_cases_multilinear_regression perform_deaths_multilinear_regression"

train_state_str="--state ${train_state}"
val_state_str="--validate_state_name ${validation_state}"

for mode in $mode_array
do
    mode_str="--mode ${mode}"
    for regression_type in $regression_type_array
    do
        regression_type_str="--regression_type ${regression_type}"
        echo ${regression_type_str}
        for train_case_counties in "${train_case_counties_array[@]}"
        do
            trn_case_county="--county"
            for case_county in $train_case_counties
            do
               trn_case_county="${trn_case_county} ${case_county}"
            done

            for validate_case_counties in "${validation_case_counties_array[@]}"
            do
                val_case_counties="--validate_county_names"
                for val_case_county in $validate_case_counties
                do
                    val_case_counties="${val_case_counties} ${val_case_county}"
                done
                echo "python managers.py ${mode_str} ${train_state_str} ${trn_case_county} ${val_state_str} ${val_case_counties} ${regression_type_str} ${regression_key_str} --ethnicity_list Black White Asian Hispanic"
                python managers.py ${mode_str} ${train_state_str} ${trn_case_county} ${val_state_str} ${val_case_counties} ${regression_type_str} ${regression_key_str} --ethnicity_list Black White Asian Hispanic
                BACK_PID=$!
                while kill -0 $BACK_PID ; do
                    echo "Process is still active..."
                    sleep 1
                    # You can add a timeout here if you want
                done
            done
        done
    done
done
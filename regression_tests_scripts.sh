#!/usr/bin/env bash

#array="Vietnam Germany Argentina"
#array2=("Asia" "Europe" "America")
#i=0
#for arr in $array; do
#    printf "%s is in %s\n" "${arr} is in ${array2[i]}"
#    i=$(( $i + 1 ))
#done

# Training regions list of lists
train_state="california"
train_case_counties_array=("None") #("alameda imperial riverside sacramento sanfrancisco sonoma" "None")
train_death_counties_array=("None") #("alameda imperial sanfrancisco santaclara" "None")

# Validation regions list of lists
validation_state="california"
validation_case_counties_array=("None") #("losangeles santaclara" "None")
validation_death_counties_array=("None") #("losangeles sacramento" "None")

# Test regions list
test_state="california"
test_case_counties_array=("sonoma riverside sacramento alameda losangeles sanfrancisco santaclara") #("None" "losangeles santaclara")
test_death_counties_array=("sacramento alameda losangeles sanfrancisco santaclara") #("None" "losangeles sacramento")

# Regression type list
regression_type_array="gp" #"multilinear_lasso"

# Regression key list
regression_key_str="--reg_key discrepancy" #"--reg_key discrepancy"

# Mode list
mode_array="test_cases_model test_deaths_model"

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

            test_idx=0
            for validate_case_counties in "${validation_case_counties_array[@]}"
            do
                val_case_counties="--validate_county_names"
                for val_case_county in $validate_case_counties
                do
                    val_case_counties="${val_case_counties} ${val_case_county}"
                done
                test_state_str="--test_state_name california"
                if [[ ${mode} = "test_cases_model" ]]
                then
                    test_county_str="--test_county_names ${test_case_counties_array[test_idx]}"
                fi
                if [[ ${mode} = "test_deaths_model" ]]
                then
                    test_county_str="--test_county_names ${test_death_counties_array[test_idx]}"
                fi
                echo "python managers.py ${mode_str} ${train_state_str} ${trn_case_county} ${val_state_str} ${val_case_counties} ${test_state_str} ${test_county_str} ${regression_type_str} ${regression_key_str} --ethnicity_list Black White Asian Hispanic"
                python managers.py ${mode_str} ${train_state_str} ${trn_case_county} ${val_state_str} ${val_case_counties} ${test_state_str} ${test_county_str} ${regression_type_str} ${regression_key_str} --ethnicity_list Black White Asian Hispanic
                BACK_PID=$!
                while kill -0 $BACK_PID ; do
                    echo "Process is still active..."
                    sleep 1
                    # You can add a timeout here if you want
                done
                test_idx=$(( $test_idx + 1 ))
            done
        done
    done
done
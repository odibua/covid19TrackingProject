#!/usr/bin/env bash
#python managers.py --mode perform_cases_multilinear_regression --state california --county alameda imperial riverside sacramento sanfrancisco sonoma --validate_state_name california --validate_county_names losangeles santaclara --ethnicity_list Black White Asian Hispanic --reg_key discrepancy

# Training regions list of lists
train_state="california"
train_case_counties_array=("alameda losangeles sacramento sanfrancisco riverside sonoma" "santaclara losangeles sacramento sanfrancisco riverside sonoma" "alameda santaclara sacramento sanfrancisco riverside sonoma" "alameda losangeles santaclara sanfrancisco riverside sonoma" "alameda losangeles sacramento santaclara riverside sonoma" "alameda losangeles sacramento sanfrancisco santaclara sonoma" "alameda losangeles sacramento sanfrancisco riverside santaclara") # "alameda losangeles sacramento sanfrancisco")
train_death_counties_array=("alameda sacramento sanfrancisco losangeles" "santaclara sacramento sanfrancisco losangeles" "alameda sacramento losangeles santaclara" "alameda santaclara sanfrancisco losangeles" "alameda sacramento sanfrancisco santaclara") # "santaclara sanfrancisco losangeles")

# Validation regions list of lists
validation_state="california"
validation_case_counties_array=("santaclara" "alameda" "losangeles" "sacramento" "sanfrancisco" "riverside" "sonoma") # "None") # "None")
validation_death_counties_array=("santaclara" "alameda" "sanfrancisco" "sacramento" "losangeles") # "None") # "None")

# Regression type list
regression_type_array="rf_reg" #"multilinear multilinear_ridge multilinear_lasso" #

# Regression key list
regression_key_str="--reg_key mortality_rate" #"--reg_key discrepancy" #"--reg_key mortality_rate"

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
        echo ${mode}
        idx=0
        for train_death_counties in "${train_death_counties_array[@]}"
        do
            echo "${train_death_counties}"
            validation_death_counties=${validation_death_counties_array[idx]}
            idx=${idx}+1
            if [ "${mode}" == 'perform_cases_multilinear_regression' ]
            then
                echo "HERE"
                    train_counties=${train_case_counties_array[idx]}
                    validate_counties=${validation_case_counties_array[idx]}
            else
                echo "HERE2"
                    train_counties=${train_death_counties_array[idx]} #${train_death_counties} #${train_death_counties_array[0]}
                    validate_counties=${validation_death_counties_array[idx]} #${validation_death_counties} #${validation_death_counties_array[0]}
            fi

            trn_county="--county"
            for county in $train_counties
            do
               trn_county="${trn_county} ${county}"
            done

            val_counties="--validate_county_names"
            for val_county in $validate_counties
            do
                val_counties="${val_counties} ${val_county}"
            done
            echo "python managers.py ${mode_str} ${train_state_str} ${trn_county} ${val_state_str} ${val_counties} ${regression_type_str} ${regression_key_str} --ethnicity_list Black White Asian Hispanic"
            python managers.py ${mode_str} ${train_state_str} ${trn_county} ${val_state_str} ${val_counties} ${regression_type_str} ${regression_key_str} --ethnicity_list Black White Asian Hispanic
            BACK_PID=$!
            while kill -0 $BACK_PID ; do
                echo "Process is still active..."
                sleep 1
                # You can add a timeout here if you want
            done
        done
    done
done
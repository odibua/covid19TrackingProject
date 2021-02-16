#!/usr/bin/env bash
# Training regions list of lists
train_state="california"
train_case_counties_array=("None") #("alameda losangeles sacramento sanfrancisco riverside sonoma santaclara")
train_death_counties_array=("None") #("alameda sacramento sanfrancisco losangeles santaclara")

# Regression type list
corr_type_array="spearman" #"multilinear multilinear_ridge multilinear_lasso" #

# Regression key list
corr_key_str="--corr_key mortality_rate" #"--reg_key discrepancy" #"--reg_key mortality_rate"

# Mode list
mode_array="perform_death_spearman_corr"

train_state_str="--state ${train_state}"

for mode in $mode_array
do
    mode_str="--mode ${mode}"
    for corr_type in $corr_type_array
    do
        corr_type_str="--corr_type ${corr_type}"
        echo ${corr_type_str}
        echo ${mode}
        idx=0
        for train_death_counties in "${train_death_counties_array[@]}"
        do
            echo "${train_death_counties}"
            if [ "${mode}" == 'perform_case_spearman_corr' ]
            then
                echo "HERE"
                    train_counties=${train_case_counties_array[0]}
            else
                echo "HERE2"
                    train_counties=${train_death_counties_array[0]} #${train_death_counties_array[0]}
            fi

            trn_county="--county"
            for county in $train_counties
            do
               trn_county="${trn_county} ${county}"
            done

            echo "python managers.py ${mode_str} ${train_state_str} ${trn_county} ${corr_type_str} ${corr_key_str} --ethnicity_list Black White Asian Hispanic"
            python managers.py ${mode_str} ${train_state_str} ${trn_county}  ${corr_type_str} ${corr_key_str} --ethnicity_list Black White Asian Hispanic
            BACK_PID=$!
            while kill -0 $BACK_PID ; do
                echo "Process is still active..."
                sleep 1
                # You can add a timeout here if you want
            done
        done
    done
done
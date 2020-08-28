# Tracking COVID19 by Ethnicity

**"Never let a good crisis go to waste"**

The purpose of his repository is to enable the collection of real-time data
regarding covid cases/deaths from different counties and states, stratified by race.
In it's most ideal form this repository will:

1. contain real-time data about covid cases/deaths by ethnicity
2. contain this daa for different states/counties

This repository of data contained above has the potential to motivate future health/pandemic 
policies, quantify disparities in health outcomes, and enable research into what factors
exacerbate/alleviate health disparities.


## Overview of README
#### [I. Packages Used](#packages-used)
#### [II. Getting Started](#getting-started)
#### [III. Overview of Code](#overview-of-code)
#### [IV. Adding Regions for Scraping Raw Data](#adding-regions-for-scraping-raw-data)
#### [V. Running Scraping Locally](#running-scraping-locally)
#### [VI. Handling Scraping Errors](#handling-scraping-errors)
#### [VII. Processing Raw Data](#processing-raw-data)
#### [VIII. Processing Raw Data Locally](#processing-raw-data-locally)
#### [IX. Handling Processing Errors](#handling-processing-errors)


#### Packages Used
The main packages used can be found in ```setup.py```. These are

```
autopep8
bokeh
beautifulsoup4
html5lib
numpy>=1.9.0
pandas
pytest
requests
requests_html
scipy
setuptools>=40.0
typing
urllib3>1.25
wheel
PyYAML
```

## Getting Started
1. Install git on your system, following these [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
1. Add ssh key to git, following the instructions [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
1. Clone the covid19Tracking repository using ```git clone git@github.com:odibua/covid19TrackingProject.git```
1. Install python on your system by following these [instructions](https://wiki.python.org/moin/BeginnersGuide/Download)
1. Install pip on your system by following these [instructions](https://pip.pypa.io/en/stable/installing/)
1. Navigate to ```covid19Tracking/``` and run ```pip install -e .```

## Overview of Code
The below diagram provides a schematic overview of this repository. The description of this schematic will make 
reference to the directory structure at the end of this section.

![Test Image](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/overview_code.png)

#####Scrape Manager
For a given state and (optionally) county, our scrape manager:

1. Uses a config saved in the configs directory to request the raw data containing case and (if applicable)
death counts stratified by ethnicity from a website associated with that state and/or county. 
1. Saves this requested data as file(s) in the relevant sub-directory e.g. 
    ```{STATE}/{COUNTY}/raw_data/{DATE}```

**NOTE: Details on how to add a new state/county for processing by the
scraper manager are described [here](#adding-regions-for-scraping-raw-data).**

#####Raw Data Parsers
The raw data parsing is split into two managers. One for parsing cases from raw data,
and one for parsing deaths from raw data. For a given state and (optionally) county:

1. Iterates through the raw data saved for a particular state and county.
1. Parses the case/death counts stratified by ethnicity based on count.
1. Calculates a disparity ratio for each ethnicity. The disparity ratio
is defined as the ratio of the proportion of cases/deaths of a particular
ethnicity and the proportion of that ethnicity in a state/county. 
1. Saves these results to the csvs directory of the state e.g.
    ```{STATE}/csvs/{STATE}_{COUNTY}_ethnicity_cases.csv``` or 
    ```{STATE}/csvs/{STATE}_{COUNTY}_ethnicity_deaths.csv```

**NOTE: Details on how to add a new state/county for parsing by the
case/death parsers are described [here](#processing-raw-data).**

    covid19Tracking/
    |   managers.py
    |   utils.py
    |   setup.py
    |   conftest.py
    |---states/
        |   data_projectors.py
        |   utils.py
        |   states_config.yaml
        |---california/
            |   california_projector.py
            |   test_california_scrape_project.py
            |---configs/
            |       california_all.yaml
            |       california_all_html_parse.yaml
            |       projector_exceptions.yaml
            |---csvs/
            |---raw_data/
            |       2020-06-14/
            |       ...
            |---counties/
                |---alameda/
                |   |   alameda_projector.py
                |   |   test_california_alameda_scrape_project.py
                |   |---configs/
                |   |       alameda_cases.yaml
                |   |       alameda_cases_json_parser.yaml
                |   |       alameda_deaths.yaml
                |   |       alameda_deaths_json_parser.yaml
                |   |       projector_exceptions.yaml
                |   |---raw_data/
                |---...
                .
                .
                .
## Adding Regions for Scraping Raw Data
## Running Scraping Locally
## Handling Scraping Errors
## Processing Raw Data
## Processing Raw Data Locally
## Handling Processing Errors
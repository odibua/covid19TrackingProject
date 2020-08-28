# Tracking COVID19 by Ethnicity

**"Never let a good crisis go to waste"**

The purpose of his repository is to enable the semi-automated collection of real-time data
regarding covid cases/deaths from different counties and states, stratified by race. The
semi-automation is enabled through a combination of Circle CI and pytest.
In it's most ideal form this repository will:

1. contain real-time data about covid cases/deaths by ethnicity
2. contain this daa for different states/counties

This repository of data contained above has the potential to motivate future health/pandemic 
policies, quantify disparities in health outcomes, and enable research into what factors
exacerbate/alleviate health disparities.


## Overview of README
#### [I. Packages Used](#packages-used)
#### [II. Getting Started](#getting-started)
#### [III. Overview of Code](#overview-of-codlone)
#### [IV. Adding Regions for Scraping Raw Data](#adding-regions-for-scraping-raw-data)
#### [V. Configuring Scraping Schedule](#configuring-scraping-schedule)
#### [VI. Running Scraping Locally](#running-scraping-locally)
#### [VII. Handling Scraping Errors](#handling-scraping-errors)
#### [VIII. Processing Raw Data](#processing-raw-data)
#### [IX. Processing Raw Data Locally](#processing-raw-data-locally)
#### [X. Handling Processing Errors](#handling-processing-errors)


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
1. Sign up for [circleci](https://circleci.com/signup/), a continous integration tool
1. Checkout a branch with a meaningful name for adding regions from which to scrape raw data

## Overview of Code
The below diagram provides a schematic overview of this repository. The description of this schematic will make 
reference to the directory structure at the end of this section.

![Test Image](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/overview_code.png)

#### Scrape Manager
For a given state and (optionally) county, our scrape manager:

1. Uses a config saved in the configs directory to request the raw data containing case and (if applicable)
death counts stratified by ethnicity from a website associated with that state and/or county. 
1. Saves this requested data as file(s) in the relevant sub-directory e.g. 
    ```{STATE}/{COUNTY}/raw_data/{DATE}```

**NOTE: Details on how to add a new state/county for processing by the
scraper manager are described [here](#adding-regions-for-scraping-raw-data).**

#### Raw Data Parsers
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
Add regions for scraping raw data involves two main steps. The first is adding
a config file that will be used to get the raw data for the state/county of interest,
and the second is creating a ``test_{STATE}_{COUNTY}_scrape_project.py``. The specifics 
of these steps are as follows:

1. Create a directory of your region of interest. E.G. if you are going to scrape the
state of California create the directory ``states/california/``. If you are going to
scrape the state of California and the county of Alameda create the directory ``states/california/counties/alameda/``

1. Within the region directory, create a configs sub-directory. E.G. ``states/california/configs/`` 

1. Add a config file in the created configs sub-directory (**examples are given
[here](#examples-of-config-files)**). The fields of this config
are ``NAME, DATA_TYPE, REQUEST, WEBSITE``. 
    - The ``NAME`` and ``DATA_TYPE`` fields are used determine the name of the file
      containing scraped raw data
    - The ``WEBSITE`` field states the website from which the raw data will be obtained.
      It will be useful for handling scraping errors.
    - The ```REQUEST``` field is used by the scrape manager to obtain the relevant raw data 
      from the added region. The```REQUEST``` field either contains paramters for a ``POST``
      or ``GET`` type of request.

### Examples of Config Files
Here we will give an example of two configs for scraping raw data. One
will be for a website based on a `GET` request and one for a `POST` request.

#### GET Request Example
For the state of California, the information on cases/death counts 
stratified by ethnicity are stored on an html page that we can obtain
using a simple ``GET`` request. The associated config is ``california_all.yaml``
and has the below fields:

```
NAME: California
DATA_TYPE: All

REQUEST:
  TYPE: GET
  URL: https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/Race-Ethnicity.aspx
  HEADERS:
    User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0
  PAYLOAD: None

WEBSITE: https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/Race-Ethnicity.aspx
```

#### POST Request Example

## Configuring Scraping Schedule
## Running Scraping Locally
## Handling Scraping Errors
## Processing Raw Data
## Processing Raw Data Locally
## Handling Processing Errors
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

To lint code type `python setup.py pylint` in terminal
## Overview of Code
The below diagram provides a schematic overview of this repository. The description of this schematic will make 
reference to the directory structure at the end of this section.

![Overview Image](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/overview_code.png)

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

1. Add a config file in the created configs sub-directory (**examples of the config
 are given [here](#examples-of-config-files)**). The fields of this config
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
![California Image](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/california_example.png)

For the state of California, the information on cases/death counts 
stratified by ethnicity are stored on an html page, shown above. We can obtain
this raw data using a simple ``GET`` request. The associated config is ``california_all.yaml``
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
![Santa Clara Image](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/santaclara_dashboard.png)

Many websites display information about COVID19 on dash boards. These are generally not
amenable to simple `GET` requests, and often times require `POST` requests. They 
also generally require multiple config files. Santa Clara is an example of a 
particularly sticky case.

For Santa Clara, their dashboard shows cases and deaths stratified by ethnicity
as percentages. It also shows the total cases/deaths. To get the relevant case
and death counts, we need all of this data. And, by proxy a config file for each
value. An example of this is the `santaclara_cases.yaml` config file.

```
NAME: SantaClara
DATA_TYPE: Cases
IFRAME: https://app.powerbigov.us/view?r=eyJrIjoiYzBiYTA0YTAtOWYyYS00NzExLTk2ZjAtOGMxOWQ4YzhlODgwIiwidCI6IjBhYzMyMDJmLWMzZTktNGY1Ni04MzBkLTAxN2QwOWQxNmIzZiJ9

REQUEST:
  TYPE: POST
  URL: https://wabi-us-gov-virginia-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true
  HEADERS:
    Accept: application/json, text/plain, */*
    Accept-Encoding: gzip, deflate, br
    Accept-Language: en-US,en;q=0.9
    ActivityId: 3f2feede-de49-52a2-2e2f-62e2f1118259
    Connection: keep-alive
    Content-Length: '2945'
    Content-Type: application/json;charset=UTF-8
    Host: wabi-us-gov-virginia-api.analysis.usgovcloudapi.net
    Origin: https://app.powerbigov.us
    Referer: https://app.powerbigov.us/view?r=eyJrIjoiYzBiYTA0YTAtOWYyYS00NzExLTk2ZjAtOGMxOWQ4YzhlODgwIiwidCI6IjBhYzMyMDJmLWMzZTktNGY1Ni04MzBkLTAxN2QwOWQxNmIzZiJ9
    RequestId: e369f8db-ecfd-d1d3-f1b3-73a9498b84e7
    Sec-Fetch-Dest: empty
    Sec-Fetch-Mode: cors
    Sec-Fetch-Site: cross-site
    User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36
    X-PowerBI-ResourceKey: c0ba04a0-9f2a-4711-96f0-8c19d8c8e880
  PAYLOAD: {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"c","Entity":"cases_race","Type":0}],"Select":[{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Race_eth"},"Name":"cases_race.Race_eth"},{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Count"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Count"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"Sum(cases_race.Count)"},{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_pop"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_pop"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"Sum(cases_race.Percent_pop)"}],"OrderBy":[{"Direction":1,"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Race_eth"}}}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0,1,2]}]},"DataReduction":{"DataVolume":4,"Primary":{"Window":{"Count":1000}}},"Version":1},"ExecutionMetricsKind":3}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"c\",\"Entity\":\"cases_race\",\"Type\":0}],\"Select\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Race_eth\"},\"Name\":\"cases_race.Race_eth\"},{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Count\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Count\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"Sum(cases_race.Count)\"},{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_pop\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_pop\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"Sum(cases_race.Percent_pop)\"}],\"OrderBy\":[{\"Direction\":1,\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Race_eth\"}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1,2]}]},\"DataReduction\":{\"DataVolume\":4,\"Primary\":{\"Window\":{\"Count\":1000}}},\"Version\":1},\"ExecutionMetricsKind\":3}}]}","QueryId":"","ApplicationContext":{"DatasetId":"9f953fbe-cd3f-4764-be79-e8d95223222f","Sources":[{"ReportId":"bb6481c0-2521-4a1e-8db6-e886e81e81c7"}]}}],"cancelQueries":[],"modelId":344052}

WEBSITE: https://www.sccgov.org/sites/covid19/Pages/dashboard-demographics-of-cases-and-deaths.aspx
```
 
### Populating Request Field in Configs
Populating the request field requires the use of network developer tools. 

For tables on simple html pages, a `GET` request should suffice, and the config should be filled out
like the `california_all.yaml` config above. The only fields that need to be changed are
the `URL` and `WEBSITE` fields.

Pages with dashboards are more complicated. Here, we will walk through an example of 
figuring out how to properly populate a config file with `POST` requests based on
the Santa Clara [website](https://www.sccgov.org/sites/covid19/Pages/dashboard-demographics-of-cases-and-deaths.aspx)

1. Click inspect near the dash board
![step1_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step1_inspect.png)



## Configuring Scraping Schedule
## Running Scraping Locally
## Handling Scraping Errors
## Processing Raw Data
## Processing Raw Data Locally
## Handling Processing Errors
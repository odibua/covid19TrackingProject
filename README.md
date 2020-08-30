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
1. Create a branch with a meaningful name for adding regions from which to scrape raw data i.e `git checkout -b {NAME}`

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

1. Create in the `states/{STATE}` or `states/{STATE}/counties/{COUNTY}`, a `test_{STATE}_{COUNTY}_scrape_project.py`. 
Examples of this type of file can be found for [california](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/test_california_scrape_project.py)
and for [alameda](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/santaclara/test_california_santaclara_scrape_project.py).
An example of the Alameda file is below. For your particular region modify `self.state_name` and `self.county_name`. If you are just
scraping from a state, set `self.county_name = None`.

```
# --------------------------
# Standard Python Imports
# --------------------------
import pytest
import unittest

# --------------------------
# Third Party Imports
# --------------------------

# --------------------------
# covid19Tracking Imports
# --------------------------
from managers import add_commit_and_push, scrape_manager, case_parser_manager, death_parser_manager


@pytest.mark.usefixtures("project_bools")
class TestCaliforniaAlamedaScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'alameda'
        self.state_county_dir = f"states/{self.state_name}/counties/{self.county_name}/raw_data/"

    def test_scrape_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                scrape_manager(state_name=self.state_name)
                add_commit_and_push(state_county_dir=self.state_county_dir)

    def test_raw_to_ethnicity_case_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                if self.project_case_bool:
                    case_parser_manager(state_name=self.state_name)

    def test_raw_to_ethnicity_death_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                if self.project_death_bool:
                    death_parser_manager(state_name=self.state_name)
```
### Examples of Config Files
Here we will give an example of two configs for scraping raw data. One
will be for a website based on a `GET` request and one for a `POST` request.

#### GET Request Example
![California Image](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/california_example.png)

For the state of California, the information on cases/death counts 
stratified by ethnicity are stored on an html page, shown above. We can obtain
this raw data using a simple ``GET`` request. The associated config is [california_all.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/configs/california_all.yaml)
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
value. An example of this is the [santaclara_cases.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/santaclara/configs/santaclara_cases.yaml) config file.

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

1. Click inspect near the dashboard
![step1_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step1_inspect.png) 

1. Select the element in the right tab that highlights the dashboard. Click the div elements until the
url for the dashboard is visible
![step2_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step2_select_dashboard.png)


1. Copy the url and navigate to it. This should result in a webpage with just the dash board
![step3_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step3_copy_link_address.png)

1. Click the network tab. If the name column is empty, reload the url
![step4_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step4_click_network_tab.png)

1. Click the preview tab and click through all the queries. For each query, search the json in the preview tab.
Stop when a dictionary that displays ethnicity case/death counts is found
![step5_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step5_search_query_preview.png)

1. Click the headers tab. 
    - If the `Request Method` is `GET` then copy and paste the `Request URL` to the `URL` field 
in the config (as in `california_all.yaml`) and make `TYPE` `GET`. 
    - If it is `POST` make the `TYPE` `POST`, and copy the `Request Headers` to the `HEADERS` field in the config
as in `santaclara_cases.yaml`. Make the `Content-Length` subfield a string.
![step6_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step6_click_headers_tab.png)

1. Click the `view source` button next to `Request Payload` and copy and paste the resulting json to the `PAYLOAD` 
field in the config, as in `sanaclara_cases.yaml`.
![step7_post](https://github.com/odibua/covid19TrackingProject/blob/odibua/README/images/step7_view_source_payload.png)

## Configuring Scraping Schedule
Scraping is run periodically using Circle CI. The configuration for this periodic run is stored in
[.circleci/config.yml](https://github.com/odibua/covid19TrackingProject/blob/master/.circleci/config.yml). The
relevant fields are replicated below.

```
      - run:
          name: run tests
          command: |
            python -m venv venv
            . venv/bin/activate
            git config user.email odibua@gmail.com
            git config user.name odibua
            python -m pytest
      - store_artifacts:
          path: test-reports
          destination: test-reports

workflows:
  build-and-test:
    jobs:
      - build-and-run-scrape
    triggers:
      - schedule:
          cron: "30 15 * * *"
          filters:
            branches:
              only:
                - master

```
In order to have Circle CI run scraping on a particular state and/or county regularly three modifications must be made.

1. Under the run section modify `git config user.email odibua@gmail.com` and `git config user.name odibua` 
   to make use of your own email and user name.
1. Under the run section modify `python -m pytest` to `python -m pytest --state={STATE}` or `python -m pytest --state={STATE} --county={COUNTY}`
   in order to have periodic scraping occur for particular states and/or counties.
1. Modify the `cron` subsection under `schedule`. The [crontab guru](https://crontab.guru/#20_21_*_*_*) shows what 
different inputs to this subsection mean. In the above case, `30 15 * * *`, means scraping will run at 15:30 UTC every day.
We note that UTC is the only time zone used in Circle CI.

## Running Scraping Locally
Scraping is run locally through pytest. It can be run in three ways.
    - `python -m pytest` scrapes, commits, and pushes raw data for every state/county with a `test_{STATE}_{COUNTY}_scrape_project.py`
       file
    - `python -m pytest --state={STATE}` does this for a particular state
    - `python -m pytest --state={STATE} --county={COUNTY}` does this for a particular state and county


## Handling Scraping Errors
## Processing Raw Data
## Processing Raw Data Locally
## Handling Processing Errors
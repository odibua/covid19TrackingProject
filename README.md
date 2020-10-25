# Tracking COVID19 by Ethnicity
TODO:
   - Make data structure for projectors to be filled in (simple interface)
   - Tableau python parser
   - Crawl through states directory (or auto-generate pytests)
**"Never let a good crisis go to waste"**

The purpose of his repository is to enable the semi-automated collection of time-series data
regarding covid cases/deaths from different counties and states, stratified by race. The
semi-automation is enabled through a combination of Circle CI and pytest.
In it's most ideal form this repository will:

1. contain time series raw data about covid cases/deaths by ethnicity
2. contain this data for a variety of different states/counties

This repository has the potential to motivate future health/pandemic 
policies, quantify disparities in health outcomes, and enable research into what factors
exacerbate/alleviate health disparities.


## Overview of README
#### [I. Data for Researchers](#data-for-researchers)
#### [II. Packages Used](#packages-used)
#### [III. Getting Started](#getting-started)
#### [IV. Overview of Code](#overview-of-codlone)
#### [V. Adding Regions for Scraping Raw Data](#adding-regions-for-scraping-raw-data)
#### [VI. Configuring Scraping Schedule](#configuring-scraping-schedule)
#### [VII. Running Scraping Locally](#running-scraping-locally)
#### [VIII. Handling Scraping Errors](#handling-scraping-errors)
#### [IX. Adding Regions for Processing Raw Data](#adding-regions-for-processing-raw-data)
#### [X. Processing Raw Data](#processing-raw-data)
#### [XI. Handling Processing Errors](#handling-processing-errors)
#### [XII. Where to Scrape More Data From](#where-to-scrape-more-data-from)
 
## Data for Researchers
For researchers who would like to directly use data, the relevant csvs are contained in directories of 
form `states/{STATE}/csvs/`. Each of these have csvs containing state/county information with names 
formatted as `{STATE}_{COUNTY}_ethnicity_cases.csv`. The directory for California is located 
[here](https://github.com/odibua/covid19TrackingProject/tree/master/states/california/csvs)

Each csv has columns of data containing the case/death count of each ethnicity, the date of that count, 
and a disparity ratio.
 
**Disparity Ratio Definition** 
  - percentage of total cases or deaths represented by an ethnicity/ percentage of total population of an ethnicity in
    a particular region
  - A disparity ratio of 2 for cases would indicate that an ethnicity has cases at twice the rate one
    would expect based on their population

**NOTES OF INTEREST**
1. In many counties/states their are people whose ethnicities are not known. Those are
ignored in the data set

1. While the raw data for each region are updated automatically, the csvs are updated manually on a periodic basis.
   If the dates in your csv of interest are not up to date, check the raw data for that county. If there is more recent
   raw data, process this data by following the instructions [here](#handling-processing-errors)

#### Packages Used
The main packages used can be found in [setup.py](https://github.com/odibua/covid19TrackingProject/blob/master/setup.py). 
They are listed below.

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
1. Install git on your system, following these [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
1. Add ssh key to git, following the instructions [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
1. Clone the covid19Tracking repository using 
    ```
    git clone git@github.com:odibua/covid19TrackingProject.git```
1. Install python on your system by following these [instructions](https://wiki.python.org/moin/BeginnersGuide/Download).
1. Install pip and a create a virtual environment on your system by following these [instructions](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
1. Navigate to the ```covid19Tracking``` directory, if not already there, and activate
   the virtual environment, following these [instructions](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
   **NOTE: Activate the virtual environment every time you work with this code**
1. Create a branch with a meaningful name for adding regions from which to scrape and process raw data i.e `git checkout -b {NAME}`
   This branch should be used to insure the scraping of raw data from a particular region is operating properly.
1. Sign up for [circleci](https://circleci.com/signup/)
1. Authorize github on circleci and create either a user key or read/write deploy key that will allow circleci to
   push commits to the repository. [Instructions](https://support.circleci.com/hc/en-us/articles/360018860473-How-to-push-a-commit-back-to-the-same-repository-as-part-of-the-CircleCI-job)

**To lint code type `python setup.py pylint` in terminal.**

## Overview of Code
The below diagram provides a schematic overview of this repository. The description of this schematic will make 
reference to the directory structure at the end of this section.

![Overview Image](https://github.com/odibua/covid19TrackingProject/blob/master/images/overview_code.png)

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
and one for parsing deaths from raw data. For a given state and (optionally) county, the parsers:

1. Iterate through the raw data saved for a particular state and county.
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
      containing scraped raw data.
    - The ``WEBSITE`` field states the website from which the raw data will be obtained.
      It will be useful for handling scraping errors.
    - The ```REQUEST``` field is used by the scrape manager to obtain the relevant raw data 
      from the added region. The```REQUEST``` field either contains paramters for a ``POST``
      or ``GET`` type of request.

1. Create in the `states/{STATE}` or `states/{STATE}/counties/{COUNTY}`, a `test_{STATE}_{COUNTY}_scrape_project.py`. 
Examples of this type of file can be found for [california](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/test_california_scrape_project.py)
and for [alameda](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/santaclara/test_california_santaclara_scrape_project.py).
An example of the Alameda file is below. For your particular region modify `self.state_name` and `self.county_name`. If you are just
scraping from a state, set `self.county_name = None`. Also modify the name of the class to reflect the region.

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
![California Image](https://github.com/odibua/covid19TrackingProject/blob/master/images/california_example.png)

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
![Santa Clara Image](https://github.com/odibua/covid19TrackingProject/blob/master/images/santaclara_dashboard.png)

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

1. Click inspect near the dashboard.
![step1_post](https://github.com/odibua/covid19TrackingProject/blob/master/images/step1_inspect.png) 

1. Select the element in the right tab that highlights the dashboard. Click the div elements until the
url for the dashboard is visible.
![step2_post](https://github.com/odibua/covid19TrackingProject/blob/master/images/step2_select_dashboard.png)


1. Copy the url and navigate to it. This should result in a webpage with just the dashboard.
![step3_post](https://github.com/odibua/covid19TrackingProject/blob/master/images/step3_copy_link_address.png)

1. Click the network tab. If the name column is empty, reload the url.
![step4_post](https://github.com/odibua/covid19TrackingProject/blob/master/images/step4_click_network_tab.png)

1. Click the preview tab and click through all the queries. For each query, search the json in the preview tab.
Stop when a dictionary that displays ethnicity case/death counts is found.
![step5_post](https://github.com/odibua/covid19TrackingProject/blob/master/images/step5_search_query_preview.png)

1. Click the headers tab. 
    - If the `Request Method` is `GET` then copy and paste the `Request URL` to the `URL` field 
in the config as in the [california_all.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/configs/california_all.yaml) and make `TYPE` `GET`. 
    - If it is `POST` make the `TYPE` `POST`, and copy the `Request Headers` to the `HEADERS` field in the config
as in the [santaclara_cases.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/santaclara/configs/santaclara_cases.yaml). Make the `Content-Length` subfield a string.
![step6_post](https://github.com/odibua/covid19TrackingProject/blob/master/images/step6_click_headers_tab.png)

1. Click the `view source` button next to `Request Payload` and copy and paste the resulting json to the `PAYLOAD` 
field in the config, as in the as in the [santaclara_cases.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/santaclara/configs/santaclara_cases.yaml). Make the `Content-Length` subfield a string.
![step7_post](https://github.com/odibua/covid19TrackingProject/blob/master/images/step7_view_source_payload.png)

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
- `python -m pytest` scrapes, commits, and pushes raw data for every state/county with a `test_{STATE}_{COUNTY}_scrape_project.py` file
- `python -m pytest --state={STATE}` does this for a particular state and every county in the state
- `python -m pytest --state={STATE} --county={COUNTY}` does this for a particular state and county


## Handling Scraping Errors
The API of websites tend to change. There are two sources of errors.

1. The API of the website being scraped has changed, resulting in request errors
1. The API of the website has changed, and there are no request errors, but you are scraping incorrect data.

The automatic scraping will always pick up on the first error, but the second error can only be found by semi-periodically
inspecting your raw data to make sure that something meaningful is being scraped. Regardless of the source of error, the
fix is the same. Re-populate the relevant scraping config file. Instructions on how to do so are located [here](#populating-request-field-in-configs).

## Adding Regions for Processing Raw Data
To make full use of raw data, it is necessary to process it into coherent numbers. The raw data is 
processed into:
- Case/Death counts straified by ethnicity.
- Disparity ratio stratified by ethnicity, where disparity ratio is defined as percentage of a particular ethnicity in cases/deaths
divided by the percentage of that ethnicity in the county.

**All processed data for a state, or a county in a particular state, is stored in a directory of the form `states/{STATE}/csvs/`**

Adding regions for processing raw data boils down to creating a parser config file containing elements that are used to 
transform raw data to case/death counts, creating a projection exception config file (that will be explained shortly), and creating
a python projector file of form `{COUNTY}_projector.py` or `{STATE}_projector.py`. The config files
are saved in the state and/or county `configs` directory. We go through these steps in more detail below.

1. Create a parser config that will be used to convert raw data to case/death counts. The parser
will either contain xpaths for html, or a list of json keys. The json keys can be found by using pythons
`json` module to load the raw data text through `json.load`. The xpaths for html can be found by opening
the raw html file, using inspect to select relevant elements on the page, and copying the full xpath of that element.
Both configs are partially replicated at the end of this section. 
    - An example of a html parser is [california_all_html_parse.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/configs/california_all_html_parse.yaml)
    - An example of a json parser is [santaclara_cases.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/santaclara/configs/santaclara_cases.yaml).
    - The `DATES` field in these files gives the date at which the html or json parsers
are valid. It is not uncommon for api responses to change slightly over time. At each date this occurs, a new element is added
under the `DATE` field with valid parsing. 

1. Create a [projector_exceptions.yaml](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/sacramento/configs/projector_exceptions.yaml) file. The processing managers have built in checks to see whether or not the 
result of processing raw data makes sense. 
    - `CASE_THRESH` and `DEATH_THRESH` field gives the threshold percentage change allowed before an error is thrown during processing case/death raw data. 
    - `CASE_DATES` and `DEATH_DATES` fields are populated with lists of dates in which checks are skipped when processing raw data. These are populated in the instances where
our checks throw an error, but, upon manual inspection, the changes recorded are valid. 

1. Create a python projector file of form `{COUNTY}_projector.py` or `{STATE}_projector.py`.
    - The [Santa Clara projector](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/santaclara/santaclara_projector.py) is
      is an example of a projector for parsing JSON raw data when one needs to combine values from different sources of raw data.
    - The [Sacrament projector](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/sacramento/sacramento_projector.py) is
      an example of a straightforward projector for parsing JSON raw data.
    - The [Los Angeles projector](https://github.com/odibua/covid19TrackingProject/blob/master/states/california/counties/losangeles/losangeles_projector.py)
      is an example of a projector for parsing html raw data. 
      
For any new region, populate the projector using one of the above examples as a reference.

**California html parse yaml**
```
DATES:
  '2020-06-13':
    LATINO_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[2]/td[2]
    LATINO_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[2]/td[4]
    WHITE_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[3]/td[2]
    WHITE_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[3]/td[4]
    ASIAN_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[4]/td[2]
    ASIAN_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[4]/td[4]
    BLACK_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[5]/td[2]
    BLACK_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[5]/td[4]
    MULTI_RACE_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[6]/td[2]
    MULTI_RACE_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[6]/td[4]
    AMERICAN_INDIAN_ALASKA_NATIVE_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[7]/td[2]
    AMERICAN_INDIAN_ALASKA_NATIVE_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[7]/td[4]
    NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[8]/td[2]
    NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[8]/td[4]
    OTHER_CASES: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[9]/td[2]
    OTHER_DEATHS: /html/body/form/div[1]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[2]/table/tbody/tr[9]/td[4]
    ...
```

**Santa Clara Cases Json Parser Yaml**
 ```
DATES:
  '2020-06-14':
    BLACK_CASES:
      - results
      - 0
      - result
      - data
      - dsr
      - DS
      - 0
      - PH
      - 1
      - DM1
      - 0
      - C
      - 1
    ASIAN_CASES:
      - results
      - 0
      - result
      - data
      - dsr
      - DS
      - 0
      - PH
      - 1
      - DM1
      - 1
      - C
      - 1
    HISPANIC_CASES:
      - results
      - 0
      - result
      - data
      - dsr
      - DS
      - 0
      - PH
      - 1
      - DM1
      - 2
      - C
      - 1
    NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES:
      - results
      - 0
      - result
      - data
      - dsr
      - DS
      - 0
      - PH
      - 1
      - DM1
      - 3
      - C
      - 1
    OTHER_CASES:
      - results
      - 0
      - result
      - data
      - dsr
      - DS
      - 0
      - PH
      - 1
      - DM1
      - 4
      - C
      - 1
    UNKNOWN_CASES:
      - results
      - 0
      - result
      - data
      - dsr
      - DS
      - 0
      - PH
      - 1
      - DM1
      - 5
      - C
      - 1
    WHITE_CASES:
      - results
      - 0
      - result
      - data
      - dsr
      - DS
      - 0
      - PH
      - 1
      - DM1
      - 6
      - C
      - 1
```

## Processing Raw Data
Processing raw data makes use of one of three commands:
- `python -m pytest --project_case_bool` processes raw data about cases from every state/county with a properly populated
 `test_{STATE}_{COUNTY}_scrape_project.py` file and pushes the resulting csv.
- `python -m pytest --state={STATE} --project_case_bool` does this for a state and every county in the state.
- `python -m pytest --state={STATE} --county={COUNTY} --projecth_case_bool` does this for a particular state and county.

To process raw data for deaths replace `project_case_bool` with `project_death_bool`. It is possible to
make processing raw data a regularly scheduled by adding one of the above commands to the circleci config
as illustrated [here](#configuring-scraping-schedule). **Sceduling processing is not recommended. Processing raw data raises
frequent errors, and it is best to periodically do so locally**

## Handling Processing Errors
Three primary types of errors will occur when processing raw data. The first is that newly processed data will have abnormal
values, the second is that processed data will have inconsistent keys (ethnic categories may change, for example), and the
third is that the parsing is simply not working. The first two errors are thrown based on the output of the
 [check_valid_change](https://github.com/odibua/covid19TrackingProject/blob/62671742b142ec643bd55e5e4d6e386df7e2221f/utils.py#L28)
 function. We will iterate through example error messages from these, and how to address them.

1. **Abnormal Values**
     - Sample Error: 
        ```
        ValueError: CASES: ERROR state: california county: santaclara Max difference 0.14239324565676245 is greater than thresh: 0.1 
        {'White': 1487, 'Hispanic': 7036, 'Asian': 1357, 'Black': 247, 'Native Hawaiian/Pacific Islander': 83, 'Other': 884, 'date': '2020-08-12'} 
        != {'White': 1376, 'Hispanic': 6159, 'Asian': 1260, 'Black': 226, 'Native Hawaiian/Pacific Islander': 80, 'Other': 850, 'date': '2020-08-08'}
        ```

    - For this error, check the raw data from which this is being pulled:
        - If the data is correct, add the date, `2020-08-12`, to this state/counties `projection_exception.yaml`
        - If the data is incorrect, update the html or json parser with new xpaths/json keys at this new
          date as shown [here](#adding-regions-for-processing-raw-data)

1. **Inconsistent Keys**
    - Sample Error:
        ```
        ValueError: ERROR state: {'White': 1487, 'Hispanic': 7036, 'Asian': 1357, 'Black': 247, 'Native Hawaiian/Pacific Islander': 83, 'Other': 884, 'date': '2020-08-12'} 
        != {'White': 1376, 'Hispanic': 6159, 'Asian/Pacific Islander': 1260, 'Black': 226, 'Native Hawaiian': 80, 'Other': 850, 'date': '2020-08-08'}
        ```
    - For this error perform the same checks as above

1. **Error in Parsing**
    - Sample Error:he 
        ```
         ValueError: Issue with parsing cases at date: 2020-08-12 with most recent entry {'hispanic': 1431, 'white': 505, 'asian_pacific_islander': 48, 'non_hispanic': 103, 'date': '2020-08-08'}
        ```
    - For this error update the html or json parser with new xpaths/json keys at this new date as shown [here](#adding-regions-for-processing-raw-data)
    - Sometimes it can be the case that, though no error was thrown in scraping, you are collecting the wrong raw data. In this case, 
      update the raw scraper for a particular region.

## Where to Scrape More Data From
Below we will give more sources from which to potentially scrape data. If you are interested in
a particular county, you will likely need to search for it online. This is an evolving list and
all are welcomed to it.

1. COVID Tracking Project:
   This is a great project run by the Atlantic, and low hanging fruit. They update the covid case/deaths for states reporting it by ethnicity
   twice a week. The link is [here](https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vR_xmYt4ACPDZCDJcY12kCiMiH0ODyx3E1ZvgOHB8ae1tRcjXbs_yWBOA4j4uoCEADVfC1PS2jYO68B/pubhtml#)
   
1. San Mateo County, California Dashboard:
   Dash board that shows cases/deaths stratified by ethnicity for San Mateo county. Linked [here](https://www.smchealth.org/data-dashboard/county-data-dashboard)
   
1. Denver County, Colorado:
   Website containing dashboard that shows cases/deaths stratified by ethnicity for Denver County. Linked [here](https://storymaps.arcgis.com/stories/50dbb5e7dfb6495292b71b7d8df56d0a).

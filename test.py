import ast
import datetime
import requests
import os
import pandas as pd
import json
from requests_html import HTMLSession
#
# payload = {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"c","Entity":"cases_race"}],"Select":[{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Race_eth"},"Name":"cases_race.Race_eth"},{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_group"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_group"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"Sum(cases_race.Percent_group)"},{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_pop"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_pop"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"Sum(cases_race.Percent_pop)"}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0,1,2],"Subtotal":1}]},"DataReduction":{"DataVolume":3,"Primary":{"Window":{"Count":500}}},"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"c\",\"Entity\":\"cases_race\"}],\"Select\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Race_eth\"},\"Name\":\"cases_race.Race_eth\"},{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_group\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_group\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"Sum(cases_race.Percent_group)\"},{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_pop\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_pop\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"Sum(cases_race.Percent_pop)\"}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1,2],\"Subtotal\":1}]},\"DataReduction\":{\"DataVolume\":3,\"Primary\":{\"Window\":{\"Count\":500}}},\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"366bfc6b-cdb9-43a4-9208-89ffd773dfe7","Sources":[{"ReportId":"f863b97c-85d7-431e-a6bf-20549f18d10f"}]}}],"cancelQueries":[],"modelId":320392}
#
titles = ["riverside_cases", "imperial_county_deaths", "imperial_county_cases", "losangeles_all", "sacramento_total_cases", "sacramento_cases", "sacramento_deaths", "kern_cases", "sonoma_all", "santaclara_deaths", "santaclara_totalcases", "california_all", "alameda_deaths", "alameda_cases", "sf_deaths", "sf_cases", "santaclara_totaldeaths", "santaclara_cases"]
methods = ["get", "get", "get", "get", "get", "get", "get", "post", "get", "post", "post", "get", "get", "get", "post", "post", "post", "post"]
urls = [
    "https://services1.arcgis.com/pWmBUdSlVpXStHU6/arcgis/rest/services/COVID19_Race_Graph/FeatureServer/0/query?f=json&where=Race%20NOT%20IN(%27Unknown%27)%20AND%20Race%20NOT%20IN(%27Total%20Pop%27)&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=Rate_100K%20asc&outSR=102100&resultOffset=0&resultRecordCount=32000&resultType=standard&cacheHint=true",
    "https://services7.arcgis.com/RomaVqqozKczDNgd/arcgis/rest/services/s0vvL/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&groupByFieldsForStatistics=ETHNICITY&outStatistics=%5B%7B%22statisticType%22%3A%22count%22%2C%22onStatisticField%22%3A%22ID%22%2C%22outStatisticFieldName%22%3A%22value%22%7D%5D&resultType=standard&cacheHint=true",
    "https://services7.arcgis.com/RomaVqqozKczDNgd/arcgis/rest/services/jtXgg/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&outStatistics=%5B%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22ETHNICITY_HISPANIC_OR_LATINO%22%2C%22outStatisticFieldName%22%3A%22ETHNICITY_HISPANIC_OR_LATINO%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22ETHNICITY_NON_x002d_HISPANIC_OR%22%2C%22outStatisticFieldName%22%3A%22ETHNICITY_NON_x002d_HISPANIC_OR%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22ETHNICITY_OTHER%22%2C%22outStatisticFieldName%22%3A%22ETHNICITY_OTHER%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22ETHNICITY_UNKNOWN%22%2C%22outStatisticFieldName%22%3A%22ETHNICITY_UNKNOWN%22%7D%5D&resultType=standard&cacheHint=true",
    "http://publichealth.lacounty.gov/media/Coronavirus/locations.htm",
    "https://services6.arcgis.com/yeTSZ1znt7H7iDG7/arcgis/rest/services/Cases/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&resultOffset=0&resultRecordCount=50&resultType=standard&cacheHint=true",
    "https://services6.arcgis.com/yeTSZ1znt7H7iDG7/arcgis/rest/services/COVID19_Race_Ethnicity/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=Percent_%20desc&resultOffset=0&resultRecordCount=20&resultType=standard&cacheHint=true",
    "https://services6.arcgis.com/yeTSZ1znt7H7iDG7/arcgis/rest/services/COVID19_Deaths_by_Race_Ethnicity/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=Percent_%20desc&resultOffset=0&resultRecordCount=20&resultType=standard&cacheHint=true",
    "https://wabi-us-gov-iowa-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true",
    "https://socoemergency.org/emergency/novel-coronavirus/coronavirus-cases/",
    "https://wabi-us-gov-virginia-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true",
    "https://wabi-us-gov-virginia-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true",
    "https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/Race-Ethnicity.aspx",
    "https://services3.arcgis.com/1iDJcsklY3l3KIjE/arcgis/rest/services/AC_deaths_rates/FeatureServer/0/query?f=json&where=Geography%3D%27Alameda%20County%27&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&outStatistics=%5B%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Deaths_Hispanic_Latino%22%2C%22outStatisticFieldName%22%3A%22Deaths_Hispanic_Latino%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Deaths_Asian%22%2C%22outStatisticFieldName%22%3A%22Deaths_Asian%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Deaths_African_American_Black%22%2C%22outStatisticFieldName%22%3A%22Deaths_African_American_Black%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Deaths_White%22%2C%22outStatisticFieldName%22%3A%22Deaths_White%22%7D%5D&resultType=standard&cacheHint=true",
    "https://services3.arcgis.com/1iDJcsklY3l3KIjE/arcgis/rest/services/AC_cases/FeatureServer/0/query?f=json&where=Geography%3D%27Alameda%20County%27&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&outStatistics=%5B%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Hispanic_Latino%22%2C%22outStatisticFieldName%22%3A%22Hispanic_Latino%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Asian%22%2C%22outStatisticFieldName%22%3A%22Asian%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22African_American_Black%22%2C%22outStatisticFieldName%22%3A%22African_American_Black%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22White%22%2C%22outStatisticFieldName%22%3A%22White%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Pacific_Islander%22%2C%22outStatisticFieldName%22%3A%22Pacific_Islander%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Native_American%22%2C%22outStatisticFieldName%22%3A%22Native_American%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Multirace%22%2C%22outStatisticFieldName%22%3A%22Multirace%22%7D%2C%7B%22statisticType%22%3A%22avg%22%2C%22onStatisticField%22%3A%22Unknown_Race%22%2C%22outStatisticFieldName%22%3A%22Unknown_Race%22%7D%5D&resultType=standard&cacheHint=true",
    "https://wabi-us-gov-iowa-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true",
    "https://wabi-us-gov-iowa-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true",
    "https://wabi-us-gov-virginia-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true",
    "https://wabi-us-gov-virginia-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true",
]
payloads = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"n","Entity":"nCoV Condition"}],"Select":[{"Column":{"Expression":{"SourceRef":{"Source":"n"}},"Property":"nCoV17"},"Name":"nCoV Condition.nCoV17"},{"Measure":{"Expression":{"SourceRef":{"Source":"n"}},"Property":"CurrentCaseCount"},"Name":"nCoV Condition.CurrentCaseCount"}],"Where":[{"Condition":{"Not":{"Expression":{"In":{"Expressions":[{"Column":{"Expression":{"SourceRef":{"Source":"n"}},"Property":"AgeGroup"}}],"Values":[[{"Literal":{"Value":"null"}}]]}}}}},{"Condition":{"Not":{"Expression":{"In":{"Expressions":[{"Column":{"Expression":{"SourceRef":{"Source":"n"}},"Property":"Region"}}],"Values":[[{"Literal":{"Value":"'Check Address'"}}],[{"Literal":{"Value":"null"}}]]}}}}}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0,1]}]},"DataReduction":{"DataVolume":3,"Primary":{"Top":{}}},"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"n\",\"Entity\":\"nCoV Condition\"}],\"Select\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"n\"}},\"Property\":\"nCoV17\"},\"Name\":\"nCoV Condition.nCoV17\"},{\"Measure\":{\"Expression\":{\"SourceRef\":{\"Source\":\"n\"}},\"Property\":\"CurrentCaseCount\"},\"Name\":\"nCoV Condition.CurrentCaseCount\"}],\"Where\":[{\"Condition\":{\"Not\":{\"Expression\":{\"In\":{\"Expressions\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"n\"}},\"Property\":\"AgeGroup\"}}],\"Values\":[[{\"Literal\":{\"Value\":\"null\"}}]]}}}}},{\"Condition\":{\"Not\":{\"Expression\":{\"In\":{\"Expressions\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"n\"}},\"Property\":\"Region\"}}],\"Values\":[[{\"Literal\":{\"Value\":\"'Check Address'\"}}],[{\"Literal\":{\"Value\":\"null\"}}]]}}}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1]}]},\"DataReduction\":{\"DataVolume\":3,\"Primary\":{\"Top\":{}}},\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"058e2058-6a36-4b57-95c5-61a9a4154411","Sources":[{"ReportId":"7ef83e30-2a44-40c9-bfad-2b8e55ca6cac"}]}}],"cancelQueries":[],"modelId":283897},
    None,
    {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"d","Entity":"death_race"}],"Select":[{"Column":{"Expression":{"SourceRef":{"Source":"d"}},"Property":"Race_eth"},"Name":"death_race.Race_eth"},{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"d"}},"Property":"Counts"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"d"}},"Property":"Counts"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"Sum(death_race.Counts)"}],"OrderBy":[{"Direction":1,"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"d"}},"Property":"Race_eth"}}}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0,1]}]},"DataReduction":{"DataVolume":4,"Primary":{"Window":{"Count":1000}}},"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"d\",\"Entity\":\"death_race\"}],\"Select\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"d\"}},\"Property\":\"Race_eth\"},\"Name\":\"death_race.Race_eth\"},{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"d\"}},\"Property\":\"Counts\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"d\"}},\"Property\":\"Counts\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"Sum(death_race.Counts)\"}],\"OrderBy\":[{\"Direction\":1,\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"d\"}},\"Property\":\"Race_eth\"}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1]}]},\"DataReduction\":{\"DataVolume\":4,\"Primary\":{\"Window\":{\"Count\":1000}}},\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"366bfc6b-cdb9-43a4-9208-89ffd773dfe7","Sources":[{"ReportId":"f863b97c-85d7-431e-a6bf-20549f18d10f"}]}}],"cancelQueries":[],"modelId":320392},
    {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"c","Entity":"counts"}],"Select":[{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Total"}},"Function":0},"Name":"Sum(counts.Total)"}],"Where":[{"Condition":{"In":{"Expressions":[{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Category"}}],"Values":[[{"Literal":{"Value":"'Cases'"}}]]}}}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0]}]},"DataReduction":{"DataVolume":3,"Primary":{"Top":{}}},"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"c\",\"Entity\":\"counts\"}],\"Select\":[{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total\"}},\"Function\":0},\"Name\":\"Sum(counts.Total)\"}],\"Where\":[{\"Condition\":{\"In\":{\"Expressions\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Category\"}}],\"Values\":[[{\"Literal\":{\"Value\":\"'Cases'\"}}]]}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0]}]},\"DataReduction\":{\"DataVolume\":3,\"Primary\":{\"Top\":{}}},\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"366bfc6b-cdb9-43a4-9208-89ffd773dfe7","Sources":[{"ReportId":"f863b97c-85d7-431e-a6bf-20549f18d10f"}]}}],"cancelQueries":[],"modelId":320392},
    None,
    None,
    None,
    {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"d1","Entity":"Deaths_Ethnicity"}],"Select":[{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"d1"}},"Property":"Total Cases"}},"Function":0},"Name":"Sum(Deaths_Ethnicity.Total Cases)"},{"Column":{"Expression":{"SourceRef":{"Source":"d1"}},"Property":"raceethnicity"},"Name":"Deaths_Ethnicity.raceethnicity"}],"OrderBy":[{"Direction":1,"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"d1"}},"Property":"Total Cases"}},"Function":0}}}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0,1]}]},"DataReduction":{"DataVolume":4,"Primary":{"Window":{"Count":1000}}},"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"d1\",\"Entity\":\"Deaths_Ethnicity\"}],\"Select\":[{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"d1\"}},\"Property\":\"Total Cases\"}},\"Function\":0},\"Name\":\"Sum(Deaths_Ethnicity.Total Cases)\"},{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"d1\"}},\"Property\":\"raceethnicity\"},\"Name\":\"Deaths_Ethnicity.raceethnicity\"}],\"OrderBy\":[{\"Direction\":1,\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"d1\"}},\"Property\":\"Total Cases\"}},\"Function\":0}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1]}]},\"DataReduction\":{\"DataVolume\":4,\"Primary\":{\"Window\":{\"Count\":1000}}},\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"507f48cb-a9bd-432f-af06-de81cb290a5e","Sources":[{"ReportId":"4b76105e-31ab-44bf-89d7-66a4323e3e02"}]}}],"cancelQueries":[],"modelId":282807},
    {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"c","Entity":"Cases_Ethnicity"}],"Select":[{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Total Cases"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Total Cases"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"CountNonNull(Cases_Ethnicity.Total Cases)"},{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Total Cases"}},"Function":0},"Name":"CountNonNull(Cases_Ethnicity.Total Cases)1"},{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"raceethnicity"},"Name":"Cases_Ethnicity.raceethnicity"}],"OrderBy":[{"Direction":1,"Expression":{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Total Cases"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Total Cases"}},"Function":0}},"Scope":[]}},"Operator":3}}}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0,1,2]}]},"DataReduction":{"DataVolume":4,"Primary":{"Window":{"Count":1000}}},"SuppressedJoinPredicates":[1],"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"c\",\"Entity\":\"Cases_Ethnicity\"}],\"Select\":[{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"CountNonNull(Cases_Ethnicity.Total Cases)\"},{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0},\"Name\":\"CountNonNull(Cases_Ethnicity.Total Cases)1\"},{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"raceethnicity\"},\"Name\":\"Cases_Ethnicity.raceethnicity\"}],\"OrderBy\":[{\"Direction\":1,\"Expression\":{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1,2]}]},\"DataReduction\":{\"DataVolume\":4,\"Primary\":{\"Window\":{\"Count\":1000}}},\"SuppressedJoinPredicates\":[1],\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"507f48cb-a9bd-432f-af06-de81cb290a5e","Sources":[{"ReportId":"4b76105e-31ab-44bf-89d7-66a4323e3e02"}]}}],"cancelQueries":[],"modelId":282807},
    {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"c","Entity":"counts"}],"Select":[{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Total"}},"Function":0},"Name":"Sum(counts.Total)"}],"Where":[{"Condition":{"In":{"Expressions":[{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Category"}}],"Values":[[{"Literal":{"Value":"'Deaths'"}}]]}}}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0]}]},"DataReduction":{"DataVolume":3,"Primary":{"Top":{}}},"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"c\",\"Entity\":\"counts\"}],\"Select\":[{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total\"}},\"Function\":0},\"Name\":\"Sum(counts.Total)\"}],\"Where\":[{\"Condition\":{\"In\":{\"Expressions\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Category\"}}],\"Values\":[[{\"Literal\":{\"Value\":\"'Deaths'\"}}]]}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0]}]},\"DataReduction\":{\"DataVolume\":3,\"Primary\":{\"Top\":{}}},\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"366bfc6b-cdb9-43a4-9208-89ffd773dfe7","Sources":[{"ReportId":"f863b97c-85d7-431e-a6bf-20549f18d10f"}]}}],"cancelQueries":[],"modelId":320392},
    {"version":"1.0.0","queries":[{"Query":{"Commands":[{"SemanticQueryDataShapeCommand":{"Query":{"Version":2,"From":[{"Name":"c","Entity":"cases_race"}],"Select":[{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Race_eth"},"Name":"cases_race.Race_eth"},{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_group"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_group"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"Sum(cases_race.Percent_group)"},{"Arithmetic":{"Left":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_pop"}},"Function":0}},"Right":{"ScopedEval":{"Expression":{"Aggregation":{"Expression":{"Column":{"Expression":{"SourceRef":{"Source":"c"}},"Property":"Percent_pop"}},"Function":0}},"Scope":[]}},"Operator":3},"Name":"Sum(cases_race.Percent_pop)"}]},"Binding":{"Primary":{"Groupings":[{"Projections":[0,1,2],"Subtotal":1}]},"DataReduction":{"DataVolume":3,"Primary":{"Window":{"Count":500}}},"Version":1}}}]},"CacheKey":"{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"c\",\"Entity\":\"cases_race\"}],\"Select\":[{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Race_eth\"},\"Name\":\"cases_race.Race_eth\"},{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_group\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_group\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"Sum(cases_race.Percent_group)\"},{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_pop\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Percent_pop\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"Sum(cases_race.Percent_pop)\"}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1,2],\"Subtotal\":1}]},\"DataReduction\":{\"DataVolume\":3,\"Primary\":{\"Window\":{\"Count\":500}}},\"Version\":1}}}]}","QueryId":"","ApplicationContext":{"DatasetId":"366bfc6b-cdb9-43a4-9208-89ffd773dfe7","Sources":[{"ReportId":"f863b97c-85d7-431e-a6bf-20549f18d10f"}]}}],"cancelQueries":[],"modelId":320392}
]
headers = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    { 'Accept': 'application/json, text/plain, */*',
                      'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'ActivityId': 'fc9ca2a8-2117-4ff1-8b02-f37fecc2af30',
                        'Connection': 'keep-alive',
                        'Content-Length': '2193',
                        'Content-Type': 'application/json;charset=UTF-8',
                        'Host': 'wabi-us-gov-iowa-api.analysis.usgovcloudapi.net',
                        'Origin': 'https://app.powerbigov.us',
                        'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiNDg0ZmE3MjgtZDg0OC00N2QxLWE4ZWQtOGNkZDFhYWM3ZTlmIiwidCI6ImUwZjJlNGI1LTA1MTUtNDAyOC05OWYyLTJlN2E0M2ZlNTM3OSJ9',
                        'RequestId': '18265ce9-e36c-c130-35ba-213714367ed6',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        'X-PowerBI-ResourceKey': '484fa728-d848-47d1-a8ed-8cdd1aac7e9f',
            },
    None,
    { 'Accept': 'application/json, text/plain, */*',
                      'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'ActivityId': '4892726d-e072-4491-a2ab-8c7395a5a053',
                        'Connection': 'keep-alive',
                        'Content-Length': '2075',
                        'Content-Type': 'application/json;charset=UTF-8',
                        'Host': 'wabi-us-gov-virginia-api.analysis.usgovcloudapi.net',
                        'Origin': 'https://app.powerbigov.us',
                        'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiMWEyYTRhYTAtMTdjYi00YTE1LWJiMTQtYTY3NmJmMjJhOThkIiwidCI6IjIyZDVjMmNmLWNlM2UtNDQzZC05YTdmLWRmY2MwMjMxZjczZiJ9&navContentPaneEnabled=false&filterPaneEnabled=false',
                        'RequestId': '512a3a33-aa3a-0973-0dce-0a8503e83ac0',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        'X-PowerBI-ResourceKey': 'e8619a1d-ea98-4d27-a860-157af0d4e93f',
            },
    { 'Accept': 'application/json, text/plain, */*',
                      'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'ActivityId': '235ea993-40b5-8975-b7de-2d182a1add43',
                        'Connection': 'keep-alive',
                        'Content-Length': '1461',
                        'Content-Type': 'application/json;charset=UTF-8',
                        'Host': 'wabi-us-gov-virginia-api.analysis.usgovcloudapi.net',
                        'Origin': 'https://app.powerbigov.us',
                        'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiMWEyYTRhYTAtMTdjYi00YTE1LWJiMTQtYTY3NmJmMjJhOThkIiwidCI6IjIyZDVjMmNmLWNlM2UtNDQzZC05YTdmLWRmY2MwMjMxZjczZiJ9&navContentPaneEnabled=false&filterPaneEnabled=false',
                        'RequestId': 'a0adb383-081a-b77e-09d7-472b5d74885f',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        'X-PowerBI-ResourceKey': 'e8619a1d-ea98-4d27-a860-157af0d4e93f',
            },
    None,
    None,
    None,
    { 'Accept': 'application/json, text/plain, */*',
                      'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'ActivityId': '92ab9535-b7d3-477b-adbb-eb163a4b24e9',
                        'Connection': 'keep-alive',
                        'Content-Length': '1815',
                        'Content-Type': 'application/json;charset=UTF-8',
                        'Host': 'wabi-us-gov-iowa-api.analysis.usgovcloudapi.net',
                        'Origin': 'https://app.powerbigov.us',
                        'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiMWEyYTRhYTAtMTdjYi00YTE1LWJiMTQtYTY3NmJmMjJhOThkIiwidCI6IjIyZDVjMmNmLWNlM2UtNDQzZC05YTdmLWRmY2MwMjMxZjczZiJ9&navContentPaneEnabled=false&filterPaneEnabled=false',
                        'RequestId': 'b82a0201-b443-b44d-a6b6-55c10bdf7971',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        'X-PowerBI-ResourceKey': '1a2a4aa0-17cb-4a15-bb14-a676bf22a98d',
            },
    { 'Accept': 'application/json, text/plain, */*',
                      'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'ActivityId': '7c3d6a5e-411d-e919-7d0f-9cf97758c7ae',
                        'Connection': 'keep-alive',
                        'Content-Length': '3171',
                        'Content-Type': 'application/json;charset=UTF-8',
                        'Host': 'wabi-us-gov-iowa-api.analysis.usgovcloudapi.net',
                        'Origin': 'https://app.powerbigov.us',
                        'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiMWEyYTRhYTAtMTdjYi00YTE1LWJiMTQtYTY3NmJmMjJhOThkIiwidCI6IjIyZDVjMmNmLWNlM2UtNDQzZC05YTdmLWRmY2MwMjMxZjczZiJ9&navContentPaneEnabled=false&filterPaneEnabled=false',
                        'RequestId': '79e54f51-a254-f1c7-7df7-cca9d4aeba97',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        'X-PowerBI-ResourceKey': '1a2a4aa0-17cb-4a15-bb14-a676bf22a98df',
            },
    { 'Accept': 'application/json, text/plain, */*',
                      'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'ActivityId': '235ea993-40b5-8975-b7de-2d182a1add43',
                        'Connection': 'keep-alive',
                        'Content-Length': '1463',
                        'Content-Type': 'application/json;charset=UTF-8',
                        'Host': 'wabi-us-gov-virginia-api.analysis.usgovcloudapi.net',
                        'Origin': 'https://app.powerbigov.us',
                        'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiZTg2MTlhMWQtZWE5OC00ZDI3LWE4NjAtMTU3YWYwZDRlOTNmIiwidCI6IjBhYzMyMDJmLWMzZTktNGY1Ni04MzBkLTAxN2QwOWQxNmIzZiJ9',
                        'RequestId': '0f27d1d9-adfe-45c9-8b69-e3b5ddcb1cea',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        'X-PowerBI-ResourceKey': 'e8619a1d-ea98-4d27-a860-157af0d4e93f',
            },
        { 'Accept': 'application/json, text/plain, */*',
                      'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'ActivityId': '235ea993-40b5-8975-b7de-2d182a1add43',
                        'Connection': 'keep-alive',
                        'Content-Length': '2691',
                        'Content-Type': 'application/json;charset=UTF-8',
                        'Host': 'wabi-us-gov-virginia-api.analysis.usgovcloudapi.net',
                        'Origin': 'https://app.powerbigov.us',
                        'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiZTg2MTlhMWQtZWE5OC00ZDI3LWE4NjAtMTU3YWYwZDRlOTNmIiwidCI6IjBhYzMyMDJmLWMzZTktNGY1Ni04MzBkLTAxN2QwOWQxNmIzZiJ9',
                        'RequestId': '442e39f9-eed1-af32-ef33-dc6bfbc48c8e',
                        'Sec-Fetch-Dest': 'empty',
                        'Sec-Fetch-Mode': 'cors',
                        'Sec-Fetch-Site': 'cross-site',
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        'X-PowerBI-ResourceKey': 'e8619a1d-ea98-4d27-a860-157af0d4e93f',
            }
    ]

# g = requests.post(url=urls[0], headers=headers[0], json=json.loads(json.dumps(payload[0])))
# print(g.status_code)
# print(g.text)

success_list = []
failure_list = []
path_to_raw = 'state/california/data/raw_pages'
today = datetime.datetime.now()
today_str = today.isoformat()
for title, url, method, header, payload in zip(titles, urls, methods, headers, payloads):
    status_code = -1
    if method == 'get':
        response = requests.get(url)
        status_code = response.status_code
    elif method == 'post':
        response = requests.post(url=url, headers=header, json=json.loads(json.dumps(payload)))
        status_code = response.status_code
        # print(title)
        # print(isinstance(response.text, str))

    if status_code == 200:
        full_path = os.path.join(path_to_raw, today_str)
        if not os.path.isdir(full_path):
            os.mkdir(full_path)
        text_file = open(f"{full_path}/{title}", "w")
        n = text_file.write(response.text)
        response.close()
        text_file.close()
        success_list.append(title)
    else:
        print(status_code)
        print(method)
        print(header)
        print(payload)
        print(url)
        failure_list.append(title)
print(success_list)
print(failure_list)


# g = requests.get(url="https://services6.arcgis.com/yeTSZ1znt7H7iDG7/arcgis/rest/services/Cases/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&resultOffset=0&resultRecordCount=50&resultType=standard&cacheHint=true")
# print(g.status_code)
# print(g.text)

# session = HTMLSession()
# r = session.get('https://cofgisonline.maps.arcgis.com/apps/MapSeries/index.html?appid=1f82e8eb24c0403c90e774202c5dafea')
# r.html.render()
# print(r.text)

# path = 'state/california/data'
# source_data_df = pd.read_csv('state/california/data/covid_ethinic_online_data.csv')
#
# path_to_raw = 'state/california/data/raw_pages'
# success_list = []
# failure_list = []
# cnt_list = [ 'fresno_all', 'kern_cases', 'tulane_cases', 'santa barbara_all', 'san diego_cases', 'san diego_cases']
# for idx, row in source_data_df.iterrows():
#     status = 403
#     try:
#         county = row['county'].lower()
#         type = str(row['type']).lower()
#     # if f"{county}_{type}" in cnt_list:
#     #     print(row['method'])
#         method = str(row['method']).lower()
#         url = row['request_url']
#         path_to_raw_county = os.path.join(path_to_raw, county)
#         if not os.path.isdir(path_to_raw_county):
#             os.mkdir(path_to_raw_county)
#
#         if method == 'get':
#             response = requests.get(url)
#             status = response.status_code
#         elif method == 'post':
#             # print(''.join(row['header'].split()))
#             # headers = json.dumps(''.join(row['header'].split()).replace("'",'"'))
#             headers = ast.literal_eval(''.join(row['header'].split()))
#             if isinstance(headers, dict):
#                 pass
#             else:
#                 print(isinstance(headers, dict))
#                 print(headers)
#             payload = ast.literal_eval(''.join(row['payload'].split()))
#             response = requests.post(url, headers=headers, json=json.loads(json.dumps(payload)))
#             status = response.status_code
#             print(f"{county}_{type}")
#             print(response.text)
#             print(' ')
#             # import ipdb
#             # ipdb.set_trace()
#     except:
#         # print(type)
#         pass
#
#     if int(status) == 200:
#         success_list.append(f"{county}_{type}")
#         # print(response.text)
#     else:
#         # print(rf"{county}_{type}")
#         # print(status)
#         # print(method)
#         # if method != 'get' and method != 'post':
#         failure_list.append(f"{county}_{type}")
#
# print(success_list)
# print(len(success_list))
# print(failure_list)

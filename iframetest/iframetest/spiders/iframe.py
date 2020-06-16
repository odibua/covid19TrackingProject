import json
import scrapy
from scrapy.http import *

class MySpider(scrapy.Spider):
    name = "myspider"
    def start_requests(self):
        url = "https://wabi-us-gov-iowa-api.analysis.usgovcloudapi.net/public/reports/querydata?synchronous=true"
        payload = {"version": "1.0.0", "queries": [{"Query": {"Commands": [{"SemanticQueryDataShapeCommand": {
            "Query": {"Version": 2, "From": [{"Name": "c", "Entity": "Cases_Ethnicity"}],
                      "Select": [{"Arithmetic": {"Left": {
                          "Aggregation": {
                              "Expression": {
                                  "Column": {"Expression": {"SourceRef": {"Source": "c"}}, "Property": "Total Cases"}},
                              "Function": 0}}, "Right": {"ScopedEval": {"Expression": {"Aggregation": {
                          "Expression": {
                              "Column": {"Expression": {"SourceRef": {"Source": "c"}}, "Property": "Total Cases"}},
                          "Function": 0}}, "Scope": []}}, "Operator": 3},
                          "Name": "CountNonNull(Cases_Ethnicity.Total Cases)"}, {
                          "Aggregation": {
                              "Expression": {
                                  "Column": {
                                      "Expression": {
                                          "SourceRef": {
                                              "Source": "c"}},
                                      "Property": "Total Cases"}},
                              "Function": 0},
                          "Name": "CountNonNull(Cases_Ethnicity.Total Cases)1"},
                          {"Column": {"Expression": {
                              "SourceRef": {
                                  "Source": "c"}},
                              "Property": "raceethnicity"},
                              "Name": "Cases_Ethnicity.raceethnicity"}],
                      "OrderBy": [{"Direction": 1, "Expression": {"Arithmetic": {"Left": {"Aggregation": {
                          "Expression": {
                              "Column": {"Expression": {"SourceRef": {"Source": "c"}}, "Property": "Total Cases"}},
                          "Function": 0}}, "Right": {"ScopedEval": {"Expression": {"Aggregation": {
                          "Expression": {
                              "Column": {"Expression": {"SourceRef": {"Source": "c"}}, "Property": "Total Cases"}},
                          "Function": 0}}, "Scope": []}}, "Operator": 3}}}]},
            "Binding": {"Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                        "DataReduction": {"DataVolume": 4, "Primary": {"Window": {"Count": 1000}}},
                        "SuppressedJoinPredicates": [1], "Version": 1}}}]},
            "CacheKey": "{\"Commands\":[{\"SemanticQueryDataShapeCommand\":{\"Query\":{\"Version\":2,\"From\":[{\"Name\":\"c\",\"Entity\":\"Cases_Ethnicity\"}],\"Select\":[{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3},\"Name\":\"CountNonNull(Cases_Ethnicity.Total Cases)\"},{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0},\"Name\":\"CountNonNull(Cases_Ethnicity.Total Cases)1\"},{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"raceethnicity\"},\"Name\":\"Cases_Ethnicity.raceethnicity\"}],\"OrderBy\":[{\"Direction\":1,\"Expression\":{\"Arithmetic\":{\"Left\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Right\":{\"ScopedEval\":{\"Expression\":{\"Aggregation\":{\"Expression\":{\"Column\":{\"Expression\":{\"SourceRef\":{\"Source\":\"c\"}},\"Property\":\"Total Cases\"}},\"Function\":0}},\"Scope\":[]}},\"Operator\":3}}}]},\"Binding\":{\"Primary\":{\"Groupings\":[{\"Projections\":[0,1,2]}]},\"DataReduction\":{\"DataVolume\":4,\"Primary\":{\"Window\":{\"Count\":1000}}},\"SuppressedJoinPredicates\":[1],\"Version\":1}}}]}",
            "QueryId": "",
            "ApplicationContext": {"DatasetId": "507f48cb-a9bd-432f-af06-de81cb290a5e",
                                   "Sources": [{
                                       "ReportId": "4b76105e-31ab-44bf-89d7-66a4323e3e02"}]}}],
                   "cancelQueries": [], "modelId": 282807}
        headers = { 'Accept': 'application/json, text/plain, */*',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'ActivityId': 'c5f42d32-5a9b-6e57-910d-14c752f36ff5',
                    'Connection': 'keep-alive',
                    'Content-Length': '3171',
                    'Content-Type': 'application/json;charset=UTF-8',
                    'Host': 'wabi-us-gov-iowa-api.analysis.usgovcloudapi.net',
                    'Origin': 'https://app.powerbigov.us',
                    'Referer': 'https://app.powerbigov.us/view?r=eyJrIjoiMWEyYTRhYTAtMTdjYi00YTE1LWJiMTQtYTY3NmJmMjJhOThkIiwidCI6IjIyZDVjMmNmLWNlM2UtNDQzZC05YTdmLWRmY2MwMjMxZjczZiJ9&navContentPaneEnabled=false&filterPaneEnabled=false',
                    'RequestId': 'b6f43f3b-b415-4c24-a9d7-eff18bd5bbd0',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'cross-site',
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                    'X-PowerBI-ResourceKey': '1a2a4aa0-17cb-4a15-bb14-a676bf22a98d',
        }
        yield Request(url, self.parse, method="POST", headers=headers, body=json.dumps(payload))

    def parse(self, response):
        data = json.loads(response.body)
        text = json.loads(response.text)
        print(data)
        print(text)


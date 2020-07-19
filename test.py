# from lxml import etree
#
# path = 'states/california/raw_data/2020-07-07/california_all.html'
# htm = open(path, 'r')
# htm = htm.read()
# lxml = etree.HTML(htm)
# elem = lxml.xpath('/html/body/form/div[3]/div/section/div/div[2]/div[3]/div[1]/div/div[2]/div[2]/div[1]/div/div/div/div/div/div[1]/div[1]/table/tbody/tr[2]/td[2]')
# text = elem[0].text
# print(elem)
# print(type(text))
# print(text.replace(',', ''))

# from states.california import california_projector
#
# import os
#
# date_string = '2020-06-29'
# state = "california"
# county = None
# state_projector = california_projector.CaliforniaEthnicDataProjector(state=state, county=county, date_string=date_string)
# print(state_projector.process_raw_data_to_cases())
# print(state_projector.process_raw_data_to_deaths())
# print(state_projector.ethnicity_cases_discrepancies)
# print(state_projector.ethnicity_deaths_discrepancies)


# from states.california.counties.losangeles import losangeles_projector
# import os
# date_string = '2020-07-08'
# state = "california"
# county = 'losangeles'
# state_projector = losangeles_projector.LosAngelesEthnicDataProjector(state=state, county=county, date_string=date_string)
# print(state_projector.process_raw_data_to_cases())
# print(state_projector.process_raw_data_to_deaths())
# print(state_projector.ethnicity_cases_discrepancies)
# print(state_projector.ethnicity_deaths_discrepancies)

# from states.california.counties.sonoma import sonoma_projector
# import os
# date_string = '2020-07-09'
# state = "california"
# county = 'sonoma'
# state_projector = sonoma_projector.SonomaEthnicDataProjector(state=state, county=county, date_string=date_string)
# print(state_projector.process_raw_data_to_cases())
# print(state_projector.process_raw_data_to_deaths())
# print(state_projector.ethnicity_cases_discrepancies)
# print(state_projector.ethnicity_deaths_discrepancies)

# import os
# import json
# date_string_list = os.listdir('states/california/counties/sacramento/raw_data')
# date_string_list.sort()
# county_dir = "states/california/counties/sacramento/raw_data/"
# case_file = f"{county_dir}/{date_string_list[0]}/sacramento_deaths"
# case_obj = open(case_file, 'r')
# case_obj_dict = json.load(case_obj)
# idx_list = list(range(6))
# keys = ['features', 0, 'attributes', 'Race']
# for idx in idx_list:
#     if idx < len(case_obj_dict['features']):
#         print(f"Race: {case_obj_dict['features'][idx]['attributes']['Race_Ethnicity']} Keys: {['features', idx, 'attributes', 'Race_Ethnicity']}")
# for date_string in date_string_list[1:]:
#     try:
#         case_file = f"{county_dir}/{date_string}/sacramento_deaths"
#         tmp_obj = open(case_file, 'r')
#         tmp_dict = json.load(tmp_obj)
#     except:
#         case_file = f"{county_dir}/{date_string}/sacramento_deaths.html"
#         tmp_obj = open(case_file, 'r')
#         tmp_dict = json.load(tmp_obj)
#     change_bool = False
#     for idx in idx_list:
#         if idx < len(case_obj_dict['features']):
#             if case_obj_dict['features'][idx]['attributes']['Race_Ethnicity'] != tmp_dict['features'][idx]['attributes']['Race_Ethnicity']:
#                 change_bool = True
#                 break;
#     if len(case_obj_dict['features']) != len(tmp_dict['features']):
#         change_bool = True
#     if change_bool:
#         print(date_string)
#         for idx in idx_list:
#             if idx < len(tmp_dict['features']):
#                 print(f"Race: {tmp_dict['features'][idx]['attributes']['Race_Ethnicity']} Keys: {['features', idx, 'attributes','Race_Ethnicity']}")
#     case_obj_dict = tmp_dict

# import os
# import json
# date_string_list = os.listdir('states/california/counties/santaclara/raw_data')
# date_string_list.sort()
# date_string_list = date_string_list[9:]
# county_dir = "states/california/counties/santaclara/raw_data/"
# case_file = f"{county_dir}/{date_string_list[0]}/santaclara_totaldeaths"
# case_obj = open(case_file, 'r')
# case_obj_dict = json.load(case_obj)
# idx_list = list(range(1))
#
#
# for idx in idx_list:
#     print(f"Race: {case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][0].keys()} Keys: {['results',0,'result','data','dsr','DS',0,'PH',0,'DM0',0]}")
#     # if idx < len(case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#     #     print(f"Race: {case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0]} Keys: {['results',0,'result','data','dsr','DS',0,'PH',0,'DM0',idx,'C', 0]}")
# for date_string in date_string_list[1:]:
#     try:
#         case_file = f"{county_dir}/{date_string}/santaclara_totaldeaths"
#         tmp_obj = open(case_file, 'r')
#         tmp_dict = json.load(tmp_obj)
#         change_bool = False
#         if case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][0].keys() != tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][0].keys():
#             change_bool = True
#         if change_bool:
#             print(f"Race: {case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][0].keys()}")
#         # for idx in idx_list:
#         #     if idx < len(case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#         #         if case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0] != \
#         #                 tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0]:
#         #             change_bool = True
#         #             break;
#         # if len(case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']) != len(tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#         #     change_bool = True
#         # if change_bool:
#         #     print(date_string)
#         #     for idx in idx_list:
#         #         if idx < len(tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#         #             print(f"Race: {tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0]} Keys: {['results',0,'result','data','dsr','DS',0,'PH',0,'DM0',idx,'C', 0]}")
#         # case_obj_dict = tmp_dict
#     except:
#         pass

# import os
# import json
# date_string_list = os.listdir('states/california/counties/santaclara/raw_data')
# date_string_list.sort()
# date_string_list = date_string_list[9:]
# county_dir = "states/california/counties/santaclara/raw_data/"
# case_file = f"{county_dir}/{date_string_list[0]}/santaclara_cases"
# case_obj = open(case_file, 'r')
# case_obj_dict = json.load(case_obj)
# idx_list = list(range(10))
#
# for idx in idx_list:
#     if idx < len(case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#         print(f"Race: {case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0]} Keys: {['results',0,'result','data','dsr','DS',0,'PH',0,'DM0',idx,'C', 0]}")
# for date_string in date_string_list[1:]:
#     try:
#         case_file = f"{county_dir}/{date_string}/santaclara_cases"
#         tmp_obj = open(case_file, 'r')
#         tmp_dict = json.load(tmp_obj)
#         change_bool = False
#         for idx in idx_list:
#             if idx < len(case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#                 if case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0] != \
#                         tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0]:
#                     change_bool = True
#                     break;
#         if len(case_obj_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']) != len(tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#             change_bool = True
#         if change_bool:
#             print(date_string)
#             for idx in idx_list:
#                 if idx < len(tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']):
#                     print(f"Race: {tmp_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0'][idx]['C'][0]} Keys: {['results',0,'result','data','dsr','DS',0,'PH',0,'DM0',idx,'C', 0]}")
#         case_obj_dict = tmp_dict
#     except:
#         pass

from states.california.counties.santaclara import santaclara_projector
date_string = '2020-06-14'
state = "california"
county = 'santaclara'
state_projector = santaclara_projector.SantaClaraEthnicDataProjector(state=state, county=county, date_string=date_string)
print(state_projector.process_raw_data_to_cases())
print(state_projector.process_raw_data_to_deaths())
print(state_projector.ethnicity_cases_discrepancies)
print(state_projector.ethnicity_cases)
print(state_projector.total_cases)
print(state_projector.ethnicity_cases_percentages)
print(state_projector.ethnicity_deaths_discrepancies)
print(state_projector.ethnicity_deaths)
print(state_projector.ethnicity_deaths_percentages)
print(state_projector.total_deaths)

import ipdb
ipdb.set_trace()

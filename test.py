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

import ipdb
from states.california.counties.alameda import alameda_projector
import os
date_string = '2020-07-10'
state = "california"
county = 'alameda'
state_projector = alameda_projector.AlamedaEthnicDataProjector(state=state, county=county, date_string=date_string)
print(state_projector.process_raw_data_to_cases())
print(state_projector.process_raw_data_to_deaths())
print(state_projector.ethnicity_cases_discrepancies)
print(state_projector.ethnicity_deaths_discrepancies)
import ipdb
ipdb.set_trace()

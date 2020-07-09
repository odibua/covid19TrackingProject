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

from states.california import california_projector

date_string = '2020-06-13'
state_projector = california_projector.CaliforniaEthnicDataProjector(date_string=date_string)
state_projector.process_raw_data_to_cases()
state_projector.process_raw_data_to_deaths()
import ipdb

ipdb.set_trace()
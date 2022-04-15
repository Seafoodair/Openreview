from time import sleep
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
#executable_path = '/Users/waltersun/Desktop/chromedriver'  # path to your executable browser
#options = Options()
#url = 'https://openreview.net/group?id=ICLR.cc/2020/Conference#all-submissions'
driver = webdriver.Edge('msedgedriver.exe')
#browser = webdriver.Chrome(options=options, executable_path=executable_path)
#listadd=['https://openreview.net/group?id=ICLR.cc/2020/Conference#accept-spotlight','https://openreview.net/group?id=ICLR.cc/2020/Conference#accept-talk','https://openreview.net/group?id=ICLR.cc/2020/Conference#reject']
#for url in listadd:
#url='https://openreview.net/group?id=ICLR.cc/2020/Conference#accept-spotlight'
url='https://openreview.net/group?id=ICLR.cc/2020/Conference#reject'
driver.get(url)
time.sleep(10)
data_id_list = []
done = False
page = 1
title_id_list=[]
while not done:
    print('Page: {}'.format(page))
    time.sleep(2)
    for item in driver.find_elements(by=By.CLASS_NAME, value="note"):
        data_id = item.get_attribute('data-id')
        title = item.find_element_by_xpath('./h4/a[1]')
        title = title.text.strip().replace('\t', ' ').replace('\n', ' ')
        print(title)#title 获取成功

        data_id_list.append(data_id)
        #title_id_list.append(title)

        time.sleep(1)

    #text_center =driver.find_elements(by=By.CLASS_NAME, value="pagination-container text-center")
    #right_arrow = driver.find_elements(by=By.CLASS_NAME, value="right_arrow")


    done = True
data_id_list = list(set(data_id_list))
print('Number of submission: {}'.format(len(data_id_list)))
with open('2020Furls.txt', 'a+') as urls_txt:
    for data_id in data_id_list:
        urls_txt.write('https://openreview.net/forum?id={}\n'.format(data_id))
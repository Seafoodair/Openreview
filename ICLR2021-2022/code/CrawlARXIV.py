#encoding=utf-8
"""
@author=wanggang
"""
import os
import time
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import random

driver = webdriver.Edge('msedgedriver.exe')
arxivtime=[]
df = pd.read_csv('G:/爬虫代码/数据/2022.tsv', sep='\t', index_col=0,encoding='gb18030')
link='https://arxiv.org/search/?query=&searchtype=all&abstracts=show&order=-announced_date_first&size=50'#这一版代码只能一个有用。
driver.get(link)
citenum=[]
for i in df['title']:  #输出title
    i=i.lower()#title都变为小写
    #print(i)
    try:
        elem=driver.find_element(By.XPATH,value='//*[@id="query"]')  #这里替换arxiv的连接
        elem.clear()
        time.sleep(3)
        #print('-------------上title，下搜索')
        elem.send_keys(i)#输入关键词
        try:
            driver.find_element(By.XPATH,value='//*[@id="main-container"]/div[2]/form/div[1]/div[3]/button').click()#点击
            time.sleep(random.randint(1,3))
            try:
                arxivtitle = driver.find_element(By.XPATH, value='//*[@id="main-container"]/div[2]/ol/li/p[1]').text
            # Citetitle=Citeclass.find_element_by_xpath('./div[2]/h3/a').text
                arxivtitle = arxivtitle.lower()
                #print(arxivtitle)
                if i == arxivtitle:  # 判断是否相等
                    tt=driver.find_element(By.XPATH,value='//*[@id="main-container"]/div[2]/ol/li/p[4]').text
                    arxivtime.append(tt)
                    #print(tt)
                else:
                    #print("失败")
                    arxivtime.append("failed")
            except Exception as e:
                arxivtime.append(-1)
                continue
        except Exception as e:
            arxivtime.append(-1)
            continue
    except Exception as e:
        arxivtime.append(-1)
        continue

with open('2022年的防12arxiv.txt', 'w') as f:
    for data_id in arxivtime:
        f.write(str(data_id)+'\n')
print("成功")

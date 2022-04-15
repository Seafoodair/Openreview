import os
import time
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
def Dealwith(i):
    i = i.lower()  # title都变为小写
    # print(i)
    elem = driver.find_element(By.XPATH, value='//*[@id="gs_hdr_tsi"]')
    elem.clear()
    time.sleep(1)
    print('-------------上title，下搜索')
    elem.send_keys(i)  # 输入关键词
    driver.find_element(By.XPATH, value='//*[@id="gs_hdr_tsb"]').click()  # 点击
    time.sleep(3)
    Citetitle = driver.find_element(By.XPATH, value='//*[@id="gs_res_ccl_mid"]/div/div[2]/h3/a').text
    # Citetitle=Citeclass.find_element_by_xpath('./div[2]/h3/a').text
    Citetitle = Citetitle.lower()
    print(Citetitle)
    if i == Citetitle:  # 判断是否相等
        # print(111111111)
        waitsave = driver.find_element(By.XPATH, value='//*[@id="gs_res_ccl_mid"]/div/div[2]/div[3]')
        # print(waitsave.text)
        if waitsave.text.find("被引用次数："):
            start = waitsave.text.find("：") + 1
            end = waitsave.text.find("相关文章")
            print(waitsave.text[start:end])
            citenum.append(waitsave.text[start:end])
        else:
            # print(22222222)
            citenum.append(-1)



driver = webdriver.Edge('msedgedriver.exe')

#df = pd.read_csv('paperlist44.tsv', sep='\t', index_col=0)
df = pd.read_csv('C:/Users/wg/Downloads/ICLR2021-OpenReviewData-master/ICLR2021-OpenReviewData-master/paperlistwg.tsv',sep='\t',index_col=0)
"""
link='https://scholar.google.com/'#这一版代码只能一个有用。
driver.get(link)
citenum=[]
for i in df['title']:  #输出title
    i=i.lower()#title都变为小写
    print(i)
    elem=driver.find_element(By.XPATH,value='//*[@id="gs_hdr_tsi"]')
    elem.send_keys(i)#输入关键词
    driver.find_element(By.XPATH,value='//*[@id="gs_hdr_tsb"]').click()#点击
    Citetitle=driver.find_element(By.XPATH,value='//*[@id="iVqdeAlAJcgJ"]').text
    Citetitle=Citetitle.lower()
    print(Citetitle)
    if i == Citetitle:#判断是否相等
        #print(111111111)
        waitsave=driver.find_element(By.XPATH,value='//*[@id="gs_res_ccl_mid"]/div/div[2]/div[3]')
        #print(waitsave.text)
        if waitsave.text.find("被引用次数："):
            start=waitsave.text.find("：") + 1
            end=waitsave.text.find("相关文章")
            print(waitsave.text[start:end])
            citenum.append(waitsave.text[start:end])


        else:
            #print(22222222)
            citenum.append(-1)
            continue
    else:
        citenum.append(-1)
        """
#link='https://xs2.dailyheadlines.cc/scholar?hl=zh-CN&as_sdt=0%2C5&q=&btnG='
link='https://xs.dailyheadlines.cc/scholar?hl=zh-CN&as_sdt=0%2C5&q=&btnG=' #镜像站
#link='https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=&btnG='#这一版代码只能一个有用。
driver.get(link)
citenum=[]
paperlist=[]
for i in df['title']:  #输出title

        i=i.lower()#title都变为小写
        #print(i)
        try:
            elem=driver.find_element(By.XPATH,value='//*[@id="gs_hdr_tsi"]')
            try:
                elem.clear()
                time.sleep(0.5)
                print('-------------上title，下搜索')
                try:
                    elem.send_keys(i)#输入关键词
                    try:
                        driver.find_element(By.XPATH,value='//*[@id="gs_hdr_tsb"]').click()#点击
                        time.sleep(2)
                        try:
                            Citetitle=driver.find_element(By.XPATH,value='//*[@id="gs_res_ccl_mid"]/div/div[2]/h3/a').text
                            #Citetitle=Citeclass.find_element_by_xpath('./div[2]/h3/a').text
                            Citetitle=Citetitle.lower()
                            print(Citetitle)
                            if i == Citetitle:#判断是否相等
                                #print(111111111)
                                waitsave=driver.find_element(By.XPATH,value='//*[@id="gs_res_ccl_mid"]/div/div[2]/div[3]')
                                #print(waitsave.text)
                                if waitsave.text.find("被引用次数："):
                                    start=waitsave.text.find("：") + 1
                                    end=waitsave.text.find("相关文章")
                                    print(waitsave.text[start:end])
                                    citenum.append(waitsave.text[start:end])
                                    paperlist.append(i)



                                else:
                                    print(22222222)

                        except Exception as e:

                                pass
                        continue
                    except Exception as e:
                        pass
                    continue
                except Exception as e:
                    pass
                continue
            except Exception as e:
                pass
            continue
        except Exception as e:
            pass
        continue




with open('2020wgcite.txt', 'a+',encoding='utf-8') as f:
    for data_id in citenum:
        f.write(str(data_id)+'\n')
with open('2020wgtitle.txt', 'a+',encoding='utf-8') as f:
    for data_ids in paperlist:
        f.write(str(data_ids)+'\n')


#xpath = '//div[@id="note_children"]//span[@class="note_content_value"]/..'




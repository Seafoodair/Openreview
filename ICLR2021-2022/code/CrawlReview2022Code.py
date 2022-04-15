import os
import time
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

driver = webdriver.Edge('msedgedriver.exe')

df = pd.read_csv('paperlist33.tsv', sep='\t', index_col=0,encoding='gb18030')
conference = dict()
ratings = dict()
decisions = dict()
review=dict()
chair=[]
reviewer0=[]
reviewer1=[]
reviewer2=[]
reviewer3=[]
reviewer4=[]
conf1=[]
conf2=[]
conf3=[]
conf4=[]
conf5=[]
num=0
for paper_id, link in tqdm(list(df.link.items())):
    num+=1
    try:
        driver.get(link)
        xpath = '//div[@id="note_children"]//span[@class="note_content_value"]/..'
        xpathreview= '//div[@id="note_children"]//div[@class="note_contents"]//span[@class="note_content_value markdown-rendered"]'
        ppreview = '//div[@id="note_children"]//div[@class="note_contents"]//span[@class="note_content_field"]'
        xpathre='//div[@id="note_children"]//span[@class="note_content_field"]'
        zzpath='//div[@id="note_children"]'
        cond = EC.presence_of_element_located((By.XPATH, xpath))
        WebDriverWait(driver, 60).until(cond)

        elems = driver.find_elements_by_xpath(xpath)
        assert len(elems), 'empty ratings'   #判断该程序为真
        ratings[paper_id] = pd.Series([
            int(x.text.split(': ')[1]) for x in elems if x.text.startswith('Recommendation:')
        ], dtype=int)
        conference = pd.Series([
            int(x.text.split(': ')[1]) for x in elems if x.text.startswith('Confidence:')
        ], dtype=int)
        review = pd.Series([
            float(x.text.split(': ')[1]) for x in elems if x.text.startswith('Main Review:')
        ], dtype=float)
        decision = [x.text.split(': ')[1] for x in elems if x.text.startswith('Decision:')]

        decisions[paper_id] = decision[0] if decision else 'Unknown'
        for kk in elems:
            kk.text.startswith('Main Review:')
            #print(kk.text)
        reviewnum=driver.find_elements(By.XPATH,value=xpathreview)
        ppt = driver.find_elements(By.XPATH, value=ppreview)
        zpath=driver.find_elements(By.XPATH, value=zzpath)
        count=0

        for aa in zpath:
            for bb in aa.find_elements(By.CLASS_NAME,value='note_contents'):# 这个函数与能够拿到review
               # print(bb.text)
                if bb.find_element_by_xpath('./span[1]').text.startswith('Main Review:'):
                    count=count+1
                    #print(bb.find_element_by_xpath('./span[2]').text)
                    if count==1:
                        reviewer0.append(bb.find_element_by_xpath('./span[2]').text)
                    elif count==2:
                        reviewer1.append(bb.find_element_by_xpath('./span[2]').text)
                    elif count==3:
                        reviewer2.append(bb.find_element_by_xpath('./span[2]').text)
                    elif count==4:
                        reviewer3.append(bb.find_element_by_xpath('./span[2]').text)
                    elif count==5:
                        reviewer4.append(bb.find_element_by_xpath('./span[2]').text)

                    #print("-------------------------\n")
                else:
                    continue



        #这个函数是拿chair的评价
        # for aa in zpath:
        #     for bb in aa.find_elements(By.CLASS_NAME,value='note_contents'):
        #         if bb.find_element_by_xpath('./span[1]').text.startswith('Comment:'):
        #             #print(bb.find_element_by_xpath('./span[2]').text)
        #             chair.append(bb.find_element_by_xpath('./span[2]').text)
        #             break

        #for pp,jj in zip(ppt,reviewnum):
           # if pp.text.startswith('Review:'): #这个代码完全过滤了review
             #   print(jj.text)
            #if jj.text.startswith('Review:'):
                #print(jj.text.split(': ')[1])
                #tt=jj.find_elements(By.XPATH,value='../span[@class="note_content_field"]')
                #for zz in tt:
                    #print(zz.text)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(paper_id, e)
        ratings[paper_id] = pd.Series(dtype=int)
        decisions[paper_id] = 'Unknown'
#print(len(conference))
#ds = pd.DataFrame(conference).T # 这个是信用评分
"""
for k,v in enumerate(conference):
 print(k,v)
 if k == 0:
     conf1.append(v)
 elif k == 1:
     conf2.append(v)
 elif k == 3:
     conf3.append(v)
 elif k == 4:
     conf4.append(v)
 elif k == 5:
     conf5.append(v)
print(conf1)
print("---------------")
print(conf5)
df['conf1']=conf1

while(len(conf5)<len(conf1)):
    conf5.append("-1")
print(conf5)
while(len(conf4)<len(conf1)):
    conf4.append("-1")
while(len(conf3)<len(conf1)):
    conf3.append("-1")
df['conf2']=conf2
df['conf3']=conf3
df['conf4']=conf4
df['conf5']=conf5
"""
#print(ds)
#print(review)#对review进行输出。
#print(len(conference))
df = pd.DataFrame(ratings).T
print(df)
#print(df)
#df = pd.DataFrame(conference).T
df['decision'] = pd.Series(decisions)
while len(reviewer0)<num:
    reviewer0.append(-1)
while len(reviewer4)<len(reviewer0): #如果为空 这个还有点儿问题。
    reviewer4.append("-1")
df['reviewer4'] = reviewer4
print(len(reviewer4))
print("长度：")
print(len(reviewer0))
df['reviewer0']=reviewer0
while len(reviewer1)!=len(reviewer0): #如果为空
    reviewer1.append("-1")
df['reviewer1']=reviewer1
while len(reviewer2)!=len(reviewer0): #如果为空
    reviewer2.append("-1")
df['reviewer2']=reviewer2
while len(reviewer3)!=len(reviewer0): #如果为空
    reviewer3.append("-1")
df['reviewer3']=reviewer3


print(df)
#df=pd.concat([df, ds], axis=1) #列追加拼接
#df=pd.merge(df,ds,on='paper_id')
#print(ds[2])
"""
for k in range(len(ds)):
    print(k,ds[k])
    df['conference'+str(k)]=ds[k]
"""
#df=df+ds
df.index.name = 'paper_id'#可以换成link
df.to_csv('成功2.tsv', sep='\t',mode='a+')

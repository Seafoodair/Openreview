#encoding=utf-8
import urllib
from urllib import request
import feedparser
import time
import sys
import xlsxwriter

workbook = xlsxwriter.Workbook('2020arxiv.xlsx') # 

worksheet = workbook.add_worksheet() # 建立sheet， 可以work.add_worksheet('employee')
#print(sys.getdefaultencoding())
import requests
import urllib.request as libreq
url='http://export.arxiv.org/api/query?search_query=ti:'
#linelint='&sortBy=submittedDate&sortOrder=None'
file=open('titledeal.txt','r',encoding='utf-8')
list_query=[]
for k in file.readlines():
    line=k.strip('\n')
    list_query.append(line)
#print(list_query)
query1='Guided+Exploration in Deep+Reinforcement+Learning'

list=['Image Segmentation by Iterative Inference from Conditional Score Estimation','Guided+Exploration in Deep+Reinforcement+Learning','A Constructive Formalization of the Weak Perfect Graph Theorem']
#print(len(list_query))
j=0
for i in range(len(list_query)):
    #list.append(url+i+linelint)
    #list.append(i)
    #text_to_check = urllib.parse.quote(i)  # 增加此段
    #p=len(list)
    #text_to_check=urllib.parse.quote(list[p-1])
    #print(text_to_check)
    #print(url +i+linelint)
    j=j+1
    #print(j)
    response = urllib.request.urlopen(url+list_query[i])




    # parse the response using feedparser
    feed = feedparser.parse(response)
    for entry in feed.entries:
        worksheet.write(j,0,entry.title)
        worksheet.write(j,1,entry.published)

        print(j,entry.title,'time:  %s' % entry.published)
        #print('\n')
        #print('time:  %s' % entry.published)
        break
    #time.sleep(1)
workbook.close()
    #time.sleep(3)
    #response.encoding = 'utf-8'
    #print(response)
    #feed = feedparser.parse(response)




#coding=utf-8
import numpy as np
import pandas as pd
import collections
import pandas as pd
import numpy as np
import sys
#wr=pd.read_excel('data.xlsx')
wr=pd.read_excel('../../data/sentiment analysis data/sentimentanalysisresult/predictresult.xlsx')
#wn=wr.novel
#wd=wr.noveltag
# wn=wr.motivation
# wd=wr.motivationtag
# wn=wr.experiment
# wd=wr.experimenttag
# wn=wr.relatework
# wd=wr.relateworktag
"""
introduce :per aspects should be run.this only run a aspect
"""
wn=wr.readable
wd=wr.readabletag
listn=[]
listd=[]
for i in wn:
    listn.append(i)
for j in wd:
    listd.append(j)
# for a,b in zip(listn,listd):
#     print(a,"            ",b)
#求一个列表的众数。
def GetMultiData(L):
    x = dict((a, L.count(a)) for a in L)
    y = [k for k, v in x.items() if max(x.values()) == v]
    Y = sorted(y)
    return Y[0]
for c in range(1,16844):
    listc = []
    for a, b in zip(listn, listd):
        #print(a, "            ", b, "            ",c)
        if a==c:
            #print(b)
            listc.append(b)
        else:
            continue
    if listc==[]:
        print(c,"            ",3)
    else:
        print(c, "            ", GetMultiData(listc))
    #break
    #listc.append()

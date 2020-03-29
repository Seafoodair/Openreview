#coding=utf-8
"""

"""
def ForCicle():
    for ii in range(4):
        if kk==1:
            continue
        else:
            ForCicle()
def CalcAvg(pp):
    print('avg')

def Trans(tt):
    list=[]
    for i in tt:
        list.append(i)
    return list
import pandas as pd
wr=pd.read_excel("处理表格.xlsx")
score=wr.score
du1=wr.novel
du2=wr.motivation
du3=wr.experiment
du4=wr.relatework
du5=wr.readable
listscore=Trans(score)
list1=Trans(du1)
list2=Trans(du2)
list3=Trans(du3)
list4=Trans(du4)
list5=Trans(du5)
#for a,b,c,d,e,f in zip(list1,list2,list3,list4,list5,listscore):

for kk in range(4):
    if kk==1:
        continue
    else:
        for k1 in range(4):
            if k1 == 1:
                continue
            else:
                for k2 in range(4):
                    if k2 == 1:
                        continue
                    else:
                        for k3 in range(4):
                            if k3 == 1:
                                continue
                            else:
                                for k4 in range(4):
                                    if k4 == 1:
                                        continue
                                    else:
                                        sum = 0
                                        count = 0
                                        for a, b, c, d, e, f in zip(list1, list2, list3, list4, list5, listscore):
                                            if a==kk and b ==k1 and c==k2 and d==k3 and e==k4 :
                                                sum += f
                                                count+=1
                                            else:
                                                continue
                                        if sum!=0:
                                            avg=sum/count
                                            avg=round(avg,2)
                                            print(kk ,k1,k2,k3,k4,"==================",avg,count)
                                        else:
                                            print(kk, k1, k2, k3, k4, "==================", 0,'         0')

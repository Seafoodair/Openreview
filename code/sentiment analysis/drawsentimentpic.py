#coding=utf-8
"""

"""
import  matplotlib.pyplot  as plt
import random
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x/3.00124732715609
def graphpic(w,h,p,q,list66,listgg):
    sx = 0
    num = 2
    sy = 0.2 + 4* (h + q)
    sum = 0
    for i,kkk in zip(list66,listgg):
        if i==0:
            rect3 = plt.Rectangle((sx, sy), w+p, h+q, color='tomato',alpha=1)
            ax.add_patch(rect3)
        elif i==3:
            rect3=plt.Rectangle((sx, sy), (w+p), h+q, color='tan',alpha=0.6)
            ax.add_patch(rect3)
        else:
            rect3 = plt.Rectangle((sx, sy), (w+p), h + q, color='lime',alpha=0.6)
            ax.add_patch(rect3)
        sum+=kkk
        sx+=(w+p)
        #q=2*q
def graphpic2(w,h,p,q,list66,listgg):
    sx = 0
    num = 2
    sy = 0.2 + 3 * (h + q)
    sum = 0
    for i,kkk in zip(list66,listgg):
        if i==0:
            rect3 = plt.Rectangle((sx, sy), w+p, h+q, color='tomato',alpha=1)
            ax.add_patch(rect3)
        elif i==3:
            rect3=plt.Rectangle((sx, sy), (w+p), h+q, color='tan',alpha=0.6)
            ax.add_patch(rect3)
        else:
            rect3 = plt.Rectangle((sx, sy), (w+p), h + q, color='lime',alpha=0.6)
            ax.add_patch(rect3)
        sum+=kkk
        sx+=(w+p)
        #p=2*p
        #q=2*q
def graphpic3(w,h,p,q,list66,listgg):
    sx=0
    num=2
    sy=0.2+2*(h+q)
    sum=0
    for i,kkk in zip(list66,listgg):
        if i==0:
            rect3 = plt.Rectangle((sx, sy), w+p, h+q, color='tomato',alpha=1)
            ax.add_patch(rect3)
        elif i==3:
            rect3=plt.Rectangle((sx, sy), (w+p), h+q, color='tan',alpha=0.6)
            ax.add_patch(rect3)
        else:
            rect3 = plt.Rectangle((sx, sy), (w+p), h + q, color='lime',alpha=0.6)
            ax.add_patch(rect3)
        sum+=kkk
        sx+=(w+p)
        #p=2*p
        #q=2*q
def graphpic4(w,h,p,q,list66,listgg):
    sx=0
    num=3
    sy=0.2+h+q
    sum = 0
    for i,kkk in zip(list66,listgg):
        if i==0:
            rect3 = plt.Rectangle((sx, sy), w+p, h+q, color='tomato',alpha=1)
            ax.add_patch(rect3)
        elif i==3:
            rect3=plt.Rectangle((sx, sy), (w+p), h+q, color='tan',alpha=0.6)
            ax.add_patch(rect3)
        else:
            rect3 = plt.Rectangle((sx, sy), (w+p), h + q, color='lime',alpha=0.6)
            ax.add_patch(rect3)
        sum+=kkk
        sx+=(w+p)
        #p=2*p
        #q=2*q
def graphpic5(w,h,p,q,list66,listgg):
    sx=0
    num=4
    sy=0.2
    sum=0
    for i,kkk in zip(list66,listgg):
        if i==0:
            rect3 = plt.Rectangle((sx, sy), w+p, h+q, color='tomato',alpha=1)
            ax.add_patch(rect3)
        elif i==3:
            rect3=plt.Rectangle((sx, sy), (w+p), h+q, color='tan',alpha=0.6)
            ax.add_patch(rect3)
        else:
            rect3 = plt.Rectangle((sx, sy), (w+p), h + q, color='lime',alpha=0.6)
            ax.add_patch(rect3)
        sum+=kkk
        sx+=(w+p)
        #p=2*p
        #q=2*q

def Randomlist(t,no):
    randomlist=[]
    for qq in range(no):
        randomlist.append(t)
    return randomlist
def getDu(zeze):
    listzeze=[]
    for boy in zeze:
        listzeze.append(boy)
    return listzeze
def pdread(getdd):
    nba=[]
    for getpp in  getdd:
        nba.append(getpp)
    return nba
if __name__ == '__main__':
    import os
    list = []
    

    wr = pd.read_excel(os.path.abspath('../../data/sentiment analysis data/sentimentanalysisresult/result.xlsx'),sheet_name='use')
    ws = wr.num
    length = []
    for kk in ws:
        length.append(kk)
    for i in length:
        t = MaxMinNormalization(i, max(length), min(length))
        list.append(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)   #创建子图

    # list=[0,2,3,0,2]
    # list2=[3,3,0,2,0]
    # list3=[2,3,0,0,2]
    # list4=[0,0,2,3,0]
    # list5=[3,3,2,2,3]
    # list=Randomlist(0,243)
    # list2=Randomlist(2,243)
    # list3=Randomlist(3,243)
    list1=pdread(wr.novel)
    list2=pdread(wr.motavition)
    list3=pdread(wr.experiment)
    list4=pdread(wr.relatework)
    list5=pdread(wr.readable)
    widlen=1/134
    graphpic(0,0,widlen,0.1,list1,list)#0.0088495575221239这个数字是单位1/个数
    graphpic2(0,0,widlen,0.1,list2,list)
    graphpic3(0,0,widlen,0.1,list3,list)
    graphpic4(0, 0,widlen, 0.1, list4,list)
    #list55=[2,2,3,0,3]
    #listss=[0.1, 0.2, 0.6, 0.05, 0.05]
    graphpic5(0, 0,widlen, 0.1, list5,list)
    labletext=['novelty','motavition','experiment','relate work','presentation']
    listscore=[]
    numcount=0
    for score in wr.avg:
        score=round(score,2)
        listscore.append(score)
    for avgscore in range(134):#这是横坐标这里多少条数据就要多少
        print(avgscore)
        if avgscore%5==0:
            zzz=0.0373134328358209
            plt.text(0 + zzz* numcount, 0.1, listscore[avgscore], rotation=90, fontsize=15)
            #plt.text(0+0.0442477876106195*numcount,0.1,listscore[avgscore],rotation=90,fontsize=15)
            numcount += 1
    plt.text(-0.12, 0.1, '         avg.\n review score', fontsize=15)
    for label in range(5):
        plt.text(-0.11,0.23+0.1*label,labletext[label],fontsize=15)
    # graphpic4(0,0,0.1,0.1,list4)
    # graphpic5(0,0,0.1,0.1,list5)
    # rect = plt.Rectangle((0.1,0.2),0.4,0.3, color="red")# （0.1，0.2）为左下角的坐标，0.4，0.3为宽和高，负数为反方向，红色填充
    # rect2=plt.Rectangle((0.1+0.4,0.2),0.4,0.3, color="green")
    #
    # ax.add_patch(rect)
    # ax.add_patch(rect2)
    labels = [u'positive', u'negative', u'neutral/not mentioned']

    color = ['lime', 'tomato', 'tan']
    alpha = [0.6, 1, 0.6]
    #plt.axis('scaled')

    # labels = ['confidence1', 'confidence2', 'confidence3', 'confidence4','confidence5']  #legend标签列表，上面的color即是颜色列表
    # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i]), alpha=alpha[i]) for i in
               range(len(color))]
    ax.legend(handles=patches, loc=[0.2,0.72] , ncol=3,fontsize=15)
    plt.axis('off')
    plt.show()
